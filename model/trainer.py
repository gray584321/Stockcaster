import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from datetime import datetime
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import os
from contextlib import nullcontext, contextmanager
from torch.cuda.amp import GradScaler, autocast

def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Calculate multiple metrics for evaluation using vectorized operations"""
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)
    
    # Calculate all metrics in one pass
    diff = predictions - targets
    squared_diff = diff * diff
    abs_diff = np.abs(diff)
    
    mse = np.mean(squared_diff)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_diff)
    
    # Calculate MAPE
    mask = targets != 0  # Avoid division by zero
    mape = np.mean(np.abs(diff[mask] / targets[mask])) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }

@contextmanager
def autocast_context(device: str, use_amp: bool):
    """Unified autocast context manager for different devices"""
    if device == 'cuda':
        with autocast():
            yield
    elif device == 'mps' and use_amp:
        with torch.autocast(device):
            yield
    else:
        with nullcontext():
            yield

class PatchTSTTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: str = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        num_epochs: int = 100,
        patience: int = 10,
        use_wandb: bool = True,
        gradient_accumulation_steps: int = 1
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.use_wandb = use_wandb
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Initialize wandb only if needed
        if self.use_wandb:
            # Log detailed model architecture and parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            wandb.init(
                project="stockcaster",
                config={
                    "architecture": model.__class__.__name__,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "patience": patience,
                    "device": device,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "optimizer": "AdamW",
                    "scheduler": "CosineAnnealingLR",
                    "model_structure": str(model),
                }
            )
            
            # Log model parameter histograms
            for name, param in model.named_parameters():
                if param.requires_grad:
                    wandb.log({f"parameter_dist/{name}": wandb.Histogram(param.detach().cpu().numpy())})
        
        # Use AMP for faster training on supported devices
        self.use_amp = device in ['cuda', 'mps']
        self.scaler = GradScaler() if device == 'cuda' else None
            
        # Use a more memory-efficient optimizer configuration
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            foreach=True
        )
        
        # Learning rate scheduler for better convergence
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=num_epochs
        )
        
        self.best_val_loss = float('inf')
        self.best_model_path = 'best_model.pth'
        self.pretrained_model_path = 'pretrained_model.pth'
        
    def create_dataloader(self, sequences: np.ndarray, targets: Optional[np.ndarray] = None) -> DataLoader:
        """Create optimized dataloader based on device type"""
        # Convert to torch tensors with correct dtype and device placement
        sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
        if targets is not None:
            targets_tensor = torch.tensor(targets, dtype=torch.float32)
            dataset = TensorDataset(sequences_tensor, targets_tensor)
        else:
            dataset = TensorDataset(sequences_tensor)
        
        # Optimize dataloader settings based on device
        num_workers = 0
        pin_memory = False
        
        if self.device == 'cuda':
            num_workers = min(4, os.cpu_count() or 1)
            pin_memory = True
        elif self.device == 'mps':
            num_workers = 1  # MPS works better with minimal workers
            pin_memory = True

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None
        )

    def _forward_pass(self, sequences: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Unified forward pass with automatic mixed precision support"""
        with autocast_context(self.device, self.use_amp):
            if targets is None:
                # Pretraining forward pass
                reconstructed = self.model.forward_pretrain(sequences)
                loss = self.criterion(reconstructed, sequences)
                return loss, reconstructed
            else:
                # Training forward pass
                predictions = self.model(sequences)
                if predictions.shape != targets.shape:
                    predictions = predictions.view(targets.shape)
                loss = self.criterion(predictions, targets)
                return loss, predictions

    def _optimization_step(self, loss: torch.Tensor, accumulation_step: int):
        """Unified optimization step with gradient accumulation and AMP support"""
        # Scale loss by accumulation steps for consistent gradients
        scaled_loss = loss / self.gradient_accumulation_steps
        
        if self.device == 'cuda':
            self.scaler.scale(scaled_loss).backward()
            if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
        else:
            scaled_loss.backward()
            if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

    def _log_validation_predictions(self, val_seq, val_targets, epoch):
        """Log validation predictions and groundtruth to W&B with optimized memory usage"""
        self.model.eval()
        predictions_list = []
        batch_size = min(1024, len(val_seq))  # Process validation in reasonable chunks
        
        with autocast_context(self.device, self.use_amp):
            for i in range(0, len(val_seq), batch_size):
                batch_seq = torch.tensor(
                    val_seq[i:i + batch_size], 
                    dtype=torch.float32, 
                    device=self.device
                )
                with torch.no_grad():
                    batch_pred = self.model(batch_seq)
                predictions_list.append(batch_pred.cpu().numpy())
                del batch_pred, batch_seq
            
            val_pred = np.concatenate(predictions_list)
            
            # Calculate metrics
            metrics = calculate_metrics(val_pred, val_targets)
            
            if self.use_wandb:
                # Create visualization efficiently
                plt.figure(figsize=(12, 6))
                # Plot only a subset for visualization
                plot_size = min(1000, len(val_targets))
                plt.plot(val_targets[:plot_size].reshape(-1), label='Ground Truth', alpha=0.7)
                plt.plot(val_pred[:plot_size].reshape(-1), label='Predictions', alpha=0.7)
                plt.title(f'Validation Predictions vs Ground Truth - Epoch {epoch}')
                plt.legend()
                plt.grid(True)
                
                # Log metrics and visualizations
                wandb.log({
                    f"val/mse": metrics['mse'],
                    f"val/rmse": metrics['rmse'],
                    f"val/mae": metrics['mae'],
                    f"val/mape": metrics['mape'],
                    "validation_predictions": wandb.Image(plt),
                    "epoch": epoch
                })
                
                # Log distributions and scatter plot with reduced data
                wandb.log({
                    "val/prediction_distribution": wandb.Histogram(val_pred[:plot_size].reshape(-1)),
                    "val/error_distribution": wandb.Histogram((val_pred[:plot_size] - val_targets[:plot_size]).reshape(-1)),
                    "val/predicted_vs_actual_scatter": wandb.plot.scatter(
                        wandb.Table(data=[[x, y] for x, y in zip(
                            val_targets[:plot_size].reshape(-1), 
                            val_pred[:plot_size].reshape(-1)
                        )], columns=["ground_truth", "predictions"]),
                        "ground_truth",
                        "predictions"
                    )
                })
                
                plt.close()
                
            # Clean up
            del val_pred, predictions_list
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                
            return metrics
            
    def _log_test_metrics(self, test_seq, test_targets):
        """Log test metrics with optimized memory usage"""
        self.model.eval()
        predictions_list = []
        batch_size = min(1024, len(test_seq))
        
        with autocast_context(self.device, self.use_amp):
            for i in range(0, len(test_seq), batch_size):
                batch_seq = torch.tensor(
                    test_seq[i:i + batch_size], 
                    dtype=torch.float32, 
                    device=self.device
                )
                with torch.no_grad():
                    batch_pred = self.model(batch_seq)
                predictions_list.append(batch_pred.cpu().numpy())
                del batch_pred, batch_seq
            
            test_pred = np.concatenate(predictions_list)
            
            # Calculate metrics
            metrics = calculate_metrics(test_pred, test_targets)
            
            if self.use_wandb:
                # Create visualization efficiently
                plt.figure(figsize=(12, 6))
                # Plot only a subset for visualization
                plot_size = min(1000, len(test_targets))
                plt.plot(test_targets[:plot_size].reshape(-1), label='Ground Truth', alpha=0.7)
                plt.plot(test_pred[:plot_size].reshape(-1), label='Predictions', alpha=0.7)
                plt.title('Test Set: Predictions vs Ground Truth')
                plt.legend()
                plt.grid(True)
                
                # Log metrics and visualizations
                wandb.log({
                    "test/mse": metrics['mse'],
                    "test/rmse": metrics['rmse'],
                    "test/mae": metrics['mae'],
                    "test/mape": metrics['mape'],
                    "test/predictions_plot": wandb.Image(plt),
                    "test/prediction_distribution": wandb.Histogram(test_pred[:plot_size].reshape(-1)),
                    "test/error_distribution": wandb.Histogram((test_pred[:plot_size] - test_targets[:plot_size]).reshape(-1)),
                    "test/predicted_vs_actual_scatter": wandb.plot.scatter(
                        wandb.Table(data=[[x, y] for x, y in zip(
                            test_targets[:plot_size].reshape(-1), 
                            test_pred[:plot_size].reshape(-1)
                        )], columns=["ground_truth", "predictions"]),
                        "ground_truth",
                        "predictions"
                    )
                })
                
                plt.close()
                
            # Clean up
            del test_pred, predictions_list
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                
            return metrics
    
    def _log_model_gradients(self):
        """Log model gradient statistics"""
        if not self.use_wandb:
            return
            
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
        wandb.log({"gradients/global_norm": grad_norm})
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_data = param.grad.detach().cpu().numpy()
                wandb.log({
                    f"gradients/{name}/mean": np.mean(grad_data),
                    f"gradients/{name}/std": np.std(grad_data),
                    f"gradients/{name}/norm": np.linalg.norm(grad_data),
                    f"gradients/{name}/histogram": wandb.Histogram(grad_data)
                })

    def _log_layer_outputs(self, prefix, sample_input):
        """Log intermediate layer outputs for visualization"""
        if not self.use_wandb:
            return
            
        self.model.eval()
        with torch.no_grad():
            # Get sample predictions and log activations
            if hasattr(self.model, 'get_layer_outputs'):
                layer_outputs = self.model.get_layer_outputs(sample_input)
                for layer_name, output in layer_outputs.items():
                    # Log first channel/feature map
                    if len(output.shape) >= 3:
                        output_data = output[0, 0].detach().cpu().numpy()
                        plt.figure(figsize=(8, 6))
                        plt.imshow(output_data, cmap='viridis')
                        plt.colorbar()
                        plt.title(f'{layer_name} Activation')
                        wandb.log({f"{prefix}/layer_outputs/{layer_name}": wandb.Image(plt)})
                        plt.close()
                        
                        # Log statistics
                        wandb.log({
                            f"{prefix}/layer_stats/{layer_name}/mean": output.mean().item(),
                            f"{prefix}/layer_stats/{layer_name}/std": output.std().item(),
                            f"{prefix}/layer_stats/{layer_name}/histogram": wandb.Histogram(output.detach().cpu().numpy())
                        })

    def _log_optimization_step(self, loss, step, epoch, total_steps):
        """Log detailed optimization metrics"""
        if not self.use_wandb:
            return
            
        # Log basic metrics
        wandb.log({
            "optimization/loss": loss.item(),
            "optimization/learning_rate": self.scheduler.get_last_lr()[0],
            "optimization/epoch": epoch,
            "optimization/step": step,
            "optimization/progress": step / total_steps
        })
        
        # Log gradient statistics
        self._log_model_gradients()
        
        # Log parameter statistics
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_data = param.detach().cpu().numpy()
                wandb.log({
                    f"parameters/{name}/mean": np.mean(param_data),
                    f"parameters/{name}/std": np.std(param_data),
                    f"parameters/{name}/norm": np.linalg.norm(param_data)
                })

    def _log_epoch_summary(self, epoch, train_metrics, val_metrics=None, test_metrics=None):
        """Log comprehensive epoch summary"""
        if not self.use_wandb:
            return
            
        summary = {
            "epoch": epoch,
            "train/epoch_metrics": train_metrics
        }
        
        if val_metrics:
            summary["validation/epoch_metrics"] = val_metrics
            
        if test_metrics:
            summary["test/epoch_metrics"] = test_metrics
            
        # Create comparison plots
        if val_metrics and test_metrics:
            metrics = ['mse', 'mae', 'rmse', 'mape']
            values = {
                'train': [train_metrics.get(m, 0) for m in metrics],
                'val': [val_metrics.get(m, 0) for m in metrics],
                'test': [test_metrics.get(m, 0) for m in metrics]
            }
            
            # Bar plot comparing metrics
            plt.figure(figsize=(12, 6))
            x = np.arange(len(metrics))
            width = 0.25
            
            plt.bar(x - width, values['train'], width, label='Train')
            plt.bar(x, values['val'], width, label='Validation')
            plt.bar(x + width, values['test'], width, label='Test')
            
            plt.xlabel('Metrics')
            plt.ylabel('Value')
            plt.title(f'Metrics Comparison - Epoch {epoch}')
            plt.xticks(x, metrics)
            plt.legend()
            
            summary["epoch_summary/metrics_comparison"] = wandb.Image(plt)
            plt.close()
        
        wandb.log(summary)

    def _log_test_reconstruction_samples(self, test_seq, epoch):
        """Log test dataset reconstruction samples"""
        if not self.use_wandb or test_seq is None:
            return
            
        self.model.eval()
        with torch.no_grad():
            # Take a few samples from test set
            sample_indices = np.random.choice(len(test_seq), min(5, len(test_seq)), replace=False)
            samples = torch.tensor(test_seq[sample_indices], dtype=torch.float32, device=self.device)
            
            # Get reconstructions
            reconstructions = self.model.forward_pretrain(samples)
            
            # Plot multiple samples
            fig, axes = plt.subplots(len(sample_indices), 1, figsize=(12, 4*len(sample_indices)))
            if len(sample_indices) == 1:
                axes = [axes]
                
            for i, (sample, recon) in enumerate(zip(samples, reconstructions)):
                sample_data = sample.detach().cpu().numpy()
                recon_data = recon.detach().cpu().numpy()
                
                axes[i].plot(sample_data.reshape(-1), label='Original', alpha=0.7)
                axes[i].plot(recon_data.reshape(-1), label='Reconstructed', alpha=0.7, linestyle='--')
                axes[i].set_title(f'Sample {i+1} Reconstruction')
                axes[i].legend()
                axes[i].grid(True)
            
            plt.tight_layout()
            wandb.log({
                "test_reconstruction/samples": wandb.Image(plt),
                "epoch": epoch
            })
            plt.close()
            
            # Log reconstruction error distribution
            error = (samples - reconstructions).detach().cpu().numpy()
            wandb.log({
                "test_reconstruction/error_dist": wandb.Histogram(error),
                "test_reconstruction/error_mean": np.mean(np.abs(error)),
                "test_reconstruction/error_std": np.std(error),
                "epoch": epoch
            })

    def pretrain(self, train_seq, val_seq=None, test_seq=None):
        """Unsupervised pre-training with enhanced logging"""
        train_loader = self.create_dataloader(train_seq)
        if val_seq is not None:
            val_loader = self.create_dataloader(val_seq)
        if test_seq is not None:
            test_loader = self.create_dataloader(test_seq)
            
        best_pretrain_loss = float('inf')
        patience_counter = 0
        
        def log_reconstruction_samples(prefix, sequences, epoch):
            """Helper function to log reconstruction samples"""
            with autocast_context(self.device, self.use_amp):
                sample_seq = sequences[:8].to(self.device, non_blocking=True)
                sample_recon = self.model.forward_pretrain(sample_seq)
                
                # Move to CPU and convert to numpy efficiently
                sample_data = sample_seq[0, :100, 0].detach().cpu().numpy()
                recon_data = sample_recon[0, :100, 0].detach().cpu().numpy()
                
                plt.figure(figsize=(12, 6))
                plt.plot(sample_data, label='Original', alpha=0.7)
                plt.plot(recon_data, label='Reconstructed', alpha=0.7, linestyle='--')
                plt.title(f'{prefix} Sample Reconstruction - Epoch {epoch + 1}')
                plt.legend()
                plt.grid(True)
                wandb.log({f"{prefix}/sample_reconstruction": wandb.Image(plt)})
                plt.close()
                
                # Calculate reconstruction metrics
                mse = F.mse_loss(sample_recon, sample_seq).item()
                mae = F.l1_loss(sample_recon, sample_seq).item()
                wandb.log({
                    f"{prefix}/reconstruction_mse": mse,
                    f"{prefix}/reconstruction_mae": mae,
                    "epoch": epoch
                })
                
                del sample_recon, sample_seq
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            running_mse = 0
            running_mae = 0
            running_count = 0
            
            with tqdm(train_loader, desc=f'Pretrain Epoch {epoch + 1}/{self.num_epochs}') as pbar:
                for step, batch in enumerate(pbar):
                    sequences = batch[0].to(self.device, non_blocking=True)
                    
                    # Forward pass and loss calculation with AMP support
                    loss, reconstructed = self._forward_pass(sequences)
                    
                    # Optimization step with gradient accumulation
                    self._optimization_step(loss, step)
                    
                    # Update running statistics
                    batch_size = sequences.size(0)
                    running_count += batch_size
                    total_loss += loss.item() * batch_size
                    
                    # Calculate batch metrics efficiently
                    with torch.no_grad():
                        batch_mse = F.mse_loss(reconstructed, sequences, reduction='mean').item()
                        batch_mae = F.l1_loss(reconstructed, sequences, reduction='mean').item()
                        running_mse += batch_mse * batch_size
                        running_mae += batch_mae * batch_size
                    
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'mse': batch_mse,
                        'mae': batch_mae,
                        'lr': self.scheduler.get_last_lr()[0]
                    })
                    
                    if self.use_wandb:
                        wandb.log({
                            "pretrain/batch_loss": loss.item(),
                            "pretrain/batch_mse": batch_mse,
                            "pretrain/batch_mae": batch_mae,
                            "pretrain/learning_rate": self.scheduler.get_last_lr()[0],
                            "pretrain/batch": epoch * len(train_loader) + pbar.n
                        })
                    
                    # Free memory
                    del reconstructed, loss
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
            
            # Step the scheduler
            self.scheduler.step()
            
            # Calculate epoch metrics
            avg_loss = total_loss / running_count
            avg_mse = running_mse / running_count
            avg_mae = running_mae / running_count
            
            if self.use_wandb:
                # Log train metrics and sample reconstructions
                wandb.log({
                    "pretrain/epoch_loss": avg_loss,
                    "pretrain/epoch": epoch,
                    "pretrain/mse": avg_mse,
                    "pretrain/mae": avg_mae
                })
                
                # Log train reconstructions
                log_reconstruction_samples("pretrain/train", sequences, epoch)
                
                # Log validation metrics and reconstructions if available
                if val_seq is not None:
                    self.model.eval()
                    val_loss = 0
                    val_mse = 0
                    val_mae = 0
                    val_count = 0
                    
                    with torch.no_grad():
                        for val_batch in val_loader:
                            val_sequences = val_batch[0].to(self.device, non_blocking=True)
                            val_recon = self.model.forward_pretrain(val_sequences)
                            
                            val_batch_loss = F.mse_loss(val_recon, val_sequences)
                            val_batch_mse = F.mse_loss(val_recon, val_sequences, reduction='mean').item()
                            val_batch_mae = F.l1_loss(val_recon, val_sequences, reduction='mean').item()
                            
                            batch_size = val_sequences.size(0)
                            val_loss += val_batch_loss.item() * batch_size
                            val_mse += val_batch_mse * batch_size
                            val_mae += val_batch_mae * batch_size
                            val_count += batch_size
                            
                            del val_recon
                    
                    avg_val_loss = val_loss / val_count
                    avg_val_mse = val_mse / val_count
                    avg_val_mae = val_mae / val_count
                    
                    wandb.log({
                        "pretrain/val_loss": avg_val_loss,
                        "pretrain/val_mse": avg_val_mse,
                        "pretrain/val_mae": avg_val_mae,
                        "epoch": epoch
                    })
                    
                    # Log validation reconstructions
                    log_reconstruction_samples("pretrain/val", val_sequences, epoch)
                
                # Log test metrics and reconstructions if available
                if test_seq is not None:
                    self.model.eval()
                    test_loss = 0
                    test_mse = 0
                    test_mae = 0
                    test_count = 0
                    
                    with torch.no_grad():
                        for test_batch in test_loader:
                            test_sequences = test_batch[0].to(self.device, non_blocking=True)
                            test_recon = self.model.forward_pretrain(test_sequences)
                            
                            test_batch_loss = F.mse_loss(test_recon, test_sequences)
                            test_batch_mse = F.mse_loss(test_recon, test_sequences, reduction='mean').item()
                            test_batch_mae = F.l1_loss(test_recon, test_sequences, reduction='mean').item()
                            
                            batch_size = test_sequences.size(0)
                            test_loss += test_batch_loss.item() * batch_size
                            test_mse += test_batch_mse * batch_size
                            test_mae += test_batch_mae * batch_size
                            test_count += batch_size
                            
                            del test_recon
                    
                    avg_test_loss = test_loss / test_count
                    avg_test_mse = test_mse / test_count
                    avg_test_mae = test_mae / test_count
                    
                    wandb.log({
                        "pretrain/test_loss": avg_test_loss,
                        "pretrain/test_mse": avg_test_mse,
                        "pretrain/test_mae": avg_test_mae,
                        "epoch": epoch
                    })
                    
                    # Log test reconstructions
                    log_reconstruction_samples("pretrain/test", test_sequences, epoch)
            
            # Add test reconstruction logging
            if test_seq is not None:
                self._log_test_reconstruction_samples(test_seq, epoch)
                
            # Log epoch summary
            self._log_epoch_summary(
                epoch,
                train_metrics={'loss': avg_loss, 'mse': avg_mse, 'mae': avg_mae},
                val_metrics={'loss': avg_val_loss, 'mse': avg_val_mse, 'mae': avg_val_mae} if val_seq is not None else None,
                test_metrics={'loss': avg_test_loss, 'mse': avg_test_mse, 'mae': avg_test_mae} if test_seq is not None else None
            )
            
            if avg_loss < best_pretrain_loss:
                best_pretrain_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': best_pretrain_loss,
                }, self.pretrained_model_path)
                patience_counter = 0
                if self.use_wandb:
                    wandb.log({
                        "pretrain/best_loss": avg_loss,
                        "pretrain/best_mse": avg_mse,
                        "pretrain/best_mae": avg_mae
                    })
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
            
            # Memory cleanup at epoch end
            torch.cuda.empty_cache() if self.device == 'cuda' else None

    def train(self, train_seq, train_targets, val_seq=None, val_targets=None, test_seq=None, test_targets=None):
        """Supervised training with enhanced logging"""
        print(f"Training data shapes - Sequences: {train_seq.shape}, Targets: {train_targets.shape}")
        
        train_loader = self.create_dataloader(train_seq, train_targets)
        if val_seq is not None and val_targets is not None:
            val_loader = self.create_dataloader(val_seq, val_targets)
            print(f"Validation data shapes - Sequences: {val_seq.shape}, Targets: {val_targets.shape}")
        if test_seq is not None and test_targets is not None:
            test_loader = self.create_dataloader(test_seq, test_targets)
            print(f"Test data shapes - Sequences: {test_seq.shape}, Targets: {test_targets.shape}")
        
        def log_prediction_samples(prefix, sequences, targets, epoch):
            """Helper function to log prediction samples"""
            with autocast_context(self.device, self.use_amp):
                sample_seq = sequences[:8].to(self.device, non_blocking=True)
                sample_targets = targets[:8].to(self.device, non_blocking=True)
                with torch.no_grad():
                    sample_pred = self.model(sample_seq)
                
                # Move to CPU and convert to numpy efficiently
                target_data = sample_targets[0].detach().cpu().numpy()
                pred_data = sample_pred[0].detach().cpu().numpy()
                
                plt.figure(figsize=(12, 6))
                plt.plot(target_data, label='Target', alpha=0.7)
                plt.plot(pred_data, label='Prediction', alpha=0.7, linestyle='--')
                plt.title(f'{prefix} Sample Predictions - Epoch {epoch + 1}')
                plt.legend()
                plt.grid(True)
                wandb.log({f"{prefix}/sample_predictions": wandb.Image(plt)})
                plt.close()
                
                # Calculate prediction metrics for samples
                mse = F.mse_loss(sample_pred, sample_targets).item()
                mae = F.l1_loss(sample_pred, sample_targets).item()
                wandb.log({
                    f"{prefix}/sample_mse": mse,
                    f"{prefix}/sample_mae": mae,
                    "epoch": epoch
                })
                
                del sample_pred
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            running_mse = 0
            running_mae = 0
            running_count = 0
            
            with tqdm(train_loader, desc=f'Train Epoch {epoch + 1}/{self.num_epochs}') as pbar:
                for step, (sequences, targets) in enumerate(pbar):
                    sequences = sequences.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    
                    # Forward pass and loss calculation with AMP support
                    loss, predictions = self._forward_pass(sequences, targets)
                    
                    # Optimization step with gradient accumulation
                    self._optimization_step(loss, step)
                    
                    # Update running statistics
                    batch_size = sequences.size(0)
                    running_count += batch_size
                    total_loss += loss.item() * batch_size
                    
                    # Calculate batch metrics efficiently
                    with torch.no_grad():
                        batch_mse = F.mse_loss(predictions, targets, reduction='mean').item()
                        batch_mae = F.l1_loss(predictions, targets, reduction='mean').item()
                        running_mse += batch_mse * batch_size
                        running_mae += batch_mae * batch_size
                    
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'mse': batch_mse,
                        'mae': batch_mae,
                        'lr': self.scheduler.get_last_lr()[0]
                    })
                    
                    if self.use_wandb:
                        wandb.log({
                            "train/batch_loss": loss.item(),
                            "train/batch_mse": batch_mse,
                            "train/batch_mae": batch_mae,
                            "train/learning_rate": self.scheduler.get_last_lr()[0],
                            "train/batch": step + epoch * len(train_loader)
                        })
                    
                    # Free memory
                    del predictions, loss
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
            
            # Step the scheduler
            self.scheduler.step()
            
            # Calculate epoch metrics
            avg_loss = total_loss / running_count
            avg_mse = running_mse / running_count
            avg_mae = running_mae / running_count
            
            if self.use_wandb:
                # Log train metrics and sample predictions
                wandb.log({
                    "train/epoch_loss": avg_loss,
                    "train/mse": avg_mse,
                    "train/mae": avg_mae,
                    "epoch": epoch
                })
                
                # Log train predictions
                log_prediction_samples("train", sequences, targets, epoch)
                
                # Validation step with enhanced logging
                if val_seq is not None and val_targets is not None:
                    val_metrics = self._log_validation_predictions(val_seq, val_targets, epoch)
                    log_prediction_samples("val", val_seq, val_targets, epoch)
                    
                    # Update best model if validation improves
                    if val_metrics['mse'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['mse']
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'loss': self.best_val_loss,
                        }, self.best_model_path)
                        patience_counter = 0
                    else:
                        patience_counter += 1
                
                # Log test metrics and predictions if available
                if test_seq is not None and test_targets is not None:
                    test_metrics = self._log_test_metrics(test_seq, test_targets)
                    log_prediction_samples("test", test_seq, test_targets, epoch)
                    
                    # Additional test metrics logging
                    wandb.log({
                        "test/epoch_loss": test_metrics['mse'],  # Using MSE as loss
                        "test/mse": test_metrics['mse'],
                        "test/rmse": test_metrics['rmse'],
                        "test/mae": test_metrics['mae'],
                        "test/mape": test_metrics['mape'],
                        "epoch": epoch
                    })
            
            # Add test reconstruction logging if in pretraining mode
            if hasattr(self.model, 'forward_pretrain') and test_seq is not None:
                self._log_test_reconstruction_samples(test_seq, epoch)
                
            # Log epoch summary
            self._log_epoch_summary(
                epoch,
                train_metrics={'loss': avg_loss, 'mse': avg_mse, 'mae': avg_mae},
                val_metrics=val_metrics if val_seq is not None else None,
                test_metrics=test_metrics if test_seq is not None else None
            )
            
            if patience_counter >= self.patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
            
            # Memory cleanup at epoch end
            torch.cuda.empty_cache() if self.device == 'cuda' else None

    def validate(self, val_loader):
        """Validation step with optimized memory usage"""
        self.model.eval()
        total_val_loss = 0
        total_samples = 0
        
        with autocast_context(self.device, self.use_amp):
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                with torch.no_grad():
                    predictions = self.model(sequences)
                    if predictions.shape != targets.shape:
                        predictions = predictions.view(targets.shape)
                        
                    val_loss = self.criterion(predictions, targets)
                    batch_size = sequences.size(0)
                    total_val_loss += val_loss.item() * batch_size
                    total_samples += batch_size
                
                # Free memory
                del predictions
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        return total_val_loss / total_samples
    
    def predict(self, sequences, batch_size=None):
        """Generate predictions with optimized memory usage"""
        self.model.eval()
        if batch_size is None:
            batch_size = min(self.batch_size, 1024)  # Use a reasonable batch size
            
        predictions_list = []
        
        with autocast_context(self.device, self.use_amp):
            for i in range(0, len(sequences), batch_size):
                batch_seq = torch.tensor(
                    sequences[i:i + batch_size],
                    dtype=torch.float32,
                    device=self.device
                )
                
                with torch.no_grad():
                    batch_predictions = self.model(batch_seq)
                    predictions_list.append(batch_predictions.cpu().numpy())
                
                # Free memory
                del batch_predictions, batch_seq
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        predictions = np.concatenate(predictions_list)
        del predictions_list
        return predictions
    
    def load_pretrained(self):
        """Load pre-trained weights with device handling"""
        state_dict = torch.load(self.pretrained_model_path, map_location=self.device)
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        
    def load_best_model(self):
        """Load best model weights with device handling"""
        state_dict = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        
    def __del__(self):
        """Cleanup resources"""
        if self.use_wandb:
            try:
                wandb.finish()
            except:
                pass
        
        # Clear CUDA cache if available
        if self.device == 'cuda':
            try:
                torch.cuda.empty_cache()
            except:
                pass 