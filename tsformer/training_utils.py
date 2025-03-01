import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
import wandb
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from .adaptive_scheduler import AdaptiveTrainingScheduler
from .terminal_utils import (
    Colors, print_header, print_subheader, print_success, print_error, 
    print_warning, print_info, print_progress, print_metric, print_metrics_table,
    print_progress_bar, format_time, print_time_elapsed, print_config,
    print_timestamp, print_section_separator
)
import random
import gc
import sys
import copy


def set_random_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Set deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_memory():
    """Clear memory caches"""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing for the model to save memory"""
    # Check model type first
    if hasattr(model, '__class__') and 'PatchTST' in model.__class__.__name__:
        # For PatchTST models, handle channel-independent encoder layers
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            for module in model.encoder.layers:
                original_forward = module.forward
                
                def checkpointed_forward(*args, **kwargs):
                    # Make sure no src_mask is passed to PatchTST encoder layers
                    if 'src_mask' in kwargs:
                        del kwargs['src_mask']
                    return torch.utils.checkpoint.checkpoint(original_forward, *args, **kwargs, use_reentrant=False)
                
                module.forward = checkpointed_forward
        
        print_info("Gradient checkpointing enabled for PatchTST model")
        return
    
    # For standard TimeSeriesTransformer models
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        for module in model.encoder.layers:
            if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'forward'):
                # Enable gradient checkpointing
                original_forward = module.forward
                
                def checkpointed_forward(*args, **kwargs):
                    return torch.utils.checkpoint.checkpoint(original_forward, *args, **kwargs, use_reentrant=False)
                
                module.forward = checkpointed_forward
    
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
        for module in model.decoder.layers:
            if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'forward'):
                # Enable gradient checkpointing
                original_forward = module.forward
                
                def checkpointed_forward(*args, **kwargs):
                    return torch.utils.checkpoint.checkpoint(original_forward, *args, **kwargs, use_reentrant=False)
                
                module.forward = checkpointed_forward
    
    print_info("Gradient checkpointing enabled for Transformer model")


def compute_metrics(predictions, targets):
    """
    Computes evaluation metrics between predictions and targets.
    Works with both PyTorch tensors and numpy arrays.
    
    Args:
        predictions: Predicted values (tensor or numpy array)
        targets: Ground truth values (tensor or numpy array)
        
    Returns:
        Dictionary of metrics
    """
    # Convert tensors to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
        
    # Compute Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predictions - targets))
    
    # Compute Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # Compute Symmetric Mean Absolute Percentage Error (SMAPE)
    epsilon = 1e-8  # Small constant to avoid division by zero
    smape = 100 * np.mean(2 * np.abs(predictions - targets) / (np.abs(predictions) + np.abs(targets) + epsilon))
    
    # Compute Mean Absolute Scaled Error (MASE)
    # Using simple persistence forecast (t=t-1) as the baseline
    # For each feature separately
    mase_values = []
    for i in range(targets.shape[-1]):
        feature_targets = targets[:, :, i]
        feature_preds = predictions[:, :, i]
        
        # Compute naive forecasts using persistence (shifted target)
        # We would need full time series for proper MASE, using approximation
        naive_errors = np.abs(np.diff(feature_targets, axis=1))
        naive_error_mean = np.mean(naive_errors) + epsilon
        
        forecast_errors = np.abs(feature_preds - feature_targets)
        feature_mase = np.mean(forecast_errors) / naive_error_mean
        mase_values.append(feature_mase)
    
    mase = np.mean(mase_values)
    
    # Compute Direction Accuracy (for first feature, typically close price)
    feature_idx = 0  # Typically close price is index 0
    correct_directions = 0
    total_directions = 0
    
    for i in range(predictions.shape[0]):
        pred_diff = np.diff(predictions[i, :, feature_idx])
        target_diff = np.diff(targets[i, :, feature_idx])
        
        # Count where direction matches (both positive or both negative)
        matches = (np.sign(pred_diff) == np.sign(target_diff))
        correct_directions += np.sum(matches)
        total_directions += len(matches)
    
    direction_accuracy = (correct_directions / total_directions * 100) if total_directions > 0 else 0.0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'smape': smape,
        'mase': mase,
        'direction_accuracy': direction_accuracy
    }


def compute_individual_losses(pred, target, criterion):
    """
    Compute individual loss components for monitoring
    """
    loss_components = {}
    
    # Standard losses
    mse_loss = nn.MSELoss()(pred, target).item()
    mae_loss = nn.L1Loss()(pred, target).item()
    loss_components["mse"] = mse_loss
    loss_components["mae"] = mae_loss
    
    # Combined loss (what we're optimizing)
    combined_loss = criterion(pred, target).item()
    loss_components["combined_loss"] = combined_loss
    
    # Check if criterion has the component weights attributes
    if hasattr(criterion, 'mse_weight'):
        loss_components["mse_weight"] = criterion.mse_weight
    if hasattr(criterion, 'mae_weight'):
        loss_components["mae_weight"] = criterion.mae_weight
    if hasattr(criterion, 'directional_weight'):
        loss_components["directional_weight"] = criterion.directional_weight
    if hasattr(criterion, 'asymmetric_weight'):
        loss_components["asymmetric_weight"] = criterion.asymmetric_weight
    
    # Calculate directional component if available
    if hasattr(criterion, 'directional_loss') and criterion.directional_loss is not None:
        pred_diff = pred[:, 1:, 0] - pred[:, :-1, 0]
        target_diff = target[:, 1:, 0] - target[:, :-1, 0]
        direction_match = torch.sign(pred_diff) * torch.sign(target_diff)
        direction_accuracy = (direction_match > 0).float().mean().item() * 100
        directional_penalty = torch.clamp(-direction_match, min=0).mean().item()
        loss_components["directional_penalty"] = directional_penalty
        loss_components["direction_accuracy"] = direction_accuracy
    
    # Calculate asymmetric component if available
    if hasattr(criterion, 'asymmetric_loss') and criterion.asymmetric_loss is not None:
        errors = pred - target
        under_predictions = (errors < 0).float().mean().item()
        loss_components["under_prediction_rate"] = under_predictions
    
    return loss_components


def create_prediction_visualization(predictions, targets, epoch, prefix="val", feature_idx=0):
    """
    Create prediction visualization for wandb logging
    Shows multiple samples from the batch for better visualization
    """
    fig = plt.figure(figsize=(12, 8))
    
    # Create a grid of subplots (2x2)
    num_samples = min(4, predictions.shape[0])
    for i in range(num_samples):
        ax = fig.add_subplot(2, 2, i+1)
        
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        pred_series = pred_np[i, :, feature_idx]
        target_series = target_np[i, :, feature_idx]
        
        time_steps = range(1, len(pred_series) + 1)
        ax.plot(time_steps, target_series, label="Ground Truth", marker='o', markersize=4)
        ax.plot(time_steps, pred_series, label="Prediction", marker='x', markersize=4)
        
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.set_title(f"Sample {i+1}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f"{prefix.capitalize()}: Ground Truth vs Prediction (Epoch {epoch+1})")
    plt.subplots_adjust(top=0.9)
    
    return fig


def create_warmup_scheduler(optimizer, num_warmup_steps, total_steps, start_factor=0.1):
    """Create a learning rate scheduler with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return start_factor + (1.0 - start_factor) * current_step / num_warmup_steps
        else:
            # Cosine decay after warmup
            progress = (current_step - num_warmup_steps) / max(1, total_steps - num_warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def apply_test_time_augmentation(model, x_encoder, x_decoder, decoder_time=None, n_augmentations=5, device=None):
    """
    Apply test-time augmentation by adding small noise to input data
    and averaging predictions for more robust results
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Original prediction without augmentation
        original_pred = model(x_encoder, x_decoder, None, decoder_time)
        predictions.append(original_pred)
        
        # Add noise augmentations
        for _ in range(n_augmentations - 1):
            # Add small Gaussian noise
            noise_level = 0.01  # Very subtle noise
            
            # Apply noise to encoder and decoder inputs
            noisy_encoder = x_encoder + noise_level * torch.randn_like(x_encoder)
            noisy_decoder = x_decoder + noise_level * torch.randn_like(x_decoder)
            
            # Get prediction with noisy inputs
            noisy_pred = model(noisy_encoder, noisy_decoder, None, decoder_time)
            predictions.append(noisy_pred)
    
    # Average all predictions
    avg_prediction = torch.stack(predictions).mean(dim=0)
    return avg_prediction


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, 
                scheduler=None, patience=None, use_amp=False, model_type='transformer',
                model_forward_args=None, clip_grad_norm=None, skip_val_until=5):
    """
    Trains a model with advanced features like early stopping, gradient clipping, and mixed precision.
    
    Args:
        model: The PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: PyTorch optimizer
        device: Device to train on ('cuda', 'mps', or 'cpu')
        num_epochs: Number of epochs to train for
        scheduler: Optional learning rate scheduler
        patience: Number of epochs to wait for improvement before early stopping
        use_amp: Whether to use automatic mixed precision training
        model_type: Type of model being trained ('transformer' or 'patchtst')
        model_forward_args: Function that prepares batch inputs for the model based on model type
        clip_grad_norm: Optional maximum norm for gradient clipping
        skip_val_until: Skip validation for the first N epochs to save time
        
    Returns:
        best_model: The best model based on validation loss
        train_losses: List of training losses for each epoch
        val_losses: List of validation losses for each epoch
        val_metrics: Dictionary of validation metrics
    """
    # For early stopping
    best_val_loss = float('inf')
    best_model = None
    wait = 0
    
    # For tracking metrics
    train_losses = []
    val_losses = []
    val_metrics = {'mase': [], 'smape': [], 'rmse': [], 'mae': [], 'direction_accuracy': []}
    
    # Set up wandb
    wandb_enabled = 'wandb' in sys.modules and wandb.run is not None
    print_info = print if 'print_info' in globals() else (lambda x: None)
    
    # Define scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    
    # Default model forward args if not provided
    if model_forward_args is None:
        model_forward_args = {
            'transformer': lambda batch, device: {
                'encoder_x': batch['encoder'].to(device),
                'decoder_x': batch['decoder'].to(device),
                'target': batch['target'].to(device),
                'decoder_time': batch.get('decoder_time', None)
            },
            'patchtst': lambda batch, device: {
                'x': batch['encoder'].to(device),
                'x_mark': batch.get('encoder_time', None)
            }
        }
    
    # Main training loop
    for epoch in range(num_epochs):
        # Track time for this epoch
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Zero gradients
            optimizer.zero_grad()
            
            # Prepare inputs based on model type
            if model_type == 'transformer':
                forward_args = model_forward_args['transformer'](batch, device)
                
                # Mixed precision training
                if use_amp and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = model(**forward_args)
                        loss = criterion(outputs, forward_args['target'][:, -outputs.size(1):])
                    
                    # Scale gradients and optimize
                    scaler.scale(loss).backward()
                    if clip_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Regular training
                    outputs = model(**forward_args)
                    loss = criterion(outputs, forward_args['target'][:, -outputs.size(1):])
                    loss.backward()
                    if clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    optimizer.step()
            
            elif model_type == 'patchtst':
                forward_args = model_forward_args['patchtst'](batch, device)
                target = batch['target'].to(device)
                
                # Mixed precision training
                if use_amp and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = model(**forward_args)
                        # Ensure target shape matches outputs shape for PatchTST model
                        loss = criterion(outputs, target)
                    
                    # Scale gradients and optimize
                    scaler.scale(loss).backward()
                    if clip_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Regular training
                    outputs = model(**forward_args)
                    # Ensure target shape matches outputs shape for PatchTST model
                    loss = criterion(outputs, target)
                    loss.backward()
                    if clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    optimizer.step()
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Update training metrics
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log step metrics to wandb
            if wandb_enabled and batch_idx % 10 == 0:  # Log every 10 batches
                wandb.log({
                    "batch": epoch * len(train_loader) + batch_idx,
                    "train_step_loss": loss.item()
                })
            
            # Clear memory
            if device.type in ['cuda', 'mps']:
                torch.cuda.empty_cache() if device.type == 'cuda' else torch.mps.empty_cache()
        
        # Calculate average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase (only after a few epochs of initial training, to save time)
        avg_val_loss = None
        metrics = None
        
        if epoch >= skip_val_until:  # Skip validation for the first few epochs to save time
            model.eval()
            val_loss = 0.0
            all_targets = []
            all_outputs = []
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            
            with torch.no_grad():
                for batch in progress_bar:
                    if model_type == 'transformer':
                        forward_args = model_forward_args['transformer'](batch, device)
                        outputs = model(**forward_args)
                        targets = forward_args['target'][:, -outputs.size(1):]
                    elif model_type == 'patchtst':
                        forward_args = model_forward_args['patchtst'](batch, device)
                        outputs = model(**forward_args)
                        targets = batch['target'].to(device)
                    
                    # Ensure targets and outputs have the same shape
                    if outputs.shape != targets.shape:
                        print_info(f"Warning: Output shape {outputs.shape} doesn't match target shape {targets.shape}")
                        
                    # Calculate validation loss
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    # Store outputs and targets for metric calculation
                    all_outputs.append(outputs.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                    
                    progress_bar.set_postfix({'loss': loss.item()})
            
            # Calculate average validation loss
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Calculate additional metrics
            all_outputs = np.vstack(all_outputs)
            all_targets = np.vstack(all_targets)
            metrics = compute_metrics(all_outputs, all_targets)
            
            # Update metrics dictionary
            for k, v in metrics.items():
                val_metrics[k].append(v)
        else:
            # If skipping validation, just append None
            val_losses.append(None)
            for k in val_metrics:
                val_metrics[k].append(None)
        
        # Update learning rate scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if avg_val_loss is not None:  # Only step if we performed validation
                    scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # Early stopping check
        if patience is not None and avg_val_loss is not None:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = copy.deepcopy(model)
                wait = 0
                # Save the best model
                torch.save(model.state_dict(), f"best_{model_type}_model.pt")
                print_success(f"Saved best model with val_loss={best_val_loss:.4f}")
            else:
                wait += 1
                if wait >= patience:
                    print_info(f"Early stopping at epoch {epoch+1}")
                    break
        elif best_model is None or (avg_val_loss is not None and avg_val_loss < best_val_loss):
            # If not using early stopping but we want to keep the best model
            best_val_loss = float('inf') if avg_val_loss is None else avg_val_loss
            best_model = copy.deepcopy(model)
            # Save the best model
            torch.save(model.state_dict(), f"best_{model_type}_model.pt")
            print_success(f"Saved model at epoch {epoch+1}")
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        epoch_summary = f"Epoch {epoch+1}/{num_epochs} - "
        epoch_summary += f"Train Loss: {avg_train_loss:.4f}"
        if avg_val_loss is not None:
            epoch_summary += f" | Val Loss: {avg_val_loss:.4f}"
            if metrics:
                epoch_summary += f" | RMSE: {metrics['rmse']:.4f}"
                epoch_summary += f" | Dir Acc: {metrics['direction_accuracy']:.2f}%"
        epoch_summary += f" | Time: {epoch_time:.1f}s"
        print_info(epoch_summary)
        
        # Log to wandb
        if wandb_enabled:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch_time": epoch_time
            }
            if avg_val_loss is not None:
                log_dict["val_loss"] = avg_val_loss
                if metrics:
                    for k, v in metrics.items():
                        log_dict[f"val_{k}"] = v
            wandb.log(log_dict)
        
        # Clear memory after epoch
        clear_memory()
    
    # Return the best model
    if best_model is None:
        best_model = model  # If no best model was saved, return the final model
    
    return best_model, train_losses, val_losses, val_metrics


def evaluate_model(model, test_loader, criterion, device, model_type='transformer', model_forward_args=None):
    """
    Evaluates a model on the test dataset and returns metrics.
    
    Args:
        model: The PyTorch model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on ('cuda', 'mps', or 'cpu')
        model_type: Type of model being evaluated ('transformer' or 'patchtst')
        model_forward_args: Function that prepares batch inputs for the model based on model type
        
    Returns:
        Dictionary of evaluation metrics
    """
    print_info = print if 'print_info' in globals() else (lambda x: None)
    model.eval()
    
    # Default model forward args if not provided
    if model_forward_args is None:
        model_forward_args = {
            'transformer': lambda batch, device: {
                'encoder_x': batch['encoder'].to(device),
                'decoder_x': batch['decoder'].to(device),
                'target': batch['target'].to(device),
                'decoder_time': batch.get('decoder_time', None)
            },
            'patchtst': lambda batch, device: {
                'x': batch['encoder'].to(device),
                'x_mark': batch.get('encoder_time', None)
            }
        }
    
    print_info("Evaluating model on test set")
    test_loss = 0.0
    all_outputs = []
    all_targets = []
    all_datetimes = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Get batch data based on model type
            if model_type == 'transformer':
                forward_args = model_forward_args['transformer'](batch, device)
                outputs = model(**forward_args)
                targets = forward_args['target'][:, -outputs.size(1):]
            elif model_type == 'patchtst':
                forward_args = model_forward_args['patchtst'](batch, device)
                outputs = model(**forward_args)
                targets = batch['target'].to(device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Calculate loss
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # Store outputs and targets
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            # Store datetimes if available
            if 'datetime' in batch:
                all_datetimes.extend(batch['datetime'])
    
    # Calculate average loss and metrics
    avg_test_loss = test_loss / len(test_loader)
    all_outputs = np.vstack(all_outputs)
    all_targets = np.vstack(all_targets)
    
    # Compute metrics
    metrics = compute_metrics(all_outputs, all_targets)
    metrics['loss'] = avg_test_loss
    
    # Clear memory
    clear_memory()
    
    return metrics 