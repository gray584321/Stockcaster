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
from .adaptive_scheduler import AdaptiveTrainingScheduler
from .terminal_utils import (
    Colors, print_header, print_subheader, print_success, print_error, 
    print_warning, print_info, print_progress, print_metric, print_metrics_table,
    print_progress_bar, format_time, print_time_elapsed, print_config,
    print_timestamp, print_section_separator
)


def set_random_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(pred, target):
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    mae = np.mean(np.abs(pred_np - target_np))
    rmse = np.sqrt(np.mean((pred_np - target_np) ** 2))
    mape = np.mean(np.abs((pred_np - target_np) / (target_np + 1e-5))) * 100
    return mae, rmse, mape


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
        directional_penalty = torch.clamp(-direction_match, min=0).mean().item()
        loss_components["directional_penalty"] = directional_penalty
    
    # Calculate asymmetric component if available
    if hasattr(criterion, 'asymmetric_loss') and criterion.asymmetric_loss is not None:
        errors = pred - target
        under_predictions = (errors < 0).float().mean().item()
        loss_components["under_prediction_rate"] = under_predictions
    
    return loss_components


def create_prediction_visualization(predictions, targets, epoch, prefix="val"):
    """
    Create prediction visualization for wandb logging
    """
    fig = plt.figure(figsize=(10, 6))
    sample_idx = 0
    feature_idx = 0
    pred_np = predictions.cpu().numpy()
    target_np = targets.cpu().numpy()
    pred_series = pred_np[sample_idx, :, feature_idx]
    target_series = target_np[sample_idx, :, feature_idx]
    time_steps = range(1, len(pred_series) + 1)
    plt.plot(time_steps, target_series, label="Ground Truth", marker='o')
    plt.plot(time_steps, pred_series, label="Prediction", marker='x')
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title(f"{prefix.capitalize()}: Ground Truth vs Prediction (Epoch {epoch+1})")
    plt.legend()
    plt.grid(True)
    return fig


def train_model(model, train_loader, val_loader, test_loader, num_epochs, optimizer, criterion, device, config=None):
    print_header("Stockcaster Training", style="double")
    print_timestamp("Training started at")
    
    if config:
        print_config(config)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    scaler = GradScaler() if device.type == 'cuda' else None
    scheduler = AdaptiveTrainingScheduler(optimizer, scheduler_type='plateau', mode='min', patience=2, factor=0.5, verbose=True)
    
    if wandb.run is None and config is not None:
        wandb.init(project="stockcaster", config=config)
        wandb.run.summary["model_architecture"] = str(model)
    
    print_info(f"Training on {device} device")
    print_info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Print epoch header
        print_section_separator()
        print_subheader(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        print_progress(f"Training phase")
        model.train()
        total_loss = 0.0
        batch_count = len(train_loader)
        
        for i, batch in enumerate(train_loader):
            encoder = batch['encoder'].to(device)
            decoder = batch['decoder'].to(device)
            target = batch['target'].to(device)
            optimizer.zero_grad()
            
            if scaler:
                with autocast():
                    output = model(encoder, decoder, target)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(encoder, decoder, target)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_percentage = (i + 1) / batch_count
            progress_bar_length = 30
            filled_length = int(progress_percentage * progress_bar_length)
            bar = '█' * filled_length + '-' * (progress_bar_length - filled_length)
            current_loss = loss.item()
            print(f'\r{Colors.BRIGHT_BLUE}Training:{Colors.RESET} |{Colors.BRIGHT_CYAN}{bar}{Colors.RESET}| {progress_percentage*100:.1f}% Loss: {Colors.BRIGHT_YELLOW}{current_loss:.6f}{Colors.RESET}', end='')
            
            if wandb.run is not None:
                wandb.log({"batch_loss": loss.item()})
        
        # Print new line after progress bar
        print()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        print_progress(f"Validation phase")
        model.eval()
        total_val_loss = 0.0
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                encoder = batch['encoder'].to(device)
                decoder = batch['decoder'].to(device)
                target = batch['target'].to(device)
                output = model(encoder, decoder, target)
                loss = criterion(output, target)
                total_val_loss += loss.item()
                val_predictions.append(output)
                val_targets.append(target)
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_predictions = torch.cat(val_predictions, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        val_mae, val_rmse, val_mape = compute_metrics(val_predictions, val_targets)
        val_loss_components = compute_individual_losses(val_predictions, val_targets, criterion)
        
        # Test phase - only for monitoring, not for model selection
        print_progress(f"Test phase")
        total_test_loss = 0.0
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                encoder = batch['encoder'].to(device)
                decoder = batch['decoder'].to(device)
                target = batch['target'].to(device)
                output = model(encoder, decoder, target)
                loss = criterion(output, target)
                total_test_loss += loss.item()
                test_predictions.append(output)
                test_targets.append(target)
        
        avg_test_loss = total_test_loss / len(test_loader)
        test_predictions = torch.cat(test_predictions, dim=0)
        test_targets = torch.cat(test_targets, dim=0)
        test_mae, test_rmse, test_mape = compute_metrics(test_predictions, test_targets)
        test_loss_components = compute_individual_losses(test_predictions, test_targets, criterion)
        
        # Log all metrics
        epoch_duration = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics tables
        val_metrics = {
            "Loss": avg_val_loss,
            "MAE": val_mae,
            "RMSE": val_rmse,
            "MAPE": f"{val_mape:.2f}%"
        }
        print_metrics_table(val_metrics, "Validation Metrics")
        
        test_metrics = {
            "Loss": avg_test_loss,
            "MAE": test_mae,
            "RMSE": test_rmse,
            "MAPE": f"{test_mape:.2f}%"
        }
        print_metrics_table(test_metrics, "Test Metrics")
        
        # Print loss components if available
        if val_loss_components:
            print_subheader("Loss Components")
            for component, value in val_loss_components.items():
                print_metric(component, f"{value:.6f}", is_good=component.lower() not in ["mse", "mae", "combined_loss", "directional_penalty"])
                
        # Print additional info
        print_time_elapsed(epoch_start_time, "Epoch duration")
        print_info(f"Learning rate: {current_lr:.6e}")
        
        if wandb.run is not None:
            metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_mape": val_mape,
                "test_loss": avg_test_loss,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_mape": test_mape,
                "epoch_duration": epoch_duration,
                "learning_rate": current_lr,
            }
            
            # Add individual loss components to metrics
            for component_name, component_value in val_loss_components.items():
                metrics[f"val_{component_name}"] = component_value
            
            for component_name, component_value in test_loss_components.items():
                metrics[f"test_{component_name}"] = component_value
                
            wandb.log(metrics)
            
            # Create and log visualizations every epoch
            val_fig = create_prediction_visualization(val_predictions, val_targets, epoch, prefix="val")
            test_fig = create_prediction_visualization(test_predictions, test_targets, epoch, prefix="test")
            
            wandb.log({"val_predictions": wandb.Image(val_fig)})
            wandb.log({"test_predictions": wandb.Image(test_fig)})
            
            plt.close(val_fig)
            plt.close(test_fig)
            
            # Log model weights and gradients
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        wandb.log({f"weights/{name}": wandb.Histogram(param.detach().cpu().numpy())})
                        if param.grad is not None:
                            wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.detach().cpu().numpy())})

        # Model selection based on validation loss only
        scheduler.step(epoch, metric=avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            model_path = "best_informer_model.pt"
            torch.save(model.state_dict(), model_path)
            print_success(f"New best validation loss: {best_val_loss:.6f} - Model saved to {model_path}")
            if wandb.run is not None:
                artifact = wandb.Artifact(f"best_model", type="model")
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)
        else:
            epochs_without_improvement += 1
            print_warning(f"Validation loss did not improve. Best: {best_val_loss:.6f}, Current: {avg_val_loss:.6f}")
            print_info(f"Epochs without improvement: {epochs_without_improvement}")
        
        if epochs_without_improvement >= 3:
            print_warning("Early stopping triggered after 3 epochs without improvement.")
            break
    
    print_section_separator()
    print_success(f"Training completed with best validation loss: {best_val_loss:.6f}")
    print_timestamp("Training finished at")
    
    if wandb.run is not None:
        wandb.run.summary["best_val_loss"] = best_val_loss


def evaluate_model(model, test_loader, criterion, device):
    print_header("Model Evaluation", style="double")
    print_timestamp("Evaluation started at")
    
    start_time = time.time()
    model.eval()
    total_test_loss = 0.0
    predictions = []
    targets = []
    datetime_list = []
    
    print_progress("Evaluating model on test dataset")
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            encoder = batch['encoder'].to(device)
            decoder = batch['decoder'].to(device)
            target = batch['target'].to(device)
            output = model(encoder, decoder, target)
            loss = criterion(output, target)
            total_test_loss += loss.item()
            predictions.append(output)
            targets.append(target)
            if 'datetime' in batch:
                datetime_list.append(batch['datetime'])
            
            # Update progress
            progress_percentage = (i + 1) / len(test_loader)
            print_progress_bar(i+1, len(test_loader), prefix="Testing", suffix=f"Loss: {loss.item():.6f}")
    
    avg_test_loss = total_test_loss / len(test_loader)
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    datetimes = [dt for sublist in datetime_list for dt in sublist] if datetime_list else None
    
    # Calculate metrics
    mae, rmse, mape = compute_metrics(predictions, targets)
    
    # Print metrics table
    test_metrics = {
        "Loss": avg_test_loss,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": f"{mape:.2f}%"
    }
    print_metrics_table(test_metrics, "Final Test Metrics")
    
    # Calculate and print loss components
    loss_components = compute_individual_losses(predictions, targets, criterion)
    if loss_components:
        print_subheader("Loss Components")
        for component, value in loss_components.items():
            print_metric(component, f"{value:.6f}", is_good=component.lower() not in ["mse", "mae", "combined_loss", "directional_penalty"])
    
    print_time_elapsed(start_time, "Total evaluation time")
    print_timestamp("Evaluation finished at")
    print_section_separator()
    
    if wandb.run is not None:
        # Compute and log individual loss components for test data
        test_metrics = {
            "test_loss": avg_test_loss,
            "test_mae": mae,
            "test_rmse": rmse,
            "test_mape": mape,
        }
        
        # Add individual loss components to metrics
        for component_name, component_value in loss_components.items():
            test_metrics[f"test_{component_name}"] = component_value
            
        wandb.log(test_metrics)
        wandb.run.summary.update(test_metrics)
    
    return predictions, targets, datetimes 