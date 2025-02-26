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


def train_model(model, train_loader, val_loader, num_epochs, optimizer, criterion, device, config=None):
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    scaler = GradScaler() if device.type == 'cuda' else None
    scheduler = AdaptiveTrainingScheduler(optimizer, scheduler_type='plateau', mode='min', patience=2, factor=0.5, verbose=True)
    
    if wandb.run is None and config is not None:
        wandb.init(project="stockcaster", config=config)
        wandb.run.summary["model_architecture"] = str(model)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for batch in pbar:
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
            pbar.set_postfix(loss=f"{loss.item():.6f}")
            
            if wandb.run is not None:
                wandb.log({"batch_loss": loss.item()})
            
        avg_train_loss = total_loss / len(train_loader)
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
        mae, rmse, mape = compute_metrics(val_predictions, val_targets)
        epoch_duration = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
              f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.2f}% , Duration: {epoch_duration:.2f}s, "
              f"Best Val Loss: {best_val_loss:.6f}, LR: {current_lr:.6e}")
        
        if wandb.run is not None:
            metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "epoch_duration": epoch_duration,
                "learning_rate": current_lr
            }
            wandb.log(metrics)
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                fig = plt.figure(figsize=(10, 6))
                sample_idx = 0
                feature_idx = 0
                pred_np = val_predictions.cpu().numpy()
                target_np = val_targets.cpu().numpy()
                pred_series = pred_np[sample_idx, :, feature_idx]
                target_series = target_np[sample_idx, :, feature_idx]
                time_steps = range(1, len(pred_series) + 1)
                plt.plot(time_steps, target_series, label="Ground Truth", marker='o')
                plt.plot(time_steps, pred_series, label="Prediction", marker='x')
                plt.xlabel("Time Step")
                plt.ylabel("Value")
                plt.title(f"Validation: Ground Truth vs Prediction (Epoch {epoch+1})")
                plt.legend()
                plt.grid(True)
                wandb.log({"val_predictions": wandb.Image(fig)})
                plt.close(fig)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    wandb.log({f"weights/{name}": wandb.Histogram(param.detach().cpu().numpy())})
                    if param.grad is not None:
                        wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.detach().cpu().numpy())})

        scheduler.step(epoch, metric=avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            model_path = "best_informer_model.pt"
            torch.save(model.state_dict(), model_path)
            if wandb.run is not None:
                artifact = wandb.Artifact(f"best_model", type="model")
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= 3:
            print("Early stopping triggered.")
            break
    
    if wandb.run is not None:
        wandb.run.summary["best_val_loss"] = best_val_loss


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_test_loss = 0.0
    predictions = []
    targets = []
    datetime_list = []
    
    with torch.no_grad():
        for batch in test_loader:
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
    
    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.6f}")
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    datetimes = [dt for sublist in datetime_list for dt in sublist] if datetime_list else None
    
    if wandb.run is not None:
        mae, rmse, mape = compute_metrics(predictions, targets)
        test_metrics = {
            "test_loss": avg_test_loss,
            "test_mae": mae,
            "test_rmse": rmse,
            "test_mape": mape
        }
        wandb.log(test_metrics)
        wandb.run.summary.update(test_metrics)
    
    return predictions, targets, datetimes 