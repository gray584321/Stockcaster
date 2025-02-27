import os
import torch
import torch.nn as nn
from torch.optim import Adam
import wandb
import pandas as pd
import numpy as np

from informer.model_components import Informer
from informer.training_utils import set_random_seed, train_model, evaluate_model, compute_metrics
from informer.dataloader import StockDataLoader
from informer.loss_functions import CombinedLoss, DirectionalLoss, AsymmetricLoss
from informer.terminal_utils import (
    print_header, print_subheader, print_success, print_info, 
    print_warning, print_section_separator, print_config, clear_terminal,
    print_timestamp
)


if __name__ == "__main__":
    # Clear the terminal for a fresh start
    clear_terminal()
    
    print_header("🚀 Stockcaster - Time Series Stock Prediction 📈", style="double")
    print_timestamp("Script started at")
    
    # Model hyperparameters
    d_model = 128
    d_ff = 512
    nhead = 8
    enc_layers = 3
    dec_layers = 3
    dropout = 0.1
    encoder_length = 96
    decoder_length = 48
    prediction_length = 24
    batch_size = 64
    num_epochs = 50
    learning_rate = 4e-5

    # Loss function hyperparameters
    mse_weight = 0.7
    mae_weight = 0.3
    directional_weight = 0.25  # Set > 0 to enable
    asymmetric_weight = 0.25   # Set > 0 to enable
    asymmetric_alpha = 1.5    # > 1 to penalize under-predictions more
    asymmetric_beta = 1.0     # > 1 to penalize over-predictions more

    # Set up device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Set random seed
    seed = 69
    set_random_seed(seed)
    print_info(f"Random seed set to: {seed}")
    print_info(f"Using device: {device}")

    # Load data
    print_section_separator()
    print_subheader("Data Loading")
    csv_path = "data/processed/SPY.csv"
    print_info(f"Loading data from: {csv_path}")
    
    stock_data_loader = StockDataLoader(csv_path, use_cyclical_encoding=True, include_technical_indicators=False)
    stock_data_loader.prepare_datasets()
    train_loader, val_loader, test_loader = stock_data_loader.get_dataloaders(batch_size=batch_size)
    
    print_success(f"Data loaded successfully")
    print_info(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Dynamically determine input dimension from the dataloader features
    data_input_dim = stock_data_loader.features.shape[1]
    print_info(f"Input dimension: {data_input_dim}")
    
    # Create config dictionary for wandb with dynamic input_dim
    config = {
        "model_type": "Informer",
        "input_dim": data_input_dim,
        "d_model": d_model,
        "d_ff": d_ff,
        "nhead": nhead,
        "enc_layers": enc_layers,
        "dec_layers": dec_layers,
        "dropout": dropout,
        "encoder_length": encoder_length,
        "decoder_length": decoder_length,
        "prediction_length": prediction_length,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": str(device),
        # Loss function parameters
        "loss_function": "CombinedLoss",
        "mse_weight": mse_weight,
        "mae_weight": mae_weight,
        "directional_weight": directional_weight,
        "asymmetric_weight": asymmetric_weight,
        "asymmetric_alpha": asymmetric_alpha,
        "asymmetric_beta": asymmetric_beta
    }
    
    # Initialize wandb with the updated config
    print_section_separator()
    print_subheader("Wandb Initialization")
    wandb.init(project="stockcaster", config=config)
    print_success("Wandb initialized")
    
    # Create model using dynamic input dimension
    print_section_separator()
    print_subheader("Model Creation")
    model = Informer(input_dim=data_input_dim, d_model=d_model, d_ff=d_ff, nhead=nhead,
                     enc_layers=enc_layers, dec_layers=dec_layers, dropout=dropout,
                     encoder_length=encoder_length, decoder_length=decoder_length,
                     prediction_length=prediction_length, use_prob_sparse=True)
    model.to(device)
    print_success("Model created and moved to device")
    
    # Log model architecture
    wandb.run.summary["model_architecture"] = str(model)
    
    # Set up loss and optimizer
    print_section_separator()
    print_subheader("Loss and Optimizer Setup")
    criterion = CombinedLoss(
        mse_weight=mse_weight, 
        mae_weight=mae_weight,
        directional_weight=directional_weight,
        asymmetric_weight=asymmetric_weight,
        asymmetric_alpha=asymmetric_alpha,
        asymmetric_beta=asymmetric_beta
    )
    optimizer = Adam(model.parameters(), lr=learning_rate)
    print_success("Loss function and optimizer initialized")
    
    # Train model and evaluate on test data during training
    print_section_separator()
    train_model(model, train_loader, val_loader, test_loader, num_epochs, optimizer, criterion, device, config)  
    
    # Final evaluation on test data
    print_section_separator()
    print_subheader("Final Model Evaluation")
    predictions, targets, datetimes = evaluate_model(model, test_loader, criterion, device)
    
    # Additionally calculate metrics with standard MSE for comparison
    standard_criterion = nn.MSELoss()
    with torch.no_grad():
        standard_loss = standard_criterion(predictions, targets).item()
    mae, rmse, mape = compute_metrics(predictions, targets)
    print_info(f"Standard MSE Loss: {standard_loss:.6f}")
    
    # Denormalize predictions for visualization
    print_section_separator()
    print_subheader("Visualizing Results")
    pred_np = predictions.cpu().numpy()
    target_np = targets.cpu().numpy()
    close_price_mean = 100.0
    close_price_std = 20.0
    pred_np[:, :, 0] = pred_np[:, :, 0] * close_price_std + close_price_mean
    target_np[:, :, 0] = target_np[:, :, 0] * close_price_std + close_price_mean
    
    # Create and save visualization
    sample_idx = 0
    feature_idx = 0
    pred_series = pred_np[sample_idx, :, feature_idx]
    target_series = target_np[sample_idx, :, feature_idx]
    time_steps = range(1, len(pred_series) + 1)
    
    results_folder = "results"
    os.makedirs(results_folder, exist_ok=True)
    print_info(f"Created results folder: {results_folder}")
    
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 6))
    plt.plot(time_steps, target_series, label="Ground Truth", marker='o')
    plt.plot(time_steps, pred_series, label="Prediction", marker='x')
    plt.xlabel("Time Step")
    plt.ylabel("Close Price")
    plt.title(f"Ground Truth vs Prediction (Sample {sample_idx}, Feature {feature_idx})")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(results_folder, "ground_truth_vs_prediction.png")
    plt.savefig(plot_path)
    
    if wandb.run is not None:
        wandb.log({"final_test_predictions": wandb.Image(fig)})
    
    plt.close()
    print_success(f"Saved prediction plot to {plot_path}")
    
    # Create and save results dataframe
    print_section_separator()
    print_subheader("Saving Results")
    
    num_samples, pred_length, _ = pred_np.shape
    sample_ids = np.repeat(np.arange(num_samples), pred_length)
    time_steps_full = np.tile(np.arange(pred_length), num_samples)
    ground_truth_flat = target_np[:, :, feature_idx].flatten()
    prediction_flat = pred_np[:, :, feature_idx].flatten()
    
    df = pd.DataFrame({
        "sample": sample_ids,
        "time_step": time_steps_full,
        "ground_truth": ground_truth_flat,
        "prediction": prediction_flat
    })
    if datetimes is not None:
        if len(datetimes) == num_samples * pred_length:
            df["datetime"] = datetimes
        elif len(datetimes) == num_samples:
            df["datetime"] = np.repeat(datetimes, pred_length)
        else:
            df["datetime"] = None
    else:
        df["datetime"] = None
    
    csv_path = os.path.join(results_folder, "test_predictions_vs_ground_truth.csv")
    df.to_csv(csv_path, index=False)
    print_success(f"Saved raw predictions and ground truth data to {csv_path}")
    
    if wandb.run is not None:
        wandb.log({"predictions_table": wandb.Table(dataframe=df)})
        wandb.finish()
        print_success("Wandb run completed and finalized")
    
    print_section_separator()
    print_header("Stockcaster Execution Completed! 🎉", style="hash")
    print_timestamp("Script finished at") 