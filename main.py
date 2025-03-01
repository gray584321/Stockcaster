import os
import torch
import wandb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from io import BytesIO

from models.patch_tst import PatchTST
from models.data_processor import DataProcessor
from models.trainer import PatchTSTTrainer

# Configuration dictionary for easy modification
CONFIG = {
    # Data parameters
    'seq_len': 100,          # Length of input sequence
    'pred_len': 30,          # Length of prediction sequence
    'data_path': 'data/processed/SPY.csv',
    
    # Training parameters
    'batch_size': 128,
    'num_epochs': 100,
    'learning_rate': 1e-5,
    'patience': 10,
    
    # Model parameters
    'd_model': 128,          # Dimension of model
    'n_heads': 4,            # Number of attention heads
    'n_layers': 3,           # Number of transformer layers
    'd_ff': 256,            # Dimension of feedforward network
    'dropout': 0.1,         # Dropout rate
    'fc_dropout': 0.1,      # Fully connected dropout rate
    'head_dropout': 0.1,    # Head dropout rate
    
    # Patch parameters will be set dynamically
    'patch_len_factor': 0.1,  # patch_len will be this fraction of seq_len
    'stride_factor': 0.5,     # stride will be this fraction of patch_len
}

def calculate_patch_params(seq_len):
    """Calculate patch_len and stride based on sequence length"""
    # Calculate patch_len as a fraction of sequence length
    patch_len = max(int(seq_len * CONFIG['patch_len_factor']), 1)
    # Calculate stride as a fraction of patch_len
    stride = max(int(patch_len * CONFIG['stride_factor']), 1)
    return patch_len, stride

def print_gpu_memory_usage():
    """Print GPU memory usage for either CUDA or MPS"""
    if torch.backends.mps.is_available():
        print(f"MPS Memory: Current allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
        print(f"MPS Memory: Driver allocated: {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")
    elif torch.cuda.is_available():
        print(f"CUDA Memory: Current allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"CUDA Memory: Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

def load_data(file_path, use_chunks=False, chunksize=100000):
    """Load and preprocess the data"""
    print(f"Loading data from {file_path}...")
    
    # If use_chunks is enabled, process CSV in chunks to reduce memory footprint
    if use_chunks:
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            try:
                chunk['datetime'] = pd.to_datetime(chunk['datetime'])
            except (ValueError, TypeError):
                try:
                    chunk['datetime'] = pd.to_datetime(chunk['datetime'], format='%Y-%m-%d %H:%M:%S')
                except ValueError:
                    chunk['datetime'] = pd.to_datetime(chunk['datetime'], infer_datetime_format=True)
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    else:
        try:
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
        except (ValueError, TypeError):
            try:
                df = pd.read_csv(file_path, parse_dates=['datetime'], date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
            except ValueError:
                df = pd.read_csv(file_path)
                df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True)
    
    # Ensure 'datetime' column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    # Convert timezone-aware dates to naive dates if timezone info exists
    if df['datetime'].dt.tz is not None:
        df['datetime'] = df['datetime'].dt.tz_localize(None)
    
    df.set_index('datetime', inplace=True)
    
    # Sort index to ensure chronological order
    df.sort_index(inplace=True)
    
    # Determine the data frequency
    freq = pd.infer_freq(df.index)
    if freq is None:
        # If can't infer, check most common time difference
        time_diff = df.index.to_series().diff().mode()[0]
        freq = pd.Timedelta(time_diff).resolution_string
    print(f"Detected data frequency: {freq}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("Found missing values:")
        print(missing_values[missing_values > 0])
        
        # Handle missing values using appropriate methods for financial time series
        # 1. For very short gaps (1-2 points), use linear interpolation
        df = df.interpolate(method='linear', limit=2)
        
        # 2. For slightly longer gaps (3-5 points), use cubic interpolation
        df = df.interpolate(method='cubic', limit=5)
        
        # 3. For remaining gaps, use forward fill followed by backward fill
        df = df.ffill().bfill()
        
    # Verify no missing values remain
    if df.isnull().sum().any():
        print("Warning: Some missing values could not be filled!")
        
    # Remove any duplicate indices
    df = df[~df.index.duplicated(keep='first')]
    
    # Ensure the index is continuous at the detected frequency
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_idx)
    
    # Fill any new missing values created by reindexing
    df = df.ffill().bfill()
    
    # Optimize memory usage: convert float64 columns to float32
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    print("Data cleaning completed.")
    print(f"Final dataset shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def plot_predictions_vs_actual(y_true, y_pred, title):
    """Create a plot comparing predictions vs actual values"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Ground Truth', color='blue')
    plt.plot(y_pred, label='Predictions', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    
    # Save plot to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return buf

def main():
    print("\nRunning with configuration:")
    for key, value in CONFIG.items():
        print(f"{key}: {value}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() 
                         else "mps" if torch.backends.mps.is_available() 
                         else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    df = load_data(CONFIG['data_path'])
    
    # Initialize data processor
    data_processor = DataProcessor(seq_len=CONFIG['seq_len'], pred_len=CONFIG['pred_len'])
    
    # Prepare sequences and targets with timing
    print("Preparing data sequences...")
    start_time = time.time()
    sequences, targets = data_processor.prepare_data(df.copy())
    prep_time = time.time() - start_time
    print(f"Data preparation took {prep_time:.2f} seconds")
    
    # Print sequence shape for debugging
    print(f"Sequence shape: {sequences.shape}")
    
    # Calculate dynamic patch parameters based on sequence length
    patch_len, stride = calculate_patch_params(sequences.shape[1])
    print(f"\nDynamic patch parameters:")
    print(f"patch_len: {patch_len}")
    print(f"stride: {stride}")
    
    # Split data into train, validation, and test sets
    train_seq, temp_seq, train_targets, temp_targets = train_test_split(
        sequences, targets, test_size=0.3, random_state=42
    )
    val_seq, test_seq, val_targets, test_targets = train_test_split(
        temp_seq, temp_targets, test_size=0.5, random_state=42
    )
    
    print(f"Data shapes:")
    print(f"Train: {train_seq.shape}")
    print(f"Validation: {val_seq.shape}")
    print(f"Test: {test_seq.shape}")
    
    # Initialize model with adjusted patch parameters
    print("\nInitializing PatchTST model...")
    model = PatchTST(
        input_dim=train_seq.shape[2],  # Number of features
        output_dim=1,  # Predicting close price
        patch_len=patch_len,
        stride=stride,
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        d_ff=CONFIG['d_ff'],
        dropout=CONFIG['dropout'],
        seq_len=CONFIG['seq_len'],
        pred_len=CONFIG['pred_len'],
        fc_dropout=CONFIG['fc_dropout'],
        head_dropout=CONFIG['head_dropout']
    )
    
    # Initialize trainer with W&B logging
    trainer = PatchTSTTrainer(
        model=model,
        device=device,
        batch_size=CONFIG['batch_size'],
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        patience=CONFIG['patience'],
        use_wandb=True
    )
    
    # Print GPU memory usage
    print("\nInitial GPU memory usage:")
    print_gpu_memory_usage()
    
    # Unsupervised pre-training
    print("\nStarting unsupervised pre-training...")
    trainer.pretrain(train_seq)
    
    # Load best pre-trained weights
    trainer.load_pretrained()
    
    # Supervised training with test metrics logging
    print("\nStarting supervised training...")
    trainer.train(
        train_seq=train_seq,
        train_targets=train_targets,
        val_seq=val_seq,
        val_targets=val_targets,
        test_seq=test_seq,
        test_targets=test_targets
    )
    
    # Load best model for evaluation
    trainer.load_best_model()
    
    # Generate predictions on test set
    print("\nGenerating test predictions...")
    predictions = trainer.predict(test_seq)
    
    # Inverse transform predictions and targets
    predictions = data_processor.inverse_transform(predictions.reshape(-1, 1))
    test_targets = data_processor.inverse_transform(test_targets.reshape(-1, 1))
    
    # Calculate metrics
    mse = np.mean((predictions - test_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - test_targets))
    mape = calculate_mape(test_targets, predictions)
    
    print("\nTest Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Create and log prediction vs actual plot
    plot_buf = plot_predictions_vs_actual(
        test_targets, 
        predictions, 
        'Test Set: Predictions vs Ground Truth'
    )
    
    # Log final test metrics and plot to W&B
    if wandb.run is not None:
        wandb.log({
            "final_test/mse": mse,
            "final_test/rmse": rmse,
            "final_test/mae": mae,
            "final_test/mape": mape,
            "final_test/predictions_vs_actual": wandb.Image(plot_buf),
            "final_test/predictions": wandb.Table(
                data=[[i, float(true), float(pred)] for i, (true, pred) in enumerate(zip(test_targets, predictions))],
                columns=["timestep", "ground_truth", "prediction"]
            )
        })
        
        # Log histograms of predictions and errors
        wandb.log({
            "final_test/prediction_distribution": wandb.Histogram(predictions),
            "final_test/error_distribution": wandb.Histogram(predictions - test_targets),
        })
        
        # Log scatter plot of predicted vs actual values
        wandb.log({
            "final_test/predicted_vs_actual_scatter": wandb.plot.scatter(
                wandb.Table(data=[[x, y] for x, y in zip(test_targets, predictions)],
                          columns=["ground_truth", "predictions"]),
                "ground_truth",
                "predictions"
            )
        })
    
    # Example of transfer learning on new data
    print("\nExample of transfer learning on new data:")
    new_df = df.copy()  # In practice, this would be new data
    new_sequences, new_targets = data_processor.prepare_data(new_df, is_train=False)
    
    # Fine-tune on new data
    trainer.train(new_sequences, new_targets)
    
    # Final GPU memory usage
    print("\nFinal GPU memory usage:")
    print_gpu_memory_usage()

if __name__ == "__main__":
    main() 