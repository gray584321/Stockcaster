import os
import torch
import torch.nn as nn
from torch.optim import AdamW  # Changed from Adam to AdamW
import wandb
import pandas as pd
import numpy as np
import gc  # Add garbage collection
import math

from tsformer.model_components import TimeSeriesTransformer, PatchTST
from tsformer.training_utils import (
    set_random_seed, train_model, evaluate_model, compute_metrics,
    enable_gradient_checkpointing, clear_memory
)
from tsformer.dataloader import StockDataLoader
from tsformer.loss_functions import CombinedLoss, DirectionalLoss, AsymmetricLoss, TemporalWeightedLoss
from tsformer.terminal_utils import (
    print_header, print_subheader, print_success, print_info, 
    print_warning, print_section_separator, print_config, clear_terminal,
    print_timestamp
)

# Function to print current MPS memory usage
def print_mps_memory_usage():
    if torch.backends.mps.is_available():
        print_info(f"MPS Memory: Current allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
        print_info(f"MPS Memory: Driver allocated: {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")
        print_info(f"MPS Memory: Max allowed: {torch.mps.recommended_max_memory() / 1e9:.2f} GB")
    elif torch.cuda.is_available():
        print_info(f"CUDA Memory: Current allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print_info(f"CUDA Memory: Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print_info(f"CUDA Memory: Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

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
    dropout = 0.1  # Increased from 0.05 to better regularize Transformer
    encoder_length = 96
    decoder_length = 48
    prediction_length = 24
    batch_size = 32  # Further reduced from 32 to 24 for memory efficiency
    num_epochs = 60
    learning_rate = 3e-4  # Slightly adjusted for Transformer architecture

    # Enhanced model parameters
    model_type = 'patchtst'  # Options: 'transformer', 'patchtst'
    attention_type = 'probsparse'  # 'standard', 'probsparse', or 'autocorrelation'
    use_decomposition = True
    use_convolutional = True
    
    # PatchTST specific parameters
    patch_len = 16
    stride = 8
    
    # Training parameters
    use_scheduler = True
    use_early_stopping = True
    early_stopping_patience = 7
    use_amp = torch.cuda.is_available()  # Use mixed precision if CUDA is available
    clip_grad_norm = 1.0  # Maximum gradient norm for gradient clipping
    
    # Loss function hyperparameters
    mse_weight = 0.6
    mae_weight = 0.2
    directional_weight = 0.15
    asymmetric_weight = 0.05
    asymmetric_alpha = 1.5  # > 1 to penalize under-predictions more
    asymmetric_beta = 1.0   # > 1 to penalize over-predictions more
    use_time_features = True

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
    
    # Print initial memory state
    print_mps_memory_usage()

    # Load data
    print_section_separator()
    print_subheader("Data Loading")
    csv_path = "data/processed/SPY.csv"
    print_info(f"Loading data from: {csv_path}")
    
    # Use reduced set of technical indicators for memory efficiency
    max_features = 16 # Further reduced from 10 to 8 for memory efficiency
    print_info(f"Using feature selection with max_features={max_features}")
    
    # Create sequence config to pass to data loader
    sequence_config = {
        "encoder_length": encoder_length,
        "decoder_length": decoder_length,
        "prediction_length": prediction_length
    }
    
    stock_data_loader = StockDataLoader(
        csv_path, 
        sequence_config=sequence_config,
        use_cyclical_encoding=True, 
        include_technical_indicators=True,
        max_features=max_features,
        use_log_returns=True  # Enable log-based close pricing
    )
    
    # Print interim memory status before data processing
    print_info("Before data processing:")
    print_mps_memory_usage()
    
    # Prepare datasets with more detailed progress logging
    stock_data_loader.prepare_datasets()
    
    # Use pin_memory=True for faster data transfer to GPU
    pin_memory = True if device.type != 'cpu' else False
    print_info(f"Using pin_memory={pin_memory} for data transfer")
    
    # Use 0 workers to avoid memory issues, can increase if memory allows
    num_workers = 0
    print_info(f"Using num_workers={num_workers} for dataloaders")
    
    # Use a smaller batch size for memory efficiency
    print_info(f"Using reduced batch size of {batch_size} for memory efficiency")
    
    train_loader, val_loader, test_loader = stock_data_loader.get_dataloaders(
        batch_size=batch_size, 
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    
    # Clear memory after data loading
    clear_memory()
    print_info("After data loading:")
    print_mps_memory_usage()
    
    print_success(f"Data loaded successfully")
    print_info(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Print selected features if available
    if hasattr(stock_data_loader, 'selected_feature_names') and stock_data_loader.selected_feature_names:
        print_info(f"Selected features: {', '.join(stock_data_loader.selected_feature_names)}")
    
    # Dynamically determine input dimension from the dataloader features
    if hasattr(stock_data_loader, 'features') and stock_data_loader.features is not None:
        data_input_dim = stock_data_loader.features.shape[1]
    else:
        # Infer from the first batch if features are not directly accessible
        for batch in train_loader:
            data_input_dim = batch['encoder'].shape[2]  # [batch, seq_len, features]
            break
    
    print_info(f"Input dimension: {data_input_dim}")
    
    # Create config dictionary for wandb with dynamic input_dim
    config = {
        "model_type": model_type,
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
        "use_time_features": use_time_features,
        # Enhanced model parameters
        "attention_type": attention_type,
        "use_decomposition": use_decomposition,
        "use_convolutional": use_convolutional,
        # PatchTST specific parameters
        "patch_len": patch_len,
        "stride": stride,
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
    
    if model_type == 'transformer':
        model = TimeSeriesTransformer(
            input_dim=data_input_dim, 
            d_model=d_model, 
            d_ff=d_ff, 
            nhead=nhead,
            enc_layers=enc_layers, 
            dec_layers=dec_layers, 
            dropout=dropout,
            encoder_length=encoder_length, 
            decoder_length=decoder_length,
            prediction_length=prediction_length, 
            use_time_features=use_time_features,
            attention_type=attention_type,
            use_decomposition=use_decomposition,
            use_convolutional=use_convolutional
        )
        print_success("Enhanced Time Series Transformer model created and moved to device")
        print_info(f"Using attention type: {attention_type}")
        print_info(f"Series decomposition: {'Enabled' if use_decomposition else 'Disabled'}")
        print_info(f"Convolutional features: {'Enabled' if use_convolutional else 'Disabled'}")
    elif model_type == 'patchtst':
        model = PatchTST(
            input_dim=data_input_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=enc_layers,
            d_ff=d_ff,
            dropout=dropout,
            encoder_length=encoder_length,
            prediction_length=prediction_length,
            patch_len=patch_len,
            stride=stride
        )
        print_success("PatchTST model created and moved to device")
        print_info(f"Using patch length: {patch_len}")
        print_info(f"Using stride: {stride}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.to(device)
    print_mps_memory_usage()
    
    # Log model architecture
    wandb.run.summary["model_architecture"] = str(model)
    
    # Set up loss and optimizer
    print_section_separator()
    print_subheader("Loss and Optimizer Setup")
    
    # Create feature weights if needed - putting higher weight on close price (assumed to be index 0)
    feature_weights = None
    if data_input_dim > 1:
        feature_weights = torch.ones(data_input_dim)
        feature_weights[0] = 2.0  # Double weight for close price
        feature_weights = feature_weights / feature_weights.sum()  # Normalize
    
    criterion = CombinedLoss(
        mse_weight=mse_weight, 
        mae_weight=mae_weight,
        directional_weight=directional_weight,
        asymmetric_weight=asymmetric_weight,
        asymmetric_alpha=asymmetric_alpha,
        asymmetric_beta=asymmetric_beta,
        feature_weights=feature_weights,
        close_price_index=0  # Assuming close price is the first feature
    )
    
    # Use AdamW optimizer which has better weight decay handling for Transformers
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    print_success("Loss function and optimizer initialized")
    
    # Set up learning rate scheduler
    if use_scheduler:
        print_info("Setting up learning rate scheduler")
        # Warmup followed by cosine annealing
        total_steps = num_epochs * len(train_loader)
        warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print_success(f"Using warmup for {warmup_steps} steps followed by cosine decay")
    else:
        scheduler = None
    
    # Training loop
    print_section_separator()
    print_subheader("Training")
    
    # Check for model-specific forward method signature
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
    
    # Enable gradient checkpointing if available (for memory efficiency)
    if model_type in ['transformer', 'patchtst']:
        print_info(f"Enabling gradient checkpointing for {model_type} model")
        enable_gradient_checkpointing(model)
    else:
        print_info("Skipping gradient checkpointing - unsupported model type")
    
    # Train model
    best_model, train_losses, val_losses, val_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        model_type=model_type,
        model_forward_args=model_forward_args,
        scheduler=scheduler,
        patience=early_stopping_patience if use_early_stopping else None,
        use_amp=use_amp,  # Mixed precision training
        clip_grad_norm=clip_grad_norm  # Gradient clipping
    )
    
    # Evaluate the best model on the test set
    print_section_separator()
    print_subheader("Test Evaluation")
    
    # Load the best model
    try:
        model.load_state_dict(torch.load(f"best_{model_type}_model.pt"))
        print_success(f"Loaded best {model_type} model weights")
    except:
        print_warning("Could not load best model weights, using current model")
    
    # Evaluate on test data
    model.eval()
    test_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        model_type=model_type,
        model_forward_args=model_forward_args
    )
    
    # Print test metrics
    print_info("Test metrics:")
    for metric_name, metric_value in test_metrics.items():
        formatted_value = f"{metric_value:.4f}" if isinstance(metric_value, float) else metric_value
        print_info(f"  - {metric_name}: {formatted_value}")
    
    # Log final test metrics to wandb
    if wandb.run is not None:
        for metric_name, metric_value in test_metrics.items():
            wandb.run.summary[f"test_{metric_name}"] = metric_value
    
    print_section_separator()
    print_success("Experiment completed successfully!")
    print_timestamp("Script finished at")
    
    # Close wandb
    if wandb.run is not None:
        wandb.finish()
    
    # Final memory cleanup
    clear_memory()
    print_mps_memory_usage()
    
    print_section_separator()
    print_header("Stockcaster Execution Completed! 🎉", style="hash")
    print_timestamp("Script finished at") 