# Stockcaster

## Overview
Stockcaster is a machine learning-based stock price prediction system that uses state-of-the-art time series forecasting techniques with transformer models. It implements the PatchTST (Patch Time Series Transformer) architecture to analyze historical stock data and predict future price movements.

## Features
- Advanced time series forecasting using PatchTST (Patch Time Series Transformer)
- Data fetching from financial data providers
- Comprehensive technical indicator generation
- Robust data preprocessing pipeline
- Model training with early stopping and checkpointing
- Visualization of predictions vs. actual prices
- Pre-trained models for immediate use
- Support for multiple stocks (currently includes SPY, AAPL, TSLA)
- Integration with Weights & Biases for experiment tracking

## Architecture
Stockcaster uses a patched-based transformer architecture (PatchTST) that divides time series data into patches for more efficient and effective processing:

1. **Data Processing**: 
   - Historical price data is loaded and cleaned
   - Technical indicators are computed using the TA-Lib wrapper
   - Data is normalized and prepared as input sequences

2. **Model Architecture**:
   - **Patch Embedding**: Converts input sequences into patches and projects them into embedding space
   - **Transformer Encoder**: Processes embedded patches with self-attention mechanisms
   - **Prediction Head**: Generates predictions from the CLS token representation

3. **Training Pipeline**:
   - Supervised training for price prediction
   - Unsupervised pre-training using reconstruction objectives
   - Early stopping and model checkpointing

## Directory Structure
```
stockcaster/
│
├── data/                   # Data handling and processing
│   ├── raw/                # Raw data files
│   ├── processed/          # Processed data (SPY, AAPL, TSLA)
│   ├── config.py           # Data configuration parameters
│   ├── data_fetcher.py     # Data fetching utilities
│   ├── data_fetcher_mass.py# Batch data fetching 
│   └── dataset_creator.py  # Dataset preparation
│
├── models/                 # Model definitions
│   ├── patch_tst.py        # PatchTST model architecture
│   ├── data_processor.py   # Data preprocessing utilities
│   └── trainer.py          # Training and evaluation utilities
│
├── checkpoint/             # Saved model checkpoints
│
├── wandb/                  # Weights & Biases experiment logs
│
├── main.py                 # Main execution script
├── pretrained_model.pth    # Pre-trained model weights
├── requirements.txt        # Project dependencies
└── .env                    # Environment variables (API keys)
```

## Requirements
- Python 3.8+
- PyTorch 2.0+
- pandas 1.5+
- numpy 1.21+
- matplotlib 3.5+
- scikit-learn 1.0+
- TA-Lib (technical analysis library)
- Weights & Biases for experiment tracking

See `requirements.txt` for the complete list of dependencies.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stockcaster.git
   cd stockcaster
   ```

2. Create a virtual environment (recommended):
   ```bash
   # Using venv
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Or using conda
   conda create -n stockcaster python=3.10
   conda activate stockcaster
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your Polygon API key to the `.env` file

## Usage

### Data Collection
To fetch new stock data:
```bash
python data/data_fetcher.py --symbol SPY --start 2010-01-01 --end 2023-12-31
```

To fetch multiple stocks at once:
```bash
python data/data_fetcher_mass.py --symbols SPY,AAPL,TSLA --start 2010-01-01 --end 2023-12-31
```

### Running Predictions
To train a model and make predictions:
```bash
python main.py
```

This will:
1. Load and preprocess the stock data
2. Train the PatchTST model (or use a pre-trained model)
3. Make predictions for the configured prediction horizon
4. Display and log the results

### Configuration
Model and training parameters can be adjusted in the `CONFIG` dictionary in `main.py`:

```python
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
    'd_ff': 256,             # Dimension of feedforward network
    # ...other parameters...
}
```

## Performance
The PatchTST model generally outperforms traditional time-series forecasting methods and many deep learning approaches on financial time series data. The actual performance will vary depending on:

- The specific stock being predicted
- The market conditions
- The configuration parameters
- The quality and quantity of training data

Typical metrics on test data:
- MAPE (Mean Absolute Percentage Error): ~2-5%
- MAE (Mean Absolute Error): varies by stock price range
- Directional Accuracy: ~55-65%

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Future Work
- Support for multi-variate forecasting (predicting multiple values)
- Integration with more data sources
- Ensemble methods combining multiple models
- Reinforcement learning for trading strategies
- Web interface for easy visualization and interaction
