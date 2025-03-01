import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import ta
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class DataProcessor:
    def __init__(self, seq_len=100, pred_len=30):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scalers = {}
        self._cached_indicators = {}
        self.price_scaler = None
        self.volume_scaler = None
        self.returns_scaler = None
        
    def _compute_indicator(self, df, indicator_fn, name):
        """Compute a single technical indicator with caching"""
        if name not in self._cached_indicators:
            self._cached_indicators[name] = indicator_fn(df)
        return self._cached_indicators[name]
        
    def add_technical_indicators(self, df):
        """Add technical indicators using parallel processing"""
        # Price-based indicators (vectorized)
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Define indicator functions
        indicator_fns = {
            'volume_ma5': lambda d: ta.volume.volume_weighted_average_price(
                high=d['high'], low=d['low'], close=d['close'], volume=d['volume'], window=5
            ),
            'ema_9': lambda d: ta.trend.ema_indicator(d['close'], window=9),
            'sma_20': lambda d: ta.trend.sma_indicator(d['close'], window=20),
            'macd': lambda d: ta.trend.macd_diff(d['close']),
            'rsi': lambda d: ta.momentum.rsi(d['close']),
            'stoch_k': lambda d: ta.momentum.stoch(d['high'], d['low'], d['close']),
            'bb_high': lambda d: ta.volatility.bollinger_hband(d['close']),
            'bb_low': lambda d: ta.volatility.bollinger_lband(d['close']),
            'atr': lambda d: ta.volatility.average_true_range(d['high'], d['low'], d['close'])
        }
        
        # Compute indicators in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                name: executor.submit(self._compute_indicator, df, fn, name)
                for name, fn in indicator_fns.items()
            }
            
            # Get results
            for name, future in futures.items():
                df[name] = future.result()
        
        return df
    
    def _normalize_price_data(self, df, column, is_train=True):
        """Normalize price-based columns using log-returns and z-score normalization"""
        if is_train:
            # Store the last price for denormalization
            self.last_price = df[column].iloc[-1]
            # Calculate returns
            returns = np.log(df[column]) - np.log(df[column].shift(1))
            self.returns_mean = returns.mean()
            self.returns_std = returns.std()
            normalized = (returns - self.returns_mean) / (self.returns_std + 1e-8)
        else:
            returns = np.log(df[column]) - np.log(df[column].shift(1))
            normalized = (returns - self.returns_mean) / (self.returns_std + 1e-8)
        
        return normalized.fillna(0)
    
    def _normalize_volume_data(self, df, column, is_train=True):
        """Normalize volume data using log transformation and z-score normalization"""
        if is_train:
            log_volume = np.log1p(df[column])
            self.volume_mean = log_volume.mean()
            self.volume_std = log_volume.std()
            normalized = (log_volume - self.volume_mean) / (self.volume_std + 1e-8)
        else:
            log_volume = np.log1p(df[column])
            normalized = (log_volume - self.volume_mean) / (self.volume_std + 1e-8)
        
        return normalized
    
    def _normalize_indicator_data(self, df, column, is_train=True):
        """Normalize technical indicators using z-score normalization"""
        if is_train:
            self.scalers[column] = {
                'mean': df[column].mean(),
                'std': df[column].std()
            }
        
        normalized = (df[column] - self.scalers[column]['mean']) / (self.scalers[column]['std'] + 1e-8)
        return normalized
    
    def _denormalize_price(self, normalized_values):
        """Denormalize the price predictions"""
        # Convert back to returns
        returns = normalized_values * (self.returns_std + 1e-8) + self.returns_mean
        # Convert returns to price levels
        price_levels = self.last_price * np.exp(returns.cumsum())
        return price_levels
        
    def prepare_data(self, df, is_train=True):
        """Prepare data sequences using vectorized operations with improved normalization and data quality checks"""
        print(f"\nPreparing data with shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Data quality checks
        if df.index.duplicated().any():
            print("Warning: Duplicate timestamps found. Keeping first occurrence.")
            df = df[~df.index.duplicated(keep='first')]
            
        # Check for infinite values
        inf_mask = np.isinf(df.values)
        if inf_mask.any():
            print("Warning: Infinite values found. Replacing with NaN.")
            df = df.replace([np.inf, -np.inf], np.nan)
            
        # Add technical indicators
        df = self.add_technical_indicators(df)
        print(f"Shape after adding technical indicators: {df.shape}")
        
        # Forward fill any remaining NaN values
        df = df.ffill()
        
        # Backward fill any remaining NaN values at the start
        df = df.bfill()
        
        # Verify data quality after cleaning
        if df.isnull().any().any():
            raise ValueError("Data still contains NaN values after cleaning!")
            
        # Normalize different types of features
        price_columns = ['open', 'high', 'low', 'close']
        volume_columns = ['volume', 'volume_ma5']
        indicator_columns = ['ema_9', 'sma_20', 'macd', 'rsi', 'stoch_k', 'bb_high', 'bb_low', 'atr']
        
        normalized_data = {}
        
        # Normalize price data
        for col in price_columns:
            normalized_data[col] = self._normalize_price_data(df, col, is_train)
            
        # Normalize volume data
        for col in volume_columns:
            normalized_data[col] = self._normalize_volume_data(df, col, is_train)
            
        # Normalize technical indicators
        for col in indicator_columns:
            if col in df.columns:
                normalized_data[col] = self._normalize_indicator_data(df, col, is_train)
        
        # Create feature matrix
        feature_columns = price_columns + volume_columns + indicator_columns
        feature_matrix = np.column_stack([normalized_data[col] for col in feature_columns if col in normalized_data])
        print(f"Feature matrix shape: {feature_matrix.shape}")
        
        # Verify no NaN or infinite values in feature matrix
        if not np.isfinite(feature_matrix).all():
            raise ValueError("Feature matrix contains NaN or infinite values after normalization!")
        
        # Create sequences and targets using sliding window view for memory efficient operation
        n_samples = len(df) - self.seq_len - self.pred_len + 1
        print(f"Number of samples that will be created: {n_samples}")

        # Ensure feature matrix is float32 to reduce memory usage
        feature_matrix = np.asarray(feature_matrix, dtype=np.float32)

        # Use sliding_window_view to create sequences
        sequences = np.lib.stride_tricks.sliding_window_view(feature_matrix, window_shape=self.seq_len, axis=0)[:n_samples]

        # Prepare targets from the normalized 'close' column, cast to float32
        close_array = np.asarray(normalized_data['close'], dtype=np.float32)
        targets = np.lib.stride_tricks.sliding_window_view(close_array, window_shape=self.pred_len, axis=0)[self.seq_len:self.seq_len+n_samples]
        
        # Final verification of sequences and targets
        if not np.isfinite(sequences).all() or not np.isfinite(targets).all():
            raise ValueError("Final sequences or targets contain NaN or infinite values!")
            
        print(f"Final shapes - Sequences: {sequences.shape}, Targets: {targets.shape}")
        return sequences, targets
    
    def inverse_transform(self, scaled_values):
        """Inverse transform the scaled values back to original price scale"""
        return self._denormalize_price(scaled_values) 