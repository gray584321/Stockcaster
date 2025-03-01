import pandas as pd
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest
import math
import gc  # Add gc for garbage collection
import os
import psutil  # For memory tracking


class StockDataset(Dataset):
    def __init__(self, encoder_inputs, decoder_inputs, targets, encoder_times=None, decoder_times=None, pin_memory=False):
        # Ensure all data is float32 to reduce memory usage
        self.encoder_inputs = encoder_inputs.astype(np.float32) if not isinstance(encoder_inputs, torch.Tensor) else encoder_inputs
        self.decoder_inputs = decoder_inputs.astype(np.float32) if not isinstance(decoder_inputs, torch.Tensor) else decoder_inputs
        self.targets = targets.astype(np.float32) if not isinstance(targets, torch.Tensor) else targets
        
        if encoder_times is not None and decoder_times is not None:
            self.encoder_times = encoder_times.astype(np.float32) if not isinstance(encoder_times, torch.Tensor) else encoder_times
            self.decoder_times = decoder_times.astype(np.float32) if not isinstance(decoder_times, torch.Tensor) else decoder_times
            self.has_time_features = True
        else:
            self.encoder_times = None
            self.decoder_times = None
            self.has_time_features = False
            
        self.pin_memory = pin_memory

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Convert to tensor with appropriate dtype (already float32 from initialization)
        encoder = torch.tensor(self.encoder_inputs[idx], dtype=torch.float32)
        decoder = torch.tensor(self.decoder_inputs[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        result = {
            'encoder': encoder,
            'decoder': decoder,
            'target': target
        }
        
        if self.has_time_features:
            result['encoder_time'] = torch.tensor(self.encoder_times[idx], dtype=torch.float32)
            result['decoder_time'] = torch.tensor(self.decoder_times[idx], dtype=torch.float32)
        
        # Pin memory if requested (helps with GPU transfer)
        if self.pin_memory:
            for k in result:
                result[k] = result[k].pin_memory()
            
        return result


class StockDataLoader:
    def __init__(self, csv_path, sequence_config=None, date_col='datetime', use_cyclical_encoding=True, 
                 include_technical_indicators=True, max_features=10, use_log_returns=True):
        """
        Initializes the data loader for historical stock data.

        Parameters:
            csv_path (str): Path to the CSV file containing stock data.
            sequence_config (dict): Dict specifying 'encoder_length', 'decoder_length', and 'prediction_length'.
               Defaults to {encoder_length:96, decoder_length:48, prediction_length:24} if not provided.
            date_col (str): Column name for the combined timestamp. Defaults to 'datetime'.
            use_cyclical_encoding (bool): If True, applies cyclical encoding for hour and minute features.
            include_technical_indicators (bool): If True, includes technical indicators in the feature set.
            max_features (int): Maximum number of features to keep after feature selection (default reduced to 10).
            use_log_returns (bool): If True, adds log transformation of close prices and log returns.
        """
        if sequence_config is None:
            sequence_config = {
                "encoder_length": 96,
                "decoder_length": 48,
                "prediction_length": 24
            }
        self.csv_path = csv_path
        self.sequence_config = sequence_config
        self.date_col = date_col
        self.use_cyclical_encoding = use_cyclical_encoding
        self.include_technical_indicators = include_technical_indicators
        self.max_features = max_features
        self.use_log_returns = use_log_returns

        self.dataframe = None
        self.scaler = None
        self.global_feature_names = None  # will store names of engineered time features
        self.time_features = None  # will store raw time features for TimeSeriesTransformer
        self.selected_feature_names = None  # will store names of selected features

        # Placeholders for processed features and datasets
        self.features = None
        self.all_windows = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def load_csv(self):
        """Reads the CSV file and parses the timestamps, combining date and time if needed."""
        self.dataframe = pd.read_csv(self.csv_path)
        
        # Ensure data is chronologically ordered by the timestamp
        self.dataframe.sort_values(by=self.date_col, inplace=True)
        
        # If CSV has separate 'date' and 'time' columns and no combined timestamp column
        if 'date' in self.dataframe.columns and 'time' in self.dataframe.columns and self.date_col not in self.dataframe.columns:
            self.dataframe[self.date_col] = pd.to_datetime(self.dataframe['date'] + ' ' + self.dataframe['time'], utc=True).dt.tz_convert(None)
        else:
            self.dataframe[self.date_col] = pd.to_datetime(self.dataframe[self.date_col], utc=True).dt.tz_convert(None)

    def engineer_time_features(self):
        """Extracts and processes time features from the timestamp."""
        df = self.dataframe
        ts = df[self.date_col]
        
        # Extract basic time components
        df['year'] = ts.dt.year.astype(float)  # Keep year as a raw value (could be normalized if needed)
        df['month'] = ts.dt.month.astype(float)  
        df['week'] = ts.dt.isocalendar().week.astype(float)
        df['day'] = ts.dt.day.astype(float)
        df['hour'] = ts.dt.hour.astype(float)
        df['minute'] = ts.dt.minute.astype(float)
        
        # Extract day of week and add weekend flag for TimeSeriesTransformer
        df['dayofweek'] = ts.dt.dayofweek.astype(float)
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(float)  # 5,6 = weekend

        if self.use_cyclical_encoding:
            # Apply cyclical encoding for hour and minute
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
            df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
            # Drop raw hour and minute
            df.drop(columns=['hour', 'minute'], inplace=True)
            self.global_feature_names = ['year', 'month', 'week', 'day', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'dayofweek', 'is_weekend']
        else:
            # Normalize the time components with simple scaling
            df['month'] = df['month'] / 12.0
            df['week'] = df['week'] / 52.0
            df['hour'] = df['hour'] / 23.0
            df['minute'] = df['minute'] / 59.0
            df['dayofweek'] = df['dayofweek'] / 6.0
            self.global_feature_names = ['year', 'month', 'week', 'day', 'hour', 'minute', 'dayofweek', 'is_weekend']
        
        # For TimeSeriesTransformer: save raw time features in a separate array
        # Format: [year, month, week, day, hour, minute, is_weekend, 0]
        time_features = np.zeros((len(df), 8))
        time_features[:, 0] = df['year'].values
        time_features[:, 1] = ts.dt.month.values
        time_features[:, 2] = ts.dt.isocalendar().week.values
        time_features[:, 3] = ts.dt.day.values
        time_features[:, 4] = ts.dt.hour.values
        time_features[:, 5] = ts.dt.minute.values
        time_features[:, 6] = df['is_weekend'].values
        # Column 7 is reserved for holiday flags (if available)
        
        self.time_features = time_features
        self.dataframe = df

    def prepare_features(self):
        """Prepares features from the loaded dataframe."""
        # 1. Extract time features
        print_info = print if 'print_info' in globals() else (lambda x: None)
        print_info("Preparing features")
        self.load_csv()
        self.engineer_time_features()
        
        # 2. Calculate technical indicators if needed
        if self.include_technical_indicators:
            print_info("Computing technical indicators")
            self.compute_technical_indicators()
        else:
            print_info("Skipping technical indicators")
            
        # 3. Add market/calendar features
        print_info("Adding market and calendar features")
        self.add_market_features()
        
        # 4. Add advanced statistical features
        print_info("Adding advanced statistical features")
        self.add_statistical_features()
            
        # 5. Add log-based features if enabled
        if self.use_log_returns:
            print_info("Adding log-based features")
            # Collect all log-based features in a dictionary
            log_features = {}
            
            # Log transform of close price
            log_features['log_close'] = np.log(self.dataframe['close'])
            
            # Log returns (daily percentage change in log space)
            log_features['log_return_1d'] = log_features['log_close'].diff(1)
            
            # Add 5-day and 20-day log returns for trend capture
            log_features['log_return_5d'] = log_features['log_close'].diff(5)
            log_features['log_return_20d'] = log_features['log_close'].diff(20)
            
            # Rolling volatility of log returns (standard deviation of log returns)
            log_features['log_volatility_14d'] = log_features['log_return_1d'].rolling(window=14).std()
            
            # Log returns squared (proxy for volatility)
            log_features['log_return_1d_squared'] = log_features['log_return_1d'] ** 2
            
            # Log return momentum (difference between short and long-term returns)
            log_features['log_return_momentum'] = log_features['log_return_5d'] - log_features['log_return_20d']
            
            # Add all log features at once
            self.dataframe = pd.concat([self.dataframe, pd.DataFrame(log_features)], axis=1)
            
            # Drop missing values after log calculations
            self.dataframe.dropna(inplace=True)
            
        # 6. Set final feature columns
        print_info("Combining and preparing final features")
        
        # Define OHLCV columns and time features
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # If using log returns, add them to feature list
        if self.use_log_returns:
            log_cols = ['log_close', 'log_return_1d', 'log_return_5d', 'log_return_20d', 'log_volatility_14d', 
                          'log_return_1d_squared', 'log_return_momentum']
            base_cols = ohlcv_cols + log_cols
        else:
            base_cols = ohlcv_cols
            
        # Get market feature columns and statistical feature columns
        market_cols = [col for col in self.dataframe.columns if col.startswith('market_') or col.startswith('calendar_')]
        stat_cols = [col for col in self.dataframe.columns if col.startswith('stat_')]
        base_cols = base_cols + market_cols + stat_cols
            
        # Get technical indicator columns if available
        if self.include_technical_indicators:
            all_cols = self.dataframe.columns.tolist()
            tech_cols = [col for col in all_cols if col not in ohlcv_cols + self.global_feature_names + [self.date_col] + 
                         (log_cols if self.use_log_returns else []) + market_cols + stat_cols]
            
            # Select most important features if using feature selection
            if len(tech_cols) > 0 and self.max_features < len(tech_cols) + len(base_cols):
                tech_cols = self.select_important_features(tech_cols)
                feature_cols = base_cols + tech_cols
            else:
                feature_cols = base_cols + tech_cols
        else:
            feature_cols = base_cols
            
        print_info(f"Final feature count: {len(feature_cols)}")
        self.features = self.dataframe[feature_cols].values
        self.selected_feature_names = feature_cols
        
        # 7. Normalize all features
        self.normalize_features()

    def select_important_features(self, tech_columns):
        """
        Select the most important technical features based on mutual information with future price.
        
        Args:
            tech_columns: List of technical indicator column names
            
        Returns:
            List of selected technical indicator column names
        """
        df = self.dataframe
        
        # Create target: future return (shift close price by prediction length)
        prediction_length = self.sequence_config.get("prediction_length", 24)
        future_close = df['close'].shift(-prediction_length)
        future_return = (future_close / df['close'] - 1) * 100
        
        # Drop NaNs created by the shift
        valid_idx = ~future_return.isna()
        X = df[tech_columns].loc[valid_idx]
        y = future_return.loc[valid_idx]
        
        # Use mutual information to select features with strongest relationship to target
        try:
            # Number of features to keep (subtract base features like OHLCV from max)
            n_to_select = min(self.max_features - 5, len(tech_columns))
            
            # Use SelectKBest with mutual_info_regression to identify important features
            selector = SelectKBest(mutual_info_regression, k=n_to_select)
            selector.fit(X, y)
            
            # Get feature scores and indices of selected features
            scores = selector.scores_
            selected_indices = selector.get_support(indices=True)
            
            # Create pairs of (feature, score) and sort by score
            feature_scores = [(tech_columns[i], scores[i]) for i in selected_indices]
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Extract just the feature names
            selected_features = [feature for feature, _ in feature_scores]
            return selected_features
            
        except Exception as e:
            print(f"Feature selection error: {e}. Using all features.")
            return tech_columns[:self.max_features - 5]  # Fall back to first n features

    def normalize_features(self):
        """Normalizes features using StandardScaler fit on the training portion (first 70%)."""
        num_data = self.features.shape[0]
        train_end = int(num_data * 0.7)
        self.scaler = StandardScaler()
        self.scaler.fit(self.features[:train_end])
        self.features = self.scaler.transform(self.features)

    def generate_windows(self):
        """Generates sliding windows for encoder input, decoder input, and target output with improved memory efficiency."""
        encoder_length = self.sequence_config["encoder_length"]
        decoder_length = self.sequence_config["decoder_length"]
        prediction_length = self.sequence_config["prediction_length"]
        total_window = encoder_length + decoder_length + prediction_length
        
        # Get dimensions for pre-allocation
        num_data = self.features.shape[0]
        num_features = self.features.shape[1]
        total_windows = num_data - total_window + 1
        
        print_info = print if 'print_info' in globals() else (lambda x: None)
        print_info(f"Total windows to generate: {total_windows}")
        
        # Pre-allocate arrays with correct data type (float32 instead of float64)
        # This immediately reduces memory usage by half for these arrays
        self.all_windows = {
            'encoder': np.zeros((total_windows, encoder_length, num_features), dtype=np.float32),
            'decoder': np.zeros((total_windows, decoder_length, num_features), dtype=np.float32),
            'target': np.zeros((total_windows, prediction_length, num_features), dtype=np.float32)
        }
        
        # Pre-allocate time feature arrays if available
        if self.time_features is not None:
            time_features_dim = self.time_features.shape[1]
            self.all_windows['encoder_time'] = np.zeros((total_windows, encoder_length, time_features_dim), dtype=np.float32)
            self.all_windows['decoder_time'] = np.zeros((total_windows, decoder_length + prediction_length, time_features_dim), dtype=np.float32)
        
        # Use smaller chunk size to reduce peak memory usage
        chunk_size = 100  # Reduced from 250
        
        # Process in smaller chunks with more frequent gc
        for chunk_start in range(0, total_windows, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_windows)
            print_info(f"Processing window chunk {chunk_start} to {chunk_end}")
            
            # Process each window in the chunk
            for i, window_idx in enumerate(range(chunk_start, chunk_end)):
                # Extract data for this window
                start_idx = window_idx
                end_idx = window_idx + total_window
                
                # Encoder window
                enc_start = start_idx
                enc_end = start_idx + encoder_length
                self.all_windows['encoder'][window_idx] = self.features[enc_start:enc_end].astype(np.float32)
                
                # Decoder window
                dec_start = enc_end
                dec_end = dec_start + decoder_length
                self.all_windows['decoder'][window_idx] = self.features[dec_start:dec_end].astype(np.float32)
                
                # Target window
                target_start = dec_end
                target_end = target_start + prediction_length
                self.all_windows['target'][window_idx] = self.features[target_start:target_end].astype(np.float32)
                
                # Time features if available
                if self.time_features is not None:
                    self.all_windows['encoder_time'][window_idx] = self.time_features[enc_start:enc_end].astype(np.float32)
                    self.all_windows['decoder_time'][window_idx] = self.time_features[dec_start:target_end].astype(np.float32)
            
            # Force garbage collection after each chunk
            gc.collect()
            
            # Print memory usage periodically
            if (chunk_end - chunk_start) % (chunk_size * 5) == 0:
                if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                    print_info(f"MPS Memory: Current allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
                elif torch.cuda.is_available():
                    print_info(f"CUDA Memory: Current allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        print_info(f"Window generation complete. Total windows: {len(self.all_windows['encoder'])}")

    def split_data(self):
        """Splits the generated windows chronologically into training (70%), validation (10%), and test (20%)."""
        total_windows = self.all_windows['encoder'].shape[0]
        print_info = print if 'print_info' in globals() else (lambda x: None)
        
        # Calculate split indices
        train_end = int(total_windows * 0.7)
        val_end = int(total_windows * 0.8)
        
        print_info(f"Splitting data: train={train_end}, val={val_end-train_end}, test={total_windows-val_end}")
        
        # Process train split
        print_info("Creating training dataset")
        train_encoder = self.all_windows['encoder'][:train_end]
        train_decoder = self.all_windows['decoder'][:train_end]
        train_target = self.all_windows['target'][:train_end]
        
        if 'encoder_time' in self.all_windows and 'decoder_time' in self.all_windows:
            train_encoder_time = self.all_windows['encoder_time'][:train_end]
            train_decoder_time = self.all_windows['decoder_time'][:train_end]
            self.train_dataset = StockDataset(train_encoder, train_decoder, train_target, 
                                             train_encoder_time, train_decoder_time)
        else:
            self.train_dataset = StockDataset(train_encoder, train_decoder, train_target)
        
        # Clear variables to free memory
        train_encoder, train_decoder, train_target = None, None, None
        if 'encoder_time' in self.all_windows and 'decoder_time' in self.all_windows:
            train_encoder_time, train_decoder_time = None, None
        gc.collect()
        
        # Process validation split
        print_info("Creating validation dataset")
        val_encoder = self.all_windows['encoder'][train_end:val_end]
        val_decoder = self.all_windows['decoder'][train_end:val_end]
        val_target = self.all_windows['target'][train_end:val_end]
        
        if 'encoder_time' in self.all_windows and 'decoder_time' in self.all_windows:
            val_encoder_time = self.all_windows['encoder_time'][train_end:val_end]
            val_decoder_time = self.all_windows['decoder_time'][train_end:val_end]
            self.val_dataset = StockDataset(val_encoder, val_decoder, val_target, 
                                           val_encoder_time, val_decoder_time)
        else:
            self.val_dataset = StockDataset(val_encoder, val_decoder, val_target)
        
        # Clear variables to free memory
        val_encoder, val_decoder, val_target = None, None, None
        if 'encoder_time' in self.all_windows and 'decoder_time' in self.all_windows:
            val_encoder_time, val_decoder_time = None, None
        gc.collect()
        
        # Process test split
        print_info("Creating test dataset")
        test_encoder = self.all_windows['encoder'][val_end:]
        test_decoder = self.all_windows['decoder'][val_end:]
        test_target = self.all_windows['target'][val_end:]
        
        if 'encoder_time' in self.all_windows and 'decoder_time' in self.all_windows:
            test_encoder_time = self.all_windows['encoder_time'][val_end:]
            test_decoder_time = self.all_windows['decoder_time'][val_end:]
            self.test_dataset = StockDataset(test_encoder, test_decoder, test_target, 
                                            test_encoder_time, test_decoder_time)
        else:
            self.test_dataset = StockDataset(test_encoder, test_decoder, test_target)
        
        # Clear variables to free memory
        test_encoder, test_decoder, test_target = None, None, None
        if 'encoder_time' in self.all_windows and 'decoder_time' in self.all_windows:
            test_encoder_time, test_decoder_time = None, None
        gc.collect()
        
        print_info("Data splitting complete")

    def get_dataloaders(self, batch_size=32, shuffle_train=True, num_workers=0, pin_memory=False):
        """Creates PyTorch DataLoader objects for train, validation, and test sets with device-specific optimizations."""
        # For MPS/CUDA, using pin_memory=True can help with data transfer
        dataloader_args = {
            'batch_size': batch_size,
            'pin_memory': pin_memory,
            'num_workers': num_workers
        }
        
        # Create data loaders with optimized settings
        train_loader = DataLoader(self.train_dataset, shuffle=shuffle_train, **dataloader_args)
        val_loader = DataLoader(self.val_dataset, shuffle=False, **dataloader_args)
        test_loader = DataLoader(self.test_dataset, shuffle=False, **dataloader_args)
        
        return train_loader, val_loader, test_loader
    
    def prepare_datasets(self):
        """Runs the entire pipeline: CSV ingestion, feature engineering, normalization, window generation, and splitting."""
        print_info = print if 'print_info' in globals() else (lambda x: None)
        
        print_info(f"Starting dataset preparation. Initial memory usage: {get_memory_usage():.2f} GB")
        
        print_info("Loading CSV data")
        self.load_csv()
        
        # Optimize DataFrame memory usage after loading
        print_info("Optimizing initial DataFrame memory usage")
        self.dataframe = optimize_df_dtypes(self.dataframe)
        
        print_info("Engineering time features")
        self.engineer_time_features()
        print_info(f"After time features. Memory usage: {get_memory_usage():.2f} GB")
        
        # Only compute technical indicators (and additional features) if enabled
        if self.include_technical_indicators:
            print_info("Computing technical indicators")
            self.compute_technical_indicators()
            print_info(f"After technical indicators. Memory usage: {get_memory_usage():.2f} GB")
        else:
            print_info("Skipping technical indicators")
            
        # Add market features
        print_info("Adding market and calendar features")
        self.add_market_features()
        print_info(f"After market features. Memory usage: {get_memory_usage():.2f} GB")
        
        # Add statistical features
        print_info("Adding statistical features")
        self.add_statistical_features()
        print_info(f"After statistical features. Memory usage: {get_memory_usage():.2f} GB")
        
        print_info("Preparing features")
        self.prepare_features()
        print_info(f"After feature preparation. Memory usage: {get_memory_usage():.2f} GB")
        
        # We don't need the raw dataframe anymore - free memory
        print_info("Freeing dataframe memory")
        del self.dataframe
        self.dataframe = None
        gc.collect()
        print_info(f"After freeing dataframe. Memory usage: {get_memory_usage():.2f} GB")
        
        print_info("Normalizing features")
        self.normalize_features()
        
        # Convert features to float32 to reduce memory usage
        print_info("Converting features to float32")
        self.features = self.features.astype(np.float32)
        if self.time_features is not None:
            self.time_features = self.time_features.astype(np.float32)
        gc.collect()
        print_info(f"After normalization. Memory usage: {get_memory_usage():.2f} GB")
        
        print_info("Generating windows")
        self.generate_windows()
        print_info(f"After window generation. Memory usage: {get_memory_usage():.2f} GB")
        
        # Free the features array as we now have windows
        print_info("Freeing features memory")
        del self.features
        self.features = None
        gc.collect()
        print_info(f"After freeing features. Memory usage: {get_memory_usage():.2f} GB")
        
        print_info("Splitting data into train/val/test")
        self.split_data()
        print_info(f"After splitting data. Memory usage: {get_memory_usage():.2f} GB")
        
        # Free window data once datasets are created
        if self.train_dataset is not None and self.val_dataset is not None and self.test_dataset is not None:
            print_info("Freeing window memory")
            del self.all_windows
            self.all_windows = None
            gc.collect()
            print_info(f"Final memory usage: {get_memory_usage():.2f} GB")
            print_info("Dataset preparation complete")

    def compute_technical_indicators(self):
        """
        Computes a comprehensive set of technical indicators for better stock price prediction.
        """
        df = self.dataframe
        print_info = print if 'print_info' in globals() else (lambda x: None)
        print_info("Computing enhanced technical indicators set")

        # Create a dictionary to collect all technical indicators
        tech_indicators = {}
        
        # Extract base price and volume data to avoid repeated DataFrame access
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_price = df['open'].values
        volume = df['volume'].values
        
        # Convert to pandas Series for calculations
        close_series = pd.Series(close)
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        open_series = pd.Series(open_price)
        volume_series = pd.Series(volume)

        # 1. TREND INDICATORS
        # Simple Moving Averages - multiple timeframes
        for window in [5, 10, 20, 50, 200]:
            tech_indicators[f'sma_{window}'] = close_series.rolling(window=window, min_periods=1).mean()
        
        # Exponential Moving Averages - multiple timeframes
        for window in [5, 12, 26]:
            tech_indicators[f'ema_{window}'] = close_series.ewm(span=window, adjust=False).mean()
        
        # MACD (Moving Average Convergence Divergence)
        ema12 = close_series.ewm(span=12, adjust=False).mean()
        ema26 = close_series.ewm(span=26, adjust=False).mean()
        tech_indicators['macd_line'] = ema12 - ema26
        tech_indicators['macd_signal'] = tech_indicators['macd_line'].ewm(span=9, adjust=False).mean()
        tech_indicators['macd_histogram'] = tech_indicators['macd_line'] - tech_indicators['macd_signal']
        
        # Price Rate of Change (ROC)
        for window in [5, 10, 20]:
            tech_indicators[f'price_roc_{window}'] = close_series.pct_change(periods=window) * 100

        # 2. MOMENTUM INDICATORS
        # Relative Strength Index (RSI)
        for window in [6, 14, 28]:
            delta = close_series.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=window, min_periods=window).mean()
            avg_loss = loss.rolling(window=window, min_periods=window).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            tech_indicators[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        window = 14
        lowest_low = low_series.rolling(window=window).min()
        highest_high = high_series.rolling(window=window).max()
        tech_indicators['stoch_k'] = 100 * ((close_series - lowest_low) / (highest_high - lowest_low + 1e-10))
        tech_indicators['stoch_d'] = tech_indicators['stoch_k'].rolling(window=3).mean()
        
        # Williams %R
        tech_indicators['williams_r'] = -100 * ((highest_high - close_series) / (highest_high - lowest_low + 1e-10))

        # 3. VOLATILITY INDICATORS
        # Bollinger Bands
        for window in [20, 50]:
            sma = close_series.rolling(window=window, min_periods=window).mean()
            std = close_series.rolling(window=window, min_periods=window).std()
            tech_indicators[f'bb_upper_{window}'] = sma + (2 * std)
            tech_indicators[f'bb_lower_{window}'] = sma - (2 * std)
            tech_indicators[f'bb_width_{window}'] = (tech_indicators[f'bb_upper_{window}'] - tech_indicators[f'bb_lower_{window}']) / sma
            tech_indicators[f'bb_percent_{window}'] = (close_series - tech_indicators[f'bb_lower_{window}']) / (tech_indicators[f'bb_upper_{window}'] - tech_indicators[f'bb_lower_{window}'] + 1e-10)
        
        # Average True Range (ATR)
        tr1 = abs(high_series - low_series)
        tr2 = abs(high_series - close_series.shift())
        tr3 = abs(low_series - close_series.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        tech_indicators['atr_14'] = true_range.rolling(window=14).mean()
        tech_indicators['atr_percent'] = tech_indicators['atr_14'] / close_series * 100  # ATR as percentage of price
        
        # Historical Volatility
        for window in [10, 20, 30]:
            tech_indicators[f'hist_vol_{window}'] = close_series.pct_change().rolling(window=window).std() * (252 ** 0.5)  # Annualized
        
        # 4. VOLUME INDICATORS
        # On-Balance Volume (OBV)
        tech_indicators['obv'] = (np.sign(close_series.diff()) * volume_series).fillna(0).cumsum()
        
        # Volume Rate of Change
        for window in [5, 10]:
            tech_indicators[f'volume_roc_{window}'] = volume_series.pct_change(periods=window) * 100
        
        # Price-Volume Trend
        tech_indicators['pvt'] = ((close_series - close_series.shift()) / close_series.shift() * volume_series).cumsum()
        
        # 5. PRICE PATTERNS & RELATIONSHIPS
        # Gaps
        tech_indicators['gap_up'] = (open_series > high_series.shift()) * 1.0
        tech_indicators['gap_down'] = (open_series < low_series.shift()) * 1.0
        
        # Candle patterns: Doji
        body_size = abs(close_series - open_series)
        shadow_size = high_series - low_series
        tech_indicators['doji'] = (body_size < (0.1 * shadow_size)) * 1.0
        
        # Candle patterns: Hammer/Shooting Star (rough approximation)
        lower_wick = pd.concat([open_series, close_series], axis=1).min(axis=1) - low_series
        upper_wick = high_series - pd.concat([open_series, close_series], axis=1).max(axis=1)
        tech_indicators['hammer'] = ((lower_wick > (2 * body_size)) & (upper_wick < (0.2 * body_size))) * 1.0
        tech_indicators['shooting_star'] = ((upper_wick > (2 * body_size)) & (lower_wick < (0.2 * body_size))) * 1.0
        
        # 6. ADVANCED INDICATORS
        # Ichimoku Cloud components (simplified)
        high_9 = high_series.rolling(window=9).max()
        low_9 = low_series.rolling(window=9).min()
        tech_indicators['tenkan_sen'] = (high_9 + low_9) / 2  # Conversion Line
        
        high_26 = high_series.rolling(window=26).max()
        low_26 = low_series.rolling(window=26).min()
        tech_indicators['kijun_sen'] = (high_26 + low_26) / 2  # Base Line
        
        tech_indicators['senkou_span_a'] = ((tech_indicators['tenkan_sen'] + tech_indicators['kijun_sen']) / 2).shift(26)  # Leading Span A
        
        # 7. PRICE RELATIVE TO MOVING AVERAGES
        for ma in [50, 200]:
            tech_indicators[f'close_to_sma_{ma}_pct'] = (close_series / tech_indicators[f'sma_{ma}'] - 1) * 100
        
        # 8. MEAN REVERSION INDICATORS
        # Z-Score of price (how many std devs away from moving average)
        for window in [20, 50]:
            mean = close_series.rolling(window=window).mean()
            std = close_series.rolling(window=window).std()
            tech_indicators[f'z_score_{window}'] = (close_series - mean) / (std + 1e-10)
        
        # 9. CUSTOM FEATURES
        # Price acceleration (second derivative of price)
        tech_indicators['price_acceleration'] = close_series.diff().diff()
        
        # Volume-weighted price changes
        tech_indicators['volume_weighted_return'] = close_series.pct_change() * (volume_series / volume_series.rolling(window=5).mean())
        
        # Trend strength indicators
        tech_indicators['adx_14'] = self._calculate_adx(df, window=14)
        
        # Price and volume divergence
        price_5d_change = close_series.pct_change(periods=5)
        volume_5d_change = volume_series.pct_change(periods=5)
        tech_indicators['price_volume_divergence'] = ((price_5d_change > 0) & (volume_5d_change < 0)) | ((price_5d_change < 0) & (volume_5d_change > 0))
        tech_indicators['price_volume_divergence'] = tech_indicators['price_volume_divergence'] * 1.0  # Convert boolean to float
        
        # Create a DataFrame from the collected indicators
        tech_df = pd.DataFrame(tech_indicators)
        
        # Optimize memory usage
        tech_df = optimize_df_dtypes(tech_df)
        
        # Combine with original DataFrame
        self.dataframe = pd.concat([df, tech_df], axis=1)
        
        # Drop rows with any missing values after indicator calculations
        self.dataframe.dropna(inplace=True)
        
        # Force garbage collection
        gc.collect()
        
        print_info(f"Computed enhanced technical indicators")
        
    def _calculate_adx(self, df, window=14):
        """Helper function to calculate ADX (Average Directional Index)"""
        # Calculate the +DM and -DM
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = ((high_diff > 0) & (high_diff > low_diff)) * high_diff
        minus_dm = ((low_diff > 0) & (low_diff > high_diff)) * low_diff
        
        # Calculate ATR
        tr1 = abs(df['high'] - df['low'])
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / (atr + 1e-10))
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=window).mean()
        
        return adx

    def inverse_transform(self, data):
        """Inverse transforms the normalized features using the fitted scaler.
        
        Parameters:
            data (np.array): Normalized data array to transform back to the original scale.

        Returns:
            np.array: Data transformed back to the original scale.
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Run prepare_datasets first.")
        return self.scaler.inverse_transform(data)

    def add_market_features(self):
        """
        Add market-related and calendar-based features.
        This method adds features that relate to the broader market context and time-based patterns.
        """
        df = self.dataframe
        dates = pd.to_datetime(df[self.date_col])
        
        # Dictionary to collect all market features
        market_features = {}
        
        # Extract base data to avoid repeated DataFrame access
        close = df['close'].values
        close_series = pd.Series(close, index=df.index)
        volume = df['volume'].values
        volume_series = pd.Series(volume, index=df.index)
        
        # 1. CALENDAR FEATURES
        # Month of year as cyclical features
        market_features['calendar_month_sin'] = np.sin(2 * np.pi * dates.dt.month / 12)
        market_features['calendar_month_cos'] = np.cos(2 * np.pi * dates.dt.month / 12)
        
        # Day of month normalized
        market_features['calendar_day_of_month'] = dates.dt.day / dates.dt.daysinmonth
        
        # Week of year as cyclical features
        market_features['calendar_week_sin'] = np.sin(2 * np.pi * dates.dt.isocalendar().week / 52)
        market_features['calendar_week_cos'] = np.cos(2 * np.pi * dates.dt.isocalendar().week / 52)
        
        # Quarter of year
        market_features['calendar_quarter'] = dates.dt.quarter / 4
        
        # Beginning/end of month markers (often have different price behavior)
        market_features['calendar_start_of_month'] = (dates.dt.day <= 5).astype(float)
        market_features['calendar_end_of_month'] = (dates.dt.day >= 25).astype(float)
        
        # Day of week as cyclical features
        market_features['calendar_day_of_week_sin'] = np.sin(2 * np.pi * dates.dt.dayofweek / 7)
        market_features['calendar_day_of_week_cos'] = np.cos(2 * np.pi * dates.dt.dayofweek / 7)
        
        # 2. MARKET REGIME FEATURES
        # Volatility regimes (based on recent volatility compared to longer term)
        vol_short = close_series.pct_change().rolling(window=5).std()
        vol_long = close_series.pct_change().rolling(window=20).std()
        market_features['market_volatility_regime'] = (vol_short / vol_long).replace([np.inf, -np.inf], np.nan)
        
        # Trend regimes (based on price relative to moving averages)
        ma_short = close_series.rolling(window=10).mean()
        ma_long = close_series.rolling(window=30).mean()
        market_features['market_trend_regime'] = ((ma_short / ma_long) - 1) * 100
        
        # Volume regimes (is current volume higher or lower than recent average)
        market_features['market_volume_regime'] = volume_series / volume_series.rolling(window=20).mean() - 1
        
        # 3. DERIVATIVE PRICE FEATURES
        # Rate of change of volatility
        market_features['market_volatility_change'] = vol_short.pct_change(5)
        
        # Momentum strength based on consecutive moves in same direction
        price_change = close_series.pct_change()
        market_features['market_consec_up'] = ((price_change > 0) & (price_change.shift(1) > 0) & 
                                  (price_change.shift(2) > 0)).astype(float)
        market_features['market_consec_down'] = ((price_change < 0) & (price_change.shift(1) < 0) & 
                                    (price_change.shift(2) < 0)).astype(float)
        
        # 4. PRICE RANGE FEATURES
        # Trading range measures
        rolling_high = df['high'].rolling(window=10).max()
        rolling_low = df['low'].rolling(window=10).min()
        market_features['market_trading_range'] = (rolling_high - rolling_low) / rolling_low * 100
        
        # Position within recent range (0 = at bottom, 1 = at top)
        market_features['market_range_percentile'] = (close_series - rolling_low) / (rolling_high - rolling_low + 1e-10)
        
        # 5. FRACTAL DIMENSIONS (measure of market complexity/choppiness)
        market_features['market_hurst_exponent'] = self._calculate_rolling_hurst(close_series, window=30)
        
        # Create a DataFrame from the collected features
        # Convert all features to Series with the same index if they aren't already
        for key in market_features:
            if not isinstance(market_features[key], pd.Series):
                market_features[key] = pd.Series(market_features[key], index=df.index)
        
        # Create DataFrame with aligned indices
        market_df = pd.DataFrame(market_features)
        
        # Optimize memory usage
        market_df = optimize_df_dtypes(market_df)
        
        # Combine with original DataFrame
        self.dataframe = pd.concat([df, market_df], axis=1)
        
        # Clean up NaN values
        self.dataframe.dropna(inplace=True)
        
        # Force garbage collection
        gc.collect()

    def _calculate_rolling_hurst(self, series, window=30):
        """
        Calculate the Hurst exponent over a rolling window for time series.
        The Hurst exponent measures the long-term memory of a time series.
        H < 0.5 indicates mean-reverting series
        H = 0.5 indicates random walk
        H > 0.5 indicates trend-following series
        """
        result = np.zeros_like(series)
        result[:] = np.nan
        
        for i in range(window, len(series)):
            ts = series.iloc[i-window:i].values
            try:
                # Calculate simple version of Hurst exponent
                lags = range(2, 20)  # Use fewer lags for performance
                tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                result[i] = poly[0] / 2.0  # Hurst = slope/2
            except:
                continue  # Keep as NaN in case of numerical issues
        
        return result

    def add_statistical_features(self):
        """
        Add advanced statistical features that capture non-linear relationships 
        and complex statistical properties of the time series.
        """
        df = self.dataframe
        new_features = {}  # Dictionary to collect all new features
        
        # 1. SERIAL CORRELATION FEATURES
        # Autocorrelation features for close price at multiple lags
        close_returns = df['close'].pct_change().dropna()
        for lag in [1, 5, 10]:
            if len(close_returns) > lag + 5:  # Ensure enough data points
                autocorr_values = close_returns.rolling(window=30).apply(
                    lambda x: x.autocorr(lag=lag) if len(x) > lag + 5 else np.nan, raw=False)
                # Reindex to match the original dataframe
                new_features[f'stat_autocorr_{lag}'] = autocorr_values.reindex(df.index)
        
        # 2. DISTRIBUTIONAL FEATURES
        # Calculate rolling skewness and kurtosis of returns (non-normality measures)
        returns = df['close'].pct_change()
        for window in [10, 30]:
            new_features[f'stat_return_skew_{window}'] = returns.rolling(window=window).skew()
            new_features[f'stat_return_kurt_{window}'] = returns.rolling(window=window).kurt()
        
        # Quantile-based features (useful for capturing tail behavior)
        window = 50
        for quantile in [0.1, 0.9]:
            new_features[f'stat_return_q{int(quantile*100)}_{window}'] = returns.rolling(
                window=window).quantile(quantile)
        
        # 3. ENTROPY-BASED FEATURES (measure of randomness/predictability)
        # Approximated rolling entropy using normalized bins
        entropy_values = self._calculate_rolling_entropy(returns, window=10)
        new_features['stat_entropy_10'] = pd.Series(entropy_values, index=df.index)
        
        # 4. TURNING POINTS / ZIGZAG FEATURES
        # Detect local maxima and minima (turning points in price)
        local_max = self._detect_local_extrema(df['close'], window=5, find_max=True)
        local_min = self._detect_local_extrema(df['close'], window=5, find_max=False)
        new_features['stat_is_local_max'] = pd.Series(local_max, index=df.index)
        new_features['stat_is_local_min'] = pd.Series(local_min, index=df.index)
        
        # 5. CROSS-CORRELATION BETWEEN PRICE AND VOLUME
        # Rolling correlation between price changes and volume changes
        volume_changes = df['volume'].pct_change()
        for window in [10, 30]:
            corrs = returns.rolling(window=window).corr(volume_changes)
            new_features[f'stat_price_vol_corr_{window}'] = corrs
        
        # 6. TAIL RISK MEASURES
        # Conditional Value at Risk (Expected Shortfall)
        window = 50
        cvar_values = returns.rolling(window=window).apply(
            lambda x: -np.mean(np.sort(x)[:int(0.05*len(x))]) if len(x) >= 20 else np.nan, raw=True)
        new_features['stat_cvar_95'] = cvar_values
            
        # 7. MEAN REVERSION / MOMENTUM STRENGTH
        # Crossing of price and moving average (sign changes in price - MA)
        ma20 = df['close'].rolling(window=20).mean()
        price_above_ma = (df['close'] > ma20).astype(int)
        new_features['stat_price_ma_cross'] = price_above_ma.diff().abs()
        
        # 8. NON-LINEAR FEATURES
        # Square, cube, and square root transformations of returns
        new_features['stat_return_squared'] = returns ** 2
        new_features['stat_return_sign'] = np.sign(returns)
        new_features['stat_return_sign_change'] = new_features['stat_return_sign'].diff().abs() / 2
        
        # 9. RANGE-BASED VOLATILITY FEATURES
        # Parkinson volatility estimator (based on high-low range)
        parkinson_vol = (1 / (4 * np.log(2)) * 
                        (np.log(df['high'] / df['low']) ** 2)).rolling(window=10).mean() ** 0.5
        new_features['stat_parkinsons_vol_10'] = parkinson_vol
        
        # 10. RELATIVE STRENGTH / NORMALIZED PRICE FEATURES
        # Price normalized by recent trading range
        for window in [20, 50]:
            high_max = df['high'].rolling(window=window).max()
            low_min = df['low'].rolling(window=window).min()
            range_width = high_max - low_min
            new_features[f'stat_normalized_price_{window}'] = (df['close'] - low_min) / (range_width + 1e-10)
        
        # Ensure all features have the same index as the dataframe
        for key in list(new_features.keys()):
            if not isinstance(new_features[key], pd.Series):
                new_features[key] = pd.Series(new_features[key], index=df.index)
            elif new_features[key].index.equals(df.index) == False:
                # Reindex to match the original dataframe
                new_features[key] = new_features[key].reindex(df.index)
            
        # Update DataFrame all at once
        df = pd.concat([df, pd.DataFrame(new_features)], axis=1)
        
        # Clean up and handle any NaN values
        df.dropna(inplace=True)
        self.dataframe = df
    
    def _calculate_rolling_entropy(self, series, window=20, bins=10):
        """Calculate approximate entropy for time series in rolling windows"""
        result = np.zeros_like(series)
        result[:] = np.nan
        
        # Use rolling window approach
        for i in range(window, len(series)):
            if i < window:
                continue
                
            # Take window of data
            data = series.iloc[i-window:i].dropna().values
            if len(data) < window * 0.8:  # Require at least 80% of data points
                continue
                
            try:
                # Discretize the data into bins for entropy calculation
                hist, _ = np.histogram(data, bins=bins)
                prob = hist / len(data)
                # Filter out zeros to avoid log(0)
                prob = prob[prob > 0]
                # Calculate entropy
                entropy = -np.sum(prob * np.log2(prob))
                result[i] = entropy
            except:
                continue
                
        return result
    
    def _detect_local_extrema(self, series, window=5, find_max=True):
        """Detect local maxima or minima in a time series"""
        result = np.zeros_like(series)
        
        # Determine the extrema detection function
        extrema_func = np.argmax if find_max else np.argmin
        
        # For each point, check if it's a local extrema in the window
        for i in range(window, len(series) - window):
            window_data = series.iloc[i-window:i+window+1]
            extrema_idx = extrema_func(window_data.values)
            
            # If the center point is the extrema, mark it
            if extrema_idx == window:
                result[i] = 1.0
                
        return result

def get_memory_usage():
    """
    Get current memory usage of the process.
    
    Returns:
        float: Memory usage in GB
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 ** 3)  # Convert bytes to GB
    return memory_gb

def optimize_df_dtypes(df):
    """
    Optimize memory usage of a DataFrame by downcasting numeric columns to the smallest possible type.
    
    Args:
        df (pd.DataFrame): DataFrame to optimize
        
    Returns:
        pd.DataFrame: Memory-optimized DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage before optimization: {start_mem:.2f} MB")
    
    # Optimize numeric columns
    for col in df.columns:
        col_type = df[col].dtype
        
        # Handle pandas extension dtypes
        if hasattr(col_type, 'name'):
            type_name = col_type.name
            
            # Convert pandas extension float types to numpy float32
            if 'float' in type_name.lower():
                df[col] = df[col].astype(np.float32)
            continue
            
        # Optimize integers
        if col_type != 'object' and np.issubdtype(col_type, np.integer):
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Find the smallest integer type that can hold the data
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                df[col] = df[col].astype(np.uint8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                df[col] = df[col].astype(np.uint16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                df[col] = df[col].astype(np.uint32)
                
        # Optimize floats
        elif col_type != 'object' and np.issubdtype(col_type, np.floating):
            # Use float32 instead of float64 for most financial data
            # This provides sufficient precision while using half the memory
            df[col] = df[col].astype(np.float32)
    
    # Force garbage collection
    gc.collect()
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage after optimization: {end_mem:.2f} MB")
    print(f"Memory reduced by {100 * (start_mem - end_mem) / start_mem:.2f}%")
    
    return df


if __name__ == '__main__':
    # Example usage: Adjust the CSV file path as needed.
    csv_path = 'data/processed/SPY.csv'

    # Initialize the loader with cyclical encoding enabled for hour and minute features
    data_loader = StockDataLoader(csv_path, use_cyclical_encoding=True, include_technical_indicators=False)
    data_loader.prepare_datasets()
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(batch_size=32)
    
    # Iterate over one batch from the training loader to verify shapes
    for batch in train_loader:
        print('Encoder batch shape:', batch['encoder'].shape)
        print('Decoder batch shape:', batch['decoder'].shape)
        print('Target batch shape:', batch['target'].shape)
        if 'encoder_time' in batch:
            print('Encoder time shape:', batch['encoder_time'].shape)
            print('Decoder time shape:', batch['decoder_time'].shape)
        break
