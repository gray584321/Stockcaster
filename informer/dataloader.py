import pandas as pd
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import math


class StockDataset(Dataset):
    def __init__(self, encoder_inputs, decoder_inputs, targets):
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            'encoder': torch.tensor(self.encoder_inputs[idx], dtype=torch.float32),
            'decoder': torch.tensor(self.decoder_inputs[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets[idx], dtype=torch.float32)
        }


class StockDataLoader:
    def __init__(self, csv_path, sequence_config=None, date_col='datetime', use_cyclical_encoding=True, include_technical_indicators=True):
        """
        Initializes the data loader for historical stock data.

        Parameters:
            csv_path (str): Path to the CSV file containing stock data.
            sequence_config (dict): Dict specifying 'encoder_length', 'decoder_length', and 'prediction_length'.
               Defaults to {encoder_length:96, decoder_length:48, prediction_length:24} if not provided.
            date_col (str): Column name for the combined timestamp. Defaults to 'datetime'.
            use_cyclical_encoding (bool): If True, applies cyclical encoding for hour and minute features.
            include_technical_indicators (bool): If True, includes technical indicators in the feature set.
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

        self.dataframe = None
        self.scaler = None
        self.global_feature_names = None  # will store names of engineered time features

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
        df['hour'] = ts.dt.hour.astype(float)
        df['minute'] = ts.dt.minute.astype(float)

        if self.use_cyclical_encoding:
            # Apply cyclical encoding for hour and minute
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
            df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
            # Drop raw hour and minute
            df.drop(columns=['hour', 'minute'], inplace=True)
            self.global_feature_names = ['year', 'month', 'week', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos']
        else:
            # Normalize the time components with simple scaling
            df['month'] = df['month'] / 12.0
            df['week'] = df['week'] / 52.0
            df['hour'] = df['hour'] / 23.0
            df['minute'] = df['minute'] / 59.0
            self.global_feature_names = ['year', 'month', 'week', 'hour', 'minute']
        
        self.dataframe = df

    def prepare_features(self):
        """Combines stock features and engineered time features into a single numpy array."""
        df = self.dataframe
        stock_columns = ['open', 'high', 'low', 'close', 'volume']
        if self.include_technical_indicators:
            technical_columns = [
                'sma_close', 'ema_close', 'rsi_14', 'macd',
                'macd_signal', 'macd_histogram', 'bollinger_upper',
                'bollinger_lower', 'obv',
                '5min_mean_close', '15min_mean_close',
                '5min_rolling_volatility', '15min_rolling_volatility',
                '5min_realized_variance', '15min_realized_variance',
                'price_return', 'return_volume_interaction', 'log_volume'
            ]
            stock_columns.extend(technical_columns)
        stock_features = df[stock_columns]
        time_features = df[self.global_feature_names]
        full_features = pd.concat([stock_features, time_features], axis=1)
        self.features = full_features.values  # shape: (num_timesteps, feature_dimension)

    def normalize_features(self):
        """Normalizes features using StandardScaler fit on the training portion (first 70%)."""
        num_data = self.features.shape[0]
        train_end = int(num_data * 0.7)
        self.scaler = StandardScaler()
        self.scaler.fit(self.features[:train_end])
        self.features = self.scaler.transform(self.features)

    def generate_windows(self):
        """Generates sliding windows for encoder input, decoder input, and target output."""
        encoder_length = self.sequence_config["encoder_length"]
        decoder_length = self.sequence_config["decoder_length"]
        prediction_length = self.sequence_config["prediction_length"]
        total_window = encoder_length + decoder_length + prediction_length
        
        enc_windows = []
        dec_windows = []
        target_windows = []
        num_data = self.features.shape[0]
        
        for i in range(num_data - total_window + 1):
            window = self.features[i: i + total_window]
            encoder_input = window[:encoder_length]
            decoder_input = window[encoder_length: encoder_length + decoder_length]
            target = window[encoder_length + decoder_length:]
            enc_windows.append(encoder_input)
            dec_windows.append(decoder_input)
            target_windows.append(target)
        
        self.all_windows = {
            'encoder': np.array(enc_windows),
            'decoder': np.array(dec_windows),
            'target': np.array(target_windows)
        }
        
    def split_data(self):
        """Splits the generated windows chronologically into training (70%), validation (10%), and test (20%)."""
        total_windows = self.all_windows['encoder'].shape[0]
        train_idx, val_idx, test_idx = [], [], []

        for i in range(total_windows):
            # Using the window index relative to total windows for chronological splitting
            ratio = i / total_windows
            if ratio < 0.7:
                train_idx.append(i)
            elif ratio < 0.8:
                val_idx.append(i)
            else:
                test_idx.append(i)
        
        def subset_windows(indices):
            encoder = self.all_windows['encoder'][indices]
            decoder = self.all_windows['decoder'][indices]
            target = self.all_windows['target'][indices]
            return encoder, decoder, target
        
        train_enc, train_dec, train_tar = subset_windows(train_idx)
        val_enc, val_dec, val_tar = subset_windows(val_idx)
        test_enc, test_dec, test_tar = subset_windows(test_idx)
        
        self.train_dataset = StockDataset(train_enc, train_dec, train_tar)
        self.val_dataset = StockDataset(val_enc, val_dec, val_tar)
        self.test_dataset = StockDataset(test_enc, test_dec, test_tar)
        
    def get_dataloaders(self, batch_size=32, shuffle_train=True):
        """Creates PyTorch DataLoader objects for train, validation, and test sets."""
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle_train)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    
    def prepare_datasets(self):
        """Runs the entire pipeline: CSV ingestion, feature engineering, normalization, window generation, and splitting."""
        self.load_csv()
        self.engineer_time_features()
        # Only compute technical indicators (and additional features) if enabled
        if self.include_technical_indicators:
            self.compute_technical_indicators()
        self.prepare_features()
        self.normalize_features()
        self.generate_windows()
        self.split_data()

    def compute_technical_indicators(self):
        """Optionally computes technical indicators for the 'close' price.
        Added indicators: SMA (window=5), EMA (span=5), RSI (14), MACD, Bollinger Bands, OBV,
        as well as mixed‑frequency aggregated features, rolling volatility/variance, and interaction terms.
        Then removes rows with any missing values.
        """
        df = self.dataframe

        # Basic technical indicators
        df['sma_close'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['ema_close'] = df['close'].ewm(span=5, adjust=False).mean()

        # RSI calculation
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD calculation
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        window = 20
        sma_20 = df['close'].rolling(window=window, min_periods=window).mean()
        std_20 = df['close'].rolling(window=window, min_periods=window).std()
        df['bollinger_upper'] = sma_20 + (std_20 * 2)
        df['bollinger_lower'] = sma_20 - (std_20 * 2)

        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

        # -------------------------------
        # Mixed‑Frequency and Aggregated Features
        df['5min_mean_close'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['15min_mean_close'] = df['close'].rolling(window=15, min_periods=1).mean()
        df['30min_mean_close'] = df['close'].rolling(window=30, min_periods=1).mean()  # Added 30min aggregation

        # Rolling Volatility and Realized Variance
        df['5min_rolling_volatility'] = df['close'].pct_change().rolling(window=5, min_periods=1).std()
        df['15min_rolling_volatility'] = df['close'].pct_change().rolling(window=15, min_periods=1).std()
        df['5min_realized_variance'] = (df['close'].pct_change() ** 2).rolling(window=5, min_periods=1).sum()
        df['15min_realized_variance'] = (df['close'].pct_change() ** 2).rolling(window=15, min_periods=1).sum()

        # Interaction Terms and Nonlinear Transformations
        df['price_return'] = df['close'].pct_change()
        df['return_volume_interaction'] = df['price_return'] * df['volume']
        df['log_volume'] = np.log(df['volume'] + 1)
        
        # NEW: Advanced features and indicators
        
        # 1. Price gap calculations
        df['gap_open'] = df['open'] - df['close'].shift(1)
        df['gap_ratio'] = df['gap_open'] / df['close'].shift(1)
        
        # 2. Candlestick pattern features
        df['candle_body'] = df['close'] - df['open']
        df['candle_wick_upper'] = df['high'] - np.maximum(df['open'], df['close'])
        df['candle_wick_lower'] = np.minimum(df['open'], df['close']) - df['low']
        df['candle_body_ratio'] = df['candle_body'] / (df['high'] - df['low'] + 1e-10)
        
        # 3. Trend strength indicators
        df['adx_up_move'] = df['high'] - df['high'].shift(1)
        df['adx_down_move'] = df['low'].shift(1) - df['low']
        df['adx_plus_di'] = 100 * df['adx_up_move'].clip(lower=0).rolling(window=14).mean() / df['true_range']
        df['adx_minus_di'] = 100 * df['adx_down_move'].clip(lower=0).rolling(window=14).mean() / df['true_range']
        df['adx_di_diff'] = abs(df['adx_plus_di'] - df['adx_minus_di'])
        df['adx_di_sum'] = df['adx_plus_di'] + df['adx_minus_di']
        df['adx'] = 100 * df['adx_di_diff'].rolling(window=14).mean() / df['adx_di_sum']
        
        # 4. Momentum indicators
        # Rate of Change (ROC)
        df['roc_5'] = (df['close'] / df['close'].shift(5) - 1) * 100
        df['roc_10'] = (df['close'] / df['close'].shift(10) - 1) * 100
        
        # Williams %R
        highest_high = df['high'].rolling(window=14).max()
        lowest_low = df['low'].rolling(window=14).min()
        df['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-10)
        
        # 5. Volume-based indicators
        # Chaikin Money Flow (CMF)
        money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        money_flow_volume = money_flow_multiplier * df['volume']
        df['cmf'] = money_flow_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        
        # 6. Volatility indicators
        # Average True Range (ATR)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = np.maximum(np.maximum(tr1, tr2), tr3)
        df['atr'] = df['true_range'].rolling(window=14).mean()
        
        # 7. Mean reversion indicators
        df['zscore_5'] = (df['close'] - df['close'].rolling(window=5).mean()) / df['close'].rolling(window=5).std()
        df['zscore_10'] = (df['close'] - df['close'].rolling(window=10).mean()) / df['close'].rolling(window=10).std()
        
        # 8. Stochastic Oscillator
        df['stoch_k'] = 100 * (df['close'] - df['low'].rolling(window=14).min()) / (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min() + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # 9. Time series decomposition features (simplified)
        # Trend component using long-term moving average
        df['trend_component'] = df['close'].rolling(window=50, min_periods=1).mean()
        # Seasonal component (daily pattern) using hour-of-day normalization
        if 'hour' in df.columns:
            df['hour_avg'] = df.groupby('hour')['close'].transform('mean')
            df['seasonal_component'] = df['close'] / df['hour_avg']
            
        # 10. Cross-asset features (if we had multiple assets, we'd add correlations here)
        
        # 11. Price acceleration
        df['price_acceleration'] = df['price_return'].diff()
        
        # 12. Volume profile
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma15'] = df['volume'].rolling(window=15).mean()
        df['relative_volume'] = df['volume'] / df['volume_ma15']
        
        # Drop rows with any missing values after indicator calculations
        df.dropna(inplace=True)
        self.dataframe = df

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
        break
