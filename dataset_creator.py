#!/usr/bin/env python3
"""
Dataset Creator for Financial Ticker Data

This script loads a CSV file (e.g., test.csv) containing minute-level financial data,
inspects the data types, checks for missing values in both the columns and expected
timestamps, and logs detailed information to both a log file and the console.

Usage:
    python dataset_creator.py --file test.csv --trading_start "09:00:00" --trading_end "16:00:00"

If trading_start and trading_end are not provided, the script uses the first and last
timestamps recorded for each day as the boundaries for generating an expected minute-
by-minute timeline.
"""

import pandas as pd
import numpy as np
import logging
import sys
import pandas_market_calendars as mcal
import os

# ---------------------------
# Setup Logging Configuration
# ---------------------------
logger = logging.getLogger("dataset_creator")
logger.setLevel(logging.DEBUG)

# Formatter for both console and file logs
formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

# File handler to log to 'dataset_creator.log'
file_handler = logging.FileHandler("dataset_creator.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler to also log to stdout
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# ---------------------------
# Utility Functions
# ---------------------------
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the CSV file into a pandas DataFrame.
    Expects a 'datetime' column for proper time parsing.
    """
    logger.info(f"Loading data from {file_path}")
    try:
        # Parse 'datetime' column as a datetime object directly
        df = pd.read_csv(file_path, parse_dates=["datetime"])
        logger.info(f"Data loaded successfully with shape: {df.shape}")
    except Exception as e:
        logger.error("Error loading data from CSV.", exc_info=True)
        raise e
    return df


def convert_numeric_columns(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    Convert listed columns to numeric types.
    If a column is missing, log a warning.
    """
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            logger.warning(f"Expected numeric column '{col}' is missing from the dataset.")
    return df


def check_missing_timestamps(df: pd.DataFrame, trading_start: str, trading_end: str, freq: str) -> list:
    """
    For each unique day in the DataFrame, generate an expected DateTime index based
    on either the provided trading start/end times or the day's observed min/max times.
    
    Returns a list of missing timestamps across all days.
    """
    missing_ts_list = []

    # Group data by the date part of the index
    for date, group in df.groupby(df.index.date):
        # If trading_start and trading_end are provided, use them; otherwise use observed boundaries
        if trading_start and trading_end:
            expected_start = pd.Timestamp(f"{date} {trading_start}")
            expected_end = pd.Timestamp(f"{date} {trading_end}")
        else:
            expected_start = group.index.min()
            expected_end = group.index.max()

        expected_index = pd.date_range(start=expected_start, end=expected_end, freq=freq)
        actual_index = group.index
        # Find the timestamps in the expected timeline that are missing in the actual data
        missing = expected_index.difference(actual_index)
        logger.info(f"Date: {date} - Expected timestamps: {len(expected_index)}, "
                    f"Actual: {len(actual_index)}, Missing: {len(missing)}")
        missing_ts_list.extend(missing)

    logger.info(f"Total missing timestamps across all dates: {len(missing_ts_list)}")
    return missing_ts_list


def check_missing_timestamps_with_calendar(df: pd.DataFrame, calendar_name: str = 'NYSE', freq: str = 'min') -> list:
    """
    Enhanced check for missing intraday timestamps using the official market calendar.

    It uses the pandas_market_calendars package to generate the expected trading sessions
    for the specified exchange (default 'NYSE') and then compares them to the actual data's index.
    
    Returns:
        missing_ts_list (list): List of missing datetime stamps.
    """
    calendar = mcal.get_calendar(calendar_name)
    start_date = df.index.min().date()
    end_date = df.index.max().date()

    schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    if schedule["market_open"].dtype.tz is None:
         schedule["market_open"] = schedule["market_open"].dt.tz_localize("UTC")
    schedule["market_open"] = schedule["market_open"].dt.tz_convert("America/New_York")
    if schedule["market_close"].dtype.tz is None:
         schedule["market_close"] = schedule["market_close"].dt.tz_localize("UTC")
    schedule["market_close"] = schedule["market_close"].dt.tz_convert("America/New_York")
    missing_ts_list = []

    for day in schedule.index:
        market_open = schedule.loc[day, 'market_open']
        market_close = schedule.loc[day, 'market_close']
        expected_index = pd.date_range(start=market_open, end=market_close, freq=freq)
        group = df[(df.index >= market_open) & (df.index <= market_close)]
        actual_index = group.index
        missing = expected_index.difference(actual_index)
        logger.info(f"{day} ({market_open.time()}-{market_close.time()}): Expected={len(expected_index)}, "
                    f"Actual={len(actual_index)}, Missing={len(missing)}")
        missing_ts_list.extend(missing)

    logger.info(f"Total missing timestamps (using '{calendar_name}' calendar) across all dates: {len(missing_ts_list)}")
    return missing_ts_list


def count_complete_days(df: pd.DataFrame, calendar_name: str = 'NYSE', freq: str = 'min') -> tuple:
    """
    Count the number of trading days with a full dataset (no missing timestamps)
    based on the official market calendar.
    
    Returns:
        (int, list): A tuple where the first element is the number of complete days,
                     and the second element is a list of the dates (as Timestamps) that are complete.
    """
    calendar = mcal.get_calendar(calendar_name)
    start_date = df.index.min().date()
    end_date = df.index.max().date()
    
    # Get the trading schedule from the calendar
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    # Check if the entire column is tz-naive using its dtype
    if schedule["market_open"].dtype.tz is None:
         schedule["market_open"] = schedule["market_open"].dt.tz_localize("UTC")
    # Use .dt accessor to ensure proper conversion
    schedule["market_open"] = schedule["market_open"].dt.tz_convert("America/New_York")
    if schedule["market_close"].dtype.tz is None:
         schedule["market_close"] = schedule["market_close"].dt.tz_localize("UTC")
    schedule["market_close"] = schedule["market_close"].dt.tz_convert("America/New_York")
    
    complete_count = 0
    complete_days = []
    
    for day in schedule.index:
        market_open = schedule.loc[day, 'market_open']
        market_close = schedule.loc[day, 'market_close']
        expected_index = pd.date_range(start=market_open, end=market_close, freq=freq)
        group = df[(df.index >= market_open) & (df.index <= market_close)]
        missing = expected_index.difference(group.index)
        if len(missing) == 0:
            complete_count += 1
            complete_days.append(day)
    
    return complete_count, complete_days


def fill_missing_intraday_data_with_mid_price(df: pd.DataFrame, calendar_name: str = 'NYSE', freq: str = 'min') -> pd.DataFrame:
    """
    Fill missing intraday data with mid price estimations using the market calendar schedule.
    The mid price is estimated as (high + low) / 2.
    For missing rows, the open, high, low, close, and vwap columns will be filled with this mid price,
    and volume and num_trades are assumed to be 0.
    Only trading hours (between market open and market close) are considered.
    """
    calendar = mcal.get_calendar(calendar_name)
    start_date = df.index.min().date()
    end_date = df.index.max().date()
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    if schedule["market_open"].dtype.tz is None:
         schedule["market_open"] = schedule["market_open"].dt.tz_localize("UTC")
    schedule["market_open"] = schedule["market_open"].dt.tz_convert("America/New_York")
    if schedule["market_close"].dtype.tz is None:
         schedule["market_close"] = schedule["market_close"].dt.tz_localize("UTC")
    schedule["market_close"] = schedule["market_close"].dt.tz_convert("America/New_York")
    processed_dfs = []
    
    # For storing the last available row from previous trading day
    last_available_row = None

    for day in schedule.index:
        market_open = schedule.loc[day, 'market_open']
        market_close = schedule.loc[day, 'market_close']
        expected_index = pd.date_range(start=market_open, end=market_close, freq=freq)
        day_df = df[(df.index >= market_open) & (df.index <= market_close)].copy()
        if day_df.empty:
            if last_available_row is not None:
                filled_day = pd.DataFrame(index=expected_index)
                for col in ['open', 'high', 'low', 'close', 'vwap']:
                    filled_day[col] = last_available_row[col]
                filled_day['volume'] = 0
                filled_day['num_trades'] = 0
                processed_dfs.append(filled_day)
                logger.info(f"{day} had no data; filled with previous day's last value.")
            else:
                logger.warning(f"{day} had no data and no previous data available to forward fill.")
            continue
        day_df['mid'] = ((day_df['high'] + day_df['low']) / 2).round(3)
        day_df = day_df.reindex(expected_index)
        day_df['mid'] = day_df['mid'].interpolate(method='time', limit_direction='both').ffill().bfill().round(3)
        
        for col in ['open', 'high', 'low', 'close', 'vwap']:
            day_df[col] = day_df[col].fillna(day_df['mid']).round(3)
            
        for col in ['volume', 'num_trades']:
            day_df[col] = day_df[col].fillna(0).astype(int)
            
        last_available_row = day_df.iloc[-1]
        
        processed_dfs.append(day_df)
    
    processed_df = pd.concat(processed_dfs).sort_index()
    processed_df.drop(columns=['mid'], inplace=True)
    return processed_df


# ---------------------------
# Main Functionality
# ---------------------------
def main():
    # Hardcoded parameters:
    file_path = "/Users/grantray/Desktop/infomer/data/raw/SPY.csv"
    freq = "min"  # Expected frequency (1 minute intervals)
    calendar_name = "NYSE"  # Market calendar to use

    logger.info("Starting dataset creation and inspection.")

    # ---- Capture original column names from CSV file ----
    try:
        # Read the header only (single row) to capture the original columns exactly as in file
        original_columns = pd.read_csv(file_path, nrows=0).columns.tolist()
        logger.info(f"Original CSV columns are: {original_columns}")
    except Exception as e:
        logger.warning("Unable to capture original columns from the CSV file, using later DataFrame columns.", exc_info=True)
        original_columns = None

    # Load the data
    try:
        df = load_data(file_path)
    except Exception as e:
        logger.error("Failed to load the CSV file. Exiting.")
        sys.exit(1)

    if "datetime" not in df.columns:
        logger.error("CSV file must contain a 'datetime' column. Exiting.")
        sys.exit(1)

    try:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df.set_index("datetime", inplace=True)
        # Normalize timestamps to minute resolution:
        df.index = df.index.floor("min")  # This ensures the timestamps align with the 'min' frequency expected later.
        # If the index is tz-naive, assume it's in America/New_York and use it as such (no conversion to UTC)
        if df.index.tz is None:
            df.index = df.index.tz_localize("America/New_York")
    except Exception as e:
        logger.error("Error processing the 'datetime' column.", exc_info=True)
        sys.exit(1)

    expected_numeric_cols = ["open", "high", "low", "close", "volume", "vwap", "num_trades"]
    df = convert_numeric_columns(df, expected_numeric_cols)
    logger.info("Completed data type conversions.")

    missing_counts = df.isnull().sum()
    total_rows = len(df)
    logger.info("Missing values per column:")
    for col, count in missing_counts.items():
        perc = (count / total_rows) * 100 if total_rows > 0 else 0
        logger.info(f"  {col}: {count} missing ({perc:.2f}% of {total_rows} rows)")

    # Count the number of days that have a complete dataset (no missing timestamps)
    complete_count, complete_days = count_complete_days(df, calendar_name=calendar_name, freq=freq)
    logger.info(f"Number of complete days (with full dataset): {complete_count}")

    # Calculate the incomplete days (days with missing dataset)
    calendar = mcal.get_calendar(calendar_name)
    start_date = df.index.min().date()
    end_date = df.index.max().date()
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    all_days = list(schedule.index)
    incomplete_days = [day for day in all_days if day not in complete_days]
    incomplete_count = len(incomplete_days)

    logger.info(f"Number of incomplete days (with missing dataset): {incomplete_count}")
    logger.info("Incomplete days:")
    for day in incomplete_days:
        logger.info(day)

    logger.info("Dataset inspection completed.")

    # Extract ticker from input filename (e.g., "A.csv" -> "A")
    ticker = os.path.splitext(os.path.basename(file_path))[0]
    
    # Fill missing intraday data using mid price estimation
    df_filled = fill_missing_intraday_data_with_mid_price(df, calendar_name=calendar_name, freq=freq)

    # Reset index to include the 'datetime' column in the output CSV file
    df_filled.reset_index(inplace=True)

    # ---- Restore the original column names ----
    if original_columns is not None and len(df_filled.columns) == len(original_columns):
        df_filled.columns = original_columns
        logger.info("Restored column names from the original CSV header.")
    else:
        logger.warning("The number of columns in processed data differs from the original CSV header. Column names may not match exactly.")

    # Save the processed trading data into the "processed" folder with ticker name
    processed_dir = os.path.join(os.path.dirname(file_path), "../processed")
    os.makedirs(processed_dir, exist_ok=True)
    processed_file_path = os.path.join(processed_dir, f"{ticker}.csv")
    df_filled.to_csv(processed_file_path, index=False)
    logger.info(f"Processed trading data saved to {processed_file_path}")


if __name__ == "__main__":
    main() 