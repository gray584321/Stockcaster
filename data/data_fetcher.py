import os
import csv
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
from config import TICKERS
import logging
import pandas as pd
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
data_fetcher.py

This script is responsible for fetching historical 1-minute aggregated stock data
for each ticker provided in config.TICKERS from Polygon.io's REST API.

Key Features:
  - Fetches data from 5 years ago until 5 days ago (to avoid the issues with incomplete current day data).
  - Retrieves data in fixed 75-day chunks.
  - Converts API timestamps (provided in UTC) to Eastern Time (America/New_York) before saving.
  - Saves the data to CSV files located in a "raw" folder.
  - If a CSV file for a ticker already exists, the script resumes fetching from one minute after the last
    recorded timestamp.
  - Provides functionality to check for and fill missing intraday data.

Ensure that a .env file with POLYGON_API_KEY is present and that config.TICKERS is properly defined.
"""

# Initialize logging
logging.getLogger(__name__)





# Configuration
def get_date_range():
    print("Getting date range for data fetching.")
    today = datetime.utcnow().date()
    end_date = today - timedelta(days=5)  # 5 days ago (for Free api access use 2 years)
    start_date = end_date - timedelta(days=365*5)  # 5 years before end_date (changed from 2 years)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    print(f"Date range set: Start Date = {start_date_str}, End Date = {end_date_str}")
    return start_date_str, end_date_str

# Get the date range when module is loaded
START_DATE, END_DATE = get_date_range()

def fetch_data(ticker, api_key, start_date=None, end_date=None, min_datetime=None):
    """
    Retrieve 1-minute aggregated data for a given ticker from Polygon.io.
    Data range is from 5 years ago until yesterday (no current day data).
    Fetches data in fixed 30-day chunks. If using free api key, use 13 second delay between requests.
    """
    print(f"Fetching data for ticker: {ticker}, from {start_date} to {end_date}.")
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = END_DATE

    # Ensure we never try to get today's data
    today = datetime.utcnow().date()
    end_datetime_capped = min(
        datetime.strptime(end_date, "%Y-%m-%d").date(),
        today - timedelta(days=1)
    )
    end_date = end_datetime_capped.strftime("%Y-%m-%d")

    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    # Fetch data in fixed 30-day chunks
    chunk_size = 75  # days per chunk
    while current_date <= end_datetime:
        # Calculate chunk_end so that each chunk covers 30 consecutive days
        chunk_end = min(current_date + timedelta(days=chunk_size - 1), end_datetime)

        log_msg = f"Processing {current_date.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}"
        print(log_msg)

        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{current_date.strftime('%Y-%m-%d')}/{chunk_end.strftime('%Y-%m-%d')}"
        params = {
            "limit": 50000,
            "sort": "asc"
        }

        total_records = 0
        while url:
            response = requests.get(url, headers=headers, params=params if "?" not in url else None)

            if response.status_code != 200:
                error_msg = f"Error {response.status_code}: {response.text[:100]}..."
                logging.error(error_msg)
                print(f"✗ {error_msg}")
                if response.status_code == 429:  # Rate limit error from the API side
                    logging.warning("Rate limit hit, waiting 65 seconds...")
                    print("Rate limit hit, waiting 65 seconds...")
                    time.sleep(65)
                    continue  # Retry the same request
                else:
                    time.sleep(30)
                    break

            data = response.json()
            if data.get("status") != "OK":
                api_error_msg = f"API Error: {data.get('error', 'Unknown error')}"
                logging.error(api_error_msg)
                print(f"✗ {api_error_msg}")
                break

            if "results" in data:
                records = data["results"]
                total_records += len(records)
                save_to_csv(ticker, records, min_datetime)

            url = data.get("next_url")
            if url and "?" in url:
                url = url.split("&apiKey=")[0]

        print(f"Retrieved {total_records} total records for current chunk")
        current_date = chunk_end + timedelta(days=1)

    print(f"Data fetching completed for ticker: {ticker}.")

def save_to_csv(ticker, data, min_datetime=None):
    print(f"Saving {len(data)} records to CSV for ticker: {ticker}.")
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Create path relative to script location
    output_dir = os.path.join(script_dir, "raw")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        error_msg = f"Error creating directory: {e}"
        logging.error(error_msg)
        print(f"Error creating directory: {e}")
        return
        
    output_file = os.path.join(output_dir, f"{ticker}.csv")
    file_exists = os.path.exists(output_file)

    try:
        with open(output_file, "a", newline="") as csvfile:
            fieldnames = ["datetime", "open", "high", "low", "close", "volume", "vwap", "num_trades"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            # Define Eastern time zone
            eastern_tz = pytz.timezone("America/New_York")
            for entry in data:
                # Get the UTC datetime from the API timestamp (milliseconds)
                utc_dt = datetime.utcfromtimestamp(entry["t"] / 1000)
                # Mark it as UTC
                utc_dt = utc_dt.replace(tzinfo=pytz.utc)
                # Convert to Eastern time
                eastern_dt = utc_dt.astimezone(eastern_tz)
                # If a minimum datetime (resume point) is provided, skip records older than it.
                if min_datetime is not None and eastern_dt < min_datetime:
                    continue
                dt = eastern_dt.strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow({
                    "datetime": dt,
                    "open": entry.get("o"),
                    "high": entry.get("h"),
                    "low": entry.get("l"),
                    "close": entry.get("c"),
                    "volume": entry.get("v"),
                    "vwap": entry.get("vw"),
                    "num_trades": entry.get("n"),
                })
        log_msg = f"Saved {len(data)} records to {ticker}.csv"
        print(log_msg)
        print(f"✓ {log_msg}")
    except Exception as e:
        error_msg = f"Error writing to file: {e}"
        logging.error(error_msg)
        print(f"✗ {error_msg}")

def get_last_datetime_from_csv(ticker):
    print(f"Getting last datetime from CSV for ticker: {ticker}.")
    """
    Reads the existing CSV file for the given ticker (if it exists) and returns
    the maximum datetime (as a datetime object) found in the "datetime" column.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "raw", f"{ticker}.csv")
    
    if not os.path.exists(file_path):
        print(f"No existing CSV file found for ticker: {ticker}.")
        return None
        
    last_dt = None
    with open(file_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dt = datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S")
            if last_dt is None or dt > last_dt:
                last_dt = dt
    print(f"Last datetime found in CSV for ticker {ticker}: {last_dt}")
    return last_dt

def fill_missing_intraday_data(ticker, threshold_minutes=5):
    """
    Checks the raw CSV file for the given ticker for missing one-minute intraday data.
    For each trading day, it compares the actual timestamps to the expected one-minute frequency.
    If gaps are found (within the same trading day), it fetches data for that day using fetch_data()
    and updates the raw CSV.
    """
    print(f"Filling missing intraday data for ticker: {ticker} ...")

    # Load API key from .env
    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("Missing API key. Cannot fetch missing data.")
        return

    # Locate the raw CSV for the ticker.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(script_dir, "raw")
    raw_file = os.path.join(raw_dir, f"{ticker}.csv")
    if not os.path.exists(raw_file):
        print(f"No raw CSV file exists for ticker: {ticker}.")
        return

    df = pd.read_csv(raw_file)
    if "datetime" not in df.columns:
        print("Column 'datetime' is missing in CSV.")
        return
    df["DateTime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")

    # For each trading day, determine if there are missing one-minute intervals.
    df["Date"] = df["DateTime"].dt.date
    unique_days = df["Date"].unique()
    new_data_fetched = False

    for day in unique_days:
        day_mask = df["Date"] == day
        day_df = df.loc[day_mask].copy().sort_values("DateTime")
        day_start = day_df["DateTime"].min()
        day_end = day_df["DateTime"].max()
        full_range = pd.date_range(start=day_start, end=day_end, freq="1min")
        missing_times = full_range.difference(day_df["DateTime"])
        if not missing_times.empty:
            print(f"[{ticker}] Detected {len(missing_times)} missing minutes on {day}. Fetching data for that day...")
            day_str = pd.Timestamp(day).strftime("%Y-%m-%d")
            fetch_data(ticker, api_key, start_date=day_str, end_date=day_str)
            new_data_fetched = True

    if new_data_fetched:
        # Re-read the updated CSV.
        df_updated = pd.read_csv(raw_file)
        print(f"Updated raw CSV for {ticker} after fetching missing intraday data.")
    else:
        print(f"No missing intraday data found for {ticker}.")

    return

def validate_ticker(ticker, api_key):
    """
    Validate if a ticker exists and is active using Polygon.io's Ticker Details endpoint.
    """
    print(f"Validating ticker: {ticker}.")
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "OK" and data.get("results"):
                is_active = data["results"].get("active", False)
                print(f"Ticker {ticker} validation successful. Active status: {is_active}")
                return is_active
        elif response.status_code == 404:
            log_msg = f"{ticker} not found"
            logging.warning(log_msg)
            print(f"✗ {log_msg}")
            return False
        else:
            error_msg = f"Error validating {ticker}: {response.status_code}, {response.text}"
            logging.error(error_msg)
            print(f"✗ {error_msg}")
            return False
    except Exception as e:
        error_msg = f"Validation error: {str(e)}"
        logging.error(error_msg)
        print(f"✗ {error_msg}")
        return False
    print(f"Ticker {ticker} validation failed or ticker not active.")
    return False

def process_ticker(ticker, api_key):
    """
    Processes an individual ticker by validating it, determining the correct start date for fetching,
    calling fetch_data, and logging the completion time.
    """
    print(f"\n🔍 Processing ticker: {ticker}")

    if not validate_ticker(ticker, api_key):
        print(f"⏩ Ticker {ticker} is not active. Skipping.")
        return

    last_dt = get_last_datetime_from_csv(ticker)
    if last_dt:
        # Compute the minimum datetime to fetch new data from: one minute past the last recorded entry.
        min_datetime = last_dt + timedelta(minutes=1)
        fetch_start = min_datetime.strftime("%Y-%m-%d")
        print(f"Existing CSV found for {ticker}. Resuming data fetch from {fetch_start} (records after {last_dt.strftime('%Y-%m-%d %H:%M:%S')}) to {END_DATE}.")
    else:
        fetch_start = START_DATE
        min_datetime = None
        print(f"No CSV exists for {ticker}. Fetching full history from {fetch_start} to {END_DATE}.")

    fetch_data(ticker, api_key, start_date=fetch_start, end_date=END_DATE, min_datetime=min_datetime)

    completion_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Completed processing {ticker} at {completion_time}")

def main():
    print("Starting data fetcher main function.")
    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY")
    
    if not api_key:
        error_msg = "Missing API key. Create a .env file with POLYGON_API_KEY=your_key"
        logging.error(error_msg)
        print(error_msg)
        exit(1)

    log_msg = f"\n🚀 Starting data fetch for {len(TICKERS)} tickers"
    print(log_msg)
    print(log_msg)
    log_msg = f"⏳ Historical range: {START_DATE} to {END_DATE}\n"
    print(log_msg)
    print(log_msg)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_ticker, ticker, api_key): ticker for ticker in TICKERS}

        for future in as_completed(futures):
            ticker = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {ticker}: {e}")

    completion_all_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"\nCompleted all processing at {completion_all_time}"
    print(log_msg)
    print("Data fetcher main function completed.")

if __name__ == "__main__":
    main()