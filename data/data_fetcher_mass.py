import os
import csv
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz
from tqdm import tqdm
import pandas as pd

"""
data_fetcher_mass.py

This script is responsible for fetching historical 1-minute aggregated stock data
for each active ticker retrieved from Polygon.io's REST API.

Key Features:
  - Fetches active tickers from Polygon.io.
  - For each active ticker, it retrieves 5 years of historical data (until 5 days ago) in fixed 75-day chunks.
  - Converts API timestamps (provided in UTC) to Eastern Time (America/New_York) before saving.
  - Saves the data to CSV files located in a "raw" folder.
  - If a CSV file for a ticker already exists, the script resumes fetching from one minute after the last
    recorded timestamp.
  - Provides functionality to check for and fill missing intraday data.
  
Ensure that a .env file with POLYGON_API_KEY is present.
"""

# Initialize logging and load environment once
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
load_dotenv()

# Constants
CHUNK_SIZE = 75  # days per chunk
MAX_WORKERS = 24  # Increased to use more threads for concurrent processing
RATE_LIMIT_SLEEP = 65
API_BASE_URL = "https://api.polygon.io/v2"
EASTERN_TZ = pytz.timezone("America/New_York")
UTC_TZ = pytz.UTC
REQUEST_TIMEOUT = 30  # seconds

def get_date_range():
    """Calculate date range for data fetching."""
    today = datetime.now(UTC_TZ).date()
    end_date = today  # use today's date to get the absolute latest data available
    start_date = end_date - timedelta(days=365*5)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

START_DATE, END_DATE = get_date_range()

class PolygonAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=200,
            pool_maxsize=200,
            max_retries=3,
            pool_block=True
        )
        self.session = requests.Session()
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "polygon-data-fetcher/1.0"
        })

    def get_active_tickers(self):
        """Fetch active American tickers using session for connection pooling."""
        url = 'https://api.polygon.io/v3/reference/tickers'
        params = {
            'market': 'stocks',
            'active': 'true',
            'locale': 'us',           # Ensure we target American stocks
            'limit': 1000,
            'apiKey': self.api_key
        }
        
        all_tickers = []
        with tqdm(desc="Fetching American tickers", unit=" tickers") as pbar:
            while True:
                try:
                    response = self.session.get(url, params=params)
                    if response.status_code == 429:
                        time.sleep(RATE_LIMIT_SLEEP)
                        continue
                    
                    data = response.json()
                    results = data.get('results', [])
                    if not results:
                        break
                        
                    new_tickers = [ticker['ticker'] for ticker in results]
                    all_tickers.extend(new_tickers)
                    pbar.update(len(new_tickers))
                    
                    next_url = data.get('next_url')
                    if not next_url:
                        break
                        
                    url = next_url.split('&apiKey=')[0]
                    params = {'apiKey': self.api_key}
                    
                except Exception as e:
                    logger.error(f"Error fetching tickers: {e}")
                    time.sleep(5)
                    
        return all_tickers

    def fetch_ticker_data(self, ticker, start_date, end_date):
        """Fetch data for a single ticker chunk with improved error handling."""
        url = f"{API_BASE_URL}/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}"
        params = {
            "limit": 50000,
            "sort": "asc",
            "apiKey": self.api_key
        }
        
        retries = 3
        while retries > 0:
            try:
                response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                if response.status_code == 429:  # Rate limit
                    time.sleep(RATE_LIMIT_SLEEP)
                    continue
                elif response.status_code != 200:
                    logger.warning(f"Error {response.status_code} fetching {ticker}: {response.text}")
                    retries -= 1
                    time.sleep(5)
                    continue
                    
                return response.json().get('results', [])
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error for {ticker}: {str(e)}")
                retries -= 1
                time.sleep(5)
                
        return []  # Return empty list if all retries failed

class DataManager:
    def __init__(self, api):
        self.api = api
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
        os.makedirs(self.output_dir, exist_ok=True)

    def get_last_datetime(self, ticker):
        """Get the last datetime from CSV by reading only the last line."""
        file_path = os.path.join(self.output_dir, f"{ticker}.csv")
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, 'rb') as f:
                # Move to the end of file
                f.seek(-2, os.SEEK_END)
                # Read backwards until newline is found
                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)
                last_line = f.readline().decode()

            # Parse CSV line using csv.reader (assumes CSV format is maintained)
            import csv
            reader = csv.reader([last_line])
            row = next(reader)
            dt_str = row[0]  # Assumes the first column is "datetime"
            last_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            return EASTERN_TZ.localize(last_dt)
        except Exception as e:
            logger.error(f"Error reading last datetime for {ticker}: {e}")
            return None

    def save_data(self, ticker, data, min_datetime=None):
        """Save data to CSV efficiently."""
        if not data:
            return
            
        file_path = os.path.join(self.output_dir, f"{ticker}.csv")
        file_exists = os.path.exists(file_path)
        
        try:
            new_records = []
            for entry in data:
                utc_dt = datetime.fromtimestamp(entry["t"] / 1000, UTC_TZ)
                eastern_dt = utc_dt.astimezone(EASTERN_TZ)
                
                if min_datetime and eastern_dt <= min_datetime:
                    continue
                    
                new_records.append({
                    "datetime": eastern_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "open": entry.get("o"),
                    "high": entry.get("h"),
                    "low": entry.get("l"),
                    "close": entry.get("c"),
                    "volume": entry.get("v"),
                    "vwap": entry.get("vw"),
                    "num_trades": entry.get("n")
                })
            
            if new_records:
                df = pd.DataFrame(new_records)
                df.to_csv(file_path, mode='a', header=not file_exists, index=False)
                
        except Exception as e:
            logger.error(f"Error saving {ticker} data: {e}")

def process_ticker(ticker, api, data_manager, pbar=None):
    """Process individual ticker and update it to the absolute latest data available."""
    try:
        now = datetime.now(UTC_TZ)
        last_dt = data_manager.get_last_datetime(ticker)
        
        if last_dt:
            start_date = last_dt.date()  # resume from the last recorded timestamp regardless of staleness
        else:
            # No CSV exists; perform a preliminary check of recent data
            recent_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
            recent_data = api.fetch_ticker_data(ticker, recent_start, END_DATE)
            if not recent_data:
                tqdm.write(f"⏩ {ticker}: No recent data in the last week. Skipping.")
                if pbar:
                    pbar.update(1)
                return
            start_date = datetime.strptime(START_DATE, "%Y-%m-%d").date()

        # Calculate chunks for the date range
        end_date = datetime.strptime(END_DATE, "%Y-%m-%d").date()
        current_start = start_date
        
        while current_start <= end_date:
            current_end = min(current_start + timedelta(days=CHUNK_SIZE), end_date)
            chunk_data = api.fetch_ticker_data(
                ticker,
                current_start.strftime("%Y-%m-%d"),
                current_end.strftime("%Y-%m-%d")
            )
            
            if chunk_data:
                data_manager.save_data(ticker, chunk_data, last_dt)
                tqdm.write(f"✓ {ticker}: Saved chunk {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
            
            # Add a small delay between chunks to avoid rate limiting
            time.sleep(0.5)
            current_start = current_end + timedelta(days=1)
            
        if pbar:
            pbar.update(1)
            
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        if pbar:
            pbar.update(1)

def main():
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        logger.error("Missing API key")
        return

    api = PolygonAPI(api_key)
    data_manager = DataManager(api)
    
    tickers = api.get_active_tickers()
    if not tickers:
        logger.error("No tickers found")
        return

    with tqdm(total=len(tickers), desc="Processing tickers") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(process_ticker, ticker, api, data_manager, pbar)
                for ticker in tickers
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Future error: {e}")

if __name__ == "__main__":
    main() 