
import yfinance as yf
import pandas as pd
import sqlite3
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'trading_data.db')

class DataService:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prices (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, date)
            )
        ''')
        conn.commit()
        conn.close()

    def get_data(self, ticker: str, start_date: str = None, end_date: str = None, interval: str = '1d'):
        """
        Get historical data for a ticker. 
        Checks local DB first, then fetches from yfinance if needed.
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # normalize ticker
        ticker = ticker.upper()

        # 1. Try to fetch from DB
        df_db = self._fetch_from_db(ticker, start_date, end_date)
        
        # 2. If data is sufficient, return it
        # Logic: If we have data up to recently (e.g. yesterday or today), we might not need to download.
        # But for simplicity, if we request a range, we want to ensure we have it.
        # Efficient sync: Check the max date in DB. If < end_date, fetch missing chunk.
        
        last_date_in_db = self._get_last_date(ticker)
        
        if last_date_in_db:
            last_date_dt = datetime.strptime(last_date_in_db, '%Y-%m-%d')
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            if last_date_dt < end_date_dt - timedelta(days=1):
                 # We need to update
                logger.info(f"Updating data for {ticker} from {last_date_dt} to {end_date}")
                self._download_and_save(ticker, start=last_date_dt + timedelta(days=1), end=end_date_dt + timedelta(days=1))
        else:
            # No data, download all
            logger.info(f"Downloading all data for {ticker}")
            self._download_and_save(ticker, start=start_date, end=end_date)
            
        # Re-fetch from DB to get full range
        return self._fetch_from_db(ticker, start_date, end_date)

    def _download_and_save(self, ticker, start, end):
        try:
            # yfinance download
            df = yf.download(ticker, start=start, end=end, interval='1d', progress=False, auto_adjust=False)
            if df.empty:
                logger.warning(f"No data found for {ticker}")
                return

            # Handle MultiIndex columns (remove ticker level)
            if isinstance(df.columns, pd.MultiIndex):
                # We expect the ticker to be at level 1. Dropping it leaves ['Open', 'High', 'Low', 'Close', 'Volume']
                df.columns = df.columns.droplevel(1)

            # Ensure index is datetime
            df.index = pd.to_datetime(df.index)
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Rename columns to ensure standard names (Title Case)
            # Yfinance usually gives: Date, Open, High, Low, Close, Volume
            # Sometimes 'Adj Close' is present.
            df = df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
            
            # Standardize column names to lowercase for local processing if preferred, 
            # but our DB schema uses lowercase.
            
            # Check required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                # Fallback: maybe columns are capitalized?
                df.columns = [c.lower() for c in df.columns]
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                     logger.error(f"Missing columns for {ticker}: {missing}. Found: {df.columns}")
                     return

            # Convert date to string for SQLite
            # Vectorized creation of list of tuples
            records = list(zip(
                [ticker] * len(df),
                df['date'].dt.strftime('%Y-%m-%d'),
                df['open'],
                df['high'],
                df['low'],
                df['close'],
                df['volume'].fillna(0).astype(int)
            ))
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT OR REPLACE INTO prices (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', records)
            conn.commit()
            conn.close()
            logger.info(f"Saved {len(records)} records for {ticker}")
            
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
            import traceback
            traceback.print_exc()

    def _fetch_from_db(self, ticker, start_date, end_date):
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT date, open, high, low, close, volume 
            FROM prices 
            WHERE ticker = ? AND date >= ? AND date <= ?
            ORDER BY date ASC
        '''
        df = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))
        conn.close()
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        
        return df

    def _get_last_date(self, ticker):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT MAX(date) FROM prices WHERE ticker = ?', (ticker,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    def get_sp500_tickers(self):
        # Fallback or scraper
        try:
            import requests
            from io import StringIO
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers)
            table = pd.read_html(StringIO(response.text))
            df = table[0]
            tickers = df['Symbol'].tolist()
            return [t.replace('.', '-') for t in tickers]
        except Exception as e:
            logger.error(f"Error fetching SP500 tickers: {e}")
            return []

    def get_nasdaq_tickers(self):
         # Placeholder for simple list or scraping
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'PYPL', 'INTC', 'CSCO', 'NFLX', 'ADBE', 'AMD']

if __name__ == "__main__":
    ds = DataService()
    print("Fetching SPY data...")
    df = ds.get_data("SPY")
    print(df.tail())
