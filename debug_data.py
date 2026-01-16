
import yfinance as yf
import pandas as pd
import sys
import os
sys.path.append(os.getcwd())

from app.services.data_service import DataService

def test_yfinance_raw():
    print("--- Testing Raw yfinance ---")
    ticker = "SPY"
    try:
        df = yf.download(ticker, period="1mo", interval="1d", progress=False, auto_adjust=False) 
        print("Columns:", df.columns)
        print("Index:", df.index.name)
        print("Head:\n", df.head())
        
        df.reset_index(inplace=True)
        print("After reset_index columns:", df.columns)
        if 'Date' in df.columns:
            print("Date column type:", type(df.iloc[0]['Date']))
        else:
            print("Date column MISSING")
            
    except Exception as e:
        print(f"YFinance Error: {e}")

def test_data_service():
    print("\n--- Testing DataService ---")
    ds = DataService()
    # Force fresh download by using a random ticker or clearing db?
    # We'll just try SPY
    df = ds.get_data("SPY", start_date="2023-01-01", end_date="2023-01-10")
    if df is not None and not df.empty:
        print("DataService Success. Rows:", len(df))
        print(df.head())
    else:
        print("DataService FAILED: Returned None or Empty")

if __name__ == "__main__":
    test_yfinance_raw()
    test_data_service()
