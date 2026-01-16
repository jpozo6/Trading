
import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(__file__))

from app.services.data_service import DataService
from app.services.backtester import Backtester
from strategies.coppock import CoppockStrategy
from strategies.macd import MACDStrategy
from strategies.moving_averages import MovingAveragesStrategy
from strategies.weinstein import WeinsteinStrategy

def verify():
    print("Verifying Trading System...")
    ds = DataService()
    bt = Backtester()
    
    ticker = 'SPY'
    print(f"Fetching data for {ticker}...")
    df = ds.get_data(ticker)
    
    if df is None or df.empty:
        print("FAIL: No data fetched")
        return

    print("Data Fetched. Running Strategies...")
    
    strategies = [CoppockStrategy(), MACDStrategy(), MovingAveragesStrategy()]
    
    for strategy in strategies:
        print(f"\nTesting {strategy.name}...")
        try:
            results_df = strategy.analyze(df)
            backtest_res = bt.run(results_df, strategy)
            print(f"Success. Total Return: {backtest_res['total_return_pct']:.2f}%")
            print(f"Trades: {len(backtest_res['trades'])}")
        except Exception as e:
            print(f"FAIL: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    verify()
