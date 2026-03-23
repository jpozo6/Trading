
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pandas as pd
import numpy as np
import sys
import os

# Add root directory to sys.path to allow importing strategies
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.services.data_service import DataService
from app.services.backtester import Backtester
from strategies.coppock import CoppockStrategy
from strategies.macd import MACDStrategy
from strategies.moving_averages import MovingAveragesStrategy
from strategies.weinstein import WeinsteinStrategy
from strategies.market_breadth import DollarRatioStrategy, PutCallStrategy, ADNHNLStrategy

app = FastAPI(title="Trading Analysis API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_service = DataService()
backtester = Backtester(initial_capital=10000)

# Instantiate Strategies
strategies = {
    'coppock': CoppockStrategy(),
    'macd': MACDStrategy(),
    'golden_cross': MovingAveragesStrategy(),
    'weinstein': WeinsteinStrategy(),
    'dollar_ratio': DollarRatioStrategy(),
    'put_call': PutCallStrategy(),
    'ad_nhnl': ADNHNLStrategy()
}

def get_market_context():
    # Fetch common context data
    tickers = ['^GSPC', 'DX-Y.NYB', '^ADD', '^CPC']
    data = {}
    for t in tickers:
        df = data_service.get_data(t)
        if df is not None:
            data[t] = df
    return data

@app.get("/tickers/{index}")
def get_tickers(index: str):
    if index.lower() == 'sp500':
        return data_service.get_sp500_tickers()
    elif index.lower() == 'nasdaq':
        return data_service.get_nasdaq_tickers()
    else:
        raise HTTPException(status_code=404, detail="Index not found")

@app.get("/strategies")
def get_strategies():
    return [{"id": k, "name": v.name} for k, v in strategies.items()]

@app.post("/analyze")
def analyze(ticker: str = Body(...), strategy_id: str = Body(...), years: int = Body(5)):
    strategy = strategies.get(strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
        
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
    df = data_service.get_data(ticker, start_date=start_date)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Data not found for ticker")
        
    # Context data for Breadth strategies
    context = get_market_context()
    
    # Calculate
    # Some strategies (Weinstein) need benchmark, others need context
    # We can pass context to all, BaseStrategy handles it if updated, 
    # but currently BaseStrategy.analyze signature is `analyze(df)`.
    # Weinstein overrides it. Breadth overrides calculate_indicators.
    # We need a unified call or check type.
    
    if strategy_id == 'weinstein':
        results = strategy.analyze(df, benchmark_df=context.get('^GSPC'))
    elif strategy_id in ['dollar_ratio', 'put_call', 'ad_nhnl']:
        # These define calculate_indicators with context
        results = strategy.analyze(df) # Analyze calls calculate_indicators
        # Wait, BaseStrategy.analyze calls calc_ind(df). It doesn't pass context.
        # I need to manually call the pipeline if context is needed or update BaseStrategy.
        # Current BaseStrategy.analyze:
        # df = self.calculate_indicators(df.copy())
        # df = self.generate_signals(df)
        
        # Override for breadth in python logic:
        df_ind = strategy.calculate_indicators(df.copy(), context_data=context)
        results = strategy.generate_signals(df_ind)
    else:
        results = strategy.analyze(df)
        
    latest = strategy.get_latest_signal(results)
    
    # Helper function to sanitize NaN/Inf values for JSON
    def sanitize_for_json(obj):
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(item) for item in obj]
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        elif isinstance(obj, (np.floating, np.integer)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif pd.isna(obj):
            return None
        return obj
    
    # Sanitize latest_signal
    latest = sanitize_for_json(latest)
    
    # Return all records for charting (user controls time range via years param)
    # Convert index (datetime) to string for JSON serialization
    results_copy = results.reset_index()
    if 'date' in results_copy.columns:
        results_copy['date'] = results_copy['date'].dt.strftime('%Y-%m-%d')
    elif 'index' in results_copy.columns:
        results_copy['date'] = pd.to_datetime(results_copy['index']).dt.strftime('%Y-%m-%d')
        results_copy.drop(columns=['index'], inplace=True)
    
    # Replace NaN/Inf with None for JSON compliance
    results_copy = results_copy.replace([np.inf, -np.inf], np.nan)
    chart_data = results_copy.where(pd.notnull(results_copy), None).to_dict(orient='records')
    chart_data = [sanitize_for_json(record) for record in chart_data]
    
    return {
        "ticker": ticker,
        "strategy": strategy.name,
        "latest_signal": latest,
        "chart_data": chart_data
    }

@app.post("/backtest")
def run_backtest(ticker: str = Body(...), strategy_id: str = Body(...), years: int = Body(5)):
    strategy = strategies.get(strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")

    start_date = (pd.Timestamp.now() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
    df = data_service.get_data(ticker, start_date=start_date)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Data not found")
        
    context = get_market_context()
    
    # Prepare Data
    if strategy_id == 'weinstein':
        df_processed = strategy.analyze(df, benchmark_df=context.get('^GSPC'))
    elif strategy_id in ['dollar_ratio', 'put_call', 'ad_nhnl']:
        df_ind = strategy.calculate_indicators(df.copy(), context_data=context)
        df_processed = strategy.generate_signals(df_ind)
    else:
        df_processed = strategy.analyze(df)
        
    # Run Backtest
    # Need to normalize Stop Loss (User defined per strategy).
    # Default 8% (0.08) for many.
    
    stop_loss = 0.08
    if strategy_id == 'weinstein':
        stop_loss = None # handled internally by 'stop_risk' signal logic? 
        # Strategy logic has exit condition: stop_risk >= 30.
        # But also Backtester has "Hard Stop".
        # User: "Riesgo stop ... será inferior al 9% (entry)... riesgo stop ... >= 30% (exit)".
        # This is a dynamic stop? Or trailing? 
        # I'll let the strategy signals handle the dynamic exit, and maybe set a hard catastrophe stop.
        pass
        
    results = backtester.run(df_processed, strategy, stop_loss_pct=stop_loss)
    
    return results

@app.post("/backtest_all")
def run_backtest_all(ticker: str = Body(...), years: int = Body(5)):
    """Run all strategies on a single ticker and return comparison table."""
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
    df = data_service.get_data(ticker, start_date=start_date)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Data not found for ticker")
    
    context = get_market_context()
    results_table = []
    
    for strategy_id, strategy in strategies.items():
        try:
            # Process data based on strategy type
            if strategy_id == 'weinstein':
                df_processed = strategy.analyze(df.copy(), benchmark_df=context.get('^GSPC'))
            elif strategy_id in ['dollar_ratio', 'put_call', 'ad_nhnl']:
                df_ind = strategy.calculate_indicators(df.copy(), context_data=context)
                df_processed = strategy.generate_signals(df_ind)
            else:
                df_processed = strategy.analyze(df.copy())
            
            # Run backtest
            stop_loss = 0.08 if strategy_id != 'weinstein' else None
            backtest_result = backtester.run(df_processed, strategy, stop_loss_pct=stop_loss)
            
            results_table.append({
                "strategy_id": strategy_id,
                "strategy_name": strategy.name,
                "total_return_pct": backtest_result.get("total_return_pct", 0),
                "win_rate": backtest_result.get("win_rate", 0),
                "final_balance": backtest_result.get("final_balance", 0),
                "total_trades": len(backtest_result.get("trades", []))
            })
        except Exception as e:
            # If a strategy fails, add error entry
            results_table.append({
                "strategy_id": strategy_id,
                "strategy_name": strategy.name,
                "total_return_pct": 0,
                "win_rate": 0,
                "final_balance": 0,
                "total_trades": 0,
                "error": str(e)
            })
    
    # Sort by total return descending
    results_table.sort(key=lambda x: x.get("total_return_pct", 0), reverse=True)
    
    return {"ticker": ticker, "years": years, "results": results_table}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
