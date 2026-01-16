
import pandas as pd
import numpy as np
from datetime import datetime
from strategies.base import BaseStrategy

class Backtester:
    def __init__(self, initial_capital=10000.0, commission=0.0):
        self.initial_capital = initial_capital
        self.commission = commission

    def run(self, df: pd.DataFrame, strategy: BaseStrategy, stop_loss_pct: float = None):
        """
        Runs the backtest.
        :param df: DataFrame with 'Close', 'High', 'Low' and strategy indicators/signals.
                   Must have 'Signal' column populated by strategy.
        :param stop_loss_pct: Stop loss percentage (e.g. 0.08 for 8%). 
                              If None, uses strategy default or internal logic.
        """
        if df is None or df.empty or 'Signal' not in df.columns:
            return {'error': 'Invalid Data or No Signals'}

        # Ensure sorted
        df = df.sort_index()

        trades = []
        equity = []
        cash = self.initial_capital
        position = 0
        entry_price = 0
        max_price_in_trade = 0 # For trailing stop logic if needed, or fixed stop
        
        # Stop Loss Logic:
        # User specified "Stop-loss del 8% relativo al precio de la señal"
        # MACD: "Stop-loss del 8% desde la entrada"
        
        for date, row in df.iterrows():
            price = row['Close']
            high = row['High']
            low = row['Low']
            signal = row['Signal']
            
            # Check Stop Loss if in position
            if position > 0 and stop_loss_pct:
                stop_price = entry_price * (1 - stop_loss_pct)
                if low <= stop_price:
                    # Trigger Stop Loss
                    exit_price = stop_price # Assume we got filled at stop
                    # Slippage could be modeled but keep simple
                    
                    pnl = (exit_price - entry_price) * position - self.commission
                    cash += (position * exit_price)
                    trades.append({
                        'type': 'loss_stop',
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'return_pct': (exit_price - entry_price) / entry_price
                    })
                    position = 0
                    entry_price = 0
                    
            # Execute Signals
            # Buy
            if signal == 1 and position == 0:
                # Enter Long
                entry_price = price 
                # Position Size: Use all cash
                position = cash / entry_price # Fractional shares allowed? Assume yes or calc floor
                cash = 0
                entry_date = date
                
            # Sell
            elif signal == -1 and position > 0:
                # Exit Long
                exit_price = price
                pnl = (exit_price - entry_price) * position - self.commission
                cash += (position * exit_price)
                trades.append({
                    'type': 'sell_signal',
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'return_pct': (exit_price - entry_price) / entry_price
                })
                position = 0
                entry_price = 0
                
            # Update Equity Curve
            current_val = cash + (position * price)
            equity.append({'date': date, 'equity': current_val})

        # Close open position at end
        if position > 0:
            final_price = df.iloc[-1]['Close']
            pnl = (final_price - entry_price) * position
            cash += position * final_price
            trades.append({
                'type': 'open_end',
                'entry_date': entry_date,
                'exit_date': df.index[-1],
                'entry_price': entry_price,
                'exit_price': final_price,
                'pnl': pnl,
                'return_pct': (final_price - entry_price) / entry_price
            })
            
        equity_df = pd.DataFrame(equity).set_index('date')
        
        # Calculate Metrics
        total_return = (cash - self.initial_capital) / self.initial_capital
        df_trades = pd.DataFrame(trades)
        
        if not df_trades.empty:
            win_rate = len(df_trades[df_trades['pnl'] > 0]) / len(df_trades)
            avg_return = df_trades['return_pct'].mean()
        else:
            win_rate = 0
            avg_return = 0
            
        return {
            'final_balance': cash,
            'total_return_pct': total_return * 100,
            'trades': trades,
            'win_rate': win_rate,
            'equity_curve': equity_df.to_dict(orient='index')
        }
