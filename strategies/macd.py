
import pandas as pd
import numpy as np
import ta
from .base import BaseStrategy

class MACDStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("MACD Semanal")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Resample to Weekly
        df_weekly = df.resample('W-FRI').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        })
        
        # MACD (12, 23, 9) - Note 23 per user spec (standard is 26)
        macd = ta.trend.MACD(close=df_weekly['Close'], window_slow=23, window_fast=12, window_sign=9)
        df_weekly['macd'] = macd.macd()
        df_weekly['signal_line'] = macd.macd_signal()
        df_weekly['histogram'] = macd.macd_diff()
        
        # WMA 30 (Weekly)
        weights = np.arange(1, 31)
        df_weekly['wma30'] = df_weekly['Close'].rolling(window=30).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        
        # Distance WMA30 to Low
        # "La distancia de la media de 30 semanas ponderada con respecto al mínimo semanal"
        df_weekly['dist_wma30_low'] = (abs(df_weekly['wma30'] - df_weekly['Low']) / df_weekly['Low']) * 100
        
        # Merge back to daily
        df['week_key'] = df.index.to_period('W-FRI')
        df_weekly.index = df_weekly.index.to_period('W-FRI')
        
        cols = ['macd', 'signal_line', 'wma30', 'dist_wma30_low']
        df = df.merge(df_weekly[cols], left_on='week_key', right_index=True, how='left')
        df[cols] = df[cols].ffill()
        
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Signal'] = 0
        
        # Buy: Signal Line >= 0 (Zero cross up OR > 0)
        # Logic: "La línea de señal ... cruza cero al alza o es mayor que cero."
        # This implies checking if it IS positive.
        # But usually you don't buy EVERY day it is positive. You buy WHEN it becomes positive?
        # Or is it a filter? "Señal de compra" usually means the TRIGGER.
        # If I interpret "or is > 0" literally, it generates a purchase signal EVERY tick it is > 0.
        # That means "Stay Long" while > 0.
        # However, standard backtesting triggers on the transition.
        # But the User also adds: "La distancia... debe ser menor del 2%... Así se busca evitar vender en pánico". 
        # Wait, the 2% condition is listed under "Señal de venta" in my summary?
        # User: "Señal de venta: ... < 0. La distancia ... menor del 2%. Así se busca evitar vender em pánico." -> Buying filter or Selling filter?
        # "Avoid selling in panic" implies: IF Signal causes Sell, BUT Price is too far from Mean (Oversold), DON'T SELL.
        # So it's a filter for the SELL signal.
        
        # Buy Signal:
        # Cross 0 Up OR (>0 - implies trend following). I will treat it as a State. 
        # If Signal > 0 -> Buy Zone.
        # To avoid re-buying every day, we use (Current > 0) AND (Prev <= 0) for ENTRY?
        # User said "cruza cero al alza o es mayor que cero". 
        # I'll implement ENTRY when Cross Up occurs.
        # NOTE: If we only implement "Cross Up", and we start data in the middle of a trend, we miss it.
        # But "or is > 0" suggests we can enter anytime? That's risky.
        # I will assume Entry on Cross Up.
        
        # Sell Signal:
        # Cross 0 Down OR (<0). 
        # FILTER: Dist WMA30 vs Low < 2%. prevents selling?
        # "Evitar vender en pánico" -> If panic (price crash far below mean), don't sell?
        # Wait, if price is far below mean, distance is LARGE.
        # If "distancia ... menor del 2%", that means Price is CLOSE to Mean.
        # So "Don't sell if Price is CLOSE to Mean"? That logic seems inverted for "Panic".
        # Panic = Price far below mean. Dist = Large.
        # Maybe text means "Do NOT sell if Dist > X%"?
        # Text: "La distancia ... deberá ser menor del 2%. Así se busca evitar vender em pánico."
        # This implies: You CAN sell ONLY if Distance is SMALL (< 2%).
        # If Distance is LARGE (> 2%), it implies we are far from mean (Panic?), so HOLD (don't sell).
        # Yes, that makes sense. Sell only on Pullback/Correction, not Crash.
        
        # Implementation:
        # Buy: Signal Line crosses 0 Up.
        # Sell: Signal Line crosses 0 Down AND Dist(WMA30, Low) < 2%.
        
        # Create shifted columns for crossover check
        df['signal_line_prev'] = df['signal_line'].shift(1)
        
        # Buy
        # Cross Up: Prev < 0 AND Curr >= 0
        cond_buy = (df['signal_line_prev'] < 0) & (df['signal_line'] >= 0)
        
        # Sell
        # Cross Down: Prev > 0 AND Curr <= 0
        cross_down = (df['signal_line_prev'] > 0) & (df['signal_line'] <= 0)
        filter_dist = df['dist_wma30_low'] < 2.0
        
        cond_sell = cross_down & filter_dist
        
        df.loc[cond_buy, 'Signal'] = 1
        df.loc[cond_sell, 'Signal'] = -1
        
        return df
