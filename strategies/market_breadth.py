
import pandas as pd
import numpy as np
import ta
from .base import BaseStrategy
from .indicators import calculate_wma

class DollarRatioStrategy(BaseStrategy):
    def __init__(self, sp500_ticker='^GSPC', dollar_ticker='DX-Y.NYB'):
        super().__init__("Ratios Bursátiles (Dolar)")
        self.sp500_ticker = sp500_ticker
        self.dollar_ticker = dollar_ticker

    def calculate_indicators(self, df: pd.DataFrame, context_data: dict = None) -> pd.DataFrame:
        if context_data is None:
            return df
        
        dollar_df = context_data.get(self.dollar_ticker)
        sp500_df = context_data.get(self.sp500_ticker)
        
        if dollar_df is None or sp500_df is None:
            return df
            
        # Realign
        common_index = df.index.intersection(dollar_df.index).intersection(sp500_df.index)
        d_close = dollar_df.loc[common_index]['Close']
        s_close = sp500_df.loc[common_index]['Close']
        
        ratio = d_close / s_close
        
        # WMA 30 (Weekly? Text says "WMA30 de la ratio ... pendiente negativa"). 
        # "Temporalidad semanal" wasn't explicitly stated but WMA30 usually implies weekly.
        # "Temporalidad mensual" for Coppock, "Semanal" for MACD. Dollar Ratio?
        # "WMA30 de la ratio" -> likely Weekly.
        
        # Resample logic if needed. Assuming inputs are Daily.
        # Construct Weekly Ratio.
        
        ratio_weekly = ratio.resample('W-FRI').last()
        wma30 = calculate_wma(ratio_weekly, 30)
        
        # Slope
        slope = wma30.diff()
        
        # Distance Ratio to WMA30 > 12%
        dist = (abs(ratio_weekly - wma30) / wma30) * 100
        
        # Merge back
        # Create a DF for the ratio signals
        signals_df = pd.DataFrame(index=ratio_weekly.index)
        signals_df['ratio'] = ratio_weekly
        signals_df['wma30'] = wma30
        signals_df['slope'] = slope
        signals_df['dist'] = dist
        
        # Map to daily df
        df['week_key'] = df.index.to_period('W-FRI')
        signals_df.index = signals_df.index.to_period('W-FRI')
        
        df = df.merge(signals_df, left_on='week_key', right_index=True, how='left')
        df[['ratio', 'wma30', 'slope', 'dist']] = df[['ratio', 'wma30', 'slope', 'dist']].ffill()
        
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Signal'] = 0
        if 'slope' not in df.columns:
            return df
            
        # Buy:
        # WMA30 Slope Negative (descending)
        # Dist Ratio to WMA30 > 12%
        # (Wait, if Dist > 12%, it means Ratio is FAR from WMA. 
        # If Ratio is Dollar/SPY.
        # If Ratio falls (Dollar down, Stocks Up).
        # We want to buy Stocks when Ratio is falling? Yes.
        # If Ratio is far from WMA (12%), it means it crashed?
        # Usually Mean Reversion suggests if it is too far, it might snap back.
        # But User says: "Distancia ... superior al 12%".
        # This implies "Deep Overextended Trend" -> High Confidence?
        
        cond_buy = (df['slope'] < 0) & (df['dist'] > 12.0)
        
        # Sell:
        # WMA30 Slope Positive
        # Ratio crosses WMA30 Up
        
        cross_up = (df['ratio'] > df['wma30']) & (df['ratio'].shift(1) <= df['wma30'].shift(1))
        cond_sell = (df['slope'] > 0) & cross_up
        
        df.loc[cond_buy, 'Signal'] = 1
        df.loc[cond_sell, 'Signal'] = -1
        
        return df
        
class PutCallStrategy(BaseStrategy):
    def __init__(self, pc_ticker='^CPC'): # CBOE Put Call Ratio
        super().__init__("Sistema Ratio Put:Call")
        self.pc_ticker = pc_ticker

    def calculate_indicators(self, df: pd.DataFrame, context_data: dict = None) -> pd.DataFrame:
        if context_data is None: return df
        pc_df = context_data.get(self.pc_ticker)
        if pc_df is None: return df
        
        # Align
        pc_series = pc_df['Close'].reindex(df.index).ffill()
        
        # Sentimiento = SMA(10) of CPC
        df['sma10_pc'] = pc_series.rolling(window=10).mean()
        
        # BB [Sentimiento, 42, 2]
        # Bollinger Bands on the SMA10??
        # User: "Sistema P/C = BB [Sentimiento, 42, 2]"
        # usually BB is on Price. Here input is SMA10.
        
        indicator = ta.volatility.BollingerBands(close=df['sma10_pc'], window=42, window_dev=2)
        df['bb_high'] = indicator.bollinger_hband()
        df['bb_low'] = indicator.bollinger_lband()
        df['pc_ratio'] = pc_series
        
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Signal'] = 0
        if 'pc_ratio' not in df.columns: return df
        
        # Buy:
        # SMA10 closes ABOVE BB Upper
        # Cross > 0.97 level (of the ratio itself?? or SMA?)
        # User: "El cruce se produce en un nivel superior a 0.97 de la ratio put:call"
        # Implies PC Ratio > 0.97. (Lots of Puts -> Bearish Sentiment -> Contrarian Buy)
        
        cond_bb_up = df['sma10_pc'] > df['bb_high'] # "Cierra por encima"
        cond_level_buy = df['pc_ratio'] > 0.97
        
        cond_buy = cond_bb_up & cond_level_buy
        
        # Sell:
        # SMA10 closes BELOW BB Lower
        # Cross < 0.82
        
        cond_bb_low = df['sma10_pc'] < df['bb_low']
        cond_level_sell = df['pc_ratio'] < 0.82
        
        cond_sell = cond_bb_low & cond_level_sell
        
        df.loc[cond_buy, 'Signal'] = 1
        df.loc[cond_sell, 'Signal'] = -1
        
        return df

class ADNHNLStrategy(BaseStrategy):
    def __init__(self, ad_ticker='^ADD'): # NYSE Advance Decline
        super().__init__("Línea AD NHNL")
        self.ad_ticker = ad_ticker
        
    def calculate_indicators(self, df: pd.DataFrame, context_data: dict = None) -> pd.DataFrame:
        if context_data is None: return df
        ad_df = context_data.get(self.ad_ticker)
        if ad_df is None: return df
        
        # AD Line is usually Cumulative Sum of (Adv - Dec).
        # ^ADD returns the daily net (Adv - Dec) or count?
        # Usually ^ADD is net number.
        # "Línea AD NHNL" -> implies Advance-Decline Line.
        # We need to construct the Cumulative Line.
        
        ad_vals = ad_df['Close'].reindex(df.index).fillna(0)
        df['ad_line'] = ad_vals.cumsum()
        
        # WMA 60 of AD Line
        df['wma60_ad'] = calculate_wma(df['ad_line'], 60)
        
        return df
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Signal'] = 0
        if 'ad_line' not in df.columns: return df
        
        # Buy: AD Line Crosses WMA60 Up
        cross_up = (df['ad_line'] > df['wma60_ad']) & (df['ad_line'].shift(1) <= df['wma60_ad'].shift(1))
        
        # Sell: Cross Down
        cross_down = (df['ad_line'] < df['wma60_ad']) & (df['ad_line'].shift(1) >= df['wma60_ad'].shift(1))
        
        df.loc[cross_up, 'Signal'] = 1
        df.loc[cross_down, 'Signal'] = -1
        
        return df
