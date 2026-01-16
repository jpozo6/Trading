
import pandas as pd
import numpy as np
import ta
from .base import BaseStrategy

class MovingAveragesStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Cruce de Medias (Dorado/Muerte)")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Daily SMAs
        df['sma50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        df['sma200'] = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()
        
        df['dist_sma200_low'] = (abs(df['sma200'] - df['Low']) / df['Low']) * 100
        
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Signal'] = 0
        
        # Buy:
        # Golden Cross Environment: SMA50 > SMA200.
        # Trigger: When it happens? Or anytime?
        # Usually Cross Strategy trades the Cross itself. 
        # Condition: "Estar en un entorno de Cruce Dorado ... comprar cuando SMA50 > SMA200"
        # AND "Dist(SMA200, Low) < 1%".
        # This implies: If SMA50 > SMA200 (Golden), AND Price is very close to SMA200 (Pullback?), BUY.
        # This is a "Buy the Dip in Uptrend" strategy, not just "Buy the Cross".
        # Ah, "La distancia de la SMA200 con respecto al mínimo diario debe ser menor al 1%".
        # This means Price Low is within 1% of SMA200. 
        # Since in Golden Cross SMA200 is support, this is buying support.
        
        # Buy Trigger: SMA50 > SMA200 AND Dist(SMA200, Low) <= 1%
        
        # Sell:
        # "Estar en un entorno de Cruce de la Muerte" -> SMA50 < SMA200.
        # Does it mean Sell immediately when Cross happens? Or just "Be Short"?
        # Usually "Señal de venta" is the trigger.
        # If I am Long, and SMA50 crosses below SMA200 -> Sell.
        
        # Logic:
        # Buy: (SMA50 > SMA200) AND (Dist < 1%)
        # Sell: Cross Down (SMA50 crosses below SMA200)
        
        # To avoid rapid Buy signals every day the price is near SMA200, we might need to filter.
        # But strictly speaking, each day satisfying this is a buy signal candidate.
        # In backtesting, we only act if we are not in position. 
        
        cond_buy = (df['sma50'] > df['sma200']) & (df['dist_sma200_low'] < 1.0)
        
        # Cross Down
        df['sma50_prev'] = df['sma50'].shift(1)
        df['sma200_prev'] = df['sma200'].shift(1)
        
        cross_down = (df['sma50_prev'] >= df['sma200_prev']) & (df['sma50'] < df['sma200'])
        
        cond_sell = cross_down
        
        df.loc[cond_buy, 'Signal'] = 1
        df.loc[cond_sell, 'Signal'] = -1
        
        return df
