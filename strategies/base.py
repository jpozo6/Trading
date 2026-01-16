
import pandas as pd
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates necessary indicators for the strategy.
        Should return the DataFrame with new columns.
        """
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates Buy (1), Sell (-1), or Neutral (0) signals.
        Returns DataFrame with a 'Signal' column.
        """
        pass
        
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full analysis pipeline.
        """
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = self.calculate_indicators(df.copy())
        df = self.generate_signals(df)
        return df

    def get_latest_signal(self, df: pd.DataFrame) -> dict:
        """
        Returns the latest signal and relevant values.
        """
        if df is None or df.empty:
            return {'signal': 0, 'date': None}
        
        last_row = df.iloc[-1]
        return {
            'signal': int(last_row.get('Signal', 0)),
            'date': last_row.name,
            'details': last_row.to_dict()
        }
