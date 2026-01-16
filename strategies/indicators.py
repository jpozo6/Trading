
import pandas as pd
import numpy as np
import ta

def calculate_wma(series, window):
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_coppock(series, wma_period=10, roc1_period=14, roc2_period=11):
    roc1 = series.pct_change(periods=roc1_period) * 100
    roc2 = series.pct_change(periods=roc2_period) * 100
    coppock = calculate_wma(roc1 + roc2, window=wma_period)
    return coppock

def calculate_rsc_mansfield(series, benchmark_series, window=52):
    """
    Calculates Mansfield Relative Strength.
    """
    # Ensure alignment
    aligned_series, aligned_bench = series.align(benchmark_series, join='inner')
    
    ratio = aligned_series / aligned_bench
    sma_ratio = ratio.rolling(window=window).mean()
    mansfield = ((ratio / sma_ratio) - 1) * 100
    return mansfield

def calculate_slope(series, window=3):
    """
    Simple slope calculation based on change over window.
    """
    return series.diff(window)
