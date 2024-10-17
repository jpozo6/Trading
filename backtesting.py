# backtesting_script.py

import pandas as pd
import numpy as np
import yfinance as yf
import ta
import os
from datetime import datetime, timedelta

def download_data(ticker, start_date, end_date, interval='1d'):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=False, progress=False)
        if df.empty:
            print(f"No hay datos para el ticker {ticker}")
            return None
        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"Error al descargar datos para {ticker}: {e}")
        return None

def calculate_indicators(df, benchmark_df):
    # Calcular indicadores directamente sobre datos diarios
    df = df.copy()
    # SMA
    df['sma50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['sma200'] = ta.trend.SMAIndicator(close=df['Close'], window=200).sma_indicator()
    df['sma30'] = ta.trend.SMAIndicator(close=df['Close'], window=30).sma_indicator()
    # MACD
    macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    # Coppock Curve
    df['coppock'] = calculate_coppock_curve(df['Close'])
    # Calcular CPM52 y sus medias móviles
    df['CPM'] = df['Close'] * df['Volume']
    df['CPM52'] = df['CPM'].rolling(window=252).mean()
    df['sma5_cpm52'] = df['CPM52'].rolling(window=5).mean()
    df['sma20_cpm52'] = df['CPM52'].rolling(window=20).mean()
    # Calcular RSC Mansfield
    df['benchmark_close'] = benchmark_df['Close']
    df['rsc_ratio'] = df['Close'] / df['benchmark_close']
    df['rsc_sma52'] = df['rsc_ratio'].rolling(window=252).mean()
    df['rsc_mansfield'] = (df['rsc_ratio'] / df['rsc_sma52'] - 1) * 100
    # Riesgo stop
    df['stop_risk'] = abs(df['Close'] - df['sma30']) / df['Close']
    # Distancia al máximo de 52 semanas
    df['max_52weeks'] = df['High'].rolling(window=252).max()
    df['distance_to_max'] = abs(df['max_52weeks'] - df['Close']) / df['max_52weeks']
    return df

def calculate_coppock_curve(series, wma_period=10, roc1_period=14, roc2_period=11):
    roc1 = series.pct_change(periods=roc1_period) * 100
    roc2 = series.pct_change(periods=roc2_period) * 100
    coppock = (roc1 + roc2).rolling(window=wma_period).mean()
    return coppock

def backtest_ticker(ticker, start_date, end_date, benchmark_df):
    print(f"Descargando datos para {ticker}")
    df = download_data(ticker, start_date=start_date, end_date=end_date, interval='1d')
    if df is None or len(df) < 252 * 2:
        print(f"No hay suficientes datos para {ticker}")
        return None
    # Unir los datos del benchmark
    df = df.join(benchmark_df['Close'], rsuffix='_benchmark', how='left')
    df.rename(columns={'Close_benchmark': 'benchmark_close'}, inplace=True)
    df.dropna(subset=['benchmark_close'], inplace=True)
    # Calcular indicadores
    df = calculate_indicators(df, benchmark_df)
    # Resetear el índice para usar posiciones numéricas
    df = df.reset_index()
    # Inicializar variables de trading para cada estrategia
    strategies = ['Alfayate', 'MACD', 'Coppock', 'CruceMedias']
    positions = {strategy: 0 for strategy in strategies}
    buy_prices = {strategy: 0.0 for strategy in strategies}
    buy_dates = {strategy: None for strategy in strategies}
    portfolios = {strategy: [] for strategy in strategies}
    for i in range(1, len(df)):
        date = df.loc[i, 'Date']
        row = df.loc[i]
        prev_row = df.loc[i - 1]
        # Verificar si tenemos todos los datos necesarios
        required_columns = ['sma50', 'sma200', 'sma30', 'sma5_cpm52', 'sma20_cpm52', 'rsc_mansfield',
                            'stop_risk', 'distance_to_max', 'macd', 'macd_signal', 'coppock']
        if any(pd.isna(row[col]) for col in required_columns):
            continue
        close_price = row['Close']
        ### Estrategia Alfayate ###
        # Señal de Compra Alfayate
        condition1_buy = row['distance_to_max'] <= 0.02
        condition2_buy = row['rsc_mansfield'] > 0.10
        condition3_buy = row['sma5_cpm52'] >= 10
        condition4_buy = row['stop_risk'] <= 0.09
        signal_alfayate_buy = condition1_buy and condition2_buy and condition3_buy and condition4_buy
        # Señal de Venta Alfayate
        condition1_sell = row['rsc_mansfield'] < -0.90
        condition2_sell = row['sma20_cpm52'] <= -15
        condition3_sell = row['stop_risk'] >= 0.30
        signal_alfayate_sell = condition1_sell and condition2_sell and condition3_sell
        # Lógica de trading Alfayate
        positions, portfolios = trading_logic('Alfayate', positions, portfolios, buy_prices, buy_dates,
                                              signal_alfayate_buy, signal_alfayate_sell, close_price, date, ticker)
        ### Estrategia MACD ###
        # Señal de Compra MACD
        macd_cross_up = (row['macd'] > row['macd_signal']) and (prev_row['macd'] <= prev_row['macd_signal'])
        signal_macd_buy = macd_cross_up
        # Señal de Venta MACD
        macd_cross_down = (row['macd'] < row['macd_signal']) and (prev_row['macd'] >= prev_row['macd_signal'])
        signal_macd_sell = macd_cross_down
        # Lógica de trading MACD
        positions, portfolios = trading_logic('MACD', positions, portfolios, buy_prices, buy_dates,
                                              signal_macd_buy, signal_macd_sell, close_price, date, ticker)
        ### Estrategia Coppock ###
        # Señal de Compra Coppock
        coppock_slope_positive = row['coppock'] > prev_row['coppock']
        coppock_below_zero = row['coppock'] < 0
        signal_coppock_buy = coppock_slope_positive and coppock_below_zero
        # Señal de Venta Coppock
        coppock_slope_negative = row['coppock'] < prev_row['coppock']
        coppock_above_zero = row['coppock'] > 0
        signal_coppock_sell = coppock_slope_negative and coppock_above_zero
        # Lógica de trading Coppock
        positions, portfolios = trading_logic('Coppock', positions, portfolios, buy_prices, buy_dates,
                                              signal_coppock_buy, signal_coppock_sell, close_price, date, ticker)
        ### Estrategia Cruce de Medias ###
        # Señal de Compra Cruce de Medias
        golden_cross = (row['sma50'] > row['sma200']) and (prev_row['sma50'] <= prev_row['sma200'])
        signal_crucemedias_buy = golden_cross
        # Señal de Venta Cruce de Medias
        death_cross = (row['sma50'] < row['sma200']) and (prev_row['sma50'] >= prev_row['sma200'])
        signal_crucemedias_sell = death_cross
        # Lógica de trading Cruce de Medias
        positions, portfolios = trading_logic('CruceMedias', positions, portfolios, buy_prices, buy_dates,
                                              signal_crucemedias_buy, signal_crucemedias_sell, close_price, date, ticker)
    # Cerrar posiciones al final si están abiertas
    for strategy in strategies:
        if positions[strategy] == 1:
            sell_price = df['Close'].iloc[-1]
            sell_date = df['Date'].iloc[-1]
            profit = (sell_price - buy_prices[strategy]) / buy_prices[strategy]
            portfolios[strategy].append({
                'Ticker': ticker,
                'Buy_Date': buy_dates[strategy].date(),
                'Buy_Price': buy_prices[strategy],
                'Sell_Date': sell_date.date(),
                'Sell_Price': sell_price,
                'Profit_%': profit * 100,
                'Strategy': strategy
            })
            print(f"{sell_date.date()}: Vendido {ticker} ({strategy}) a {sell_price:.2f} con ganancia de {profit*100:.2f}% (posición cerrada al final)")
    return portfolios

def trading_logic(strategy, positions, portfolios, buy_prices, buy_dates, signal_buy, signal_sell, close_price, date, ticker):
    if positions[strategy] == 0 and signal_buy:
        # Comprar
        positions[strategy] = 1
        buy_prices[strategy] = close_price
        buy_dates[strategy] = date
        print(f"{date.date()}: Comprado {ticker} ({strategy}) a {close_price:.2f}")
    elif positions[strategy] == 1 and signal_sell:
        # Vender
        positions[strategy] = 0
        sell_price = close_price
        sell_date = date
        profit = (sell_price - buy_prices[strategy]) / buy_prices[strategy]
        portfolios[strategy].append({
            'Ticker': ticker,
            'Buy_Date': buy_dates[strategy].date(),
            'Buy_Price': buy_prices[strategy],
            'Sell_Date': sell_date.date(),
            'Sell_Price': sell_price,
            'Profit_%': profit * 100,
            'Strategy': strategy
        })
        print(f"{date.date()}: Vendido {ticker} ({strategy}) a {sell_price:.2f} con ganancia de {profit*100:.2f}%")
        # Reiniciar variables
        buy_prices[strategy] = 0.0
        buy_dates[strategy] = None
    return positions, portfolios

if __name__ == "__main__":
    # Lista de los 10 tickers más importantes del NASDAQ
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'PYPL', 'INTC', 'CSCO']
    # Definir las fechas para el backtesting
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # Últimos 5 años
    # Descargar datos del índice de referencia (por ejemplo, el Nasdaq Composite)
    benchmark_ticker = '^IXIC'  # Índice Nasdaq Composite
    print("Descargando datos del índice de referencia")
    benchmark_df = download_data(benchmark_ticker, start_date=start_date, end_date=end_date, interval='1d')
    if benchmark_df is None:
        print("No se pudo descargar los datos del índice de referencia.")
        exit()
    # Asegurarnos de que el índice sea datetime
    benchmark_df.index = pd.to_datetime(benchmark_df.index)
    total_portfolios = {'Alfayate': [], 'MACD': [], 'Coppock': [], 'CruceMedias': []}
    for ticker in tickers:
        print(f"\nBacktesting para {ticker}")
        portfolios = backtest_ticker(ticker, start_date, end_date, benchmark_df)
        if portfolios:
            for strategy in portfolios:
                total_portfolios[strategy].extend(portfolios[strategy])
    # Analizar y guardar resultados para cada estrategia
    for strategy in total_portfolios:
        results_df = pd.DataFrame(total_portfolios[strategy])
        if not results_df.empty:
            # Calcular métricas de rendimiento
            total_trades = len(results_df)
            profitable_trades = results_df[results_df['Profit_%'] > 0]
            num_profitable = len(profitable_trades)
            average_profit = results_df['Profit_%'].mean()
            print(f"\nResultados para la estrategia {strategy}:")
            print(f"Total de operaciones: {total_trades}")
            print(f"Operaciones ganadoras: {num_profitable}")
            print(f"Porcentaje de operaciones ganadoras: {num_profitable/total_trades*100:.2f}%")
            print(f"Ganancia promedio por operación: {average_profit:.2f}%")
            # Guardar resultados
            today_str = datetime.now().strftime('%Y%m%d')
            output_filename = f'backtesting/resultados_backtesting_{strategy}_{today_str}.csv'
            results_df.to_csv(output_filename, index=False)
            print(f"Resultados guardados en: {output_filename}")
        else:
            print(f"\nNo se obtuvieron resultados para la estrategia {strategy}.")
