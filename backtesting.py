# backtesting_script.py

import pandas as pd
import numpy as np
import yfinance as yf
import ta
import sys
from datetime import datetime, timedelta

def download_data(ticker, start_date, end_date, interval='1d'):
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            print(f"No hay datos para el ticker {ticker}")
            return None
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error al descargar datos para {ticker}: {e}")
        return None

def get_tickers(index_name):
    if index_name.lower() == 'sp500':
        # Obtener la lista de tickers del S&P500
        sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df_sp500 = sp500_table[0]
        tickers = df_sp500['Symbol'].tolist()
        # Corregir posibles símbolos
        tickers = [ticker.replace('.', '-') for ticker in tickers]
    elif index_name.lower() == 'nasdaq':
        # Lista de los 10 tickers más importantes del NASDAQ
        tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'PYPL', 'INTC', 'CSCO']
    else:
        print("Índice no reconocido. Por favor, elige 'sp500' o 'nasdaq'.")
        sys.exit()
    return tickers

def calculate_indicators(df_daily, benchmark_daily):
    # Resampleo y cálculo de indicadores en diferentes temporalidades
    df_weekly = df_daily.resample('W-FRI').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    })
    df_monthly = df_daily.resample('M').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    })

    # Calcular indicadores en temporalidad diaria (Alfayate y Cruce de Medias)
    # Alfayate Indicators
    df_daily['sma50'] = ta.trend.SMAIndicator(df_daily['Close'], window=50).sma_indicator()
    df_daily['sma200'] = ta.trend.SMAIndicator(df_daily['Close'], window=200).sma_indicator()
    df_daily['sma30'] = ta.trend.SMAIndicator(df_daily['Close'], window=30).sma_indicator()
    df_daily['CPM'] = df_daily['Close'] * df_daily['Volume']
    df_daily['CPM252'] = df_daily['CPM'].rolling(window=252).mean()
    df_daily['sma5_cpm252'] = df_daily['CPM252'].rolling(window=5).mean()
    df_daily['sma20_cpm252'] = df_daily['CPM252'].rolling(window=20).mean()
    df_daily['max_52weeks'] = df_daily['High'].rolling(window=252).max()
    df_daily['distance_to_max'] = abs(df_daily['max_52weeks'] - df_daily['Close']) / df_daily['max_52weeks']
    df_daily['stop_risk'] = abs(df_daily['Close'] - df_daily['sma30']) / df_daily['Close']

    # RSC Mansfield (Daily)
    df_daily['benchmark_close'] = benchmark_daily['Close']
    df_daily['rsc_ratio'] = df_daily['Close'] / df_daily['benchmark_close']
    df_daily['rsc_sma252'] = df_daily['rsc_ratio'].rolling(window=252).mean()
    df_daily['rsc_mansfield'] = (df_daily['rsc_ratio'] / df_daily['rsc_sma252'] - 1) * 100

    # MACD semanal
    macd_weekly = ta.trend.MACD(df_weekly['Close'])
    df_weekly['macd'] = macd_weekly.macd()
    df_weekly['macd_signal'] = macd_weekly.macd_signal()

    # Coppock Curve mensual
    df_monthly['coppock'] = calculate_coppock_curve(df_monthly['Close'])

    # Unir los DataFrames
    df_daily = df_daily.reset_index().rename(columns={'index': 'Date'})
    df_weekly = df_weekly.reset_index().rename(columns={'index': 'Date'})
    df_monthly = df_monthly.reset_index().rename(columns={'index': 'Date'})

    # Merge MACD (Weekly) into daily DataFrame
    df = df_daily.merge(df_weekly[['Date', 'macd', 'macd_signal']], on='Date', how='left')
    # Merge Coppock (Monthly) into daily DataFrame
    df = df.merge(df_monthly[['Date', 'coppock']], on='Date', how='left')

    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    # Rellenar valores NaN hacia adelante para indicadores semanales y mensuales
    df[['macd', 'macd_signal', 'coppock']] = df[['macd', 'macd_signal', 'coppock']].fillna(method='ffill')

    return df

def calculate_coppock_curve(series, wma_period=10, roc1_period=14, roc2_period=11):
    roc1 = series.pct_change(periods=roc1_period)
    roc2 = series.pct_change(periods=roc2_period)
    coppock = (roc1 + roc2).rolling(window=wma_period).mean()
    return coppock

def backtest_ticker(ticker, start_date, end_date, benchmark_df):
    df_daily = download_data(ticker, start_date=start_date, end_date=end_date)
    if df_daily is None or len(df_daily) < 252 * 2:
        print(f"No hay suficientes datos para {ticker}")
        return None

    # Descargar datos diarios del benchmark
    benchmark_daily = benchmark_df.reindex(df_daily.index).fillna(method='ffill')

    # Calcular indicadores
    df = calculate_indicators(df_daily, benchmark_daily)
    df.reset_index(inplace=True)

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
        close_price = row['Close']

        ### Estrategia Alfayate (Diario) ###
        indicators = ['distance_to_max', 'rsc_mansfield', 'sma5_cpm252', 'stop_risk']
        if all(pd.notna(row[ind]) for ind in indicators):
            condition1_buy = row['distance_to_max'] <= 0.02
            condition2_buy = row['rsc_mansfield'] > 0.10
            condition3_buy = row['sma5_cpm252'] >= 10
            condition4_buy = row['stop_risk'] <= 0.09
            signal_alfayate_buy = condition1_buy and condition2_buy and condition3_buy and condition4_buy

            condition1_sell = row['rsc_mansfield'] < -0.90
            condition2_sell = row['sma20_cpm252'] <= -15
            condition3_sell = row['stop_risk'] >= 0.30
            signal_alfayate_sell = condition1_sell and condition2_sell and condition3_sell

            positions, portfolios = trading_logic('Alfayate', positions, portfolios, buy_prices, buy_dates,
                                                  signal_alfayate_buy, signal_alfayate_sell, close_price, date, ticker)

        ### Estrategia MACD (Semanal) ###
        if pd.notna(row['macd']) and pd.notna(prev_row['macd']):
            macd_cross_up = (row['macd'] > row['macd_signal']) and (prev_row['macd'] <= prev_row['macd_signal'])
            signal_macd_buy = macd_cross_up

            macd_cross_down = (row['macd'] < row['macd_signal']) and (prev_row['macd'] >= prev_row['macd_signal'])
            signal_macd_sell = macd_cross_down

            positions, portfolios = trading_logic('MACD', positions, portfolios, buy_prices, buy_dates,
                                                  signal_macd_buy, signal_macd_sell, close_price, date, ticker)

        ### Estrategia Coppock (Mensual) ###
        if pd.notna(row['coppock']) and pd.notna(prev_row['coppock']):
            coppock_slope_positive = row['coppock'] > prev_row['coppock']
            coppock_below_zero = row['coppock'] < 0
            signal_coppock_buy = coppock_slope_positive and coppock_below_zero

            coppock_slope_negative = row['coppock'] < prev_row['coppock']
            coppock_above_zero = row['coppock'] > 0
            signal_coppock_sell = coppock_slope_negative and coppock_above_zero

            positions, portfolios = trading_logic('Coppock', positions, portfolios, buy_prices, buy_dates,
                                                  signal_coppock_buy, signal_coppock_sell, close_price, date, ticker)

        ### Estrategia Cruce de Medias (Diario) ###
        if pd.notna(row['sma50']) and pd.notna(prev_row['sma50']):
            golden_cross = (row['sma50'] > row['sma200']) and (prev_row['sma50'] <= prev_row['sma200'])
            signal_crucemedias_buy = golden_cross

            death_cross = (row['sma50'] < row['sma200']) and (prev_row['sma50'] >= prev_row['sma200'])
            signal_crucemedias_sell = death_cross

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
        positions[strategy] = 1
        buy_prices[strategy] = close_price
        buy_dates[strategy] = date
        print(f"{date.date()}: Comprado {ticker} ({strategy}) a {close_price:.2f}")
    elif positions[strategy] == 1 and signal_sell:
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
        buy_prices[strategy] = 0.0
        buy_dates[strategy] = None
    return positions, portfolios

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python backtesting_script.py [sp500|nasdaq]")
        sys.exit()
    index_name = sys.argv[1]
    tickers = get_tickers(index_name)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)
    benchmark_ticker = '^GSPC' if index_name.lower() == 'sp500' else '^IXIC'
    print("Descargando datos del índice de referencia")
    benchmark_df = download_data(benchmark_ticker, start_date=start_date, end_date=end_date)
    if benchmark_df is None:
        print("No se pudo descargar los datos del índice de referencia.")
        exit()
    benchmark_df.index = pd.to_datetime(benchmark_df.index)
    total_portfolios = {'Alfayate': [], 'MACD': [], 'Coppock': [], 'CruceMedias': []}

    for ticker in tickers:
        print(f"\nBacktesting para {ticker}")
        portfolios = backtest_ticker(ticker, start_date, end_date, benchmark_df)
        if portfolios:
            for strategy in portfolios:
                total_portfolios[strategy].extend(portfolios[strategy])
    for strategy in total_portfolios:
        results_df = pd.DataFrame(total_portfolios[strategy])
        if not results_df.empty:
            total_trades = len(results_df)
            profitable_trades = results_df[results_df['Profit_%'] > 0]
            num_profitable = len(profitable_trades)
            average_profit = results_df['Profit_%'].mean()
            print(f"\nResultados para la estrategia {strategy}:")
            print(f"Total de operaciones: {total_trades}")
            print(f"Operaciones ganadoras: {num_profitable}")
            print(f"Porcentaje de operaciones ganadoras: {num_profitable/total_trades*100:.2f}%")
            print(f"Ganancia promedio por operación: {average_profit:.2f}%")
            today_str = datetime.now().strftime('%Y%m%d')
            output_filename = f'backtesting/resultados_backtesting_{strategy}_{index_name}_{today_str}.csv'
            results_df.to_csv(output_filename, index=False)
            print(f"Resultados guardados en: {output_filename}")
        else:
            print(f"\nNo se obtuvieron resultados para la estrategia {strategy}.")
