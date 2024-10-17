# analysis_script.py

import pandas as pd
import numpy as np
import yfinance as yf
import ta
import os
import requests
from io import StringIO
from datetime import datetime, timedelta

def get_nasdaq_tickers():
    url = 'http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt'
    response = requests.get(url)
    data = response.content.decode('utf-8')
    df = pd.read_csv(StringIO(data), sep='|')
    # Eliminar la última fila que contiene 'File Creation Time'
    df = df[:-1]
    # Asegurarse de que la columna 'Symbol' es de tipo cadena y eliminar valores nulos
    df['Symbol'] = df['Symbol'].astype(str)
    df = df[df['Symbol'].notnull()]
    tickers = df['Symbol'].tolist()
    # Filtrar los símbolos correctamente
    tickers = [
        ticker for ticker in tickers
        if isinstance(ticker, str) and 'test' not in ticker.lower() and ticker.isalpha()
    ]
    return tickers

def download_data(ticker, start_date, end_date, interval='1d'):
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        print(f"No hay datos para el ticker {ticker}")
        return None
    df.dropna(inplace=True)
    return df

def calculate_indicators(df, indicators, time_frame):
    df = df.copy()
    if time_frame == 'monthly':
        df = df.resample('M').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Adj Close': 'last',
            'Volume': 'sum'
        })
    elif time_frame == 'weekly':
        df = df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Adj Close': 'last',
            'Volume': 'sum'
        })
    df.dropna(inplace=True)
    # Calcular los indicadores seleccionados
    if 'coppock' in indicators:
        df['coppock'] = calculate_coppock_curve(df['Close'], wma_period=10, roc1_period=12, roc2_period=6)
    if 'macd' in indicators:
        macd = ta.trend.MACD(close=df['Close'], window_slow=23, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
    if 'sma5' in indicators:
        df['sma5'] = ta.trend.SMAIndicator(close=df['Close'], window=5).sma_indicator()
    if 'sma20' in indicators:
        df['sma20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
    if 'sma30' in indicators:
        df['sma30'] = ta.trend.SMAIndicator(close=df['Close'], window=30).sma_indicator()
    if 'sma50' in indicators:
        df['sma50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
    if 'sma200' in indicators:
        df['sma200'] = ta.trend.SMAIndicator(close=df['Close'], window=200).sma_indicator()
    return df

def calculate_coppock_curve(series, wma_period=10, roc1_period=12, roc2_period=6):
    roc1 = series.pct_change(periods=roc1_period) * 100
    roc2 = series.pct_change(periods=roc2_period) * 100
    coppock = (roc1 + roc2).rolling(window=wma_period).mean()
    return coppock

def calculate_rsc_mansfield(df, benchmark_df):
    """
    Calcula la fuerza RSC Mansfield entre el activo y el índice de referencia.
    """
    df = df.copy()
    benchmark_df = benchmark_df.copy()
    df['Close'] = df['Close'].fillna(method='ffill')
    benchmark_df['Close'] = benchmark_df['Close'].fillna(method='ffill')
    ratio = df['Close'] / benchmark_df['Close']
    sma_ratio = ratio.rolling(window=52).mean()
    mansfield = (ratio / sma_ratio - 1) * 100
    return mansfield

def analyze_tickers(tickers, analysis_date=None):
    if analysis_date is None:
        analysis_date = datetime.now()
    else:
        analysis_date = datetime.strptime(analysis_date, '%Y-%m-%d')

    # Definir las fechas para obtener los datos necesarios
    end_date = analysis_date + timedelta(days=1)  # Agregar un día para incluir la fecha de análisis
    start_date = analysis_date - timedelta(days=365*5)  # Últimos 5 años

    # Descargar datos del índice de referencia (por ejemplo, el Nasdaq Composite)
    benchmark_ticker = '^IXIC'  # Índice Nasdaq Composite
    benchmark_df = download_data(benchmark_ticker, start_date=start_date, end_date=end_date, interval='1d')
    if benchmark_df is None:
        print("No se pudo descargar los datos del índice de referencia.")
        return

    results = []

    total_tickers = len(tickers)
    processed_tickers = 0

    for ticker in tickers:
        processed_tickers += 1
        print(f"Procesando ticker ({processed_tickers}/{total_tickers}): {ticker}")
        try:
            # Descargar datos diarios
            df_daily = download_data(ticker, start_date=start_date, end_date=end_date, interval='1d')
            if df_daily is None or len(df_daily) < 200:
                continue  # Necesitamos al menos 200 días de datos para las SMA

            # Calcular indicadores diarios
            df_daily = calculate_indicators(df_daily, indicators=['sma50', 'sma200'], time_frame='daily')
            if df_daily.empty:
                continue

            # Calcular indicadores semanales
            df_weekly = calculate_indicators(df_daily, indicators=['macd', 'sma5', 'sma20', 'sma30'], time_frame='weekly')
            if df_weekly.empty:
                continue

            # Calcular indicadores mensuales
            df_monthly = calculate_indicators(df_daily, indicators=['coppock'], time_frame='monthly')
            if df_monthly.empty:
                continue

            # Verificar que las SMA han sido calculadas
            if df_daily['sma50'].isnull().all() or df_daily['sma200'].isnull().all():
                continue

            analysis_date_str = analysis_date.strftime('%Y-%m-%d')
            # Asegurar que la fecha de análisis está en los DataFrames
            if analysis_date_str not in df_daily.index or analysis_date_str not in df_weekly.index or analysis_date_str not in df_monthly.index:
                continue

            # Señales de Compra y Venta

            ## Señales de Coppock
            # Señal de Compra
            coppock_current = df_monthly.loc[analysis_date_str, 'coppock']
            coppock_prev = df_monthly.shift(1).loc[analysis_date_str, 'coppock']
            if pd.isna(coppock_current) or pd.isna(coppock_prev):
                continue
            coppock_slope_positive = (coppock_current - coppock_prev) > 0
            coppock_below_zero = coppock_current < 0
            signal_coppock_buy = coppock_slope_positive and coppock_below_zero

            # Señal de Venta
            coppock_slope_negative = (coppock_current - coppock_prev) < 0
            signal_coppock_sell = coppock_slope_negative and coppock_current < 0

            ## Señales de MACD
            macd_signal_current = df_weekly.loc[analysis_date_str, 'macd_signal']
            macd_signal_prev = df_weekly.shift(1).loc[analysis_date_str, 'macd_signal']
            if pd.isna(macd_signal_current) or pd.isna(macd_signal_prev):
                continue
            # Señal de Compra
            macd_crosses_zero_up = (macd_signal_prev < 0) and (macd_signal_current > 0)
            macd_signal_positive = macd_signal_current > 0
            signal_macd_buy = macd_crosses_zero_up or macd_signal_positive

            # Señal de Venta
            macd_crosses_zero_down = (macd_signal_prev > 0) and (macd_signal_current < 0)
            macd_signal_negative = macd_signal_current < 0
            signal_macd_sell = macd_crosses_zero_down or macd_signal_negative

            ## Señales de Cruce de Medias
            sma50_current = df_daily.loc[analysis_date_str, 'sma50']
            sma200_current = df_daily.loc[analysis_date_str, 'sma200']
            if pd.isna(sma50_current) or pd.isna(sma200_current):
                continue
            # Señal de Compra
            golden_cross = sma50_current > sma200_current
            min_price = df_daily.loc[analysis_date_str, 'Low']
            sma200_distance = abs(sma200_current - min_price) / min_price
            sma200_close_to_min = sma200_distance < 0.01  # Menor al 1%
            signal_sma_buy = golden_cross and sma200_close_to_min

            # Señal de Venta
            death_cross = sma50_current < sma200_current
            signal_sma_sell = death_cross

            ## Señales de Alfayate_Weinstein
            # Señal de Compra
            # Condición 1: Distancia del precio de cierre al máximo de 52 semanas <= 2%
            max_52weeks = df_daily['High'].rolling(window=252).max().loc[analysis_date_str]
            close_price = df_daily.loc[analysis_date_str, 'Close']
            distance_to_max = abs(max_52weeks - close_price) / max_52weeks
            condition1_buy = distance_to_max <= 0.02

            # Condición 2: RSC Mansfield de 52 semanas > 0.10
            # Calcular RSC Mansfield para el activo
            df_weekly_for_rsc = df_weekly.copy()
            benchmark_weekly = benchmark_df.copy()
            benchmark_weekly = calculate_indicators(benchmark_weekly, indicators=[], time_frame='weekly')
            rsc_mansfield = calculate_rsc_mansfield(df_weekly_for_rsc, benchmark_weekly)
            rsc_current = rsc_mansfield.loc[analysis_date_str]
            condition2_buy = rsc_current > 0.10

            # Condición 3: SMA5 del CPM52 >= 10
            df_weekly['CPM'] = df_weekly['Volume'] * df_weekly['Close']
            cpm52 = df_weekly['CPM'].rolling(window=52).mean()
            sma5_cpm52 = cpm52.rolling(window=5).mean()
            cpm52_current = sma5_cpm52.loc[analysis_date_str]
            condition3_buy = cpm52_current >= 10  # Ajusta el umbral según sea necesario

            # Condición 4: Riesgo stop <= 9%
            sma30_current = df_weekly.loc[analysis_date_str, 'sma30']
            stop_risk = abs(close_price - sma30_current) / close_price
            condition4_buy = stop_risk <= 0.09

            signal_alfayate_buy = condition1_buy and condition2_buy and condition3_buy and condition4_buy

            # Señal de Venta
            # Condición 1: RSC Mansfield de 52 semanas < -0.90
            condition1_sell = rsc_current < -0.90

            # Condición 2: SMA20 del CPM52 <= -15
            sma20_cpm52 = cpm52.rolling(window=20).mean()
            cpm52_current_sma20 = sma20_cpm52.loc[analysis_date_str]
            condition2_sell = cpm52_current_sma20 <= -15  # Ajusta el umbral según sea necesario

            # Condición 3: Riesgo stop >= 30%
            condition3_sell = stop_risk >= 0.30

            signal_alfayate_sell = condition1_sell and condition2_sell and condition3_sell

            # Añadir los resultados para este ticker
            results.append({
                'Ticker': ticker,
                'Date': analysis_date_str,
                # Señales de Compra
                'Signal_Coppock_Buy': signal_coppock_buy,
                'Signal_MACD_Buy': signal_macd_buy,
                'Signal_SMA_Buy': signal_sma_buy,
                'Signal_Alfayate_Buy': signal_alfayate_buy,
                # Señales de Venta
                'Signal_Coppock_Sell': signal_coppock_sell,
                'Signal_MACD_Sell': signal_macd_sell,
                'Signal_SMA_Sell': signal_sma_sell,
                'Signal_Alfayate_Sell': signal_alfayate_sell,
                # Indicadores y valores calculados
                'Coppock': coppock_current,
                'MACD_Signal': macd_signal_current,
                'SMA50': sma50_current,
                'SMA200': sma200_current,
                'SMA200_Dist_Min': sma200_distance,
                'Distance_to_Max': distance_to_max,
                'RSC_Mansfield': rsc_current,
                'SMA5_CPM52': cpm52_current,
                'SMA20_CPM52': cpm52_current_sma20,
                'Stop_Risk': stop_risk
            })

        except Exception as e:
            print(f"Error al procesar {ticker}: {e}")
            continue

    # Convertir la lista de resultados en DataFrame
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.set_index('Ticker')
        print("\nResultados del análisis:")
        print(results_df)
    else:
        print("\nNo se pudieron obtener resultados del análisis.")

    # Guardar el DataFrame en un archivo CSV con el formato especificado
    date_str = analysis_date.strftime('%Y%m%d')
    output_filename = f'resultados_{date_str}.csv'
    results_df.to_csv(output_filename)
    print(f"\nResultados guardados en: {output_filename}")

    return results_df

if __name__ == "__main__":
    # Ejecutar el análisis
    tickers = get_nasdaq_tickers()
    # Para pruebas, puedes limitar el número de tickers
    # tickers = tickers[:50]

    # Puedes especificar la fecha de análisis en formato 'YYYY-MM-DD'
    # Si no se especifica, se utilizará la fecha de hoy
    import argparse

    parser = argparse.ArgumentParser(description='Análisis de señales de compra y venta para los tickers del NASDAQ.')
    parser.add_argument('--date', type=str, help='Fecha para el análisis en formato YYYY-MM-DD (por defecto, hoy).')

    args = parser.parse_args()
    analysis_date = args.date

    results_df = analyze_tickers(tickers, analysis_date)
