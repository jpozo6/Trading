import pandas as pd
import numpy as np
import yfinance as yf
import ta
import os
import matplotlib.pyplot as plt
import requests
from io import StringIO
from datetime import datetime

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

def download_data(ticker, period='5y', interval='1d'):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
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

"""
def calculate_rsc_mansfield_simple(df, benchmark_df):
    
    #Calcula la fuerza RSC Mansfield utilizando el cálculo simplificado.
    
    # Asegurarse de que los índices están en formato de fecha
    df.index = pd.to_datetime(df.index)
    benchmark_df.index = pd.to_datetime(benchmark_df.index)

    # Alinear las fechas entre ambos DataFrames
    df['benchmark_close'] = benchmark_df['Close']

    # Calcular RSC Mansfield
    df['rsc_ratio'] = df['Close'] / df['benchmark_close']
    df['rsc_sma252'] = df['rsc_ratio'].rolling(window=252).mean()
    df['rsc_mansfield'] = (df['rsc_ratio'] / df['rsc_sma252'] - 1) * 100

    return df
    """

def analyze_tickers(tickers):
    # Descargar datos del índice de referencia (por ejemplo, el Nasdaq Composite)
    benchmark_ticker = '^IXIC'  # Índice Nasdaq Composite
    benchmark_df = download_data(benchmark_ticker, period='5y', interval='1d')
    if benchmark_df is None or benchmark_df.empty:
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
            df_daily = download_data(ticker, period='1y', interval='1d')
            if df_daily is None or len(df_daily) < 200:
                continue  # Necesitamos al menos 200 días de datos para las SMA

            # Descargar datos semanales
            df_weekly = download_data(ticker, period='5y', interval='1d')
            if df_weekly is None or len(df_weekly) < 100:
                continue  # Necesitamos suficientes datos para calcular indicadores semanales
            df_weekly = calculate_indicators(df_weekly, indicators=['macd', 'sma5', 'sma20', 'sma30'], time_frame='weekly')
            if df_weekly.empty:
                continue

            # Descargar datos mensuales
            df_monthly = download_data(ticker, period='10y', interval='1d')
            if df_monthly is None or len(df_monthly) < 120:
                continue  # Necesitamos suficientes datos para calcular la Curva de Coppock
            df_monthly = calculate_indicators(df_monthly, indicators=['coppock'], time_frame='monthly')
            if df_monthly.empty:
                continue

            # Calcular SMA50 y SMA200 en datos diarios
            df_daily = calculate_indicators(df_daily, indicators=['sma50', 'sma200'], time_frame='daily')

            # Verificar que las SMA han sido calculadas
            if df_daily['sma50'].isnull().all() or df_daily['sma200'].isnull().all():
                continue

            # Señales de Compra y Venta

            ## Señales de Coppock
            # Señal de Compra
            coppock_current = df_monthly['coppock'].iloc[-1]
            coppock_prev = df_monthly['coppock'].iloc[-2]
            coppock_slope_positive = (coppock_current - coppock_prev) > 0
            coppock_below_zero = coppock_current < 0
            signal_coppock_buy = coppock_slope_positive and coppock_below_zero

            # Señal de Venta
            coppock_slope_negative = (coppock_current - coppock_prev) < 0
            signal_coppock_sell = coppock_slope_negative and coppock_current < 0

            ## Señales de MACD
            # Señal de Compra
            macd_signal_current = df_weekly['macd_signal'].iloc[-1]
            macd_signal_prev = df_weekly['macd_signal'].iloc[-2]
            macd_crosses_zero_up = (macd_signal_prev < 0) and (macd_signal_current > 0)
            macd_signal_positive = macd_signal_current > 0
            signal_macd_buy = macd_crosses_zero_up or macd_signal_positive

            # Señal de Venta
            macd_crosses_zero_down = (macd_signal_prev > 0) and (macd_signal_current < 0)
            macd_signal_negative = macd_signal_current < 0
            signal_macd_sell = macd_crosses_zero_down or macd_signal_negative

            ## Señales de Cruce de Medias
            # Señal de Compra
            sma50_current = df_daily['sma50'].iloc[-1]
            sma200_current = df_daily['sma200'].iloc[-1]
            golden_cross = sma50_current > sma200_current
            min_price = df_daily['Low'].iloc[-1]
            sma200_distance = abs(sma200_current - min_price) / min_price
            sma200_close_to_min = sma200_distance < 0.01  # Menor al 1%
            signal_sma_buy = golden_cross and sma200_close_to_min

            # Señal de Venta
            death_cross = sma50_current < sma200_current
            signal_sma_sell = death_cross

            ## Señales de Alfayate_Weinstein
            # Señal de Compra
            # Condición 1: Distancia del precio de cierre al máximo de 52 semanas <= 2%
            max_52weeks = df_daily['High'].rolling(window=252).max().iloc[-1]
            close_price = df_daily['Close'].iloc[-1]
            # Calcular máximo de 52 semanas, verificando que haya suficientes datos
            if len(df_daily) >= 252:
                max_52weeks = df_daily['High'].rolling(window=252).max().iloc[-1]
                close_price = df_daily['Close'].iloc[-1]
                distance_to_max = abs(max_52weeks - close_price) / max_52weeks
            else:
                distance_to_max = np.nan  # Valor NaN si no hay suficientes datos
            condition1_buy = distance_to_max <= 0.02
            """
            # Condición 2: RSC Mansfield de 52 semanas > 0.10
            # Calcular RSC Mansfield para el activo
            print(f"Datos semanales del ticker {ticker}: {len(df_weekly)} filas.")
            print(f"Datos semanales del benchmark: {len(benchmark_df)} filas.")
            if len(df_weekly) >= 52 and len(benchmark_df) >= 52:  # Asegurar datos suficientes
                rsc_mansfield = calculate_rsc_mansfield(df_weekly, benchmark_df)
                rsc_current = rsc_mansfield.iloc[-1] if not rsc_mansfield.isnull().all() else np.nan
            else:
                rsc_mansfield = pd.Series(index=df_weekly.index, data=np.nan)
                rsc_current = np.nan  # Valor NaN si no hay suficientes datos
            condition2_buy = rsc_current > 0.10
            """
            # Condición 3: SMA5 del CPM52 >= 10
            df_weekly['CPM'] = df_weekly['Volume'] * df_weekly['Close']
            cpm52 = df_weekly['CPM'].rolling(window=52).mean()
            sma5_cpm52 = cpm52.rolling(window=5).mean()
            cpm52_current = sma5_cpm52.iloc[-1]
            condition3_buy = cpm52_current >= 10  # Ajusta el umbral según sea necesario

            # Condición 4: Riesgo stop <= 9%
            sma30_current = df_weekly['sma30'].iloc[-1]
            stop_risk = abs(close_price - sma30_current) / close_price
            condition4_buy = stop_risk <= 0.09

            # Señal de Compra Alfayate_Weinstein
            signal_alfayate_buy = condition1_buy and condition3_buy and condition4_buy

            # Señal de Venta
            """
            # Condición 1: RSC Mansfield de 52 semanas < -0.90
            condition1_sell = rsc_current < -0.90
            """
            # Condición 2: SMA20 del CPM52 <= -15
            sma20_cpm52 = cpm52.rolling(window=20).mean()
            cpm52_current_sma20 = sma20_cpm52.iloc[-1]
            condition2_sell = cpm52_current_sma20 <= -15  # Ajusta el umbral según sea necesario

            # Condición 3: Riesgo stop >= 30%
            condition3_sell = stop_risk >= 0.30

            # Señal de Venta Alfayate_Weinstein
            signal_alfayate_sell = condition2_sell and condition3_sell

            # Añadir los resultados para este ticker
            results.append({
                'Ticker': ticker,
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
                #'RSC_Mansfield': rsc_current,
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
    today = datetime.now().strftime('%Y%m%d')
    output_dir = 'analysis_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, f'resultados_{today}.csv')
    results_df.to_csv(output_filename)
    print(f"\nResultados guardados en: {output_filename}")

    return results_df

if __name__ == '__main__':
    # Obtener la lista de tickers del NASDAQ
    #tickers = get_nasdaq_tickers()

    # Debido a que el número de tickers es grande, puedes limitarlo para pruebas
    #tickers = tickers[:100]  # Analizar solo los primeros 100 tickers

    # top 100 tickers nasdaq
    tickers = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "COST",
    "NFLX", "ASML", "TMUS", "CSCO", "PEP", "AMD", "LIN", "ADBE", "ISRG", "INTU",
    "QCOM", "TXN", "HON", "AMGN", "SBUX", "INTC", "MDLZ", "AMAT", "PYPL", "ADI",
    "LRCX", "BKNG", "MRVL", "GILD", "FISV", "PANW", "ADP", "MU", "CSX", "VRTX",
    "REGN", "KLAC", "SNPS", "MELI", "NXPI", "LULU", "FTNT", "ATVI", "IDXX", "ROST",
    "MNST", "CTAS", "ORLY", "KDP", "MAR", "TEAM", "CDNS", "EXC", "DXCM", "ODFL",
    "PCAR", "PAYX", "XEL", "FAST", "WBA", "VRSK", "BIIB", "SGEN", "CHTR", "AEP",
    "SIRI", "ILMN", "DLTR", "CPRT", "EA", "WDAY", "PDD", "DDOG", "ZM", "DOCU",
    "CRWD", "ZS", "LCID", "BKR", "CEG", "GFS", "MRNA", "APP", "DASH", "ARM", "GEHC",
    "SMCI", "WBD", "AZN", "KHC", "CSGP", "MCHP", "ON", "TTWO", "MDB", "ANSS"
]

    # Ejecutar el análisis
    results_df = analyze_tickers(tickers)
