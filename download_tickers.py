import pandas as pd
import yfinance as yf
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna
import requests
import os
import time

# Función actualizada para obtener los 50 principales símbolos del NASDAQ
def get_top_nasdaq_tickers(n=50):
    # Utilizar la lista del NASDAQ-100 como base
    nasdaq_100_url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
    tables = pd.read_html(nasdaq_100_url)
    
    # Buscar la tabla que contiene los componentes
    for table in tables:
        if 'Ticker' in table.columns or 'Ticker symbol' in table.columns or 'Symbol' in table.columns:
            components_table = table
            break
    else:
        raise ValueError("No se pudo encontrar la tabla de componentes del NASDAQ-100.")
    
    # Determinar el nombre correcto de la columna de tickers
    if 'Ticker' in components_table.columns:
        ticker_column = 'Ticker'
    elif 'Ticker symbol' in components_table.columns:
        ticker_column = 'Ticker symbol'
    elif 'Symbol' in components_table.columns:
        ticker_column = 'Symbol'
    else:
        raise ValueError("No se encontró una columna de tickers en la tabla de componentes.")
    
    # Obtener la lista de tickers
    tickers = components_table[ticker_column].tolist()
    
    # Limpiar los tickers (eliminar símbolos adicionales si es necesario)
    tickers = [ticker.split('^')[0].strip() for ticker in tickers]
    
    # Seleccionar los primeros n tickers
    top_tickers = tickers[:n]
    return top_tickers

# Resto del código permanece igual...
def download_and_analyze_ticker(ticker, period='1y', interval='1d'):
    try:
        # Descargar datos
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, threads=False)
        if df.empty:
            print(f"No hay datos para el ticker {ticker}")
            return None

        # Limpiar valores NaN
        df = dropna(df)

        # Asegurar que las columnas estén nombradas correctamente
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"})

        # Calcular indicadores técnicos
        df = add_all_ta_features(
            df, open="open", high="high", low="low", close="close", volume="volume")

        # Guardar en CSV
        df.to_csv(os.path.join(output_dir, f"{ticker}_{interval}_analysis.csv"))
        print(f"Análisis guardado para {ticker}")
        return df
    except Exception as e:
        print(f"Error al procesar {ticker}: {e}")
        return None

def main():
    # Crear directorio de salida
    global output_dir
    output_dir = 'nasdaq_analysis_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Obtener lista de los 50 principales tickers
    tickers = get_top_nasdaq_tickers(n=50)

    # Solicitar al usuario el intervalo de tiempo
    interval = input("Ingrese el intervalo deseado (1d, 1wk, 1mo): ")
    period = input("Ingrese el período a descargar (ejemplo, 1y, 5y, max): ")

    for ticker in tickers:
        print(f"Procesando {ticker}...")
        df = download_and_analyze_ticker(ticker, period=period, interval=interval)
        # Retraso opcional para evitar sobrecarga
        time.sleep(1)  # Esperar 1 segundo entre solicitudes

if __name__ == "__main__":
    main()
