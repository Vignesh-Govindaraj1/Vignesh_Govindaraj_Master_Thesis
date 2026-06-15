
import os
import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv
import random

load_dotenv()

# Load DB credentials
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASS
)

def read_all_symbols(folder_path):
    all_symbols = set()
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            symbol_col = next((c for c in df.columns if c.strip().lower() == 'symbol'), None)
            if symbol_col:
                all_symbols.update(df[symbol_col].dropna().unique())
    return list(all_symbols)

def get_1min_data(symbol):
    query = """
        SELECT date, open, high, low, close, volume
        FROM stock_history_1min
        WHERE stock_symbol = %s
        ORDER BY date ASC
    """
    try:
        df = pd.read_sql(query, conn, params=(symbol,))
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def detect_support_resistance(df, window=5):
    supports = []
    resistances = []
    for i in range(window, len(df) - window):
        low = df['low'].iloc[i]
        high = df['high'].iloc[i]
        if all(df['low'].iloc[i - j] > low for j in range(1, window + 1)) and \
           all(df['low'].iloc[i + j] > low for j in range(1, window + 1)):
            supports.append((df['date'].iloc[i], low))
        if all(df['high'].iloc[i - j] < high for j in range(1, window + 1)) and \
           all(df['high'].iloc[i + j] < high for j in range(1, window + 1)):
            resistances.append((df['date'].iloc[i], high))
    return supports, resistances

def main():
    symbols = read_all_symbols("stock_name_data")
    print(f"Found {len(symbols)} symbols.")
    selected = random.sample(symbols, min(300, len(symbols)))
    for symbol in selected:
        df = get_1min_data(symbol)
        if df.empty or len(df) < 100:
            continue
        supports, resistances = detect_support_resistance(df)
        if supports or resistances:
            print(f"\n Symbol: {symbol}")
            print(f" Supports: {[f'{s[1]:.2f}' for s in supports[-3:]]}")
            print(f" Resistances: {[f'{r[1]:.2f}' for r in resistances[-3:]]}")

if __name__ == "__main__":
    main()