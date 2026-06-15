import os
import pandas as pd
import psycopg2

DB_CONFIG = {
    "host": "13.60.0.106", 
    "port": "5432",
    "dbname": "dalalstreet",
    "user": "username", #Hint -name
    "password": "pass" *hint:pa
}

BROAD_FOLDER = "C:/Users/srisa/OneDrive - Hochschule Luzern/Master's Thesis/Data/Broad market indices"

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS broad_market_symbols (
        id SERIAL PRIMARY KEY,
        symbol TEXT UNIQUE,
        source_file TEXT
    );
""")
conn.commit()
total_inserted = 0

for file in os.listdir(BROAD_FOLDER):
    if file.endswith(".csv"):
        file_path = os.path.join(BROAD_FOLDER, file)
        df = pd.read_csv(file_path)

        # Try to detect column with stock symbols
        symbol_col = next((col for col in df.columns if col.strip().lower() == "symbol"), None)
        if not symbol_col:
            print(f"Skipping file (no 'symbol' column): {file}")
            continue

        for symbol in df[symbol_col].dropna().unique():
            try:
                cur.execute(
                    "INSERT INTO broad_market_symbols (symbol, source_file) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (symbol.strip(), file)
                )
                total_inserted += 1
            except Exception as e:
                print(f" Failed to insert {symbol} from {file}: {e}")

        conn.commit()
        print(f"Uploaded symbols from {file}")

print(f"\nTotal new symbols inserted: {total_inserted}")
cur.close()
conn.close()
