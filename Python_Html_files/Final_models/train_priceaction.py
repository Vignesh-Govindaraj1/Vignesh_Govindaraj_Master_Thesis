import os
from dotenv import load_dotenv
import urllib.parse
import pandas as pd
import numpy as np
import traceback
from sqlalchemy import create_engine, text
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import pickle
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

# Database connection
DB_USER = os.getenv("DB_USER")
DB_PASS = urllib.parse.quote_plus(os.getenv("DB_PASS"))
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

def fetch_random_symbols(n=300):
    query = """
        SELECT stock_symbol FROM (
            SELECT stock_symbol
            FROM stock_history_1min
            GROUP BY stock_symbol
            ORDER BY RANDOM()
            LIMIT :n
        ) t
    """
    df = pd.read_sql(text(query), engine, params={'n': n})
    return df['stock_symbol'].tolist()

def fetch_stock_data(symbol):
    query = """
        SELECT date, open, high, low, close, volume
        FROM stock_history_1min
        WHERE stock_symbol = :symbol
        AND date >= '2015-01-01'
        ORDER BY date ASC
    """
    df = pd.read_sql(text(query), engine, params={'symbol': symbol})
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def add_features(df):
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(window=10).std()
    df['rsi'] = compute_rsi(df['close'])
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    df['ma_100'] = df['close'].rolling(window=100).mean()
    df['bollinger_upper'] = df['ma_20'] + 2 * df['close'].rolling(window=20).std()
    df['bollinger_lower'] = df['ma_20'] - 2 * df['close'].rolling(window=20).std()
    df['atr_14'] = compute_atr(df)
    df['log_volume'] = np.log1p(df['volume'])
    df['price_range'] = df['high'] - df['low']
    for lag in range(1, 4):
        df[f'close_lag{lag}'] = df['close'].shift(lag)
        df[f'vol_lag{lag}'] = df['volume'].shift(lag)
    return df.dropna()

# ---- UPDATED label_data ----
def label_data(df, threshold=0.02, forward=3):
    df['target'] = 0
    df['target'] = np.where(df['close'].shift(-forward) > df['close'] * (1 + threshold), 1, df['target'])
    df['target'] = np.where(df['close'].shift(-forward) < df['close'] * (1 - threshold), -1, df['target'])
    df['target'] = df['target'].astype(int)
    # Map -1→0 (Sell), 0→1 (Hold), 1→2 (Buy) for XGBoost
    df['target'] = df['target'].map({-1: 0, 0: 1, 1: 2})
    return df

def prepare_data(symbols, batch_size=10):
    all_data = []
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(symbols)+batch_size-1)//batch_size}...")
        for symbol in batch:
            try:
                df = fetch_stock_data(symbol)
                df = add_features(df)
                df = label_data(df)
                df['symbol'] = symbol
                all_data.append(df)
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                traceback.print_exc()
    return pd.concat(all_data)

def train_model(df):
    df = df.dropna()
    feature_cols = [
        'close', 'volume', 'volatility', 'rsi', 'ma_20', 'ma_50', 'ma_100',
        'bollinger_upper', 'bollinger_lower', 'atr_14', 'log_volume', 'price_range',
        'close_lag1', 'close_lag2', 'close_lag3', 'vol_lag1', 'vol_lag2', 'vol_lag3'
    ]
    df = df.dropna(subset=feature_cols + ['target'])
    X = df[feature_cols]
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Grid search (expand as needed)
    param_grid = {
        'learning_rate': [0.1],
        'max_depth': [7],
        'n_estimators': [200],
        'subsample': [0.8]
        
    }

    model = GridSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    model.fit(X_train, y_train)
    print(f"Best params: {model.best_params_}")
    y_pred = model.predict(X_test)
    # For multi-class: ['Sell', 'Hold', 'Buy']
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=2, target_names=['Sell', 'Hold', 'Buy']))

    with open("breakout_model_priceaction.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("breakout_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("breakout_features.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    print("Model, scaler, and feature list saved.")
    return model, scaler, feature_cols

def plot_random_symbol(model, scaler, features, symbols):
    import matplotlib.pyplot as plt
    import random
    from datetime import datetime, timedelta

    for _ in range(10):
        symbol = random.choice(symbols)
        try:
            df = fetch_stock_data(symbol)
            df = add_features(df)
            df = label_data(df)
            end_date = df.index.max()
            start_date = end_date - timedelta(days=15)
            df_vis = df[df.index >= start_date]
            X_vis = scaler.transform(df_vis[features].dropna())
            df_vis = df_vis.iloc[-X_vis.shape[0]:]
            df_vis['pred'] = model.predict(X_vis)
            plt.figure(figsize=(16, 6))
            plt.plot(df_vis.index, df_vis['close'], label='Close', color='black')
            plt.scatter(df_vis[df_vis['pred'] == 2].index, df_vis[df_vis['pred'] == 2]['close'], color='green', marker='^', label='Buy', s=50)
            plt.scatter(df_vis[df_vis['pred'] == 0].index, df_vis[df_vis['pred'] == 0]['close'], color='red', marker='v', label='Sell', s=50)
            plt.scatter(df_vis[df_vis['pred'] == 1].index, df_vis[df_vis['pred'] == 1]['close'], color='orange', marker='o', label='Hold', s=50)
            plt.title(f'Breakout Model Prediction (Last 15 days) for {symbol}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            fname = f'breakout_plot_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(fname)
            print(f"Chart saved: {fname}")
            plt.show()
            return
        except Exception as e:
            print(f"Skipping {symbol} for plot: {e}")
    print("No valid symbol found to plot.")

if __name__ == "__main__":
    print("Fetching 300 random stock symbols...")
    symbols = fetch_random_symbols(300)
    print(f"Fetched {len(symbols)} symbols. Starting batching and data preparation...")
    df_all = prepare_data(symbols, batch_size=10)
    print(f"Total samples collected: {len(df_all)}. Starting training...")
    model, scaler, features = train_model(df_all)
    print("Plotting prediction for a random symbol (last 15 days)...")
    plot_random_symbol(model, scaler, features, symbols)