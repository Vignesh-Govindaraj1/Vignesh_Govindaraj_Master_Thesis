from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_required, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import os
from urllib.parse import quote_plus
import pandas as pd
import numpy as np
import traceback
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import psycopg2
from sqlalchemy import text, create_engine

# Load environment variables
load_dotenv()

# App config
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.getenv("APP_SECRET_KEY", "default_secret")

# Database config
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = quote_plus(os.getenv("DB_PASS", "postgres"))
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "dalalstreet")

# SQLAlchemy URI
app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
bcrypt = Bcrypt(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Raw psycopg2 connection
def get_conn():
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=os.getenv("DB_PASS")
    )

# SQLAlchemy engine for pandas
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind=engine)
db_session = Session()

# Other environment vars
Z_API_KEY = os.getenv("Z_API_KEY")
Z_API_SECRET = os.getenv("Z_API_SECRET")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "default_secret")
STOCK_NAME_DIR = '/home/ubuntu/dalalstreet/stock_name_data'

# UNIVERSAL LIVE DATA STORE (For all symbols)
live_data_store = {}

def preload_live_data():
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT stock_symbol FROM stock_history_1min LIMIT 1000;")
        symbols = cur.fetchall()
        cur.close()
        conn.close()
        for row in symbols:
            sym = row[0].upper()
            live_data_store[sym] = {'price': 1000 + hash(sym) % 1000}
    except Exception as e:
        print("Could not preload live_data_store:", e)

preload_live_data()

# Login manager
class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Load trained model 
model = pickle.load(open("breakout_model_priceaction.pkl", "rb"))

@app.route('/')
def index():
    return redirect(url_for("login"))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT id, password, role, approved FROM users WHERE username=%s", (username,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row or not check_password_hash(row[1], password):
            return render_template('login.html', error="Invalid credentials")
        if not row[3]:
            return render_template('login.html', error="Admin approval pending.")
        session['user'] = username
        session['role'] = row[2]
        session['user_id'] = row[0]
        return redirect('/chart')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        hashed = generate_password_hash(password)
        conn = get_conn()
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users (username, password, email) VALUES (%s, %s, %s)", (username, hashed, email))
            conn.commit()
        except:
            conn.rollback()
            return render_template('register.html', error="User already exists.")
        finally:
            cur.close()
            conn.close()
        return redirect('/login')
    return render_template('register.html')

@app.route('/admin')
def admin():
    if session.get('role') != 'admin':
        return redirect('/login')
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, username, email, role, approved FROM users")
    users = cur.fetchall()
    cur.close()
    conn.close()
    return render_template("admin.html", users=users)

@app.route('/approve/<int:user_id>')
def approve(user_id):
    if session.get('role') != 'admin':
        return redirect('/login')
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE users SET approved=true WHERE id=%s", (user_id,))
    conn.commit()
    cur.close()
    conn.close()
    return redirect('/admin')

@app.route("/zerodha-login")
def zerodha_login():
    """
    Starts the Zerodha login flow by redirecting to the Kite login URL.
    """
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=Z_API_KEY)
    login_url = kite.login_url()
    return redirect(login_url)


@app.route('/callback')
def zerodha_callback():
    from kiteconnect import KiteConnect
    request_token = request.args.get('request_token')
    if not request_token:
        return "Error: Missing request token from Zerodha", 400

    kite = KiteConnect(api_key=Z_API_KEY)
    try:
        data = kite.generate_session(request_token, api_secret=Z_API_SECRET)
        access_token = data["access_token"]

        # Save to file used by scraper
        token_file_path = "/home/ubuntu/dalalstreet/access_token.txt"
        os.makedirs(os.path.dirname(token_file_path), exist_ok=True)
        with open(token_file_path, "w") as f:
            f.write(access_token)

        print(f" Access token saved to {token_file_path}")
        return redirect('/chart')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Failed to generate access token: {e}", 500

@app.route('/chart')
def chart():
    return render_template("chart.html")

@app.route('/sector')
def sector_page():
    return render_template('sector_compass.html')

@app.route('/api/sectors')
def get_sectors():
    try:
        sector_dir = Path(STOCK_NAME_DIR)
        sectors = [f.stem for f in sector_dir.glob('*.csv') if f.is_file()]
        return jsonify({'sectors': sectors})
    except Exception as e:
        print("Error fetching sectors:", str(e))
        return jsonify({'sectors': []})


@app.route('/api/sector-trend', methods=['GET'])
def sector_trend_api():
    try:
        sector = request.args.get('sector')
        interval = request.args.get('interval')

        if not sector or not interval:
            return jsonify({'error': 'Missing sector or interval'}), 400

        # How much history to look back for each interval
        interval_map = {
            '1min': 15,      # 15 min for 1min interval
            '5min': 60,      # 1 hour for 5min interval
            '15min': 180,    # 3 hours for 15min interval
            '1h': 180,       # 3 hours for 1h interval
            '1d': 4320,      # 3 days for 1d interval (3*24*60)
            '1w': 43200,     # 1 month for 1w interval (30*24*60)
            '1mo': 86400     # 2 months for 1mo interval (60*24*60)
        }
        delta_min = interval_map.get(interval)
        if not delta_min:
            return jsonify({'error': 'Invalid interval'}), 400

        csv_path = os.path.expanduser(f'~/dalalstreet/stock_name_data/{sector}.csv')
        if not os.path.exists(csv_path):
            return jsonify({'error': f'CSV for sector {sector} not found'}), 404

        df = pd.read_csv(csv_path)
        symbol_col = next((col for col in df.columns if col.lower() in ['symbol', 'symbols']), None)
        if not symbol_col:
            return jsonify({'error': 'Symbol column not found in CSV'}), 500

        symbols = df[symbol_col].dropna().unique().tolist()
        if not symbols:
            return jsonify({'error': 'No symbols found in CSV'}), 404

        trend_data = []
        total_volume = 0

        for sym in symbols:
            sym = sym.strip().upper()
            # 1 Find latest timestamp for this symbol
            res_latest = db.session.execute(
                text("SELECT MAX(date) FROM stock_history_1min WHERE stock_symbol = :sym"),
                {"sym": sym}
            ).fetchone()
            latest_time = res_latest[0]
            if not latest_time:
                trend_data.append({
                    "symbol": sym, "current_volume": 0, "previous_volume": 0, "change_pct": 0
                })
                continue

            # 2 Set the range based on latest_time 
            latest_time = pd.to_datetime(str(latest_time))
            start_time = latest_time - pd.Timedelta(minutes=delta_min)
            prev_start = start_time - pd.Timedelta(minutes=delta_min)
            prev_end = start_time

            # 3Current and previous volume
            res_curr = db.session.execute(
                text("""SELECT SUM(volume) FROM stock_history_1min 
                        WHERE stock_symbol = :sym AND date > :start_time AND date <= :end_time"""),
                {"sym": sym, "start_time": start_time, "end_time": latest_time}
            ).fetchone()
            curr_vol = res_curr[0] if res_curr and res_curr[0] else 0

            res_prev = db.session.execute(
                text("""SELECT SUM(volume) FROM stock_history_1min 
                        WHERE stock_symbol = :sym AND date > :prev_start AND date <= :prev_end"""),
                {"sym": sym, "prev_start": prev_start, "prev_end": prev_end}
            ).fetchone()
            prev_vol = res_prev[0] if res_prev and res_prev[0] else 0

            change = ((curr_vol - prev_vol) / prev_vol * 100) if prev_vol > 0 else 0
            total_volume += curr_vol

            trend_data.append({
                "symbol": sym,
                "current_volume": int(curr_vol),
                "previous_volume": int(prev_vol),
                "change_pct": round(change, 2)
            })

        for entry in trend_data:
            entry["strength"] = round((entry["current_volume"] / total_volume) * 100, 2) if total_volume > 0 else 0

        trend_data.sort(key=lambda x: x['current_volume'], reverse=True)
        return jsonify(trend_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/api/symbols')
def api_symbols():
    """
    Returns all symbols from your master stock table (populated from CSVs).
    Assumes a table named 'stock_master' with 'symbol' as the column.
    """
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT symbol FROM stock_master ORDER BY symbol")
        symbols = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        return jsonify(symbols)
    except Exception as e:
        print("Error in /api/symbols:", e)
        return jsonify([]), 500

@app.route('/api/candles-live')
def candles_live():
    try:
        symbol = request.args.get("symbol", "").strip()
        interval = request.args.get("interval", "1min")
        if not symbol:
            return jsonify({"error": "Missing symbol"}), 400
        print(f"candles-live: Got symbol={symbol!r} interval={interval!r}")
        conn = get_conn()
        df = pd.read_sql("""
            SELECT date, open, high, low, close, volume
            FROM stock_history_1min
            WHERE stock_symbol = %s AND date >= NOW() - INTERVAL '2000 days'
            ORDER BY date ASC
        """, conn, params=(symbol,))
        conn.close()
        if df.empty:
            print(f"No data found for symbol: {symbol}")
            return jsonify([])
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df.set_index('date', inplace=True)
        if interval == "1min":
            df_resampled = df
        else:
            rule = {
                "5min": "5T", "15min": "15T", "1h": "1H", "1d": "1D", "1w": "1W"
            }.get(interval)
            if rule:
                df_resampled = df.resample(rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            else:
                return jsonify({"error": "Unsupported interval"}), 400

        candles = [ {
            "time": int(ts.timestamp()),
            "open": row['open'],
            "high": row['high'],
            "low": row['low'],
            "close": row['close'],
            "volume": row['volume']
        } for ts, row in df_resampled.iterrows()]
        print(f"candles-live: Returning {len(candles)} candles for {symbol}")
        return jsonify(candles)
    except Exception as e:
        print("Error in /api/candles-live:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict-breakouts')
def predict_breakouts():
    try:
        symbol = request.args.get('symbol', '').upper()
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400

        model_path = 'breakout_model_priceaction.pkl'
        scaler_path = 'breakout_scaler.pkl'
        features_path = 'breakout_features.pkl'
        if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
            return jsonify({'error': 'Model or scaler file not found'}), 500

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(features_path, 'rb') as f:
            feature_cols = pickle.load(f)

        #USE THE MOST RECENT 20000 ROWS NOT TO OVERLOAD SERVER
        query = text("""
            SELECT date, open, high, low, close, volume
            FROM stock_history_1min
            WHERE stock_symbol = :symbol
            ORDER BY date DESC
            LIMIT 20000
        """)
        result = db_session.execute(query, {'symbol': symbol}).fetchall()
        if not result:
            return jsonify({'error': f'No data found for symbol {symbol}'}), 404

        df = pd.DataFrame(result, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df.sort_values('date').reset_index(drop=True)  

        
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
        df = df.dropna().reset_index(drop=True)

        if df.shape[0] == 0:
            return jsonify([])

        X = df[feature_cols]
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)

        #Show only signals from the most recent 30 days
        latest = df['date'].max()
        cutoff = latest - pd.Timedelta(days=30)
        mask = df['date'] >= cutoff

        result = []
        for i in range(len(preds)):
            if not mask.iloc[i]:
                continue
            if preds[i] == 2:  #BUY
                result.append({
                    "timestamp": int(df.loc[i, 'date'].timestamp()),
                    "pred": 2,
                    "label": "BUY",
                    "color": "lime",
                    "shape": "arrowUp",
                    "position": "belowBar"
                })
            elif preds[i] == 0:  #SELL
                result.append({
                    "timestamp": int(df.loc[i, 'date'].timestamp()),
                    "pred": 0,
                    "label": "SELL",
                    "color": "red",
                    "shape": "arrowDown",
                    "position": "aboveBar"
                })
        print(f"Breakout API {symbol}: {len(result)} signals in last 30 days")
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/support-resistance')
def support_resistance():
    try:
        symbol = request.args.get('symbol', '').upper()
        interval_min = int(request.args.get('interval', 15))

        if not symbol:
            return jsonify({'error': 'Missing symbol parameter'}), 400

        conn = get_conn()
        try:
            df = pd.read_sql("""
                SELECT date, open, high, low, close, volume
                FROM stock_history_1min
                WHERE stock_symbol = %s AND date >= NOW() - INTERVAL '30 days'
                ORDER BY date ASC
            """, conn, params=(symbol,))
        finally:
            conn.close()
        if df.empty:
            return jsonify([])

        df['date'] = pd.to_datetime(df['date'], utc=True)
        df.set_index('date', inplace=True)
        df_resampled = df.resample(f'{interval_min}T').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna().reset_index()

        if df_resampled.empty or len(df_resampled) < 7:
            return jsonify([])

        window = 3
        supports, resistances = [], []
        for i in range(window, len(df_resampled) - window):
            low = df_resampled['low'][i]
            high = df_resampled['high'][i]
            if all(low < df_resampled['low'][i - w] for w in range(1, window + 1)) and \
               all(low < df_resampled['low'][i + w] for w in range(1, window + 1)):
                supports.append(i)
            if all(high > df_resampled['high'][i - w] for w in range(1, window + 1)) and \
               all(high > df_resampled['high'][i + w] for w in range(1, window + 1)):
                resistances.append(i)

        zones = []
        for idx in supports:
            start_idx = max(idx - 2, 0)
            end_idx = min(idx + 2, len(df_resampled) - 1)
            zones.append({
                'type': 'support',
                'start_time': int(df_resampled['date'][start_idx].timestamp()),
                'end_time': int(df_resampled['date'][end_idx].timestamp()),
                'price': float(df_resampled['low'][idx])
            })
        for idx in resistances:
            start_idx = max(idx - 2, 0)
            end_idx = min(idx + 2, len(df_resampled) - 1)
            zones.append({
                'type': 'resistance',
                'start_time': int(df_resampled['date'][start_idx].timestamp()),
                'end_time': int(df_resampled['date'][end_idx].timestamp()),
                'price': float(df_resampled['high'][idx])
            })

        def dedup(zones, pct=0.0025):
            res = []
            for z in sorted(zones, key=lambda x: x['price']):
                if all(abs(z['price'] - zz['price']) > pct * z['price'] for zz in res):
                    res.append(z)
            return res

        supports_final = dedup([z for z in zones if z['type'] == 'support'])
        resistances_final = dedup([z for z in zones if z['type'] == 'resistance'])
        final_zones = supports_final + resistances_final

        return jsonify(final_zones)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify([]), 500
@app.route('/market-sentiment')
def market_sentiment():
    csv_file = 'market_sentiment_latest.csv'
    if not os.path.exists(csv_file):
        return render_template('market_sentiment.html', results=[], error="No sentiment data found. Please run the analysis script.")
    df = pd.read_csv(csv_file)
    # Sort so highest positive or negative are at the top
    df_sorted = df.reindex(df['avg_score'].abs().sort_values(ascending=False).index)
    results = df_sorted[['symbol', 'avg_score', 'net_sentiment']].to_dict(orient='records')
    return render_template('market_sentiment.html', results=results, error=None)

if __name__ == "__main__":
    app.run(debug=True)