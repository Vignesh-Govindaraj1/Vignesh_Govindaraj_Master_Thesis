import os
import urllib.parse
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
if not all([DB_USER, DB_PASS, DB_HOST, DB_NAME]):
    raise Exception("Missing DB_USER/DB_PASS/DB_HOST/DB_NAME in .env or env vars!")
DB_PASS_ENC = urllib.parse.quote_plus(DB_PASS)
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS_ENC}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

def fetch_all_symbols(engine):
    df = pd.read_sql("SELECT symbol FROM stock_master", engine)
    return df['symbol'].unique().tolist()

def fetch_yahoo_headlines(symbol):
    url = f'https://finance.yahoo.com/quote/{symbol}/news?p={symbol}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        headlines = [h3.get_text(strip=True) for h3 in soup.find_all('h3')]
        return headlines
    except Exception as e:
        print(f"Failed to fetch news for {symbol}: {e}")
        return []

tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

def finbert_sentiment(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()[0]
        idx = np.argmax(probs)
        sentiment = {0: "negative", 1: "neutral", 2: "positive"}[idx]
        score = float(probs[2]) - float(probs[0])
        return sentiment, score
    except Exception as e:
        print(f"Sentiment error for '{text}': {e}")
        return "neutral", 0.0

def analyze_sentiment_all(engine):
    symbols = fetch_all_symbols(engine)
    all_results = []
    for symbol in symbols:
        yfin_symbol = symbol if symbol.endswith('.NS') else f"{symbol}.NS"
        headlines = fetch_yahoo_headlines(yfin_symbol)
        sentiments = []
        for h in headlines:
            sent, score = finbert_sentiment(h)
            sentiments.append((h, sent, score))
        if sentiments:
            scores = [s[2] for s in sentiments]
            avg_score = np.mean(scores)
            net_sent = "positive" if avg_score > 0.1 else "negative" if avg_score < -0.1 else "neutral"
        else:
            avg_score, net_sent = 0, "neutral"
        all_results.append({
            "symbol": symbol,
            "avg_score": avg_score,
            "net_sentiment": net_sent,
            "n_headlines": len(sentiments),
            "headlines": "; ".join([s[0] for s in sentiments])
        })
        print(f"{symbol}: {net_sent} ({avg_score:.2f}) from {len(sentiments)} headlines")
    return pd.DataFrame(all_results)

def evaluate_sentiment(df, ground_truth_file='sentiment_manual_labels.csv'):
    if not os.path.exists(ground_truth_file):
        print("No ground truth file found. Skipping evaluation.")
        return None
    gt = pd.read_csv(ground_truth_file)
    merged = df[['symbol', 'net_sentiment']].merge(gt, on='symbol', suffixes=('_pred', '_true'))
    accuracy = (merged['net_sentiment'] == merged['true_sentiment']).mean()
    print(f"\nSentiment model accuracy (manual validation): {accuracy:.2%}")
    return accuracy

def plot_sentiment_distribution(df):
    counts = df['net_sentiment'].value_counts()
    counts.plot(kind='bar', color=['green', 'grey', 'red'])
    plt.title('Market Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of stocks')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    df_result = analyze_sentiment_all(engine)
    df_result.to_csv('market_sentiment_latest.csv', index=False)
    plot_sentiment_distribution(df_result)
    
    evaluate_sentiment(df_result, ground_truth_file='sentiment_manual_labels.csv')