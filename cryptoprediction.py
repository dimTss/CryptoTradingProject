import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import yfinance as yf
from praw import Reddit
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#from tweepy import twitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

SYMBOL = "BTC-USD"
INTERVAL = "15m"
LOOKBACK = 4 # hours of history
PRED_HOURS = 24
MODEL_FILE = "rf_btc_indicator.pkl"
SCALER_FILE = "rf_btc_scaler.pkl"

REDDIT_CFG = dict(client_id="8ZYOUfRVuLdwCV74VfkCOw", client_secret="vajteyckL10dyE1x7HdH_dULOc2uJw", user_agent="btc-sentiment")
#TWITTER_CFG = dict(bearer_token="")

sia = SentimentIntensityAnalyzer()
reddit = Reddit(**REDDIT_CFG)
#twitter = tweepy.Client(**TWITTER_CFG)

def download_data():
    df = yf.download(SYMBOL, period="25d", interval="15m", auto_adjust=True).dropna()
    df.index = pd.to_datetime(df.index)
    return df

def generate_features(df):
    df['ret6'] = df['Close'].pct_change(6)
    df['ret12'] = df['Close'].pct_change(12)
    df['hour'] = df.index.hour
    df['dow'] = df.index.dayofweek
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['dow_sin'] = np.sin(2*np.pi*df['dow']/7)
    df['dow_cos'] = np.cos(2*np.pi*df['dow']/7)
    df['future_ret'] = df['Close'].pct_change(PRED_HOURS).shift(-PRED_HOURS)
    df['trend'] = (df['future_ret'] > 0).astype(int)
    return df.dropna()

from datetime import timezone

def fetch_reddit_sentiment(df):
    from datetime import timezone
    import pandas as pd
    import numpy as np
    
    # Get the full date range
    start_time = df.index.min()
    end_time = df.index.max()
    
    # Ensure timezone consistency - convert to UTC if needed
    if start_time.tz is None:
        start_time = start_time.tz_localize('UTC')
    if end_time.tz is None:
        end_time = end_time.tz_localize('UTC')
    
    # Fetch all recent posts once (adjust the number to cover your range)
    print("🔄 Fetching Reddit posts in bulk...")
    
    try:
        all_posts = list(reddit.subreddit("Bitcoin").new(limit=1000))
    except NameError:
        print("⚠️  Reddit instance not found. Please ensure PRAW is properly configured.")
        return df
    
    # Extract relevant posts with timestamps
    post_data = []
    for post in all_posts:
        try:
            created = pd.Timestamp(post.created_utc, unit='s', tz='UTC')
            
            # Check if post falls within our time range
            if start_time <= created <= end_time:
                text = post.title + " " + getattr(post, 'selftext', '')
                
                try:
                    score = sia.polarity_scores(text)['compound']
                except NameError:
                    print("⚠️  Sentiment analyzer not found. Please ensure NLTK VADER is initialized.")
                    return df
                
                post_data.append((created, score))
        except Exception as e:
            print(f"⚠️  Error processing post: {e}")
            continue
    
    print(f"📊 Found {len(post_data)} relevant posts in time range")
    
    # Initialize reddit_sent column
    df['reddit_sent'] = 0.0
    
    # Group sentiment by hour - more efficient approach
    if post_data:
        # Create a DataFrame for easier manipulation
        sentiment_df = pd.DataFrame(post_data, columns=['timestamp', 'sentiment'])
        sentiment_df.set_index('timestamp', inplace=True)
        
        # Resample to hourly averages
        hourly_sentiment = sentiment_df.resample('H')['sentiment'].mean()
        
        # Map to original DataFrame index
        for t in df.index:
            # Find the hour bucket this timestamp belongs to
            hour_start = t.floor('H')
            if hour_start in hourly_sentiment.index:
                df.at[t, 'reddit_sent'] = hourly_sentiment[hour_start]
            else:
                df.at[t, 'reddit_sent'] = 0.0
    else:
        print("⚠️  No posts found in the specified time range")
    
    return df




"""def fetch_twitter_sentiment(df):
    df['twitter_sent'] = 0.0
    for t in df.index:
        start, end = t, t + pd.Timedelta(hours=1)
        tweets = twitter.search_recent_tweets(query="bitcoin -is:retweet", start_time=start.isoformat(),
                                              end_time=end.isoformat(), max_results=50)
        s = [sia.polarity_scores(t.text)['compound'] for t in tweets.data] if tweets.data else []
        df.at[t, 'twitter_sent'] = np.mean(s) if s else 0
    return df"""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.svm import SVC

def generate_new_features(df):
    df['future_ret'] = df['Close'].shift(-PRED_HOURS) / df['Close'] - 1
    df['trend'] = df['future_ret'].gt(0).astype(int)
    df['prev_future_ret'] = df['future_ret'].shift(1)  # previous t's future return (t−1 → t−1+H)
    df['avg_past_future_ret'] = df['future_ret'].shift(1).rolling(3).mean()
    df['prev_future_ret'] = df['future_ret'].shift(PRED_HOURS)  # now it's from the past
    df['avg_past_future_ret'] = df['future_ret'].shift(PRED_HOURS).rolling(3).mean()
    df['ma_6'] = df['Close'].rolling(6).mean()
    df['ma_12'] = df['Close'].rolling(12).mean()
    df['volatility'] = df['Close'].rolling(6).std()
    df['volume_change'] = df['Volume'].pct_change()
    df['rsi'] = compute_rsi(df['Close'], period=14)
    df['macd'] = compute_macd(df['Close'])
    df['bollinger_bandwidth'] = -1* compute_bollinger_bandwidth(df['Close'])
    df['roc'] = compute_roc(df['Close'])
    df['sma_fast'] = df['Close'].rolling(window=5).mean()
    df['sma_slow'] = df['Close'].rolling(window=20).mean()
    df['sma_cross'] = (df['sma_fast'] > df['sma_slow']).astype(int)
    df['ret1'] = df['Close'].pct_change(1)
    df['ret3'] = df['Close'].pct_change(3)
    df['range'] = df['High'] - df['Low']
    df['log_volume'] = np.log1p(df['Volume'])
    df['dif'] = df['Close'] - df['Open']
    # ---- Momentum indicators ----
    df['momentum_3']  = df['Close'] - df['Close'].shift(3)
    df['momentum_6']  = df['Close'] - df['Close'].shift(6)

    # ---- Exponential Moving Averages (EMAs) ----
    df['ema_6']   = df['Close'].ewm(span=6, adjust=False).mean()
    df['ema_12']  = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26']  = df['Close'].ewm(span=26, adjust=False).mean()

    # ---- Additional Rate of Change ----
    df['roc_6']   = ((df['Close'] - df['Close'].shift(6)) / df['Close'].shift(6)) * 100
    df['roc_12']  = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)) * 100

    # ---- Average True Range (ATR) ----
    high_low       = df['High'] - df['Low']
    high_close     = (df['High'] - df['Close'].shift(1)).abs()
    low_close      = (df['Low'] - df['Close'].shift(1)).abs()
    df['tr']       = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14']   = df['tr'].rolling(window=14).mean()

    # ---- On-Balance Volume (OBV) ----
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv

    # ---- Williams %R ----
    df['williams_r'] = (df['High'].rolling(14).max() - df['Close']) / (df['High'].rolling(14).max() - df['Low'].rolling(14).min()) * -100

    # ---- Chaikin Money Flow (CMF) ----
    mf_mult = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mf_vol = mf_mult * df['Volume']
    df['cmf'] = mf_vol.rolling(20).sum() / df['Volume'].rolling(20).sum()

    # ---- Stochastic Oscillator (%K and %D) ----
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # ---- VWAP (Volume Weighted Average Price) ----
    cum_vol_price = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum()
    cum_vol = df['Volume'].cumsum()
    df['vwap'] = cum_vol_price / cum_vol

    # ---- Price Change (1-step return) ----
    df['price_change'] = df['Close'].pct_change()

    # Slope of 6-period MA — approximates recent trend steepness
    df['ma_slope_6'] = df['ma_6'] - df['ma_6'].shift(1)

    # Position of Close relative to rolling max/min
    df['price_pos_in_range_6'] = (df['Close'] - df['Low'].rolling(6).min()) / (df['High'].rolling(6).max() - df['Low'].rolling(6).min())

    # Rolling volatility ratio
    df['volatility_ratio'] = df['Close'].rolling(6).std() / df['Close'].rolling(20).std()

    # Bollinger Band position — normalized distance from mid-band
    middle_bb = df['Close'].rolling(20).mean()
    upper_bb = middle_bb + 2 * df['Close'].rolling(20).std()
    lower_bb = middle_bb - 2 * df['Close'].rolling(20).std()
    df['bb_percent'] = (df['Close'] - lower_bb) / (upper_bb - lower_bb)

    # Candle body and wicks
    df['candle_body'] = (df['Close'] - df['Open']).abs()
    df['upper_wick'] = df['High'] - np.maximum(df['Close'], df['Open'])
    df['lower_wick'] = np.minimum(df['Close'], df['Open']) - df['Low']

    # Shape of the candle as signal
    df['candle_signal'] = np.sign(df['Close'] - df['Open'])  # 1 if bullish, -1 if bearish

    # Volatility breakout pattern
    df['volatility_breakout'] = (df['High'] > df['High'].shift(1)) & (df['Low'] < df['Low'].shift(1))

    # Reversal pattern indicator
    df['inverted_hammer_like'] = ((df['upper_wick'] > 2 * df['candle_body']) & (df['lower_wick'] < df['candle_body']))

    # Rolling volume z-score (standardized anomaly detection)
    df['volume_zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()

    # True Range to Previous Close (relative gap size)
    df['gap_pct'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

    # Percentage change from Open to High and Low
    df['open_to_high_pct'] = (df['High'] - df['Open']) / df['Open']
    df['open_to_low_pct']  = (df['Open'] - df['Low']) / df['Open']

    # Absorption of range
    df['body_to_range'] = df['candle_body'] / (df['High'] - df['Low'] + 1e-8)

    # Rolling realized volatility (standard deviation of returns)
    df['realized_vol_10'] = df['Close'].pct_change().rolling(10).std()

    # Return autocorrelation (short-term)
    df['ret_autocorr_5'] = df['Close'].pct_change().rolling(5).apply(lambda x: x.autocorr(), raw=False)

    # Volume as fraction of rolling median
    df['vol_med_ratio'] = df['Volume'] / (df['Volume'].rolling(20).median() + 1e-8)

    # Price movement per unit volume (impact indicator)
    df['price_per_vol'] = df['Close'].pct_change() / (df['Volume'] + 1e-8)

    df['rsi_x_vol'] = df['rsi'] * df['volatility']
    df['obv_x_vol'] = df['obv'] * df['volatility']
    



    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import requests
from bs4 import BeautifulSoup
from datetime import timedelta
import pandas as pd
import feedparser
from datetime import timedelta
from urllib.parse import quote_plus

def fetch_web_sentiment(df, query="Bitcoin"):
    sia = SentimentIntensityAnalyzer()
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    else:
        df = df.tz_convert("UTC")
        
    start = df.index.min().strftime("%Y%m%d%H%M%S")
    end   = df.index.max().strftime("%Y%m%d%H%M%S")
    
    url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={query}&mode=ArtList&maxrecords=250&format=JSON&startdatetime={start}&enddatetime={end}"
    r = requests.get(url)
    if r.status_code != 200:
        print("Error fetching:", r.status_code)
        df["web_sent"] = 0.0
        return df

    data = r.json()
    if "articles" not in data or not data["articles"]:
        df["web_sent"] = 0.0
        return df

    rows = []
    for art in data["articles"]:
        try:
            ts = pd.to_datetime(art["seendate"], utc=True)
        except Exception:
            continue
        text = f"{art.get('title','')} {art.get('sourceURL','')}"
        score = sia.polarity_scores(text)["compound"]
        rows.append((ts, score))

    if not rows:
        df["web_sent"] = 0.0
        return df

    news_df = pd.DataFrame(rows, columns=["timestamp", "sentiment"])
    news_df.set_index("timestamp", inplace=True)
    hourly = news_df.resample("H").mean()

    out = df.copy()
    out["hour"] = out.index.floor("H")
    out = out.merge(hourly, left_on="hour", right_index=True, how="left")
    out.rename(columns={"sentiment": "web_sent"}, inplace=True)
    out["web_sent"].fillna(0.0, inplace=True)
    out.drop(columns=["hour"], inplace=True)
    
    return out



def new_features2(df, PRED_HOURS=6):
    df['rsi_x_vol'] = df['rsi'] * df['volatility']
    df['obv_x_vol'] = df['obv'] * df['volatility']

    # BASIC RETURNS
    df['ret1'] = df['Close'].pct_change(1)
    df['ret3'] = df['Close'].pct_change(3)
    df['ret6'] = df['Close'].pct_change(6)
    df['ret12'] = df['Close'].pct_change(12)
    
    # === NEW: EXTENDED RETURNS ===
    df['ret24'] = df['Close'].pct_change(24)
    df['ret48'] = df['Close'].pct_change(48)
    df['ret72'] = df['Close'].pct_change(72)

    # MOMENTUM
    df['momentum_3'] = df['Close'] - df['Close'].shift(3)
    df['momentum_6'] = df['Close'] - df['Close'].shift(6)
    
    # === NEW: ADVANCED MOMENTUM ===
    df['momentum_12'] = df['Close'] - df['Close'].shift(12)
    df['momentum_24'] = df['Close'] - df['Close'].shift(24)
    df['price_acceleration'] = df['ret1'] - df['ret1'].shift(1)
    df['momentum_ratio'] = df['momentum_3'] / (df['momentum_6'] + 1e-8)

    # VOLATILITY
    df['volatility_6'] = df['Close'].rolling(6).std()
    df['volatility_12'] = df['Close'].rolling(12).std()
    df['volatility_ratio'] = df['volatility_6'] / (df['volatility_12'] + 1e-8)
    df['realized_vol_10'] = df['ret1'].rolling(10).std()
    
    # === NEW: ADVANCED VOLATILITY ===
    df['volatility_3'] = df['Close'].rolling(3).std()
    df['volatility_24'] = df['Close'].rolling(24).std()
    df['vol_of_vol'] = df['volatility_6'].rolling(10).std()
    df['garch_proxy'] = df['ret1'].rolling(10).std() / df['ret1'].rolling(50).std()
    df['volatility_regime'] = (df['volatility_6'] > df['volatility_6'].rolling(50).quantile(0.8)).astype(int)
    df['vol_skew'] = df['ret1'].rolling(20).skew()
    df['vol_kurt'] = df['ret1'].rolling(20).kurt()

    # VOLUME & FLOW
    df['volume_change'] = df['Volume'].pct_change()
    df['log_volume'] = np.log1p(df['Volume'])
    df['volume_zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
    df['vol_med_ratio'] = df['Volume'] / (df['Volume'].rolling(20).median() + 1e-8)
    
    # === NEW: ADVANCED VOLUME ===
    df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(50).mean()
    df['volume_acceleration'] = df['volume_change'] - df['volume_change'].shift(1)
    df['turnover_ratio'] = df['Volume'] / df['Volume'].rolling(30).mean()
    df['volume_trend'] = df['Volume'].rolling(10).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0,1], raw=False)

    # MOVING AVERAGES
    df['ma_6'] = df['Close'].rolling(6).mean()
    df['ma_12'] = df['Close'].rolling(12).mean()
    df['ma_slope_6'] = df['ma_6'] - df['ma_6'].shift(1)
    df['ema_6'] = df['Close'].ewm(span=6, adjust=False).mean()
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['sma_fast'] = df['Close'].rolling(5).mean()
    df['sma_slow'] = df['Close'].rolling(20).mean()
    df['sma_cross'] = (df['sma_fast'] > df['sma_slow']).astype(int)
    
    # === NEW: ADVANCED MOVING AVERAGES ===
    df['ma_50'] = df['Close'].rolling(50).mean()
    df['ma_100'] = df['Close'].rolling(100).mean()
    df['hull_ma'] = (2 * df['Close'].rolling(6).mean() - df['Close'].rolling(12).mean()).rolling(3).mean()
    df['kama'] = df['Close'].ewm(alpha=0.2).mean()  # Kaufman Adaptive MA proxy
    df['distance_from_sma'] = (df['Close'] - df['sma_slow']) / df['sma_slow']
    df['ma_envelope_upper'] = df['sma_slow'] * 1.05
    df['ma_envelope_lower'] = df['sma_slow'] * 0.95
    df['price_above_ma'] = (df['Close'] > df['sma_slow']).astype(int)

    # PRICE ACTION
    df['range'] = df['High'] - df['Low']
    df['dif'] = df['Close'] - df['Open']
    df['price_change'] = df['Close'].pct_change()
    df['candle_body'] = (df['Close'] - df['Open']).abs()
    df['upper_wick'] = df['High'] - np.maximum(df['Close'], df['Open'])
    df['lower_wick'] = np.minimum(df['Close'], df['Open']) - df['Low']
    df['body_to_range'] = df['candle_body'] / (df['range'] + 1e-9)
    
    # === NEW: ADVANCED PRICE ACTION ===
    df['range_ma'] = df['range'].rolling(20).mean()
    df['range_ratio'] = df['range'] / (df['range_ma'] + 1e-8)
    df['price_position'] = (df['Close'] - df['Low']) / (df['range'] + 1e-8)
    df['buying_pressure'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
    df['selling_pressure'] = (df['High'] - df['Close']) / (df['High'] - df['Low'] + 1e-8)

    # CANDLE PATTERNS
    df['candle_signal'] = np.sign(df['Close'] - df['Open'])  # 1/-1
    df['inverted_hammer_like'] = ((df['upper_wick'] > 2 * df['candle_body']) & (df['lower_wick'] < df['candle_body'])).astype(int)
    df['volatility_breakout'] = ((df['High'] > df['High'].shift(1)) & (df['Low'] < df['Low'].shift(1))).astype(int)
    df['trend_flip'] = (np.sign(df['ret1']) != np.sign(df['ret1'].shift(1))).astype(int)
    
    # === NEW: ADVANCED CANDLE PATTERNS ===
    df['doji'] = (df['candle_body'] < (df['range'] * 0.1)).astype(int)
    df['hammer'] = ((df['lower_wick'] > 2 * df['candle_body']) & (df['upper_wick'] < df['candle_body'] * 0.5)).astype(int)
    df['shooting_star'] = ((df['upper_wick'] > 2 * df['candle_body']) & (df['lower_wick'] < df['candle_body'] * 0.5)).astype(int)
    df['marubozu'] = (df['candle_body'] > df['range'] * 0.9).astype(int)
    df['spinning_top'] = ((df['upper_wick'] > df['candle_body']) & (df['lower_wick'] > df['candle_body'])).astype(int)

    # INDICATORS
    df['price_pos_in_range_6'] = (df['Close'] - df['Low'].rolling(6).min()) / (df['High'].rolling(6).max() - df['Low'].rolling(6).min() + 1e-8)
    df['bb_percent'] = (
        (df['Close'] - df['Close'].rolling(20).mean() + 1e-8) /
        (2 * df['Close'].rolling(20).std() + 1e-8)
    )
    df['rsi'] = compute_rsi(df['Close'], period=14)
    df['macd'] = compute_macd(df['Close'])
    df['roc_6'] = 100 * (df['Close'] - df['Close'].shift(6)) / (df['Close'].shift(6) + 1e-8)
    df['roc_12'] = 100 * (df['Close'] - df['Close'].shift(12)) / (df['Close'].shift(12) + 1e-8)
    df['williams_r'] = (df['High'].rolling(14).max() - df['Close']) / (df['High'].rolling(14).max() - df['Low'].rolling(14).min() + 1e-8) * -100
    df['cmf'] = (((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-8) * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
    
    # === NEW: ADVANCED INDICATORS ===
    df['rsi_14'] = compute_rsi(df['Close'], period=14)
    df['rsi_21'] = compute_rsi(df['Close'], period=21)
    df['rsi_sma'] = df['rsi'].rolling(5).mean()
    df['rsi_divergence'] = df['rsi'] - df['rsi'].shift(5)
    df['atr_14'] = df['range'].rolling(14).mean()  # Simplified ATR
    df['atr_ratio'] = df['range'] / (df['atr_14'] + 1e-8)
    df['cci'] = (df['Close'] - df['Close'].rolling(20).mean()) / (0.015 * df['Close'].rolling(20).std())
    df['mfi'] = (df['buying_pressure'] * df['Volume']).rolling(14).sum() / df['Volume'].rolling(14).sum() * 100

    # STOCHASTIC
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14 + 1e-8))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # === NEW: ADVANCED STOCHASTIC ===
    df['stoch_rsi'] = 100 * ((df['rsi'] - df['rsi'].rolling(14).min()) / (df['rsi'].rolling(14).max() - df['rsi'].rolling(14).min() + 1e-8))

    # VWAP
    vwap_numerator = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum()
    vwap_denominator = df['Volume'].cumsum()
    df['vwap'] = vwap_numerator / (vwap_denominator + 1e-8)
    
    # === NEW: ADVANCED VWAP ===
    df['distance_from_vwap'] = (df['Close'] - df['vwap']) / df['vwap']
    df['vwap_slope'] = df['vwap'] - df['vwap'].shift(1)

    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    
    # === NEW: ADVANCED OBV ===
    df['obv_sma'] = df['obv'].rolling(20).mean()
    df['obv_divergence'] = df['obv'] - df['obv'].shift(10)

    # ADVANCED: GAPS & ORDERBOOK SURROGATES
    df['gap_pct'] = (df['Open'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-8)
    df['open_to_high_pct'] = (df['High'] - df['Open']) / (df['Open'] + 1e-8)
    df['open_to_low_pct'] = (df['Open'] - df['Low']) / (df['Open'] + 1e-8)
    df['price_per_vol'] = df['Close'].pct_change() / (df['Volume'] + 1e-8)
    
    # === NEW: MICROSTRUCTURE FEATURES ===
    df['amihud_illiq'] = df['ret1'].abs() / (df['Volume'] + 1e-8)  # Amihud illiquidity measure
    df['volume_price_trend'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * df['Volume']).rolling(10).sum()

    # === NEW: FRACTAL & PATTERN FEATURES ===
    df['fractal_high'] = ((df['High'] > df['High'].shift(2)) & 
                         (df['High'] > df['High'].shift(1)) & 
                         (df['High'] > df['High'].shift(-1)) & 
                         (df['High'] > df['High'].shift(-2))).astype(int)
    df['fractal_low'] = ((df['Low'] < df['Low'].shift(2)) & 
                        (df['Low'] < df['Low'].shift(1)) & 
                        (df['Low'] < df['Low'].shift(-1)) & 
                        (df['Low'] < df['Low'].shift(-2))).astype(int)

    # === NEW: BREAKOUT & REGIME FEATURES ===
    df['breakout_high'] = (df['Close'] >= df['Close'].rolling(20).max()).astype(int)
    df['breakout_low'] = (df['Close'] <= df['Close'].rolling(20).min()).astype(int)
    df['support_distance'] = (df['Close'] - df['Low'].rolling(20).min()) / df['Close']
    df['resistance_distance'] = (df['High'].rolling(20).max() - df['Close']) / df['Close']

    # === NEW: TREND & AUTOCORRELATION ===
    df['trend_strength'] = df['Close'].rolling(20).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0,1] if len(x) > 1 else 0, raw=False)
    df['autocorr_1'] = df['ret1'].rolling(10).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False)
    df['autocorr_3'] = df['ret1'].rolling(10).apply(lambda x: x.autocorr(lag=3) if len(x) > 3 else 0, raw=False)

    # === NEW: ENTROPY & COMPLEXITY ===
    """def shannon_entropy(series, bins=10):
        try:
            counts, _ = np.histogram(series.dropna(), bins=bins)
            probs = counts / counts.sum()
            return -np.sum(probs * np.log2(probs + 1e-8))
        except:
            return 0
    
    df['price_entropy'] = df['ret1'].rolling(20).apply(lambda x: shannon_entropy(x), raw=False)
    df['volume_entropy'] = df['volume_change'].rolling(20).apply(lambda x: shannon_entropy(x), raw=False)"""

    # === NEW: HIGHER-ORDER STATISTICS ===
    for window in [10, 20]:
        df[f'skewness_{window}'] = df['ret1'].rolling(window).skew()
        df[f'kurtosis_{window}'] = df['ret1'].rolling(window).kurt()

    # === NEW: REGIME & BETA FEATURES ===
    df['rolling_beta'] = df['ret1'].rolling(20).cov(df['ret1'].shift(1)) / (df['ret1'].shift(1).rolling(20).var() + 1e-8)
    df['market_beta'] = df['ret1'].rolling(50).cov(df['ret1']) / (df['ret1'].rolling(50).var() + 1e-8)

    # INTERACTION FEATURES
    df['rsi_x_vol'] = df['rsi'] * df['volatility_6']
    df['obv_x_vol'] = df['obv'] * df['volatility_6']
    
    # ADDITIONAL INTERACTION FEATURES
    df['momentum_vol'] = df['momentum_6'] * df['volatility_6']
    df['rsi_momentum'] = df['rsi'] * df['momentum_3']
    df['volume_volatility'] = df['log_volume'] * df['volatility_6']
    df['gap_volume'] = df['gap_pct'] * df['log_volume']
    
    # === NEW: ADVANCED INTERACTIONS ===
    df['macd_rsi'] = df['macd'] * df['rsi']
    df['bb_rsi'] = df['bb_percent'] * df['rsi']
    df['atr_volume'] = df['atr_14'] * df['log_volume']
    df['trend_momentum'] = df['trend_strength'] * df['momentum_6']
    df['volatility_trend'] = df['volatility_6'] * df['trend_strength']
    df['price_volume_interaction'] = df['Close'] * df['Volume'] / 1e6  # Scaled for numerical stability
    
    # === NEW: BOLLINGER BANDS EXTENSIONS ===
    df['bb_upper'] = df['Close'].rolling(20).mean() + 2 * df['Close'].rolling(20).std()
    df['bb_lower'] = df['Close'].rolling(20).mean() - 2 * df['Close'].rolling(20).std()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['Close'].rolling(20).mean()
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).quantile(0.2)).astype(int)
    
    return df.dropna()



import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pathlib import Path

MODEL_FILE = Path("btc_trend_model.joblib")

# ------------------------------------------------------------------
# FIXED HAND-PICKED FEATURES (same order every run)
# ------------------------------------------------------------------
FEATS = [
    'rsi', 'macd', 'volatility', 'trend_strength', 'momentum_6',
    'volume_change', 'log_volume', 'ma_6', 'ema_6',
    'bb_percent', 'stoch_k', 'stoch_d', 'vwap', 'obv', 'gap_pct',
    'open_to_high_pct', 'open_to_low_pct', 'candle_body', 'upper_wick', 
    'lower_wick', 'body_to_range', 'price_pos_in_range_6',
]

# ------------------------------------------------------------------
# MAIN TRAIN-AND-SAVE FUNCTION

def train_and_save(csv_path="cached_BTC_sentiment_data5.csv"):
    # 1. -------- Data ------------------------------------------------
    df = (
        pd.read_csv(csv_path, skiprows=[1, 2], index_col=0, parse_dates=True)
          .pipe(new_features2)                     # your custom feature builder
    )

    # Quick sanity-check on required columns
    missing = [f for f in FEATS + ['trend'] if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input DF: {missing}")

    # 2. -------- Clean ------------------------------------------------
    df_clean = df[FEATS + ['trend']].dropna()
    X = df_clean[FEATS].values
    y = df_clean['trend'].values

    # 3. -------- 5-Fold Forward CV -----------------------------------
    n = len(X)
    block = n // 5
    fold_acc = []

    for i in range(5):
        lo, hi = i * block, (i + 1) * block if i < 4 else n

        X_test,  y_test  = X[lo:hi],  y[lo:hi]
        X_train = np.concatenate([X[:lo], X[hi:]])
        y_train = np.concatenate([y[:lo], y[hi:]])

        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        # ------- choose ONE model here -------
        model = SVC(kernel='poly', C=17.3, degree=3) 
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        fold_acc.append(acc)
        print(f"Fold {i+1}: {acc:.4f}")

    avg_acc = np.mean(fold_acc)
    print(f"\nAverage 5-fold accuracy: {avg_acc:.4f}")

    # 4. -------- Train on ALL data -----------------------------------
    scaler_full = StandardScaler().fit(X)
    X_all_s = scaler_full.transform(X)

    final_model = LogisticRegression(max_iter=1_000).fit(X_all_s, y)

    # 5. -------- Persist ---------------------------------------------
    joblib.dump(
        {'model': final_model,
         'scaler': scaler_full,
         'features': FEATS,
         'cv_accuracy': fold_acc,
         'avg_accuracy': avg_acc},
        MODEL_FILE
    )
    print(f"\n✅ Saved model to {MODEL_FILE}")
    print(f"   Using {len(FEATS)} fixed features")

def make_windowed_dataset(X, y, window_size):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X.iloc[i:i+window_size].values.flatten())
        ys.append(y.iloc[i+window_size])  # notice: target is AFTER window
    return np.array(Xs), np.array(ys)

def new_train_and_save():
    # Data loading (same as before)
    df = download_data()
    #df.to_csv("cached_BTC_sentiment_data9.csv", index=True)
    #df = fetch_reddit_sentiment(df)
    #df = df.iloc[2:]  # Remove first row (NaN from shift)
    #print("downloaded data")
    #df.to_csv("cached_BTC_sentiment_data9.csv", index=True)

    # Define strings that should be treated as NaN
    na_values = ['None', '', ' ']

    # Read CSV, skip first two rows, treat specified na_values as NaN
    df = pd.read_csv(
        "cached_BTC_sentiment_data5.csv",
        skiprows=[1, 2],
        index_col=0,
        parse_dates=True,
        na_values=na_values,
        keep_default_na=True,
        skip_blank_lines=True,
    )

    """df = df.iloc[3985:]

    print("read cached data")

    # Drop rows with any NaN values (including those from empty or whitespace cells)
    df.dropna(inplace=True)

    print("dropped na")"""

    #df.to_csv("cached_BTC_sentiment_data16.csv", index=True)

    #df.to_csv("cached_BTC_sentiment_data91.csv", index=True)
    df = generate_features(df) 
    #df.to_csv("cached_BTC_sentiment_data92.csv", index=True)
    df = generate_new_features(df) 
    #df.to_csv("cached_BTC_sentiment_data93.csv", index=True)
    df = new_features2(df, PRED_HOURS=PRED_HOURS)
    #df.to_csv("cached_BTC_sentiment_data94.csv", index=True)
    #df = fetch_web_sentiment(df, query="Bitcoin")
    #df.to_csv("cached_BTC_sentiment_data95.csv", index=True)
    #df = fetch_reddit_sentiment(df)
    #df.to_csv("cached_BTC_sentiment_data99.csv", index=True)

    print("Data loaded and features generated.")

    # Same feature list as your original
    feats = ['ret6', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'rsi', 'macd', # roc
              'volatility', 'trend_strength', 'ma_6', 'ema_6', 'bb_percent', # 'momentum_6', 'log_volume',
              'stoch_k', 'stoch_d', 'vwap', 'obv', 'gap_pct', 'open_to_high_pct', 'open_to_low_pct',
              'candle_body', 'upper_wick', 'lower_wick', 'price_pos_in_range_6',
              'volume_zscore', 'vol_med_ratio', 'rsi_x_vol', 'obv_x_vol', 'ma_slope_6',
              'price_change', 'candle_signal', 'price_acceleration', 'hour', 'vol_kurt', 'ma_envelope_lower', 'ma_envelope_upper', 'price_above_ma'] 
    
    # Optional: keep your correlation plot
    # plot_feature_correlation(df, feats)

    X, y = df[feats], df['trend']
    
    # Clean data (drop NaNs)
    df_clean = df[feats + ['trend']].dropna()
    X = df_clean[feats].values
    y = df_clean['trend'].values
    
    # === 5-FOLD TIME-BLOCKED CROSS VALIDATION ===
    n = len(X)
    block_size = n // 5
    fold_results = []
    
    print("=== 5-FOLD CROSS VALIDATION ===")

    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier as LightGBMClassifier
    
    for i in range(5):
        test_start = i * block_size
        test_end = (i + 1) * block_size if i < 4 else n

        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        X_train = np.concatenate([X[:test_start], X[test_end:]])
        y_train = np.concatenate([y[:test_start], y[test_end:]])

        # Scale features (same as before)
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Same model as your original
        #model = SVC(kernel='linear', C=70)
        model = LogisticRegression(max_iter=1500, solver='saga', C=1.0, penalty='l2')
        #model = XGBClassifier(n_estimators=1700, learning_rate=0.05, max_depth=4, random_state=42)
        #model = LightGBMClassifier(n_estimators=900, max_depth=100, random_state=42)
        #model = RandomForestClassifier(n_estimators=1000, max_depth=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Evaluation
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        fold_results.append(acc)
        print(f"  Fold {i+1}: {acc:.4f}")

    # Average results
    avg_acc = np.mean(fold_results)
    print(f"\nAverage 5-fold accuracy: {avg_acc:.4f}")
    
    # === TRAIN FINAL MODEL ON ALL DATA ===
    scaler_full = StandardScaler().fit(X)
    X_scaled_full = scaler_full.transform(X)
    
    final_model = SVC(kernel='poly', C=17.3, degree=3)
    final_model.fit(X_scaled_full, y)
    
    # Save model + scaler (same as before)
    joblib.dump((final_model, scaler_full), MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Using {len(feats)} features")


def new_train_and_save_stratified():
    # Data loading (same as before)
    df = download_data()
    print("downloaded data")
    df = pd.read_csv("cached_BTC_sentiment_data5.csv", skiprows=[1,2], index_col=0, parse_dates=True)
    df = generate_new_features(df)
    df = new_features2(df)
    print("Data loaded and features generated.")

    # Same feature list as your original
    feats = ['ret6', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'rsi', 'macd', 'roc',
              'volatility', 'trend_strength', 'momentum_6', 'ma_6', 'ema_6', 'bb_percent',
              'stoch_k', 'stoch_d', 'vwap', 'obv', 'gap_pct', 'open_to_high_pct', 'open_to_low_pct',
              'candle_body', 'upper_wick', 'lower_wick', 'price_pos_in_range_6',
              'volume_zscore', 'vol_med_ratio', 'rsi_x_vol', 'obv_x_vol', 'ma_slope_6',
              'price_change', 'candle_signal', 'inverted_hammer_like', 'volatility_breakout'] 
    
    X, y = df[feats], df['trend']
    
    # Clean data (drop NaNs)
    df_clean = df[feats + ['trend']].dropna()
    X = df_clean[feats].values
    y = df_clean['trend'].values
    
    # === ONLY CHANGE: STRATIFIED CROSS VALIDATION ===
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier as LightGBMClassifier
    
    # Replace manual block splitting with StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    
    print("=== STRATIFIED 5-FOLD CROSS VALIDATION ===")
    
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features (same as before)
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Same model as your original
        model = LogisticRegression(max_iter=2000, random_state=42, solver='saga', C=2.0)
        #model = XGBClassifier(n_estimators=1100, max_depth=100, random_state=42)
        #model = SVC(kernel='poly', C=70, degree=1) 
        #model = LightGBMClassifier(n_estimators=900, max_depth=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Evaluation
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        fold_results.append(acc)
        print(f"  Fold {i+1}: {acc:.4f}")

    # Average results
    avg_acc = np.mean(fold_results)
    print(f"\nAverage 5-fold accuracy: {avg_acc:.4f}")
    
    # === TRAIN FINAL MODEL ON ALL DATA ===
    scaler_full = StandardScaler().fit(X)
    X_scaled_full = scaler_full.transform(X)
    
    final_model = SVC(kernel='poly', C=17.3, degree=3)
    final_model.fit(X_scaled_full, y)
    
    # Save model + scaler (same as before)
    joblib.dump((final_model, scaler_full), MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Using {len(feats)} features")


def plot_feature_correlation(df, features, target='trend'):
    corr_df = df[features + [target]].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()

def load_model():
    return joblib.load(MODEL_FILE)

def predict_trend(df_slice, model_scaler=None):
    if model_scaler is None: model, scaler = load_model()
    else: model, scaler = model_scaler
    df_slice = generate_features(df_slice).iloc[[-1]]
    df_slice = fetch_reddit_sentiment(fetch_reddit_sentiment(df_slice))
    X = df_slice[['ret6','ret12','hour_sin','hour_cos','dow_sin','dow_cos','reddit_sent']]
    p = model.predict(scaler.transform(X))[0]
    prob = model.predict_proba(scaler.transform(X))[0][p]
    return int(p), float(prob)

def trend_predictor_indicator(df):
    model, scaler = load_model()
    p, prob = predict_trend(df, (model, scaler))
    return pd.Series([p]*len(df), index=df.index, name='trend_pred')

def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line  # MACD histogram

def compute_bollinger_bandwidth(close, window=20):
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return (upper - lower) / ma

def compute_roc(close, period=12):
    return close.pct_change(period)

def download_few_data():
    df = yf.download(SYMBOL, period="500d", interval="1h", auto_adjust=True).dropna()
    df.index = pd.to_datetime(df.index)
    return df


def use():
    model, scaler = joblib.load(MODEL_FILE)
    print("1")

    df = download_few_data()
    df = generate_features(df)
    df.to_csv("cached_BTC_sentiment_data6.csv")
    df = pd.read_csv("cached_BTC_sentiment_data6.csv", skiprows=[1,2], index_col=0, parse_dates=True)
    df = generate_new_features(df)
    df.to_csv("cached_BTC_sentiment_data7.csv")

    print("2")

    feats = ['ret6', 'ret12', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'rsi', 'sma_cross', 'obv', 'bollinger_bandwidth', 'volatility',
             'ret3', 'log_volume', 'sma_fast', 'sma_slow', 'range', 'ret1', 'momentum_3', 'momentum_6', 'ema_6', 'ema_12', 'ema_26',
             'roc_6', 'roc_12', 'atr_14', 'williams_r', 'cmf', 'stoch_k', 'stoch_d', 'vwap', 'price_change', 'dif', 'prev_future_ret',
             'avg_past_future_ret']

    X = df[feats]
    print("3")

    # If you are using windows in train, likely also needed for test data (adjust depending on use)
    window_size = 5
    X_windowed = []

    for i in range(len(X) - window_size):
        X_windowed.append(X.iloc[i:i+window_size].values.flatten())
    X_windowed = np.array(X_windowed)
    print(X_windowed.shape)

    print("4")

    X_scaled = scaler.transform(X_windowed)
    print("X_scaled shape:", X_scaled.shape)
    predictions = model.predict_proba(X_scaled)

    print("Label distribution in use():")
    print(df['trend'].value_counts())



    print("5")

    import time
    print("Predictions shape:", predictions.shape)
    zer = 0
    one = 0
    for i in range(len(predictions)):
        #print("Predictions example:", predictions[i])
        if predictions[i][0] > predictions[i][1]:
            zer += 1
        else:
            one += 1

    print(zer, one)

    # Further logic to use predictions for trading signals, alerts, or reporting...
 
def train_donttest_and_save():
    df = download_data()
    df = generate_features(df)
    #df = fetch_reddit_sentiment(df)
    #df = generate_new_features(df)
    df.to_csv("cached_BTC_sentiment_data4.csv")
    df = pd.read_csv("cached_BTC_sentiment_data4.csv", skiprows=[1,2], index_col=0, parse_dates=True)
    #df = pd.read_csv("cached_BTC_sentiment_data.csv", skiprows=[1,2], index_col=0, parse_dates=True)
    #df = fetch_reddit_sentiment(df)
    df = generate_new_features(df)
    df.to_csv("cached_BTC_sentiment_data5.csv")
    #df.to_csv("cached_BTC_sentiment_data.csv")
    print("Data loaded and features generated.")
    
    #feats = ['ret6', 'ret12', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'rsi', 'sma_cross', 'bollinger_bandwidth', 'ret3']     
    feats = ['ret6', 'ret12', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'rsi', 'sma_cross', 'obv', 'bollinger_bandwidth', 'volatility', 'ret3', 
             'log_volume', 'sma_fast', 'sma_slow', 'range', 'ret1', 'momentum_3', 'momentum_6', 'ema_6', 'ema_12', 'ema_26', 'roc_6', 'roc_12', 'atr_14',
             'williams_r', 'cmf', 'stoch_k', 'stoch_d', 'vwap', 'price_change', 'dif', 'prev_future_ret', 'avg_past_future_ret'] 

    #plot_feature_correlation(df, feats)

    X, y = df[feats], df['trend']
    X_windowed, y_windowed = make_windowed_dataset(X, y, window_size=5)
    
    n = len(X_windowed)
    block_size = n // 5
    results = []

    from sklearn.linear_model import SGDClassifier
    from sklearn.svm import SVC


    for i in range(5):
        test_start = i * block_size
        test_end = (i + 1) * block_size if i < 4 else n

        X_test = X_windowed[test_start:test_end]
        y_test = y_windowed[test_start:test_end]

        X_train = np.concatenate([X_windowed[:test_start], X_windowed[test_end:]])
        y_train = np.concatenate([y_windowed[:test_start], y_windowed[test_end:]])

        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Final model training after CV (for saving)
        # Use ALL data (no holdout), but do this AFTER CV!
        scaler_full = StandardScaler().fit(X_windowed)
        X_scaled_full = scaler_full.transform(X_windowed)

        final_model = LogisticRegression(max_iter=1000, C=0.2, solver='saga', random_state=42)
        final_model.fit(X_scaled_full, y_windowed)

        joblib.dump((final_model, scaler_full), MODEL_FILE)
        print("✅ Final model trained on full dataset and saved to:", MODEL_FILE)

        # Train model
        #model =  SVC(kernel='poly', C=11.5, degree=3, max_iter=100000) 
        model = LogisticRegression(max_iter=1000, C=0.2, solver='saga', random_state=42)
        #model = XGBClassifier(n_estimators=1350, max_depth=300, learning_rate=0.03, use_label_encoder=False, eval_metric='logloss')
        #model = LightGBM(n_estimators=1000, max_depth=50, learning_rate=0.04)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        print(df['trend'].value_counts(normalize=True))

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        rev_acc = 1 - acc
        print(f"✅ Block {i+1} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        """plt.title(f"Confusion Matrix (Block {i+1})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()"""

        results.append(acc)


    avg_acc = np.mean(results)
    print(f"\n📊 Average Accuracy across 4 time blocks: {avg_acc:.4f}")

    for i in range(100):
        print(y_pred[-100+i] == y_test[-100+i])


def improved_train_and_save_v2():
    """
    Improved function based on successful old code patterns
    """
    import pandas as pd
    import numpy as np
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
    import joblib

    # Data loading
    df = pd.read_csv("cached_BTC_sentiment_data5.csv", skiprows=[1,2], index_col=0, parse_dates=True)
    df = generate_new_features(df)
    df = new_features2(df)
    print("Data loaded and features generated.")

    # IMPROVEMENT 1: DRASTICALLY REDUCE FEATURES (like old code)
    # Select only the most predictive features based on old code success
    core_feats = [
        'ret6',           # Lagged return (core predictor)
        'rsi',            # RSI (proven indicator)
        'macd',           # MACD (proven indicator)
        'volatility',     # Market volatility
        'momentum_6',     # Momentum indicator
        'ma_6',           # Moving average
        'ema_6',          # Exponential moving average
        'stoch_k',        # Stochastic oscillator
        'volume_zscore',  # Volume anomaly detection
        'bb_percent'      # Bollinger band position
    ]

    print(f"Reduced features from 31 to {len(core_feats)}")

    # Clean data
    df_clean = df[core_feats + ['trend']].dropna()
    X = df_clean[core_feats].values
    y = df_clean['trend'].values

    print(f"Dataset size: {len(X)} samples, {X.shape[1]} features")

    # IMPROVEMENT 2: USE TEMPORAL VALIDATION (like old code)
    # Single holdout test maintaining temporal order
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, shuffle=False  # Maintain temporal order
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # IMPROVEMENT 3: ADD PCA DIMENSIONALITY REDUCTION (like old code)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA to reduce to essential components
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"PCA reduced dimensions to: {X_train_pca.shape[1]} components")

    # IMPROVEMENT 4: USE SIMPLE LINEAR SVM (like old code)
    model = SVC(kernel='linear', C=70, probability=True, random_state=42)
    model.fit(X_train_pca, y_train)

    # Evaluation
    y_pred = model.predict(X_test_pca)
    y_proba = model.predict_proba(X_test_pca)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    print(f"\n=== RESULTS ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Brier Score: {brier:.4f}")

    # Check class distribution
    unique, counts = np.unique(y_test, return_counts=True)
    print(f"\nTest set class distribution:")
    for class_val, count in zip(unique, counts):
        print(f"  Class {class_val}: {count} ({count/len(y_test)*100:.1f}%)")

    # IMPROVEMENT 5: TRAIN FINAL MODEL ON ALL DATA
    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X)
    pca_full = PCA(n_components=0.95)
    X_pca_full = pca_full.fit_transform(X_scaled_full)

    final_model = SVC(kernel='linear', C=70, probability=True, random_state=42)
    final_model.fit(X_pca_full, y)

    # Save model with all components
    model_data = {
        'model': final_model,
        'scaler': scaler_full,
        'pca': pca_full,
        'features': core_feats,
        'accuracy': accuracy,
        'auc': auc,
        'brier': brier
    }

    joblib.dump(model_data, MODEL_FILE)
    print(f"\nâœ… Model saved to: {MODEL_FILE}")
    print(f"âœ… Using {len(core_feats)} core features")
    print(f"âœ… PCA components: {X_pca_full.shape[1]}")

    return accuracy, auc, brier

#acc, auc, brier = improved_train_and_save_v2()
#print(acc, auc, brier)

new_train_and_save()

def improved_train_and_save_with_dim_reduction():
    # Load data
    df = pd.read_csv("cached_BTC_sentiment_data5.csv", skiprows=[1,2], index_col=0, parse_dates=True)
    df = new_features2(df)
    print("Data loaded and features generated.")
    
    # === 1. INITIAL CLEANUP ===
    all_features = [col for col in df.columns if col not in ['trend', 'future_ret'] and df[col].dtype in ['float64', 'int64']]
    exclude_features = ['Unnamed: 0', 'Close', 'Open', 'High', 'Low', 'Volume']
    feats = [f for f in all_features if f not in exclude_features]
    print(f"Starting features: {len(feats)}")
    
    # === 2. AGGRESSIVE FEATURE FILTERING (Multi-Stage) ===
    df_clean = df[feats + ['trend']].dropna()
    
    feats_var = feats
    print(f"After variance filtering: {len(feats_var)} features")
    
    # Stage 2: Remove highly correlated features (redundancy removal)
    corr_matrix = df_clean[feats_var].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    feats_decorr = [f for f in feats_var if f not in high_corr_features]
    print(f"After correlation filtering (>0.95): {len(feats_decorr)} features")
    
    # Stage 3: Target correlation filtering
    correlations = {feat: abs(df_clean[feat].corr(df_clean['trend'])) for feat in feats_decorr}
    sorted_feats = sorted(correlations.items(), key=lambda x: -x[1])
    
    # More aggressive threshold based on research findings[1][3]
    threshold = 0.02  # Keep only features with >2% correlation
    high_corr_features = [feat for feat, corr in correlations.items() if corr >= threshold]
    print(f"After target correlation filtering (>{threshold}): {len(high_corr_features)} features")
    
    # === 3. ADVANCED MULTI-METHOD FEATURE SELECTION ===
    from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel, mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LassoCV
    from lightgbm import LGBMClassifier
    
    X_temp = df_clean[high_corr_features]
    y_temp = df_clean['trend']
    
    # Method 1: Statistical (F-test + Mutual Information)
    n_select = min(25, len(high_corr_features))
    f_selector = SelectKBest(f_classif, k=n_select)
    f_selector.fit(X_temp, y_temp)
    f_selected = [high_corr_features[i] for i in f_selector.get_support(indices=True)]
    
    mi_selector = SelectKBest(mutual_info_classif, k=n_select)
    mi_selector.fit(X_temp, y_temp)
    mi_selected = [high_corr_features[i] for i in mi_selector.get_support(indices=True)]
    
    # Method 2: Tree-based importance (Random Forest + Extra Trees)
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_selector.fit(X_temp, y_temp)
    rf_importances = rf_selector.feature_importances_
    rf_threshold = np.percentile(rf_importances, 70)  # Top 30%
    rf_selected = [feat for feat, imp in zip(high_corr_features, rf_importances) if imp >= rf_threshold]
    
    et_selector = ExtraTreesClassifier(n_estimators=100, random_state=42)
    et_selector.fit(X_temp, y_temp)
    et_importances = et_selector.feature_importances_
    et_threshold = np.percentile(et_importances, 70)
    et_selected = [feat for feat, imp in zip(high_corr_features, et_importances) if imp >= et_threshold]
    
    # Method 3: L1 Regularization (Lasso)
    lasso_selector = SelectFromModel(LassoCV(cv=5, random_state=42), threshold='median')
    lasso_selector.fit(X_temp, y_temp)
    lasso_selected = [feat for feat, selected in zip(high_corr_features, lasso_selector.get_support()) if selected]
    
    # Method 4: RFE (Recursive Feature Elimination)
    rfe_selector = RFE(LogisticRegression(max_iter=1000, random_state=42), n_features_to_select=min(20, len(high_corr_features)))
    rfe_selector.fit(X_temp, y_temp)
    rfe_selected = [feat for feat, selected in zip(high_corr_features, rfe_selector.support_) if selected]
    
    # === 4. ENSEMBLE FEATURE SELECTION (VOTING) ===
    # Count how many methods selected each feature
    all_selected = f_selected + mi_selected + rf_selected + et_selected + lasso_selected + rfe_selected
    feature_votes = {}
    for feat in set(all_selected):
        feature_votes[feat] = sum([
            feat in f_selected,
            feat in mi_selected, 
            feat in rf_selected,
            feat in et_selected,
            feat in lasso_selected,
            feat in rfe_selected
        ])
    
    # Select features that got votes from at least 3 methods (majority voting)
    min_votes = 3
    final_features = [feat for feat, votes in feature_votes.items() if votes >= min_votes]
    
    # If too few features, lower the threshold
    if len(final_features) < 10:
        min_votes = 2
        final_features = [feat for feat, votes in feature_votes.items() if votes >= min_votes]
    
    print(f"\n=== ENSEMBLE FEATURE SELECTION RESULTS ===")
    print(f"Features selected by ≥{min_votes} methods: {len(final_features)}")
    
    # Show top features by votes
    sorted_by_votes = sorted(feature_votes.items(), key=lambda x: -x[1])
    print(f"\nTop 15 features by method consensus:")
    for feat, votes in sorted_by_votes[:15]:
        print(f"  {feat:30} {votes} votes")
    
    # === 5. OPTIONAL: PCA AS FINAL STEP (only if still too many features) ===
    use_pca = len(final_features) > 30  # Only if still high-dimensional
    
    if use_pca:
        print(f"\n=== APPLYING PCA (features > 30) ===")
        from sklearn.decomposition import PCA
        
        # Use PCA to reduce to ~20 components explaining 90% variance
        pca = PCA(n_components=0.99, random_state=42)
        X_temp_final = df_clean[final_features]
        pca.fit(X_temp_final)
        
        n_components = pca.n_components_
        print(f"PCA components needed for 90% variance: {n_components}")
        
        # If PCA reduces significantly, use it
        if n_components < len(final_features) * 0.7:  # At least 30% reduction
            use_pca_final = True
            print("✅ Using PCA for final dimensionality reduction")
        else:
            use_pca_final = False
            print("❌ PCA doesn't provide enough reduction, keeping feature selection")
    else:
        use_pca_final = False
        pca = None
    
    # === 6. PREPARE FINAL DATASET ===
    if use_pca_final:
        X = df_clean[final_features]
        # We'll apply PCA during cross-validation to avoid data leakage
        features_for_model = final_features
        print(f"Final approach: Feature Selection → PCA ({len(final_features)} → ~{pca.n_components_})")
    else:
        X = df_clean[final_features]
        features_for_model = final_features
        print(f"Final approach: Feature Selection only ({len(final_features)} features)")
    
    y = df_clean['trend']
    
    # === 7. WINDOWING ===
    X_windowed, y_windowed = make_windowed_dataset(X, y, window_size=10)
    
    # === 8. CROSS-VALIDATION WITH DIMENSIONALITY REDUCTION ===
    n = len(X_windowed)
    block_size = n // 5
    results = []
    
    # Use fewer, more robust models
    models = {
        #'SVC': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, C=0.1, solver='saga', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42, verbose=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42),
        'SVC linear': SVC(kernel='linear', C=70.0, probability=True, random_state=42),
        'SVC poly': SVC(kernel='poly', C=12.0, degree=3, probability=True, random_state=42),

    }
    
    model_results = {}
    
    for model_name, model in models.items():
        print(f"\n=== TRAINING {model_name} ===")
        fold_results = []
        
        for i in range(5):
            test_start = i * block_size
            test_end = (i + 1) * block_size if i < 4 else n

            X_test = X_windowed[test_start:test_end]
            y_test = y_windowed[test_start:test_end]
            X_train = np.concatenate([X_windowed[:test_start], X_windowed[test_end:]])
            y_train = np.concatenate([y_windowed[:test_start], y_windowed[test_end:]])

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Apply PCA if selected (within CV to avoid leakage)
            if use_pca_final:
                pca_fold = PCA(n_components=0.90, random_state=42)
                X_train_scaled = pca_fold.fit_transform(X_train_scaled)
                X_test_scaled = pca_fold.transform(X_test_scaled)

            # Train model
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            fold_results.append(acc)
            print(f"  Fold {i+1}: {acc:.4f}")

        avg_acc = np.mean(fold_results)
        model_results[model_name] = avg_acc
        print(f"  Average: {avg_acc:.4f}")
    
    # === 9. FINAL RESULTS ===
    print(f"\n=== DIMENSIONALITY REDUCTION SUMMARY ===")
    print(f"Original features: {len(feats)}")
    print(f"After filtering: {len(final_features)}")
    if use_pca_final:
        print(f"After PCA: ~{pca.n_components_}")
    print(f"Reduction: {len(feats)} → {len(final_features)} ({(1-len(final_features)/len(feats))*100:.1f}% reduction)")
    
    print(f"\n=== MODEL PERFORMANCE ===")
    for model_name, acc in sorted(model_results.items(), key=lambda x: -x[1]):
        print(f"{model_name:20} {acc:.4f}")
    
    # === 10. SAVE BEST MODEL ===
    best_model_name = max(model_results.items(), key=lambda x: x[1])[0]
    best_model = models[best_model_name]
    
    # Train final model on all data
    scaler_final = StandardScaler()
    X_scaled_final = scaler_final.fit_transform(X_windowed)
    
    if use_pca_final:
        pca_final = PCA(n_components=0.90, random_state=42)
        X_scaled_final = pca_final.fit_transform(X_scaled_final)
    else:
        pca_final = None
    
    best_model.fit(X_scaled_final, y_windowed)
    
    # Save with all preprocessing components
    joblib.dump({
        'model': best_model,
        'scaler': scaler_final,
        'pca': pca_final,
        'features': final_features,
        'feature_votes': feature_votes,
        'performance': model_results,
        'dimensionality_reduction': {
            'original_features': len(feats),
            'final_features': len(final_features),
            'use_pca': use_pca_final,
            'pca_components': pca_final.n_components_ if pca_final else None
        }
    }, MODEL_FILE)
    
    print(f"✅ Best model ({best_model_name}) saved with dimensionality reduction pipeline")
    
    return model_results, final_features, feature_votes

# === USAGE ===
"""results, features, votes = improved_train_and_save_with_dim_reduction()
print("\n=== FINAL RESULTS ===")
for model_name, acc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"{model_name:20} {acc:.4f}")"""
