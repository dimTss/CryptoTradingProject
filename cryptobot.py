import requests
import ccxt
import datetime
import time
from ta.volume import OnBalanceVolumeIndicator

# Setup
bot_token = '7008762247:AAE6h63UBuQ95o3M1aGvYwzYYxo-L_Tu5OY'
chat_ids = ['7219843078', '8008570616', '6274831312', '6242030754']
symbol = 'BTC/USDT'
exchange = ccxt.binance()

success = {
    "lm10": 0, "sm10": 0,
    "lm30": 0, "sm30": 0,
    "lh1": 0,  "sh1": 0,
    "lh2": 0,  "sh2": 0,
    "lh3": 0,  "sh3": 0,
    "lh6": 0,  "sh6": 0,
    "lh12": 0, "sh12": 0,
    "lh24": 0, "sh24": 0,
}

total = {
    "lm10": 0, "sm10": 0,
    "lm30": 0, "sm30": 0,
    "lh1": 0,  "sh1": 0,
    "lh2": 0,  "sh2": 0,
    "lh3": 0,  "sh3": 0,
    "lh6": 0,  "sh6": 0,
    "lh12": 0, "sh12": 0,
    "lh24": 0, "sh24": 0,
}

countdown = {
    "lm10": [], "sm10": [],
    "lm30": [], "sm30": [],
    "lh1": [],  "sh1": [],
    "lh2": [],  "sh2": [],
    "lh3": [],  "sh3": [],
    "lh6": [],  "sh6": [],
    "lh12": [], "sh12": [],
    "lh24": [], "sh24": [],
}

value = {
    "lm10": [], "sm10": [],
    "lm30": [], "sm30": [],
    "lh1": [],  "sh1": [],
    "lh2": [],  "sh2": [],
    "lh3": [],  "sh3": [],
    "lh6": [],  "sh6": [],
    "lh12": [], "sh12": [],
    "lh24": [], "sh24": [],
}

ltimevalues = [    
    "lm10", 
    "lm30", 
    "lh1",  
    "lh2",  
    "lh3", 
    "lh6",  
    "lh12", 
    "lh24"
]

stimevalues = [    
    "sm10",
    "sm30",
    "sh1",
    "sh2",
    "sh3",
    "sh6",
    "sh12",
    "sh24"
]

keytotime = {
    "lm10": 10, 
    "sm10": 10,
    "lm30": 30, 
    "sm30": 30,
    "lh1": 60,  
    "sh1": 60,
    "lh2": 120, 
    "sh2": 120,
    "lh3": 180, 
    "sh3": 180,
    "lh6": 360, 
    "sh6": 360,
    "lh12": 720, 
    "sh12": 720,
    "lh24": 1440, 
    "sh24": 1440
}

def print_stats():
    global ltimevalues, stimevalues, success, total, statscountdown

    message = "\n\n--- Long Signal Stats ---"
    for timevalue in ltimevalues:
        if total[timevalue] > 0:
            success_rate = (success[timevalue] / total[timevalue]) * 100
            message += f"\nLong signal {timevalue}: {success[timevalue]} / {total[timevalue]} ({success_rate:.2f}%)"
        else:
            message += f"\nLong signal {timevalue}: No data yet"

    message += "\n--- Short Signal Stats ---"
    for timevalue in stimevalues:
        if total[timevalue] > 0:
            success_rate = (success[timevalue] / total[timevalue]) * 100
            message += f"\nShort signal {timevalue}: {success[timevalue]} / {total[timevalue]} ({success_rate:.2f}%)"
        else:
            message += f"\nShort signal {timevalue}: No data yet"

    message += "\n--------------------\n"
    print(message)
    
    broadcast_message(message)
    statscountdown = 1440

# ── 1.  Improved RSI helper ────────────────────────────────────────────────────
def calculate_rsi(close_prices, period=14):
    if len(close_prices) < period + 1:
        return 50
    deltas = np.diff(close_prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = [100 - 100 / (1 + rs)]
    for delta in deltas[period:]:
        up_val, down_val = max(delta, 0), -min(delta, 0)
        up = (up * (period - 1) + up_val) / period
        down = (down * (period - 1) + down_val) / period
        rs = up / down if down != 0 else 0
        rsi.append(100 - 100 / (1 + rs))
    return rsi[-1]

# ── 2.  Core pattern detectors (feedback-based) ────────────────────────────────
def _classic_bullish_div(data):
    ok = (
        25 <= data['rsi_5m'] <= 35 and            # oversold RSI window
        data['price'] < data['price_past'] and    # price lower-low
        data['rsi_5m'] > data['rsi_5m_past']      # RSI higher-low
    )
    if not ok: return False, []
    cond = ['RSI bullish divergence']
    if data['stoch_rsi'] < 20 and data['stoch_k'] > data['stoch_d']:
        cond.append('StochRSI hook-up')
    if data['obv_current'] > data['obv_past']:
        cond.append('OBV positive divergence')
    if data['volume'] > 1.5 * data['avg_volume']:
        cond.append('Volume climax')
    return len(cond) >= 2, cond

def _hidden_bullish_div(data):
    ok = (
        data['price'] > data['price_past'] and     # higher-low price (up-trend)
        data['rsi_5m'] < data['rsi_5m_past'] < 45 # lower-low RSI
    )
    if not ok: return False, []
    cond = ['Hidden bullish divergence']
    if data['obv_current'] >= data['obv_past']:
        cond.append('OBV rising')
    if data['volume'] < data['avg_volume']:
        cond.append('Low-volume pullback')
    if data['rsi_5m'] < 35:
        cond.append('RSI oversold in up-trend')
    return len(cond) >= 2, cond

def _accumulation(data):
    flat = abs(data['price'] - data['price_past']) / data['price_past'] < 0.02
    if not flat: return False, []
    cond = []
    if data['obv_current'] > data['obv_past']:
        cond.append('OBV accumulation')
    if data['cvd_current'] > data['cvd_past']:
        cond.append('CVD buying pressure')
    if 40 <= data['rsi_5m'] <= 50:
        cond.append('RSI neutral')
    if data['volume'] >= 0.8 * data['avg_volume']:
        cond.append('Sustained volume')
    return len(cond) >= 2, cond

def _classic_bearish_div(data):
    ok = (
        70 <= data['rsi_5m'] <= 75 and
        data['price'] > data['price_past'] and
        data['rsi_5m'] < data['rsi_5m_past']
    )
    if not ok: return False, []
    cond = ['RSI bearish divergence']
    if data['stoch_rsi'] > 80 and data['stoch_k'] < data['stoch_d']:
        cond.append('StochRSI rolling over')
    if data['obv_current'] < data['obv_past']:
        cond.append('OBV bearish divergence')
    if data['volume'] < data['volume_past']:
        cond.append('Volume fading')
    return len(cond) >= 2, cond

def _distribution(data):
    if data['price'] < data['price_past']*0.998: return False, []
    cond = []
    if data['obv_current'] < data['obv_past']:
        cond.append('OBV distribution')
    if data['cvd_current'] < data['cvd_past']:
        cond.append('CVD selling pressure')
    if 60 <= data['rsi_5m'] <= 70:
        cond.append('RSI high/flat')
    if data['volume'] > data['avg_volume']:
        cond.append('High volume sell-off')
    return len(cond) >= 2, cond

def _blowoff_top(data):
    cond = []
    if data['rsi_5m'] > 85: cond.append('RSI >85')
    if data['volume'] > 3 * data['avg_volume']: cond.append('3× volume spike')
    if (data['price']-data['price_past'])/data['price_past'] > 0.05: cond.append('Parabolic 5%+')
    if data['rsi_5m'] < data['rsi_5m_past'] and data['price'] > data['price_past']:
        cond.append('Divergence in spike')
    return (len(cond) >= 3), cond



def update_stats(current_price):
    global ltimevalues, stimevalues, countdown, value, success, total, statscountdown

    showStats = False

    for timevalue in ltimevalues + stimevalues:
        if len(countdown[timevalue]) > 0:
            if countdown[timevalue][0] > 1:
                countdown[timevalue] = [x - 1 for x in countdown[timevalue]]
            else:
                # Signal matured
                countdown[timevalue].pop(0)
                value_at_signal = value[timevalue].pop(0)
                total[timevalue] += 1

                if (timevalue[0] == 'l' and current_price > value_at_signal) or \
                   (timevalue[0] == 's' and current_price < value_at_signal):
                    success[timevalue] += 1

                # Decrement remaining countdowns
                countdown[timevalue] = [x - 1 for x in countdown[timevalue]]
                showStats = True

    statscountdown -= 1

    if statscountdown <= 0:
        print_stats()


def get_24h_high_low():
    now = exchange.milliseconds()
    since = now - 24 * 60 * 60 * 1000

    chunk_size = 60 * 60 * 1000  # 1 hour
    high_price = float('-inf')
    low_price = float('inf')
    high_ts = None
    low_ts = None

    for chunk_start in range(since, now, chunk_size):
        chunk_end = min(chunk_start + chunk_size, now)
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, '1m', since=chunk_start, limit=1000)
        except Exception as e:
            print(f"Error fetching chunk {chunk_start}: {e}")
            time.sleep(5)
            continue

        for candle in ohlcv:
            high, low, ts = candle[2], candle[3], candle[0]

            if high > high_price:
                high_price = high
                high_ts = ts

            if low < low_price:
                low_price = low
                low_ts = ts

        time.sleep(1.5)  # Avoid rate limit + give memory time to settle

    return (high_price, high_ts), (low_price, low_ts)

# Initialize with last 24h high/low and their timestamps
(high_price, high_ts), (low_price, low_ts) = get_24h_high_low()

def check_rsi_oversold(rsi_5m, rsi_15m):
    return rsi_5m < 20 and (rsi_15m is None or rsi_15m < 30)

def check_bollinger_band_touch(price, lower_band):
    return price <= lower_band

def check_volume_spike(current_volume, avg_volume):
    return current_volume > 1.5 * avg_volume

def check_obv_divergence(obv_15m_trend):
    return obv_15m_trend == 'rising'

def check_cvd_exhaustion(cvd_trend, cvd_value):
    return cvd_trend in ['flattening', 'up'] and cvd_value < 0

def check_stoch_rsi_hook_up(stoch_k, stoch_d):
    return stoch_k > 20 and stoch_k > stoch_d



def evaluate_trade_continuation(data, signal_state):
    """Assess continuation, weakening, or invalidation of the current signal."""
    if not signal_state['active']:
        return None, []

    direction = signal_state['type']
    entry_price = signal_state['entry_price']
    conditions = []

    # Long continuation
    if direction == 'long':
        if data['rsi_5m'] > 50 and data['obv_15m_trend'] == 'rising' and data['volume'] > data['avg_volume']:
            conditions.append('RSI rising')
            conditions.append('OBV increasing')
            return 'continuation', conditions

        elif data['rsi_5m'] < 40 or data['obv_15m_trend'] == 'flat' or data['volume'] < data['avg_volume']:
            conditions.append('RSI fading')
            conditions.append('Volume dropping')
            return 'weakening', conditions

        elif data['price'] < entry_price * 0.98:
            return 'invalidation', ['Price broke below entry-level']
    
    # Short continuation
    elif direction == 'short':
        if data['rsi_5m'] < 50 and data['obv_15m_trend'] == 'falling' and data['volume'] > data['avg_volume']:
            conditions.append('RSI falling')
            conditions.append('OBV dropping')
            return 'continuation', conditions

        elif data['rsi_5m'] > 60 or data['obv_15m_trend'] == 'flat' or data['volume'] < data['avg_volume']:
            conditions.append('RSI bouncing')
            conditions.append('Volume weakening')
            return 'weakening', conditions

        elif data['price'] > entry_price * 1.02:
            return 'invalidation', ['Price broke above short entry-level']

    return None, []

import numpy as np
import ta
import pandas as pd
import random

def calculate_bollinger_bands(close_prices, period=20, std_dev=2):
    ma = np.mean(close_prices[-period:])
    std = np.std(close_prices[-period:])
    upper = ma + std_dev * std
    lower = ma - std_dev * std
    return upper, lower

def calculate_stoch_rsi(close_prices, period=14):
    rsi_vals = [calculate_rsi(close_prices[-(period + i):], period) for i in range(14)]
    rsi_series = np.array(rsi_vals)
    lowest = rsi_series.min()
    highest = rsi_series.max()
    stoch_rsi = (rsi_series[-1] - lowest) / (highest - lowest) * 100 if highest != lowest else 0
    return stoch_rsi

def detect_short_signal(data):
    conditions = []

    # --- RSI(14) Overbought + OBV Bearish Divergence + Volume Climax ---
    if data.get('rsi_5m', 0) > 67.5:
        conditions.append(f'RSI 5m = {data["rsi_5m"]:.1f} (overbought)')

        if (
            data.get('obv_current') is not None and
            data.get('obv_past') is not None and
            data.get('price') is not None and
            data.get('price_past') is not None and
            data['obv_current'] < data['obv_past'] and
            data['price'] > data['price_past']
        ):
            conditions.append('+ OBV Divergence')

        if data.get('volume') and data.get('volume_past') and data['volume'] > data['volume_past']:
            conditions.append('+ Volume Climax')

    # --- StochRSI + BB Break + Sell Wall ---
    if (
        data.get('stoch_rsi', 0) > 0.9 and
        data.get('price') > data.get('bb_upper', float('inf')) and
        data.get('sell_wall')
    ):
        conditions.append('StochRSI>90 + BB Break + Sell Wall')

    # --- CVD Bearish Divergence + Whale Inflows ---
    if (
        data.get('cvd_current') is not None and
        data.get('cvd_past') is not None and
        data['cvd_current'] < data['cvd_past'] and
        data.get('price') >= data.get('price_past', 0)
    ):
        inflows = data.get('exchange_inflows')
        if inflows and len(inflows) >= 10:
            if inflows[-1] > np.mean(inflows[-10:]):
                conditions.append('CVD Bearish Div + Whale Inflows')

    # --- MACD Bearish Cross + Volume Drop ---
    if (
        data.get('macd') is not None and
        data.get('macd_signal') is not None and
        data.get('volume') is not None and
        data.get('volume_past') is not None and
        data['macd'] < data['macd_signal'] and
        data['volume'] < data['volume_past']
    ):
        conditions.append('MACD Bearish Cross + Volume↓')

    # --- Final Signal Decision ---
    if len(conditions) >= 2:
        return True, conditions
    return False, []


def detect_long_signal(data):
    conditions = []

    # --- RSI(14) Oversold + OBV Bullish Divergence + Volume Spike ---
    if data.get('rsi_5m', 100) < 37.5:
        conditions.append(f'RSI 5m = {data["rsi_5m"]:.1f} (oversold)')

        if (
            data.get('obv_current') is not None and
            data.get('obv_past') is not None and
            data.get('price') is not None and
            data.get('price_past') is not None and
            data['obv_current'] > data['obv_past'] and
            data['price'] < data['price_past']
        ):
            conditions.append('+ OBV Divergence')

        if data.get('volume') and data.get('volume_past') and data['volume'] > data['volume_past']:
            conditions.append('+ Volume Spike')

    # --- StochRSI + Bollinger Lower Band + Buy Wall ---
    if (
        data.get('stoch_rsi', 1) < 0.1 and
        data.get('price') <= data.get('bb_lower', 0) and
        data.get('buy_wall')
    ):
        conditions.append('StochRSI<10 + BB Touch + Buy Wall')

    # --- CVD Bullish Divergence + Whale Outflows ---
    if (
        data.get('cvd_current') is not None and
        data.get('cvd_past') is not None and
        data['cvd_current'] > data['cvd_past'] and
        data.get('price') < data.get('price_past', float('inf'))
    ):
        outflows = data.get('exchange_outflows')
        if outflows and len(outflows) >= 10:
            if outflows[-1] > np.mean(outflows[-10:]):
                conditions.append('CVD Bullish Div + Whale Outflows')

    # --- RSI(2) Oversold + MACD Bullish Cross + ADX Down ---
    if (
        data.get('rsi_2', 100) < 5 and
        data.get('macd') > data.get('macd_signal') and
        data.get('adx') < data.get('adx_past', 100)
    ):
        conditions.append('RSI(2)<5 + MACD Cross + ADX↓')

    # --- Final Signal Decision ---
    if len(conditions) >= 2:
        return True, conditions
    return False, []


def format_ts(ts):
    return datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S')

def broadcast_message(message):
    message += f"\nCurrent Price: {current_price:.2f} USDT"
    #message += f"\n24H High: {high_price:.2f} USDT at {format_ts(high_ts)}"
    #message += f"\n24H Low: {low_price:.2f} USDT at {format_ts(low_ts)}"
    message += f"\n5min RSI: {rsi_5m:.1f}, 15min RSI: {rsi_15m:.1f}"
    message += f"\n5min StochRSI: {stoch_rsi:.1f}"
    message += f"\n5min Volume: {volume:.2f}"
    message += f"\nOBV Value: {obv.iloc[-1]} OBV Trend: {obv_trend}"
    message += f"\nCVD Trend: {cvd_trend}, CVD Value: {cvd_value:.2f}"
    for cid in chat_ids:
        try:
            requests.get(f"https://api.telegram.org/bot{bot_token}/sendMessage", params={
                'chat_id': cid,
                'text': message
            })
        except Exception as e:
            print(f"Failed to send message to {cid}: {e}")

def detect_trade_signal(data):
    long_signal, long_conditions = detect_long_signal(data)
    short_signal, short_conditions = detect_short_signal(data)
    return long_signal, long_conditions, short_signal, short_conditions


def detect_buy_wall(df):
    return random.choice([True, False])  # or add logic based on volume or order book data

def detect_sell_wall(df):
    return random.choice([True, False])

longcooldown = 0
shortcooldown = 0
statscountdown = 1440

while True:
    try:
        now_ts = exchange.milliseconds()

        # Fetch OHLCV data
        df_1m = pd.DataFrame(exchange.fetch_ohlcv(symbol, '1m', limit=100), columns=["timestamp", "open", "high", "low", "close", "volume"])
        df_5m = pd.DataFrame(exchange.fetch_ohlcv(symbol, '5m', limit=100), columns=["timestamp", "open", "high", "low", "close", "volume"])
        df_15m = pd.DataFrame(exchange.fetch_ohlcv(symbol, '15m', limit=100), columns=["timestamp", "open", "high", "low", "close", "volume"])

        current_price = df_1m.iloc[-1]['close']
        close_5m = df_5m['close'].values
        close_15m = df_15m['close'].values
        volume_5m = df_5m['volume'].values

        # Indicators (manual calculations)
        rsi_5m = calculate_rsi(close_5m)
        rsi_15m = calculate_rsi(close_15m)
        bb_upper, bb_lower = calculate_bollinger_bands(close_5m)
        volume = volume_5m[-1]
        avg_volume = np.mean(volume_5m[-21:-1])  # last 20 candles, excluding the current

        # Placeholders for OBV and CVD logic
        obv = OnBalanceVolumeIndicator(close=df_15m["close"], volume=df_15m["volume"]).on_balance_volume()
        obv_trend = "rising" if obv.iloc[-1] > obv.iloc[-5] else "falling"
        delta = df_15m["close"].diff()
        buy_volume = pd.Series(np.where(delta > 0, df_15m["volume"], 0), index=df_15m.index)
        sell_volume = pd.Series(np.where(delta < 0, df_15m["volume"], 0), index=df_15m.index)
        cvd = (buy_volume - sell_volume).cumsum()

        cvd_value = cvd.iloc[-1]  # current net value

        # Simple CVD trend logic
        if cvd.iloc[-1] > cvd.iloc[-5]:
            cvd_trend = "up"
        elif cvd.iloc[-1] < cvd.iloc[-5]:
            cvd_trend = "down"
        else:
            cvd_trend = "flattening"

        stoch_rsi = calculate_stoch_rsi(close_5m)
        stoch_k = stoch_rsi
        stoch_d = stoch_rsi  # No smoothing here for simplicity

        # 24H high/low logic
        if now_ts - high_ts > 24 * 60 * 60 * 1000 or now_ts - low_ts > 24 * 60 * 60 * 1000:
            (high_price, high_ts), (low_price, low_ts) = get_24h_high_low()

        if current_price > high_price and now_ts - high_ts > 15 * 60 * 1000:
            high_price = current_price
            high_ts = now_ts
            broadcast_message(f'New {symbol} 24H HIGH: {current_price:.2f}')

        if current_price < low_price and now_ts - low_ts > 15 * 60 * 1000:
            low_price = current_price
            low_ts = now_ts
            msg = f'New {symbol} 24H LOW: {current_price:.2f}'
            broadcast_message(msg)
        
        signal_state = {
            'type': None,            # 'long' / 'short' / None
            'entry_price': None,
            'entry_time': None,
            'active': False,
            'last_alert': None
        }

        # Check for continuation/weakening/invalidation
        status, status_conditions = evaluate_trade_continuation({
            'price': current_price,
            'rsi_5m': rsi_5m,
            'volume': volume,
            'avg_volume': avg_volume,
            'obv_15m_trend': obv_trend,  # needs real calc
        }, signal_state)

        if status and status != signal_state.get('last_alert'):
            if status == 'continuation':
                msg = f"✅ Continuation: {signal_state['type'].capitalize()} from ${signal_state['entry_price']:.2f} still strong – " + ", ".join(status_conditions)
            elif status == 'weakening':
                msg = f"⚠️ Weakening: {signal_state['type'].capitalize()} trade losing steam – " + ", ".join(status_conditions)
            elif status == 'invalidation':
                msg = f"❌ Invalidation: {signal_state['type'].capitalize()} setup failed – " + ", ".join(status_conditions)
                signal_state['active'] = False  # Close the signal

            broadcast_message(msg)
            signal_state['last_alert'] = status

        # --- Calculate MACD ---
        macd_line = df_5m['close'].ewm(span=12, adjust=False).mean() - df_5m['close'].ewm(span=26, adjust=False).mean()
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()

        # --- ADX (using TA or custom logic) ---
        # If not using TA-Lib, you can use `ta.trend.adx`
        high = df_5m['high']
        low = df_5m['low']
        close = df_5m['close']
        adx = ta.trend.ADXIndicator(high, low, close).adx()

        # --- RSI(2) ---
        rsi_2 = calculate_rsi(close_5m, period=2)  # update your `calculate_rsi()` function to accept `period` arg if needed

        # --- Placeholder Buy/Sell Wall Detection (replace with real logic or use dummy) ---
        buy_wall = detect_buy_wall(df_1m)  # Implement or stub this
        sell_wall = detect_sell_wall(df_1m)  # Implement or stub this

        # --- Exchange Whale Flows (dummy data or real from API) ---
        # You can replace with actual on-chain or book flow data
        exchange_outflows = [random.uniform(10, 100) for _ in range(20)]
        exchange_inflows = [random.uniform(10, 100) for _ in range(20)]

        rsi_2 = calculate_rsi(close_5m, period=2)

        # --- Pack all into data_dict ---
        data_dict = {
            'price': current_price,
            'price_past': df_5m['close'].iloc[-5],
            'rsi_5m': rsi_5m,
            'rsi_15m': rsi_15m,
            'rsi_2': rsi_2,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'volume': volume, 
            'volume_past': df_5m['volume'].iloc[-5],
            'avg_volume': avg_volume,
            'obv_current': obv.iloc[-1],
            'obv_past': obv.iloc[-5],
            'cvd_current': cvd.iloc[-1],
            'cvd_past': cvd.iloc[-5],
            'cvd_trend': cvd_trend,
            'cvd_value': cvd_value,
            'macd': macd_line.iloc[-1],
            'macd_signal': macd_signal.iloc[-1],
            'adx': adx.iloc[-1],
            'adx_past': adx.iloc[-5],
            'stoch_rsi': stoch_rsi,
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'buy_wall': buy_wall,
            'sell_wall': sell_wall,
            'exchange_outflows': exchange_outflows,
            'exchange_inflows': exchange_inflows
        }


        long_signal, long_conditions, short_signal, short_conditions = detect_trade_signal(data_dict)

        update_stats(current_price)

        # === LONG SIGNAL ===
        if longcooldown < 1:
            if long_signal:
                signal_state.update({
                    'type': 'long',
                    'entry_price': current_price,
                    'entry_time': now_ts,
                    'active': True,
                    'last_alert': 'entry'
                })
                msg = f"🔺 Long Signal – {symbol} @ {current_price:.2f}:\n• " + "\n• ".join(long_conditions)
                broadcast_message(msg)
                longcooldown = 3
                for lt in ltimevalues:
                    countdown[lt].append(keytotime[lt])
                    value[lt].append(current_price)
        else:
            longcooldown -= 1

        # === SHORT SIGNAL ===
        if shortcooldown < 1:
            if short_signal:
                signal_state.update({
                    'type': 'short',
                    'entry_price': current_price,
                    'entry_time': now_ts,
                    'active': True,
                    'last_alert': 'entry'
                })
                msg = f"🔻 Short Signal – {symbol} @ {current_price:.2f}:\n• " + "\n• ".join(short_conditions)
                broadcast_message(msg)
                shortcooldown = 3
                for st in stimevalues:
                    countdown[st].append(keytotime[st])
                    value[st].append(current_price)
        else:
            shortcooldown -= 1


        time.sleep(59)

    except Exception as e:
        import traceback
        print("Error occurred:")
        traceback.print_exc()
        time.sleep(10)


