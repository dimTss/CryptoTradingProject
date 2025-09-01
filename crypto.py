import streamlit as st
import ccxt
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import time
st.query_params["refresh"] = int(time.time())

st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title("Real-time Crypto Exchange Dashboard")

st.sidebar.header("🔧 Settings")

if "settings_mode" not in st.session_state:
    st.session_state.settings_mode = False
if "settings_mode_temp" not in st.session_state:
    st.session_state.settings_mode_temp = False

st.session_state.settings_mode_temp = st.sidebar.checkbox("Settings only mode")

st.sidebar.subheader("Technical indicators")

exchange = ccxt.binance({
    'enableRateLimit': True
})

if "base_currency_temp" not in st.session_state:
    st.session_state.base_currency_temp = "BTC"
if "quote_currency_temp" not in st.session_state:
    st.session_state.quote_currency_temp = "USDT"
if "base_currency" not in st.session_state:
    st.session_state.base_currency = "BTC"
if "quote_currency" not in st.session_state:
    st.session_state.quote_currency = "USDT"

# User input for base and quote currencies
st.session_state.base_currency_temp = st.sidebar.text_input("Enter Base Currency (e.g., BTC, ETH)", "BTC").upper()
st.session_state.quote_currency_temp = st.sidebar.text_input("Enter Quote Currency (e.g., USDT, EUR)", "USDT").upper()

# Construct symbol
symbol = f"{st.session_state.base_currency}/{st.session_state.quote_currency}"

# Load available markets from Binance
markets = exchange.load_markets()
available_symbols = list(markets.keys())

if symbol not in available_symbols:
    st.error(f"❌ Trading pair {symbol} not available on Binance.")
    
    # Suggest valid quote currencies for the selected base
    suggested = [s for s in available_symbols if s.startswith(f"{st.session_state.base_currency}/")]
    if suggested:
        st.info("✅ Did you mean one of these?\n\n" + ", ".join(suggested[:10]))
    else:
        st.info("ℹ️ No matching pairs found.")
    
    st.stop()

from datetime import timedelta, datetime, timezone
import pytz

if "selected_temp" not in st.session_state:
    st.session_state.selected_temp = "1 Day"
if "selected" not in st.session_state:
    st.session_state.selected = "1 Day"

timeframes = {
    "1 Day": ("1m", timedelta(days=1)),
    "1 Week": ("5m", timedelta(weeks=1)),
    "1 Month": ("30m", timedelta(days=30)),
    "1 Year": ("1d", timedelta(days=365)),
    "500 Days": ("1d", timedelta(days=500)),
}

st.session_state.selected_temp = st.sidebar.selectbox("Select Time Range", list(timeframes.keys()), index=0)

if "hide_technicals" not in st.session_state:
    st.session_state.hide_technicals = True
if "hide_technicals_temp" not in st.session_state:
    st.session_state.hide_technicals_temp = True

st.session_state.hide_technicals_temp = st.sidebar.checkbox("Binance", value = True)

if "hide_technicals2" not in st.session_state:
    st.session_state.hide_technicals2 = False
if "hide_technicals_temp2" not in st.session_state:
    st.session_state.hide_technicals_temp2 = False

st.session_state.hide_technicals_temp2 = st.sidebar.checkbox("Kraken")

if "hide_technicals3" not in st.session_state:
    st.session_state.hide_technicals3 = False
if "hide_technicals_temp3" not in st.session_state:
    st.session_state.hide_technicals_temp3 = False


if False:
    st.session_state.hide_technicals_temp3 = st.sidebar.checkbox("Bybit")
if "hide_technicals4" not in st.session_state:
    st.session_state.hide_technicals4 = False
if "hide_technicals_temp4" not in st.session_state:
    st.session_state.hide_technicals_temp4 = False


if False:
    st.session_state.hide_technicals_temp4 = st.sidebar.checkbox("Crypto.com")

if "show_bollinger" not in st.session_state:
    st.session_state.show_bollinger = False
if "show_bollinger_temp" not in st.session_state:
    st.session_state.show_bollinger_temp = False

#st.session_state.show_bollinger_temp = st.sidebar.checkbox("Show Bollinger Bands", value = False)

st.sidebar.subheader("Order Book")

# Initialize default values in session_state if not already set
if "small_max_temp" not in st.session_state:
    st.session_state.small_max_temp = 0.5
if "medium_max_temp" not in st.session_state:
    st.session_state.medium_max_temp = 5.0
if "small_max" not in st.session_state:
    st.session_state.small_max = 0.5
if "medium_max" not in st.session_state:
    st.session_state.medium_max = 5.0

# Temporary sliders (buffer values)
st.session_state.small_max_temp = st.sidebar.slider(
    "Max size for Small (Pending)", 0.01, 5.0, st.session_state.small_max_temp
)
st.session_state.medium_max_temp = st.sidebar.slider(
    "Max size for Medium (Pending)", st.session_state.small_max_temp, 50.0, st.session_state.medium_max_temp
)

if "hide_order" not in st.session_state:
    st.session_state.hide_order = False
if "hide_order_temp" not in st.session_state:
    st.session_state.hide_order_temp = False

st.session_state.hide_order_temp = st.sidebar.checkbox("Hide Order Book")

st.sidebar.subheader("ChronoAlpha Analysis")

if "selected_range_label_temp" not in st.session_state:
    st.session_state.selected_range_label_temp = "500 Days"
if "selected_range_label" not in st.session_state:
    st.session_state.selected_range_label = "500 Days"

range_options = {
    "500 Days": timedelta(days=500),
    "1 Year": timedelta(days=365),
    "6 Months": timedelta(days=180),
    "3 Months": timedelta(days=90)
}
st.session_state.selected_range_label_temp = st.sidebar.selectbox("Time Range for Hourly Strength Analysis", list(range_options.keys()))

if "time_gaps_temp" not in st.session_state:
    st.session_state.time_gaps_temp = "30 minutes"
if "time_gaps" not in st.session_state:
    st.session_state.time_gaps = "30 minutes"

time_gap_range_options = {
    "30 minutes": "30m",
    "15 minutes": "15m",
    "1 hour": "1h",
    "2 hours": "2h",
}
st.session_state.time_gaps_temp = st.sidebar.selectbox("Time gaps for Hourly Strength Analysis", list(time_gap_range_options.keys()))

if "hide_ChronoAnalysis" not in st.session_state:
    st.session_state.hide_ChronoAnalysis = False
if "hide_ChronoAnalysis_temp" not in st.session_state:
    st.session_state.hide_ChronoAnalysis_temp = False

st.session_state.hide_ChronoAnalysis_temp = st.sidebar.checkbox("Hide ChronoAlpha")

st.sidebar.subheader("Auto refresh")

if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False
if "auto_refresh_temp" not in st.session_state:
    st.session_state.auto_refresh_temp = False

st.session_state.auto_refresh_temp = st.sidebar.checkbox("Stop auto refresh")


# Button to apply changes
if st.sidebar.button("Apply Settings"):
    st.session_state.settings_mode = st.session_state.settings_mode_temp
    st.session_state.base_currency = st.session_state.base_currency_temp
    st.session_state.quote_currency = st.session_state.quote_currency_temp
    st.session_state.selected = st.session_state.selected_temp
    st.session_state.small_max = st.session_state.small_max_temp
    st.session_state.medium_max = st.session_state.medium_max_temp
    st.session_state.selected_range_label = st.session_state.selected_range_label_temp
    st.session_state.time_gaps = st.session_state.time_gaps_temp
    st.session_state.auto_refresh = st.session_state.auto_refresh_temp

    st.session_state.hide_technicals = st.session_state.hide_technicals_temp
    st.session_state.hide_technicals2 = st.session_state.hide_technicals_temp2
    st.session_state.hide_technicals3 = st.session_state.hide_technicals_temp3
    st.session_state.hide_technicals4 = st.session_state.hide_technicals_temp4
    st.session_state.hide_order = st.session_state.hide_order_temp
    st.session_state.hide_ChronoAnalysis = st.session_state.hide_ChronoAnalysis_temp

def format_duration(td: timedelta) -> str:
    days = td.days
    return f"{days} day{'s' if days != 1 else ''}"

# Display current applied settings
st.sidebar.subheader("Current Settings")
st.sidebar.write(f"")

if st.session_state.hide_technicals:
    st.sidebar.write(f"**Technical Indicators (hidden)**")
else:
    st.sidebar.write(f"**Technical Indicators**")
st.sidebar.write(f"Base Currency: {st.session_state.base_currency}")
st.sidebar.write(f"Quote Currency: {st.session_state.quote_currency}")
selected_resolution, selected_duration = timeframes[st.session_state.selected]
st.sidebar.write(f"Resolution: {selected_resolution}")
st.sidebar.write(f"Duration: {format_duration(selected_duration)}")
st.sidebar.write(f"")

if st.session_state.hide_order:
    st.sidebar.write(f"**Order book (hidden)**")
else:
    st.sidebar.write(f"**Order book**")
st.sidebar.write(f"Small Max: {st.session_state.small_max}")
st.sidebar.write(f"Medium Max: {st.session_state.medium_max}")
st.sidebar.write(f"")

if st.session_state.hide_ChronoAnalysis:
    st.sidebar.write(f"**ChronoAlpha Analysis (hidden)**")
else:
    st.sidebar.write(f"**ChronoAlpha Analysis**")
selected_range = range_options[st.session_state.selected_range_label]
st.sidebar.write(f"Time range: {format_duration(selected_range)}")
selected_gaps = time_gap_range_options[st.session_state.time_gaps]
st.sidebar.write(f"Time range: {st.session_state.time_gaps}")
st.sidebar.write(f"")

if(st.session_state.auto_refresh):
    st.sidebar.write(f"**Auto refresh deactivated**")
else:
    st.sidebar.write(f"**Auto refresh activated**")


if(st.session_state.settings_mode):
    st.subheader("Please disable the 'settings only' mode to access the content")
    st.stop()

def showpercentage(pct: float) -> str:
    color = "green" if pct > 0 else "red"
    sign = "+" if pct > 0 else ""
    return f"<span style='color:{color}'>{sign}{pct:.2f}%</span>"

def adjust_tf_for_kraken(original_tf, delta):
    if delta is None:
        return original_tf
    
    days = delta.days
    if days <= 1:
        return '5m'  # Changed from 2m → 5m for Kraken compatibility
    elif days <= 7:
        return '15m'
    elif days <= 31:
        return '1h'
    else:
        return '1d'
        
athens_tz = pytz.timezone("Europe/Athens")
current_offset = athens_tz.utcoffset(datetime.utcnow())
fixed_offset = timezone(current_offset)

import requests

def fetch_binance_ohlcv_rest(symbol, interval, start_ms, end_ms):
    limit = 1000
    all_klines = []
    base_url = "https://api.binance.com/api/v3/klines"

    while start_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }

        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Binance API error: {response.text}")
            time.sleep(5)
            continue

        klines = response.json()
        if not klines:
            break

        all_klines.extend(klines)
        last_time = klines[-1][0]
        start_ms = last_time + 1
        time.sleep(0.6)  # respect rate limits

    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])

    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    df = df.astype({
        'timestamp': 'int64',
        'open': 'float',
        'high': 'float',
        'low': 'float',
        'close': 'float',
        'volume': 'float'
    })

    return df

def fetch_all_ohlcv(exchange, symbol, timeframe, since, to):
    all_ohlcv = []

    interval_ms = exchange.parse_timeframe(timeframe) * 1000
    max_candles = 720 if exchange.id == 'kraken' else 1000

    while since < to:
        try:
            print(f"Fetching {exchange.id} {symbol} from {datetime.utcfromtimestamp(since/1000)}")
            limit = min(max_candles, (to - since) // interval_ms)
            data = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

            if not data:
                break

            all_ohlcv.extend(data)

            last_ts = data[-1][0]

            # Break if we've reached or passed the last timestamp
            if last_ts + interval_ms >= to:
                break

            since = last_ts + interval_ms
            time.sleep(exchange.rateLimit / 1000.0)

        except Exception as e:
            print(f"⚠️ Error while fetching: {e}")
            break

    return all_ohlcv

# --- Signal Detection Rules ---
def detect_trade_signal(df, buy_wall=False, sell_wall=False, exchange_outflows=None, exchange_inflows=None):
    from ta.momentum import RSIIndicator, StochRSIIndicator
    from ta.trend import MACD, ADXIndicator
    from ta.volume import OnBalanceVolumeIndicator
    from ta.volatility import BollingerBands
    import numpy as np
    import pandas as pd

    signals = []

    close = df["close"]
    volume = df["volume"]

    # Indicators
    rsi = RSIIndicator(close=close, window=14).rsi()
    rsi2 = RSIIndicator(close=close, window=2).rsi()
    stoch_rsi = StochRSIIndicator(close=close).stochrsi()
    obv = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    bb = BollingerBands(close=close)
    bb_lower = bb.bollinger_lband()
    bb_upper = bb.bollinger_hband()
    macd = MACD(close=close)
    macd_line = macd.macd()
    macd_signal = macd.macd_signal()
    adx = ADXIndicator(high=df["high"], low=df["low"], close=close).adx()

    # CVD
    delta = close.diff()
    buy_volume = pd.Series(np.where(delta > 0, volume, 0), index=df.index)
    sell_volume = pd.Series(np.where(delta < 0, volume, 0), index=df.index)
    cvd = (buy_volume - sell_volume).cumsum()

    # --- LONG Signals ---
    if rsi.iloc[-1] < 20:
        if obv.iloc[-1] > obv.iloc[-5] and close.iloc[-1] < close.iloc[-5]:
            if volume.iloc[-1] > volume.iloc[-5]:
                signals.append(("LONG", "RSI<20 + OBV Div + Volume Spike", "90–95%"))
            else:
                signals.append(("LONG", "RSI<20 + OBV Bullish Divergence", "85–90%"))

    if stoch_rsi.iloc[-1] < 0.1 and close.iloc[-1] <= bb_lower.iloc[-1]:
        if buy_wall:
            signals.append(("LONG", "StochRSI<10 + BB Touch + Buy Wall", "~90%"))

    if cvd.iloc[-1] > cvd.iloc[-5] and close.iloc[-1] <= close.iloc[-5]:
        if exchange_outflows and exchange_outflows[-1] > np.mean(exchange_outflows[-10:]):
            signals.append(("LONG", "CVD Bullish Divergence + Whale Outflows", "80–90%"))

    if rsi2.iloc[-1] < 5 and macd_line.iloc[-1] > macd_signal.iloc[-1] and adx.iloc[-1] < adx.iloc[-5]:
        signals.append(("LONG", "RSI(2)<5 + MACD Cross + ADX↓", "85%"))

    # --- SHORT Signals ---
    if rsi.iloc[-1] > 80:
        if obv.iloc[-1] < obv.iloc[-5] and close.iloc[-1] > close.iloc[-5]:
            if volume.iloc[-1] > volume.iloc[-5]:
                signals.append(("SHORT", "RSI>80 + OBV Div + Volume Climax", "90–95%"))
            else:
                signals.append(("SHORT", "RSI>80 + OBV Bearish Divergence", "85–90%"))

    if stoch_rsi.iloc[-1] > 0.9 and close.iloc[-1] > bb_upper.iloc[-1]:
        if sell_wall:
            signals.append(("SHORT", "StochRSI>90 + BB Break + Sell Wall", "~90%"))

    if cvd.iloc[-1] < cvd.iloc[-5] and close.iloc[-1] >= close.iloc[-5]:
        if exchange_inflows and exchange_inflows[-1] > np.mean(exchange_inflows[-10:]):
            signals.append(("SHORT", "CVD Bearish Divergence + Whale Inflows", "80–90%"))

    if macd_line.iloc[-1] < macd_signal.iloc[-1] and volume.iloc[-1] < volume.iloc[-5]:
        signals.append(("SHORT", "MACD Bearish Cross + Vol↓", "~85%"))

    # Return all detected signals 
    return signals


if False:
    tf, delta = timeframes[st.session_state.selected]

    now = exchange.milliseconds()
    if delta is not None:
        since = now - int(delta.total_seconds() * 1000)
        ohlcv = fetch_all_ohlcv(exchange, symbol, tf, since, now)
    else:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=5000)

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    if df.empty:
        st.error("⚠️ No data returned for the selected timeframe.")
        st.stop()

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['timestamp'] = df['timestamp'].dt.tz_convert(fixed_offset)


    y_min = 0.99999 * df['low'].min()
    y_max = 1.00001 * df['high'].max()

    latest = df.iloc[-1]
    first = df.iloc[0]
    percentage = (latest['close'] - first['close']) * 100 / first['close']

    fig_price = go.Figure()

    fig_price.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['close'],
        name='Close Price',
        mode='lines',
        line=dict(color='#00cc96')
    ))

    fig_price.update_layout(
        title="Price Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        height=350,
        margin=dict(t=50, b=20),
        xaxis=dict(tickangle=45)
    )


    st.header("Technical Indicators")

    st.plotly_chart(fig_price, use_container_width=True)

    fig_volume = go.Figure()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    now_athens = datetime.now(tz=athens_tz).strftime("%Y-%m-%d %H:%M:%S")
    st.metric(label=f"Price (as of {now_athens} Athens Time)", value=f"{st.session_state.quote_currency} {latest['close']:,.2f}")

    fig_volume.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['volume'],
        name='Volume',
        marker_color='rgba(210, 210, 210, 1)',
        opacity=0.9
    ))

    fig_volume.update_layout(
        title="Volume Chart",
        xaxis_title="Time",
        yaxis_title="Volume",
        template="plotly_dark",
        height=350,
        margin=dict(t=50, b=20),
        xaxis=dict(tickangle=45)
    )

    st.plotly_chart(fig_volume, use_container_width=True)

    vol = sum(df['volume'])
    st.metric(label="Total Volume", value=vol)

    from ta.volume import OnBalanceVolumeIndicator
    import numpy as np

    # --- OBV ---
    obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()

    # --- CVD ---
    df["delta"] = df["close"].diff()
    df["buy_volume"] = np.where(df["delta"] > 0, df["volume"], 0)
    df["sell_volume"] = np.where(df["delta"] < 0, df["volume"], 0)
    df["cvd"] = (df["buy_volume"] - df["sell_volume"]).cumsum()

    # --- Combined OBV + CVD Chart ---
    fig_obv_cvd = go.Figure()

    fig_obv_cvd.add_trace(go.Scatter(
        x=df["timestamp"],
        y=obv,
        name="OBV",
        line=dict(color='aqua')
    ))

    fig_obv_cvd.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["cvd"],
        name="CVD",
        line=dict(color='orange')
    ))

    fig_obv_cvd.update_layout(
        title="On-Balance Volume (OBV) and Cumulative Volume Delta (CVD)",
        template="plotly_dark",
        height=350,
        xaxis_title="Time (Athens Time)",
        yaxis_title="Volume",
        legend=dict(x=0, y=1)
    )

    st.plotly_chart(fig_obv_cvd, use_container_width=True)




    from ta.momentum import RSIIndicator
    # --- RSI ---
    rsi = RSIIndicator(close=df["close"], window=14).rsi()
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df["timestamp"], y=rsi, name="RSI", line=dict(color='lime')))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", annotation_position="top left")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="blue", annotation_text="Oversold", annotation_position="bottom left")
    fig_rsi.update_layout(title="Relative Strength Index (RSI)", template="plotly_dark", height=300)
    st.plotly_chart(fig_rsi, use_container_width=True)
elif st.session_state.hide_technicals or st.session_state.hide_technicals2 or st.session_state.hide_technicals3 or st.session_state.hide_technicals4:
    
    st.header("Technical Indicators")
    exchanges_to_use = {}

    if st.session_state.hide_technicals2:
        exchanges_to_use["Kraken"] = ccxt.kraken()
    if st.session_state.hide_technicals:
        exchanges_to_use["Binance"] = exchange

    #if st.session_state.hide_technicals3:
        #exchanges_to_use["Bybit"] = ccxt.bybit()

    #if st.session_state.hide_technicals4:
        #exchanges_to_use["Crypto.com"] = ccxt.cryptocom()

    from ta.volatility import BollingerBands
    import numpy as np

    # Configurable parameters
    VOLUME_LOOKBACK = 20
    VOLUME_SPIKE_THRESHOLD = 1.5
    BODY_LOOKBACK = 20
    BODY_THRESHOLD = 1.5

    tf, delta = timeframes[st.session_state.selected]
    now = ccxt.binance({
        'enableRateLimit': True
    }).milliseconds()
    dataframes_by_exchange = {}
    bollinger_df = None  # Will hold the DataFrame for Bollinger Bands

    # -------------------- Fetch Data --------------------
    for ex_name, exchange in exchanges_to_use.items():
        exchange_tf = tf
        if ex_name.lower() == "kraken":
            exchange_tf = adjust_tf_for_kraken(tf, delta)

        try:
            since = now - int(delta.total_seconds() * 1000) if delta else now - 30 * 24 * 60 * 60 * 1000

            # Fetch OHLCV data
            if ex_name == "Binance":
                ohlcv = fetch_binance_ohlcv_rest(symbol="BTCUSDT", interval=tf, start_ms=since, end_ms=now)
            else:
                ohlcv = fetch_all_ohlcv(exchange, symbol, exchange_tf, since, now)

            if ohlcv is None or len(ohlcv) == 0:
                continue

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(fixed_offset)

            if df.empty:
                continue

            # Calculate volume and body signals
            df['vol_ma'] = df['volume'].rolling(VOLUME_LOOKBACK).mean()
            df['body'] = (df['close'] - df['open']).abs()
            df['body_ma'] = df['body'].rolling(BODY_LOOKBACK).mean()

            df['is_spike'] = df['volume'] > VOLUME_SPIKE_THRESHOLD * df['vol_ma']
            df['is_big_body'] = df['body'] > BODY_THRESHOLD * df['body_ma']

            # Classify spikes
            df['spike_type'] = np.where(
                df['is_spike'] & df['is_big_body'],
                np.where(df['close'] < df['open'], 'buy_signal', 'sell_warning'),
                None
            )

            dataframes_by_exchange[ex_name] = df

            # Reference DF for Bollinger Bands
            if len(exchanges_to_use) == 1 or (len(exchanges_to_use) > 1 and ex_name == "Binance"):
                bollinger_df = df

        except Exception as e:
            st.warning(f"⚠️ Failed to fetch data from {ex_name}: {e}")

    # -------------------- Final Checks --------------------
    if not dataframes_by_exchange:
        st.error("⚠️ No data returned for any exchange.")
        st.stop()

    # -------------------- Price Chart --------------------
    fig_price = go.Figure()
    price_changes = []

    for ex_name, df in dataframes_by_exchange.items():
        fig_price.add_trace(go.Scatter(
            x=df['timestamp'], y=df['close'],
            name=f'{ex_name} Close', mode='lines'
        ))

        # Add signal markers (filtered)
        buy_spikes = df[df['spike_type'] == 'buy_signal']
        sell_spikes = df[df['spike_type'] == 'sell_warning']

        if not buy_spikes.empty:
            fig_price.add_trace(go.Scatter(
                x=buy_spikes['timestamp'], y=buy_spikes['low'],
                mode='markers',
                name=f'{ex_name} Buy Signal',
                marker=dict(size=10, color='lime', symbol='triangle-up'),
                hovertemplate="BUY Spike<br>Time: %{x}<br>Low: %{y}<br>Vol: %{customdata[0]:,.0f}<extra></extra>",
                customdata=buy_spikes[['volume']].values
            ))

        if not sell_spikes.empty:
            fig_price.add_trace(go.Scatter(
                x=sell_spikes['timestamp'], y=sell_spikes['high'],
                mode='markers',
                name=f'{ex_name} Sell Warning',
                marker=dict(size=10, color='red', symbol='triangle-down'),
                hovertemplate="SELL Spike<br>Time: %{x}<br>High: %{y}<br>Vol: %{customdata[0]:,.0f}<extra></extra>",
                customdata=sell_spikes[['volume']].values
            ))

        # Price % change
        pct_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
        price_changes.append(pct_change)


    # -------------------- Bollinger Bands --------------------
    if bollinger_df is not None and len(bollinger_df) >= 21:
        bb = BollingerBands(close=bollinger_df['close'], window=20, window_dev=2)
        bollinger_df['bb_mavg'] = bb.bollinger_mavg()
        bollinger_df['bb_upper'] = bb.bollinger_hband()
        bollinger_df['bb_lower'] = bb.bollinger_lband()

        suffix = " (Binance)" if len(dataframes_by_exchange) > 1 and bollinger_df.equals(dataframes_by_exchange.get("Binance")) else ""

        fig_price.add_trace(go.Scatter(x=bollinger_df['timestamp'], y=bollinger_df['bb_mavg'],
                                    mode='lines', name=f'BB Mid{suffix}',
                                    line=dict(color='orange', width=1, dash='dash')))
        fig_price.add_trace(go.Scatter(x=bollinger_df['timestamp'], y=bollinger_df['bb_upper'],
                                    mode='lines', name='BB Upper',
                                    line=dict(color='lightgray', width=1)))
        fig_price.add_trace(go.Scatter(x=bollinger_df['timestamp'], y=bollinger_df['bb_lower'],
                                    mode='lines', name='BB Lower',
                                    line=dict(color='lightgray', width=1)))
        fig_price.add_trace(go.Scatter(
            x=pd.concat([bollinger_df['timestamp'], bollinger_df['timestamp'][::-1]]),
            y=pd.concat([bollinger_df['bb_upper'], bollinger_df['bb_lower'][::-1]]),
            fill='toself', fillcolor='rgba(255, 255, 255, 0.05)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=False
        ))

    # -------------------- Price Chart Layout --------------------
    avg_pct_change = price_changes[0] if len(price_changes) == 1 else sum(price_changes) / len(price_changes)
    avg_color = "green" if avg_pct_change >= 0 else "red"
    avg_pct_str = f"{avg_pct_change:+.2f}%"

    fig_price.update_layout(
        title=f"<b>Price Chart</b><br><span style='color:{avg_color}; font-size:16px;'>Change: {avg_pct_str}</span>",
        xaxis_title="Time", yaxis_title="Price",
        template="plotly_dark", height=350,
        margin=dict(t=50, b=20), xaxis=dict(tickangle=45)
    )

    from ta.volume import OnBalanceVolumeIndicator
    import numpy as np

    figs_volume = {}
    figs_obv_cvd = {}

    for ex_name, df in dataframes_by_exchange.items():
        # Volume
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name=f'{ex_name} Volume'
        ))
        fig_volume.update_layout(
            title=f"{ex_name} Volume Chart",
            xaxis_title="Time",
            yaxis_title="Volume",
            template="plotly_dark",
            height=350,
            margin=dict(t=50, b=20),
            xaxis=dict(tickangle=45)
        )
        figs_volume[ex_name] = fig_volume

        from ta.volume import OnBalanceVolumeIndicator
        # OBV, CVD
        df["delta"] = df["close"].diff()
        df["buy_volume"] = np.where(df["delta"] > 0, df["volume"], 0)
        df["sell_volume"] = np.where(df["delta"] < 0, df["volume"], 0)
        df["cvd"] = (df["buy_volume"] - df["sell_volume"]).cumsum()
        obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()

        fig_obv_cvd = go.Figure()
        fig_obv_cvd.add_trace(go.Scatter(x=df["timestamp"], y=obv, name="OBV"))
        fig_obv_cvd.add_trace(go.Scatter(x=df["timestamp"], y=df["cvd"], name="CVD"))
        fig_obv_cvd.update_layout(
            title=f"{ex_name} OBV, CVD",


            template="plotly_dark",
            height=350,
            xaxis_title="Time (Athens Time)",
            yaxis_title="Volume",
            legend=dict(x=0, y=1)
        )
        figs_obv_cvd[ex_name] = fig_obv_cvd


    

    from ta.momentum import RSIIndicator

    # ------------------ RSI Chart ------------------
    fig_rsi = go.Figure()
    for ex_name, df in dataframes_by_exchange.items():
        rsi = RSIIndicator(close=df["close"], window=14).rsi()
        fig_rsi.add_trace(go.Scatter(x=df["timestamp"], y=rsi, name=f"{ex_name} RSI"))

    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", annotation_position="top left")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="blue", annotation_text="Oversold", annotation_position="bottom left")
    fig_rsi.update_layout(
        title="Relative Strength Index (RSI)",
        template="plotly_dark",
        height=300
    )

    from plotly.subplots import make_subplots

    # -- 📈 Combined Price + RSI chart --
    fig_combined = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price", "RSI")
    )

    # Add price line(s)
    for ex_name, df in dataframes_by_exchange.items():
        fig_combined.add_trace(
            go.Scatter(x=df['timestamp'], y=df['close'], mode='lines', name=f'{ex_name} Close'),
            row=1, col=1
        )

        # Buy/Sell markers
        buy_spikes = df[df['spike_type'] == 'buy_signal']
        sell_spikes = df[df['spike_type'] == 'sell_warning']

        if not buy_spikes.empty:
            fig_combined.add_trace(
                go.Scatter(
                    x=buy_spikes['timestamp'], y=buy_spikes['low'],
                    mode='markers',
                    name=f'{ex_name} Buy Signal',
                    marker=dict(size=10, color='lime', symbol='triangle-up'),
                    showlegend=False,
                    customdata=buy_spikes[['volume']].values,
                    hovertemplate="BUY Spike<br>Time: %{x}<br>Low: %{y}<br>Vol: %{customdata[0]:,.0f}<extra></extra>",
                ),
                row=1, col=1
            )

        if not sell_spikes.empty:
            fig_combined.add_trace(
                go.Scatter(
                    x=sell_spikes['timestamp'], y=sell_spikes['high'],
                    mode='markers',
                    name=f'{ex_name} Sell Warning',
                    marker=dict(size=10, color='red', symbol='triangle-down'),
                    showlegend=False,
                    customdata=sell_spikes[['volume']].values,
                    hovertemplate="SELL Spike<br>Time: %{x}<br>High: %{y}<br>Vol: %{customdata[0]:,.0f}<extra></extra>",
                ),
                row=1, col=1
            )

        # RSI Line
        rsi = RSIIndicator(close=df["close"], window=14).rsi()
        fig_combined.add_trace(
            go.Scatter(x=df['timestamp'], y=rsi, name=f'{ex_name} RSI'),
            row=2, col=1
        )

    # RSI bands
    fig_combined.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1,
                        annotation_text="Overbought", annotation_position="top left")
    fig_combined.add_hline(y=30, line_dash="dash", line_color="blue", row=2, col=1,
                        annotation_text="Oversold", annotation_position="bottom left")

    fig_combined.update_layout(
        title="Price and RSI Combined View",
        template="plotly_dark",
        height=600,
        showlegend=True,
        margin=dict(t=60, b=40),
        xaxis=dict(tickangle=45)
    )


    summary_data = []
    checkedBinance = False

    for name, exchange in exchanges_to_use.items():
        tf, delta = timeframes[st.session_state.selected]

        now = exchange.milliseconds()
        if delta is not None:
            since = now - int(delta.total_seconds() * 1000)
            ohlcv = fetch_all_ohlcv(exchange, symbol, tf, since, now)
        else:
            # Default to 30 days back if no delta is defined
            since = now - 30 * 24 * 60 * 60 * 1000
            ohlcv = fetch_all_ohlcv(exchange, symbol, tf, since, now)

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        if df.empty:
            continue

        # Ensure consistent time handling
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert(fixed_offset)

        # Basic metrics
        latest_close = df["close"].iloc[-1]
        if ex_name == "Binance":
            bin_latest_close = latest_close
            checkedBinance = True
        first_close = df["close"].iloc[0]
        pct_change = (latest_close - first_close) / first_close * 100
        total_volume = df["volume"].sum()

        summary_data.append({
            "Exchange": name,
            "Price": f"{st.session_state.quote_currency} {latest_close:,.2f}",
            "Volume": f"{total_volume:,.0f}",
            "% Change": f"{showpercentage(pct_change)}"
        })

    summary_rows = []

    for ex_name, df in dataframes_by_exchange.items():
        latest_close = df["close"].iloc[-1]
        first_close = df["close"].iloc[0]
        pct_change = (latest_close - first_close) / first_close * 100

        # RSI
        rsi_value = RSIIndicator(close=df["close"], window=14).rsi().iloc[-1]

        # OBV
        obv_value = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume().iloc[-1]

        # CVD
        df["delta"] = df["close"].diff()
        df["buy_volume"] = np.where(df["delta"] > 0, df["volume"], 0)
        df["sell_volume"] = np.where(df["delta"] < 0, df["volume"], 0)
        df["cvd"] = (df["buy_volume"] - df["sell_volume"]).cumsum()
        cvd_value = df["cvd"].iloc[-1]

        volume_total = df["volume"].sum()

        current_metrics = {
            "Price": latest_close,
            "% Change": pct_change,
            "RSI": rsi_value,
            "OBV": obv_value,
            "CVD": cvd_value,
            "Volume": volume_total
        }

        # ✅ Always create color_flags dictionary
        color_flags = {}
        if f"shown_{ex_name}" not in st.session_state:
            st.session_state[f"shown_{ex_name}"] = True
            st.session_state[f"prev_values_{ex_name}"] = current_metrics
            # All neutral if first time
            color_flags = {k: "default" for k in current_metrics}
        else:
            prev = st.session_state[f"prev_values_{ex_name}"]
            for key in current_metrics:
                if current_metrics[key] > prev[key]:
                    color_flags[key] = "up"
                elif current_metrics[key] < prev[key]:
                    color_flags[key] = "down"
                else:
                    color_flags[key] = "default"
            st.session_state[f"prev_values_{ex_name}"] = current_metrics

        # ✅ Add both values and color keys to row
        row = {"Exchange": ex_name, **current_metrics}
        for k, v in color_flags.items():
            row[f"color_{k}"] = v
        summary_rows.append(row)

    # -- ✅ Convert to DataFrame
    summary_df = pd.DataFrame(summary_rows)
    summary_df.index += 1  # 1-based index

    # Drop hidden color columns
    visible_columns = ["Exchange", "Price", "% Change", "RSI", "OBV", "CVD", "Volume"]
    display_df = summary_df[visible_columns]  # <- drop all color_*

    # Apply formatting and styling
    styled_df = display_df.style \
        .format({
            "Price": lambda x: f"{st.session_state.quote_currency} {x:,.2f}",
            "% Change": "{:+.2f}%",
            "RSI": "{:.1f}",
            "OBV": "{:,.1f}",
            "CVD": "{:,.1f}",
            "Volume": "{:,.1f}"
        })
    
    # -- 🧾 Display --
    st.markdown("### Market Summary")
    st.write(styled_df)

    # --- 📊 Signal Table per Exchange ---
    signal_rows = []

    for ex_name, df in dataframes_by_exchange.items():
        detected_signals = detect_trade_signal(df)
        
        if detected_signals:
            for sig_type, sig_name, conf in detected_signals:
                signal_rows.append({
                    "Exchange": ex_name,
                    "Signal Type": sig_type,
                    "Pattern": sig_name,
                    "Confidence": conf
                })
        else:
            signal_rows.append({
                "Exchange": ex_name,
                "Signal Type": "NEUTRAL",
                "Pattern": "—",
                "Confidence": "—"
            })

    signal_df = pd.DataFrame(signal_rows, index=range(1, len(signal_rows)+1))

    st.markdown("### Trade Signal Detector")
    st.dataframe(signal_df, use_container_width=True)





    st.plotly_chart(fig_price, use_container_width=True, key="price_chart")
    now_athens = datetime.now(tz=athens_tz).strftime("%Y-%m-%d %H:%M:%S")
    if (checkedBinance):
        st.metric(label=f"Binance Price (as of {now_athens} Athens Time)", value=f"{st.session_state.quote_currency} {bin_latest_close:,.2f}")
    else:
        st.metric(label=f"Kraken Price (as of {now_athens} Athens Time)", value=f"{st.session_state.quote_currency} {latest_close:,.2f}")

    # Render volume charts
    for ex_name in dataframes_by_exchange.keys():
        st.plotly_chart(figs_volume[ex_name], use_container_width=True, key=f"volume_{ex_name}")


    # ------------------ Volume Total ------------------
    total_vol = sum(df['volume'].sum() for df in dataframes_by_exchange.values())
    total_volume_string = "Total Volume"

    if st.session_state.selected == "1 Day":
        total_volume_string += " at the last 24 hours"
    elif st.session_state.selected == "1 Week":
        total_volume_string += " at the last 7 days"
    elif st.session_state.selected == "1 Month":
        total_volume_string += " at the last 30 days"
    elif st.session_state.selected == "1 Year":
        total_volume_string += " at the last 365 days"
    elif st.session_state.selected == "500 Days":
        total_volume_string += " at the last 500 days"
    
    if st.session_state.hide_technicals and st.session_state.hide_technicals2:
        total_volume_string += " (Binance + Kraken)"
    elif st.session_state.hide_technicals:
        total_volume_string += " (Binance only)"
    elif st.session_state.hide_technicals2:
        total_volume_string += " (Kraken only)"
    
    st.metric(label=total_volume_string, value=total_vol)


    # Render OBV, CVD charts
    for ex_name in dataframes_by_exchange.keys():
        st.plotly_chart(figs_obv_cvd[ex_name], use_container_width=True, key=f"obv_cvd_{ex_name}")

    st.plotly_chart(fig_rsi, use_container_width=True)

    from ta.momentum import StochRSIIndicator

    # ------------------ Stochastic RSI Chart ------------------
    stochrsi_data = {}
    shared_scale = True

    for ex_name, df in dataframes_by_exchange.items():
        stoch_rsi = StochRSIIndicator(close=df["close"], window=14, smooth1=3, smooth2=3)
        k = stoch_rsi.stochrsi_k()
        d = stoch_rsi.stochrsi_d()

        # Save data to check ranges
        stochrsi_data[ex_name] = {
            "timestamp": df["timestamp"],
            "%K": k*100,
            "%D": d*100
        }

    # Heuristic: if any max-min across exchanges differs significantly, split graphs
    ranges = [data["%K"].max() - data["%K"].min() for data in stochrsi_data.values()]
    max_range = max(ranges)
    min_range = min(ranges)
    if max_range == 0 or min_range == 0 or max_range / min_range > 2:
        shared_scale = False

    # Plot
    if shared_scale:
        fig_stochrsi = go.Figure()
        for ex_name, data in stochrsi_data.items():
            fig_stochrsi.add_trace(go.Scatter(x=data["timestamp"], y=data["%K"], name=f"{ex_name} %K"))
            fig_stochrsi.add_trace(go.Scatter(x=data["timestamp"], y=data["%D"], name=f"{ex_name} %D", line=dict(dash='dot')))

        fig_stochrsi.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought", annotation_position="top left")
        fig_stochrsi.add_hline(y=20, line_dash="dash", line_color="blue", annotation_text="Oversold", annotation_position="bottom left")

        fig_stochrsi.update_layout(
            title="Stochastic RSI",
            template="plotly_dark",
            height=300,
            yaxis_title="%K / %D",
            xaxis_title="Time (Athens Time)",
            xaxis=dict(tickangle=45)
        )
        st.plotly_chart(fig_stochrsi, use_container_width=True)
    else:
        for ex_name, data in stochrsi_data.items():
            fig_individual = go.Figure()
            fig_individual.add_trace(go.Scatter(x=data["timestamp"], y=data["%K"], name="%K"))
            fig_individual.add_trace(go.Scatter(x=data["timestamp"], y=data["%D"], name="%D", line=dict(dash='dot')))

            fig_individual.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought", annotation_position="top left")
            fig_individual.add_hline(y=20, line_dash="dash", line_color="blue", annotation_text="Oversold", annotation_position="bottom left")

            fig_individual.update_layout(
                title=f"Stochastic RSI - {ex_name}",
                template="plotly_dark",
                height=300,
                yaxis_title="%K / %D",
                xaxis_title="Time (Athens Time)",
                xaxis=dict(tickangle=45)
            )
            st.plotly_chart(fig_individual, use_container_width=True)

    st.markdown("### TESTING: Combined Price + RSI Chart")
    st.plotly_chart(fig_combined, use_container_width=True)





if not st.session_state.hide_order:
    import plotly.express as px

    # Fetch order book
    order_book = ccxt.binance({
        'enableRateLimit': True
    }).fetch_order_book(symbol)
    bids_all = order_book['bids']   # All bids for charts
    asks_all = order_book['asks']   # All asks for charts

    # For display only (first 100 rows for performance)
    bids_display = bids_all[:100]
    asks_display = asks_all[:100]

    # --- CATEGORIZATION FUNCTION FOR VOLUME ---
    def categorize_orders(orders):
        categories = {"Small": 0, "Medium": 0, "Big": 0}
        for _, amount in orders:
            if amount < st.session_state.small_max:
                categories["Small"] += amount
            elif amount < st.session_state.medium_max:
                categories["Medium"] += amount
            else:
                categories["Big"] += amount
        return categories

    # --- CATEGORIZATION FUNCTION FOR COUNT ---
    def count_orders(orders):
        count = {"Small": 0, "Medium": 0, "Big": 0}
        for _, amount in orders:
            if amount < st.session_state.small_max:
                count["Small"] += 1
            elif amount < st.session_state.medium_max:
                count["Medium"] += 1
            else:
                count["Big"] += 1
        return count

    # Categorize full order book for volume
    bid_vol_cat = categorize_orders(bids_all)
    ask_vol_cat = categorize_orders(asks_all)

    # Categorize full order book for count (for the extra bar chart you want)
    bid_count_cat = count_orders(bids_all)
    ask_count_cat = count_orders(asks_all)

    # --- PIE CHART DATA (volume-based) ---
    pie_df = pd.DataFrame({
        "Category": [
            "Small Bids", "Medium Bids", "Big Bids",
            "Small Asks", "Medium Asks", "Big Asks"
        ],
        "Volume": [
            bid_vol_cat["Small"], bid_vol_cat["Medium"], bid_vol_cat["Big"],
            ask_vol_cat["Small"], ask_vol_cat["Medium"], ask_vol_cat["Big"]
        ]
    })

    fig_pie = px.pie(
        pie_df,
        names="Category",
        values="Volume",
        hole=0.45,
        color="Category",
        color_discrete_map={
            "Small Bids": "#00cc96",
            "Medium Bids": "#f5c518",
            "Big Bids": "#0074D9",
            "Small Asks": "#ff4d4d",
            "Medium Asks": "#FF851B",
            "Big Asks": "#B10DC9"
        }
    )

    fig_pie.update_traces(
        textinfo="percent",
        hoverinfo="label+value+percent",
        pull=[0.03] * 6,
        sort=False,
        marker=dict(line=dict(color='#1e1e1e', width=2))
    )

    fig_pie.update_layout(
        title_text="Order Volume Split by Type & Size",
        title_x=0.02,
        showlegend=True,
        height=360,
        legend_title="Category",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        margin=dict(t=50, b=20, l=10, r=10),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font_color="white"
    )

    colors = {
        "Small Bids": "#00cc96",
        "Medium Bids": "#f5c518",
        "Big Bids": "#0074D9",
        "Small Asks": "#ff4d4d",
        "Medium Asks": "#FF851B",
        "Big Asks": "#B10DC9"
    }


    # --- BAR CHART VOLUME ---
    fig_bar_vol = go.Figure(data=[
        go.Bar(name="Small Bids", x=["Small"], y=[bid_vol_cat["Small"]], marker_color=colors["Small Bids"]),
        go.Bar(name="Medium Bids", x=["Medium"], y=[bid_vol_cat["Medium"]], marker_color=colors["Medium Bids"]),
        go.Bar(name="Big Bids", x=["Big"], y=[bid_vol_cat["Big"]], marker_color=colors["Big Bids"]),
        go.Bar(name="Small Asks", x=["Small"], y=[ask_vol_cat["Small"]], marker_color=colors["Small Asks"]),
        go.Bar(name="Medium Asks", x=["Medium"], y=[ask_vol_cat["Medium"]], marker_color=colors["Medium Asks"]),
        go.Bar(name="Big Asks", x=["Big"], y=[ask_vol_cat["Big"]], marker_color=colors["Big Asks"]),
    ])
    fig_bar_vol.update_layout(
        barmode='group',
        title_text="Order Volume by Size",
        xaxis_title="Order Size",
        yaxis_title="Volume",
        height=320,
        template="plotly_dark",
        margin=dict(t=50, b=20),
    )


    # --- BAR CHART COUNT ---
    fig_bar_count = go.Figure(data=[
        go.Bar(name="Small Bids", x=["Small"], y=[bid_count_cat["Small"]], marker_color=colors["Small Bids"]),
        go.Bar(name="Medium Bids", x=["Medium"], y=[bid_count_cat["Medium"]], marker_color=colors["Medium Bids"]),
        go.Bar(name="Big Bids", x=["Big"], y=[bid_count_cat["Big"]], marker_color=colors["Big Bids"]),
        go.Bar(name="Small Asks", x=["Small"], y=[ask_count_cat["Small"]], marker_color=colors["Small Asks"]),
        go.Bar(name="Medium Asks", x=["Medium"], y=[ask_count_cat["Medium"]], marker_color=colors["Medium Asks"]),
        go.Bar(name="Big Asks", x=["Big"], y=[ask_count_cat["Big"]], marker_color=colors["Big Asks"]),
    ])
    fig_bar_count.update_layout(
        barmode='group',
        title_text="Order Count by Size",
        xaxis_title="Order Size",
        yaxis_title="Count",
        height=320,
        template="plotly_dark",
        margin=dict(t=50, b=20),
    )


    # --- DISPLAY ---
    st.subheader("BinanceOrder Book (Top 100)")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Bids (Buy Orders)")
        st.dataframe(pd.DataFrame(bids_display, columns=[f"Price {st.session_state.quote_currency}", "Amount"], index=range(1, len(bids_display)+1)))
    with col2:
        st.write("Asks (Sell Orders)")
        st.dataframe(pd.DataFrame(asks_display, columns=[f"Price {st.session_state.quote_currency}", "Amount"], index=range(1, len(asks_display)+1)))

    st.plotly_chart(fig_pie, use_container_width=True)
    st.plotly_chart(fig_bar_vol, use_container_width=True)
    st.plotly_chart(fig_bar_count, use_container_width=True)


if not st.session_state.hide_ChronoAnalysis:
    st.header("ChronoAlpha Analysis")

    exchange = ccxt.binance({
        'enableRateLimit': True
    })

    selected_delta = range_options[st.session_state.selected_range_label]
    gap_floor_mapping = {
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "2h": "2h",
    }
    floor_freq = gap_floor_mapping[selected_gaps]

    since_ms = exchange.milliseconds() - int(selected_delta.total_seconds() * 1000)
    ohlcv = fetch_all_ohlcv(exchange, symbol, selected_gaps, since_ms, exchange.milliseconds())
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert(fixed_offset)
    df['datetime'] = df['datetime'].dt.floor(floor_freq)
    df['window_start'] = df['datetime'].dt.floor('D')

    daily_groups = df.groupby('window_start')

    score_map = {}
    sell_map = {}
    buy_map = {}

    if(selected_gaps == '1h'):
        gap_minutes = 60
    elif(selected_gaps == '2h'):
        gap_minutes = 120
    else:
        gap_minutes = int(selected_gaps.rstrip("m"))
    expected_rows = int(24 * 60 / gap_minutes)

    for window_start, group in daily_groups:
        if len(group) != expected_rows:
            continue

        ranked = group.sort_values('close').reset_index(drop=True)
        for rank, row in enumerate(ranked.itertuples(index=False), 1):
            slot = row.datetime.strftime('%H:%M')
            score_map.setdefault(slot, []).append(rank)

        max_close = group['close'].max()
        min_close = group['close'].min()
        price_range = max_close - min_close
        if price_range == 0:
            continue

        for _, row in group.iterrows():
            slot = row['datetime'].strftime('%H:%M')
            close_price = row['close']
            sell_score = (close_price - min_close) / price_range
            buy_score = 1 - sell_score
            sell_map.setdefault(slot, []).append(sell_score)
            buy_map.setdefault(slot, []).append(buy_score)

    # ChronoAlpha (position-based)
    score_summary = [
        (slot, (sum(scores)/len(scores) - 1) / (expected_rows-1)*10)
        for slot, scores in score_map.items()
    ]
    score_summary.sort(key=lambda x: x[1], reverse=True)
    df_top = pd.DataFrame(score_summary[:10], columns=["Time (Athens Time)", "ChronoAlpha Score"])
    df_top.index += 1

    df_bottom = pd.DataFrame(score_summary[-10:], columns=["Time (Athens Time)", "ChronoAlpha Score"])
    df_bottom = df_bottom.sort_values("ChronoAlpha Score", ascending=True)
    df_bottom.index = range(1, 11)

    # Line chart (normalized scores)
    normalized_scores = sorted(score_summary, key=lambda x: x[0])
    score_df = pd.DataFrame(normalized_scores, columns=["Time", "Normalized Score"])
    y_min = min(score_df["Normalized Score"]) * 0.999
    y_max = max(score_df["Normalized Score"]) * 1.001

    # Sell/buy score transformation
    scaled_sell_scores = [
        (slot, round((sum(scores) / len(scores)) * 10, 3))
        for slot, scores in sorted(sell_map.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)
    ]
    scaled_buy_scores = list(reversed(scaled_sell_scores))

    df_sell = pd.DataFrame(scaled_sell_scores[:10], columns=["Time (Athens Time)", "ChronoAlpha Score"])
    df_sell.index += 1

    df_buy = pd.DataFrame(scaled_buy_scores[:10], columns=["Time (Athens Time)", "ChronoAlpha Score"])
    df_buy.index += 1

    st.subheader("ChronoAlpha: Time-Based Strength")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Top 10 Selling Intervals**")
        st.dataframe(df_sell)

    with col4:
        st.markdown("**Top 10 Buying Intervals**")
        st.dataframe(df_buy)

    # Final chart
    combined_scores = sorted(scaled_sell_scores, key=lambda x: x[0])
    score_df = pd.DataFrame(combined_scores, columns=["Time", "ChronoAlpha Score"])
    y_min = min(score_df["ChronoAlpha Score"]) * 0.98
    y_max = max(score_df["ChronoAlpha Score"]) * 1.02

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=score_df["Time"],
        y=score_df["ChronoAlpha Score"],
        mode='lines+markers',
        line=dict(color="#636efa"),
        name="ChronoAlpha Score",
        line_shape="spline"
    ))
    fig2.update_layout(
        title="ChronoAlpha Score by Time of Day (0–10 Scale)",
        xaxis_title="Time of Day (Athens Time)",
        yaxis_title="ChronoAlpha Score",
        template="plotly_dark",
        height=350,
        margin=dict(t=50, b=20),
        xaxis=dict(tickangle=45),
        yaxis=dict(range=[y_min, y_max]),
        modebar_remove=["zoomIn2d", "zoomOut2d"]
    )
    st.plotly_chart(fig2, use_container_width=True)





refresh_interval = 60

# Store the time of last refresh
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if not st.session_state.auto_refresh:
    now = time.time()
    elapsed = now - st.session_state.last_refresh
    remaining = int(refresh_interval - elapsed)

    with st.empty():
        for seconds in range(refresh_interval, 0, -1):
            st.info(f"Auto-refreshing in {seconds} seconds...")
            time.sleep(1)
    st.session_state.last_refresh = now
    st.rerun()


