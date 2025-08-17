# app.py
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import urllib
import plotly.graph_objects as go
import xml.etree.ElementTree as ET


def load_assets_from_xml(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    assets = {}

    for asset_class in root.findall("AssetClass"):
        class_name = asset_class.attrib["name"]
        assets[class_name] = {}

        for product in asset_class:
            tag = product.tag  # Fx, Etf, Future, Index, etc.
            if tag not in assets[class_name]:
                assets[class_name][tag] = {}

            name = product.find("Name").text.strip()
            ticker = product.find("Ticker").text.strip()
            assets[class_name][tag][name] = ticker

    return assets

def get_ticker(assets_dict, asset_class, product_type, name):
    try:
        return assets_dict[asset_class][product_type][name]
    except KeyError:
        return None
st.set_page_config(layout="wide")


# Load XML once at app start
assets_dict = load_assets_from_xml("assets.xml")


def get_data(ticker,start):
    ticker_request = ticker.replace("=", "%3D")

    try:
        endpoint_data = f"https://raw.githubusercontent.com/alitalbi/storage_data_fy/refs/heads/master/{ticker_request}.csv"
        price_df = pd.read_csv(endpoint_data)
        price_df.set_index("Date", inplace=True)
        price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
        price_df = price_df.loc[price_df.index > start]
        return price_df
    except urllib.error.HTTPError as e:
            print(f"HTTP Error: {e.code} {e.reason}")
            print(f"URL: {endpoint_data}")
            raise
# --- Helper Functions ---
def calculate_rsi(series, period):
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_williams_r(high, low, close, period):
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    return williams_r

def calculate_cci(high, low, close, period):
    tp = (high + low + close) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad)
    return cci

# --- Sidebar Navigation ---
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio("Go to", ["Momentum Dashboard", "Strategy Backtest"])

# --- Page 1: Momentum Dashboard ---
if page == "Momentum Dashboard":
    st.title("📈 Momentum Dashboard:")
    col1,col2 = st.columns(2)
    with col1:
        with st.expander("⚙️ Settings", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m"], index=0)
            with col2:
                period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
            with col3:
                start=st.date_input(label="Start Date",
                                  value=datetime.date(2025, 1, 1),
                                  min_value=datetime.date(1990, 1, 1),
                                  max_value=datetime.date.today())

            with col4:
                refresh_button = st.button("🔄 Refresh Data")
            cols = st.columns([3, 3, 3])
            with cols[0]:
                # Step 1: Choose Asset Class
                asset_class = st.selectbox("Asset Class", list(assets_dict.keys()))
            with cols[1]:
                # Step 2: Choose Product Type
                product_type = st.selectbox("Product Type", list(assets_dict[asset_class].keys()))

            with cols[2]:
                # Step 3: Choose Product Name
                name = st.selectbox("Product", list(assets_dict[asset_class][product_type].keys()))

            # Output Ticker
            ticker = get_ticker(assets_dict, asset_class, product_type, name)
            st.markdown("---")
            st.markdown("#### Indicators Setup")
            rsi_period = st.number_input("RSI Period", min_value=5, value=28)
            wlpr_period = st.number_input("Williams %R Period", min_value=5,  value=70)
            cci_period = st.number_input("CCI Period", min_value=5, value=70)

    if refresh_button or True:
        if ticker == "":
            ticker = "^GSPC"
        df = get_data(ticker, start=start.strftime("%Y-%m-%d"))
        df.index = pd.to_datetime(df.index)
        df['RSI'] = calculate_rsi(df['Close'], rsi_period)
        df['WLR%'] = calculate_williams_r(df['High'], df['Low'], df['Close'], wlpr_period)
        df['CCI'] = calculate_cci(df['High'], df['Low'], df['Close'], cci_period)

        # --- Calculate Aggregate Momentum Score (-3 to 3)
        df['Score'] = 0
        df['Score'] += np.where(df['RSI'] > 60, 1, 0)
        df['Score'] += np.where(df['RSI'] < 40, -1, 0)
        df['Score'] += np.where(df['WLR%'] > -25, 1, 0)
        df['Score'] += np.where(df['WLR%'] < -75, -1, 0)
        df['Score'] += np.where(df['CCI'] > 100, 1, 0)
        df['Score'] += np.where(df['CCI'] < -100, -1, 0)

        # --- Plot Candlestick
        st.markdown("### 📈 "+ticker+" Candlestick Chart")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'],
                                     name="Candles"))
        fig.update_layout(xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # --- Plot Aggregate Score Bar
        st.markdown("### 📊 Aggregate Momentum Score (-3 to +3)")
        fig_score = go.Figure()
        fig_score.add_trace(go.Bar(
            x=df.index,
            y=df['Score'],
            marker_color=df['Score'].apply(lambda x: 'green' if x > 0 else 'red' if x < 0 else 'gray')
        ))
        fig_score.update_layout(height=300,
                                yaxis=dict(range=[-3, 3]),
                                showlegend=False)
        st.plotly_chart(fig_score, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # --- Checkbox to show/hide indicators
        if st.checkbox("🔍 Show individual indicators (RSI, WLR%, CCI)"):
            st.markdown("### 📊 RSI Indicator")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
            fig_rsi.add_hline(y=60, line_dash="dash", line_color="green")
            fig_rsi.add_hline(y=40, line_dash="dash", line_color="red")
            fig_rsi.update_layout(height=300)
            st.plotly_chart(fig_rsi, use_container_width=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            st.markdown("### 📊 Williams %R Indicator")
            fig_wlpr = go.Figure()
            fig_wlpr.add_trace(go.Scatter(x=df.index, y=df['WLR%'], mode='lines', name='WLR%'))
            fig_wlpr.add_hline(y=-25, line_dash="dash", line_color="green")
            fig_wlpr.add_hline(y=-75, line_dash="dash", line_color="red")
            fig_wlpr.update_layout(height=300)
            st.plotly_chart(fig_wlpr, use_container_width=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            st.markdown("### 📊 CCI Indicator")
            fig_cci = go.Figure()
            fig_cci.add_trace(go.Scatter(x=df.index, y=df['CCI'], mode='lines', name='CCI'))
            fig_cci.add_hline(y=100, line_dash="dash", line_color="green")
            fig_cci.add_hline(y=-100, line_dash="dash", line_color="red")
            fig_cci.update_layout(height=300)
            st.plotly_chart(fig_cci, use_container_width=True)

# --- Page 2: Strategy Backtest ---
# --- Page 2: Strategy Backtest ---
elif page == "Strategy Backtest":

    def calculate_hurst(ts):
        """Safe Hurst Exponent calculation."""
        ts = np.array(ts)
        if len(ts) < 20:
            return np.nan

        lags = range(2, 20)
        tau = []

        for lag in lags:
            if lag >= len(ts):
                break
            diff = ts[lag:] - ts[:-lag]
            std = np.std(diff)
            if std > 1e-8:  # ignore almost zero std
                tau.append(std)

        if len(tau) < 2:
            return np.nan

        log_lags = np.log(np.array(range(2, 2 + len(tau))))
        log_tau = np.log(np.array(tau))

        if np.any(np.isinf(log_tau)) or np.any(np.isnan(log_tau)):
            return np.nan

        m = np.polyfit(log_lags, log_tau, 1)
        hurst = m[0]

        return hurst


    st.title("🧪 Strategy Backtest:")

    with st.expander("⚙️ Backtest Settings", expanded=True):
        # --- Date Pickers at the top ---
        st.subheader("Select Date Range:")

        col1, col2 = st.columns(2)  # Two side-by-side columns

        with col1:
            start = st.date_input(
                label="Start Date",
                value=datetime.date(2023, 1, 1),
                min_value=datetime.date(1990, 1, 1),
                max_value=datetime.date.today()
            )

        with col2:
            end = st.date_input(
                label="End Date",
                value=datetime.date.today(),
                min_value=datetime.date(1990, 1, 1),
                max_value=datetime.date.today()
            )

        start = start.strftime("%Y-%m-%d")
        end = datetime.date.today().strftime('%Y-%m-%d')
        col1, col2 = st.columns(2)
        cols = st.columns([3, 3, 3])
        with cols[0]:
            # Step 1: Choose Asset Class
            asset_class = st.selectbox("Asset Class", list(assets_dict.keys()))
        with cols[1]:
            # Step 2: Choose Product Type
            product_type = st.selectbox("Product Type", list(assets_dict[asset_class].keys()))

        with cols[2]:
            # Step 3: Choose Product Name
            name = st.selectbox("Product", list(assets_dict[asset_class][product_type].keys()))
        st.markdown("---")
        st.markdown("#### Strategy Parameters")
        sma_period = st.number_input("SMA Period", min_value=5,  value=28, key="sma_period")
        initial_cash = st.number_input("Initial Cash ($)", min_value=1000, value=100000, step=1000)
    ticker = get_ticker(assets_dict,asset_class,product_type,name)
    if ticker == "":
        ticker = "^GSPC"
    df = get_data(ticker, start=start)
    # If the columns have MultiIndex, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    hurst_window = 50  # you can expose this as a parameter too

    df['Hurst'] = df['Close'].rolling(window=hurst_window).apply(calculate_hurst, raw=False)
    #st.write(df)

    # --- Calculate Indicators
    df['SMA'] = df['Close'].rolling(window=sma_period).mean()
    df['RSI'] = calculate_rsi(df['Close'], 28)
    df['WLR%'] = calculate_williams_r(df['High'], df['Low'], df['Close'], 70)
    df['CCI'] = calculate_cci(df['High'], df['Low'], df['Close'], 70)

    # --- Create Scorecard
    df['Score'] = 0
    df['Score'] += np.where(df['RSI'] > 60, 1, 0)
    df['Score'] += np.where(df['RSI'] < 40, -1, 0)
    df['Score'] += np.where(df['WLR%'] > -25, 1, 0)
    df['Score'] += np.where(df['WLR%'] < -75, -1, 0)
    df['Score'] += np.where(df['CCI'] > 100, 1, 0)
    df['Score'] += np.where(df['CCI'] < -100, -1, 0)

    # --- Generate Entry Signals
    # Apply only during trendy cycle
    trend_condition = df['Hurst'] > 0.5
    # --- Rolling Mean and Std
    mean_close = df['Close'].rolling(window=sma_period).mean()
    std_close = df['Close'].rolling(window=sma_period).std()

    # --- Rolling SMA already calculated
    # df['SMA'] = df['Close'].rolling(window=sma_period).mean()

    # --- Define bands
    upper_band = mean_close + 0.2 * std_close
    lower_band = mean_close - 0.2 * std_close

    # --- Signal Initialization
    df['Signal'] = 0

    # --- Flat Zone (no trading if close to mean)
    flat_zone = (df['Close'] > lower_band) & (df['Close'] < upper_band)

    # --- Trading Logic
    # Long condition
    long_condition = (df['Close'] > df['SMA']) & (df['Score'] == 3)

    # Short condition
    short_condition = (df['Close'] < df['SMA']) & (df['Score'] == -3)

    # --- Apply logic
    # df.loc[flat_zone, 'Signal'] = 0  # No trade if in flat zone
    df.loc[long_condition, 'Signal'] = 1
    df.loc[short_condition, 'Signal'] = -1

   # df['Signal'] = 0
   # df.loc[(trend_condition) & (df['Close'] > df['SMA']) & (df['Score'] >= 2), 'Signal'] = 1
   # df.loc[(trend_condition) & (df['Close'] < df['SMA']) & (df['Score'] <= -2), 'Signal'] = -1

    # --- Strategy Returns
    df['Position'] = df['Signal'].shift(1).fillna(0)
    df['Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Return'] * df['Position']
    df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod() * initial_cash

    # --- PLOT 1: Price with Buy/Sell Markers and SMA
    st.markdown("### 📈 Price with Buy/Sell Signals and SMA")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name="Close Price"))
    fig_price.add_trace(go.Scatter(x=df.index, y=df['SMA'], mode='lines', name=f"SMA {sma_period}", line=dict(dash='dash')))

    # Create the columns first

    buys = df.loc[df['Signal'] == 1]
    sells = df.loc[df['Signal'] == -1]
    st.write("Buy signals:", len(buys))
    st.write("Sell signals:", len(sells))
    fig_price.add_trace(go.Scatter(x=buys.index, y=buys['Close'],
                                   mode='markers', name="Buy", marker_symbol='triangle-up', marker_color='green', marker_size=10))
    fig_price.add_trace(go.Scatter(x=sells.index, y=sells['Close'],
                                   mode='markers', name="Sell", marker_symbol='triangle-down', marker_color='red', marker_size=10))

    fig_price.update_layout(height=600)
    st.plotly_chart(fig_price, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- PLOT 2: Scorecard Bar
    st.markdown("### 📊 Momentum Score (-3 to 3)")
    fig_score = go.Figure()
    fig_score.add_trace(go.Bar(
        x=df.index,
        y=df['Score'],
        marker_color=df['Score'].apply(lambda x: 'green' if x > 0 else 'red' if x < 0 else 'gray')
    ))
    fig_score.update_layout(height=300, yaxis=dict(range=[-3, 3]))
    st.plotly_chart(fig_score, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- PLOT 3: Cumulative Return
    st.markdown("### 💰 Cumulative Strategy Return")
    fig_return = go.Figure()
    fig_return.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Return'], mode='lines', name="Strategy Equity"))
    fig_return.update_layout(height=400)
    st.plotly_chart(fig_return, use_container_width=True)

    st.markdown("### 📈 Hurst Exponent >0.5 Trending, ~0.5 random walk, <0.5 Mean Reverting")
    fig_hurst = go.Figure()
    fig_hurst.add_trace(go.Scatter(x=df.index, y=df['Hurst'], mode='lines', name="Hurst Exponent"))
    fig_hurst.update_layout(height=300)
    st.plotly_chart(fig_hurst, use_container_width=True)


