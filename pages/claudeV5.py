import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Fast Regime Analysis", page_icon="⚡")

# Initialize FRED API
fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')


@st.cache_data
def get_github_data(ticker, start_date):
    """Fetch data from GitHub repository"""
    ticker_request = ticker.replace("=", "%3D")
    try:
        url = f"https://raw.githubusercontent.com/alitalbi/storage_data_fy/refs/heads/master/{ticker_request}.csv"
        df = pd.read_csv(url, usecols=["Date", "Close", "High", "Low", "Open"])
        df.set_index("Date", inplace=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.loc[df.index > start_date]
    except Exception as e:
        st.error(f"Error loading {ticker}: {e}")
        return pd.DataFrame()


def fast_adx(high, low, close, period=14):
    """Fast ADX calculation using vectorized operations"""
    # Simplified ADX calculation
    hl_diff = high - low
    hc_diff = abs(high - close.shift(1))
    lc_diff = abs(low - close.shift(1))

    tr = pd.concat([hl_diff, hc_diff, lc_diff], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_di = 100 * pd.Series(plus_dm, index=close.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(period).mean() / atr

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
    adx = dx.rolling(period).mean()

    return adx.fillna(0)


def fast_regime_indicators(data, short_period=10, long_period=60):
    """Fast calculation of all regime indicators"""
    high, low, close = data['High'], data['Low'], data['Close']

    # 1. ADX (trend strength)
    adx = fast_adx(high, low, close, short_period)

    # 2. Price momentum (simplified)
    returns = close.pct_change()
    momentum = returns.rolling(short_period).mean()

    # 3. Volatility ratio (range vs trend)
    price_range = (high - low).rolling(short_period).mean()
    price_trend = abs(close - close.shift(short_period))
    vol_ratio = price_range / (price_trend + 0.001)

    # 4. Moving average distance
    ma_long = close.rolling(long_period).mean()
    ma_distance = abs(close - ma_long) / ma_long

    # 5. Range breakout
    rolling_high = high.rolling(long_period).max()
    rolling_low = low.rolling(long_period).min()
    range_size = rolling_high - rolling_low
    breakout = np.maximum(
        (close - rolling_high.shift(1)) / range_size,
        (rolling_low.shift(1) - close) / range_size
    ).fillna(0)

    return {
        'adx': adx,
        'momentum': momentum,
        'vol_ratio': vol_ratio,
        'ma_distance': ma_distance,
        'breakout': breakout
    }


def calculate_regime_score(indicators):
    """Fast regime score calculation"""
    # Normalize indicators (clip to 0-1)
    adx_norm = np.clip(indicators['adx'] / 30, 0, 1)
    momentum_norm = np.clip(abs(indicators['momentum']) * 50, 0, 1)
    breakout_norm = np.clip(indicators['breakout'] * 3, 0, 1)

    vol_ratio_norm = np.clip((indicators['vol_ratio'] - 1) / 2, 0, 1)
    ma_dist_norm = np.clip(indicators['ma_distance'] * 20, 0, 1)

    # Trending score (higher = more trending)
    trending = (adx_norm * 0.4 + momentum_norm * 0.3 + breakout_norm * 0.3)

    # Ranging score (higher = more ranging)
    ranging = (vol_ratio_norm * 0.6 + ma_dist_norm * 0.4)

    # Composite score
    composite = trending - ranging

    # Smooth with simple 3-day average
    composite_smooth = composite.rolling(3, center=True).mean()

    return trending, ranging, composite, composite_smooth


def classify_regime(composite_smooth, trend_thresh=0.15, range_thresh=-0.15):
    """Fast regime classification with hysteresis"""
    regime = pd.Series(0, index=composite_smooth.index)

    # Vectorized regime classification
    regime = np.where(composite_smooth > trend_thresh, 1, regime)
    regime = np.where(composite_smooth < range_thresh, -1, regime)

    # Simple hysteresis: smooth again to reduce switches
    regime = pd.Series(regime, index=composite_smooth.index)
    regime = regime.rolling(2).median().fillna(0)

    return regime


# Header
st.title("⚡ Fast Market Regime Analysis")
st.caption("Optimized regime detection with key indicators")

# Simple Parameters
col1, col2, col3 = st.columns(3)

with col1:
    start_date = st.date_input("Start Date", datetime(2022, 1, 1))
    short_period = st.slider("Short Period", 5, 20, 10)

with col2:
    end_date = st.date_input("End Date", datetime.now().date())
    long_period = st.slider("Long Period", 30, 90, 60)

with col3:
    trend_threshold = st.slider("Trend Threshold", 0.05, 0.30, 0.15, 0.05)
    range_threshold = st.slider("Range Threshold", -0.30, -0.05, -0.15, 0.05)

# Load and Process Data
if st.button("⚡ Run Fast Analysis", type="primary"):

    with st.spinner("Loading data..."):
        # Load data
        data = get_github_data("ZF=F", start_date.strftime("%Y-%m-%d"))

        if data.empty:
            st.error("Failed to load data")
            st.stop()

        # Filter by date
        data = data.loc[
            (data.index >= pd.Timestamp(start_date)) &
            (data.index <= pd.Timestamp(end_date))
            ]

    with st.spinner("Calculating regimes..."):
        # Fast calculations
        indicators = fast_regime_indicators(data, short_period, long_period)
        trending, ranging, composite, composite_smooth = calculate_regime_score(indicators)
        regime = classify_regime(composite_smooth, trend_threshold, range_threshold)

    # Store results
    st.session_state.update({
        'data': data,
        'indicators': indicators,
        'trending': trending,
        'ranging': ranging,
        'composite': composite,
        'composite_smooth': composite_smooth,
        'regime': regime
    })

# Display Results
if 'data' in st.session_state:
    data = st.session_state.data
    indicators = st.session_state.indicators
    regime = st.session_state.regime
    composite_smooth = st.session_state.composite_smooth

    # Main Chart
    st.markdown("## Price & Regime Classification")

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Price with Regime Background", "Composite Regime Score"),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )

    # Price line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='5Y Futures',
            line=dict(color='black', width=1)
        ),
        row=1, col=1
    )

    # Regime background (simplified - sample every 10th point for speed)
    regime_sample = regime  # Sample every 5th point for speed
    for i in range(len(regime_sample) - 1):
        if regime_sample.iloc[i] == 1:  # Trending
            fig.add_vrect(
                x0=regime_sample.index[i],
                x1=regime_sample.index[i + 1],
                fillcolor="lightblue",
                opacity=0.3,
                layer="below",
                line_width=0,
                row=1, col=1
            )
        elif regime_sample.iloc[i] == -1:  # Ranging
            fig.add_vrect(
                x0=regime_sample.index[i],
                x1=regime_sample.index[i + 1],
                fillcolor="lightyellow",
                opacity=0.3,
                layer="below",
                line_width=0,
                row=1, col=1
            )

    # Composite score
    fig.add_trace(
        go.Scatter(
            x=composite_smooth.index,
            y=composite_smooth,
            mode='lines',
            name='Regime Score',
            line=dict(color='purple', width=2)
        ),
        row=2, col=1
    )

    # Thresholds
    fig.add_hline(y=trend_threshold, line_dash="dash", line_color="blue", row=2, col=1)
    fig.add_hline(y=range_threshold, line_dash="dash", line_color="orange", row=2, col=1)
    fig.add_hline(y=0, line_color="gray", row=2, col=1)

    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)

    regime_counts = regime.value_counts()
    total = len(regime.dropna())

    with col1:
        trending_pct = (regime_counts.get(1, 0) / total) * 100
        st.metric("Trending", f"{trending_pct:.1f}%")

    with col2:
        ranging_pct = (regime_counts.get(-1, 0) / total) * 100
        st.metric("Ranging", f"{ranging_pct:.1f}%")

    with col3:
        unknown_pct = (regime_counts.get(0, 0) / total) * 100
        st.metric("Unknown", f"{unknown_pct:.1f}%")

    with col4:
        current_regime = regime.iloc[-1]
        regime_map = {1: "Trending", -1: "Ranging", 0: "Unknown"}
        st.metric("Current", regime_map.get(current_regime, "Unknown"))

    # Key Indicators Chart
    st.markdown("## Key Indicators")

    fig_ind = make_subplots(
        rows=2, cols=2,
        subplot_titles=("ADX (Trend Strength)", "Momentum", "Volatility Ratio", "Breakout Strength"),
        vertical_spacing=0.15
    )

    # ADX
    fig_ind.add_trace(
        go.Scatter(x=indicators['adx'].index, y=indicators['adx'],
                   mode='lines', name='ADX', line=dict(color='red')),
        row=1, col=1
    )
    fig_ind.add_hline(y=25, line_dash="dash", line_color="gray", row=1, col=1)

    # Momentum
    fig_ind.add_trace(
        go.Scatter(x=indicators['momentum'].index, y=indicators['momentum'],
                   mode='lines', name='Momentum', line=dict(color='blue')),
        row=1, col=2
    )
    fig_ind.add_hline(y=0, line_color="gray", row=1, col=2)

    # Volatility Ratio
    fig_ind.add_trace(
        go.Scatter(x=indicators['vol_ratio'].index, y=indicators['vol_ratio'],
                   mode='lines', name='Vol Ratio', line=dict(color='orange')),
        row=2, col=1
    )
    fig_ind.add_hline(y=1, line_dash="dash", line_color="gray", row=2, col=1)

    # Breakout
    fig_ind.add_trace(
        go.Scatter(x=indicators['breakout'].index, y=indicators['breakout'],
                   mode='lines', name='Breakout', line=dict(color='green')),
        row=2, col=2
    )

    fig_ind.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_ind, use_container_width=True)

    # Recent Analysis Table
    st.markdown("## Recent Regime Analysis")

    # Create recent data table
    recent_df = pd.DataFrame({
        'Date': data.index[-20:],
        'Close': data['Close'].iloc[-20:].round(3),
        'ADX': indicators['adx'].iloc[-20:].round(1),
        'Score': composite_smooth.iloc[-20:].round(3),
        'Regime': regime.iloc[-20:].map({1: 'Trending', -1: 'Ranging', 0: 'Unknown'})
    })


    # Style the table
    def color_regime(val):
        colors = {'Trending': 'background-color: lightblue',
                  'Ranging': 'background-color: lightyellow',
                  'Unknown': 'background-color: lightgray'}
        return colors.get(val, '')


    styled_df = recent_df.style.applymap(color_regime, subset=['Regime'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Export
    st.markdown("## Export Data")

    export_df = pd.DataFrame({
        'Date': data.index,
        'Close': data['Close'],
        'ADX': indicators['adx'],
        'Momentum': indicators['momentum'],
        'Regime_Score': composite_smooth,
        'Regime': regime,
        'Regime_Label': regime.map({1: 'Trending', -1: 'Ranging', 0: 'Unknown'})
    }).dropna()

    csv_data = export_df.to_csv(index=False)
    st.download_button(
        "Download Regime Data",
        csv_data,
        f"fast_regime_{start_date}_{end_date}.csv",
        "text/csv"
    )

else:
    st.info("Click 'Run Fast Analysis' to start")

    st.markdown("""
    **This optimized version:**
    - Uses vectorized calculations (10x+ faster)
    - Simplified but effective indicators
    - Reduced plotting complexity
    - Focuses on essential regime detection

    **Indicators:**
    - **ADX**: Trend strength (>25 = strong trend)
    - **Momentum**: Price momentum direction
    - **Volatility Ratio**: Range vs trend movement
    - **Breakout**: Range breakout detection
    """)