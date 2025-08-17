import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import urllib
from fredapi import Fred

# Initialize FRED API
fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')
st.set_page_config(layout="wide", page_title="Market Regime Analysis", page_icon="📊")
logging.basicConfig(level=logging.INFO)

# Assets dictionary
assets_dict = {
    "2y US": "DGS2",
    "5y US": "^FVX",
    "5y US Real": "DFII5",
    "5y US Future": "ZF=F"
}


def get_data(ticker, start):
    """Fetch data from GitHub repository"""
    ticker_request = ticker.replace("=", "%3D")
    try:
        endpoint_data = f"https://raw.githubusercontent.com/alitalbi/storage_data_fy/refs/heads/master/{ticker_request}.csv"
        price_df = pd.read_csv(endpoint_data, usecols=["Date", "Close", "High", "Low", "Open"])
        price_df.set_index("Date", inplace=True)
        price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
        price_df = price_df.loc[price_df.index > start]
        return price_df
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} {e.reason}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


def fred_import(ticker, start_date):
    """Import data from FRED"""
    fred_data = pd.DataFrame(fred.get_series(ticker, observation_start=start_date, freq="daily"))
    return fred_data


def compute_fractal_dimension(price_series, scaling_factor):
    """
    Compute fractal dimension for price series
    Values > 1.5: Trending market (sustainable directional movement)
    Values ~ 1.5: Random walk (efficient market)
    Values < 1.5: Mean reverting/ranging market (bounded movement)
    """
    if len(price_series) < scaling_factor:
        return np.full(len(price_series), np.nan)

    fractal_dims = []

    for i in range(len(price_series)):
        if i < scaling_factor:
            fractal_dims.append(np.nan)
            continue

        window = price_series.iloc[i - scaling_factor:i + 1]

        if len(window) < 2:
            fractal_dims.append(np.nan)
            continue

        normalized_prices = (window - window.iloc[0]) / window.iloc[0] if window.iloc[0] != 0 else window - window.iloc[
            0]
        path_length = np.sum(np.abs(np.diff(normalized_prices)))
        straight_distance = abs(normalized_prices.iloc[-1] - normalized_prices.iloc[0])

        if straight_distance == 0:
            fractal_dim = 1.0
        else:
            fractal_dim = 1 + (np.log(path_length) - np.log(straight_distance)) / np.log(2)

        fractal_dims.append(fractal_dim)

    return pd.Series(fractal_dims, index=price_series.index)


def calculate_hurst(ts):
    """
    Calculate Hurst Exponent
    Values > 0.5: Persistent/trending behavior (momentum)
    Values ~ 0.5: Random walk (no memory)
    Values < 0.5: Anti-persistent/mean reverting behavior
    """
    ts = np.array(ts)
    if len(ts) < 20:
        return np.nan

    lags = range(2, min(20, len(ts)))
    tau = []

    for lag in lags:
        if lag >= len(ts):
            break
        diff = ts[lag:] - ts[:-lag]
        std = np.std(diff)
        if std > 1e-8:
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


def calculate_rsi(series, period):
    """RSI: Values >70 overbought, <30 oversold"""
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_williams_r(high, low, close, period):
    """Williams %R: Values >-20 overbought, <-80 oversold"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    return williams_r


def calculate_cci(high, low, close, period):
    """CCI: Values >+100 overbought, <-100 oversold"""
    tp = (high + low + close) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad)
    return cci


def build_indicators(data):
    """Build carry and momentum indicators"""
    data["5_2y"] = data["5y"] - data["2y"]
    data["carry_normalized"] = data["5_2y"] / data["5_2y"].rolling(75).std()
    data["5d_ma_5y"] = data["5y"].rolling(5).mean()
    data["20d_ma_5y"] = data["5y"].rolling(20).mean()
    data["momentum"] = data["5d_ma_5y"] - data["20d_ma_5y"]
    return data


def percentile_score(window):
    """Calculate percentile score"""
    if len(window) == 0:
        return np.nan
    current_value = window[-1]
    nb_values_below = np.sum(window <= current_value)
    percentile = (nb_values_below / len(window)) * 100
    return percentile


def zscore(data, lookback):
    """Calculate Z-score"""
    return (data - data.rolling(lookback).mean()) / data.rolling(lookback).std()


def classify_regime(fractal_dim, momentum_score, hurst, fractal_trend_thresh, fractal_range_thresh,
                    hurst_trend_thresh, hurst_range_thresh, momentum_strong_thresh, momentum_weak_thresh):
    """
    Classify market regime based on indicators with customizable thresholds
    """
    if pd.isna(fractal_dim) or pd.isna(momentum_score) or pd.isna(hurst):
        return "Unknown"

    # Trending regimes
    if fractal_dim > fractal_trend_thresh and hurst > hurst_trend_thresh:
        if momentum_score > momentum_strong_thresh:
            return "Strong Uptrend"
        elif momentum_score < -momentum_strong_thresh:
            return "Strong Downtrend"
        else:
            return "Weak Trend"

    # Ranging/Mean reverting regimes
    elif fractal_dim < fractal_range_thresh and hurst < hurst_range_thresh:
        if abs(momentum_score) > momentum_strong_thresh:
            return "Volatile Range"
        else:
            return "Calm Range"

    # Mixed regimes
    else:
        if momentum_score > momentum_strong_thresh:
            return "Momentum Breakout"
        elif momentum_score < -momentum_strong_thresh:
            return "Momentum Breakdown"
        else:
            return "Transition"


# Streamlit App Header
st.title("📊 Market Regime Analysis Dashboard")
st.markdown("**Professional Fixed Income Strategy Analysis for 5Y US Treasury Futures**")

with st.expander("📖 Dashboard Overview & Methodology", expanded=False):
    st.markdown("""
    ## Dashboard Purpose
    This dashboard combines **fractal dimension analysis**, **momentum indicators**, and **Hurst exponent** to classify market regimes 
    and analyze the performance of our 5Y US Treasury futures trading strategy across different market conditions.

    ## Key Components

    ### 1. **Strategy Indicators** (from Strat_Ponctual.py)
    - **Value Component (Default 70%)**: 5Y Real rates percentile ranking over 63-day window
    - **Carry Component (Default 20%)**: 5Y-2Y spread normalized by 75-day rolling volatility
    - **Momentum Component (Default 10%)**: 5-day vs 20-day moving average of 5Y yields
    - **Aggregate Percentile**: Weighted combination triggering trades at 90%+ (Buy) or 10-20% (Sell)

    ### 2. **Market Microstructure Indicators**
    - **Fractal Dimension**: Measures market efficiency vs trending behavior
      - Values > 1.5: Trending markets (sustainable directional moves)
      - Values ≈ 1.5: Random walk (efficient pricing)  
      - Values < 1.5: Mean-reverting/ranging markets

    - **Hurst Exponent**: Measures persistence of price movements
      - Values > 0.5: Trending/persistent behavior
      - Values ≈ 0.5: Random walk (no memory)
      - Values < 0.5: Mean-reverting behavior

    - **Momentum Score**: Combines RSI, Williams %R, and CCI (-3 to +3 scale)

    ### 3. **Market Regime Classification**
    Eight distinct regimes identified by combining the above indicators:
    - **Strong Uptrend/Downtrend**: High fractal + high Hurst + strong momentum
    - **Weak Trend**: Trending but low momentum
    - **Volatile/Calm Range**: Low fractal + low Hurst with varying momentum
    - **Momentum Breakout/Breakdown**: Mixed regime signals with strong momentum
    - **Transition**: Mixed or unclear regime signals

    ## Practical Applications
    - **Strategy Optimization**: Understand which market regimes generate most profitable signals
    - **Risk Management**: Adjust position sizing based on regime volatility characteristics
    - **Signal Filtering**: Potentially avoid trading in certain regimes or adjust thresholds
    - **Performance Attribution**: Decompose strategy returns by market regime
    """)

# Settings
with st.expander("⚙️ Strategy Parameters", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📅 Time Period**")
        start_date_input = st.date_input("Start Date", value=datetime(2023, 1, 1))

        st.markdown("**💼 Strategy Weights**")
        carry_weight = st.number_input("Carry Weight (%)", value=20.0, min_value=0.0, max_value=100.0)
        value_weight = st.number_input("Value Weight (%)", value=70.0, min_value=0.0, max_value=100.0)
        momentum_weight = st.number_input("Momentum Weight (%)", value=10.0, min_value=0.0, max_value=100.0)

    with col2:
        st.markdown("**🎯 Trading Signal Thresholds**")
        buy_threshold = st.number_input("Buy Signal Threshold (%)", value=90.0, min_value=50.0, max_value=99.0)
        sell_threshold_high = st.number_input("Sell Signal Upper (%)", value=20.0, min_value=1.0, max_value=49.0)
        sell_threshold_low = st.number_input("Sell Signal Lower (%)", value=10.0, min_value=1.0, max_value=49.0)

        st.markdown("**📊 Technical Indicator Periods**")
        hurst_window = st.number_input("Hurst Window", value=50, min_value=20, max_value=100)
        rsi_period = st.number_input("RSI Period", value=14, min_value=5, max_value=50)
        fractal_scaling = st.number_input("Fractal Scaling Factor", value=65, min_value=20, max_value=150)

    with col3:
        st.markdown("**🔄 Regime Classification Thresholds**")
        fractal_trend_thresh = st.number_input("Fractal Trending Threshold", value=1.5, min_value=1.0, max_value=2.0,
                                               step=0.1)
        fractal_range_thresh = st.number_input("Fractal Ranging Threshold", value=1.5, min_value=1.0, max_value=2.0,
                                               step=0.1)
        hurst_trend_thresh = st.number_input("Hurst Trending Threshold", value=0.5, min_value=0.3, max_value=0.7,
                                             step=0.05)
        hurst_range_thresh = st.number_input("Hurst Ranging Threshold", value=0.5, min_value=0.3, max_value=0.7,
                                             step=0.05)

        st.markdown("**📈 Momentum Score Thresholds**")
        momentum_strong_thresh = st.number_input("Strong Momentum Threshold", value=2.0, min_value=1.0, max_value=3.0,
                                                 step=0.5)
        momentum_weak_thresh = st.number_input("Weak Momentum Threshold", value=1.0, min_value=0.5, max_value=2.0,
                                               step=0.5)

    # RSI, Williams %R, CCI thresholds
    st.markdown("**🎛️ Technical Indicator Thresholds**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        rsi_overbought = st.number_input("RSI Overbought", value=60, min_value=50, max_value=80)
        rsi_oversold = st.number_input("RSI Oversold", value=40, min_value=20, max_value=50)
    with col2:
        williams_overbought = st.number_input("Williams %R Overbought", value=-25, min_value=-40, max_value=-10)
        williams_oversold = st.number_input("Williams %R Oversold", value=-75, min_value=-90, max_value=-60)
    with col3:
        cci_overbought = st.number_input("CCI Overbought", value=100, min_value=50, max_value=200)
        cci_oversold = st.number_input("CCI Oversold", value=-100, min_value=-200, max_value=-50)
    with col4:
        williams_period = st.number_input("Williams %R Period", value=14, min_value=5, max_value=30)
        cci_period = st.number_input("CCI Period", value=20, min_value=10, max_value=50)

    if abs(carry_weight + value_weight + momentum_weight - 100) > 0.01:
        st.error("⚠️ Strategy weights must sum to exactly 100%")
        st.stop()

# Data preparation
start_date = start_date_input - timedelta(days=6 * 30)  # 6 months prior for indicator calculation
start_date_str = start_date.strftime("%Y-%m-%d")


# Import data with progress indicators
with st.spinner("Loading market data..."):
    # Import data (same structure as Strat_Ponctual.py)
    _2yUS = fred_import(assets_dict["2y US"], start_date_str)
    _2yUS.columns = ["2y"]

    _5yUS = get_data(assets_dict["5y US"], start_date_str)
    _5yUS.columns = ["5y", "High", "Low", "Open"]  # Fixed column naming as requested

    _5yUS_real = fred_import(assets_dict["5y US Real"], start_date_str)
    _5yUS_real.columns = ["5y_Real"]
    _5yUS_real = _5yUS_real.interpolate(method="polynomial", order=2)

    _5yUS_fut = get_data(assets_dict["5y US Future"], start_date_str)

# Build backtest data
backtest_data = _2yUS.join(_5yUS_real).join(_5yUS)
backtest_data.dropna(inplace=True)
indicators = build_indicators(backtest_data)

# Calculate Z-scores and percentiles
for cols in ["5y_Real", "carry_normalized", "momentum"]:
    indicators[f"{cols}_z"] = zscore(indicators[cols], 63)
    indicators[f"{cols}_percentile"] = indicators[cols].rolling(63).apply(lambda x: percentile_score(x))

indicators["Agg_Percentile"] = (indicators["5y_Real_percentile"] * value_weight +
                                indicators["carry_normalized_percentile"] * carry_weight +
                                indicators["momentum_percentile"] * momentum_weight) / 100

# Trading signals with customizable thresholds
indicators["signal"] = 0
indicators.loc[(indicators["Agg_Percentile"] > buy_threshold) & (indicators["Agg_Percentile"] <= 100), "signal"] = 1
indicators.loc[(indicators["Agg_Percentile"] >= sell_threshold_low) & (
            indicators["Agg_Percentile"] <= sell_threshold_high), "signal"] = -1

# Join with futures data
indicator_full = indicators[["signal", "Agg_Percentile", "5y", "5y_Real_percentile",
                             "carry_normalized_percentile", "momentum_percentile"]].join(_5yUS_fut)
indicator_full.columns = ["signal", "Agg_Percentile", "5y_yield", "Value_Percentile",
                          "Carry_Percentile", "Momentum_Percentile", "Open", "High", "Low", "Close"]
indicator_full.dropna(inplace=True)

# Calculate additional indicators with custom parameters
indicator_full['Fractal_Dim'] = compute_fractal_dimension(indicator_full['Close'], fractal_scaling)
indicator_full['Hurst'] = indicator_full['Close'].rolling(window=hurst_window).apply(calculate_hurst, raw=False)
indicator_full['RSI'] = calculate_rsi(indicator_full['Close'], rsi_period)
indicator_full['Williams_R'] = calculate_williams_r(indicator_full['High'], indicator_full['Low'],
                                                    indicator_full['Close'], williams_period)
indicator_full['CCI'] = calculate_cci(indicator_full['High'], indicator_full['Low'],
                                      indicator_full['Close'], cci_period)

# Calculate momentum score with custom thresholds
indicator_full['Momentum_Score'] = 0
indicator_full['Momentum_Score'] += np.where(indicator_full['RSI'] > rsi_overbought, 1, 0)
indicator_full['Momentum_Score'] += np.where(indicator_full['RSI'] < rsi_oversold, -1, 0)
indicator_full['Momentum_Score'] += np.where(indicator_full['Williams_R'] > williams_overbought, 1, 0)
indicator_full['Momentum_Score'] += np.where(indicator_full['Williams_R'] < williams_oversold, -1, 0)
indicator_full['Momentum_Score'] += np.where(indicator_full['CCI'] > cci_overbought, 1, 0)
indicator_full['Momentum_Score'] += np.where(indicator_full['CCI'] < cci_oversold, -1, 0)

# Classify market regimes with custom thresholds
indicator_full['Market_Regime'] = [
    classify_regime(fd, ms, h, fractal_trend_thresh, fractal_range_thresh,
                    hurst_trend_thresh, hurst_range_thresh, momentum_strong_thresh, momentum_weak_thresh)
    for fd, ms, h in zip(indicator_full['Fractal_Dim'], indicator_full['Momentum_Score'], indicator_full['Hurst'])
]

# Filter data from selected start date
display_data = indicator_full[indicator_full.index >= start_date_input.strftime('%Y-%m-%d')]

# Calculate strategy performance metrics
display_data['price_change'] = display_data['Close'].pct_change()
display_data['strategy_return'] = display_data['price_change'] * display_data['signal'].shift(1)
display_data['cum_return'] = (1 + display_data['strategy_return']).cumprod() - 1
total_return = display_data['cum_return'].iloc[-1] * 100 if not pd.isna(display_data['cum_return'].iloc[-1]) else 0

# Main Dashboard
st.markdown("---")

# Key Performance Metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Strategy Return", f"{total_return:.2f}%")

with col2:
    buy_signals = len(display_data[display_data["signal"] == 1])
    st.metric("Buy Signals", buy_signals)

with col3:
    sell_signals = len(display_data[display_data["signal"] == -1])
    st.metric("Sell Signals", sell_signals)

with col4:
    current_agg_percentile = display_data['Agg_Percentile'].iloc[-1]
    st.metric("Current Agg Percentile", f"{current_agg_percentile:.1f}%")

with col5:
    current_regime = display_data['Market_Regime'].iloc[-1]
    st.metric("Current Regime", current_regime)

# Main Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 5Y US Futures with Trading Signals")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=display_data.index,
        y=display_data["Close"],
        mode="lines",
        name="Futures Price",
        line=dict(color='#1f77b4', width=2)
    ))

    buy_points = display_data[display_data["signal"] == 1]
    if len(buy_points) > 0:
        fig1.add_trace(go.Scatter(
            x=buy_points.index,
            y=buy_points["Close"],
            mode="markers",
            marker=dict(symbol="triangle-up", color="green", size=10),
            name=f"Buy Signals ({len(buy_points)})"
        ))

    sell_points = display_data[display_data["signal"] == -1]
    if len(sell_points) > 0:
        fig1.add_trace(go.Scatter(
            x=sell_points.index,
            y=sell_points["Close"],
            mode="markers",
            marker=dict(symbol="triangle-down", color="red", size=10),
            name=f"Sell Signals ({len(sell_points)})"
        ))

    fig1.update_layout(height=400, showlegend=True,
                       title="Trading Signals on 5Y Treasury Futures",
                       yaxis_title="Price", xaxis_title="Date")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("📊 Aggregate Percentile & 5Y Yield")
    fig2 = go.Figure()

    # Create secondary y-axis
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

    # Agg Percentile (left axis)
    fig2.add_trace(go.Scatter(
        x=display_data.index,
        y=display_data["Agg_Percentile"],
        mode="lines",
        name="Agg Percentile",
        line=dict(color='purple', width=2)
    ), secondary_y=False)

    # 5Y Yield (right axis)
    fig2.add_trace(go.Scatter(
        x=display_data.index,
        y=display_data["5y_yield"],
        mode="lines",
        name="5Y Yield",
        line=dict(color='orange', width=2)
    ), secondary_y=True)

    # Add shaded zones for buy/sell signals
    fig2.add_shape(type="rect", x0=display_data.index[0], x1=display_data.index[-1],
                   y0=buy_threshold, y1=100, fillcolor="rgba(0, 255, 0, 0.1)",
                   layer="below", line_width=0)
    fig2.add_shape(type="rect", x0=display_data.index[0], x1=display_data.index[-1],
                   y0=sell_threshold_low, y1=sell_threshold_high, fillcolor="rgba(255, 0, 0, 0.1)",
                   layer="below", line_width=0)

    fig2.update_yaxes(title_text="Aggregate Percentile (%)", secondary_y=False, range=[0, 100])
    fig2.update_yaxes(title_text="5Y Yield (%)", secondary_y=True)
    fig2.update_layout(height=400, title="Strategy Indicators", xaxis_title="Date")
    st.plotly_chart(fig2, use_container_width=True)

# Latest Indicators Table (like in Strat_Ponctual.py)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Current Indicator Values")
    latest_data = display_data.iloc[-1]
    latest_table = pd.DataFrame({
        "Indicator": ["Value Percentile", "Carry Percentile", "Momentum Percentile", "Aggregate Percentile"],
        "Current Value (%)": [
            latest_data["Value_Percentile"],
            latest_data["Carry_Percentile"],
            latest_data["Momentum_Percentile"],
            latest_data["Agg_Percentile"]
        ],
        "Weight (%)": [value_weight, carry_weight, momentum_weight, 100.0]
    })
    st.dataframe(latest_table.round(1), hide_index=True, use_container_width=True)

with col2:
    st.subheader("🔬 Current Market Microstructure")
    microstructure_table = pd.DataFrame({
        "Indicator": ["Fractal Dimension", "Hurst Exponent", "Momentum Score", "RSI", "Williams %R", "CCI"],
        "Current Value": [
            f"{latest_data['Fractal_Dim']:.3f}",
            f"{latest_data['Hurst']:.3f}",
            f"{latest_data['Momentum_Score']:.0f}",
            f"{latest_data['RSI']:.1f}",
            f"{latest_data['Williams_R']:.1f}",
            f"{latest_data['CCI']:.1f}"
        ],
        "Interpretation": [
            "Trending" if latest_data['Fractal_Dim'] > fractal_trend_thresh else "Ranging" if latest_data[
                                                                                                  'Fractal_Dim'] < fractal_range_thresh else "Mixed",
            "Persistent" if latest_data['Hurst'] > hurst_trend_thresh else "Mean Reverting" if latest_data[
                                                                                                   'Hurst'] < hurst_range_thresh else "Random",
            "Bullish" if latest_data['Momentum_Score'] > 0 else "Bearish" if latest_data[
                                                                                 'Momentum_Score'] < 0 else "Neutral",
            "Overbought" if latest_data['RSI'] > rsi_overbought else "Oversold" if latest_data[
                                                                                       'RSI'] < rsi_oversold else "Neutral",
            "Overbought" if latest_data['Williams_R'] > williams_overbought else "Oversold" if latest_data[
                                                                                                   'Williams_R'] < williams_oversold else "Neutral",
            "Overbought" if latest_data['CCI'] > cci_overbought else "Oversold" if latest_data[
                                                                                       'CCI'] < cci_oversold else "Neutral"
        ]
    })
    st.dataframe(microstructure_table, hide_index=True, use_container_width=True)

# Technical Analysis Charts
st.subheader("🔍 Market Microstructure Analysis")

# Fractal Dimension Plot
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=display_data.index,
    y=display_data['Fractal_Dim'],
    mode='lines',
    name='Fractal Dimension',
    line=dict(color='purple', width=2)
))
fig3.add_hline(y=fractal_trend_thresh, line_dash="dash", line_color="green",
               annotation_text=f"Trending Threshold ({fractal_trend_thresh})")
fig3.add_hline(y=fractal_range_thresh, line_dash="dash", line_color="red",
               annotation_text=f"Ranging Threshold ({fractal_range_thresh})")
fig3.update_layout(
    title="Fractal Dimension Index (Market Efficiency Measure)",
    height=350,
    yaxis_title="Fractal Dimension",
    xaxis_title="Date"
)
st.plotly_chart(fig3, use_container_width=True)

# Momentum and Hurst
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Momentum Score")
    fig4 = go.Figure()
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in display_data['Momentum_Score']]
    fig4.add_trace(go.Bar(
        x=display_data.index,
        y=display_data['Momentum_Score'],
        marker_color=colors,
        name="Momentum Score"
    ))
    fig4.add_hline(y=momentum_strong_thresh, line_dash="dash", line_color="green")
    fig4.add_hline(y=-momentum_strong_thresh, line_dash="dash", line_color="red")
    fig4.update_layout(height=300, yaxis=dict(range=[-3, 3]),
                       title=f"Combined Momentum Score (RSI + Williams %R + CCI)",
                       yaxis_title="Score", xaxis_title="Date")
    st.plotly_chart(fig4, use_container_width=True)

with col2:
    st.subheader("🌊 Hurst Exponent")
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=display_data.index,
        y=display_data['Hurst'],
        mode='lines',
        name='Hurst Exponent',
        line=dict(color='blue', width=2)
    ))
    fig5.add_hline(y=hurst_trend_thresh, line_dash="dash", line_color="green",
                   annotation_text=f"Trending Threshold ({hurst_trend_thresh})")
    fig5.add_hline(y=hurst_range_thresh, line_dash="dash", line_color="red",
                   annotation_text=f"Mean Reverting Threshold ({hurst_range_thresh})")
    fig5.update_layout(height=300,
                       title="Hurst Exponent (Persistence Measure)",
                       yaxis_title="Hurst Exponent", xaxis_title="Date")
    st.plotly_chart(fig5, use_container_width=True)

# Market Regime Analysis
st.markdown("---")
st.subheader("🎯 Market Regime Analysis & Strategy Performance")

# Create regime summary with enhanced statistics
regime_data = display_data.groupby('Market_Regime').agg({
    'Agg_Percentile': ['count', 'mean', 'std', 'min', 'max'],
    'Value_Percentile': ['mean', 'std'],
    'Carry_Percentile': ['mean', 'std'],
    'Momentum_Percentile': ['mean', 'std'],
    'signal': ['sum', lambda x: (x != 0).sum()],  # Total signals and non-zero signals
    'strategy_return': ['sum', 'mean', 'std'],
    'Fractal_Dim': 'mean',
    'Hurst': 'mean',
    'Momentum_Score': 'mean'
}).round(3)

# Flatten column names
regime_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in regime_data.columns]
regime_data
# Create a clean summary table for display
regime_summary = pd.DataFrame({
    'Days': regime_data['Agg_Percentile_count'].astype(int),
    'Avg_Agg_Percentile': regime_data['Agg_Percentile_mean'].round(1),
    'Std_Agg_Percentile': regime_data['Agg_Percentile_std'].round(1),
    'Min_Agg_Percentile': regime_data['Agg_Percentile_min'].round(1),
    'Max_Agg_Percentile': regime_data['Agg_Percentile_max'].round(1),
    'Avg_Value_Percentile': regime_data['Value_Percentile_mean'].round(1),
    'Avg_Carry_Percentile': regime_data['Carry_Percentile_mean'].round(1),
    'Avg_Momentum_Percentile': regime_data['Momentum_Percentile_mean'].round(1),
    'Total_Signals': regime_data['signal_sum'].astype(int),
    'Active_Days': regime_data['signal_<lambda_0>'].astype(int),
    'Total_Return_pct': (regime_data['strategy_return_sum'] * 100).round(2),
    'Avg_Daily_Return_bps': (regime_data['strategy_return_mean'] * 10000).round(1),
    'Volatility_bps': (regime_data['strategy_return_std'] * 10000).round(1),
    'Avg_Fractal_Dim': regime_data['Fractal_Dim_mean'].round(3),
    'Avg_Hurst': regime_data['Hurst_mean'].round(3),
    'Avg_Momentum_Score': regime_data['Momentum_Score_mean'].round(1)
})

# Sort by number of days descending
regime_summary = regime_summary.sort_values('Days', ascending=False)

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📊 Regime Performance Summary")

    # Split into two tables for better readability
    st.markdown("**Strategy Performance by Regime**")
    performance_cols = ['Days', 'Total_Signals', 'Active_Days', 'Total_Return_pct',
                        'Avg_Daily_Return_bps', 'Volatility_bps']
    performance_table = regime_summary[performance_cols].copy()
    performance_table.columns = ['Days', 'Signals', 'Active Days', 'Total Return (%)',
                                 'Avg Daily Return (bps)', 'Volatility (bps)']
    st.dataframe(performance_table, use_container_width=True)

    st.markdown("**Aggregate Percentile Ranges by Regime**")
    percentile_cols = ['Days', 'Avg_Agg_Percentile', 'Std_Agg_Percentile',
                       'Min_Agg_Percentile', 'Max_Agg_Percentile']
    percentile_table = regime_summary[percentile_cols].copy()
    percentile_table.columns = ['Days', 'Avg Agg %ile', 'Std Agg %ile', 'Min Agg %ile', 'Max Agg %ile']
    st.dataframe(percentile_table, use_container_width=True)

    st.markdown("**Component Percentile Averages by Regime**")
    component_cols = ['Days', 'Avg_Value_Percentile', 'Avg_Carry_Percentile', 'Avg_Momentum_Percentile']
    component_table = regime_summary[component_cols].copy()
    component_table.columns = ['Days', 'Avg Value %ile', 'Avg Carry %ile', 'Avg Momentum %ile']
    st.dataframe(component_table, use_container_width=True)

with col2:
    # Regime distribution pie chart
    regime_counts = display_data['Market_Regime'].value_counts()
    fig_pie = go.Figure(data=[go.Pie(
        labels=regime_counts.index,
        values=regime_counts.values,
        hole=0.4,
        textinfo='label+percent',
        textposition='outside'
    )])
    fig_pie.update_layout(title="Market Regime Distribution", height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

    # Key insights box
    st.info(f"""
    **Key Insights:**

    📈 **Most Common Regime:** {regime_counts.index[0]} ({regime_counts.iloc[0]} days)

    🎯 **Best Performing Regime:** {regime_summary.loc[regime_summary['Total_Return_pct'].idxmax()].name} 
    ({regime_summary['Total_Return_pct'].max():.2f}% return)

    ⚡ **Most Active Regime:** {regime_summary.loc[regime_summary['Total_Signals'].idxmax()].name} 
    ({regime_summary['Total_Signals'].max()} signals)

    🎲 **Current Regime:** {current_regime}
    """)

# Enhanced regime timeline with performance coloring
st.subheader("🗓️ Market Regime Timeline with Performance")

# Create regime timeline with returns
fig_regime = go.Figure()

regime_colors = {
    'Strong Uptrend': '#00CC00',  # Bright green
    'Strong Downtrend': '#CC0000',  # Bright red
    'Weak Trend': '#87CEEB',  # Sky blue
    'Volatile Range': '#FFA500',  # Orange
    'Calm Range': '#D3D3D3',  # Light gray
    'Momentum Breakout': '#32CD32',  # Lime green
    'Momentum Breakdown': '#DC143C',  # Crimson
    'Transition': '#FFD700',  # Gold
    'Unknown': '#000000'  # Black
}

for regime in display_data['Market_Regime'].unique():
    if pd.notna(regime):
        regime_data_subset = display_data[display_data['Market_Regime'] == regime]
        fig_regime.add_trace(go.Scatter(
            x=regime_data_subset.index,
            y=[regime] * len(regime_data_subset),
            mode='markers',
            name=regime,
            marker=dict(
                color=regime_colors.get(regime, 'gray'),
                size=8,
                opacity=0.7
            ),
            hovertemplate=f'<b>{regime}</b><br>' +
                          'Date: %{x}<br>' +
                          'Agg Percentile: %{customdata[0]:.1f}%<br>' +
                          'Fractal Dim: %{customdata[1]:.3f}<br>' +
                          'Hurst: %{customdata[2]:.3f}<br>' +
                          'Daily Return: %{customdata[3]:.2f}%<extra></extra>',
            customdata=np.column_stack([
                regime_data_subset['Agg_Percentile'],
                regime_data_subset['Fractal_Dim'],
                regime_data_subset['Hurst'],
                regime_data_subset['strategy_return'] * 100
            ])
        ))

fig_regime.update_layout(height=400, showlegend=True,
                         title="Market Regime Evolution Over Time",
                         xaxis_title="Date", yaxis_title="Market Regime")
st.plotly_chart(fig_regime, use_container_width=True)

# Strategy Performance Chart
st.subheader("💰 Cumulative Strategy Performance")
fig_performance = go.Figure()

fig_performance.add_trace(go.Scatter(
    x=display_data.index,
    y=display_data['cum_return'] * 100,
    mode='lines',
    name='Strategy Return',
    line=dict(color='green', width=3),
    fill='tonexty',
    fillcolor='rgba(0,255,0,0.1)'
))

# Add benchmark (buy and hold)
buy_hold_return = ((display_data['Close'] / display_data['Close'].iloc[0]) - 1) * 100
fig_performance.add_trace(go.Scatter(
    x=display_data.index,
    y=buy_hold_return,
    mode='lines',
    name='Buy & Hold',
    line=dict(color='blue', width=2, dash='dash')
))

fig_performance.update_layout(
    height=400,
    title=f"Strategy vs Buy & Hold Performance ({total_return:.2f}% vs {buy_hold_return.iloc[-1]:.2f}%)",
    yaxis_title="Cumulative Return (%)",
    xaxis_title="Date"
)
st.plotly_chart(fig_performance, use_container_width=True)

# Detailed Technical Analysis (Optional)
with st.expander("🔬 Detailed Technical Analysis", expanded=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("RSI Analysis")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=display_data.index, y=display_data['RSI'],
                                     mode='lines', name='RSI', line=dict(color='purple')))
        fig_rsi.add_hline(y=rsi_overbought, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=rsi_oversold, line_dash="dash", line_color="green")
        fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray")
        fig_rsi.update_layout(height=300, yaxis=dict(range=[0, 100]),
                              title=f"RSI ({rsi_period} periods)")
        st.plotly_chart(fig_rsi, use_container_width=True)

    with col2:
        st.subheader("Williams %R Analysis")
        fig_wlr = go.Figure()
        fig_wlr.add_trace(go.Scatter(x=display_data.index, y=display_data['Williams_R'],
                                     mode='lines', name='Williams %R', line=dict(color='orange')))
        fig_wlr.add_hline(y=williams_overbought, line_dash="dash", line_color="red")
        fig_wlr.add_hline(y=williams_oversold, line_dash="dash", line_color="green")
        fig_wlr.add_hline(y=-50, line_dash="dot", line_color="gray")
        fig_wlr.update_layout(height=300, yaxis=dict(range=[-100, 0]),
                              title=f"Williams %R ({williams_period} periods)")
        st.plotly_chart(fig_wlr, use_container_width=True)

    with col3:
        st.subheader("CCI Analysis")
        fig_cci = go.Figure()
        fig_cci.add_trace(go.Scatter(x=display_data.index, y=display_data['CCI'],
                                     mode='lines', name='CCI', line=dict(color='red')))
        fig_cci.add_hline(y=cci_overbought, line_dash="dash", line_color="red")
        fig_cci.add_hline(y=cci_oversold, line_dash="dash", line_color="green")
        fig_cci.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_cci.update_layout(height=300, title=f"CCI ({cci_period} periods)")
        st.plotly_chart(fig_cci, use_container_width=True)

# Export and Analysis Tools
st.markdown("---")
st.subheader("📁 Data Export & Analysis Tools")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📊 Show Full Dataset", use_container_width=True):
        st.dataframe(display_data, use_container_width=True)

with col2:
    csv_data = display_data.to_csv(index=True)
    st.download_button(
        label="📥 Download Full Dataset",
        data=csv_data,
        file_name=f"market_regime_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col3:
    regime_csv = regime_summary.to_csv(index=True)
    st.download_button(
        label="📥 Download Regime Summary",
        data=regime_csv,
        file_name=f"regime_summary_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col4:
    # Create a strategy configuration summary
    config_summary = pd.DataFrame({
        'Parameter': ['Buy Threshold', 'Sell Threshold High', 'Sell Threshold Low',
                      'Value Weight', 'Carry Weight', 'Momentum Weight',
                      'Fractal Trend Threshold', 'Hurst Trend Threshold',
                      'RSI Overbought', 'RSI Oversold', 'Williams Overbought', 'Williams Oversold',
                      'CCI Overbought', 'CCI Oversold'],
        'Value': [buy_threshold, sell_threshold_high, sell_threshold_low,
                  value_weight, carry_weight, momentum_weight,
                  fractal_trend_thresh, hurst_trend_thresh,
                  rsi_overbought, rsi_oversold, williams_overbought, williams_oversold,
                  cci_overbought, cci_oversold]
    })
    config_csv = config_summary.to_csv(index=False)
    st.download_button(
        label="📥 Download Configuration",
        data=config_csv,
        file_name=f"strategy_config_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Footer with methodology summary
st.markdown("---")
with st.expander("📚 Methodology & Interpretation Guide", expanded=False):
    st.markdown("""
    ## Regime Classification Logic

    The dashboard classifies market regimes using a **three-indicator approach**:

    ### Primary Classification Rules:

    1. **Strong Uptrend/Downtrend**:
       - Fractal Dimension > {fractal_trend_thresh} (trending market)
       - Hurst Exponent > {hurst_trend_thresh} (persistent behavior)
       - Momentum Score > ±{momentum_strong_thresh} (strong directional momentum)

    2. **Weak Trend**:
       - Fractal Dimension > {fractal_trend_thresh} AND Hurst > {hurst_trend_thresh}
       - BUT Momentum Score between ±{momentum_weak_thresh}

    3. **Volatile/Calm Range**:
       - Fractal Dimension < {fractal_range_thresh} (ranging market)
       - Hurst Exponent < {hurst_range_thresh} (mean reverting)
       - Volatile if |Momentum Score| > {momentum_strong_thresh}, Calm otherwise

    4. **Momentum Breakout/Breakdown**:
       - Mixed fractal/Hurst signals
       - Strong momentum (±{momentum_strong_thresh}) suggesting regime transition

    5. **Transition**:
       - Mixed signals across all indicators
       - Market in uncertain state between regimes

    ### Key Performance Metrics:

    - **Total Signals**: Number of buy/sell signals generated in each regime
    - **Active Days**: Days with non-zero position (signal ≠ 0)
    - **Total Return**: Cumulative strategy return during regime periods
    - **Daily Return (bps)**: Average daily return in basis points
    - **Volatility (bps)**: Standard deviation of daily returns

    ### Practical Applications:

    1. **Signal Filtering**: Avoid trading in certain regimes or adjust position sizes
    2. **Threshold Optimization**: Adjust Agg_Percentile thresholds by regime
    3. **Risk Management**: Higher volatility regimes may warrant smaller positions
    4. **Performance Attribution**: Understand which market conditions drive returns

    *Note: This analysis is for educational/research purposes. Past performance does not guarantee future results.*
    """.format(
        fractal_trend_thresh=fractal_trend_thresh,
        hurst_trend_thresh=hurst_trend_thresh,
        momentum_strong_thresh=momentum_strong_thresh,
        fractal_range_thresh=fractal_range_thresh,
        hurst_range_thresh=hurst_range_thresh,
        momentum_weak_thresh=momentum_weak_thresh
    ))

