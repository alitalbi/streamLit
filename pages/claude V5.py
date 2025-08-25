import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
from statsmodels.tsa.stattools import adfuller
import warnings


warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Regime Analysis", page_icon="📊")

# Initialize FRED API
fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')


@st.cache_data
def get_github_data(ticker, start_date, end_date):
    """Fetch data from GitHub repository"""
    ticker_request = ticker.replace("=", "%3D")
    try:
        url = f"https://raw.githubusercontent.com/alitalbi/storage_data_fy/refs/heads/master/{ticker_request}.csv"
        df = pd.read_csv(url, usecols=["Date", "Close", "High", "Low", "Open"])
        df.set_index("Date", inplace=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.loc[(df.index >= start_date) & (df.index <= end_date)]
    except Exception as e:
        st.error(f"Error loading {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data
def get_fred_data(ticker, start_date, end_date):
    """Import data from FRED"""
    try:
        data = fred.get_series(ticker, observation_start=start_date, observation_end=end_date, freq="daily")
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading FRED {ticker}: {e}")
        return pd.DataFrame()


def percentile_score(window):
    """Calculate percentile score"""
    if len(window) == 0:
        return np.nan
    current_value = window[-1]
    return (np.sum(window <= current_value) / len(window)) * 100


def build_indicators(data):
    """Build carry and momentum indicators"""
    data["5_2y"] = data["5y"] - data["2y"]
    data["carry_normalized"] = data["5_2y"] / data["5_2y"].rolling(75).std()
    data["momentum"] = data["5y"].rolling(5).mean() - data["5y"].rolling(20).mean()
    return data


def calculate_zscores(data, lookback_zscore=252):
    """Calculate z-scores for indicators"""
    data_copy = data.copy()

    # Calculate z-scores for each indicator
    for col in ["5y_Real", "carry_normalized", "momentum"]:
        if col in data_copy.columns:
            rolling_mean = data_copy[col].rolling(lookback_zscore).mean()
            rolling_std = data_copy[col].rolling(lookback_zscore).std()
            data_copy[f"{col}_zscore"] = (data_copy[col] - rolling_mean) / rolling_std

    return data_copy


def calculate_agg_percentile(data, weights):
    """Calculate aggregate percentile with given weights"""
    value_w, carry_w, momentum_w = weights
    return (data['Value_Percentile'] * value_w +
            data['Carry_Percentile'] * carry_w +
            data['Momentum_Percentile'] * momentum_w) / 100


def hurst_exponent(ts, max_lag=20):
    """Calculate the Hurst Exponent of a time series"""
    if len(ts) < max_lag * 2:
        return np.nan

    try:
        # Calculate the range of the cumulative deviation from the mean
        ts = np.array(ts)
        N = len(ts)
        if N == 0:
            return np.nan

        # Mean center the series
        mean_ts = ts - np.mean(ts)

        # Cumulative sum
        cumsum_ts = np.cumsum(mean_ts)

        # Range of different lag values
        lags = range(2, min(max_lag, N // 2))
        rs_values = []

        for lag in lags:
            # Split the series into chunks
            n_chunks = N // lag
            if n_chunks == 0:
                continue

            rs_chunk = []
            for i in range(n_chunks):
                start_idx = i * lag
                end_idx = start_idx + lag

                chunk = cumsum_ts[start_idx:end_idx]
                chunk_ts = ts[start_idx:end_idx]

                if len(chunk) == 0:
                    continue

                # Range
                R = np.max(chunk) - np.min(chunk)

                # Standard deviation
                S = np.std(chunk_ts)
                if S == 0:
                    continue

                rs_chunk.append(R / S)

            if rs_chunk:
                rs_values.append(np.mean(rs_chunk))

        if len(rs_values) < 3:
            return np.nan

        # Linear regression of log(R/S) vs log(lag)
        log_lags = np.log(lags[:len(rs_values)])
        log_rs = np.log(rs_values)

        # Remove any infinite or NaN values
        mask = np.isfinite(log_lags) & np.isfinite(log_rs)
        if np.sum(mask) < 3:
            return np.nan

        log_lags = log_lags[mask]
        log_rs = log_rs[mask]

        # Calculate Hurst exponent as the slope
        hurst = np.polyfit(log_lags, log_rs, 1)[0]

        # Ensure reasonable bounds
        return np.clip(hurst, 0, 1)

    except Exception as e:
        return np.nan


def adf_test_result(ts):
    """Perform ADF test and return p-value"""
    if len(ts) < 10:
        return np.nan
    try:
        result = adfuller(ts.dropna())
        return result[1]  # p-value
    except:
        return np.nan


def calculate_keltner_channels(data, ema_period=20, atr_period=10, multiplier=2):
    """Calculate Keltner Channels"""
    data = data.copy()

    # Calculate EMA
    data['EMA'] = data['Close'].ewm(span=ema_period).mean()

    # Calculate ATR
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=atr_period).mean()

    # Calculate channels
    data['KC_Upper'] = data['EMA'] + (multiplier * data['ATR'])
    data['KC_Lower'] = data['EMA'] - (multiplier * data['ATR'])

    return data


def calculate_rsi(data, window=14):
    """Calculate RSI"""

    # Manual RSI calculation if talib fails
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# Header
st.title("📊 Regime Analysis Dashboard")
st.caption("Analyze market regimes using ADF test, Hurst exponent, and technical indicators")

# Configuration
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**📅 Date Range**")
    start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=365 * 2))
    end_date = st.date_input("End Date", datetime.now().date())

    if start_date >= end_date:
        st.error("Start date must be before end date")
        st.stop()

with col2:
    st.markdown("**⚖️ Aggregate Percentile Weights**")
    value_weight = st.slider("Value Weight (%)", 1, 98, 33)
    carry_weight = st.slider("Carry Weight (%)", 1, 99 - value_weight, 33)
    momentum_weight = 100 - value_weight - carry_weight
    st.caption(f"Momentum: {momentum_weight}%")

with col3:
    st.markdown("**📊 Analysis Settings**")
    freq_choice = st.selectbox("Calculation Frequency", ["Daily", "Weekly"])
    rolling_window = st.number_input("Rolling Window (days)", 30, 252, 63)
    show_keltner = st.checkbox("Show Keltner Channels")

with col4:
    st.markdown("**🔄 Technical Indicators**")
    show_rsi = st.checkbox("Show RSI")
    rsi_window = st.number_input("RSI Window", 5, 50, 14) if show_rsi else 14
    keltner_multiplier = st.number_input("Keltner Multiplier", 1.0, 3.0, 2.0, 0.1) if show_keltner else 2.0

# Load data
if st.button("📈 Load & Analyze Data", type="primary"):
    with st.spinner("Loading data..."):
        # Load futures data
        futures_data = get_github_data("ZF=F", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

        # Load yield data for aggregate percentile
        _2yUS = get_fred_data("DGS2", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        _2yUS.columns = ["2y"]

        _5yUS = get_github_data("^FVX", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        _5yUS.columns = ["5y", "High", "Low", "Open"]

        _5yUS_real = get_fred_data("DFII5", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        _5yUS_real.columns = ["5y_Real"]
        _5yUS_real = _5yUS_real.interpolate(method="polynomial", order=2)

        if futures_data.empty:
            st.error("Failed to load futures data")
            st.stop()

        # Build indicators for percentile calculation
        if not any(df.empty for df in [_2yUS, _5yUS_real, _5yUS]):
            backtest_data = _2yUS.join(_5yUS_real).join(_5yUS).dropna()
            indicators = build_indicators(backtest_data)

            # Calculate z-scores first (using 1-year lookback for z-score calculation)
            indicators_with_zscores = calculate_zscores(indicators, lookback_zscore=252)

            # Calculate percentiles of z-scores (with 3-month lookback for percentiles)
            lookback_percentile = 63  # ~3 months
            for col in ["5y_Real", "carry_normalized", "momentum"]:
                zscore_col = f"{col}_zscore"
                if zscore_col in indicators_with_zscores.columns:
                    indicators_with_zscores[f"{col}_percentile"] = indicators_with_zscores[zscore_col].rolling(
                        lookback_percentile).apply(percentile_score)

            # Join with futures and calculate aggregate percentile
            final_data = indicators_with_zscores[
                ["5y_Real_percentile", "carry_normalized_percentile", "momentum_percentile"]].join(futures_data)
            final_data.columns = ["Value_Percentile", "Carry_Percentile", "Momentum_Percentile", "Open", "High", "Low",
                                  "Close"]
            final_data = final_data.dropna()

            # Add z-scores to final data for display
            for col in ["5y_Real", "carry_normalized", "momentum"]:
                zscore_col = f"{col}_zscore"
                if zscore_col in indicators_with_zscores.columns:
                    final_data[zscore_col] = indicators_with_zscores[zscore_col]

            # Calculate aggregate percentile
            weights = [value_weight, carry_weight, momentum_weight]
            final_data['Agg_Percentile'] = calculate_agg_percentile(final_data, weights)

        else:
            final_data = futures_data
            final_data['Agg_Percentile'] = np.nan
            st.warning("Could not calculate aggregate percentile - using futures data only")

        st.success(f"Loaded {len(final_data)} days of data from {start_date} to {end_date}")

        # Calculate Keltner Channels if requested
        if show_keltner:
            final_data = calculate_keltner_channels(final_data)

        # Calculate RSI if requested
        if show_rsi:
            final_data['RSI'] = calculate_rsi(final_data, rsi_window)

        # Calculate rolling ADF and Hurst
        st.info("Calculating rolling ADF test and Hurst exponent...")
        progress_bar = st.progress(0)

        prices = final_data['Close']

        if freq_choice == "Weekly":
            # Resample to weekly for calculation, then map back to daily
            weekly_prices = prices.resample('W').last().dropna()

            adf_results = []
            hurst_results = []
            dates = []

            for i in range(rolling_window, len(weekly_prices)):
                if i % 10 == 0:
                    progress_bar.progress(i / len(weekly_prices))

                window_data = weekly_prices.iloc[i - rolling_window:i]

                # ADF test
                adf_p = adf_test_result(window_data)
                adf_results.append(adf_p)

                # Hurst exponent
                hurst = hurst_exponent(window_data.values)
                hurst_results.append(hurst)

                dates.append(weekly_prices.index[i])

            # Create weekly results dataframe
            weekly_results = pd.DataFrame({
                'ADF_pvalue': adf_results,
                'Hurst': hurst_results
            }, index=dates)

            # Forward fill to daily frequency
            final_data['ADF_pvalue'] = weekly_results['ADF_pvalue'].reindex(final_data.index, method='ffill')
            final_data['Hurst'] = weekly_results['Hurst'].reindex(final_data.index, method='ffill')

        else:  # Daily
            adf_results = []
            hurst_results = []

            for i in range(rolling_window, len(prices)):
                if i % 50 == 0:
                    progress_bar.progress(i / len(prices))

                window_data = prices.iloc[i - rolling_window:i]

                # ADF test
                adf_p = adf_test_result(window_data)
                adf_results.append(adf_p)

                # Hurst exponent
                hurst = hurst_exponent(window_data.values)
                hurst_results.append(hurst)

            # Pad with NaN for initial window
            final_data['ADF_pvalue'] = [np.nan] * rolling_window + adf_results
            final_data['Hurst'] = [np.nan] * rolling_window + hurst_results

        progress_bar.progress(1.0)

        # Store in session state
        st.session_state.regime_data = final_data
        st.session_state.config = {
            'weights': weights,
            'show_keltner': show_keltner,
            'show_rsi': show_rsi,
            'freq_choice': freq_choice,
            'rolling_window': rolling_window
        }

# Display results
if 'regime_data' in st.session_state:
    data = st.session_state.regime_data
    config = st.session_state.config

    st.markdown("---")

    # Price and Aggregate Percentile Charts
    st.subheader("📈 Price Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### 5Y US Futures Price")

        fig_price = go.Figure()

        # Main price line
        fig_price.add_trace(go.Scatter(
            x=data.index, y=data['Close'],
            mode='lines', name='5Y Futures',
            line=dict(color='black', width=1.5)
        ))

        # Keltner Channels
        if config['show_keltner'] and 'KC_Upper' in data.columns:
            fig_price.add_trace(go.Scatter(
                x=data.index, y=data['KC_Upper'],
                mode='lines', name='KC Upper',
                line=dict(color='blue', width=1, dash='dash'),
                opacity=0.6
            ))

            fig_price.add_trace(go.Scatter(
                x=data.index, y=data['KC_Lower'],
                mode='lines', name='KC Lower',
                line=dict(color='blue', width=1, dash='dash'),
                opacity=0.6, fill='tonexty', fillcolor='rgba(0,100,200,0.1)'
            ))

            fig_price.add_trace(go.Scatter(
                x=data.index, y=data['EMA'],
                mode='lines', name='EMA',
                line=dict(color='blue', width=1)
            ))

        fig_price.update_layout(
            height=400, xaxis_title="Date", yaxis_title="Price",
            title=f"Weights: Value={config['weights'][0]}% Carry={config['weights'][1]}% Momentum={config['weights'][2]}%"
        )
        st.plotly_chart(fig_price, use_container_width=True)

    with col2:
        st.markdown("### Aggregate Percentile")

        if not data['Agg_Percentile'].isna().all():
            fig_agg = go.Figure()

            fig_agg.add_trace(go.Scatter(
                x=data.index, y=data['Agg_Percentile'],
                mode='lines', name='Agg Percentile',
                line=dict(color='purple', width=2)
            ))

            # Add regime zones
            fig_agg.add_hrect(y0=80, y1=100, fillcolor="green", opacity=0.2, annotation_text="Buy Zone")
            fig_agg.add_hrect(y0=0, y1=20, fillcolor="red", opacity=0.2, annotation_text="Sell Zone")
            fig_agg.add_hrect(y0=20, y1=80, fillcolor="yellow", opacity=0.1, annotation_text="Neutral Zone")

            fig_agg.update_layout(
                height=400, xaxis_title="Date", yaxis_title="Percentile (%)",
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_agg, use_container_width=True)
        else:
            st.info("Aggregate percentile data not available")

    # RSI Chart
    if config['show_rsi'] and 'RSI' in data.columns:
        st.subheader("🔄 RSI Analysis")

        fig_rsi = go.Figure()

        fig_rsi.add_trace(go.Scatter(
            x=data.index, y=data['RSI'],
            mode='lines', name='RSI',
            line=dict(color='orange', width=2)
        ))

        # RSI levels
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral (50)")

        fig_rsi.update_layout(
            height=300, xaxis_title="Date", yaxis_title="RSI",
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

    # Regime Analysis Charts
    st.subheader(f"📊 Regime Analysis ({config['freq_choice']} calculation, {config['rolling_window']}-day window)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ADF Test P-Value")

        fig_adf = go.Figure()

        valid_adf = data['ADF_pvalue'].dropna()
        if len(valid_adf) > 0:
            fig_adf.add_trace(go.Scatter(
                x=valid_adf.index, y=valid_adf.values,
                mode='lines', name='ADF p-value',
                line=dict(color='red', width=2)
            ))

            # Significance levels
            fig_adf.add_hline(y=0.05, line_dash="dash", line_color="black",
                              annotation_text="5% significance")
            fig_adf.add_hline(y=0.01, line_dash="dash", line_color="darkred",
                              annotation_text="1% significance")

            # Color background based on stationarity
            stationary_data = data[data['ADF_pvalue'] < 0.05]
            if len(stationary_data) > 0:
                for i, (idx, row) in enumerate(stationary_data.iterrows()):
                    if i == 0:
                        fig_adf.add_vrect(
                            x0=idx, x1=idx + timedelta(days=1),
                            fillcolor="lightgreen", opacity=0.3,
                            annotation_text="Stationary" if i == 0 else "",
                            layer="below", line_width=0
                        )

        fig_adf.update_layout(
            height=400, xaxis_title="Date", yaxis_title="P-Value",
            title="Lower values indicate stationarity (mean reversion)"
        )
        st.plotly_chart(fig_adf, use_container_width=True)

    with col2:
        st.markdown("### Hurst Exponent")

        fig_hurst = go.Figure()

        valid_hurst = data['Hurst'].dropna()
        if len(valid_hurst) > 0:
            fig_hurst.add_trace(go.Scatter(
                x=valid_hurst.index, y=valid_hurst.values,
                mode='lines', name='Hurst Exponent',
                line=dict(color='blue', width=2)
            ))

            # Reference lines
            fig_hurst.add_hline(y=0.5, line_dash="solid", line_color="black",
                                annotation_text="Random Walk (0.5)")
            fig_hurst.add_hline(y=0.4, line_dash="dash", line_color="green",
                                annotation_text="Mean Reverting")
            fig_hurst.add_hline(y=0.6, line_dash="dash", line_color="red",
                                annotation_text="Trending")

        fig_hurst.update_layout(
            height=400, xaxis_title="Date", yaxis_title="Hurst Exponent",
            title="H < 0.5: Mean Reverting | H > 0.5: Trending",
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig_hurst, use_container_width=True)

    # Summary Statistics
    st.subheader("📈 Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    valid_adf = data['ADF_pvalue'].dropna()
    valid_hurst = data['Hurst'].dropna()

    with col1:
        if len(valid_adf) > 0:
            stationary_pct = (valid_adf < 0.05).mean() * 100
            st.metric("Stationary Periods", f"{stationary_pct:.1f}%",
                      "Mean reverting behavior")
        else:
            st.metric("Stationary Periods", "N/A")

    with col2:
        if len(valid_hurst) > 0:
            mean_hurst = valid_hurst.mean()
            st.metric("Average Hurst", f"{mean_hurst:.3f}",
                      "Trending" if mean_hurst > 0.5 else "Mean Reverting")
        else:
            st.metric("Average Hurst", "N/A")

    with col3:
        if len(valid_hurst) > 0:
            trending_pct = (valid_hurst > 0.6).mean() * 100
            st.metric("Strong Trending", f"{trending_pct:.1f}%",
                      "H > 0.6")
        else:
            st.metric("Strong Trending", "N/A")

    with col4:
        if len(valid_hurst) > 0:
            mean_rev_pct = (valid_hurst < 0.4).mean() * 100
            st.metric("Strong Mean Reverting", f"{mean_rev_pct:.1f}%",
                      "H < 0.4")
        else:
            st.metric("Strong Mean Reverting", "N/A")

    # Export data
    st.markdown("### 📁 Export Data")
    if st.button("📥 Download Analysis Results"):
        export_data = data[['Close', 'Agg_Percentile', 'ADF_pvalue', 'Hurst']].copy()
        if 'RSI' in data.columns:
            export_data['RSI'] = data['RSI']

        csv_data = export_data.to_csv()
        st.download_button(
            "Download CSV", csv_data,
            f"regime_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

else:
    st.info("Configure settings above and click 'Load & Analyze Data' to begin analysis")

# Information panel
st.markdown("---")
st.markdown("""
### 📋 Interpretation Guide:

**ADF Test (Augmented Dickey-Fuller):**
- **p-value < 0.05**: Series is stationary → **Mean Reverting**
- **p-value > 0.05**: Series is non-stationary → **Trending** or Random Walk

**Hurst Exponent:**
- **H < 0.5**: **Mean Reverting** behavior (anti-persistent)
- **H = 0.5**: **Random Walk** (no memory)
- **H > 0.5**: **Trending** behavior (persistent)

**Trading Implications:**
- **Mean Reverting periods**: Use mean reversion strategies, fade extremes
- **Trending periods**: Use momentum strategies, follow trends  
- **Transition periods**: Exercise caution, consider regime change
""")