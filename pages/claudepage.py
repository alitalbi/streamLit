import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
from statsmodels.tsa.stattools import adfuller
import warnings
import json

warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Regime Analysis", page_icon="📊")

# Initialize FRED API
fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')


@st.cache_data
def get_github_data(ticker, start_date, end_date):
    """
    Fetch futures price data from GitHub repository.
    This function handles the URL encoding and data processing needed
    to retrieve historical price data for analysis.
    """
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
    """
    Import economic data from FRED (Federal Reserve Economic Data).
    This is used for yield curve data that forms the foundation
    of our value and carry indicators.
    """
    try:
        data = fred.get_series(ticker, observation_start=start_date, observation_end=end_date, freq="daily")
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading FRED {ticker}: {e}")
        return pd.DataFrame()


def percentile_score(window):
    """
    Calculate the percentile rank of the most recent value within a rolling window.
    This tells us where the current value sits relative to recent history.
    A value of 90% means the current reading is higher than 90% of recent observations.
    """
    if len(window) == 0:
        return np.nan
    current_value = window[-1]
    return (np.sum(window <= current_value) / len(window)) * 100


def kalman_filter_1d(data, process_variance=1e-5, measurement_variance=1e-1):
    """
    Apply a 1D Kalman filter to smooth time series data while preserving trends.

    The Kalman filter is a recursive algorithm that estimates the true underlying
    signal from noisy observations. It works by:
    1. Making a prediction based on the previous state
    2. Updating this prediction based on new observations
    3. Balancing between the prediction and observation based on their uncertainties

    Parameters:
    - process_variance (Q): How much the true value can change between periods
    - measurement_variance (R): How much noise we expect in our observations

    Lower process variance creates smoother signals but slower response to changes.
    Higher measurement variance assumes more noise and applies more smoothing.
    """
    if len(data) == 0:
        return data.copy()

    filtered_data = []

    # Initialize with first observation
    x_est = data.iloc[0]  # Initial state estimate
    P_est = 1.0  # Initial estimation error covariance

    Q = process_variance  # Process noise covariance
    R = measurement_variance  # Measurement noise covariance

    for observation in data:
        # Prediction step: where do we think the signal is going?
        x_pred = x_est  # State prediction (assuming constant velocity model)
        P_pred = P_est + Q  # Prediction error covariance

        # Update step: incorporate new observation
        K = P_pred / (P_pred + R)  # Kalman gain (how much to trust new observation)
        x_est = x_pred + K * (observation - x_pred)  # Updated state estimate
        P_est = (1 - K) * P_pred  # Updated estimation error covariance

        filtered_data.append(x_est)

    return pd.Series(filtered_data, index=data.index)


def build_indicators(data, use_kalman=True, kalman_process_var=1e-5, kalman_measurement_var=1e-1):
    """
    Build the fundamental trading indicators from yield curve data.

    These indicators capture different aspects of market behavior:
    1. Carry: The yield spread (5Y-2Y) normalized by volatility
    2. Momentum: Short-term vs long-term moving average difference

    Optional Kalman filtering reduces noise while preserving underlying trends.
    """
    # Calculate raw carry indicator (yield curve steepness)
    data["5_2y"] = data["5y"] - data["2y"]
    data["carry_normalized"] = data["5_2y"] / data["5_2y"].rolling(75).std()

    # Calculate momentum indicator (short vs long moving averages)
    data["momentum"] = data["5y"].rolling(5).mean() - data["5y"].rolling(20).mean()

    if use_kalman:
        # Apply Kalman filtering to smooth out market noise
        # This helps create more stable signals for regime analysis
        data["5y_Real_filtered"] = kalman_filter_1d(data["5y_Real"], kalman_process_var, kalman_measurement_var)
        data["carry_normalized_filtered"] = kalman_filter_1d(data["carry_normalized"], kalman_process_var,
                                                             kalman_measurement_var)
        data["momentum_filtered"] = kalman_filter_1d(data["momentum"], kalman_process_var, kalman_measurement_var)

        # Replace raw signals with filtered versions
        data["5y_Real"] = data["5y_Real_filtered"]
        data["carry_normalized"] = data["carry_normalized_filtered"]
        data["momentum"] = data["momentum_filtered"]

    return data


def calculate_zscores(data, lookback_zscore=66):
    """
    Calculate z-scores to standardize indicators across different time periods.

    Z-scores tell us how many standard deviations away from the recent average
    each indicator is. This normalization allows us to compare different indicators
    on the same scale and identify when they are at extreme levels.

    A z-score of +2 means the indicator is 2 standard deviations above average,
    which historically occurs only about 5% of the time.
    """
    data_copy = data.copy()

    for col in ["5y_Real", "carry_normalized", "momentum"]:
        if col in data_copy.columns:
            rolling_mean = data_copy[col].rolling(lookback_zscore).mean()
            rolling_std = data_copy[col].rolling(lookback_zscore).std()
            data_copy[f"{col}_zscore"] = (data_copy[col] - rolling_mean) / rolling_std

    return data_copy


def calculate_agg_percentile(data, weights):
    """
    Combine individual indicator percentiles into a single aggregate signal.

    This weighted combination allows us to emphasize different market factors
    based on our trading thesis. The result is a single number from 0-100
    that summarizes the overall market condition.
    """
    value_w, carry_w, momentum_w = weights
    return (data['Value_Percentile'] * value_w +
            data['Carry_Percentile'] * carry_w +
            data['Momentum_Percentile'] * momentum_w) / 100


def hurst_exponent(ts, max_lag=20):
    """
    Calculate the Hurst Exponent to measure market regime characteristics.

    The Hurst exponent tells us about the "memory" in a time series:
    - H < 0.5: Mean reverting behavior (what goes up tends to come down)
    - H = 0.5: Random walk (no predictable pattern)
    - H > 0.5: Trending behavior (momentum persists)

    This is crucial for regime analysis because it tells us whether to apply
    mean reversion or momentum strategies.
    """
    if len(ts) < max_lag * 2:
        return np.nan

    try:
        ts = np.array(ts)
        N = len(ts)
        if N == 0:
            return np.nan

        # Mean center the series
        mean_ts = ts - np.mean(ts)

        # Calculate cumulative sum for range analysis
        cumsum_ts = np.cumsum(mean_ts)

        # Test different lag periods
        lags = range(2, min(max_lag, N // 2))
        rs_values = []

        for lag in lags:
            # Split series into non-overlapping chunks
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

                # Calculate range (R) and standard deviation (S)
                R = np.max(chunk) - np.min(chunk)
                S = np.std(chunk_ts)
                if S == 0:
                    continue

                rs_chunk.append(R / S)

            if rs_chunk:
                rs_values.append(np.mean(rs_chunk))

        if len(rs_values) < 3:
            return np.nan

        # Linear regression to find Hurst exponent
        log_lags = np.log(lags[:len(rs_values)])
        log_rs = np.log(rs_values)

        # Remove infinite or NaN values
        mask = np.isfinite(log_lags) & np.isfinite(log_rs)
        if np.sum(mask) < 3:
            return np.nan

        log_lags = log_lags[mask]
        log_rs = log_rs[mask]

        # The Hurst exponent is the slope of this relationship
        hurst = np.polyfit(log_lags, log_rs, 1)[0]

        # Ensure reasonable bounds
        return np.clip(hurst, 0, 1)

    except Exception as e:
        return np.nan


def adf_test_result(ts):
    """
    Perform Augmented Dickey-Fuller test for stationarity (mean reversion).

    The ADF test tells us whether a time series has a unit root (trending)
    or is stationary (mean reverting). The p-value is key:
    - p < 0.05: Series is stationary (mean reverting)
    - p > 0.05: Series has unit root (trending or random walk)

    This complements the Hurst exponent in regime classification.
    """
    if len(ts) < 10:
        return np.nan
    try:
        result = adfuller(ts.dropna())
        return result[1]  # Return p-value
    except:
        return np.nan


def calculate_keltner_channels(data, ema_period=20, atr_period=10, multiplier=2):
    """
    Calculate Keltner Channels for volatility-based support/resistance levels.

    Keltner Channels use the Average True Range (ATR) to create dynamic
    bands around an exponential moving average. These bands expand during
    volatile periods and contract during calm periods, helping identify
    potential reversal points in mean-reverting regimes.
    """
    data = data.copy()

    # Calculate exponential moving average as centerline
    data['EMA'] = data['Close'].ewm(span=ema_period).mean()

    # Calculate Average True Range for volatility measurement
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=atr_period).mean()

    # Create upper and lower bands
    data['KC_Upper'] = data['EMA'] + (multiplier * data['ATR'])
    data['KC_Lower'] = data['EMA'] - (multiplier * data['ATR'])

    return data


def calculate_rsi(data, window=14):
    """
    Calculate Relative Strength Index manually without external dependencies.

    RSI measures the speed and magnitude of price changes to identify
    overbought (>70) and oversold (<30) conditions. In our regime analysis,
    extreme RSI readings can confirm mean reversion signals.
    """
    try:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except Exception as e:
        st.error(f"Error calculating RSI: {e}")
        return pd.Series(np.nan, index=data.index, name='RSI')


# Streamlit App Layout
st.title("📊 Regime Analysis Dashboard")
st.caption(
    "Analyze market regimes using ADF test, Hurst exponent, and technical indicators with advanced Kalman filtering")

# Configuration Panel
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
    zscore_lookback = st.number_input("Z-Score Lookback (days)", 30, 500, 66)
    percentile_lookback = st.number_input("Percentile Lookback (days)", 30, 126, 63)
    show_keltner = st.checkbox("Show Keltner Channels")

with col4:
    st.markdown("**🔧 Kalman Filter Settings**")
    use_kalman = st.checkbox("Filter Indicators", value=True)
    use_kalman_price = st.checkbox("Filter Futures Prices", value=False)
    if use_kalman or use_kalman_price:
        kalman_process_var = st.number_input("Process Variance", 1e-6, 1e-3, 1e-5, format="%.0e")
        kalman_measurement_var = st.number_input("Measurement Variance", 1e-3, 1.0, 1e-1, format="%.0e")
        st.caption("Lower process variance = smoother signal")
    else:
        kalman_process_var = 1e-5
        kalman_measurement_var = 1e-1

# Additional Settings
col1, col2 = st.columns(2)

with col1:
    st.markdown("**🔄 Technical Indicators**")
    show_rsi = st.checkbox("Show RSI")
    rsi_window = st.number_input("RSI Window", 5, 50, 14) if show_rsi else 14
    keltner_multiplier = st.number_input("Keltner Multiplier", 1.0, 3.0, 2.0, 0.1) if show_keltner else 2.0

# Main Analysis Execution
if st.button("📈 Load & Analyze Data", type="primary"):
    with st.spinner("Loading data..."):
        # Load futures data first - this is our primary dataset
        futures_data = get_github_data("ZF=F", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

        if futures_data.empty:
            st.error("❌ Failed to load futures data - please check your date range and try again")
            st.stop()
        else:
            st.success(f"✅ Loaded {len(futures_data)} days of futures data")

        # Apply Kalman filter to futures prices if enabled
        if use_kalman_price:
            st.info("Applying Kalman filter to futures prices...")
            for price_col in ['Open', 'High', 'Low', 'Close']:
                if price_col in futures_data.columns:
                    futures_data[f'{price_col}_filtered'] = kalman_filter_1d(
                        futures_data[price_col], kalman_process_var, kalman_measurement_var
                    )
                    # Replace original with filtered version
                    futures_data[price_col] = futures_data[f'{price_col}_filtered']

        # Load yield curve data for fundamental analysis
        st.info("Loading yield curve data...")
        _2yUS = get_fred_data("DGS2", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        _5yUS = get_github_data("^FVX", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        _5yUS_real = get_fred_data("DFII5", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

        # Check what data we successfully loaded
        data_status = {
            "2Y Treasury": not _2yUS.empty,
            "5Y Treasury": not _5yUS.empty,
            "5Y Real Yield": not _5yUS_real.empty
        }

        loaded_data = [k for k, v in data_status.items() if v]
        missing_data = [k for k, v in data_status.items() if not v]

        if loaded_data:
            st.success(f"✅ Loaded: {', '.join(loaded_data)}")
        if missing_data:
            st.warning(f"⚠️ Missing: {', '.join(missing_data)}")

        # Initialize final_data with futures data as baseline
        final_data = futures_data.copy()

        # Build indicators if we have sufficient yield curve data
        if not _2yUS.empty and not _5yUS_real.empty and not _5yUS.empty:
            try:
                st.info("Building fundamental indicators...")

                # Prepare data for indicator calculation
                _2yUS.columns = ["2y"]
                _5yUS.columns = ["5y", "High", "Low", "Open"]
                _5yUS_real.columns = ["5y_Real"]
                _5yUS_real = _5yUS_real.interpolate(method="polynomial", order=2)

                # Combine yield curve data
                backtest_data = _2yUS.join(_5yUS_real).join(_5yUS).dropna()

                if len(backtest_data) > 100:  # Need sufficient data for indicators
                    # Build indicators with optional Kalman filtering
                    indicators = build_indicators(backtest_data, use_kalman, kalman_process_var, kalman_measurement_var)

                    # Transform to z-scores
                    indicators_with_zscores = calculate_zscores(indicators, lookback_zscore=zscore_lookback)

                    # Convert to percentiles
                    for col in ["5y_Real", "carry_normalized", "momentum"]:
                        zscore_col = f"{col}_zscore"
                        if zscore_col in indicators_with_zscores.columns:
                            indicators_with_zscores[f"{col}_percentile"] = indicators_with_zscores[zscore_col].rolling(
                                percentile_lookback).apply(percentile_score)

                    # Create indicator subset and join with futures
                    indicator_subset = indicators_with_zscores[
                        ["5y_Real_percentile", "carry_normalized_percentile", "momentum_percentile"]]

                    # Join indicators with futures data
                    combined_data = indicator_subset.join(futures_data, how='inner')

                    if not combined_data.empty:
                        # Rename columns for clarity
                        combined_data.columns = ["Value_Percentile", "Carry_Percentile", "Momentum_Percentile", "Open",
                                                 "High", "Low", "Close"]

                        # Add z-scores for analysis
                        for col in ["5y_Real", "carry_normalized", "momentum"]:
                            zscore_col = f"{col}_zscore"
                            if zscore_col in indicators_with_zscores.columns:
                                combined_data[zscore_col] = indicators_with_zscores[zscore_col]

                        # Add filtered signals if Kalman was used
                        if use_kalman:
                            for col in ["5y_Real", "carry_normalized", "momentum"]:
                                filtered_col = f"{col}_filtered"
                                if filtered_col in indicators_with_zscores.columns:
                                    combined_data[filtered_col] = indicators_with_zscores[filtered_col]

                        # Calculate aggregate percentile
                        weights = [value_weight, carry_weight, momentum_weight]
                        combined_data['Agg_Percentile'] = calculate_agg_percentile(combined_data, weights)

                        # Use combined data as our final dataset
                        final_data = combined_data
                        st.success(f"✅ Built indicators with {len(final_data)} data points")

                    else:
                        st.warning("⚠️ No overlapping dates between indicators and futures - using futures only")
                        final_data['Agg_Percentile'] = 50.0  # Neutral value

                else:
                    st.warning(f"⚠️ Insufficient yield curve data ({len(backtest_data)} points) - using futures only")
                    final_data['Agg_Percentile'] = 50.0  # Neutral value

            except Exception as e:
                st.error(f"❌ Error building indicators: {str(e)}")
                final_data['Agg_Percentile'] = 50.0  # Neutral fallback

        else:
            st.warning("⚠️ Missing essential yield curve data - showing futures price analysis only")
            final_data['Agg_Percentile'] = 50.0  # Neutral value when no indicators available

        # Ensure we have the minimum required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in final_data.columns]

        if missing_cols:
            st.error(f"❌ Missing required price columns: {missing_cols}")
            st.stop()

        st.success(
            f"✅ Final dataset ready with {len(final_data)} data points from {final_data.index[0].strftime('%Y-%m-%d')} to {final_data.index[-1].strftime('%Y-%m-%d')}")

        # Add technical indicators if requested
        if show_keltner:
            final_data = calculate_keltner_channels(final_data)

        if show_rsi:
            final_data['RSI'] = calculate_rsi(final_data, rsi_window)

        # Perform regime analysis calculations
        st.info("Calculating rolling regime indicators...")
        progress_bar = st.progress(0)

        prices = final_data['Close']

        if freq_choice == "Weekly":
            # Weekly calculation provides more stable regime signals
            weekly_prices = prices.resample('W').last().dropna()

            adf_results = []
            hurst_results = []
            dates = []

            for i in range(rolling_window, len(weekly_prices)):
                if i % 10 == 0:
                    progress_bar.progress(i / len(weekly_prices))

                window_data = weekly_prices.iloc[i - rolling_window:i]

                # Test for mean reversion vs trending
                adf_p = adf_test_result(window_data)
                adf_results.append(adf_p)

                # Calculate persistence/anti-persistence
                hurst = hurst_exponent(window_data.values)
                hurst_results.append(hurst)

                dates.append(weekly_prices.index[i])

            # Map weekly results back to daily frequency
            weekly_results = pd.DataFrame({
                'ADF_pvalue': adf_results,
                'Hurst': hurst_results
            }, index=dates)

            final_data['ADF_pvalue'] = weekly_results['ADF_pvalue'].reindex(final_data.index, method='ffill')
            final_data['Hurst'] = weekly_results['Hurst'].reindex(final_data.index, method='ffill')

        else:  # Daily calculation
            adf_results = []
            hurst_results = []

            for i in range(rolling_window, len(prices)):
                if i % 50 == 0:
                    progress_bar.progress(i / len(prices))

                window_data = prices.iloc[i - rolling_window:i]

                # Analyze recent price behavior
                adf_p = adf_test_result(window_data)
                adf_results.append(adf_p)

                hurst = hurst_exponent(window_data.values)
                hurst_results.append(hurst)

            # Add results to dataset
            final_data['ADF_pvalue'] = [np.nan] * rolling_window + adf_results
            final_data['Hurst'] = [np.nan] * rolling_window + hurst_results

        progress_bar.progress(1.0)

        # Store results for display
        st.session_state.regime_data = final_data
        st.session_state.config = {
            'weights': weights,
            'show_keltner': show_keltner,
            'show_rsi': show_rsi,
            'use_kalman': use_kalman,
            'use_kalman_price': use_kalman_price,
            'freq_choice': freq_choice,
            'rolling_window': rolling_window
        }

# Display Analysis Results
if 'regime_data' in st.session_state:
    data = st.session_state.regime_data
    config = st.session_state.config

    st.markdown("---")

    # Main Price and Signal Analysis
    st.subheader("📈 Price Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### 5Y US Futures Price" + (" (Kalman Filtered)" if config['use_kalman_price'] else ""))

        fig_price = go.Figure()

        # Main price chart with optional filtering indication
        fig_price.add_trace(go.Scatter(
            x=data.index, y=data['Close'],
            mode='lines', name='5Y Futures' + (' (Filtered)' if config['use_kalman_price'] else ''),
            line=dict(color='black', width=1.5)
        ))

        # Add Keltner Channels for volatility analysis
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

            # Add trading zones for visual reference
            fig_agg.add_hrect(y0=80, y1=100, fillcolor="green", opacity=0.2, annotation_text="Buy Zone")
            fig_agg.add_hrect(y0=0, y1=20, fillcolor="red", opacity=0.2, annotation_text="Sell Zone")
            fig_agg.add_hrect(y0=20, y1=80, fillcolor="yellow", opacity=0.1, annotation_text="Neutral Zone")

            fig_agg.update_layout(
                height=400, xaxis_title="Date", yaxis_title="Percentile (%)",
                yaxis=dict(range=[0, 100]),
                title=f"Percentiles of Z-Scores (Z-Score: {zscore_lookback}d, Percentile: {percentile_lookback}d)"
            )
            st.plotly_chart(fig_agg, use_container_width=True)
        else:
            st.info("Aggregate percentile data not available")

    # Component Analysis
    st.subheader("📊 Z-Scores Analysis" + (" (Kalman Filtered)" if config['use_kalman'] else ""))

    zscore_cols = ['5y_Real_zscore', 'carry_normalized_zscore', 'momentum_zscore']
    available_zscores = [col for col in zscore_cols if col in data.columns and not data[col].isna().all()]

    if available_zscores:
        col1, col2, col3 = st.columns(3)

        with col1:
            if '5y_Real_zscore' in available_zscores:
                st.markdown("#### Value Z-Score")
                fig_val_z = go.Figure()
                fig_val_z.add_trace(go.Scatter(
                    x=data.index, y=data['5y_Real_zscore'],
                    mode='lines', name='Value Z-Score' + (' (Filtered)' if config['use_kalman'] else ''),
                    line=dict(color='blue', width=1.5)
                ))
                # Add reference lines for extreme values
                fig_val_z.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.5)
                fig_val_z.add_hline(y=2, line_dash="dash", line_color="red", opacity=0.5)
                fig_val_z.add_hline(y=-2, line_dash="dash", line_color="red", opacity=0.5)
                fig_val_z.update_layout(height=250, yaxis_title="Z-Score")
                st.plotly_chart(fig_val_z, use_container_width=True)

        with col2:
            if 'carry_normalized_zscore' in available_zscores:
                st.markdown("#### Carry Z-Score")
                fig_carry_z = go.Figure()
                fig_carry_z.add_trace(go.Scatter(
                    x=data.index, y=data['carry_normalized_zscore'],
                    mode='lines', name='Carry Z-Score' + (' (Filtered)' if config['use_kalman'] else ''),
                    line=dict(color='green', width=1.5)
                ))
                fig_carry_z.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.5)
                fig_carry_z.add_hline(y=2, line_dash="dash", line_color="red", opacity=0.5)
                fig_carry_z.add_hline(y=-2, line_dash="dash", line_color="red", opacity=0.5)
                fig_carry_z.update_layout(height=250, yaxis_title="Z-Score")
                st.plotly_chart(fig_carry_z, use_container_width=True)

        with col3:
            if 'momentum_zscore' in available_zscores:
                st.markdown("#### Momentum Z-Score")
                fig_mom_z = go.Figure()
                fig_mom_z.add_trace(go.Scatter(
                    x=data.index, y=data['momentum_zscore'],
                    mode='lines', name='Momentum Z-Score' + (' (Filtered)' if config['use_kalman'] else ''),
                    line=dict(color='orange', width=1.5)
                ))
                fig_mom_z.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.5)
                fig_mom_z.add_hline(y=2, line_dash="dash", line_color="red", opacity=0.5)
                fig_mom_z.add_hline(y=-2, line_dash="dash", line_color="red", opacity=0.5)
                fig_mom_z.update_layout(height=250, yaxis_title="Z-Score")
                st.plotly_chart(fig_mom_z, use_container_width=True)

    else:
        st.info("Z-score data not available")

    # RSI Analysis (if enabled)
    if config['show_rsi'] and 'RSI' in data.columns:
        st.subheader("🔄 RSI Analysis")

        fig_rsi = go.Figure()

        fig_rsi.add_trace(go.Scatter(
            x=data.index, y=data['RSI'],
            mode='lines', name='RSI',
            line=dict(color='orange', width=2)
        ))

        # Standard RSI levels for reference
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral (50)")

        fig_rsi.update_layout(
            height=300, xaxis_title="Date", yaxis_title="RSI",
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

    # Regime Analysis Core Charts
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

            # Statistical significance thresholds
            fig_adf.add_hline(y=0.05, line_dash="dash", line_color="black",
                              annotation_text="5% significance")
            fig_adf.add_hline(y=0.01, line_dash="dash", line_color="darkred",
                              annotation_text="1% significance")

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

            # Regime classification thresholds
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

    # Performance Summary Statistics
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

    # Data Export Functionality
    st.markdown("### 📁 Export Data")
    if st.button("📥 Download Analysis Results"):
        export_data = data[['Close', 'Agg_Percentile', 'ADF_pvalue', 'Hurst']].copy()

        # Include z-scores for detailed analysis
        zscore_cols = ['5y_Real_zscore', 'carry_normalized_zscore', 'momentum_zscore']
        for col in zscore_cols:
            if col in data.columns:
                export_data[col] = data[col]

        # Include component percentiles
        percentile_cols = ['Value_Percentile', 'Carry_Percentile', 'Momentum_Percentile']
        for col in percentile_cols:
            if col in data.columns:
                export_data[col] = data[col]

        # Include filtered data if Kalman filtering was applied
        if st.session_state.config.get('use_kalman_price', False):
            for col in ['Open_filtered', 'High_filtered', 'Low_filtered', 'Close_filtered']:
                if col in data.columns:
                    export_data[col] = data[col]

        if st.session_state.config.get('use_kalman', False):
            for col in ['5y_Real_filtered', 'carry_normalized_filtered', 'momentum_filtered']:
                if col in data.columns:
                    export_data[col] = data[col]

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

# Educational Information Panel
st.markdown("---")
st.markdown("""
### 📋 Interpretation Guide:

**Calculation Process:**
1. **Raw Indicators**: Value (5Y Real Yield), Carry (5Y-2Y spread), Momentum (5d vs 20d MA)
2. **Kalman Filtering**: Optional noise reduction using adaptive filtering (if enabled)
   - **Indicators**: Smooths Value/Carry/Momentum signals
   - **Prices**: Smooths OHLC futures price data for cleaner regime analysis
3. **Z-Scores**: Standardize each indicator using rolling mean/std (default: 66 days)
4. **Percentiles**: Calculate percentile rank of z-scores using shorter rolling window (default: 63 days)
5. **Aggregate**: Weighted combination of percentiles → Buy/Sell signals

**Kalman Filter:**
- **Purpose**: Reduces noise in raw signals while preserving underlying trends
- **Process Variance**: Lower values = smoother, less responsive signals
- **Measurement Variance**: Higher values = more filtering of noisy data
- **Indicators**: Cleaner signals may lead to better regime detection and fewer false signals
- **Prices**: Smoother price data improves RSI, Keltner channels, and regime calculations

**ADF Test (Augmented Dickey-Fuller):**
- **p-value < 0.05**: Series is stationary → **Mean Reverting**
- **p-value > 0.05**: Series is non-stationary → **Trending** or Random Walk

**Hurst Exponent:**
- **H < 0.5**: **Mean Reverting** behavior (anti-persistent)
- **H = 0.5**: **Random Walk** (no memory)
- **H > 0.5**: **Trending** behavior (persistent)

**Z-Score Interpretation:**
- **|Z| > 2**: Extreme values (outside 95% of recent distribution)
- **Z > 0**: Above recent average
- **Z < 0**: Below recent average

**Trading Implications:**
- **Mean Reverting periods**: Use mean reversion strategies, fade extremes
- **Trending periods**: Use momentum strategies, follow trends  
- **Transition periods**: Exercise caution, consider regime change
""")