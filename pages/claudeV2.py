import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import urllib
from fredapi import Fred
import warnings
import json

warnings.filterwarnings('ignore')

# Initialize FRED API
fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')
st.set_page_config(layout="wide", page_title="Professional Treasury Trading", page_icon="💹")

# Professional Trading App Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px solid #dee2e6;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Assets dictionary
assets_dict = {
    "2y US": "DGS2",
    "5y US": "^FVX",
    "5y US Real": "DFII5",
    "5y US Future": "ZF=F"
}


@st.cache_data
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
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


@st.cache_data
def fred_import(ticker, start_date):
    """Import data from FRED"""
    try:
        fred_data = pd.DataFrame(fred.get_series(ticker, observation_start=start_date, freq="daily"))
        return fred_data
    except Exception as e:
        st.error(f"Error loading FRED data: {e}")
        return pd.DataFrame()


def calculate_hurst(ts):
    """Calculate Hurst Exponent using improved R/S method"""
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
    return m[0]


def adf_test(ts):
    """Perform Augmented Dickey-Fuller test for mean reversion"""
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(ts.dropna(), autolag='AIC', maxlag=1)
        return result[0], result[1]
    except:
        return np.nan, np.nan


def calculate_rsi(series, period=14):
    """RSI calculation"""
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_keltner_channel(high, low, close, period=20, multiplier=2.0):
    """Calculate Keltner Channels"""
    ema = close.ewm(span=period).mean()
    atr = ((high - low).abs()).rolling(window=period).mean()

    upper_channel = ema + (multiplier * atr)
    lower_channel = ema - (multiplier * atr)

    return upper_channel, ema, lower_channel


def classify_regime(indicator_value, regime_method, hurst_trend_thresh=0.53, hurst_range_thresh=0.47,
                    adf_trend_thresh=-2.567, adf_range_thresh=-2.862):
    """Classify regime based on selected method"""
    if pd.isna(indicator_value):
        return "UNKNOWN", 0.0

    if regime_method == "Hurst Exponent":
        confidence = min(abs(indicator_value - 0.5) * 4, 1.0)
        if indicator_value > hurst_trend_thresh:
            return "TRENDING", confidence
        elif indicator_value < hurst_range_thresh:
            return "MEAN_REVERTING", confidence
        else:
            return "UNKNOWN", confidence

    elif regime_method == "ADF Test":
        confidence = 0.5  # Base confidence for ADF
        if indicator_value < adf_range_thresh:  # Strong mean reversion
            confidence = 1.0
            return "MEAN_REVERTING", confidence
        elif indicator_value < adf_trend_thresh:  # Moderate mean reversion
            confidence = 0.7
            return "MEAN_REVERTING", confidence
        else:
            confidence = 0.3
            return "TRENDING", confidence

    else:  # Manual override
        return regime_method, 1.0


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
    return (nb_values_below / len(window)) * 100


def zscore(data, lookback):
    """Calculate Z-score"""
    return (data - data.rolling(lookback).mean()) / data.rolling(lookback).std()


def calculate_agg_percentile(data, weights):
    """Calculate aggregate percentile with given weights"""
    value_w, carry_w, momentum_w = weights
    return (data['Value_Percentile'] * value_w +
            data['Carry_Percentile'] * carry_w +
            data['Momentum_Percentile'] * momentum_w) / 100


def check_rsi_confirmation(rsi, signal_type, rsi_overbought, rsi_oversold):
    """Check RSI confirmation for signals"""
    if signal_type == "BUY" and rsi < rsi_oversold:
        return True
    elif signal_type == "SELL" and rsi > rsi_overbought:
        return True
    return False


def check_keltner_confirmation(close, upper_channel, lower_channel, signal_type, regime):
    """Check Keltner Channel confirmation for signals"""
    if signal_type == "SELL" and close > upper_channel and regime == "MEAN_REVERTING":
        return True
    elif signal_type == "BUY" and close < lower_channel and regime == "MEAN_REVERTING":
        return True
    return False


def backtest_strategy_with_confirmations(data, weights, buy_zone, sell_zone,
                                         use_rsi_confirm, use_keltner_confirm,
                                         use_regime_confirm, regime_method,
                                         rsi_period, rsi_overbought, rsi_oversold,
                                         keltner_period, keltner_multiplier,
                                         hurst_window,
                                         transaction_cost_bps, capital_allocation_pct,
                                         initial_cash=100000):
    """Enhanced backtest with RSI and Keltner confirmations"""

    if len(data) < 100:
        return {'total_pnl': -999999, 'max_drawdown': 999, 'num_trades': 0, 'win_rate': 0,
                'trades': [], 'total_return_pct': -999}

    data_copy = data.copy()
    data_copy['Agg_Percentile'] = calculate_agg_percentile(data_copy, weights)

    # Calculate additional indicators if needed
    if use_rsi_confirm:
        data_copy['RSI'] = calculate_rsi(data_copy['Close'], rsi_period)

    if use_keltner_confirm:
        upper_ch, middle_ch, lower_ch = calculate_keltner_channel(
            data_copy['High'], data_copy['Low'], data_copy['Close'],
            keltner_period, keltner_multiplier)
        data_copy['Keltner_Upper'] = upper_ch
        data_copy['Keltner_Lower'] = lower_ch

    # Calculate regime indicators
    data_copy['Hurst'] = data_copy['Close'].rolling(window=hurst_window).apply(calculate_hurst, raw=False)

    if regime_method == "ADF Test":
        def rolling_adf(series, window):
            adf_stats = []
            for i in range(len(series)):
                if i < window:
                    adf_stats.append(np.nan)
                else:
                    window_data = series.iloc[i - window:i + 1]
                    stat, _ = adf_test(window_data)
                    adf_stats.append(stat)
            return pd.Series(adf_stats, index=series.index)

        data_copy['ADF_Stat'] = rolling_adf(data_copy['Close'], hurst_window)
        regime_indicator = data_copy['ADF_Stat']
    else:
        regime_indicator = data_copy['Hurst']

    # Regime classification
    regime_results = []
    for _, row in data_copy.iterrows():
        if regime_method == "ADF Test":
            regime, confidence = classify_regime(row['ADF_Stat'], regime_method)
        else:
            regime, confidence = classify_regime(row['Hurst'], regime_method)
        regime_results.append((regime, confidence))

    data_copy['Regime'] = [r[0] for r in regime_results]
    data_copy['Regime_Confidence'] = [r[1] for r in regime_results]

    data_copy['Signal'] = 0
    data_copy['Executed'] = False

    # Generate base signals
    buy_min, buy_max = buy_zone
    sell_min, sell_max = sell_zone

    for idx, row in data_copy.iterrows():
        agg_perc = row['Agg_Percentile']
        base_signal = None

        if buy_min <= agg_perc <= buy_max:
            base_signal = "BUY"
        elif sell_min <= agg_perc <= sell_max:
            base_signal = "SELL"

        if base_signal:
            confirmed = True

            # Apply regime confirmation if enabled
            if use_regime_confirm and row['Regime'] == "UNKNOWN":
                confirmed = False

            # Apply other confirmations if enabled
            if confirmed and (use_rsi_confirm or use_keltner_confirm):
                confirmations = []

                if use_rsi_confirm:
                    rsi_confirmed = check_rsi_confirmation(
                        row['RSI'], base_signal, rsi_overbought, rsi_oversold)
                    confirmations.append(rsi_confirmed)

                if use_keltner_confirm:
                    keltner_confirmed = check_keltner_confirmation(
                        row['Close'], row['Keltner_Upper'], row['Keltner_Lower'],
                        base_signal, row['Regime'])
                    confirmations.append(keltner_confirmed)

                # If both confirmations are enabled, both must be true
                # If only one is enabled, that one must be true
                confirmed = all(confirmations) if confirmations else True

            if confirmed:
                data_copy.loc[idx, 'Signal'] = 1 if base_signal == "BUY" else -1

    # Trading simulation
    cash = initial_cash
    position = 0
    position_size = 0
    entry_price = 0
    total_pnl = 0
    trades = []
    equity_curve = [initial_cash]

    for idx, row in data_copy.iterrows():
        current_price = row['Close']
        signal = row['Signal']

        # Trading logic
        if signal != 0 and signal != position:
            # Close existing position
            if position != 0:
                pnl = (current_price - entry_price) * position_size
                total_pnl += pnl
                cash += pnl
                cash -= abs(pnl) * (transaction_cost_bps / 10000)

                return_pct = ((current_price - entry_price) / entry_price) * position * 100
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': idx,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position': 'LONG' if position > 0 else 'SHORT',
                    'pnl': pnl,
                    'return_pct': return_pct,
                    'exit_reason': 'SIGNAL'
                })

                data_copy.loc[idx, 'Executed'] = True

            # Open new position
            position = signal
            allocated_capital = cash * (capital_allocation_pct / 100)
            position_size = allocated_capital / current_price * position
            entry_price = current_price
            entry_date = idx
            cash -= abs(allocated_capital) * (transaction_cost_bps / 10000)

            data_copy.loc[idx, 'Executed'] = True

        # Update equity curve
        if position != 0:
            unrealized_pnl = (current_price - entry_price) * position_size
            current_equity = cash + unrealized_pnl
        else:
            current_equity = cash
        equity_curve.append(current_equity)

    # Close final position
    if position != 0:
        final_price = data_copy.iloc[-1]['Close']
        pnl = (final_price - entry_price) * position_size
        total_pnl += pnl
        return_pct = ((final_price - entry_price) / entry_price) * position * 100

        trades.append({
            'entry_date': entry_date,
            'exit_date': data_copy.index[-1],
            'entry_price': entry_price,
            'exit_price': final_price,
            'position': 'LONG' if position > 0 else 'SHORT',
            'pnl': pnl,
            'return_pct': return_pct,
            'exit_reason': 'FINAL'
        })

    # Calculate performance metrics
    equity_curve = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max * 100
    max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0

    return {
        'total_pnl': total_pnl,
        'total_return_pct': (total_pnl / initial_cash) * 100,
        'max_drawdown': max_drawdown,
        'num_trades': len(trades),
        'win_rate': len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100 if trades else 0,
        'equity_curve': equity_curve,
        'final_value': initial_cash + total_pnl,
        'trades': trades,
        'data_with_signals': data_copy
    }


# Streamlit App Header
st.markdown(
    '<div class="main-header"><h1>💹 Professional Treasury Trading System with Grid Search</h1><p>Enhanced with RSI & Keltner Channel Confirmations + Regime Analysis</p></div>',
    unsafe_allow_html=True)

# Date Configuration - ALWAYS VISIBLE
st.markdown("### 📅 Trading & Training Period Configuration")
col1, col2, col3 = st.columns(3)

with col1:
    start_date_input = st.date_input("Trading Start Date", value=datetime(2023, 1, 1))
with col2:
    end_date_input = st.date_input("Trading End Date", value=datetime.now().date())
with col3:
    train_years = st.selectbox("Training Years (before start)", [1, 2, 3, 5, 7, 10], index=2)

# Calculate and display training period
training_start_date = start_date_input - timedelta(days=train_years * 365)
st.info(
    f"Training Period: {training_start_date.strftime('%Y-%m-%d')} to {start_date_input.strftime('%Y-%m-%d')} ({train_years} years)")
st.info(f"Trading Period: {start_date_input.strftime('%Y-%m-%d')} to {end_date_input.strftime('%Y-%m-%d')}")

# Configuration Section
with st.expander("⚙️ Additional Trading Configuration", expanded=True):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**💰 Capital Management**")
        initial_cash = st.number_input("Initial Capital ($)", value=100000, min_value=10000, step=10000)
        capital_allocation_pct = st.slider("Capital per Trade (%)", 10, 100, 25)
        transaction_cost_bps = st.number_input("Transaction Cost (bps)", 0, 50, 2)

    with col2:
        st.markdown("**🎯 Signal Zones**")
        buy_zone_min = st.number_input("Buy Zone Min (%)", value=90, min_value=70, max_value=95)
        buy_zone_max = st.number_input("Buy Zone Max (%)", value=100, min_value=95, max_value=100)
        sell_zone_min = st.number_input("Sell Zone Min (%)", value=0, min_value=0, max_value=10)
        sell_zone_max = st.number_input("Sell Zone Max (%)", value=10, min_value=5, max_value=20)

        st.markdown("**🔍 Z-Score Configuration**")
        zscore_lookback = st.number_input("Z-Score Lookback (days)", value=63, min_value=20, max_value=252)

    with col3:
        st.markdown("**📊 RSI Configuration**")
        use_rsi_confirm = st.checkbox("Use RSI Confirmation", value=False)
        rsi_period = st.number_input("RSI Period", value=14, min_value=5, max_value=50)
        rsi_overbought = st.number_input("RSI Overbought", value=60, min_value=50, max_value=80)
        rsi_oversold = st.number_input("RSI Oversold", value=40, min_value=20, max_value=50)

        st.markdown("**🔄 Regime Configuration**")
        regime_method = st.selectbox("Regime Indicator",
                                     ["Hurst Exponent", "ADF Test", "TRENDING", "MEAN_REVERTING", "UNKNOWN"])
        use_regime_confirm = st.checkbox("Filter Unknown Regimes", value=True)
        hurst_window = st.number_input("Regime Indicator Window", value=50, min_value=20, max_value=100)

    with col4:
        st.markdown("**📈 Keltner Channel Config**")
        use_keltner_confirm = st.checkbox("Use Keltner Confirmation", value=False)
        keltner_period = st.number_input("Keltner Period", value=20, min_value=10, max_value=50)
        keltner_multiplier = st.number_input("Keltner Multiplier", value=2.0, min_value=1.0, max_value=3.0, step=0.1)

        st.caption("Keltner confirms: Mean reversion signals near bands")

# Grid Search Configuration
with st.expander("🔍 Grid Search Optimization", expanded=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        enable_grid_search = st.checkbox("Enable Grid Search Optimization", value=False)
        weight_step = st.selectbox("Weight Step Size (%)", [5, 10, 20], index=1) if enable_grid_search else 10

    with col2:
        optimization_goal = st.selectbox("Optimization Goal",
                                         ["Max Return", "Max Return/DD Ratio"]) if enable_grid_search else "Max Return"

    with col3:
        if enable_grid_search:
            st.info(f"Will test {len(range(weight_step, 101, weight_step)) ** 2} combinations")

# Manual Weight Configuration (if not using grid search)
if not enable_grid_search:
    st.markdown("**⚖️ Manual Weight Configuration**")
    col1, col2, col3 = st.columns(3)
    with col1:
        value_weight = st.slider("Value Weight (%)", 0, 100, 70)
    with col2:
        carry_weight = st.slider("Carry Weight (%)", 0, 100, 20)
    with col3:
        momentum_weight = st.slider("Momentum Weight (%)", 0, 100, 10)

    if abs(value_weight + carry_weight + momentum_weight - 100) > 0.01:
        st.error("⚠️ Weights must sum to 100%")
        st.stop()

# Main Execution Button
if st.button("🚀 Run Analysis", type="primary"):

    # Calculate training start date
    training_start_date = start_date_input - timedelta(days=train_years * 365)

    with st.spinner("Loading data..."):
        # Load data from training start
        training_start_str = training_start_date.strftime("%Y-%m-%d")

        _2yUS = fred_import(assets_dict["2y US"], training_start_str)
        _2yUS.columns = ["2y"]

        _5yUS = get_data(assets_dict["5y US"], training_start_str)
        _5yUS.columns = ["5y", "High", "Low", "Open"]

        _5yUS_real = fred_import(assets_dict["5y US Real"], training_start_str)
        _5yUS_real.columns = ["5y_Real"]
        _5yUS_real = _5yUS_real.interpolate(method="polynomial", order=2)

        _5yUS_fut = get_data(assets_dict["5y US Future"], training_start_str)

        if any(df.empty for df in [_2yUS, _5yUS, _5yUS_real, _5yUS_fut]):
            st.error("Failed to load required data")
            st.stop()

    # Build indicators
    backtest_data = _2yUS.join(_5yUS_real).join(_5yUS)
    backtest_data.dropna(inplace=True)
    indicators = build_indicators(backtest_data)

    # Calculate percentiles
    for cols in ["5y_Real", "carry_normalized", "momentum"]:
        indicators[f"{cols}_z"] = zscore(indicators[cols], zscore_lookback)
        indicators[f"{cols}_percentile"] = indicators[f"{cols}_z"].rolling(zscore_lookback).apply(
            lambda x: percentile_score(x))

    # Final dataset
    indicator_full = indicators[
        ["5y", "5y_Real_percentile", "carry_normalized_percentile", "momentum_percentile"]].join(_5yUS_fut)
    indicator_full.columns = ["5y_yield", "Value_Percentile", "Carry_Percentile", "Momentum_Percentile", "Open", "High",
                              "Low", "Close"]
    indicator_full.dropna(inplace=True)

    st.success(
        f"Training: {training_start_date.strftime('%Y-%m-%d')} to {start_date_input.strftime('%Y-%m-%d')} ({len(indicator_full[indicator_full.index < pd.to_datetime(start_date_input)])} days)")
    st.success(
        f"Trading: {start_date_input.strftime('%Y-%m-%d')} to {end_date_input.strftime('%Y-%m-%d')} ({len(indicator_full[(indicator_full.index >= pd.to_datetime(start_date_input)) & (indicator_full.index <= pd.to_datetime(end_date_input))])} days)")

    # Grid Search or Single Run
    if enable_grid_search:
        with st.spinner(f"Running grid search optimization..."):
            # Generate weight combinations
            weight_combinations = []
            for value_w in range(weight_step, 101, weight_step):
                for carry_w in range(weight_step, 101 - value_w + weight_step, weight_step):
                    momentum_w = 100 - value_w - carry_w
                    if momentum_w >= weight_step:
                        weight_combinations.append([value_w, carry_w, momentum_w])

            # Split data for optimization
            train_data = indicator_full[indicator_full.index < pd.to_datetime(start_date_input)]

            # Optimize on train data
            results = []
            progress_bar = st.progress(0)

            for i, weights in enumerate(weight_combinations):
                if i % 20 == 0:
                    progress_bar.progress(i / len(weight_combinations))

                train_results = backtest_strategy_with_confirmations(
                    train_data, weights, (buy_zone_min, buy_zone_max), (sell_zone_min, sell_zone_max),
                    use_rsi_confirm, use_keltner_confirm, use_regime_confirm, regime_method,
                    rsi_period, rsi_overbought, rsi_oversold, keltner_period, keltner_multiplier,
                    hurst_window, transaction_cost_bps, capital_allocation_pct, initial_cash
                )

                train_results['weights'] = weights
                results.append(train_results)

            progress_bar.progress(1.0)

            # Find optimal weights
            if optimization_goal == "Max Return":
                results.sort(key=lambda x: x['total_return_pct'], reverse=True)
            else:
                for r in results:
                    r['return_dd_ratio'] = r['total_return_pct'] / max(r['max_drawdown'], 0.01)
                results.sort(key=lambda x: x['return_dd_ratio'], reverse=True)

            optimal_weights = results[0]['weights']

            st.success(
                f"🏆 Optimal Weights Found: Value:{optimal_weights[0]}% | Carry:{optimal_weights[1]}% | Momentum:{optimal_weights[2]}%")

            # Display top results
            st.subheader("🔍 Top 10 Optimization Results")
            top_results = pd.DataFrame({
                'Rank': range(1, 11),
                'Value%': [r['weights'][0] for r in results[:10]],
                'Carry%': [r['weights'][1] for r in results[:10]],
                'Momentum%': [r['weights'][2] for r in results[:10]],
                'Train Return%': [f"{r['total_return_pct']:.1f}" for r in results[:10]],
                'Train MaxDD%': [f"{r['max_drawdown']:.1f}" for r in results[:10]],
                'Train Trades': [r['num_trades'] for r in results[:10]]
            })
            st.dataframe(top_results, hide_index=True, use_container_width=True)

    else:
        # Single run with manual weights
        optimal_weights = [value_weight, carry_weight, momentum_weight]

    # Test on trading period
    trading_data = indicator_full[(indicator_full.index >= pd.to_datetime(start_date_input)) &
                                  (indicator_full.index <= pd.to_datetime(end_date_input))]

    full_results = backtest_strategy_with_confirmations(
        trading_data, optimal_weights, (buy_zone_min, buy_zone_max), (sell_zone_min, sell_zone_max),
        use_rsi_confirm, use_keltner_confirm, use_regime_confirm, regime_method,
        rsi_period, rsi_overbought, rsi_oversold, keltner_period, keltner_multiplier,
        hurst_window, transaction_cost_bps, capital_allocation_pct, initial_cash
    )

    # Display Results
    st.markdown("---")
    st.subheader(
        f"📊 Strategy Performance: Value:{optimal_weights[0]}% | Carry:{optimal_weights[1]}% | Momentum:{optimal_weights[2]}%")

    # Performance Metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Return", f"{full_results['total_return_pct']:.1f}%")
    with col2:
        st.metric("Total P&L", f"${full_results['total_pnl']:,.0f}")
    with col3:
        st.metric("Max Drawdown", f"{full_results['max_drawdown']:.1f}%")
    with col4:
        st.metric("Number of Trades", full_results['num_trades'])
    with col5:
        st.metric("Win Rate", f"{full_results['win_rate']:.1f}%")

    # Data with signals
    data_with_signals = full_results['data_with_signals']

    # Charts Section
    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("**📊 Chart Display Options**")
        show_hurst = st.checkbox("Show Hurst Exponent", value=True)
        if regime_method == "ADF Test":
            show_adf = st.checkbox("Show ADF Statistic", value=True)

        st.markdown("**Component Indicators:**")
        show_value = st.checkbox("Show Value Percentile", value=False)
        show_carry = st.checkbox("Show Carry Percentile", value=False)
        show_momentum = st.checkbox("Show Momentum Percentile", value=False)

    with col1:
        st.subheader("📈 Price with Trading Signals")
        fig1 = go.Figure()

        # Price line
        fig1.add_trace(go.Scatter(x=data_with_signals.index, y=data_with_signals["Close"],
                                  mode="lines", name="5Y Futures Price", line=dict(color='black', width=2)))

        # Buy signals
        executed_buys = data_with_signals[(data_with_signals['Signal'] == 1) & (data_with_signals['Executed'] == True)]
        if len(executed_buys) > 0:
            fig1.add_trace(go.Scatter(x=executed_buys.index, y=executed_buys["Close"],
                                      mode="markers", name="Buy Executed",
                                      marker=dict(symbol="triangle-up", color="green", size=12)))

        # Sell signals
        executed_sells = data_with_signals[
            (data_with_signals['Signal'] == -1) & (data_with_signals['Executed'] == True)]
        if len(executed_sells) > 0:
            fig1.add_trace(go.Scatter(x=executed_sells.index, y=executed_sells["Close"],
                                      mode="markers", name="Sell Executed",
                                      marker=dict(symbol="triangle-down", color="red", size=12)))

        # Add Keltner Channels if enabled
        if use_keltner_confirm:
            fig1.add_trace(go.Scatter(x=data_with_signals.index, y=data_with_signals["Keltner_Upper"],
                                      mode="lines", name="Keltner Upper", line=dict(color='orange', dash='dash')))
            fig1.add_trace(go.Scatter(x=data_with_signals.index, y=data_with_signals["Keltner_Lower"],
                                      mode="lines", name="Keltner Lower", line=dict(color='orange', dash='dash')))

        fig1.update_layout(height=400, title="Trading Signals on 5Y Treasury Futures")
        st.plotly_chart(fig1, use_container_width=True)

    # Aggregate Percentile Chart with Components
    st.subheader("📊 Aggregate Percentile & Components with RSI")
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

    # Aggregate percentile
    fig2.add_trace(go.Scatter(x=data_with_signals.index, y=data_with_signals["Agg_Percentile"],
                              mode="lines", name="Agg Percentile", line=dict(color='purple', width=3)),
                   secondary_y=False)

    # Component percentiles if enabled
    if show_value:
        fig2.add_trace(go.Scatter(x=data_with_signals.index, y=data_with_signals["Value_Percentile"],
                                  mode="lines", name="Value %ile", line=dict(color='red', width=1, dash='dot')),
                       secondary_y=False)

    if show_carry:
        fig2.add_trace(go.Scatter(x=data_with_signals.index, y=data_with_signals["Carry_Percentile"],
                                  mode="lines", name="Carry %ile", line=dict(color='blue', width=1, dash='dash')),
                       secondary_y=False)

    if show_momentum:
        fig2.add_trace(go.Scatter(x=data_with_signals.index, y=data_with_signals["Momentum_Percentile"],
                                  mode="lines", name="Momentum %ile",
                                  line=dict(color='green', width=1, dash='dashdot')), secondary_y=False)

    # RSI if enabled
    if use_rsi_confirm:
        fig2.add_trace(go.Scatter(x=data_with_signals.index, y=data_with_signals["RSI"],
                                  mode="lines", name="RSI", line=dict(color='blue', width=1)), secondary_y=True)
        fig2.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", secondary_y=True)
        fig2.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", secondary_y=True)

    # Trading zones
    fig2.add_hrect(y0=buy_zone_min, y1=buy_zone_max, fillcolor="rgba(0, 255, 0, 0.1)", layer="below", secondary_y=False)
    fig2.add_hrect(y0=sell_zone_min, y1=sell_zone_max, fillcolor="rgba(255, 0, 0, 0.1)", layer="below",
                   secondary_y=False)

    fig2.update_yaxes(title_text="Percentile (%)", secondary_y=False, range=[0, 100])
    if use_rsi_confirm:
        fig2.update_yaxes(title_text="RSI", secondary_y=True, range=[0, 100])

    fig2.update_layout(height=450, title="Signal Analysis Dashboard")
    st.plotly_chart(fig2, use_container_width=True)

    # Regime Indicators Chart
    if show_hurst or (regime_method == "ADF Test" and show_adf):
        st.subheader(f"🔄 Regime Analysis - {regime_method}")
        fig3 = go.Figure()

        if show_hurst:
            fig3.add_trace(go.Scatter(x=data_with_signals.index, y=data_with_signals["Hurst"],
                                      mode="lines", name="Hurst Exponent", line=dict(color='purple', width=2)))
            fig3.add_hline(y=0.5, line_dash="dot", line_color="gray", annotation_text="Random Walk (0.5)")
            fig3.add_hline(y=0.53, line_dash="dash", line_color="green", annotation_text="Trending (>0.53)")
            fig3.add_hline(y=0.47, line_dash="dash", line_color="red", annotation_text="Mean Reverting (<0.47)")

        if regime_method == "ADF Test" and show_adf:
            fig3.add_trace(go.Scatter(x=data_with_signals.index, y=data_with_signals["ADF_Stat"],
                                      mode="lines", name="ADF Statistic", line=dict(color='orange', width=2)))
            fig3.add_hline(y=-2.567, line_dash="dash", line_color="orange", annotation_text="Moderate MR (-2.567)")
            fig3.add_hline(y=-2.862, line_dash="dash", line_color="red", annotation_text="Strong MR (-2.862)")

        fig3.update_layout(height=300, title=f"{regime_method} Analysis")
        st.plotly_chart(fig3, use_container_width=True)

    # Regime Analysis Summary
    st.subheader("🔄 Market Regime Analysis Summary")
    regime_summary = data_with_signals.groupby('Regime').agg({
        'Regime': 'count',
        'Signal': lambda x: (x != 0).sum(),
        'Executed': 'sum'
    }).round(2)
    regime_summary.columns = ['Days', 'Signals Generated', 'Signals Executed']
    st.dataframe(regime_summary, use_container_width=True)

    # Trade Log
    if full_results['trades']:
        st.subheader("📋 Trade History")
        trades_df = pd.DataFrame(full_results['trades'])
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date']).dt.strftime('%Y-%m-%d')
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date']).dt.strftime('%Y-%m-%d')
        trades_df['pnl'] = trades_df['pnl'].round(0).astype(int)
        trades_df['return_pct'] = trades_df['return_pct'].round(2)


        # Style profitable trades
        def color_pnl(val):
            return 'color: green' if val > 0 else 'color: red'


        styled_trades = trades_df.style.applymap(color_pnl, subset=['pnl', 'return_pct'])
        st.dataframe(styled_trades, hide_index=True, use_container_width=True)

    # Current Status
    st.subheader("📈 Current Market Status")
    if not data_with_signals.empty:
        latest = data_with_signals.iloc[-1]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Agg %ile", f"{latest['Agg_Percentile']:.1f}%")
            st.metric("Value %ile", f"{latest['Value_Percentile']:.1f}%")

        with col2:
            st.metric("Carry %ile", f"{latest['Carry_Percentile']:.1f}%")
            st.metric("Momentum %ile", f"{latest['Momentum_Percentile']:.1f}%")

        with col3:
            if use_rsi_confirm:
                st.metric("Current RSI", f"{latest['RSI']:.1f}")
                rsi_signal = "Overbought" if latest['RSI'] > rsi_overbought else "Oversold" if latest[
                                                                                                   'RSI'] < rsi_oversold else "Neutral"
                st.metric("RSI Signal", rsi_signal)
            else:
                st.metric("Current Regime", latest['Regime'])
                st.metric("Hurst Exponent", f"{latest['Hurst']:.3f}")

        with col4:
            confidence = latest['Regime_Confidence']
            st.metric("Regime Confidence", f"{confidence:.2f}")
            if regime_method == "ADF Test":
                st.metric("ADF Stat", f"{latest['ADF_Stat']:.2f}")

    # Export Configuration
    st.subheader("📁 Export Results")
    col1, col2 = st.columns(2)

    with col1:
        config = {
            'optimal_weights': optimal_weights,
            'performance': {
                'total_return_pct': full_results['total_return_pct'],
                'max_drawdown': full_results['max_drawdown'],
                'num_trades': full_results['num_trades'],
                'win_rate': full_results['win_rate']
            },
            'periods': {
                'training_start': training_start_date.strftime('%Y-%m-%d'),
                'trading_start': start_date_input.strftime('%Y-%m-%d'),
                'trading_end': end_date_input.strftime('%Y-%m-%d')
            },
            'settings': {
                'buy_zone': [buy_zone_min, buy_zone_max],
                'sell_zone': [sell_zone_min, sell_zone_max],
                'use_rsi_confirm': use_rsi_confirm,
                'use_keltner_confirm': use_keltner_confirm,
                'use_regime_confirm': use_regime_confirm,
                'regime_method': regime_method,
                'rsi_settings': [rsi_period, rsi_overbought, rsi_oversold],
                'keltner_settings': [keltner_period, keltner_multiplier]
            }
        }

        st.download_button("📥 Download Configuration",
                           json.dumps(config, indent=2, default=str),
                           f"trading_config_{datetime.now().strftime('%Y%m%d')}.json")

    with col2:
        if full_results['trades']:
            trades_csv = pd.DataFrame(full_results['trades']).to_csv(index=False)
            st.download_button("📥 Download Trade History", trades_csv,
                               f"trade_history_{datetime.now().strftime('%Y%m%d')}.csv")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <strong>Professional Treasury Trading System v3.0</strong><br>
    <em>Enhanced with Grid Search, RSI & Keltner Confirmations, Flexible Regime Analysis</em><br>
    <small>Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</small>
</div>
""", unsafe_allow_html=True)