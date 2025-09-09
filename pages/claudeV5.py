import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
import warnings
import json

warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Regime-Aware Backtester", page_icon="🎯")

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


@st.cache_data
def get_fred_data(ticker, start_date):
    """Import data from FRED"""
    try:
        return pd.DataFrame(fred.get_series(ticker, observation_start=start_date, freq="daily"))
    except Exception as e:
        st.error(f"Error loading FRED {ticker}: {e}")
        return pd.DataFrame()


def zscore(data, lookback):
    """Calculate Z-score"""
    return (data - data.rolling(lookback).mean()) / data.rolling(lookback).std()


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


def fast_adx(high, low, close, period=14):
    """Fast ADX calculation"""
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

    # 2. Price momentum
    returns = close.pct_change()
    momentum = returns.rolling(short_period).apply(lambda x: abs(x.mean()) * 50, raw=False)

    # 3. Volatility ratio
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


def calculate_regime_score(indicators, params=None):
    """Calculate regime score and classification with custom parameters"""
    if params is None:
        params = {
            'adx_weight': 0.4, 'momentum_weight': 0.3, 'breakout_weight': 0.3,
            'vol_ratio_weight': 0.6, 'ma_dist_weight': 0.4,
            'adx_norm_factor': 30, 'momentum_norm_factor': 50, 'breakout_norm_factor': 3,
            'vol_ratio_norm_factor': 2, 'ma_dist_norm_factor': 20
        }

    # Normalize indicators using custom parameters
    adx_norm = np.clip(indicators['adx'] / params['adx_norm_factor'], 0, 1)
    momentum_norm = np.clip(indicators['momentum'] / params['momentum_norm_factor'], 0, 1)
    breakout_norm = np.clip(indicators['breakout'] * params['breakout_norm_factor'], 0, 1)
    vol_ratio_norm = np.clip((indicators['vol_ratio'] - 1) / params['vol_ratio_norm_factor'], 0, 1)
    ma_dist_norm = np.clip(indicators['ma_distance'] * params['ma_dist_norm_factor'], 0, 1)

    # Calculate trending and ranging scores using custom weights
    trending = (adx_norm * params['adx_weight'] +
                momentum_norm * params['momentum_weight'] +
                breakout_norm * params['breakout_weight'])

    ranging = (vol_ratio_norm * params['vol_ratio_weight'] +
               ma_dist_norm * params['ma_dist_weight'])

    # Composite score
    composite = trending - ranging
    composite_smooth = composite.rolling(3, center=True).mean()

    # Regime classification
    regime = pd.Series(0, index=composite_smooth.index)
    regime = np.where(composite_smooth > 0.15, 1, regime)  # Trending
    regime = np.where(composite_smooth < -0.15, -1, regime)  # Ranging
    regime = pd.Series(regime, index=composite_smooth.index)

    return {
        'trending_score': trending,
        'ranging_score': ranging,
        'composite_score': composite,
        'composite_smooth': composite_smooth,
        'regime': regime,
        'adx_norm': adx_norm,
        'momentum_norm': momentum_norm,
        'breakout_norm': breakout_norm,
        'vol_ratio_norm': vol_ratio_norm,
        'ma_dist_norm': ma_dist_norm
    }


def calculate_agg_percentile(data, weights):
    """Calculate aggregate percentile with given weights"""
    value_w, carry_w, momentum_w = weights
    return (data['Value_Percentile'] * value_w +
            data['Carry_Percentile'] * carry_w +
            data['Momentum_Percentile'] * momentum_w) / 100


def optimize_weights_by_regime(data, regime_data, buy_zone, sell_zone, weight_step=10,
                               transaction_cost_bps=2, capital_allocation_pct=100,
                               initial_cash=100000, optimization_goal="Max PnL",
                               default_weights=None, min_regime_days=30):
    """Optimize weights separately for each regime"""

    if default_weights is None:
        default_weights = {
            1: [20, 20, 60],  # Trending: favor momentum
            -1: [50, 30, 20],  # Ranging: favor value
            0: [33, 33, 34]  # Unknown: balanced
        }

    # Generate weight combinations
    weight_combinations = []
    for value_w in range(weight_step, 101, weight_step):
        for carry_w in range(weight_step, 101 - value_w + weight_step, weight_step):
            momentum_w = 100 - value_w - carry_w
            if momentum_w >= weight_step:
                weight_combinations.append([value_w, carry_w, momentum_w])

    regime_weights = {}
    regime_names = {1: 'Trending', -1: 'Ranging', 0: 'Unknown'}

    # Check regime distribution first
    regime_counts = regime_data['regime'].value_counts()
    st.write("**Regime Distribution in Train Period:**")
    for regime_type in [1, -1, 0]:
        regime_name = regime_names[regime_type]
        count = regime_counts.get(regime_type, 0)
        percentage = (count / len(regime_data['regime'].dropna())) * 100 if len(
            regime_data['regime'].dropna()) > 0 else 0
        st.write(f"- {regime_name}: {count} days ({percentage:.1f}%)")

    # Optimize for each regime separately
    for regime_type in [1, -1, 0]:  # Trending, Ranging, Unknown
        regime_name = regime_names[regime_type]

        # Filter data for this regime
        regime_mask = regime_data['regime'] == regime_type
        regime_count = regime_mask.sum()

        if regime_count < min_regime_days:  # Minimum threshold for optimization
            st.warning(
                f"⚠️ {regime_name} regime has only {regime_count} days (< {min_regime_days}) - using default weights {default_weights[regime_type]}")
            regime_weights[regime_type] = default_weights[regime_type]
            continue

        regime_subset = data[regime_mask].copy()

        best_result = {'total_pnl': -999999, 'weights': default_weights[regime_type]}

        # Test each weight combination on this regime
        for weights in weight_combinations:
            result = backtest_strategy(
                regime_subset, weights, buy_zone, sell_zone,
                False, 0, transaction_cost_bps, capital_allocation_pct,
                initial_cash, regime_aware=False
            )

            # Apply optimization criterion
            if optimization_goal == "Max PnL":
                score = result['total_pnl']
            else:  # Max PnL/Drawdown
                score = result['total_pnl'] / max(result['max_drawdown'], 0.01)

            if score > best_result.get('score', -999999):
                best_result = result
                best_result['weights'] = weights
                best_result['score'] = score

        regime_weights[regime_type] = best_result['weights']
        st.success(
            f"✅ {regime_name} regime optimal weights: Value:{best_result['weights'][0]}% Carry:{best_result['weights'][1]}% Momentum:{best_result['weights'][2]}%")

    return regime_weights


def backtest_strategy(data, weights, buy_zone, sell_zone, use_stop_loss, stop_loss_pct,
                      transaction_cost_bps, capital_allocation_pct, initial_cash=100000,
                      regime_weights=None, regime_data=None, regime_aware=True):
    """Enhanced backtest with regime-aware position sizing"""

    if len(data) < 50:
        return {'total_pnl': -999999, 'max_drawdown': 999, 'num_trades': 0, 'win_rate': 0, 'trades': []}

    data_copy = data.copy()

    # Calculate signals
    if regime_aware and regime_weights and regime_data is not None:
        # Use dynamic weights based on regime
        data_copy['Agg_Percentile'] = 0
        for i, (idx, row) in enumerate(data_copy.iterrows()):
            if idx in regime_data['regime'].index:
                current_regime = regime_data['regime'].loc[idx]
                if current_regime in regime_weights:
                    current_weights = regime_weights[current_regime]
                else:
                    current_weights = [33, 33, 34]  # Default weights

                data_copy.loc[idx, 'Agg_Percentile'] = calculate_agg_percentile(
                    data_copy.loc[[idx]], current_weights
                ).iloc[0]
    else:
        # Use fixed weights
        data_copy['Agg_Percentile'] = calculate_agg_percentile(data_copy, weights)

    data_copy['Signal'] = 0
    data_copy['Executed'] = False

    # Generate signals
    buy_min, buy_max = buy_zone
    sell_min, sell_max = sell_zone

    for idx, row in data_copy.iterrows():
        agg_perc = row['Agg_Percentile']
        if buy_min <= agg_perc <= buy_max:
            data_copy.loc[idx, 'Signal'] = 1
        elif sell_min <= agg_perc <= sell_max:
            data_copy.loc[idx, 'Signal'] = -1

    # Trading simulation with regime-aware position sizing
    cash = initial_cash
    position = 0
    position_size = 0
    entry_price = 0
    entry_date = None
    total_pnl = 0
    trades = []
    executed_trades = []
    equity_curve = [initial_cash]

    for i, (idx, row) in enumerate(data_copy.iterrows()):
        current_price = row['Close']
        signal = row['Signal']

        # Determine position sizing adjustment based on regime uncertainty
        if regime_aware and regime_data is not None and idx in regime_data['regime'].index:
            current_regime = regime_data['regime'].loc[idx]
            if current_regime == 0:  # Unknown regime
                size_multiplier = 0.5  # Reduce position size by 50%
            else:
                size_multiplier = 1.0
        else:
            size_multiplier = 1.0

        # Trading signals with T+1 execution
        if signal != 0 and signal != position:
            if i < len(data_copy) - 1:
                next_day_price = data_copy.iloc[i + 1]['Open']

                # Close existing position
                if position != 0:
                    pnl = (next_day_price - entry_price) * position_size
                    total_pnl += pnl
                    cash += pnl
                    cash -= abs(pnl) * (transaction_cost_bps / 10000)

                    return_pct = ((next_day_price - entry_price) / entry_price) * position * 100

                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': data_copy.index[i + 1],
                        'entry_price': entry_price,
                        'exit_price': next_day_price,
                        'position': 'LONG' if position > 0 else 'SHORT',
                        'pnl': pnl,
                        'return_pct': return_pct,
                        'exit_reason': 'SIGNAL',
                        'regime': regime_data['regime'].loc[idx] if regime_data is not None and idx in regime_data[
                            'regime'].index else 0
                    })

                    executed_trades.append(data_copy.index[i + 1])

                # Open new position with regime-adjusted sizing
                position = signal
                allocated_capital = cash * (capital_allocation_pct / 100) * size_multiplier
                position_size = allocated_capital / next_day_price * position
                entry_price = next_day_price
                entry_date = data_copy.index[i + 1]
                cash -= abs(allocated_capital) * (transaction_cost_bps / 10000)

                executed_trades.append(data_copy.index[i + 1])
                data_copy.loc[data_copy.index[i + 1], 'Executed'] = True

        # Track equity
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
            'exit_reason': 'FINAL',
            'regime': regime_data['regime'].iloc[-1] if regime_data is not None else 0
        })

    # Calculate metrics
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
        'data_with_signals': data_copy,
        'executed_trades': executed_trades
    }


# Header
st.title("🎯 Regime-Aware Dynamic Weight Backtester")
st.caption("Optimize weights by regime, backtest with regime-aware position sizing")

# Parameters
col1, col2, col3, col4 = st.columns(4)

with col1:
    lookback_years = st.selectbox("Train Years", [2, 3, 5, 7], index=1)
    test_start_date = st.date_input("Test Start Date", datetime.now().date() - timedelta(days=365 * 2))
    train_start_date = test_start_date - timedelta(days=lookback_years * 365)

with col2:
    optimization_goal = st.selectbox("Optimization Goal", ["Max PnL", "Max PnL/Drawdown Ratio"])
    buy_zone = st.slider("Buy Zone (%)", 80, 100, (90, 100))
    sell_zone = st.slider("Sell Zone (%)", 0, 20, (0, 10))

with col3:
    weight_step = st.selectbox("Weight Step (%)", [5, 10, 20], index=1)
    transaction_cost = st.number_input("Transaction Cost (bps)", 0, 50, 2)
    capital_allocation_pct = st.slider("Capital Allocation per Trade (%)", 10, 100, 100)

with col4:
    initial_capital = st.number_input("Capital ($)", 10000, 1000000, 100000, step=10000)
    zscore_lookback = st.number_input("Z-Score Lookback", value=63, min_value=20, max_value=252)
    min_regime_days = st.number_input("Min Regime Days", value=30, min_value=10, max_value=100,
                                      help="Minimum days required to optimize regime weights")

# Default Weights Configuration
with st.expander("⚙️ Default Weights (when insufficient regime data)", expanded=False):
    st.markdown(f"**These weights are used when a regime has <{min_regime_days} days of data:**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Trending Regime Defaults:**")
        default_trend_value = st.slider("Value %", 0, 100, 20, 5, key="def_trend_val")
        default_trend_carry = st.slider("Carry %", 0, 100, 20, 5, key="def_trend_carry")
        default_trend_momentum = st.slider("Momentum %", 0, 100, 60, 5, key="def_trend_mom")

    with col2:
        st.markdown("**Ranging Regime Defaults:**")
        default_range_value = st.slider("Value %", 0, 100, 50, 5, key="def_range_val")
        default_range_carry = st.slider("Carry %", 0, 100, 30, 5, key="def_range_carry")
        default_range_momentum = st.slider("Momentum %", 0, 100, 20, 5, key="def_range_mom")

    with col3:
        st.markdown("**Unknown Regime Defaults:**")
        default_unknown_value = st.slider("Value %", 0, 100, 33, 5, key="def_unk_val")
        default_unknown_carry = st.slider("Carry %", 0, 100, 33, 5, key="def_unk_carry")
        default_unknown_momentum = st.slider("Momentum %", 0, 100, 34, 5, key="def_unk_mom")

    # Validation
    trending_total = default_trend_value + default_trend_carry + default_trend_momentum
    ranging_total = default_range_value + default_range_carry + default_range_momentum
    unknown_total = default_unknown_value + default_unknown_carry + default_unknown_momentum

    col1, col2, col3 = st.columns(3)
    with col1:
        if trending_total != 100:
            st.error(f"Trending total: {trending_total}% (must equal 100%)")
        else:
            st.success(f"Trending total: {trending_total}%")
    with col2:
        if ranging_total != 100:
            st.error(f"Ranging total: {ranging_total}% (must equal 100%)")
        else:
            st.success(f"Ranging total: {ranging_total}%")
    with col3:
        if unknown_total != 100:
            st.error(f"Unknown total: {unknown_total}% (must equal 100%)")
        else:
            st.success(f"Unknown total: {unknown_total}%")

    default_weights = {
        1: [default_trend_value, default_trend_carry, default_trend_momentum],  # Trending
        -1: [default_range_value, default_range_carry, default_range_momentum],  # Ranging
        0: [default_unknown_value, default_unknown_carry, default_unknown_momentum]  # Unknown
    }

    st.markdown("""
    **Logic:**
    - **Trending**: High momentum (60%) to ride trends, low value/carry
    - **Ranging**: High value (50%) for mean reversion, low momentum  
    - **Unknown**: Balanced weights when regime unclear
    """)

# Regime Score Parameters Expander
with st.expander("🔧 Composite Regime Score Parameters", expanded=False):
    st.markdown("**Customize the regime detection algorithm:**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Trending Score Weights:**")
        adx_weight = st.slider("ADX Weight", 0.0, 1.0, 0.4, 0.1, key="adx_w")
        momentum_weight = st.slider("Momentum Weight", 0.0, 1.0, 0.3, 0.1, key="mom_w")
        breakout_weight = st.slider("Breakout Weight", 0.0, 1.0, 0.3, 0.1, key="break_w")

    with col2:
        st.markdown("**Ranging Score Weights:**")
        vol_ratio_weight = st.slider("Vol Ratio Weight", 0.0, 1.0, 0.6, 0.1, key="vol_w")
        ma_dist_weight = st.slider("MA Distance Weight", 0.0, 1.0, 0.4, 0.1, key="ma_w")

    with col3:
        st.markdown("**Normalization Parameters:**")
        adx_norm_factor = st.slider("ADX Norm Factor", 20, 50, 30, 5, key="adx_norm")
        momentum_norm_factor = st.slider("Momentum Norm Factor", 30, 70, 50, 10, key="mom_norm")
        breakout_norm_factor = st.slider("Breakout Norm Factor", 1, 10, 3, 1, key="break_norm")
        vol_ratio_norm_factor = st.slider("Vol Ratio Norm Factor", 1, 5, 2, 1, key="vol_norm")
        ma_dist_norm_factor = st.slider("MA Distance Norm Factor", 10, 30, 20, 5, key="ma_norm")

    # Show total weights and validation
    trending_total = adx_weight + momentum_weight + breakout_weight
    ranging_total = vol_ratio_weight + ma_dist_weight

    col1, col2 = st.columns(2)
    with col1:
        if abs(trending_total - 1.0) > 0.01:
            st.warning(f"Trending weights total: {trending_total:.2f} (should be close to 1.0)")
        else:
            st.success(f"Trending weights total: {trending_total:.2f}")
    with col2:
        if abs(ranging_total - 1.0) > 0.01:
            st.warning(f"Ranging weights total: {ranging_total:.2f} (should be close to 1.0)")
        else:
            st.success(f"Ranging weights total: {ranging_total:.2f}")

    regime_params = {
        'adx_weight': adx_weight,
        'momentum_weight': momentum_weight,
        'breakout_weight': breakout_weight,
        'vol_ratio_weight': vol_ratio_weight,
        'ma_dist_weight': ma_dist_weight,
        'adx_norm_factor': adx_norm_factor,
        'momentum_norm_factor': momentum_norm_factor,
        'breakout_norm_factor': breakout_norm_factor,
        'vol_ratio_norm_factor': vol_ratio_norm_factor,
        'ma_dist_norm_factor': ma_dist_norm_factor
    }

    st.markdown("""
    **Parameter Guide:**
    - **Trending weights**: Higher ADX/momentum/breakout = more trending bias
    - **Ranging weights**: Higher vol ratio/MA distance = more ranging bias  
    - **Norm factors**: Higher values = less sensitive indicators (more data needed for signal)
    """)

# Run Optimization
if st.button("🚀 Run Regime-Aware Optimization", type="primary"):

    with st.spinner("Loading data..."):
        # Load all required data
        train_start_str = train_start_date.strftime("%Y-%m-%d")

        _2yUS = get_fred_data("DGS2", train_start_str)
        _2yUS.columns = ["2y"]

        _5yUS = get_github_data("^FVX", train_start_str)
        _5yUS.columns = ["5y", "High", "Low", "Open"]

        _5yUS_real = get_fred_data("DFII5", train_start_str)
        _5yUS_real.columns = ["5y_Real"]
        _5yUS_real = _5yUS_real.interpolate(method="polynomial", order=2)

        _5yUS_fut = get_github_data("ZF=F", train_start_str)

        if any(df.empty for df in [_2yUS, _5yUS, _5yUS_real, _5yUS_fut]):
            st.error("Failed to load data")
            st.stop()

    with st.spinner("Building indicators..."):
        # Build fundamental indicators
        backtest_data = _2yUS.join(_5yUS_real).join(_5yUS).dropna()
        indicators = build_indicators(backtest_data)

        # Calculate percentiles
        for cols in ["5y_Real", "carry_normalized", "momentum"]:
            indicators[f"{cols}_z"] = zscore(indicators[cols], zscore_lookback)
            indicators[f"{cols}_percentile"] = indicators[f"{cols}_z"].rolling(zscore_lookback).apply(
                lambda x: percentile_score(x))

        # Final dataset
        final_data = indicators[
            ["5y", "5y_Real_percentile", "carry_normalized_percentile", "momentum_percentile"]].join(_5yUS_fut)
        final_data.columns = ["5y_yield", "Value_Percentile", "Carry_Percentile", "Momentum_Percentile", "Open", "High",
                              "Low", "Close"]
        final_data.dropna(inplace=True)
        final_data.index = pd.Series(final_data.index).dt.date

        # Calculate regime indicators
        regime_indicators = fast_regime_indicators(final_data)
        regime_results = calculate_regime_score(regime_indicators, regime_params)

    with st.spinner("Optimizing weights by regime..."):
        # Train/Test Split
        train_data = final_data.loc[final_data.index < test_start_date].copy()
        test_data = final_data.loc[final_data.index >= test_start_date].copy()

        train_regime = {k: v.loc[v.index < test_start_date] for k, v in regime_results.items()}
        test_regime = {k: v.loc[v.index >= test_start_date] for k, v in regime_results.items()}

        # Optimize weights for each regime using train data
        regime_weights = optimize_weights_by_regime(
            train_data, train_regime, buy_zone, sell_zone,
            weight_step, transaction_cost, capital_allocation_pct,
            initial_capital, optimization_goal, default_weights, min_regime_days
        )

    with st.spinner("Running backtests..."):
        # Backtest on train period
        train_results = backtest_strategy(
            train_data, [33, 33, 34], buy_zone, sell_zone, False, 0,
            transaction_cost, capital_allocation_pct, initial_capital,
            regime_weights, train_regime, regime_aware=True
        )

        # Backtest on test period
        test_results = backtest_strategy(
            test_data, [33, 33, 34], buy_zone, sell_zone, False, 0,
            transaction_cost, capital_allocation_pct, initial_capital,
            regime_weights, test_regime, regime_aware=True
        )

    # Store results
    st.session_state.update({
        'regime_weights': regime_weights,
        'train_results': train_results,
        'test_results': test_results,
        'train_data': train_data,
        'test_data': test_data,
        'train_regime': train_regime,
        'test_regime': test_regime,
        'regime_indicators': regime_indicators,
        'final_data': final_data
    })

# Display Results
if 'regime_weights' in st.session_state:
    regime_weights = st.session_state.regime_weights
    train_results = st.session_state.train_results
    test_results = st.session_state.test_results
    train_data = st.session_state.train_data
    test_data = st.session_state.test_data
    train_regime = st.session_state.train_regime
    test_regime = st.session_state.test_regime
    regime_indicators = st.session_state.regime_indicators
    final_data = st.session_state.final_data

    # Results Header
    st.markdown("---")
    st.subheader("🏆 Regime-Optimized Weights")

    col1, col2, col3 = st.columns(3)
    regime_names = {1: 'Trending', -1: 'Ranging', 0: 'Unknown'}

    # Display all three regimes, showing missing ones
    for i, regime_type in enumerate([1, -1, 0]):
        with [col1, col2, col3][i]:
            regime_name = regime_names[regime_type]
            if regime_type in regime_weights:
                weights = regime_weights[regime_type]
                st.markdown(f"**✅ {regime_name} Regime:**")
                st.write(f"Value: {weights[0]}%")
                st.write(f"Carry: {weights[1]}%")
                st.write(f"Momentum: {weights[2]}%")
            else:
                st.markdown(f"**❌ {regime_name} Regime:**")
                st.write("Insufficient data")
                st.write("Using default weights")
                st.write("Check default weights expander")

    # Performance Comparison
    st.markdown("## 📊 Performance Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🚂 TRAIN Period")
        st.metric("P&L", f"${train_results['total_pnl']:,.0f}", f"{train_results['total_return_pct']:+.1f}%")
        st.metric("Max Drawdown", f"{train_results['max_drawdown']:.1f}%")
        st.metric("Trades", f"{train_results['num_trades']}", f"Win Rate: {train_results['win_rate']:.1f}%")

    with col2:
        st.markdown("### 🧪 TEST Period")
        st.metric("P&L", f"${test_results['total_pnl']:,.0f}", f"{test_results['total_return_pct']:+.1f}%")
        st.metric("Max Drawdown", f"{test_results['max_drawdown']:.1f}%")
        st.metric("Trades", f"{test_results['num_trades']}", f"Win Rate: {test_results['win_rate']:.1f}%")

    # Period Selection
    period_choice = st.radio("📊 Analysis Period", ["Train", "Test", "Combined"], horizontal=True)

    if period_choice == "Train":
        selected_data = train_data
        selected_results = train_results
        selected_regime = train_regime
        period_label = "TRAIN"
    elif period_choice == "Test":
        selected_data = test_data
        selected_results = test_results
        selected_regime = test_regime
        period_label = "TEST"
    else:
        selected_data = final_data
        # Combine results for display
        combined_trades = train_results['trades'] + test_results['trades']
        selected_results = {
            'trades': combined_trades,
            'data_with_signals': pd.concat([train_results['data_with_signals'], test_results['data_with_signals']])
        }
        selected_regime = {k: pd.concat([train_regime[k], test_regime[k]]) for k in train_regime.keys()}
        period_label = "COMBINED"

    # Main Charts
    st.markdown(f"## 📈 Price Chart with Regime & Trades ({period_label})")

    fig_main = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Price + Regime Background + Trading Signals",
            "Aggregate Percentile + Trading Zones",
            "Composite Regime Score"
        ),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25],
        shared_xaxes=True,
        shared_yaxes=False
    )

    # Price chart with regime background
    fig_main.add_trace(
        go.Scatter(
            x=selected_data.index,
            y=selected_data['Close'],
            mode='lines',
            name='5Y Futures',
            line=dict(color='black', width=1)
        ),
        row=1, col=1
    )


    # Add regime background (sample every 5th point for speed)
    regime_sample = selected_regime['regime'][::5]
    for i in range(len(regime_sample) - 1):
        if regime_sample.iloc[i] == 1:  # Trending
            fig_main.add_vrect(
                x0=regime_sample.index[i],
                x1=regime_sample.index[i + 1],
                fillcolor="lightblue",
                opacity=0.2,
                layer="below",
                line_width=0,
                row=1, col=1
            )
        elif regime_sample.iloc[i] == -1:  # Ranging
            fig_main.add_vrect(
                x0=regime_sample.index[i],
                x1=regime_sample.index[i + 1],
                fillcolor="lightyellow",
                opacity=0.2,
                layer="below",
                line_width=0,
                row=1, col=1
            )

    # Trading signals on price chart
    data_with_signals = selected_results['data_with_signals']

    # Executed buy signals
    executed_buys = data_with_signals[(data_with_signals['Signal'] == 1) & (data_with_signals['Executed'] == True)]
    if len(executed_buys) > 0:
        fig_main.add_trace(
            go.Scatter(
                x=executed_buys.index,
                y=executed_buys['Close'],
                mode='markers',
                name='BUY Executed',
                marker=dict(color='green', size=12, symbol='x', line=dict(width=3))
            ),
            row=1, col=1
        )

    # Executed sell signals
    executed_sells = data_with_signals[(data_with_signals['Signal'] == -1) & (data_with_signals['Executed'] == True)]
    if len(executed_sells) > 0:
        fig_main.add_trace(
            go.Scatter(
                x=executed_sells.index,
                y=executed_sells['Close'],
                mode='markers',
                name='SELL Executed',
                marker=dict(color='red', size=12, symbol='x', line=dict(width=3))
            ),
            row=1, col=1
        )

    # Non-executed signals
    non_exec_buys = data_with_signals[(data_with_signals['Signal'] == 1) & (data_with_signals['Executed'] == False)]
    if len(non_exec_buys) > 0:
        fig_main.add_trace(
            go.Scatter(
                x=non_exec_buys.index,
                y=non_exec_buys['Close'],
                mode='markers',
                name='BUY Signal',
                marker=dict(color='lightgreen', size=6, symbol='triangle-up'),
                opacity=0.6
            ),
            row=1, col=1
        )

    non_exec_sells = data_with_signals[(data_with_signals['Signal'] == -1) & (data_with_signals['Executed'] == False)]
    if len(non_exec_sells) > 0:
        fig_main.add_trace(
            go.Scatter(
                x=non_exec_sells.index,
                y=non_exec_sells['Close'],
                mode='markers',
                name='SELL Signal',
                marker=dict(color='lightcoral', size=6, symbol='triangle-down'),
                opacity=0.6
            ),
            row=1, col=1
        )

    # Aggregate percentile chart
    fig_main.add_trace(
        go.Scatter(
            x=data_with_signals.index,
            y=data_with_signals['Agg_Percentile'],
            mode='lines',
            name='Dynamic Agg Percentile',
            line=dict(color='purple', width=2),
            yaxis="y1"
        ),
        row=2, col=1
    )

    # Fixed regime weight comparisons
    st.markdown("**Compare vs Fixed Regime Weights:**")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_trending_fixed = st.checkbox("Show Trending Agg%", key="trending_fixed")
    with col2:
        show_ranging_fixed = st.checkbox("Show Ranging Agg%", key="ranging_fixed")
    with col3:
        show_unknown_fixed = st.checkbox("Show Unknown Agg%", key="unknown_fixed")

    # Calculate and display fixed weight aggregate percentiles
    if show_trending_fixed and 1 in regime_weights:
        trending_agg = calculate_agg_percentile(selected_data, regime_weights[1])
        fig_main.add_trace(
            go.Scatter(
                x=selected_data.index,
                y=trending_agg,
                mode='lines',
                name=f'Trending Fixed ({regime_weights[1][0]}/{regime_weights[1][1]}/{regime_weights[1][2]})',
                line=dict(color='lightblue', width=4, dash='dash'),
                opacity=0.7
            ),
            row=2, col=1
        )

    if show_ranging_fixed and -1 in regime_weights:
        ranging_agg = calculate_agg_percentile(selected_data, regime_weights[-1])
        fig_main.add_trace(
            go.Scatter(
                x=selected_data.index,
                y=ranging_agg,
                mode='lines',
                name=f'Ranging Fixed ({regime_weights[-1][0]}/{regime_weights[-1][1]}/{regime_weights[-1][2]})',
                line=dict(color='orange', width=4, dash='dot'),
                opacity=0.7
            ),
            row=2, col=1
        )

    if show_unknown_fixed and 0 in regime_weights:
        unknown_agg = calculate_agg_percentile(selected_data, regime_weights[0])
        fig_main.add_trace(
            go.Scatter(
                x=selected_data.index,
                y=unknown_agg,
                mode='lines',
                name=f'Unknown Fixed ({regime_weights[0][0]}/{regime_weights[0][1]}/{regime_weights[0][2]})',
                line=dict(color='gray', width=4, dash='dash'),
                opacity=0.7
            ),
            row=2, col=1
        )
    # Buy/sell zones
    fig_main.add_hrect(y0=buy_zone[0], y1=buy_zone[1],
                       fillcolor="green", opacity=0.2,
                       annotation_text="BUY ZONE", annotation_position="top left",
                       row=2, col=1)
    fig_main.add_hrect(y0=sell_zone[0], y1=sell_zone[1],
                       fillcolor="red", opacity=0.2,
                       annotation_text="SELL ZONE", annotation_position="bottom left",
                       row=2, col=1)

    # Composite regime score
    fig_main.add_trace(
        go.Scatter(
            x=selected_regime['composite_smooth'].index,
            y=selected_regime['composite_smooth'],
            mode='lines',
            name='Regime Score',
            line=dict(color='orange', width=2)
        ),
        row=3, col=1
    )

    # Regime thresholds
    fig_main.add_hline(y=0.15, line_dash="dash", line_color="blue",
                       annotation_text="Trending", row=3, col=1)
    fig_main.add_hline(y=-0.15, line_dash="dash", line_color="orange",
                       annotation_text="Ranging", row=3, col=1)
    fig_main.add_hline(y=0, line_color="gray", row=3, col=1)

    # Add train/test separator line
    if period_choice == "Combined":
        fig_main.add_vline(x=test_start_date, line_dash="solid", line_color="red",
                           annotation_text="Test Start", row=1, col=1)
        fig_main.add_vline(x=test_start_date, line_dash="solid", line_color="red", row=2, col=1)
        fig_main.add_vline(x=test_start_date, line_dash="solid", line_color="red", row=3, col=1)

    fig_main.update_layout(height=800, showlegend=True)
    fig_main.update_yaxes(title_text="Price", row=1, col=1)
    fig_main.update_yaxes(title_text="Percentile", row=2, col=1)
    fig_main.update_yaxes(title_text="Score", row=3, col=1)
    fig_main.update_xaxes(title_text="Date", row=3, col=1)


    st.plotly_chart(fig_main, use_container_width=True)

    # Key Indicators Analysis
    st.markdown(f"## 🔍 Key Indicators Analysis ({period_label})")

    fig_indicators = make_subplots(
        rows=2, cols=3,
        subplot_titles=("ADX (Trend Strength)", "Momentum", "Breakout Strength",
                        "Volatility Ratio", "MA Distance", "Trending vs Ranging"),
        vertical_spacing=0.15
    )

    # Filter indicators for selected period
    if period_choice == "Train":
        indicator_data = {k: v.loc[v.index < test_start_date] for k, v in regime_indicators.items()}
        regime_data_filtered = train_regime
    elif period_choice == "Test":
        indicator_data = {k: v.loc[v.index >= test_start_date] for k, v in regime_indicators.items()}
        regime_data_filtered = test_regime
    else:
        indicator_data = regime_indicators
        regime_data_filtered = selected_regime

    # ADX
    fig_indicators.add_trace(
        go.Scatter(x=indicator_data['adx'].index, y=indicator_data['adx'],
                   mode='lines', name='ADX', line=dict(color='red')),
        row=1, col=1
    )
    fig_indicators.add_hline(y=25, line_dash="dash", line_color="gray", row=1, col=1)

    # Momentum
    fig_indicators.add_trace(
        go.Scatter(x=indicator_data['momentum'].index, y=indicator_data['momentum'],
                   mode='lines', name='Momentum', line=dict(color='blue')),
        row=1, col=2
    )

    # Breakout
    fig_indicators.add_trace(
        go.Scatter(x=indicator_data['breakout'].index, y=indicator_data['breakout'],
                   mode='lines', name='Breakout', line=dict(color='green')),
        row=1, col=3
    )

    # Volatility Ratio
    fig_indicators.add_trace(
        go.Scatter(x=indicator_data['vol_ratio'].index, y=indicator_data['vol_ratio'],
                   mode='lines', name='Vol Ratio', line=dict(color='orange')),
        row=2, col=1
    )
    fig_indicators.add_hline(y=1, line_dash="dash", line_color="gray", row=2, col=1)

    # MA Distance
    fig_indicators.add_trace(
        go.Scatter(x=indicator_data['ma_distance'].index, y=indicator_data['ma_distance'],
                   mode='lines', name='MA Distance', line=dict(color='purple')),
        row=2, col=2
    )

    # Trending vs Ranging scores
    fig_indicators.add_trace(
        go.Scatter(x=regime_data_filtered['trending_score'].index, y=regime_data_filtered['trending_score'],
                   mode='lines', name='Trending Score', line=dict(color='blue')),
        row=2, col=3
    )
    fig_indicators.add_trace(
        go.Scatter(x=regime_data_filtered['ranging_score'].index, y=regime_data_filtered['ranging_score'],
                   mode='lines', name='Ranging Score', line=dict(color='orange')),
        row=2, col=3
    )

    fig_indicators.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_indicators, use_container_width=True)

    # Trade Log
    st.markdown(f"## 📋 Trade Log ({period_label})")

    trades = selected_results['trades']
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date']).dt.strftime('%Y-%m-%d')
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date']).dt.strftime('%Y-%m-%d')
        trades_df['pnl'] = trades_df['pnl'].round(0).astype(int)
        trades_df['return_pct'] = trades_df['return_pct'].round(2)
        trades_df['entry_price'] = trades_df['entry_price'].round(3)
        trades_df['exit_price'] = trades_df['exit_price'].round(3)

        # Add regime labels
        regime_map = {1: 'Trending', -1: 'Ranging', 0: 'Unknown'}
        trades_df['regime_label'] = trades_df['regime'].map(regime_map)

        # Add period labels for combined view
        if period_choice == "Combined":
            trades_df['period'] = trades_df['entry_date'].apply(
                lambda x: 'TRAIN' if pd.to_datetime(x).date() < test_start_date else 'TEST'
            )


        # Color coding function
        def color_pnl(val):
            return 'color: green' if val > 0 else 'color: red'


        def color_regime(val):
            colors = {'Trending': 'background-color: lightblue',
                      'Ranging': 'background-color: lightyellow',
                      'Unknown': 'background-color: lightgray'}
            return colors.get(val, '')


        styled_trades = trades_df.style.applymap(color_pnl, subset=['pnl', 'return_pct'])
        styled_trades = styled_trades.applymap(color_regime, subset=['regime_label'])

        st.dataframe(styled_trades, hide_index=True, use_container_width=True)

        # Trade Statistics
        col1, col2, col3, col4 = st.columns(4)

        profitable_trades = len([t for t in trades if t['pnl'] > 0])
        total_trades = len(trades)
        avg_trade = np.mean([t['pnl'] for t in trades])

        with col1:
            st.metric("Profitable Trades", f"{profitable_trades}/{total_trades}")
        with col2:
            st.metric("Average Trade P&L", f"${avg_trade:,.0f}")
        with col3:
            best_trade = max([t['pnl'] for t in trades])
            st.metric("Best Trade", f"${best_trade:,.0f}")
        with col4:
            worst_trade = min([t['pnl'] for t in trades])
            st.metric("Worst Trade", f"${worst_trade:,.0f}")

        # Regime-specific performance
        if len(trades_df) > 0:
            st.markdown("### 📊 Performance by Regime")

            regime_performance = trades_df.groupby('regime_label').agg({
                'pnl': ['count', 'sum', 'mean'],
                'return_pct': 'mean'
            }).round(2)

            regime_performance.columns = ['Trades', 'Total PnL', 'Avg PnL', 'Avg Return %']
            st.dataframe(regime_performance, use_container_width=True)

    else:
        st.info("No trades executed in selected period")

    # Export Results
    st.markdown("## 📁 Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Export regime weights config
        config = {
            'regime_weights': regime_weights,
            'train_performance': {
                'total_pnl': train_results['total_pnl'],
                'total_return_pct': train_results['total_return_pct'],
                'max_drawdown': train_results['max_drawdown'],
                'num_trades': train_results['num_trades'],
                'win_rate': train_results['win_rate']
            },
            'test_performance': {
                'total_pnl': test_results['total_pnl'],
                'total_return_pct': test_results['total_return_pct'],
                'max_drawdown': test_results['max_drawdown'],
                'num_trades': test_results['num_trades'],
                'win_rate': test_results['win_rate']
            },
            'parameters': {
                'buy_zone': buy_zone,
                'sell_zone': sell_zone,
                'transaction_cost_bps': transaction_cost,
                'capital_allocation_pct': capital_allocation_pct,
                'min_regime_days': min_regime_days
            }
        }

        st.download_button(
            "📥 Download Config",
            json.dumps(config, indent=2, default=str),
            f"regime_weights_config_{datetime.now().strftime('%Y%m%d')}.json"
        )

    with col2:
        # Export trades
        if trades:
            trades_csv = pd.DataFrame(trades).to_csv(index=False)
            st.download_button(
                f"📥 Download {period_label} Trades",
                trades_csv,
                f"trades_{period_label.lower()}_{datetime.now().strftime('%Y%m%d')}.csv"
            )

    with col3:
        # Export regime data
        regime_export = pd.DataFrame({
            'Date': final_data.index,
            'Close': final_data['Close'],
            'Regime_Score': selected_regime['composite_smooth'],
            'Regime': selected_regime['regime'],
            'Regime_Label': selected_regime['regime'].map({1: 'Trending', -1: 'Ranging', 0: 'Unknown'}),
            'ADX': regime_indicators['adx'],
            'Momentum': regime_indicators['momentum']
        }).dropna()

        regime_csv = regime_export.to_csv(index=False)
        st.download_button(
            "📥 Download Regime Data",
            regime_csv,
            f"regime_data_{datetime.now().strftime('%Y%m%d')}.csv"
        )

else:
    st.info("Click 'Run Regime-Aware Optimization' to start analysis")

    # Information panel
    st.markdown("""
    ## 🎯 How This Works

    **1. Regime Detection:**
    - Calculates multiple technical indicators (ADX, momentum, breakout strength, etc.)
    - Combines them into a composite regime score
    - Classifies market as Trending (+0.15), Ranging (-0.15), or Unknown

    **2. Weight Optimization by Regime:**
    - Tests all weight combinations on each regime separately during train period
    - Finds optimal Value/Carry/Momentum weights for each market condition
    - Uses Max PnL or Max PnL/Drawdown ratio as optimization criterion

    **3. Dynamic Trading:**
    - Applies regime-specific weights in real-time
    - Reduces position size by 50% during Unknown regimes
    - Uses T+1 execution (signal today, trade tomorrow at open)

    **4. Comprehensive Analysis:**
    - Train/test split validation
    - Visual regime identification on price charts
    - Individual indicator analysis for trade understanding
    - Detailed trade log with regime context
    """)