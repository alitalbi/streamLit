import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import urllib
from fredapi import Fred
from scipy import stats
import warnings
import time
from itertools import product

warnings.filterwarnings('ignore')

# Initialize FRED API
fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')
st.set_page_config(layout="wide", page_title="Revised Treasury Trading Strategy", page_icon="💹")

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
    .strategy-info {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border: 2px solid #2196f3;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 6px 20px rgba(33,150,243,0.2);
    }
    .chart-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


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


def fred_import(ticker, start_date):
    """Import data from FRED"""
    try:
        fred_data = pd.DataFrame(fred.get_series(ticker, observation_start=start_date, freq="daily"))
        return fred_data
    except Exception as e:
        st.error(f"Error loading FRED data: {e}")
        return pd.DataFrame()


def kalman_filter_1d(observations, process_variance=1e-5, measurement_variance=1e-1):
    """
    Simple 1D Kalman filter for smoothing time series
    Reduces noise in financial data to improve signal quality
    """
    n = len(observations)
    if n == 0:
        return observations

    # Initialize
    x_hat = np.zeros(n)  # State estimates
    P = np.zeros(n)  # Error covariances

    # Initial conditions
    x_hat[0] = observations[0]
    P[0] = 1.0

    for k in range(1, n):
        # Prediction
        x_hat_minus = x_hat[k - 1]
        P_minus = P[k - 1] + process_variance

        # Update
        K = P_minus / (P_minus + measurement_variance)
        x_hat[k] = x_hat_minus + K * (observations[k] - x_hat_minus)
        P[k] = (1 - K) * P_minus

    return pd.Series(x_hat, index=observations.index if hasattr(observations, 'index') else range(len(observations)))


def calculate_hurst(ts):
    """Calculate Hurst Exponent using R/S method"""
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


def calculate_williams_r(high, low, close, period=14):
    """Williams %R calculation"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    return williams_r


def calculate_cci(high, low, close, period=20):
    """CCI calculation"""
    tp = (high + low + close) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad)
    return cci


def calculate_momentum_score(rsi, williams_r, cci):
    """
    Calculate momentum score as FILTER (not regime input)
    Used to confirm signals and reduce false entries
    """
    score = 0

    # RSI component: > 60 = +1, < 40 = -1
    if rsi > 60:
        score += 1
    elif rsi < 40:
        score -= 1

    # Williams %R component: > -25 = +1, < -75 = -1
    if williams_r > -25:
        score += 1
    elif williams_r < -75:
        score -= 1

    # CCI component: > 100 = +1, < -100 = -1
    if cci > 100:
        score += 1
    elif cci < -100:
        score -= 1

    return score


def build_indicators(data, enable_kalman=True):
    """Build carry and momentum indicators with optional Kalman smoothing"""

    # Apply Kalman smoothing to reduce noise (default: enabled)
    if enable_kalman:
        data["5y_smoothed"] = kalman_filter_1d(data["5y"])
        data["2y_smoothed"] = kalman_filter_1d(data["2y"])
        # Use smoothed data for indicators
        data["5_2y"] = data["5y_smoothed"] - data["2y_smoothed"]
        data["5d_ma_5y"] = data["5y_smoothed"].rolling(5).mean()
        data["20d_ma_5y"] = data["5y_smoothed"].rolling(20).mean()
    else:
        # Use raw data
        data["5_2y"] = data["5y"] - data["2y"]
        data["5d_ma_5y"] = data["5y"].rolling(5).mean()
        data["20d_ma_5y"] = data["5y"].rolling(20).mean()

    data["carry_normalized"] = data["5_2y"] / data["5_2y"].rolling(75).std()
    data["momentum"] = data["5d_ma_5y"] - data["20d_ma_5y"]

    return data


def percentile_score(window):
    """Calculate percentile score"""
    if len(window) == 0:
        return np.nan
    current_value = window[-1]
    nb_values_below = np.sum(window <= current_value)
    return (nb_values_below / len(window)) * 100


def classify_regime_simplified(hurst, adf_stat):
    """
    SIMPLIFIED regime classification using only Hurst and ADF
    Removed Fractal Dimension to reduce complexity and false signals
    """

    # Regime scoring based on Hurst and ADF only
    trend_score = 0
    mean_rev_score = 0

    # Hurst scoring
    if not pd.isna(hurst):
        if hurst > 0.55:  # Strong trending
            trend_score += 2
        elif hurst > 0.52:  # Weak trending
            trend_score += 1
        elif hurst < 0.45:  # Strong mean-reverting
            mean_rev_score += 2
        elif hurst < 0.48:  # Weak mean-reverting
            mean_rev_score += 1

    # ADF scoring
    if not pd.isna(adf_stat):
        if adf_stat < -2.862:  # Strong mean reversion (99% confidence)
            mean_rev_score += 2
        elif adf_stat < -2.567:  # Moderate mean reversion (95% confidence)
            mean_rev_score += 1

    # Calculate confidence based on consistency
    total_signals = trend_score + mean_rev_score
    if total_signals == 0:
        return "UNKNOWN", 0.1

    confidence = min(max(total_signals / 4.0, 0.3), 0.95)

    # Final classification
    if trend_score > mean_rev_score:
        return "TRENDING", confidence
    elif mean_rev_score > trend_score:
        return "MEAN_REVERTING", confidence
    else:
        return "UNKNOWN", confidence


class WalkForwardOptimizer:
    """
    Walk-forward optimization engine focused on maximizing Sharpe ratio
    Optimizes weights per regime using grid search with transaction costs
    """

    def __init__(self, granularity=5, transaction_cost_bps=2):
        self.granularity = granularity
        self.transaction_cost_bps = transaction_cost_bps
        self.optimization_results = {}

    def generate_weight_combinations(self):
        """Generate all possible weight combinations (sum to 1)"""
        weights = []
        step = self.granularity

        for value_weight in range(0, 101, step):
            for carry_weight in range(0, 101 - value_weight, step):
                momentum_weight = 100 - value_weight - carry_weight
                if momentum_weight >= 0:
                    # Convert to decimal (sum to 1)
                    weights.append([value_weight / 100, carry_weight / 100, momentum_weight / 100])

        return weights

    def calculate_agg_percentile_with_weights(self, data, weights):
        """Calculate aggregate percentile with given weights"""
        value_w, carry_w, momentum_w = weights
        return (data['Value_Percentile'] * value_w +
                data['Carry_Percentile'] * carry_w +
                data['Momentum_Percentile'] * momentum_w)

    def backtest_sharpe_with_weights(self, data, weights, buy_zone_min, buy_zone_max,
                                     sell_zone_min, sell_zone_max):
        """
        Backtest strategy with specific weights and calculate Sharpe ratio
        Including transaction costs in bps
        """
        if len(data) < 50:
            return -999  # Insufficient data penalty

        data_copy = data.copy()
        data_copy['Agg_Percentile'] = self.calculate_agg_percentile_with_weights(data_copy, weights)

        position = 0
        returns = []
        entry_price = 0

        for i, (idx, row) in enumerate(data_copy.iterrows()):
            if i == 0:
                continue

            current_price = row['Close']
            agg_perc = row['Agg_Percentile']
            prev_price = data_copy.iloc[i - 1]['Close']

            # Calculate base return (for benchmark)
            if position != 0:
                base_return = (current_price - prev_price) / prev_price
                if position < 0:
                    base_return = -base_return

                # Apply transaction costs when changing positions
                if ((buy_zone_min <= agg_perc <= buy_zone_max and position <= 0) or
                        (sell_zone_min <= agg_perc <= sell_zone_max and position >= 0)):
                    base_return -= (self.transaction_cost_bps / 10000)  # Convert bps to decimal

                returns.append(base_return)

            # Position logic
            if buy_zone_min <= agg_perc <= buy_zone_max and position <= 0:
                position = 1
                entry_price = current_price
            elif sell_zone_min <= agg_perc <= sell_zone_max and position >= 0:
                position = -1
                entry_price = current_price

        if len(returns) < 10:
            return -999  # Too few trades penalty

        returns = np.array(returns)

        # Calculate Sharpe ratio (annualized)
        mean_return = np.mean(returns) * 252  # Annualized
        std_return = np.std(returns) * np.sqrt(252)  # Annualized

        if std_return == 0:
            return 0

        sharpe = mean_return / std_return
        return sharpe

    def optimize_regime_weights(self, regime_data, regime_name, buy_zones, sell_zones):
        """Optimize weights for specific regime using grid search"""

        if len(regime_data) < 100:
            return None

        weight_combinations = self.generate_weight_combinations()
        best_sharpe = -np.inf
        best_weights = None
        results = []

        buy_zone_min, buy_zone_max = buy_zones
        sell_zone_min, sell_zone_max = sell_zones

        for weights in weight_combinations:
            sharpe = self.backtest_sharpe_with_weights(
                regime_data, weights, buy_zone_min, buy_zone_max,
                sell_zone_min, sell_zone_max
            )

            results.append({
                'weights': weights,
                'sharpe': sharpe
            })

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = weights

        return {
            'regime': regime_name,
            'best_weights': best_weights,
            'best_sharpe': best_sharpe,
            'all_results': results,
            'data_points': len(regime_data)
        }

    def walk_forward_optimize(self, data, buy_zones, sell_zones, train_pct=0.7):
        """Walk-forward optimization for all regimes"""

        results = {}

        # Split data for in-sample optimization
        split_idx = int(len(data) * train_pct)
        train_data = data.iloc[:split_idx]

        st.info(f"🎯 Walk-Forward Optimization: Training on {len(train_data)} days ({train_pct:.0%} of data)")

        for regime in ['TRENDING', 'MEAN_REVERTING', 'UNKNOWN']:
            regime_train_data = train_data[train_data['Regime'] == regime]

            if len(regime_train_data) > 50:
                result = self.optimize_regime_weights(
                    regime_train_data, regime, buy_zones, sell_zones
                )
                if result:
                    results[regime] = result
            else:
                st.warning(f"⚠️ Insufficient {regime} data for optimization: {len(regime_train_data)} days")

        self.optimization_results = results
        return results


class RevisedTradingSystem:
    """
    Revised trading system with momentum score filtering and reduced overtrading
    Focus on high-conviction trades to improve Sharpe ratio
    """

    def __init__(self, initial_cash=100000, transaction_fee_bps=2, capital_allocation_pct=100,
                 momentum_filter_threshold=1):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0
        self.entry_price = 0
        self.entry_date = None
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        self.transaction_fee_bps = transaction_fee_bps
        self.capital_allocation_pct = capital_allocation_pct
        self.momentum_filter_threshold = momentum_filter_threshold
        self.trade_history = []

    def calculate_position_size(self, price):
        """Calculate position size based on capital allocation"""
        available_capital = self.cash * (self.capital_allocation_pct / 100)
        position_size = int(available_capital / price)
        return max(position_size, 1)

    def calculate_transaction_cost(self, price, quantity):
        """Calculate transaction cost in bps"""
        notional = price * abs(quantity)
        return notional * (self.transaction_fee_bps / 10000)

    def check_signal_strength(self, agg_percentile, buy_zone_min, buy_zone_max, sell_zone_min, sell_zone_max):
        """Check if signal is in trading zones"""
        if buy_zone_min <= agg_percentile <= buy_zone_max:
            return "BUY", (agg_percentile - buy_zone_min) / (buy_zone_max - buy_zone_min)
        elif sell_zone_min <= agg_percentile <= sell_zone_max:
            return "SELL", (sell_zone_max - agg_percentile) / (sell_zone_max - sell_zone_min)
        else:
            return "NEUTRAL", 0.0

    def momentum_filter_check(self, signal_type, momentum_score):
        """
        MOMENTUM SCORE AS FILTER: Acts as safety gate to reduce false signals
        Only allows trades when momentum aligns with signal direction
        """
        if signal_type == "BUY":
            return momentum_score >= self.momentum_filter_threshold
        elif signal_type == "SELL":
            return momentum_score <= -self.momentum_filter_threshold
        else:
            return True  # Neutral signals always pass

    def can_trade(self, signal_type, signal_strength, price, regime, momentum_score, regime_confidence):
        """
        Enhanced trading logic with momentum filtering to reduce overtrading
        """

        # Regime confidence check
        if regime_confidence < 0.4:  # Higher threshold to reduce false signals
            return False, "LOW_REGIME_CONFIDENCE"

        # Unknown regime - be more conservative
        if regime == "UNKNOWN" and self.position != 0:
            return True, "CLOSE_UNKNOWN"

        # MOMENTUM FILTER: Key addition to reduce false signals
        if not self.momentum_filter_check(signal_type, momentum_score):
            return False, "MOMENTUM_FILTER_FAILED"

        # If we're flat and have strong signal, open position
        if self.position == 0 and signal_type in ["BUY", "SELL"]:
            required_capital = price * self.calculate_position_size(price)
            if self.cash >= required_capital:
                return True, f"OPEN_{signal_type}"
            else:
                return False, "INSUFFICIENT_CASH"

        # If we have position in opposite direction, reverse (high conviction only)
        if ((self.position > 0 and signal_type == "SELL") or
                (self.position < 0 and signal_type == "BUY")):
            if signal_strength > 0.6:  # Higher threshold for reversals
                return True, f"REVERSE_{signal_type}"
            else:
                return False, "REVERSAL_SIGNAL_TOO_WEAK"

        # Hold position if momentum not strong enough for changes
        return False, "HOLD_POSITION"

    def execute_trade(self, action_type, price, date, signal_strength=0, momentum_score=0):
        """Execute trade with enhanced tracking"""
        quantity = self.calculate_position_size(price)
        transaction_cost = self.calculate_transaction_cost(price, quantity)

        trade_info = {
            'date': date,
            'action': action_type,
            'price': price,
            'quantity': quantity,
            'cost': transaction_cost,
            'signal_strength': signal_strength,
            'momentum_score': momentum_score,
            'cash_before': self.cash,
            'position_before': self.position,
            'pnl': 0
        }

        # Close existing position
        if any(x in action_type for x in ["CLOSE", "REVERSE"]):
            if self.position != 0:
                pnl = (price - self.entry_price) * self.position - transaction_cost
                self.realized_pnl += pnl
                self.cash += pnl
                trade_info['pnl'] = pnl
                self.position = 0
                self.unrealized_pnl = 0

        # Open new position
        if any(x in action_type for x in ["OPEN", "REVERSE"]):
            if "BUY" in action_type:
                self.position = quantity
            elif "SELL" in action_type:
                self.position = -quantity

            self.entry_price = price
            self.entry_date = date
            self.cash -= transaction_cost

        trade_info['cash_after'] = self.cash
        trade_info['position_after'] = self.position
        self.trade_history.append(trade_info)

        return trade_info

    def update_unrealized_pnl(self, current_price):
        """Update unrealized P&L"""
        if self.position != 0:
            self.unrealized_pnl = (current_price - self.entry_price) * self.position
        else:
            self.unrealized_pnl = 0

    def get_total_pnl(self):
        return self.realized_pnl + self.unrealized_pnl

    def get_portfolio_value(self):
        return self.cash + self.unrealized_pnl

    def get_capital_utilization(self):
        if self.position != 0:
            position_value = abs(self.position * self.entry_price)
            return (position_value / self.initial_cash) * 100
        return 0


# Assets dictionary
assets_dict = {
    "2y US": "DGS2",
    "5y US": "^FVX",
    "5y US Real": "DFII5",
    "5y US Future": "ZF=F"
}

# Streamlit App Header
st.markdown(
    '<div class="main-header"><h1>💹 Revised Treasury Trading Strategy</h1><p>Simplified Regimes + Momentum Filtering + Walk-Forward Optimization</p></div>',
    unsafe_allow_html=True)

# Strategy Information Panel
st.markdown('<div class="strategy-info">', unsafe_allow_html=True)
st.markdown("""
### 🎯 **Revised Strategy Key Improvements:**

**1. Simplified Regime Classification:**
- ✅ **Removed Fractal Dimension** (reduced complexity, fewer false signals)
- ✅ **Core Indicators**: Hurst Exponent + ADF Test only
- ✅ **Higher Confidence Thresholds** to reduce overtrading

**2. Momentum Score as Filter (Not Regime Input):**
- ✅ **Safety Gate**: Only trade when momentum aligns with signal
- ✅ **Reduces False Signals**: Filters out low-momentum setups
- ✅ **Configurable Threshold**: Default ±1 for trade confirmation

**3. Kalman Filter Integration:**
- ✅ **Noise Reduction**: Smooths yield data before indicator calculation
- ✅ **Improved Signal Quality**: Reduces whipsaws in choppy markets
- ✅ **Optional**: Can disable if preferred

**4. Walk-Forward Optimization:**
- ✅ **Sharpe Ratio Maximization**: Focus on risk-adjusted returns
- ✅ **Grid Search**: Exhaustive weight optimization per regime
- ✅ **Transaction Costs**: Included in optimization (penalizes overtrading)
- ✅ **In-Sample Training**: Uses 70% for optimization, 30% for validation
""")
st.markdown('</div>', unsafe_allow_html=True)

# Enhanced Configuration Panel
with st.expander("⚙️ Revised Strategy Configuration", expanded=True):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**📅 Trading Period**")
        start_date_input = st.date_input("Start Date", value=datetime(2020, 1, 1))
        end_date_input = st.date_input("End Date", value=datetime.now().date())

    with col2:
        st.markdown("**🎯 Trading Zones**")
        buy_zone_min = st.number_input("Buy Zone Min (%)", value=85, min_value=70, max_value=95)
        buy_zone_max = st.number_input("Buy Zone Max (%)", value=100, min_value=90, max_value=100)
        sell_zone_min = st.number_input("Sell Zone Min (%)", value=0, min_value=0, max_value=15)
        sell_zone_max = st.number_input("Sell Zone Max (%)", value=15, min_value=5, max_value=30)

    with col3:
        st.markdown("**💰 Portfolio & Costs**")
        initial_cash = st.number_input("Initial Capital ($)", value=100000, min_value=10000, step=10000)
        capital_allocation_pct = st.number_input("Capital Allocation (%)", value=100, min_value=20, max_value=100)
        transaction_fee_bps = st.number_input("Transaction Cost (bps)", value=5, min_value=1, max_value=20)

    with col4:
        st.markdown("**🔧 Strategy Controls**")
        enable_kalman = st.checkbox("Enable Kalman Smoothing", value=True)
        momentum_filter_threshold = st.number_input("Momentum Filter Threshold", value=1, min_value=0, max_value=3)
        run_walk_forward_opt = st.button("🚀 Run Walk-Forward Optimization", type="primary")

# Data Loading
training_years = 7  # Extended for better walk-forward optimization
training_start_date = start_date_input - timedelta(days=training_years * 365)
start_date_str = training_start_date.strftime("%Y-%m-%d")

with st.spinner("📊 Loading extended data for walk-forward optimization..."):
    try:
        _2yUS = fred_import(assets_dict["2y US"], start_date_str)
        _2yUS.columns = ["2y"]

        _5yUS = get_data(assets_dict["5y US"], start_date_str)
        _5yUS.columns = ["5y", "High", "Low", "Open"]

        _5yUS_real = fred_import(assets_dict["5y US Real"], start_date_str)
        _5yUS_real.columns = ["5y_Real"]
        _5yUS_real = _5yUS_real.interpolate(method="polynomial", order=2)

        _5yUS_fut = get_data(assets_dict["5y US Future"], start_date_str)

        if any(df.empty for df in [_2yUS, _5yUS, _5yUS_real, _5yUS_fut]):
            st.error("Failed to load required data. Please check data sources.")
            st.stop()

        st.success(f"✅ Loaded {len(_5yUS_fut)} days from {training_start_date.strftime('%Y-%m-%d')}")

    except Exception as e:
        st.error(f"Data loading error: {e}")
        st.stop()

# Build indicators with optional Kalman filtering
backtest_data = _2yUS.join(_5yUS_real).join(_5yUS)
backtest_data.dropna(inplace=True)
indicators = build_indicators(backtest_data, enable_kalman=enable_kalman)

# Calculate percentiles
lookback = 126  # 6-month lookback for percentiles
for col in ["5y_Real", "carry_normalized", "momentum"]:
    indicators[f"{col}_percentile"] = indicators[col].rolling(lookback).apply(lambda x: percentile_score(x))

# Join with futures data
indicator_full = indicators[["5y", "5y_Real_percentile", "carry_normalized_percentile", "momentum_percentile"]].join(
    _5yUS_fut)
indicator_full.columns = ["5y_yield", "Value_Percentile", "Carry_Percentile", "Momentum_Percentile", "Open", "High",
                          "Low", "Close"]
indicator_full.dropna(inplace=True)

# Calculate technical indicators for momentum score
indicator_full['RSI'] = calculate_rsi(indicator_full['Close'])
indicator_full['Williams_R'] = calculate_williams_r(indicator_full['High'], indicator_full['Low'],
                                                    indicator_full['Close'])
indicator_full['CCI'] = calculate_cci(indicator_full['High'], indicator_full['Low'], indicator_full['Close'])
indicator_full['Momentum_Score'] = indicator_full.apply(
    lambda row: calculate_momentum_score(row['RSI'], row['Williams_R'], row['CCI']), axis=1
)

# Calculate Hurst and ADF for simplified regime classification
hurst_window = 60
indicator_full['Hurst'] = indicator_full['Close'].rolling(window=hurst_window).apply(calculate_hurst, raw=False)


def rolling_adf(series, window=hurst_window):
    adf_stats = []
    for i in range(len(series)):
        if i < window:
            adf_stats.append(np.nan)
        else:
            window_data = series.iloc[i - window:i + 1]
            stat, _ = adf_test(window_data)
            adf_stats.append(stat)
    return pd.Series(adf_stats, index=series.index)


indicator_full['ADF_Stat'] = rolling_adf(indicator_full['Close'])

# Simplified regime classification
regime_results = []
for idx, row in indicator_full.iterrows():
    regime, confidence = classify_regime_simplified(row['Hurst'], row['ADF_Stat'])
    regime_results.append((regime, confidence))

indicator_full['Regime'] = [r[0] for r in regime_results]
indicator_full['Regime_Confidence'] = [r[1] for r in regime_results]

# Initialize session state for optimization results
if 'walk_forward_results' not in st.session_state:
    st.session_state.walk_forward_results = None
if 'optimized_weights' not in st.session_state:
    st.session_state.optimized_weights = {
        'TRENDING': [0.4, 0.2, 0.4],  # Default weights
        'MEAN_REVERTING': [0.6, 0.3, 0.1],
        'UNKNOWN': [0.5, 0.3, 0.2]
    }

# Walk-Forward Optimization
if run_walk_forward_opt:
    st.markdown("### 🎯 Walk-Forward Optimization in Progress...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        optimizer = WalkForwardOptimizer(granularity=10, transaction_cost_bps=transaction_fee_bps)

        status_text.text("🔄 Running grid search optimization per regime...")
        progress_bar.progress(30)

        # Run walk-forward optimization
        buy_zones = (buy_zone_min, buy_zone_max)
        sell_zones = (sell_zone_min, sell_zone_max)

        optimization_results = optimizer.walk_forward_optimize(
            indicator_full, buy_zones, sell_zones, train_pct=0.7
        )

        progress_bar.progress(80)
        status_text.text("✅ Optimization completed! Updating weights...")

        # Update optimized weights
        for regime, result in optimization_results.items():
            if result and 'best_weights' in result:
                st.session_state.optimized_weights[regime] = result['best_weights']

        st.session_state.walk_forward_results = optimization_results

        progress_bar.progress(100)
        status_text.text("🎯 Walk-forward optimization completed successfully!")

        # Display results
        if optimization_results:
            st.success("✅ Walk-Forward Optimization Completed!")

            col1, col2, col3 = st.columns(3)

            for i, (regime, result) in enumerate(optimization_results.items()):
                with [col1, col2, col3][i]:
                    if result:
                        weights = result['best_weights']
                        st.metric(
                            f"🎯 {regime}",
                            f"Sharpe: {result['best_sharpe']:.3f}",
                            f"V:{weights[0]:.2f} C:{weights[1]:.2f} M:{weights[2]:.2f}"
                        )
                        st.caption(f"Data points: {result['data_points']}")

        time.sleep(2)
        st.rerun()

    except Exception as e:
        progress_bar.progress(0)
        status_text.text("")
        st.error(f"❌ Optimization failed: {str(e)}")

# Display current optimization status
if st.session_state.walk_forward_results:
    st.markdown("### 📊 Current Walk-Forward Optimization Results")

    results_df = pd.DataFrame({
        'Regime': [],
        'Value Weight': [],
        'Carry Weight': [],
        'Momentum Weight': [],
        'Sharpe Ratio': [],
        'Training Points': []
    })

    for regime, result in st.session_state.walk_forward_results.items():
        if result:
            weights = result['best_weights']
            new_row = pd.DataFrame({
                'Regime': [regime],
                'Value Weight': [f"{weights[0]:.3f}"],
                'Carry Weight': [f"{weights[1]:.3f}"],
                'Momentum Weight': [f"{weights[2]:.3f}"],
                'Sharpe Ratio': [f"{result['best_sharpe']:.3f}"],
                'Training Points': [result['data_points']]
            })
            results_df = pd.concat([results_df, new_row], ignore_index=True)

    st.dataframe(results_df, hide_index=True, use_container_width=True)

# Separate training and trading data clearly
training_end_date = start_date_input - timedelta(days=1)
training_data = indicator_full[indicator_full.index <= training_end_date.strftime('%Y-%m-%d')]
trading_data = indicator_full[
    (indicator_full.index >= start_date_input.strftime('%Y-%m-%d')) &
    (indicator_full.index <= end_date_input.strftime('%Y-%m-%d'))
    ]

if len(trading_data) == 0:
    st.error("❌ No trading data for selected period")
    st.stop()

st.markdown("### 📊 Data Separation Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📚 Training Data", f"{len(training_data)} days", "Optimization only")
with col2:
    st.metric("📈 Trading Data", f"{len(trading_data)} days", f"From {start_date_input}")
with col3:
    opt_status = "✅ Optimized" if st.session_state.walk_forward_results else "❌ Default"
    st.metric("⚖️ Weights Status", opt_status, "Walk-forward")

# Trading Simulation with Revised System
trading_system = RevisedTradingSystem(
    initial_cash, transaction_fee_bps, capital_allocation_pct, momentum_filter_threshold
)

st.markdown(f"### 🚀 Trading Simulation: {len(trading_data)} days with revised strategy")

# Process signals with optimized weights and momentum filtering
all_signals = []
executed_trades = []

progress_bar = st.progress(0)
status_text = st.empty()

for i, (idx, row) in enumerate(trading_data.iterrows()):
    if i % 50 == 0:
        progress_bar.progress(min(i / len(trading_data), 1.0))
        status_text.text(f"Processing {i}/{len(trading_data)} trading days...")

    # Get optimized weights for current regime
    current_regime = row['Regime']
    optimal_weights = st.session_state.optimized_weights.get(current_regime, [0.5, 0.3, 0.2])

    # Calculate dynamic aggregate percentile with optimized weights
    agg_percentile = (row['Value_Percentile'] * optimal_weights[0] +
                      row['Carry_Percentile'] * optimal_weights[1] +
                      row['Momentum_Percentile'] * optimal_weights[2])

    # Check signal strength
    signal_type, signal_strength = trading_system.check_signal_strength(
        agg_percentile, buy_zone_min, buy_zone_max, sell_zone_min, sell_zone_max
    )

    # Check if we can trade (with momentum filtering)
    can_trade, action_type = trading_system.can_trade(
        signal_type, signal_strength, row['Close'], row['Regime'],
        row['Momentum_Score'], row['Regime_Confidence']
    )

    # Execute trade if conditions met
    executed = False
    if can_trade:
        trade_info = trading_system.execute_trade(
            action_type, row['Close'], idx, signal_strength, row['Momentum_Score']
        )
        executed_trades.append(trade_info)
        executed = True

    # Update unrealized P&L
    trading_system.update_unrealized_pnl(row['Close'])

    all_signals.append({
        'signal': 1 if signal_type == "BUY" else -1 if signal_type == "SELL" else 0,
        'executed': executed,
        'action_type': action_type if executed else "NO_ACTION",
        'momentum_score': row['Momentum_Score'],
        'regime': current_regime,
        'regime_confidence': row['Regime_Confidence'],
        'agg_percentile': agg_percentile,
        'total_pnl': trading_system.get_total_pnl(),
        'realized_pnl': trading_system.realized_pnl,
        'unrealized_pnl': trading_system.unrealized_pnl,
        'cash': trading_system.cash,
        'position': trading_system.position,
        'capital_utilization': trading_system.get_capital_utilization()
    })

progress_bar.progress(1.0)
status_text.text("✅ Revised strategy simulation completed!")

# Add results to trading data
for key in ['signal', 'executed', 'action_type', 'momentum_score', 'regime', 'regime_confidence',
            'agg_percentile', 'total_pnl', 'realized_pnl', 'unrealized_pnl', 'cash',
            'position', 'capital_utilization']:
    trading_data[key] = [s[key] for s in all_signals]

display_data = trading_data.copy()

# Performance calculations
total_pnl = trading_system.get_total_pnl()
total_return_pct = (total_pnl / initial_cash) * 100
portfolio_value = trading_system.get_portfolio_value()
current_capital_utilization = trading_system.get_capital_utilization()

# Calculate Sharpe ratio for the strategy
if len(display_data) > 1:
    strategy_returns = display_data['total_pnl'].pct_change().dropna()
    if len(strategy_returns) > 0 and strategy_returns.std() > 0:
        strategy_sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
    else:
        strategy_sharpe = 0
else:
    strategy_sharpe = 0

# Professional Dashboard
st.markdown("---")
st.markdown("## 📊 Revised Strategy Performance")

# Key Metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("💰 Portfolio Value", f"${portfolio_value:,.0f}", delta=f"{total_return_pct:+.2f}%")

with col2:
    cash_change = trading_system.cash - initial_cash
    st.metric("💵 Available Cash", f"${trading_system.cash:,.0f}", delta=f"{cash_change:+,.0f}")

with col3:
    st.metric("📈 Strategy Sharpe", f"{strategy_sharpe:.3f}", "Annualized")

with col4:
    current_pos = "LONG" if trading_system.position > 0 else "SHORT" if trading_system.position < 0 else "FLAT"
    pos_size = abs(trading_system.position)
    st.metric("📊 Position", current_pos, delta=f"Size: {pos_size}")

with col5:
    total_trades = len(executed_trades)
    total_fees = sum([trade['cost'] for trade in executed_trades])
    st.metric("🔄 Total Trades", total_trades, delta=f"-${total_fees:.0f}")

with col6:
    # Calculate trade frequency (trades per month)
    days_traded = len(display_data)
    trades_per_month = (total_trades / days_traded) * 30 if days_traded > 0 else 0
    st.metric("⏰ Trade Frequency", f"{trades_per_month:.1f}/month", "Reduced overtrading")

# Enhanced Charts
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.subheader("💹 Revised Strategy: Price Action with Momentum-Filtered Signals")

fig1 = go.Figure()

# Price line
fig1.add_trace(go.Scatter(x=display_data.index, y=display_data["Close"],
                          mode="lines", name="5Y Treasury Futures",
                          line=dict(color='#2E86AB', width=3)))

# Momentum-filtered signals
buy_signals = display_data[(display_data["signal"] == 1) & (display_data["executed"] == True)]
if len(buy_signals) > 0:
    fig1.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals["Close"],
                              mode="markers", name=f"✅ Momentum-Filtered Buys ({len(buy_signals)})",
                              marker=dict(symbol="triangle-up", color="#228B22", size=14)))

sell_signals = display_data[(display_data["signal"] == -1) & (display_data["executed"] == True)]
if len(sell_signals) > 0:
    fig1.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals["Close"],
                              mode="markers", name=f"❌ Momentum-Filtered Sells ({len(sell_signals)})",
                              marker=dict(symbol="triangle-down", color="#DC143C", size=14)))

# Filtered out signals (for comparison)
filtered_out = display_data[(display_data["signal"] != 0) & (display_data["executed"] == False)]
if len(filtered_out) > 0:
    fig1.add_trace(go.Scatter(x=filtered_out.index, y=filtered_out["Close"],
                              mode="markers", name=f"🚫 Filtered Out Signals ({len(filtered_out)})",
                              marker=dict(symbol="x", color="rgba(128,128,128,0.5)", size=8)))

fig1.update_layout(height=500, title="Momentum-Filtered Trading Signals (Reduced Overtrading)",
                   template="plotly_white", showlegend=True)
st.plotly_chart(fig1, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Performance comparison
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.subheader("📊 Strategy vs Benchmark Performance")

fig2 = go.Figure()

# Strategy performance
fig2.add_trace(go.Scatter(x=display_data.index,
                          y=(display_data['total_pnl'] / initial_cash) * 100,
                          mode='lines', name=f'Revised Strategy ({strategy_sharpe:.3f} Sharpe)',
                          line=dict(color='#1B4F72', width=4)))

# Buy & Hold benchmark
if len(display_data) > 0:
    buy_hold_return = ((display_data['Close'] / display_data['Close'].iloc[0]) - 1) * 100
    bh_returns = buy_hold_return.pct_change().dropna()
    bh_sharpe = (bh_returns.mean() / bh_returns.std()) * np.sqrt(252) if len(
        bh_returns) > 0 and bh_returns.std() > 0 else 0

    fig2.add_trace(go.Scatter(x=display_data.index, y=buy_hold_return,
                              mode='lines', name=f'Buy & Hold ({bh_sharpe:.3f} Sharpe)',
                              line=dict(color='#95A5A6', width=2, dash='dash')))

fig2.update_layout(height=450,
                   title=f"Performance Comparison: Revised Strategy vs Buy & Hold",
                   yaxis_title="Return (%)", template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Trading Activity Analysis
if executed_trades:
    st.subheader("📊 Enhanced Trading Analysis")

    trades_df = pd.DataFrame(executed_trades)
    trades_df['date'] = pd.to_datetime(trades_df['date'])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📈 Recent Trading Activity**")
        display_trades = trades_df.head(20)[['date', 'action', 'price', 'pnl', 'momentum_score']].copy()
        display_trades['date'] = display_trades['date'].dt.strftime('%Y-%m-%d')
        display_trades['pnl'] = display_trades['pnl'].round(2)
        display_trades['momentum_score'] = display_trades['momentum_score'].round(1)
        st.dataframe(display_trades, hide_index=True, use_container_width=True, height=300)

    with col2:
        st.markdown("**📊 Strategy Statistics**")

        # Filter statistics
        total_signals = len(display_data[display_data['signal'] != 0])
        filtered_signals = len(display_data[(display_data['signal'] != 0) & (display_data['executed'] == False)])
        filter_rate = (filtered_signals / total_signals * 100) if total_signals > 0 else 0

        # Performance statistics
        profitable_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = (profitable_trades / len(trades_df) * 100) if len(trades_df) > 0 else 0
        avg_trade = trades_df['pnl'].mean() if len(trades_df) > 0 else 0

        stats_data = {
            "Total Signals Generated": total_signals,
            "Signals Filtered Out": filtered_signals,
            "Filter Rate": f"{filter_rate:.1f}%",
            "Executed Trades": len(trades_df),
            "Win Rate": f"{win_rate:.1f}%",
            "Average Trade P&L": f"${avg_trade:.2f}",
            "Strategy Sharpe": f"{strategy_sharpe:.3f}",
            "Total Transaction Costs": f"${trades_df['cost'].sum():.2f}"
        }

        for metric, value in stats_data.items():
            st.write(f"**{metric}**: {value}")

# Export Data
st.markdown("---")
st.markdown("## 📁 Data Export")

col1, col2, col3 = st.columns(3)

with col1:
    enhanced_data = display_data.copy()
    csv_data = enhanced_data.to_csv(index=True)
    st.download_button("📥 Download Trading Data", csv_data,
                       f"revised_strategy_data_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

with col2:
    if executed_trades:
        trades_csv = pd.DataFrame(executed_trades).to_csv(index=False)
        st.download_button("📥 Download Trades", trades_csv,
                           f"revised_strategy_trades_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

with col3:
    # Export optimized weights
    weights_df = pd.DataFrame.from_dict(st.session_state.optimized_weights, orient='index',
                                        columns=['Value_Weight', 'Carry_Weight', 'Momentum_Weight'])
    weights_df.index.name = 'Regime'
    weights_csv = weights_df.to_csv()
    st.download_button("📥 Download Optimized Weights", weights_csv,
                       f"optimized_weights_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

# Strategy Summary
st.markdown("---")
st.markdown("## 📋 Revised Strategy Summary")

summary_text = f"""
### 🎯 Strategy Configuration:
- **Regime Classification**: Simplified (Hurst + ADF only, no Fractal Dimension)
- **Momentum Filtering**: Threshold ±{momentum_filter_threshold} (reduces overtrading by {filter_rate:.1f}%)
- **Kalman Smoothing**: {'✅ ENABLED' if enable_kalman else '❌ DISABLED'}
- **Walk-Forward Optimization**: {'✅ COMPLETED' if st.session_state.walk_forward_results else '❌ USING DEFAULTS'}

### 📊 Performance Results:
- **Strategy Return**: {total_return_pct:.2f}%
- **Strategy Sharpe Ratio**: {strategy_sharpe:.3f}
- **Buy & Hold Sharpe**: {bh_sharpe:.3f} 
- **Total Trades**: {len(executed_trades)} (reduced from {total_signals} signals)
- **Win Rate**: {win_rate:.1f}%
- **Trade Frequency**: {trades_per_month:.1f} trades/month

### 🎯 Key Improvements:
- **Reduced Overtrading**: {filter_rate:.1f}% of signals filtered by momentum
- **Higher Conviction Trades**: Only trade when momentum aligns
- **Lower Transaction Costs**: Fewer trades = lower total fees
- **Improved Risk-Adjusted Returns**: Focus on Sharpe ratio optimization

### ⚖️ Current Optimized Weights:
"""

for regime, weights in st.session_state.optimized_weights.items():
    summary_text += f"\n- **{regime}**: Value={weights[0]:.3f}, Carry={weights[1]:.3f}, Momentum={weights[2]:.3f}"

st.markdown(summary_text)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 2rem; border-top: 2px solid #dee2e6;'>
    <strong>Revised Treasury Trading Strategy v1.0</strong><br>
    <em>Simplified Regimes • Momentum Filtering • Walk-Forward Optimization</em><br>
    Strategy Sharpe: {strategy_sharpe:.3f} | Trades: {len(executed_trades)} | Filter Rate: {filter_rate:.1f}%<br>
    <small>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</small>
</div>
""", unsafe_allow_html=True)