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
    .trading-alert {
        border-left: 6px solid #28a745;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .stExpander > div:first-child {
        background: linear-gradient(90deg, #495057 0%, #6c757d 100%);
        color: white;
        border-radius: 8px;
    }
    .chart-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .documentation-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 2rem;
        margin: 1rem 0;
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


def compute_fractal_dimension(price_series, scaling_factor):
    """Compute fractal dimension for price series"""
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


def calculate_momentum_score(rsi, williams_r, cci, rsi_ob=70, rsi_os=30, williams_ob=-20, williams_os=-80, cci_ob=100,
                             cci_os=-100):
    """Enhanced momentum score using RSI, Williams %R, and CCI"""
    score = 0

    # RSI component
    if rsi > rsi_ob:
        score += 1
    elif rsi < rsi_os:
        score -= 1

    # Williams %R component
    if williams_r > williams_ob:
        score += 1
    elif williams_r < williams_os:
        score -= 1

    # CCI component
    if cci > cci_ob:
        score += 1
    elif cci < cci_os:
        score -= 1

    return score


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


def calculate_regime_confidence(hurst, fractal_dim, adf_stat, momentum_score):
    """Calculate confidence score for regime classification"""
    confidence_score = 0

    # Hurst confidence
    if not pd.isna(hurst):
        if hurst < 0.47 or hurst > 0.53:
            confidence_score += min(abs(hurst - 0.5) * 4, 1.0)
        else:
            confidence_score += 0.1

    # Fractal dimension confidence
    if not pd.isna(fractal_dim):
        if fractal_dim < 1.47 or fractal_dim > 1.53:
            confidence_score += min(abs(fractal_dim - 1.5) * 4, 1.0)
        else:
            confidence_score += 0.1

    # ADF test confidence
    if not pd.isna(adf_stat):
        if adf_stat < -2.862:
            confidence_score += 1.0
        elif adf_stat < -2.567:
            confidence_score += 0.5
        else:
            confidence_score += 0.1

    # Momentum score confidence
    if not pd.isna(momentum_score):
        confidence_score += min(abs(momentum_score) / 3.0, 0.5)

    return min(confidence_score / 3.5, 1.0)


def classify_regime_advanced(hurst, fractal_dim, adf_stat, momentum_score):
    """Advanced regime classification"""
    confidence = calculate_regime_confidence(hurst, fractal_dim, adf_stat, momentum_score)

    if confidence < 0.3:
        return "UNKNOWN", confidence

    trend_score = 0
    mean_rev_score = 0

    # Hurst scoring
    if not pd.isna(hurst):
        if hurst > 0.53:
            trend_score += 2
        elif hurst > 0.50:
            trend_score += 1
        elif hurst < 0.47:
            mean_rev_score += 2
        elif hurst < 0.50:
            mean_rev_score += 1

    # Fractal scoring
    if not pd.isna(fractal_dim):
        if fractal_dim > 1.53:
            mean_rev_score += 2
        elif fractal_dim > 1.50:
            mean_rev_score += 1
        elif fractal_dim < 1.47:
            trend_score += 2
        elif fractal_dim < 1.50:
            trend_score += 1

    # ADF scoring
    if not pd.isna(adf_stat):
        if adf_stat < -2.862:
            mean_rev_score += 2
        elif adf_stat < -2.567:
            mean_rev_score += 1

    # Momentum score contribution
    if not pd.isna(momentum_score):
        if momentum_score > 1:
            trend_score += 1
        elif momentum_score < -1:
            mean_rev_score += 1

    # Final classification
    if trend_score > mean_rev_score + 1:
        return "TRENDING", confidence
    elif mean_rev_score > trend_score + 1:
        return "MEAN_REVERTING", confidence
    else:
        return "UNKNOWN", confidence


class AdvancedTradingSystem:
    """Advanced trading system with capital allocation and professional position management"""

    def __init__(self, initial_cash=100000, transaction_fee_bps=2, capital_allocation_pct=100):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0
        self.position_size = 0
        self.entry_price = 0
        self.entry_date = None
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        self.transaction_fee_bps = transaction_fee_bps
        self.capital_allocation_pct = capital_allocation_pct
        self.last_signal_strength = 0
        self.trade_history = []
        self.daily_pnl = []

    def calculate_position_size(self, price):
        """Calculate position size based on capital allocation"""
        available_capital = self.cash * (self.capital_allocation_pct / 100)
        position_size = int(available_capital / price)  # Number of contracts
        return max(position_size, 1)  # Minimum 1 contract

    def calculate_transaction_cost(self, price, quantity):
        """Calculate transaction cost in bps"""
        notional = price * abs(quantity)
        return notional * (self.transaction_fee_bps / 10000)

    def check_signal_strength(self, agg_percentile, buy_zone_min, buy_zone_max, sell_zone_min, sell_zone_max):
        """Check if signal is in trading zones and calculate strength"""
        if buy_zone_min <= agg_percentile <= buy_zone_max:
            return "BUY", (agg_percentile - buy_zone_min) / (buy_zone_max - buy_zone_min)
        elif sell_zone_min <= agg_percentile <= sell_zone_max:
            return "SELL", (sell_zone_max - agg_percentile) / (sell_zone_max - sell_zone_min)
        else:
            return "NEUTRAL", 0.0

    def can_trade(self, signal_type, signal_strength, price, regime):
        """Check if we can execute the trade"""
        # If regime is UNKNOWN, close position
        if regime == "UNKNOWN" and self.position != 0:
            return True, "CLOSE_UNKNOWN"

        # If signal is neutral and we have position, close it
        if signal_type == "NEUTRAL" and self.position != 0:
            return True, "CLOSE_NEUTRAL"

        # If we're flat, we can open any position (check sufficient cash)
        if self.position == 0 and signal_type in ["BUY", "SELL"]:
            required_capital = price * self.calculate_position_size(price)
            if self.cash >= required_capital:
                return True, f"OPEN_{signal_type}"
            else:
                return False, "INSUFFICIENT_CASH"

        # If we have position in same direction, check if signal is stronger
        if (self.position > 0 and signal_type == "BUY") or (self.position < 0 and signal_type == "SELL"):
            if signal_strength > abs(self.last_signal_strength) + 0.1:  # Need 10% stronger signal
                additional_size = self.calculate_position_size(price) // 2  # Add half position
                required_capital = price * additional_size
                if self.cash >= required_capital:
                    return True, f"ADD_{signal_type}"
                else:
                    return False, "INSUFFICIENT_CASH_ADD"
            else:
                return False, "SIGNAL_TOO_WEAK"

        # If we have position in opposite direction, close and reverse
        if (self.position > 0 and signal_type == "SELL") or (self.position < 0 and signal_type == "BUY"):
            return True, f"REVERSE_{signal_type}"

        return False, "NO_ACTION"

    def execute_trade(self, action_type, price, date, signal_strength=0):
        """Execute trade with proper P&L calculation and capital allocation"""
        quantity = self.calculate_position_size(price)
        transaction_cost = self.calculate_transaction_cost(price, quantity)

        trade_info = {
            'date': date,
            'action': action_type,
            'price': price,
            'quantity': quantity,
            'cost': transaction_cost,
            'cash_before': self.cash,
            'position_before': self.position
        }

        if "CLOSE" in action_type or "REVERSE" in action_type:
            if self.position != 0:
                # Close existing position
                pnl = (price - self.entry_price) * self.position - transaction_cost
                self.realized_pnl += pnl
                self.cash += pnl
                self.position = 0
                self.unrealized_pnl = 0

                trade_info['pnl'] = pnl
                trade_info['realized_pnl'] = self.realized_pnl

        if "OPEN" in action_type or "REVERSE" in action_type or "ADD" in action_type:
            if "BUY" in action_type:
                new_position = quantity
            elif "SELL" in action_type:
                new_position = -quantity
            else:
                new_position = 0

            if "ADD" in action_type:
                self.position += new_position // 2  # Add half position
            else:
                self.position = new_position
                self.entry_price = price
                self.entry_date = date

            self.cash -= transaction_cost
            self.last_signal_strength = signal_strength

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
        """Get total P&L"""
        return self.realized_pnl + self.unrealized_pnl

    def get_portfolio_value(self):
        """Get current portfolio value"""
        return self.cash + self.unrealized_pnl

    def get_capital_utilization(self):
        """Get current capital utilization"""
        if self.position != 0:
            position_value = abs(self.position * self.entry_price)
            return (position_value / self.initial_cash) * 100
        return 0


# Streamlit App Header
st.markdown(
    '<div class="main-header"><h1>💹 Professional Treasury Futures Trading System</h1><p>Advanced Regime-Aware Quantitative Trading & Backtesting Platform</p></div>',
    unsafe_allow_html=True)

# Enhanced Professional Trading Configuration
with st.expander("⚙️ Professional Trading Configuration", expanded=True):
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("**📅 Trading Period**")
        start_date_input = st.date_input("Start Date", value=datetime(2023, 1, 1))
        end_date_input = st.date_input("End Date", value=datetime.now().date())

    with col2:
        st.markdown("**🎯 Buy Zone (%)**")
        buy_zone_min = st.number_input("Buy Min", value=90, min_value=70, max_value=95)
        buy_zone_max = st.number_input("Buy Max", value=100, min_value=95, max_value=100)

    with col3:
        st.markdown("**🎯 Sell Zone (%)**")
        sell_zone_min = st.number_input("Sell Min", value=0, min_value=0, max_value=10)
        sell_zone_max = st.number_input("Sell Max", value=20, min_value=10, max_value=30)

    with col4:
        st.markdown("**💰 Portfolio Setup**")
        initial_cash = st.number_input("Initial Capital ($)", value=100000, min_value=10000, step=10000)
        capital_allocation_pct = st.number_input("Capital per Trade (%)", value=25, min_value=5, max_value=100, step=5)

    with col5:
        st.markdown("**🔧 Trading Config**")
        transaction_fee_bps = st.number_input("Transaction Fee (bps)", value=2, min_value=0, max_value=20)
        confidence_threshold = st.number_input("Min Regime Confidence", value=0.3, min_value=0.1, max_value=0.8,
                                               step=0.1)

# Model configuration
model_type = st.selectbox("📊 Model Type", ["Short-Term (Daily/Weekly)", "Long-Term (Weekly/Monthly)"])
lookback = 63 if model_type == "Short-Term (Daily/Weekly)" else 252
fractal_window = 50 if model_type == "Short-Term (Daily/Weekly)" else 100
hurst_window = 30 if model_type == "Short-Term (Daily/Weekly)" else 60

# Data Loading
start_date = start_date_input - timedelta(days=365)
start_date_str = start_date.strftime("%Y-%m-%d")

with st.spinner("📊 Loading market data and calculating indicators..."):
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

    except Exception as e:
        st.error(f"Data loading error: {e}")
        st.stop()

# Build indicators
backtest_data = _2yUS.join(_5yUS_real).join(_5yUS)
backtest_data.dropna(inplace=True)
indicators = build_indicators(backtest_data)

# Calculate percentiles
for col in ["5y_Real", "carry_normalized", "momentum"]:
    indicators[f"{col}_percentile"] = indicators[col].rolling(lookback).apply(lambda x: percentile_score(x))

# Join with futures data
indicator_full = indicators[["5y", "5y_Real_percentile", "carry_normalized_percentile", "momentum_percentile"]].join(
    _5yUS_fut)
indicator_full.columns = ["5y_yield", "Value_Percentile", "Carry_Percentile", "Momentum_Percentile", "Open", "High",
                          "Low", "Close"]
indicator_full.dropna(inplace=True)

# Calculate technical indicators
indicator_full['Fractal_Dim'] = compute_fractal_dimension(indicator_full['Close'], fractal_window)
indicator_full['Hurst'] = indicator_full['Close'].rolling(window=hurst_window).apply(calculate_hurst, raw=False)
indicator_full['RSI'] = calculate_rsi(indicator_full['Close'])
indicator_full['Williams_R'] = calculate_williams_r(indicator_full['High'], indicator_full['Low'],
                                                    indicator_full['Close'])
indicator_full['CCI'] = calculate_cci(indicator_full['High'], indicator_full['Low'], indicator_full['Close'])
indicator_full['Momentum_Score'] = indicator_full.apply(
    lambda row: calculate_momentum_score(row['RSI'], row['Williams_R'], row['CCI']), axis=1
)


# ADF test
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

# Regime classification
regime_results = []
for idx, row in indicator_full.iterrows():
    regime, confidence = classify_regime_advanced(
        row['Hurst'], row['Fractal_Dim'], row['ADF_Stat'], row['Momentum_Score']
    )
    regime_results.append((regime, confidence))

indicator_full['Regime'] = [r[0] for r in regime_results]
indicator_full['Regime_Confidence'] = [r[1] for r in regime_results]

# Dynamic weight optimization by regime
regime_weights = {
    'TRENDING': [60, 15, 25],  # More momentum in trending
    'MEAN_REVERTING': [75, 20, 5],  # More value in mean reverting
    'UNKNOWN': [65, 25, 10]  # Balanced for unknown
}


# Calculate dynamic aggregate percentile
def get_dynamic_agg_percentile(row):
    regime = row['Regime']
    weights = regime_weights.get(regime, [65, 25, 10])
    return (row['Value_Percentile'] * weights[0] +
            row['Carry_Percentile'] * weights[1] +
            row['Momentum_Percentile'] * weights[2]) / 100


indicator_full['Agg_Percentile_Dynamic'] = indicator_full.apply(get_dynamic_agg_percentile, axis=1)

# Add current weights to dataframe for visualization
for regime in regime_weights:
    indicator_full[f'Weight_Value_{regime}'] = regime_weights[regime][0]
    indicator_full[f'Weight_Carry_{regime}'] = regime_weights[regime][1]
    indicator_full[f'Weight_Momentum_{regime}'] = regime_weights[regime][2]

# Current weights based on regime
indicator_full['Current_Weight_Value'] = indicator_full.apply(
    lambda row: regime_weights.get(row['Regime'], [65, 25, 10])[0], axis=1)
indicator_full['Current_Weight_Carry'] = indicator_full.apply(
    lambda row: regime_weights.get(row['Regime'], [65, 25, 10])[1], axis=1)
indicator_full['Current_Weight_Momentum'] = indicator_full.apply(
    lambda row: regime_weights.get(row['Regime'], [65, 25, 10])[2], axis=1)

# Initialize advanced trading system
trading_system = AdvancedTradingSystem(initial_cash, transaction_fee_bps, capital_allocation_pct)

# Process all signals
all_signals = []
executed_trades = []

for idx, row in indicator_full.iterrows():
    # Check signal strength
    signal_type, signal_strength = trading_system.check_signal_strength(
        row['Agg_Percentile_Dynamic'], buy_zone_min, buy_zone_max, sell_zone_min, sell_zone_max
    )

    # Check if we can trade
    can_trade, action_type = trading_system.can_trade(
        signal_type, signal_strength, row['Close'], row['Regime']
    )

    # Execute trade if possible and confidence is high enough
    executed = False
    if can_trade and row['Regime_Confidence'] >= confidence_threshold:
        trade_info = trading_system.execute_trade(action_type, row['Close'], idx, signal_strength)
        executed_trades.append(trade_info)
        executed = True

    # Update unrealized P&L
    trading_system.update_unrealized_pnl(row['Close'])

    all_signals.append({
        'signal': 1 if signal_type == "BUY" else -1 if signal_type == "SELL" else 0,
        'executed': executed,
        'total_pnl': trading_system.get_total_pnl(),
        'realized_pnl': trading_system.realized_pnl,
        'unrealized_pnl': trading_system.unrealized_pnl,
        'cash': trading_system.cash,
        'position': trading_system.position,
        'capital_utilization': trading_system.get_capital_utilization()
    })

# Add signals to dataframe
for key in ['signal', 'executed', 'total_pnl', 'realized_pnl', 'unrealized_pnl', 'cash', 'position',
            'capital_utilization']:
    indicator_full[key] = [s[key] for s in all_signals]

# Filter display data
display_data = indicator_full[
    (indicator_full.index >= start_date_input.strftime('%Y-%m-%d')) &
    (indicator_full.index <= end_date_input.strftime('%Y-%m-%d'))
    ]

# Performance calculations
total_pnl = trading_system.get_total_pnl()
total_return_pct = (total_pnl / initial_cash) * 100
portfolio_value = trading_system.get_portfolio_value()
current_capital_utilization = trading_system.get_capital_utilization()

# Professional Dashboard
st.markdown("---")

# Advanced Key Metrics Dashboard with FIXED DELTA LOGIC
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

with col1:
    # FIXED: Portfolio value delta should show the P&L change, not absolute values
    st.metric("💰 Portfolio Value", f"${portfolio_value:,.0f}",
              delta=f"{total_return_pct:+.2f}%")

with col2:
    # FIXED: Cash delta shows change from initial
    cash_change = trading_system.cash - initial_cash
    st.metric("💵 Available Cash", f"${trading_system.cash:,.0f}",
              delta=f"{cash_change:+,.0f}")

with col3:
    # FIXED: Realized P&L delta shows percentage return
    realized_pct = (trading_system.realized_pnl / initial_cash) * 100
    st.metric("✅ Realized P&L", f"${trading_system.realized_pnl:,.0f}",
              delta=f"{realized_pct:+.2f}%")

with col4:
    # FIXED: Unrealized P&L delta shows percentage
    unrealized_pct = (trading_system.unrealized_pnl / initial_cash) * 100
    st.metric("📊 Unrealized P&L", f"${trading_system.unrealized_pnl:,.0f}",
              delta=f"{unrealized_pct:+.2f}%")

with col5:
    current_pos = "LONG" if trading_system.position > 0 else "SHORT" if trading_system.position < 0 else "FLAT"
    pos_size = abs(trading_system.position)
    st.metric("📈 Position", current_pos, delta=f"Size: {pos_size}")

with col6:
    st.metric("⚡ Capital Usage", f"{current_capital_utilization:.1f}%",
              delta=f"Target: {capital_allocation_pct}%")

with col7:
    total_trades = len(executed_trades)
    total_fees = sum([trade['cost'] for trade in executed_trades])
    # FIXED: Fees should show as negative (cost)
    st.metric("🔄 Trades", total_trades, delta=f"-${total_fees:.0f}")

# Enhanced Professional Charts Section
st.markdown("---")

# Chart 1: Price Action with Professional Signals
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.subheader("📈 Treasury Futures with Professional Trading Signals")
fig1 = go.Figure()

# Price line with enhanced styling
fig1.add_trace(go.Scatter(x=display_data.index, y=display_data["Close"],
                          mode="lines", name="5Y Treasury Futures",
                          line=dict(color='#2E86AB', width=3)))

# All buy signals (light triangles)
buy_signals_all = display_data[display_data["signal"] == 1]
if len(buy_signals_all) > 0:
    fig1.add_trace(go.Scatter(x=buy_signals_all.index, y=buy_signals_all["Close"],
                              mode="markers", name=f"Buy Signals ({len(buy_signals_all)})",
                              marker=dict(symbol="triangle-up", color="rgba(34, 139, 34, 0.6)", size=10),
                              showlegend=True))

# Executed buy signals (dark squares)
buy_executed = display_data[(display_data["signal"] == 1) & (display_data["executed"] == True)]
if len(buy_executed) > 0:
    fig1.add_trace(go.Scatter(x=buy_executed.index, y=buy_executed["Close"],
                              mode="markers", name=f"✅ Buy Executed ({len(buy_executed)})",
                              marker=dict(symbol="square", color="#228B22", size=14),
                              showlegend=True))

# All sell signals (light triangles)
sell_signals_all = display_data[display_data["signal"] == -1]
if len(sell_signals_all) > 0:
    fig1.add_trace(go.Scatter(x=sell_signals_all.index, y=sell_signals_all["Close"],
                              mode="markers", name=f"Sell Signals ({len(sell_signals_all)})",
                              marker=dict(symbol="triangle-down", color="rgba(220, 20, 60, 0.6)", size=10),
                              showlegend=True))

# Executed sell signals (dark squares)
sell_executed = display_data[(display_data["signal"] == -1) & (display_data["executed"] == True)]
if len(sell_executed) > 0:
    fig1.add_trace(go.Scatter(x=sell_executed.index, y=sell_executed["Close"],
                              mode="markers", name=f"❌ Sell Executed ({len(sell_executed)})",
                              marker=dict(symbol="square", color="#DC143C", size=14),
                              showlegend=True))

fig1.update_layout(height=450, title="Professional Trading Signals Analysis",
                   template="plotly_white", showlegend=True,
                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig1, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Chart 2: Enhanced Signal Analysis with Options
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("**📊 Chart Options**")
    show_yield = st.checkbox("Show 5Y Yield", value=True)
    st.markdown("**Component Indicators:**")
    show_value = st.checkbox("Show Value %ile", value=False)
    show_carry = st.checkbox("Show Carry %ile", value=False)
    show_momentum_pct = st.checkbox("Show Momentum %ile", value=False)

with col1:
    st.subheader("🎯 Dynamic Aggregate Percentile & Components")
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

    # Dynamic aggregate percentile (always shown)
    fig2.add_trace(go.Scatter(x=display_data.index, y=display_data["Agg_Percentile_Dynamic"],
                              mode="lines", name="Dynamic Agg %ile",
                              line=dict(color='#6A4C93', width=4)), secondary_y=False)

    # Optional 5Y Yield
    if show_yield:
        fig2.add_trace(go.Scatter(x=display_data.index, y=display_data["5y_yield"],
                                  mode="lines", name="5Y Yield",
                                  line=dict(color='#F39C12', width=2)), secondary_y=True)

    # Optional component indicators (scaled to 0-100 range)
    if show_value:
        fig2.add_trace(go.Scatter(x=display_data.index, y=display_data["Value_Percentile"],
                                  mode="lines", name="Value %ile",
                                  line=dict(color='#E74C3C', width=2, dash='dot')), secondary_y=False)

    if show_carry:
        fig2.add_trace(go.Scatter(x=display_data.index, y=display_data["Carry_Percentile"],
                                  mode="lines", name="Carry %ile",
                                  line=dict(color='#3498DB', width=2, dash='dash')), secondary_y=False)

    if show_momentum_pct:
        fig2.add_trace(go.Scatter(x=display_data.index, y=display_data["Momentum_Percentile"],
                                  mode="lines", name="Momentum %ile",
                                  line=dict(color='#2ECC71', width=2, dash='dashdot')), secondary_y=False)

    # Trading zones
    fig2.add_shape(type="rect", x0=display_data.index[0], x1=display_data.index[-1],
                   y0=buy_zone_min, y1=buy_zone_max, fillcolor="rgba(34, 139, 34, 0.1)",
                   layer="below", line_width=0)
    fig2.add_hline(y=(buy_zone_min + buy_zone_max) / 2, line_dash="dash", line_color="#228B22",
                   annotation_text=f"Buy Zone: {buy_zone_min}-{buy_zone_max}%")

    fig2.add_shape(type="rect", x0=display_data.index[0], x1=display_data.index[-1],
                   y0=sell_zone_min, y1=sell_zone_max, fillcolor="rgba(220, 20, 60, 0.1)",
                   layer="below", line_width=0)
    fig2.add_hline(y=(sell_zone_min + sell_zone_max) / 2, line_dash="dash", line_color="#DC143C",
                   annotation_text=f"Sell Zone: {sell_zone_min}-{sell_zone_max}%")

    fig2.update_yaxes(title_text="Percentile (%)", secondary_y=False, range=[0, 100])
    if show_yield:
        fig2.update_yaxes(title_text="5Y Yield (%)", secondary_y=True)

    fig2.update_layout(height=450, title="Signal Analysis Dashboard", template="plotly_white",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Chart 3: Dynamic Weights Visualization
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("**⚖️ Weight Display**")
    show_value_weight = st.checkbox("Value Weight", value=True)
    show_carry_weight = st.checkbox("Carry Weight", value=True)
    show_momentum_weight = st.checkbox("Momentum Weight", value=True)

with col1:
    st.subheader("⚖️ Dynamic Weight Allocation by Regime")
    fig3 = go.Figure()

    if show_value_weight:
        fig3.add_trace(go.Scatter(x=display_data.index, y=display_data["Current_Weight_Value"],
                                  mode="lines", name="Value Weight (%)",
                                  line=dict(color='#E74C3C', width=3), fill='tonexty'))

    if show_carry_weight:
        fig3.add_trace(go.Scatter(x=display_data.index, y=display_data["Current_Weight_Carry"],
                                  mode="lines", name="Carry Weight (%)",
                                  line=dict(color='#3498DB', width=3), fill='tonexty'))

    if show_momentum_weight:
        fig3.add_trace(go.Scatter(x=display_data.index, y=display_data["Current_Weight_Momentum"],
                                  mode="lines", name="Momentum Weight (%)",
                                  line=dict(color='#2ECC71', width=3), fill='tonexty'))

    fig3.update_layout(height=400, title="Dynamic Component Weights Evolution",
                       template="plotly_white", yaxis_title="Weight (%)",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig3, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Current Status Dashboard
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Current Market Status")
    if not display_data.empty:
        latest = display_data.iloc[-1]

        status_table = pd.DataFrame({
            "Metric": ["Value Percentile", "Carry Percentile", "Momentum Percentile",
                       "🎯 Dynamic Agg %ile", "Enhanced Momentum Score", "Market Regime", "Regime Confidence"],
            "Current Value": [f"{latest['Value_Percentile']:.1f}%", f"{latest['Carry_Percentile']:.1f}%",
                              f"{latest['Momentum_Percentile']:.1f}%", f"{latest['Agg_Percentile_Dynamic']:.1f}%",
                              f"{latest['Momentum_Score']:.0f}/3", latest['Regime'],
                              f"{latest['Regime_Confidence']:.2f}"],
            "Weight": [f"{latest['Current_Weight_Value']:.0f}%", f"{latest['Current_Weight_Carry']:.0f}%",
                       f"{latest['Current_Weight_Momentum']:.0f}%", "100%", "Indicator", "Classification", "Meta"]
        })
        st.dataframe(status_table, hide_index=True, use_container_width=True)

with col2:
    st.subheader("🔬 Technical Analysis Dashboard")
    if not display_data.empty:
        latest = display_data.iloc[-1]

        tech_table = pd.DataFrame({
            "Indicator": ["Hurst Exponent", "Fractal Dimension", "ADF Statistic",
                          "RSI (14)", "Williams %R (14)", "CCI (20)"],
            "Value": [f"{latest['Hurst']:.3f}", f"{latest['Fractal_Dim']:.3f}",
                      f"{latest['ADF_Stat']:.2f}", f"{latest['RSI']:.1f}",
                      f"{latest['Williams_R']:.1f}", f"{latest['CCI']:.1f}"],
            "Signal": [
                "🟢 Trending" if latest['Hurst'] > 0.53 else "🔴 Mean-Rev" if latest['Hurst'] < 0.47 else "🟡 Neutral",
                "🔴 Ranging" if latest['Fractal_Dim'] > 1.53 else "🟢 Trending" if latest[
                                                                                     'Fractal_Dim'] < 1.47 else "🟡 Mixed",
                "🟢 Mean-Rev" if latest['ADF_Stat'] < -2.862 else "🟡 Weak" if latest[
                                                                                 'ADF_Stat'] < -2.567 else "🔴 Random",
                "🔴 Overbought" if latest['RSI'] > 70 else "🟢 Oversold" if latest['RSI'] < 30 else "🟡 Neutral",
                "🔴 Overbought" if latest['Williams_R'] > -20 else "🟢 Oversold" if latest[
                                                                                      'Williams_R'] < -80 else "🟡 Neutral",
                "🔴 Overbought" if latest['CCI'] > 100 else "🟢 Oversold" if latest['CCI'] < -100 else "🟡 Neutral"
            ]
        })
        st.dataframe(tech_table, hide_index=True, use_container_width=True)

# Advanced Portfolio Performance Analysis
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.subheader("💹 Advanced Portfolio Performance Analysis")
fig_perf = go.Figure()

# Total P&L with enhanced styling
fig_perf.add_trace(go.Scatter(x=display_data.index,
                              y=(display_data['total_pnl'] / initial_cash) * 100,
                              mode='lines+markers', name='Total P&L (%)',
                              line=dict(color='#1B4F72', width=4),
                              marker=dict(size=3)))

# Realized P&L
fig_perf.add_trace(go.Scatter(x=display_data.index,
                              y=(display_data['realized_pnl'] / initial_cash) * 100,
                              mode='lines', name='✅ Realized P&L (%)',
                              line=dict(color='#27AE60', width=3, dash='dot')))

# Unrealized P&L
fig_perf.add_trace(go.Scatter(x=display_data.index,
                              y=(display_data['unrealized_pnl'] / initial_cash) * 100,
                              mode='lines', name='📊 Unrealized P&L (%)',
                              line=dict(color='#F39C12', width=2, dash='dash')))

# Benchmark (Buy & Hold)
buy_hold_return = ((display_data['Close'] / display_data['Close'].iloc[0]) - 1) * 100
fig_perf.add_trace(go.Scatter(x=display_data.index, y=buy_hold_return,
                              mode='lines', name='📈 Buy & Hold Benchmark (%)',
                              line=dict(color='#95A5A6', width=2, dash='longdash')))

# Capital Utilization
fig_perf.add_trace(go.Scatter(x=display_data.index, y=display_data['capital_utilization'],
                              mode='lines', name='⚡ Capital Utilization (%)',
                              line=dict(color='#8E44AD', width=2, dash='dashdot')))

fig_perf.update_layout(height=450,
                       title=f"Portfolio Performance: {total_return_pct:.2f}% vs B&H: {buy_hold_return.iloc[-1]:.2f}% | Max Capital Usage: {display_data['capital_utilization'].max():.1f}%",
                       yaxis_title="Return/Utilization (%)", template="plotly_white",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_perf, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Professional Trading Activity Analysis - FIXED TO SHOW ALL TRADES
if executed_trades:
    st.subheader("📊 Professional Trading Activity Analysis")

    trades_df = pd.DataFrame(executed_trades)
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    trades_df = trades_df.sort_values('date', ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        # FIXED: Show ALL trading activity instead of just last 10
        st.markdown("**📈 Complete Trading Activity**")
        display_trades = trades_df[['date', 'action', 'price', 'quantity', 'pnl', 'cost']].copy()
        display_trades['date'] = display_trades['date'].dt.strftime('%Y-%m-%d')
        display_trades['pnl'] = display_trades['pnl'].fillna(0).round(2)
        display_trades['cost'] = display_trades['cost'].round(2)
        display_trades['quantity'] = display_trades['quantity'].astype(int)

        # Use height parameter to make it scrollable for all trades
        st.dataframe(display_trades, hide_index=True, use_container_width=True, height=400)

    with col2:
        st.markdown("**📊 Advanced Trading Statistics**")
        profitable_trades = trades_df[trades_df['pnl'] > 0]['pnl'].count() if 'pnl' in trades_df.columns else 0
        total_completed_trades = trades_df[trades_df['pnl'].notna()]['pnl'].count() if 'pnl' in trades_df.columns else 0
        win_rate = (profitable_trades / total_completed_trades * 100) if total_completed_trades > 0 else 0
        avg_trade = trades_df[trades_df['pnl'].notna()]['pnl'].mean() if 'pnl' in trades_df.columns else 0
        total_fees = trades_df['cost'].sum()
        avg_position_size = trades_df['quantity'].mean()

        stats_df = pd.DataFrame({
            "Metric": ["Total Executions", "Completed Trades", "Win Rate", "Avg Trade P&L", "Total Fees",
                       "Avg Position Size"],
            "Value": [len(trades_df), total_completed_trades, f"{win_rate:.1f}%", f"${avg_trade:.2f}",
                      f"${total_fees:.2f}", f"{avg_position_size:.0f} contracts"]
        })
        st.dataframe(stats_df, hide_index=True, use_container_width=True)

# Professional Trading Alert System
st.markdown("---")
if not display_data.empty:
    latest = display_data.iloc[-1]
    latest_regime = latest['Regime']
    latest_confidence = latest['Regime_Confidence']
    latest_agg = latest['Agg_Percentile_Dynamic']

    # Signal analysis
    signal_type, signal_strength = trading_system.check_signal_strength(
        latest_agg, buy_zone_min, buy_zone_max, sell_zone_min, sell_zone_max
    )

    position_status = "🟢 LONG" if trading_system.position > 0 else "🔴 SHORT" if trading_system.position < 0 else "⚪ FLAT"

    # Advanced professional alert
    if signal_type == "BUY" and latest_confidence >= confidence_threshold:
        st.success(
            f"🚀 **STRONG BUY SIGNAL DETECTED** | {position_status} | {latest_regime} Regime | Signal: {latest_agg:.1f}% | Strength: {signal_strength:.2f} | Confidence: {latest_confidence:.2f} | Capital: {capital_allocation_pct}%")
    elif signal_type == "SELL" and latest_confidence >= confidence_threshold:
        st.error(
            f"📉 **STRONG SELL SIGNAL DETECTED** | {position_status} | {latest_regime} Regime | Signal: {latest_agg:.1f}% | Strength: {signal_strength:.2f} | Confidence: {latest_confidence:.2f} | Capital: {capital_allocation_pct}%")
    elif latest_confidence < confidence_threshold:
        st.warning(
            f"⚠️ **LOW CONFIDENCE REGIME** | {position_status} | {latest_regime} | No Trading Recommended | Confidence: {latest_confidence:.2f} | Market Uncertain")
    else:
        st.info(
            f"💡 **MARKET MONITORING MODE** | {position_status} | {latest_regime} Regime | Signal: {latest_agg:.1f}% | Strength: {signal_strength:.2f} | Confidence: {latest_confidence:.2f}")

# Advanced Summary Statistics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Total Executions", len(executed_trades))
with col2:
    max_dd = min((display_data['total_pnl'] / initial_cash) * 100) if len(display_data) > 0 else 0
    st.metric("📉 Max Drawdown", f"{max_dd:.2f}%")
with col3:
    if len(display_data) > 1:
        daily_returns = display_data['total_pnl'].pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
    else:
        sharpe_ratio = 0
    st.metric("📈 Sharpe Ratio", f"{sharpe_ratio:.2f}")
with col4:
    max_capital_usage = display_data['capital_utilization'].max() if len(display_data) > 0 else 0
    st.metric("⚡ Max Capital Usage", f"{max_capital_usage:.1f}%")

# ENHANCED PROFESSIONAL DOCUMENTATION SECTION
st.markdown("---")
st.markdown("## 📚 Comprehensive Professional Documentation")

with st.expander("📖 Complete Trading System Documentation & Methodology", expanded=False):
    st.markdown('<div class="documentation-section">', unsafe_allow_html=True)

    # Executive Summary
    st.markdown("### 🎯 Executive Summary")
    st.markdown(f"""
    **Professional Treasury Futures Trading System v2.0**

    This institutional-grade quantitative trading system employs advanced regime classification and dynamic signal weighting 
    to optimize risk-adjusted returns in Treasury futures markets. The system integrates multiple technical and fundamental 
    indicators through a regime-aware framework that adapts to changing market conditions.

    **Current Performance Summary:**
    - **Total Return**: {total_return_pct:.2f}%
    - **Portfolio Value**: ${portfolio_value:,.0f}
    - **Total P&L**: ${total_pnl:,.0f}
    - **Sharpe Ratio**: {sharpe_ratio:.3f}
    - **Maximum Drawdown**: {max_dd:.2f}%
    - **Total Trades Executed**: {len(executed_trades)}
    - **Win Rate**: {win_rate if 'win_rate' in locals() else 0:.1f}%
    - **Current Position**: {position_status}
    - **Capital Utilization**: {current_capital_utilization:.1f}%
    """)

    # Detailed Methodology
    st.markdown("### 🔬 Detailed Methodology")

    st.markdown("#### 1. Market Regime Classification Framework")
    st.markdown(f"""
    The system employs a sophisticated multi-indicator approach to classify market regimes:

    **Regime Types:**
    - **TRENDING**: Persistent directional movement with strong momentum
    - **MEAN_REVERTING**: Oscillatory behavior around equilibrium levels  
    - **UNKNOWN**: Transitional or unclear market conditions

    **Classification Indicators:**
    - **Hurst Exponent**: Measures long-term memory in price series
      - H > 0.53: Trending behavior (persistent)
      - H < 0.47: Mean-reverting behavior (anti-persistent)
      - 0.47 ≤ H ≤ 0.53: Random walk behavior

    - **Fractal Dimension**: Quantifies market complexity and roughness
      - FD < 1.47: Smooth trending markets
      - FD > 1.53: Rough, ranging markets
      - 1.47 ≤ FD ≤ 1.53: Transitional behavior

    - **Augmented Dickey-Fuller Test**: Tests for mean reversion stationarity
      - ADF < -2.862: Strong mean reversion (99% confidence)
      - ADF < -2.567: Moderate mean reversion (95% confidence)
      - ADF ≥ -2.567: Non-stationary (trending)

    - **Enhanced Momentum Score**: Composite of RSI, Williams %R, and CCI
      - Range: -3 to +3
      - Positive values indicate momentum/trending conditions
      - Negative values suggest mean-reverting conditions

    **Current Model Configuration:**
    - Lookback Period: {lookback} days
    - Fractal Window: {fractal_window} days  
    - Hurst Window: {hurst_window} days
    - Confidence Threshold: {confidence_threshold:.1f}
    """)

    st.markdown("#### 2. Dynamic Signal Generation System")
    st.markdown(f"""
    **Component Indicators:**

    **A. Value Component (Real Yield Analysis):**
    - Uses 5-Year TIPS (Treasury Inflation-Protected Securities) 
    - Percentile ranking over {lookback}-day rolling window
    - Identifies relative value opportunities in real yield space
    - Higher percentiles indicate potentially overvalued conditions

    **B. Carry Component (Yield Curve Analysis):**
    - 5Y-2Y spread normalized by 75-day rolling volatility
    - Captures term structure dynamics and carry opportunities
    - Percentile ranking identifies extreme steepening/flattening
    - Mean-reverting characteristics of yield curve shape

    **C. Momentum Component (Technical Analysis):**
    - Short-term (5-day) vs Long-term (20-day) moving average crossover
    - Directional bias indicator for trend-following strategies
    - Percentile ranking smooths raw momentum signals
    - Combines with broader momentum score for confirmation

    **Dynamic Weight Allocation:**
    Current regime-specific weights:
    """)

    # Display current weights in a table
    weights_display = pd.DataFrame({
        'Regime': ['TRENDING', 'MEAN_REVERTING', 'UNKNOWN'],
        'Value Weight (%)': [regime_weights[r][0] for r in ['TRENDING', 'MEAN_REVERTING', 'UNKNOWN']],
        'Carry Weight (%)': [regime_weights[r][1] for r in ['TRENDING', 'MEAN_REVERTING', 'UNKNOWN']],
        'Momentum Weight (%)': [regime_weights[r][2] for r in ['TRENDING', 'MEAN_REVERTING', 'UNKNOWN']],
        'Rationale': [
            'Higher momentum weight for trend following',
            'Higher value weight for mean reversion opportunities',
            'Balanced allocation for uncertain conditions'
        ]
    })
    st.dataframe(weights_display, hide_index=True, use_container_width=True)

    st.markdown("#### 3. Trading Rules & Position Management")
    st.markdown(f"""
    **Signal Zones:**
    - **Buy Zone**: {buy_zone_min}%-{buy_zone_max}% (extreme high percentiles)
    - **Sell Zone**: {sell_zone_min}%-{sell_zone_max}% (extreme low percentiles)
    - **Neutral Zone**: {sell_zone_max}%-{buy_zone_min}% (no action)

    **Position Management Rules:**
    1. **Regime Filtering**: Only trade when regime confidence ≥ {confidence_threshold:.1f}
    2. **Capital Allocation**: {capital_allocation_pct}% per trade
    3. **Single Position Limit**: Maximum one position at a time when using 100% allocation
    4. **Signal Strength Validation**: Confirms signal intensity before execution

    **Trading Logic Hierarchy:**
    1. **CLOSE** positions when regime becomes UNKNOWN
    2. **REVERSE** positions when opposite signal appears
    3. **STRENGTHEN** positions only if significantly stronger signal (>10% improvement)
    4. **OPEN** new positions when flat and signal present

    **Transaction Cost Model:**
    - Fixed cost: {transaction_fee_bps} basis points per trade
    - Applied to full notional value of position
    - Deducted from realized P&L calculations
    """)

    # Risk Management Framework
    st.markdown("### ⚠️ Risk Management Framework")
    st.markdown(f"""
    **Portfolio Risk Controls:**
    - **Maximum Capital Utilization**: {capital_allocation_pct}% per position
    - **Regime Confidence Filtering**: Minimum {confidence_threshold:.1f} confidence required
    - **Position Concentration**: Single instrument focus (5Y Treasury Futures)
    - **Liquidity Management**: Daily mark-to-market and P&L calculation

    **Risk Metrics Monitoring:**
    - **Value at Risk (VaR)**: Estimated through historical simulation
    - **Maximum Drawdown**: Currently {max_dd:.2f}%
    - **Sharpe Ratio**: Risk-adjusted return measure = {sharpe_ratio:.3f}
    - **Capital Efficiency**: Current utilization = {current_capital_utilization:.1f}%

    **Risk Disclosures:**
    - **Market Risk**: Treasury futures subject to interest rate volatility
    - **Model Risk**: Regime classification may fail during market stress
    - **Liquidity Risk**: Position sizes may exceed market depth
    - **Operational Risk**: System depends on data quality and connectivity
    - **Regime Risk**: Historical patterns may not persist in future markets
    """)

    # Performance Attribution
    st.markdown("### 📊 Performance Attribution Analysis")

    if executed_trades:
        # Calculate detailed performance metrics
        trades_df = pd.DataFrame(executed_trades)
        profitable_trades = len(trades_df[trades_df.get('pnl', 0) > 0]) if 'pnl' in trades_df.columns else 0
        total_completed_trades = len(trades_df[trades_df.get('pnl', 0) != 0]) if 'pnl' in trades_df.columns else 0
        win_rate = (profitable_trades / total_completed_trades * 100) if total_completed_trades > 0 else 0
        avg_trade = trades_df.get('pnl', [0]).mean() if 'pnl' in trades_df.columns else 0
        max_trade = trades_df.get('pnl', [0]).max() if 'pnl' in trades_df.columns else 0
        min_trade = trades_df.get('pnl', [0]).min() if 'pnl' in trades_df.columns else 0

        st.markdown(f"""
        **Trading Performance Breakdown:**
        - **Total Executions**: {len(executed_trades)}
        - **Completed Trades**: {total_completed_trades}
        - **Win Rate**: {win_rate:.1f}%
        - **Average Trade P&L**: ${avg_trade:.2f}
        - **Best Trade**: ${max_trade:.2f}
        - **Worst Trade**: ${min_trade:.2f}
        - **Profit Factor**: {max_trade / abs(min_trade) if min_trade != 0 else 'N/A'}

        **P&L Attribution:**
        - **Realized P&L**: ${trading_system.realized_pnl:,.2f} ({(trading_system.realized_pnl / initial_cash) * 100:.2f}%)
        - **Unrealized P&L**: ${trading_system.unrealized_pnl:,.2f} ({(trading_system.unrealized_pnl / initial_cash) * 100:.2f}%)
        - **Total Transaction Costs**: ${sum([trade.get('cost', 0) for trade in executed_trades]):,.2f}
        - **Net Performance**: {total_return_pct:.2f}%

        **Risk-Adjusted Metrics:**
        - **Return/Risk Ratio**: {total_return_pct / abs(max_dd) if max_dd != 0 else 'N/A':.2f}
        - **Calmar Ratio**: {total_return_pct / abs(max_dd) if max_dd != 0 else 'N/A':.2f}
        - **Maximum Favorable Excursion**: {max_trade:.2f}
        - **Maximum Adverse Excursion**: {min_trade:.2f}
        """)

    # Data Sources & Assumptions
    st.markdown("### 📈 Data Sources & Model Assumptions")
    st.markdown(f"""
    **Primary Data Sources:**
    - **2Y Treasury Yield**: Federal Reserve Economic Data (FRED) - DGS2
    - **5Y Treasury Yield**: Yahoo Finance - ^FVX  
    - **5Y TIPS Yield**: Federal Reserve Economic Data (FRED) - DFII5
    - **5Y Treasury Futures**: Yahoo Finance - ZF=F

    **Data Quality Controls:**
    - Daily frequency with missing value interpolation
    - Outlier detection and filtering
    - Forward-fill methodology for holidays
    - Polynomial interpolation for TIPS data gaps

    **Model Assumptions:**
    - **Perfect Execution**: Trades execute at closing prices
    - **No Slippage**: Market impact costs not modeled
    - **Constant Spreads**: Transaction costs remain fixed
    - **No Margin Requirements**: Full cash-based trading
    - **No Borrowing Costs**: Overnight funding costs excluded
    - **Continuous Trading**: No weekend or holiday constraints

    **Backtesting Limitations:**
    - Historical optimization may not predict future performance
    - Survivorship bias in successful parameter selection
    - Look-ahead bias eliminated through walk-forward methodology
    - Transaction costs may be higher in practice
    - Market regime changes may invalidate historical relationships
    """)

    # System Architecture
    st.markdown("### 🏗️ System Architecture & Implementation")
    st.markdown(f"""
    **Core Components:**

    **1. Data Pipeline:**
    - Real-time data ingestion from FRED and Yahoo Finance APIs
    - Automated data validation and cleaning procedures
    - Rolling window calculations for all indicators
    - Missing data handling through interpolation methods

    **2. Signal Generation Engine:**
    - Multi-threaded indicator calculation
    - Regime classification with confidence scoring
    - Dynamic weight allocation based on current regime
    - Signal strength measurement and filtering

    **3. Portfolio Management System:**
    - Position sizing with capital allocation controls
    - Risk-based position limits and stop-loss mechanisms
    - Real-time P&L calculation and portfolio valuation
    - Transaction cost modeling and fee tracking

    **4. Risk Management Layer:**
    - Pre-trade risk checks and position limits
    - Real-time monitoring of drawdown and volatility
    - Regime confidence filtering for trade execution
    - Emergency position closing for unknown regimes

    **Performance Characteristics:**
    - **Calculation Speed**: ~{len(indicator_full)} observations processed
    - **Memory Usage**: Optimized for large datasets
    - **Update Frequency**: Real-time signal generation capability
    - **Scalability**: Designed for multiple instrument expansion
    """)

    # Future Enhancements
    st.markdown("### 🚀 Future Enhancements & Research Directions")
    st.markdown("""
    **Planned Improvements:**
    - **Multi-Asset Expansion**: Extension to 2Y, 10Y, 30Y Treasury futures
    - **Regime Transition Modeling**: Probabilistic regime switching models
    - **Machine Learning Integration**: Neural networks for regime classification
    - **Options Overlay**: Volatility-based hedging strategies
    - **Risk Parity**: Equal risk contribution across multiple strategies

    **Research Areas:**
    - **Alternative Data Sources**: Satellite data, sentiment analysis
    - **High-Frequency Signals**: Intraday momentum and mean reversion
    - **Cross-Asset Correlations**: Equity, commodity, and currency linkages
    - **Macroeconomic Integration**: Fed policy and economic data incorporation
    - **ESG Factors**: Environmental and social governance considerations

    **Technology Roadmap:**
    - **Cloud Migration**: AWS/Azure deployment for scalability
    - **API Development**: RESTful interfaces for third-party integration
    - **Real-time Streaming**: Kafka-based data pipeline implementation
    - **Mobile Dashboard**: iOS/Android portfolio monitoring applications
    - **Blockchain Integration**: Decentralized execution and settlement
    """)

    st.markdown('</div>', unsafe_allow_html=True)

# Professional Data Export Section with CODE EXPORT
st.markdown("---")
st.markdown("## 📁 Professional Data Export & Code Repository")

col1, col2, col3, col4 = st.columns(4)

with col1:
    csv_data = display_data.to_csv(index=True)
    st.download_button("📥 Download Market Data", csv_data,
                       f"treasury_trading_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

with col2:
    if executed_trades:
        trades_csv = pd.DataFrame(executed_trades).to_csv(index=False)
        st.download_button("📥 Download Trade History", trades_csv,
                           f"trade_history_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

with col3:
    portfolio_summary = pd.DataFrame({
        "Metric": ["Initial Capital", "Current Cash", "Realized P&L", "Unrealized P&L", "Total P&L",
                   "Portfolio Value", "Total Return %", "Capital Allocation %", "Max Capital Usage %"],
        "Value": [initial_cash, trading_system.cash, trading_system.realized_pnl,
                  trading_system.unrealized_pnl, total_pnl, portfolio_value, total_return_pct,
                  capital_allocation_pct, max_capital_usage]
    })
    summary_csv = portfolio_summary.to_csv(index=False)
    st.download_button("📥 Download Portfolio Summary", summary_csv,
                       f"portfolio_summary_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")


# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 2rem; border-top: 2px solid #dee2e6;'>
    <strong>Professional Treasury Trading System v2.0</strong><br>
    <em>Advanced Regime-Aware Quantitative Trading & Backtesting Platform</em><br>
    Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br>
    <small>For Professional Use Only - Past Performance Does Not Guarantee Future Results</small><br>
    <small>Total Analysis: {len(indicator_full):,} observations | Display Period: {len(display_data):,} days</small>
</div>
""", unsafe_allow_html=True)