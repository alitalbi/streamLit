import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
import json

warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Rolling Regression Backtester", page_icon="📈")

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


def rolling_regression_analysis(data, window=60, forecast_horizon=1, min_obs=30):
    """
    Perform rolling regression of factors on futures returns

    Model: Future_Return[t+h] = α + β1*Value[t] + β2*Carry[t] + β3*Momentum[t] + ε
    """
    # Prepare dependent variable (future returns)
    returns = data['Close'].pct_change(forecast_horizon).shift(-forecast_horizon) * 100  # Convert to percentage

    # Prepare independent variables (lagged by 1 to avoid look-ahead bias)
    value = data['Value_Percentile'].shift(1)
    carry = data['Carry_Percentile'].shift(1)
    momentum = data['Momentum_Percentile'].shift(1)

    # Storage for results
    results = pd.DataFrame(index=data.index, columns=[
        'alpha', 'beta_value', 'beta_carry', 'beta_momentum',
        'r_squared', 'predicted_return', 'actual_return',
        'value_impact', 'carry_impact', 'momentum_impact', 'total_signal'
    ])

    for i in range(window, len(data) - forecast_horizon):
        # Get rolling window data
        y = returns.iloc[i - window:i]
        X = pd.DataFrame({
            'value': value.iloc[i - window:i],
            'carry': carry.iloc[i - window:i],
            'momentum': momentum.iloc[i - window:i]
        }).dropna()

        y = y.loc[X.index]  # Align indices

        if len(X) >= min_obs and len(y) >= min_obs:
            try:
                # Fit regression
                reg = LinearRegression().fit(X, y)
                y_pred = reg.predict(X)

                # Store regression coefficients
                results.loc[data.index[i], 'alpha'] = reg.intercept_
                results.loc[data.index[i], 'beta_value'] = reg.coef_[0]
                results.loc[data.index[i], 'beta_carry'] = reg.coef_[1]
                results.loc[data.index[i], 'beta_momentum'] = reg.coef_[2]
                results.loc[data.index[i], 'r_squared'] = r2_score(y, y_pred)

                # Generate prediction for next period
                current_factors = [value.iloc[i], carry.iloc[i], momentum.iloc[i]]
                if not any(pd.isna(current_factors)):
                    prediction = reg.predict([current_factors])[0]
                    results.loc[data.index[i], 'predicted_return'] = prediction

                    # Calculate factor impacts using coefficient magnitude × factor direction
                    value_deviation = value.iloc[i] - 50
                    carry_deviation = carry.iloc[i] - 50
                    momentum_deviation = momentum.iloc[i] - 50

                    value_impact = reg.coef_[0] * value_deviation
                    carry_impact = reg.coef_[1] * carry_deviation
                    momentum_impact = reg.coef_[2] * momentum_deviation

                    results.loc[data.index[i], 'value_impact'] = value_impact
                    results.loc[data.index[i], 'carry_impact'] = carry_impact
                    results.loc[data.index[i], 'momentum_impact'] = momentum_impact
                    results.loc[data.index[i], 'total_signal'] = value_impact + carry_impact + momentum_impact

                # Store actual return for validation
                if i + forecast_horizon < len(returns):
                    results.loc[data.index[i], 'actual_return'] = returns.iloc[i + forecast_horizon]

            except Exception as e:
                continue  # Skip problematic observations

    return results


def generate_regression_signals(regression_results, signal_method="total_signal",
                                min_rsq=0.05, signal_threshold=2.0, beta_threshold=0.1):
    """
    Generate trading signals from regression results
    """
    signals = pd.Series(0, index=regression_results.index)

    # Quality filter: only trade when model has decent explanatory power
    quality_mask = (regression_results['r_squared'] >= min_rsq) & \
                   (regression_results['r_squared'].notna())

    if signal_method == "total_signal":
        # Method 1: Use combined factor impact signal
        signal_values = regression_results['total_signal']
        signals[quality_mask & (signal_values > signal_threshold)] = 1  # Buy
        signals[quality_mask & (signal_values < -signal_threshold)] = -1  # Sell

    elif signal_method == "predicted_return":
        # Method 2: Use predicted return directly
        pred_returns = regression_results['predicted_return']
        signals[quality_mask & (pred_returns > signal_threshold / 100)] = 1  # Buy
        signals[quality_mask & (pred_returns < -signal_threshold / 100)] = -1  # Sell

    elif signal_method == "dominant_factor":
        # Method 3: Trade based on dominant factor with significant beta
        for idx in regression_results.index:
            if not quality_mask[idx]:
                continue

            row = regression_results.loc[idx]
            if pd.isna(row['beta_value']):
                continue

            # Find factor with strongest coefficient
            betas = {
                'value': row['beta_value'],
                'carry': row['beta_carry'],
                'momentum': row['beta_momentum']
            }

            impacts = {
                'value': row['value_impact'],
                'carry': row['carry_impact'],
                'momentum': row['momentum_impact']
            }

            # Get dominant factor
            dominant_factor = max(betas.keys(), key=lambda k: abs(betas[k]))

            # Only trade if dominant factor has significant coefficient
            if abs(betas[dominant_factor]) > beta_threshold:
                impact = impacts[dominant_factor]
                if impact > signal_threshold:
                    signals[idx] = 1
                elif impact < -signal_threshold:
                    signals[idx] = -1

    return signals


def backtest_regression_strategy(data, regression_results, signals,
                                 transaction_cost_bps=2, capital_allocation_pct=100,
                                 initial_cash=100000):
    """
    Backtest the rolling regression strategy
    """
    if len(signals) == 0:
        return {'total_pnl': 0, 'max_drawdown': 0, 'num_trades': 0, 'trades': []}

    # Trading simulation with T+1 execution
    cash = initial_cash
    position = 0
    position_size = 0
    entry_price = 0
    entry_date = None
    total_pnl = 0
    trades = []
    equity_curve = [initial_cash]

    signal_series = signals.reindex(data.index, fill_value=0)

    for i, (idx, row) in enumerate(data.iterrows()):
        current_price = row['Close']
        signal = signal_series.get(idx, 0)

        # T+1 execution: if signal today, execute at next day's open
        if signal != 0 and signal != position and i < len(data) - 1:
            next_day_price = data.iloc[i + 1]['Open']

            # Close existing position
            if position != 0:
                pnl = (next_day_price - entry_price) * position_size
                total_pnl += pnl
                cash += pnl
                cash -= abs(pnl) * (transaction_cost_bps / 10000)

                return_pct = ((next_day_price - entry_price) / entry_price) * position * 100

                # Get regression context for this trade
                reg_info = regression_results.loc[idx] if idx in regression_results.index else {}

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': data.index[i + 1],
                    'entry_price': entry_price,
                    'exit_price': next_day_price,
                    'position': 'LONG' if position > 0 else 'SHORT',
                    'pnl': pnl,
                    'return_pct': return_pct,
                    'exit_reason': 'SIGNAL',
                    'predicted_return': reg_info.get('predicted_return', np.nan),
                    'actual_return': reg_info.get('actual_return', np.nan),
                    'r_squared': reg_info.get('r_squared', np.nan),
                    'total_signal': reg_info.get('total_signal', np.nan)
                })

            # Open new position
            position = signal
            allocated_capital = cash * (capital_allocation_pct / 100)
            position_size = allocated_capital / next_day_price * position
            entry_price = next_day_price
            entry_date = data.index[i + 1]
            cash -= abs(allocated_capital) * (transaction_cost_bps / 10000)

        # Track equity
        if position != 0:
            unrealized_pnl = (current_price - entry_price) * position_size
            current_equity = cash + unrealized_pnl
        else:
            current_equity = cash
        equity_curve.append(current_equity)

    # Close final position
    if position != 0:
        final_price = data.iloc[-1]['Close']
        pnl = (final_price - entry_price) * position_size
        total_pnl += pnl

        trades.append({
            'entry_date': entry_date,
            'exit_date': data.index[-1],
            'entry_price': entry_price,
            'exit_price': final_price,
            'position': 'LONG' if position > 0 else 'SHORT',
            'pnl': pnl,
            'return_pct': ((final_price - entry_price) / entry_price) * position * 100,
            'exit_reason': 'FINAL',
            'predicted_return': np.nan,
            'actual_return': np.nan,
            'r_squared': np.nan,
            'total_signal': np.nan
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
        'trades': trades
    }


# Header
st.title("📈 Rolling Regression Factor Strategy Backtester")
st.caption("Dynamic factor model with time-varying coefficients")

# Parameters
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Data Parameters**")
    start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.date_input("End Date", datetime.now().date())
    zscore_lookback = st.number_input("Z-Score Lookback", value=63, min_value=20, max_value=252)

with col2:
    st.markdown("**Regression Parameters**")
    regression_window = st.slider("Regression Window", 30, 120, 60)
    forecast_horizon = st.selectbox("Forecast Horizon", [1, 3, 5], index=0)
    min_observations = st.slider("Min Observations", 20, 60, 30)
    min_rsquared = st.slider("Min R-squared", 0.0, 0.2, 0.05, 0.01)

with col3:
    st.markdown("**Signal Parameters**")
    signal_method = st.selectbox("Signal Method", ["total_signal", "predicted_return", "dominant_factor"])
    signal_threshold = st.slider("Signal Threshold", 0.5, 5.0, 2.0, 0.5)
    beta_threshold = st.slider("Beta Threshold", 0.05, 0.5, 0.1, 0.05)

with col4:
    st.markdown("**Trading Parameters**")
    transaction_cost = st.number_input("Transaction Cost (bps)", 0, 50, 2)
    capital_allocation = st.slider("Capital Allocation (%)", 10, 100, 100)
    initial_capital = st.number_input("Initial Capital ($)", 10000, 1000000, 100000, step=10000)

# Run Analysis
if st.button("🚀 Run Rolling Regression Analysis", type="primary"):

    with st.spinner("Loading data..."):
        # Load data (same as regime-aware system)
        start_str = start_date.strftime("%Y-%m-%d")

        _2yUS = get_fred_data("DGS2", start_str)
        _2yUS.columns = ["2y"]

        _5yUS = get_github_data("^FVX", start_str)
        _5yUS.columns = ["5y", "High", "Low", "Open"]

        _5yUS_real = get_fred_data("DFII5", start_str)
        _5yUS_real.columns = ["5y_Real"]
        _5yUS_real = _5yUS_real.interpolate(method="polynomial", order=2)

        _5yUS_fut = get_github_data("ZF=F", start_str)

        if any(df.empty for df in [_2yUS, _5yUS, _5yUS_real, _5yUS_fut]):
            st.error("Failed to load data")
            st.stop()

    with st.spinner("Building indicators..."):
        # Build indicators
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
        final_data = final_data.loc[
            (final_data.index >= pd.Timestamp(start_date)) & (final_data.index <= pd.Timestamp(end_date))]

    with st.spinner("Running rolling regression..."):
        # Perform rolling regression
        regression_results = rolling_regression_analysis(
            final_data, window=regression_window,
            forecast_horizon=forecast_horizon, min_obs=min_observations
        )

        # Generate signals
        signals = generate_regression_signals(
            regression_results, signal_method=signal_method,
            min_rsq=min_rsquared, signal_threshold=signal_threshold,
            beta_threshold=beta_threshold
        )

        # Backtest strategy
        backtest_results = backtest_regression_strategy(
            final_data, regression_results, signals,
            transaction_cost, capital_allocation, initial_capital
        )

    # Store results
    st.session_state.update({
        'regression_results': regression_results,
        'backtest_results': backtest_results,
        'signals': signals,
        'final_data': final_data,
        'signal_method': signal_method
    })

# Display Results
if 'regression_results' in st.session_state:
    regression_results = st.session_state.regression_results
    backtest_results = st.session_state.backtest_results
    signals = st.session_state.signals
    final_data = st.session_state.final_data
    signal_method = st.session_state.signal_method

    # Performance Summary
    st.markdown("---")
    st.subheader("📊 Performance Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total P&L", f"${backtest_results['total_pnl']:,.0f}")
        st.metric("Total Return", f"{backtest_results['total_return_pct']:+.1f}%")

    with col2:
        st.metric("Max Drawdown", f"{backtest_results['max_drawdown']:.1f}%")
        st.metric("Number of Trades", f"{backtest_results['num_trades']}")

    with col3:
        st.metric("Win Rate", f"{backtest_results['win_rate']:.1f}%")
        avg_rsq = regression_results['r_squared'].mean()
        st.metric("Avg R-squared", f"{avg_rsq:.3f}")

    with col4:
        if backtest_results['num_trades'] > 0:
            avg_trade = backtest_results['total_pnl'] / backtest_results['num_trades']
            st.metric("Avg Trade P&L", f"${avg_trade:,.0f}")

        valid_predictions = regression_results['predicted_return'].notna().sum()
        st.metric("Valid Predictions", f"{valid_predictions}")

    # Main Charts
    st.markdown("## 📈 Analysis Charts")

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            "Price Chart with Trading Signals",
            "Rolling Regression Coefficients (Betas)",
            "Factor Impacts and Total Signal",
            "Model Quality (R-squared) and Predicted vs Actual Returns"
        ),
        vertical_spacing=0.08,
        row_heights=[0.3, 0.25, 0.25, 0.2],
        shared_xaxes=True
    )

    # 1. Price chart with signals
    fig.add_trace(
        go.Scatter(
            x=final_data.index,
            y=final_data['Close'],
            mode='lines',
            name='5Y Futures',
            line=dict(color='black', width=1)
        ),
        row=1, col=1
    )

    # Add trading signals
    buy_signals = signals[signals == 1]
    sell_signals = signals[signals == -1]

    if len(buy_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=final_data.loc[buy_signals.index, 'Close'],
                mode='markers',
                name='BUY Signals',
                marker=dict(color='green', size=8, symbol='triangle-up')
            ),
            row=1, col=1
        )

    if len(sell_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=final_data.loc[sell_signals.index, 'Close'],
                mode='markers',
                name='SELL Signals',
                marker=dict(color='red', size=8, symbol='triangle-down')
            ),
            row=1, col=1
        )

    # 2. Rolling coefficients
    fig.add_trace(
        go.Scatter(x=regression_results.index, y=regression_results['beta_value'],
                   mode='lines', name='Value Beta', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=regression_results.index, y=regression_results['beta_carry'],
                   mode='lines', name='Carry Beta', line=dict(color='orange')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=regression_results.index, y=regression_results['beta_momentum'],
                   mode='lines', name='Momentum Beta', line=dict(color='green')),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # 3. Factor impacts
    fig.add_trace(
        go.Scatter(x=regression_results.index, y=regression_results['value_impact'],
                   mode='lines', name='Value Impact', line=dict(color='lightblue')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=regression_results.index, y=regression_results['carry_impact'],
                   mode='lines', name='Carry Impact', line=dict(color='lightsalmon')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=regression_results.index, y=regression_results['momentum_impact'],
                   mode='lines', name='Momentum Impact', line=dict(color='lightgreen')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=regression_results.index, y=regression_results['total_signal'],
                   mode='lines', name='Total Signal', line=dict(color='purple', width=2)),
        row=3, col=1
    )
    fig.add_hline(y=signal_threshold, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hline(y=-signal_threshold, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", row=3, col=1)

    # 4. Model quality and predictions
    fig.add_trace(
        go.Scatter(x=regression_results.index, y=regression_results['r_squared'],
                   mode='lines', name='R-squared', line=dict(color='black')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=regression_results.index, y=regression_results['predicted_return'],
                   mode='lines', name='Predicted Return', line=dict(color='blue', dash='dot')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=regression_results.index, y=regression_results['actual_return'],
                   mode='lines', name='Actual Return', line=dict(color='red', dash='dot')),
        row=4, col=1
    )
    fig.add_hline(y=min_rsquared, line_dash="dash", line_color="orange", row=4, col=1)

    fig.update_layout(height=1000, showlegend=True)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Beta", row=2, col=1)
    fig.update_yaxes(title_text="Impact", row=3, col=1)
    fig.update_yaxes(title_text="R² / Returns", row=4, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Trade Log
    st.markdown("## 📋 Trade Log with Regression Context")

    if backtest_results['trades']:
        trades_df = pd.DataFrame(backtest_results['trades'])
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date']).dt.strftime('%Y-%m-%d')
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date']).dt.strftime('%Y-%m-%d')
        trades_df['pnl'] = trades_df['pnl'].round(0).astype(int)
        trades_df['return_pct'] = trades_df['return_pct'].round(2)
        trades_df['predicted_return'] = trades_df['predicted_return'].round(3)
        trades_df['actual_return'] = trades_df['actual_return'].round(3)
        trades_df['r_squared'] = trades_df['r_squared'].round(3)
        trades_df['total_signal'] = trades_df['total_signal'].round(2)

        # Add prediction accuracy
        trades_df['pred_accuracy'] = np.where(
            (trades_df['predicted_return'] > 0) == (trades_df['actual_return'] > 0),
            'Correct Direction', 'Wrong Direction'
        )


        def color_pnl(val):
            return 'color: green' if val > 0 else 'color: red'


        def color_accuracy(val):
            return 'background-color: lightgreen' if val == 'Correct Direction' else 'background-color: lightcoral'


        styled_trades = trades_df.style.applymap(color_pnl, subset=['pnl', 'return_pct'])
        styled_trades = styled_trades.applymap(color_accuracy, subset=['pred_accuracy'])

        st.dataframe(styled_trades, hide_index=True, use_container_width=True)

        # Trade Statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            avg_pred_accuracy = (trades_df['pred_accuracy'] == 'Correct Direction').mean() * 100
            st.metric("Prediction Accuracy", f"{avg_pred_accuracy:.1f}%")

        with col2:
            avg_rsq_trades = trades_df['r_squared'].mean()
            st.metric("Avg R² at Trade", f"{avg_rsq_trades:.3f}")

        with col3:
            avg_signal_strength = abs(trades_df['total_signal']).mean()
            st.metric("Avg Signal Strength", f"{avg_signal_strength:.2f}")

    else:
        st.info("No trades executed with current parameters")

    # Model Diagnostics
    st.markdown("## 🔍 Model Diagnostics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Coefficient Stability**")
        coef_stats = pd.DataFrame({
            'Factor': ['Value', 'Carry', 'Momentum'],
            'Mean Beta': [
                regression_results['beta_value'].mean(),
                regression_results['beta_carry'].mean(),
                regression_results['beta_momentum'].mean()
            ],
            'Std Beta': [
                regression_results['beta_value'].std(),
                regression_results['beta_carry'].std(),
                regression_results['beta_momentum'].std()
            ]
        }).round(3)
        st.dataframe(coef_stats, hide_index=True)

    with col2:
        st.markdown("**Signal Distribution**")
        signal_stats = pd.DataFrame({
            'Signal': ['Buy', 'Hold', 'Sell'],
            'Count': [
                (signals == 1).sum(),
                (signals == 0).sum(),
                (signals == -1).sum()
            ]
        })
        signal_stats['Percentage'] = (signal_stats['Count'] / len(signals) * 100).round(1)
        st.dataframe(signal_stats, hide_index=True)

    # Export Results
    st.markdown("## 📁 Export Results")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Download Regression Results"):
            csv_data = regression_results.to_csv()
            st.download_button(
                "Download CSV",
                csv_data,
                f"regression_results_{datetime.now().strftime('%Y%m%d')}.csv"
            )

    with col2:
        if backtest_results['trades']:
            trades_csv = pd.DataFrame(backtest_results['trades']).to_csv(index=False)
            st.download_button(
                "📥 Download Trades",
                trades_csv,
                f"regression_trades_{datetime.now().strftime('%Y%m%d')}.csv"
            )

else:
    st.info("Click 'Run Rolling Regression Analysis' to start")

    st.markdown("""
    ## How This Works

    **Rolling Regression Model:**
    ```
    Future_Return[t+h] = α + β1*Value[t] + β2*Carry[t] + β3*Momentum[t] + ε
    ```

    **Signal Generation:**
    - **Total Signal Method**: Factor_Impact = Beta × (Percentile - 50), combined across factors
    - **Predicted Return Method**: Trade based on model's return forecast  
    - **Dominant Factor Method**: Focus on factor with strongest coefficient

    **Key Features:**
    - Time-varying factor sensitivities (rolling betas)
    - Model quality filtering (minimum R-squared)
    - T+1 execution for realistic trading
    - Comprehensive diagnostics and trade attribution
    """)