import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from fredapi import Fred
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Simple Rolling Regression", page_icon="📈")

# Initialize FRED API
fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')


@st.cache_data
def get_github_data(ticker, start_date):
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
    try:
        return pd.DataFrame(fred.get_series(ticker, observation_start=start_date, freq="daily"))
    except Exception as e:
        st.error(f"Error loading FRED {ticker}: {e}")
        return pd.DataFrame()


def zscore(data, lookback):
    return (data - data.rolling(lookback).mean()) / data.rolling(lookback).std()


def percentile_score(window):
    if len(window) == 0:
        return np.nan
    current_value = window[-1]
    return (np.sum(window <= current_value) / len(window)) * 100


def build_indicators(data):
    data["5_2y"] = data["5y"] - data["2y"]
    data["carry_normalized"] = data["5_2y"] / data["5_2y"].rolling(75).std()
    data["momentum"] = data["5y"].rolling(5).mean() - data["5y"].rolling(20).mean()
    return data


def rolling_regression_strategy(data, window=60, threshold=0.5):
    """
    Simple strategy:
    1. Predict tomorrow's return using Value, Carry, Momentum
    2. If prediction > threshold%, buy
    3. If prediction < -threshold%, sell
    """

    # Tomorrow's return (what we predict)
    future_returns = data['Close'].pct_change().shift(-1) * 100

    # Today's factors (what we use to predict)
    factors = data[['Value_Percentile', 'Carry_Percentile', 'Momentum_Percentile']].copy()

    predictions = []
    signals = []

    for i in range(window, len(data) - 1):
        # Get training data (past 60 days)
        y_train = future_returns.iloc[i - window:i].dropna()
        X_train = factors.iloc[i - window:i].dropna()

        # Make sure we have matching indices
        common_idx = y_train.index.intersection(X_train.index)

        if len(common_idx) >= 30:  # Need enough data
            y_train = y_train.loc[common_idx]
            X_train = X_train.loc[common_idx]

            # Fit regression
            reg = LinearRegression().fit(X_train, y_train)

            # Predict tomorrow using today's factors
            today_factors = factors.iloc[i:i + 1]
            prediction = reg.predict(today_factors)[0]

            predictions.append(prediction)

            # Generate signal
            if prediction > threshold:
                signals.append(1)  # Buy
            elif prediction < -threshold:
                signals.append(-1)  # Sell
            else:
                signals.append(0)  # Hold
        else:
            predictions.append(np.nan)
            signals.append(0)

    # Create results dataframe
    results = pd.DataFrame(index=data.index[window:-1])
    results['prediction'] = predictions
    results['signal'] = signals
    results['actual_return'] = future_returns.iloc[window:-1].values

    return results


def backtest_simple(data, results, transaction_cost=0.02):
    """
    Simple backtest: buy/sell based on signals
    """
    cash = 100000
    position = 0
    trades = []
    equity = [cash]

    for i, (date, row) in enumerate(results.iterrows()):
        signal = row['signal']

        if i < len(data) - 1:  # Make sure we have next day data
            current_price = data.loc[date, 'Close']

            # Execute trades at next day's open
            next_date = data.index[data.index.get_loc(date) + 1]
            next_price = data.loc[next_date, 'Open']

            if signal != position and signal != 0:
                # Close old position
                if position != 0:
                    pnl = (next_price - entry_price) * position * (cash / entry_price)
                    cash += pnl
                    cash *= (1 - transaction_cost / 100)  # Transaction cost

                    trades.append({
                        'exit_date': next_date,
                        'exit_price': next_price,
                        'pnl': pnl,
                        'prediction': row['prediction'],
                        'actual': row['actual_return']
                    })

                # Open new position
                position = signal
                entry_price = next_price
                cash *= (1 - transaction_cost / 100)  # Transaction cost

            # Update equity
            if position != 0:
                current_value = cash + (current_price - entry_price) * position * (cash / entry_price)
            else:
                current_value = cash
            equity.append(current_value)

    return {
        'final_value': equity[-1],
        'total_return': (equity[-1] - 100000) / 100000 * 100,
        'trades': trades,
        'equity_curve': equity
    }


# UI
st.title("Simple Rolling Regression Strategy")
st.caption("Predict tomorrow's return, trade if prediction is strong enough")

# Parameters
col1, col2, col3 = st.columns(3)

with col1:
    start_date = st.date_input("Start Date", datetime(2022, 1, 1))
    regression_window = st.slider("Regression Window", 30, 120, 60)

with col2:
    threshold = st.slider("Prediction Threshold (%)", 0.1, 2.0, 0.5, 0.1)
    transaction_cost = st.slider("Transaction Cost (%)", 0.0, 0.1, 0.02, 0.01)

with col3:
    zscore_lookback = st.number_input("Z-Score Lookback", value=63, min_value=20, max_value=252)

# Run Strategy
if st.button("Run Strategy"):

    with st.spinner("Loading data..."):
        # Load data
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

    with st.spinner("Running strategy..."):
        # Run strategy
        results = rolling_regression_strategy(final_data, regression_window, threshold)
        backtest = backtest_simple(final_data, results, transaction_cost)

    # Results
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Return", f"{backtest['total_return']:.1f}%")

    with col2:
        st.metric("Final Value", f"${backtest['final_value']:,.0f}")

    with col3:
        st.metric("Number of Trades", len(backtest['trades']))

    with col4:
        if len(backtest['trades']) > 0:
            profitable = sum(1 for t in backtest['trades'] if t['pnl'] > 0)
            win_rate = profitable / len(backtest['trades']) * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")

    # Charts
    st.markdown("## Charts")

    fig = go.Figure()

    # Price
    fig.add_trace(go.Scatter(
        x=final_data.index,
        y=final_data['Close'],
        mode='lines',
        name='Price',
        line=dict(color='black')
    ))

    # Buy signals
    buy_dates = results[results['signal'] == 1].index
    if len(buy_dates) > 0:
        fig.add_trace(go.Scatter(
            x=buy_dates,
            y=final_data.loc[buy_dates, 'Close'],
            mode='markers',
            name='BUY',
            marker=dict(color='green', size=8, symbol='triangle-up')
        ))

    # Sell signals
    sell_dates = results[results['signal'] == -1].index
    if len(sell_dates) > 0:
        fig.add_trace(go.Scatter(
            x=sell_dates,
            y=final_data.loc[sell_dates, 'Close'],
            mode='markers',
            name='SELL',
            marker=dict(color='red', size=8, symbol='triangle-down')
        ))

    fig.update_layout(title="Price with Trading Signals", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Predictions vs Actual
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=results.index,
        y=results['prediction'],
        mode='lines',
        name='Predicted Return',
        line=dict(color='blue')
    ))

    fig2.add_trace(go.Scatter(
        x=results.index,
        y=results['actual_return'],
        mode='lines',
        name='Actual Return',
        line=dict(color='red', dash='dot')
    ))

    fig2.add_hline(y=threshold, line_dash="dash", line_color="green", annotation_text="Buy Threshold")
    fig2.add_hline(y=-threshold, line_dash="dash", line_color="red", annotation_text="Sell Threshold")
    fig2.add_hline(y=0, line_color="gray")

    fig2.update_layout(title="Predicted vs Actual Returns", height=400)
    st.plotly_chart(fig2, use_container_width=True)

    # Trade Details
    if len(backtest['trades']) > 0:
        st.markdown("## Recent Trades")

        trades_df = pd.DataFrame(backtest['trades'][-10:])  # Last 10 trades
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date']).dt.strftime('%Y-%m-%d')
        trades_df = trades_df.round(2)

        st.dataframe(trades_df, use_container_width=True)

        # Prediction accuracy
        correct_predictions = 0
        for trade in backtest['trades']:
            if (trade['prediction'] > 0 and trade['actual'] > 0) or (trade['prediction'] < 0 and trade['actual'] < 0):
                correct_predictions += 1

        if len(backtest['trades']) > 0:
            accuracy = correct_predictions / len(backtest['trades']) * 100
            st.metric("Prediction Accuracy", f"{accuracy:.1f}%")

else:
    st.info("Set parameters and click 'Run Strategy'")

    st.markdown("""
    ## How It Works

    **Simple Logic:**
    1. Use rolling 60-day window to fit: `Tomorrow_Return = a + b1*Value + b2*Carry + b3*Momentum`
    2. If predicted return > +0.5%, BUY
    3. If predicted return < -0.5%, SELL
    4. Otherwise, HOLD

    **That's it!** No complex factor impacts or regime detection.
    """)