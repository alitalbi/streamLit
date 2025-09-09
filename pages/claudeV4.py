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

st.set_page_config(layout="wide", page_title="Weight Optimizer", page_icon="🎯")

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


def calculate_agg_percentile(data, weights):
    """Calculate aggregate percentile with given weights"""
    value_w, carry_w, momentum_w = weights
    return (data['Value_Percentile'] * value_w +
            data['Carry_Percentile'] * carry_w +
            data['Momentum_Percentile'] * momentum_w) / 100


def backtest_strategy(data, weights, buy_zone, sell_zone, use_stop_loss, stop_loss_pct,
                      transaction_cost_bps, capital_allocation_pct, initial_cash=100000):
    """Detailed backtest with trade log and capital allocation"""
    if len(data) < 100:
        return {'total_pnl': -999999, 'max_drawdown': 999, 'num_trades': 0, 'win_rate': 0, 'trades': [],
                'executed_trades': []}

    data_copy = data.copy()
    data_copy['Agg_Percentile'] = calculate_agg_percentile(data_copy, weights)
    data_copy['Signal'] = 0  # 0=hold, 1=buy, -1=sell
    data_copy['Executed'] = False  # Track actual executions

    # Generate signals
    buy_min, buy_max = buy_zone
    sell_min, sell_max = sell_zone

    for i, (idx, row) in enumerate(data_copy.iterrows()):
        agg_perc = row['Agg_Percentile']
        if buy_min <= agg_perc <= buy_max:
            data_copy.loc[idx, 'Signal'] = 1
        elif sell_min <= agg_perc <= sell_max:
            data_copy.loc[idx, 'Signal'] = -1

    # Trading simulation
    cash = initial_cash
    position = 0  # 1 = long, -1 = short, 0 = flat
    position_size = 0  # Actual position size based on capital allocation
    entry_price = 0
    entry_date = None
    total_pnl = 0
    trades = []
    executed_trades = []  # Track actual executions
    equity_curve = [initial_cash]

    for i, (idx, row) in enumerate(data_copy.iterrows()):
        current_price = row['Close']
        signal = row['Signal']

        # Check stop loss
        if use_stop_loss and position != 0:
            pnl_pct = ((current_price - entry_price) / entry_price) * position
            if pnl_pct <= -stop_loss_pct / 100:
                # Stop loss triggered
                pnl = (current_price - entry_price) * position_size
                total_pnl += pnl
                cash += pnl
                cash -= abs(pnl) * (transaction_cost_bps / 10000)

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': idx,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position': 'LONG' if position > 0 else 'SHORT',
                    'pnl': pnl,
                    'return_pct': pnl_pct * 100,
                    'exit_reason': 'STOP_LOSS'
                })

                executed_trades.append(idx)
                data_copy.loc[idx, 'Executed'] = True
                position = 0
                position_size = 0

        # Trading signals
        if signal != 0 and signal != position:
            # Check if we can execute next day (not at end of data)
            if i < len(data_copy) - 1:
                next_day_price = data_copy.iloc[i + 1]['Open']  # Use next day's open price

                # Close existing position
                if position != 0:
                    pnl = (next_day_price - entry_price) * position_size  # FIXED: Use next_day_price
                    total_pnl += pnl
                    cash += pnl
                    cash -= abs(pnl) * (transaction_cost_bps / 10000)

                    return_pct = ((
                                              next_day_price - entry_price) / entry_price) * position * 100  # FIXED: Use next_day_price

                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': data_copy.index[i + 1],
                        'entry_price': entry_price,
                        'exit_price': next_day_price,
                        'position': 'LONG' if position > 0 else 'SHORT',
                        'pnl': pnl,
                        'return_pct': return_pct,
                        'exit_reason': 'SIGNAL'
                    })

                    executed_trades.append(data_copy.index[i + 1])

                # Open new position
                position = signal
                allocated_capital = cash * (capital_allocation_pct / 100)
                position_size = allocated_capital / next_day_price * position  # FIXED: Use next_day_price
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
            'exit_reason': 'FINAL'
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
st.title("🎯 Weight Optimizer with Train/Test Split")
st.caption("Find optimal weights in train period, test in out-of-sample period")

# Config
col1, col2, col3, col4 = st.columns(4)

with col1:
    lookback_years = st.selectbox("Train Years", [3, 5, 7, 10, 12, 15], index=2)
    test_start_date = st.date_input("Test Start Date", datetime.now().date() - timedelta(days=365 * 2))
    train_start_date = test_start_date - timedelta(days=lookback_years * 365)
    st.caption(f"Train: {train_start_date} to {test_start_date}")

with col2:
    optimization_goal = st.selectbox("Goal", ["Max PnL", "Max PnL/Drawdown Ratio"])
    buy_zone = st.slider("Buy Zone (%)", 80, 100, (90, 100))
    sell_zone = st.slider("Sell Zone (%)", 0, 20, (0, 10))

with col3:
    weight_step = st.selectbox("Weight Step (%)", [1, 5, 10, 20], index=1)
    transaction_cost = st.number_input("Transaction Cost (bps)", 0, 50, 2)
    capital_allocation_pct = st.slider("Capital Allocation per Trade (%)", 10, 100, 100)

with col4:
    initial_capital = st.number_input("Capital ($)", 10000, 1000000, 100000, step=10000)
    zscore_lookback = st.number_input("Z-Score Lookback Period (days)", value=63, min_value=20, max_value=252,
                                      help="Number of days for Z-score calculation (default: 63 ≈ 3 months)")

    use_stop_loss = st.checkbox("Use Stop Loss")
    stop_loss_pct = st.number_input("Stop Loss (%)", 1.0, 20.0, 5.0, step=0.5) if use_stop_loss else 0

# Data Loading and Optimization
if st.button("🚀 Run Optimization", type="primary"):

    with st.spinner("Loading data..."):
        # Load all data from train start date
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

    # Build indicators
    backtest_data = _2yUS.join(_5yUS_real).join(_5yUS).dropna()
    indicators = build_indicators(backtest_data)


    # Calculate percentiles
    for cols in ["5y_Real", "carry_normalized", "momentum"]:
        indicators[f"{cols}_z"] = zscore(indicators[cols], zscore_lookback)
        # MODIFIED: Calculate percentiles of the Z-scores instead of raw values
        indicators[f"{cols}_percentile"] = indicators[f"{cols}_z"].rolling(zscore_lookback).apply(
            lambda x: percentile_score(x))

    # Final dataset
    final_data = indicators[["5y", "5y_Real_percentile", "carry_normalized_percentile", "momentum_percentile"]].join(
        _5yUS_fut)
    final_data.columns = ["5y_yield", "Value_Percentile", "Carry_Percentile", "Momentum_Percentile", "Open", "High",
                          "Low", "Close"]
    final_data.dropna(inplace=True)
    final_data.index = pd.Series(final_data.index).dt.date
    # Train/Test Split by date
    train_data = final_data.loc[final_data.index < test_start_date].copy()
    test_data = final_data.loc[final_data.index >= test_start_date].copy()

    if len(train_data) == 0 or len(test_data) == 0:
        st.error("Invalid date split - no data in train or test period")
        st.stop()

    train_start = train_data.index[0].strftime("%Y-%m-%d")
    train_end = train_data.index[-1].strftime("%Y-%m-%d")
    test_start = test_data.index[0].strftime("%Y-%m-%d")
    test_end = test_data.index[-1].strftime("%Y-%m-%d")

    st.success(f"Train: {train_start} to {train_end} ({len(train_data)} days)")
    st.success(f"Test: {test_start} to {test_end} ({len(test_data)} days)")

    # Generate weight combinations (no zero weights)
    weight_combinations = []
    for value_w in range(weight_step, 101, weight_step):
        for carry_w in range(weight_step, 101 - value_w + weight_step, weight_step):
            momentum_w = 100 - value_w - carry_w
            if momentum_w >= weight_step:
                weight_combinations.append([value_w, carry_w, momentum_w])

    st.info(f"Testing {len(weight_combinations)} weight combinations on TRAIN period...")

    # Optimize on TRAIN data only
    results = []
    progress_bar = st.progress(0)

    for i, weights in enumerate(weight_combinations):
        if i % 50 == 0:
            progress_bar.progress(i / len(weight_combinations))

        metrics = backtest_strategy(
            train_data, weights, buy_zone, sell_zone,
            use_stop_loss, stop_loss_pct, transaction_cost, capital_allocation_pct, initial_capital
        )

        metrics['weights'] = weights
        results.append(metrics)

    progress_bar.progress(1.0)

    # Find best weights from TRAIN period
    if optimization_goal == "Max PnL":
        results.sort(key=lambda x: x['total_pnl'], reverse=True)
    else:
        for r in results:
            r['pnl_dd_ratio'] = r['total_pnl'] / max(r['max_drawdown'], 0.01)
        results.sort(key=lambda x: x['pnl_dd_ratio'], reverse=True)

    best_weights = results[0]['weights']

    # Test on both TRAIN and TEST periods with best weights
    train_results = backtest_strategy(train_data, best_weights, buy_zone, sell_zone,
                                      use_stop_loss, stop_loss_pct, transaction_cost, capital_allocation_pct,
                                      initial_capital)
    test_results = backtest_strategy(test_data, best_weights, buy_zone, sell_zone,
                                     use_stop_loss, stop_loss_pct, transaction_cost, capital_allocation_pct,
                                     initial_capital)

    # Store results in session state
    st.session_state.update({
        'best_weights': best_weights,
        'train_results': train_results,
        'test_results': test_results,
        'train_data': train_data,
        'test_data': test_data,
        'optimization_results': results[:10]
    })

# Display Results (if optimization has been run)
if 'best_weights' in st.session_state:
    best_weights = st.session_state.best_weights
    train_results = st.session_state.train_results
    test_results = st.session_state.test_results
    train_data = st.session_state.train_data
    test_data = st.session_state.test_data

    # Results Header
    st.markdown("---")
    st.subheader(
        f"🏆 Optimal Weights: Value:{best_weights[0]}% | Carry:{best_weights[1]}% | Momentum:{best_weights[2]}%")

    # Performance Comparison
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🚂 TRAIN Period Performance")
        st.metric("P&L", f"${train_results['total_pnl']:,.0f}", f"{train_results['total_return_pct']:+.1f}%")
        st.metric("Max Drawdown", f"{train_results['max_drawdown']:.1f}%")
        st.metric("Trades", f"{train_results['num_trades']}", f"Win Rate: {train_results['win_rate']:.1f}%")

    with col2:
        st.markdown("### 🧪 TEST Period Performance")
        st.metric("P&L", f"${test_results['total_pnl']:,.0f}", f"{test_results['total_return_pct']:+.1f}%")
        st.metric("Max Drawdown", f"{test_results['max_drawdown']:.1f}%")
        st.metric("Trades", f"{test_results['num_trades']}", f"Win Rate: {test_results['win_rate']:.1f}%")

    # Period Selection for Charts
    period_choice = st.radio("📊 Chart Period", ["Train", "Test"], horizontal=True)

    if period_choice == "Train":
        selected_data = train_data
        selected_results = train_results
        period_label = "TRAIN"
    else:
        selected_data = test_data
        selected_results = test_results
        period_label = "TEST"

    # Get data with signals
    data_with_signals = selected_results['data_with_signals']

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### 📈 5Y Futures Price + Executed Trades ({period_label})")

        fig_price = go.Figure()

        # Price line
        fig_price.add_trace(go.Scatter(
            x=data_with_signals.index,
            y=data_with_signals['Close'],
            mode='lines',
            name='5Y Futures Price',
            line=dict(color='black', width=1)
        ))

        # Executed buy signals (crosses)
        executed_buys = data_with_signals[(data_with_signals['Signal'] == 1) & (data_with_signals['Executed'] == True)]
        if len(executed_buys) > 0:
            fig_price.add_trace(go.Scatter(
                x=executed_buys.index,
                y=executed_buys['Close'],
                mode='markers',
                name='BUY Executed',
                marker=dict(color='green', size=12, symbol='x', line=dict(width=3))
            ))

        # Executed sell signals (crosses)
        executed_sells = data_with_signals[
            (data_with_signals['Signal'] == -1) & (data_with_signals['Executed'] == True)]
        if len(executed_sells) > 0:
            fig_price.add_trace(go.Scatter(
                x=executed_sells.index,
                y=executed_sells['Close'],
                mode='markers',
                name='SELL Executed',
                marker=dict(color='red', size=12, symbol='x', line=dict(width=3))
            ))

        # Non-executed signals (smaller triangles)
        non_exec_buys = data_with_signals[(data_with_signals['Signal'] == 1) & (data_with_signals['Executed'] == False)]
        if len(non_exec_buys) > 0:
            fig_price.add_trace(go.Scatter(
                x=non_exec_buys.index,
                y=non_exec_buys['Close'],
                mode='markers',
                name='BUY Signal',
                marker=dict(color='lightgreen', size=6, symbol='triangle-up'),
                opacity=0.6
            ))

        non_exec_sells = data_with_signals[
            (data_with_signals['Signal'] == -1) & (data_with_signals['Executed'] == False)]
        if len(non_exec_sells) > 0:
            fig_price.add_trace(go.Scatter(
                x=non_exec_sells.index,
                y=non_exec_sells['Close'],
                mode='markers',
                name='SELL Signal',
                marker=dict(color='lightcoral', size=6, symbol='triangle-down'),
                opacity=0.6
            ))

        fig_price.update_layout(height=400, xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_price, use_container_width=True)

    with col2:
        st.markdown(f"### 📊 Aggregate Percentile + Zones ({period_label})")

        fig_agg = go.Figure()

        # Aggregate percentile line
        fig_agg.add_trace(go.Scatter(
            x=data_with_signals.index,
            y=data_with_signals['Agg_Percentile'],
            mode='lines',
            name='Agg Percentile',
            line=dict(color='blue', width=2)
        ))

        # Buy zone
        fig_agg.add_hrect(y0=buy_zone[0], y1=buy_zone[1],
                          fillcolor="green", opacity=0.2,
                          annotation_text="BUY ZONE", annotation_position="top left")

        # Sell zone
        fig_agg.add_hrect(y0=sell_zone[0], y1=sell_zone[1],
                          fillcolor="red", opacity=0.2,
                          annotation_text="SELL ZONE", annotation_position="bottom left")

        fig_agg.update_layout(height=400, xaxis_title="Date", yaxis_title="Percentile (%)",
                              yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_agg, use_container_width=True)

    # Trade Log
    st.markdown(f"### 📋 Trade Log ({period_label} Period)")

    trades = selected_results['trades']
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date']).dt.strftime('%Y-%m-%d')
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date']).dt.strftime('%Y-%m-%d')
        trades_df['pnl'] = trades_df['pnl'].round(0).astype(int)
        trades_df['return_pct'] = trades_df['return_pct'].round(2)
        trades_df['entry_price'] = trades_df['entry_price'].round(3)
        trades_df['exit_price'] = trades_df['exit_price'].round(3)


        # Color code profitable trades
        def color_pnl(val):
            return 'color: green' if val > 0 else 'color: red'


        styled_trades = trades_df.style.applymap(color_pnl, subset=['pnl', 'return_pct'])
        st.dataframe(styled_trades, hide_index=True, use_container_width=True)

        # Trade summary
        profitable_trades = len([t for t in trades if t['pnl'] > 0])
        total_trades = len(trades)
        avg_trade = np.mean([t['pnl'] for t in trades])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Profitable Trades", f"{profitable_trades}/{total_trades}")
        with col2:
            st.metric("Average Trade P&L", f"${avg_trade:,.0f}")
        with col3:
            st.metric("Best Trade", f"${max([t['pnl'] for t in trades]):,.0f}")

    else:
        st.info("No trades executed in selected period")

    # Top Optimization Results
    st.markdown("### 🏆 Top 10 Weight Combinations (from Train period)")

    top_results = st.session_state.optimization_results
    results_table = pd.DataFrame({
        'Rank': range(1, len(top_results) + 1),
        'Value%': [r['weights'][0] for r in top_results],
        'Carry%': [r['weights'][1] for r in top_results],
        'Momentum%': [r['weights'][2] for r in top_results],
        'Train P&L': [f"${r['total_pnl']:,.0f}" for r in top_results],
        'Train Return%': [f"{r['total_return_pct']:.1f}" for r in top_results],
        'Train MaxDD%': [f"{r['max_drawdown']:.1f}" for r in top_results]
    })

    st.dataframe(results_table, hide_index=True, use_container_width=True)

    # Export
    st.markdown("### 📁 Export")
    col1, col2 = st.columns(2)

    with col1:
        config = {
            'best_weights': best_weights,
            'train_performance': {k: v for k, v in train_results.items() if
                                  k not in ['equity_curve', 'trades', 'data_with_signals']},
            'test_performance': {k: v for k, v in test_results.items() if
                                 k not in ['equity_curve', 'trades', 'data_with_signals']},
            'periods': {
                'train_start': train_data.index[0].strftime('%Y-%m-%d'),
                'train_end': train_data.index[-1].strftime('%Y-%m-%d'),
                'test_start': test_data.index[0].strftime('%Y-%m-%d'),
                'test_end': test_data.index[-1].strftime('%Y-%m-%d')
            }
        }
        st.download_button("📥 Download Config",
                           json.dumps(config, indent=2, default=str),
                           f"weights_config_{datetime.now().strftime('%Y%m%d')}.json")

    with col2:
        if trades:
            trades_csv = pd.DataFrame(trades).to_csv(index=False)
            st.download_button(f"📥 Download {period_label} Trades", trades_csv,
                               f"trades_{period_label.lower()}_{datetime.now().strftime('%Y%m%d')}.csv")

else:
    st.info("Click 'Run Optimization' to find optimal weights and see train/test performance")