import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import yfinance as yf
from fredapi import Fred
import streamlit as st
import plotly.graph_objects as go
import urllib
fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')
st.set_page_config(layout="wide")


logging.basicConfig(level=logging.INFO)

assets_dict = {
    "2y US": "DGS2",
    "5y US": "^FVX",
    "5y US Real": "DFII5",
    "5y US Future": "ZF=F"}


def get_data(ticker,start):
    ticker_request = ticker.replace("=", "%3D")

    try:
        endpoint_data = f"https://raw.githubusercontent.com/alitalbi/storage_data_fy/refs/heads/master/{ticker_request}.csv"
        price_df = pd.read_csv(endpoint_data,usecols=["Date","Close"])
        price_df.set_index("Date", inplace=True)
        price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
        price_df = price_df.loc[price_df.index > start]
        return price_df
    except urllib.error.HTTPError as e:
            print(f"HTTP Error: {e.code} {e.reason}")
            print(f"URL: {endpoint_data}")
            raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
def import_data(ticker, start_date):
    logging.info(f"ticker used : {ticker}")
    data = get_data(ticker, start=start_date)
    return data

def fred_import(ticker, start_date):
    fred_data = pd.DataFrame(
        fred.get_series(ticker, observation_start=start_date, freq="daily"))
    return fred_data

def build_indicators(data):
    data["5_2y"] = data["5y"] - data["2y"]
    data["carry_normalized"] = data["5_2y"] / data["5_2y"].rolling(75).std()
    data["5d_ma_5y"] = data["5y"].rolling(5).mean()
    data["20d_ma_5y"] = data["5y"].rolling(20).mean()
    data["momentum"] = data["20d_ma_5y"] - data["5d_ma_5y"]
    return data

def percentile_score(window):
    if len(window) == 0:
        return np.nan
    current_value = window[-1]
    nb_values_below = np.sum(window <= current_value)
    percentile = (nb_values_below / len(window)) * 100
    return percentile

def zscore(data, lookback):
    return (data - data.rolling(lookback).mean()) / data.rolling(lookback).std()

if __name__ == "__main__":
    # Customizable start date with 6 months prior data
    default_date = datetime(2023, 10, 1)
    cols = st.columns([2,3,3])
    with cols[0]:
        start_date_input = st.date_input("Select Start Date", value=default_date)
        subcols = st.columns([2,2,2])
        with subcols[0]:
            carry_weight = st.text_input("W Carry %",value=20)
            carry_weight = float(carry_weight)
        with subcols[1]:
            value_weight = st.text_input("W Value %",value=70)
            value_weight = float(value_weight)
        with subcols[2]:
            momentum_weight = st.text_input("W Momentum %",value=10)
            momentum_weight = float(momentum_weight)
        if momentum_weight + value_weight + carry_weight > 100:
            st.error("Careful Weights > 100%")
    start_date = start_date_input - timedelta(days=6*30)  # 6 months prior
    start_date_str = start_date.strftime("%Y-%m-%d")

    # Import data
    _2yUS = fred_import(assets_dict["2y US"], start_date_str)
    _2yUS.columns = ["2y"]
    _5yUS = import_data(assets_dict["5y US"], start_date_str)
    _5yUS.columns = ["5y"]
    _5yUS_real = fred_import(assets_dict["5y US Real"], start_date_str)
    _5yUS_real.columns = ["5y_Real"]
    _5yUS_real = _5yUS_real.interpolate(method="polynomial", order=2)

    backtest_data = _2yUS.join(_5yUS_real).join(_5yUS)
    backtest_data.dropna(inplace=True)
    indicators = build_indicators(backtest_data)

    # Calculate Z-scores and percentiles
    for cols in ["5y_Real", "carry_normalized", "momentum"]:
        indicators[f"{cols}_z"] = zscore(indicators[cols], 63)
        indicators[f"{cols}_percentile"] = indicators[cols].rolling(63).apply(lambda x: percentile_score(x))
    indicators["Agg_Percentile"] = (indicators["5y_Real_percentile"]*value_weight + indicators["carry_normalized_percentile"]* carry_weight + indicators["momentum_percentile"] * momentum_weight)/100

    # Trading signals
    indicators["signal"] = 0
    indicators.loc[(indicators["Agg_Percentile"] > 90) & (indicators["Agg_Percentile"] <= 100), "signal"] = 1
    indicators.loc[(indicators["Agg_Percentile"] >= 10) & (indicators["Agg_Percentile"] <= 20), "signal"] = -1

    # Join with 5Y Futures for price-based signals
    _5yUS_fut = import_data(assets_dict["5y US Future"], start_date_str)
    indicator_full = indicators[["signal", "Agg_Percentile", "5y"]].join(_5yUS_fut)
    indicator_full.columns = ["signal", "Agg_Percentile", "5y_yield", "price"]
    indicator_full.dropna(inplace=True)
    # Calculate strategy returns
    indicator_full['price_change'] = indicator_full['price'].pct_change()
    indicator_full['strategy_return'] = indicator_full['price_change'] * indicator_full['signal'].shift(1)
    indicator_full['cum_return'] = (1 + indicator_full['strategy_return']).cumprod() - 1
    total_return = indicator_full['cum_return'].iloc[-1] * 100 if not np.isnan(indicator_full['cum_return'].iloc[-1]) else 0
    cols = st.columns([3,2,2,3])
    with cols[0]:
        if st.checkbox("Display Indicator Table"):
            st.write(indicator_full)
        st.download_button(data=indicator_full.to_csv(index=False).encode('utf-8'), label="Signal Download", file_name="Signal_5yUS_Percentile.csv")
        st.download_button(data=indicators.to_csv(index=False).encode('utf-8'), label="Indicators Download", file_name="Indicators_5yUS_Percentile.csv")
    # Layout with two columns
    col1, col2 = st.columns(2)


    # Second Plot: Agg Percentile and 5Y Yield with Zones (in col1)
    with col1:
        fig2 = go.Figure()

        # Agg Percentile (left axis)
        fig2.add_trace(go.Scatter(
            x=indicator_full.index,
            y=indicator_full["Agg_Percentile"],
            mode="lines",
            name="Agg Percentile",
            yaxis="y1"
        ))

        # 5Y Yield (right axis)
        fig2.add_trace(go.Scatter(
            x=indicator_full.index,
            y=indicator_full["5y_yield"],
            mode="lines",
            name="5Y Yield",
            yaxis="y2"
        ))

        # Add shaded zones
        fig2.add_trace(go.Scatter(
            x=indicator_full.index[[0, -1, -1, 0]],
            y=[90, 90, 100, 100],
            fill="toself",
            fillcolor="rgba(0, 255, 0, 0.2)",
            line_color="rgba(255, 255, 255, 0)",
            name="Buy Zone"
        ))
        fig2.add_trace(go.Scatter(
            x=indicator_full.index[[0, -1, -1, 0]],
            y=[10, 10, 20, 20],
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.2)",
            line_color="rgba(255, 255, 255, 0)",
            name="Sell Zone"
        ))

        # Update layout for dual axes
        fig2.update_layout(
            yaxis1=dict(title="Agg Percentile", range=[0, 100], side="left", overlaying="y2"),
            yaxis2=dict(title="5Y Yield", range=[min(indicator_full["5y_yield"]) * 0.9, max(indicator_full["5y_yield"]) * 1.1], side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Table with latest percentile values (in col2)
    with col2:
        latest_data = indicators.iloc[-1]
        percentile_table = pd.DataFrame({
            "5Y Treasuries": ["Value", "Carry", "Momentum", "Aggregate"],
            "Percentile %": [
                latest_data["5y_Real_percentile"],
                latest_data["carry_normalized_percentile"],
                latest_data["momentum_percentile"],
                latest_data["Agg_Percentile"]
            ]
        })
        st.subheader("Indicators Table")
        st.dataframe(percentile_table.round(0), hide_index=True,width=200)
    col1, col2 = st.columns(2)
    # First Plot: Futures Price with Buy/Sell Signals (in col1)
    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=indicator_full.index,
            y=indicator_full["price"],
            mode="lines",
            name="Futures Price"
        ))
        buy_points = indicator_full[indicator_full["signal"] == 1]
        fig1.add_trace(go.Scatter(
            x=buy_points.index,
            y=buy_points["price"],
            mode="markers",
            marker=dict(symbol="triangle-up", color="green", size=12),
            name="Buy Signal"
        ))
        sell_points = indicator_full[indicator_full["signal"] == -1]
        fig1.add_trace(go.Scatter(
            x=sell_points.index,
            y=sell_points["price"],
            mode="markers",
            marker=dict(symbol="triangle-down", color="red", size=12),
            name="Sell Signal"
        ))
        st.plotly_chart(fig1, use_container_width=True)

    # Third Plot: Cumulative Returns (in col1)
    with col2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=indicator_full.index,
            y=indicator_full["cum_return"] * 100,
            mode="lines",
            name="Cumulative Return (%)"
        ))
        fig3.update_layout(
            title="Strategy Cumulative Return",
            yaxis=dict(title="Cumulative Return (%)", range=[min(indicator_full["cum_return"] * 100) * 1.1, max(indicator_full["cum_return"] * 100) * 1.1])
        )
        #st.plotly_chart(fig3, use_container_width=True)

    # Display total return
    st.write(f"Total Strategy Return: {total_return:.2f}%")

    print("hi")
