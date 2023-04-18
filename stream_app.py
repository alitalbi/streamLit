import streamlit as st
import pandas as pd
#import yfinance as yf

def get_sp500_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",flavor="html5lib")
    df = df[0]
    tickers = df["Symbol"].to_list()
    tickers_companies_dict = dict(zip(df["Symbol"], df["Security"]))
    return tickers, tickers_companies_dict

st.title("Stock Data Analysis")
st.write("A simple app to download stock data and apply technical analysis indicators.")

st.sidebar.header("Stock Parameters")

"""available_tickers, tickers_companies_dict = get_sp500_components()

ticker = st.sidebar.selectbox(
    "Ticker", available_tickers, format_func=tickers_companies_dict.get
)

start = st.sidebar.date_input("Start date:", pd.Timestamp("2020-01-01"))
end = st.sidebar.date_input("End date:", pd.Timestamp("2021-12-31"))

#data = yf.download(ticker, start, end)

#selected_indicator = st.selectbox("Select a technical analysis indicator:", indicators)

#indicator_data = apply_indicator(selected_indicator, data)

#st.write(f"{selected_indicator} for {ticker}")
#st.line_chart(indicator_data)

#st.write("Stock data for", ticker)

"""