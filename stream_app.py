import streamlit as st
import pandas as pd
import yfinance as yf


# Set title and description of the app
st.title("Stock Price Analysis")
st.write("A simple app to download and plot stock prices.")

# Set up the search bar and date inputs
search_term = st.text_input("Enter a stock ticker (e.g. AAPL):")
start_date = st.date_input("Start date:", pd.Timestamp("2015-01-01"))
end_date = st.date_input("End date:", pd.Timestamp("2022-01-01"))

# Download the data and plot the close price
if search_term:
    data = yf.download(search_term, start=start_date, end=end_date)
    st.line_chart(data["Close"])
