import streamlit as st
import pandas as pd
import yfinance as yf
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots


#sys.path.insert(1, '/Users/talbi/PycharmProjects/streamLit/venv/lib/python3.10/site-packages')

st.set_page_config(page_title="yahoo finance search",page_icon="📈")
# Set title and description of the app
st.title("Close price from yahoo finance ")
st.write("Talbi & Co Eco Framework (not ESG complaint) ")
st.sidebar.header("Yahoo Finance Stock search")
# Set up the search bar and date inputs
search_term = st.text_input("Enter a stock ticker (e.g. AAPL):")
start_date = st.date_input("Start date:", pd.Timestamp("2015-01-01"))
end_date = st.date_input("End date:", pd.Timestamp("2022-01-01"))
print(start_date)

# Download the data and plot the close price
if search_term:
    data = yf.download(search_term, start=start_date, end=end_date)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index.to_list(), y=data.Close / 100,
                          name=search_term+ " Close Price",
                          mode="lines", line=dict(width=2)))
    st.plotly_chart(fig, use_container_width=True)
