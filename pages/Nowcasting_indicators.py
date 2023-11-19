import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import backtesting
from backtesting import Backtest
from datetime import datetime,timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backtesting.test import SMA
from backtesting.lib import SignalStrategy, TrailingStrategy
from backtesting import backtesting as backt
import streamlit.components.v1 as components
date_start = "2002-01-01"
date_end = datetime.today().strftime("%Y-%m-%d")


class SmaCross(SignalStrategy,
               TrailingStrategy):
    n1 = 10
    n2 = 25

    def init(self):
        # In init() and in next() it is important to call the
        # super method to properly initialize the parent classes
        super().init()

        # Precompute the two moving averages
        sma1 = self.I(SMA, self.data.Close, self.n1)
        sma2 = self.I(SMA, self.data.Close, self.n2)

        # Where sma1 crosses sma2 upwards. Diff gives us [-1,0, *1*]
        signal = (pd.Series(sma1) > sma2).astype(int).diff().fillna(0)
        signal = signal.replace(-1, 0)  # Upwards/long only

        # Use 95% of available liquidity (at the time) on each order.
        # (Leaving a value of 1. would instead buy a single share.)
        entry_size = signal * .95

        # Set order entry sizes using the method provided by
        # `SignalStrategy`. See the docs.
        self.set_signal(entry_size=entry_size)

        # Set trailing stop-loss to 2x ATR using
        # the method provided by `TrailingStrategy`
        self.set_trailing_sl(2)
def import_data(url):
    df = pd.read_csv(url)
    df.set_index('As Of Date', inplace=True)
    df.drop(['Time Series'], axis=1, inplace=True)
    return df

def export_yfinance_data(ticker):
    price_df = yf.download(ticker,start =date_start ,end =date_end,interval="1d")
    return price_df

ticker = st.selectbox("Ticker",("UST Bond","2y T-Note","5y T-Note","10y T-Note"))

mapping_dict_ticker  = {"UST Bond":"ZB=F",
                        "2y T-Note":"ZT=F",
                        "5y T-Note":"ZF=F",
                        "10y T-Note":"ZN=F"}

fig_ = go.Figure()
if ticker:
    price_df = export_yfinance_data(mapping_dict_ticker[ticker])
    price_df.drop("Adj Close",axis=1,inplace=True)
    print(price_df)
    fig_.add_trace(go.Scatter(x=price_df.index.to_list(),
                          y=price_df["Close"].to_list(),
                          name=ticker,
                          mode="lines", line=dict(width=2, color='white')))

    st.plotly_chart(fig_, use_container_width=True)
    fig_.update_layout(xaxis=dict(rangeselector=dict(font=dict(color="black"))))
    bt = Backtest(price_df,SmaCross,commission=0.002)

    bt.run()
    bt.plot(open_browser=False)

    HtmlFile = open("C:/Users/Administrateur/PycharmProjects/macro/SmaCross.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code,width=800,height=900)
"""
url_tbills_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDPOSGS-B.csv"
url_tbills_2022 ="https://markets.newyorkfed.org/api/pd/get/SBN2022/timeseries/PDPOSGS-B.csv"

url_coupons_2y_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDPOSGSC-L2.csv"
url_coupons_2y_2022 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDPOSGSC-L2.csv"

tbills_2015 = import_data(url_tbills_2015)
tbills_2022 = import_data(url_tbills_2022)

coupons_2y_2015 = import_data(url_coupons_2y_2015)
coupons_2y_2022 = import_data(url_coupons_2y_2022)

tbills_df = pd.concat([tbills_2015,tbills_2022])

coupons_2y_df = pd.concat([coupons_2y_2015,coupons_2y_2022])
tbills_df.plot()
##US Treasury (Excluding TIPS)
plt.title("T-Bills Net positioning")
plt.show()
coupons_2y_df.plot()
plt.title("Net positioning Coupons 2y ")
plt.show()
"""
