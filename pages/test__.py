import backtesting.backtesting
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from backtesting import Backtest
from datetime import datetime,timedelta
import streamlit as st
import plotly.graph_objects as go
from backtesting.test import SMA
import numpy as np
from backtesting.lib import SignalStrategy, TrailingStrategy
#from streamLit.Indicators import indicators
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
date_start = "2002-01-01"
date_end = datetime.today().strftime("%Y-%m-%d")
n1 = 10
n2 = 25
class SmaCross(SignalStrategy,
               TrailingStrategy):


    def init(self):
        # In init() and in next() it is important to call the
        # super method to properly initialize the parent classes
        super().init()

        # Precompute the two moving averages
        sma1 = self.I(SMA, self.data.Close, n1)
        sma2 = self.I(SMA, self.data.Close, n2)

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



class Indicator:
    def __init__(self, data):
        self.data = data

    def simple_moving_average(self, window):
        self.data["sma"+str(window)] = self.data['Close'].rolling(window=window).mean()

    def exponential_moving_average(self, span):
        self.data["ema"+str(span)] =  self.data['Close'].ewm(span=span, adjust=False).mean()

    def hull_moving_average(self, window):
        half_window = int(window / 2)
        weighted_moving_avg = 2*self.data['Close'].rolling(window=half_window, min_periods=1).mean() - self.data['Close'].rolling(window=window, min_periods=1).mean()
        sqrt_window = int(np.sqrt(window))
      # wma_sqrt_window = weighted_moving_avg.rolling(window=sqrt_window, min_periods=1).mean()
        hull_moving_avg = weighted_moving_avg.rolling(window=sqrt_window, min_periods=1).mean()
        self.data["hma" + str(window)] = hull_moving_avg
        print(self.data)

    def relative_strength_index(self, window):
        price_diff = self.data['Close'].diff(1)
        gain = price_diff.where(price_diff > 0, 0)
        loss = -price_diff.where(price_diff < 0, 0)

        average_gain = gain.rolling(window=window).mean()
        average_loss = loss.rolling(window=window).mean()

        rs = average_gain / average_loss
        rsi = 100 - (100 / (1 + rs))

        self.data["rsi"+str(window)] = rsi

class Strategy:
    def __init__(self,series1=None, series2=None):

        self.series1 = series1
        self.series2 = series2

    def crossover(self, series1, series2, direction):
        if direction == "above":
            return np.where(series1 > series2, 1, -1)
        elif direction == "below":
            return np.where(series1 < series2, 1, -1)
        else:
            raise ValueError("Invalid direction. Use 'above' or 'below'.")
class pnl:
    def __init__(self):
        self.pnl = None
        self.volatility = None

    def compute_pnl(self,trade_log):
        trade_log["pq_acc"], trade_log["q_acc"] = abs((trade_log.Close * trade_log.q)).cumsum(), abs(
            trade_log["q"]).cumsum()
        trade_log["avg_price"] = trade_log["pq_acc"] / trade_log["q_acc"]
        trade_log["pnl"] = np.zeros(len(trade_log))
        for index in range(len(trade_log)):
            if index == 0:
                continue
            trade_log["pnl"][index] = (trade_log["avg_price"][index - 1] - trade_log["Close"][index]) * trade_log.q[
                index]
        return trade_log

    def risk_metrics(self,trade_log):
        trades = [trade for trade in range(1, len(trade_log) + 1)]
        trade_log["price_std"] = ((trade_log["Close"] - trade_log["avg_price"]) ** 2).cumsum() / trades
        trade_log["pnl_std"] = trade_log["pnl"].cumsum() / trades
        self.volatility = trade_log.pnl_std
def import_data(url):

    """
        HtmlFile = open("C:/Users/Administrateur/PycharmProjects/macro/SmaCross.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code,width=800,height=900)

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

    df = pd.read_csv(url)
    df.set_index('As Of Date', inplace=True)
    df.drop(['Time Series'], axis=1, inplace=True)
    return df

def export_yfinance_data(ticker):
    price_df = yf.download(ticker,start =date_start,end =date_end,interval="1d")
    return price_df



mapping_dict_ticker  = {"UST Bond":"ZB=F",
                        "2y T-Note":"ZT=F",
                        "5y T-Note":"ZF=F",
                        "10y T-Note":"ZN=F"}

fig_ = go.Figure()
fig_signal = go.Figure()
fig_pnl = go.Figure()
c1, c2, c3 = st.columns(3)
with c1:
    ticker = st.selectbox("Ticker", ("UST Bond", "2y T-Note", "5y T-Note", "10y T-Note"))
with c2:
    date_start_ = st.date_input("Start date:", pd.Timestamp("2023-01-01"))
with c3:
    date_end = st.date_input("End date:", pd.Timestamp(datetime.now().strftime("%Y-%m-%d")))

if ticker :

    price_df = export_yfinance_data(mapping_dict_ticker[ticker])
    price_df.drop("Adj Close",axis=1,inplace=True)
    indicator = Indicator(price_df)


    c1,c2 = st.columns(2)
    with c1:
        indicators = st.selectbox("Indicator", ("sma", "ema", "hma","rsi"))
    with c2:
        period = st.selectbox("Period", ("14", "20","50","240"))


    # Example of calculating indicators

    for time_period in [14,20,50,240]:
        indicator.simple_moving_average(window=time_period)
        indicator.exponential_moving_average(span=time_period)
        indicator.hull_moving_average(window=time_period)
        indicator.relative_strength_index(window=time_period)
        data_output = indicator.data

    data_output = data_output.loc[data_output.index >= pd.to_datetime(date_start_)]

    fig_.add_trace(go.Scatter(x=data_output.index.to_list(),
                              y=data_output["Close"].to_list(),
                              name=ticker,
                              mode="lines", line=dict(width=2, color='white')))

    fig_.add_trace(go.Scatter(x=data_output.index.to_list(),
                              y=data_output[indicators+period].to_list(),
                              name=indicators+period,
                              mode="lines", line=dict(width=2, color='orange')))
    st.plotly_chart(fig_, use_container_width=True)
    fig_.update_layout(xaxis=dict(rangeselector=dict(font=dict(color="black"))))
    fig_.layout.xaxis.range = [date_start_, date_end]
    strat = Strategy()
    c1, c2,c3,c4 = st.columns(4)
    with c1:
        cross = st.selectbox("Strategy1", ("Crossover",""))
    with c2:
        indicator = st.selectbox("Indicators", ("sma", "ema", "hma", "rsi"))
    with c3:
        period1 = st.selectbox("Period1", ("14", "20","50","240"))
    with c4:
        period2 = st.selectbox("Period2", ("14", "20","50","240"),index=1)
    strategy = strat.crossover(data_output[indicator + period1], data_output[indicator + period2], "above")
    if indicator == "rsi":
        strategy = strat.crossover(data_output["rsi"+period1],70,"above")
        data_output["signal"] = strategy
    data_output["signal"] = strategy

    print(":)")



    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cross2 = st.selectbox("Strategy2", ("Crossover", ""))
    with c2:
        indicator2 = st.selectbox("Indicators2", ("sma", "ema", "hma", "rsi"))
    with c3:
        period1_ = st.selectbox("Period1_", ("14", "20", "50", "240"))
    with c4:
        period2_ = st.selectbox("Period2_", ("14", "20", "50", "240"), index=1)

    strategy2 = strat.crossover(data_output[indicator2 + period1_], data_output[indicator2 + period2_], "above")
    if indicator2 == "rsi":
        strategy2 = strat.crossover(data_output["rsi" + period1_], 70, "above")
        data_output["signal2"] = strategy2
        #h
    data_output["signal2"] = strategy2
    sum_signal = data_output["signal"]+data_output["signal2"]
    data_output["agg_signal"] = np.where(sum_signal>0,1,np.where(sum_signal==0,0,-1))
    data_output["q"] = data_output["agg_signal"] * (data_output["Volume"] * 10e-4).apply(lambda x: int(x))


    computePnL = pnl()
    data_output = computePnL.compute_pnl(data_output)
    data_output["color_pnl"] = np.where(data_output["pnl"]<0, 'red', 'green')
    data_output["color_pos"] = np.where(data_output["q"] < 0, 'red', 'green')
   # data_output = data_output.loc[data_output.index >= pd.to_datetime(date_start_)]
    fig = make_subplots(rows=3,
                        cols=2,
                        subplot_titles=('Signal','Pos', 'Pnl Per Trade','Acc_PnL','Total Pos','Price'))
    fig.add_trace(go.Scatter(x=data_output.index.to_list(),
                               y=data_output["agg_signal"].to_list(),
                               name="Strat1+2",
                               mode="lines", line=dict(width=2, color='white')),row=1,col=1)
    fig.add_trace(go.Bar(x=data_output.index.to_list(),
                                 y=data_output["q"].to_list(),
                                 name="Pos per Trade",marker_color=data_output["color_pos"]),row=1,col=2)
    fig.add_trace(go.Bar(x=data_output.index.to_list(),
                             y=data_output["pnl"],
                             name="PnL per Trade",marker_color=data_output["color_pnl"]), row=2, col=1)
    fig.add_trace(go.Scatter(x=data_output.index.to_list(),
                             y=data_output["pnl"].cumsum(),
                             name="total pnl",
                             mode="lines", line=dict(width=2, color='white')), row=2, col=2)
    fig.add_trace(go.Scatter(x=data_output.index.to_list(),
                             y=data_output["q"].cumsum(),
                             name="total pos",
                             mode="lines", line=dict(width=2, color='white')), row=3, col=1)
    fig.add_trace(go.Scatter(x=data_output.index.to_list(),
                             y=data_output["Close"],
                             name="Close Price",
                             mode="lines", line=dict(width=2, color='white')), row=3, col=2)
    fig.update_layout(width=1200,height=900)
    fig.layout.xaxis.range = [date_start_, date_end]
    st.plotly_chart(fig, use_container_width=True)

    #bt = Backtest(price_df,SmaCross,commission=0.002)

    #bt.run()
    #st.bokeh_chart(bt.plot(open_browser=False))
