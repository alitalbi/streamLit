import pandas as pd
from datetime import datetime
import yfinance as yf
from fredapi import Fred
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import os
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="RV Duration")
st.sidebar.header("Real Value & Duration Framework")
#path =os.cwd()
fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')

date_start = "2010-01-01"
date_end = datetime.now().strftime("%Y-%m-%d")

st.title("A framework for RV in duration")
def quantiles_(data):
    # Compute the 5 quantiles
    quantiles = np.percentile(data, [0, 25, 50, 75, 100])

    # Print the quantiles
    print("Quantiles: ", quantiles)

    # Determine which quantile the last data point belongs to
    last_data_point = data.iloc[-1,0]
    if last_data_point <= quantiles[1]:
        return 1
    elif last_data_point <= quantiles[2]:
        return 2
    elif last_data_point <= quantiles[3]:
        return 3
    elif last_data_point <= quantiles[4]:
        return 4
    else:
        return 5

def dot(data1,data2):
    return [data1[i]*data2[i] for i in range(len(data1))]
def filter_color(val):
    if val == 1:
        return 'background-color: rgba( 220, 20, 60, 1 )'
    elif val == 2:
        return 'background-color: rgba( 240, 128, 128, 1 )'
    elif val == 3:
        pass
    elif val == 4:
        return 'background-color: rgba( 152, 251, 152, 1 )'
    elif val == 5:
        return 'background-color: rgba( 124, 252, 0, 1 )'
#Citi Surprise Index
US_citi_surprise_index = pd.read_csv("https://raw.githubusercontent.com/alitalbi/streamLit/master/data/EU_citi_surprise_index.csv",skiprows=[0],index_col=['Date'])
#EU_citi_surprise_index = pd.read_csv(path + "EU_citi_surprise_index.csv",skiprows=[0],index_col=['Date'])

#10 US treasury future price : Treasury Yield 10 Years (^TNX)
zscore_citi_surprise = stats.zscore(US_citi_surprise_index)

#Bond Momentum : 1M change in 10Y future contract

#US_10Y = pd.read_csv(path + "US 10 Year T-Note Futures Historical Data.csv",index_col=["Date"])['Price'][::-1].apply(lambda x:float(x.replace(",",".")))
_10Y_T_Note = yf.download("^TNX", start=date_start, end=date_end, interval="1d")[['Close']]
US_10Y = yf.download("ZN=F", start=date_start, end=date_end, interval="1d")[['Close']]
_1m_momentum_US_10Y = US_10Y - US_10Y.shift(22)
_1m_momentum_US_10Y.dropna(inplace=True)

zscore_momentum_10y = stats.zscore((_1m_momentum_US_10Y))

#Bund = pd.read_csv(path + "Futures Euro Bund - Données historiques.csv",index_col=["Date"])['Dernier'][::-1].apply(lambda x:float(x.replace(",",".")))
#UK_Gilt = pd.read_csv(path + "UK Gilt Futures Historical Data.csv",index_col=["Date"])['Price'][::-1].apply(lambda x:float(x.replace(",",".")))
#Euro_BTP = pd.read_csv(path + "Euro BTP Futures Historical Data.csv",index_col=["Date"])['Price'][::-1].apply(lambda x:float(x.replace(",",".")))
#je dois juste calcluer le momentum là pour ceux là

#Equity Momentum : 1M Change

SP500 = yf.download("^GSPC", start=date_start, end=date_end, interval="1d")[['Close']]
_1m_momentum_SP = SP500 - SP500.shift(22)
_1m_momentum_SP.dropna(inplace=True)
z_score_momentum_SP = (stats.zscore(_1m_momentum_SP))


#Value

ticker_real_gdp_us = 'GDPC1'
infla_breakeven = "T10YIE"
frequency = "monthly"
#importing data
real_gdp_us = pd.DataFrame(
    fred.get_series(ticker_real_gdp_us, observation_start=date_start, observation_end=date_end, freq=frequency))
infla_breakeven_us = pd.DataFrame(
    fred.get_series(infla_breakeven, observation_start=date_start, observation_end=date_end, freq=frequency))

real_gdp_us_ch = real_gdp_us.pct_change()
#real gdp data ends 2022-10-01
#infla breakeven doesnt have 2022-10-01 point but start the month at day 3
infla_breakeven_us.dropna(inplace=True)

gdp_infla_breakeven = (real_gdp_us_ch+infla_breakeven_us)
gdp_infla_breakeven.columns = ['Close']
gdp_infla_breakeven.dropna(inplace=True)
FV = _10Y_T_Note - gdp_infla_breakeven
FV.dropna(inplace=True)
FV.plot()
plt.show()
zscore_FV = stats.zscore(FV)

#Carry spread 3M and 10Y
_3M_US_Bill= yf.download("^IRX", start=date_start, end=date_end, interval="1d")[['Close']]
carry = _10Y_T_Note - _3M_US_Bill
zscore_carry = stats.zscore(carry)

#list quantiles strategies
list_q_strategies = list(map(quantiles_,[zscore_citi_surprise,zscore_momentum_10y,z_score_momentum_SP,zscore_FV,zscore_carry]))

weights = [0.18,0.18,0.18,0.28,0.18]
list_q_strategies.append(sum(dot(weights,list_q_strategies)))
score_table_merged = pd.DataFrame({"Strategy":["Macro Surprise","Bond Momentum","Equity Momentum","Value","Carry","Total Score"],"US":list_q_strategies})

st.table(score_table_merged.style.applymap(filter_color,subset=['US']))

fig = make_subplots(rows=3, cols=2,subplot_titles=["US Citi Surprise Index","Bond Momentum (1M on 10 YTN)","Equity Momentum (1M on S&P)","Value","Carry"])

fig.add_trace(go.Scatter(x=US_citi_surprise_index.index.to_list(), y=US_citi_surprise_index.iloc[:,0], name="US Citi Surprise Index",
                                          mode="lines", line=dict(width=2, color='white'), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=US_citi_surprise_index.index.to_list(), y=_10Y_T_Note.iloc[:,0], name="10Y T Note",
                                          mode="lines", line=dict(width=2, color='white'), showlegend=False), row=1, col=1)
fig.add_trace(
    go.Scatter(x=_1m_momentum_US_10Y.index.to_list(), y=_1m_momentum_US_10Y.iloc[:, 0], name="Bond Momentum (1M on 10 YTN)",
               mode="lines", line=dict(width=2, color='green'),showlegend=False), row=1, col=2)
fig.add_trace(
    go.Scatter(x=_1m_momentum_SP.index.to_list(), y=_1m_momentum_SP.iloc[:,0], name="Equity Momentum (1M on S&P)",
               mode="lines", line=dict(width=2, color='orange'), showlegend=False), row=2, col=1)
fig.add_trace(
    go.Scatter(x=FV.index.to_list(), y=FV.iloc[:, 0], name="Value",
               mode="lines", line=dict(width=2, color='green'), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=carry.index.to_list(), y=carry.iloc[:,0], name="Carry",
                                          mode="lines", line=dict(width=2, color='purple'), showlegend=False), row=3, col=1)


fig.update_layout(
    template="plotly_dark",
    title={
        'text': "Inflation Outlook",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig.update_layout(  # customize font and legend orientation & position
    title_font_family="Arial Black",
    font=dict(
        family="Rockwell",
        size=16),
    legend=dict(
        title=None, orientation="h", y=0.97, yanchor="bottom", x=0.5, xanchor="center"
    )
)
fig.update_layout(height=650, width=1500)

st.plotly_chart(fig, use_container_width=True)


