import streamlit as st
import pandas as pd
from fredapi import Fred
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wget
import yfinance as yf
from datetime import datetime
import os

cwd = os.getcwd()
st.set_page_config(page_title="Inflation Outlook",page_icon="📈")
# Set title and description of the app
st.markdown("Inflation Outlook ")
st.sidebar.header("Inflation Outlook ")

frequency = "monthly"
fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')

date_start_ = st.date_input("Start date:", pd.Timestamp("2015-01-01"))

date_start2_ = datetime.strptime("2004-01-01","%Y-%m-%d").date()

date_end_ = st.date_input("End date:", pd.Timestamp("2022-01-01"))


def score_table(index, data_, data_10):
    score_table = pd.DataFrame.from_dict({"trend vs history ": 1 if data_.iloc[-1, 0] > data_10.iloc[-1, 0] else 0,
                                          "growth": 1 if data_.iloc[-1, 0] > 0 else 0,
                                          "Direction of Trend": 1 if data_.diff().iloc[-1][
                                                                         0] > 0 else 0}, orient="index").T
    score_table['Score'] = score_table.sum(axis=1)
    score_table['Indicators'] = index

    return score_table

def filter_color(val):
    print(val, type(val))
    if val == 0:
        return 'background-color: rgba(255, 36, 71, 1)'
    elif val == 1:
        return 'background-color: rgba(255, 36, 71, 0.4)'
    elif val == 2:
        return 'background-color: rgba(53, 108, 0, 1)'
    elif val == 3:
        return 'background-color: rgba(138, 255,0, 1)'
def smooth_data(internal_ticker, date_start, date_start2, date_end):
    data_ = pd.DataFrame(
        fred.get_series(internal_ticker, observation_start=date_start, observation_end=date_end, freq="monthly"))


    data_ = data_.loc[(data_.index > date_start) & (data_.index < date_end)]
    data_.index = pd.to_datetime(data_.index)

    data_2 = pd.DataFrame(
        fred.get_series(internal_ticker, observation_start=date_start2, observation_end=date_end, freq="monthly"))


    data_2 = data_2.loc[(data_2.index > date_start2) & (data_2.index < date_end)]
    data_2.index = pd.to_datetime(data_2.index)
    # creating 6m smoothing growth column and 10yr average column
    # Calculate the smoothed average
    average = data_.iloc[:, 0].rolling(11).mean()
    shifted = data_.iloc[:, 0].shift(11)
    # Calculate the annualized growth rate
    annualized_6m_smoothed_growth_rate = (data_.iloc[11:, 0] / average) ** 2 - 1

    # Multiply the result by 100 and store it in the _6m_smoothing_growth column
    data_['_6m_smoothing_growth'] = 100 * annualized_6m_smoothed_growth_rate
    data_2['mom_average'] = 1000 * data_2.iloc[:, 0].pct_change(periods=1)
    data_2['10 yr average'] = data_2['mom_average'].rolling(120).mean()
    data_.dropna(inplace=True)
    data_2.dropna(inplace=True)
    return data_[['_6m_smoothing_growth']], data_2[['10 yr average']]

def commo_smooth_data(internal_ticker, date_start, date_start2, date_end):
    data_ = yf.download(internal_ticker, start=date_start, end=date_end, interval="1d")[['Close']]

    data_ = data_.loc[(data_.index > date_start) & (data_.index < date_end)]
    data_.index = pd.to_datetime(data_.index)

    #data_2 = yf.download(internal_ticker, start=date_start2, end=date_end, interval="1d")[['Close']]


    # creating 6m smoothing growth column and 10yr average column
    # Calculate the smoothed average
    data_["average"] = data_.rolling(66).mean()
    # average = data_.iloc[:, 0].rolling(22).mean()
    data_["14d_growth_rate"] = ((data_["Close"] / data_["average"]) ** (252 / 66) - 1) * 100
    data_["28d_ma"] = data_["Close"].rolling(28).mean()
    data_["100d_ma"] = data_["Close"].rolling(100).mean()

    data_["growth_daily"] = data_["Close"].pct_change(periods=5)


    data_.dropna(inplace=True)
   # data_2.dropna(inplace=True)
    return data_[["100d_ma",'Close']]

date_start = date_start_.strftime("%Y-%m-%d")
date_start2 = date_start2_.strftime("%Y-%m-%d")
date_end = date_end_.strftime("%Y-%m-%d")
wheat= commo_smooth_data("ZW=F", date_start,date_start2,date_end)
print(wheat)

gas= commo_smooth_data("NG=F", date_start,date_start2,date_end)
print(wheat)

oil= commo_smooth_data("CL=F", date_start,date_start2,date_end)
print(wheat)

cooper_prices= commo_smooth_data("HG=F", date_start,date_start2,date_end)
print(wheat)


# Data importing
employment_level, employment_level_10 = smooth_data("CE16OV", date_start, date_start2, date_end)
employment_level.dropna(inplace=True)

pcec96, pcec96_10 = smooth_data("PCEC96", date_start, date_start2, date_end)

cpi, cpi_10 = smooth_data("CPIAUCSL", date_start, date_start2, date_end)
# single_search, single_search_10 = smooth_data(ticker_fred, date_start, date_start2, date_end)

shelter_prices, shelter_prices_10 = smooth_data("CUSR0000SAH1", date_start, date_start2, date_end)
shelter_prices = shelter_prices[["_6m_smoothing_growth"]] - cpi[["_6m_smoothing_growth"]]
shelter_prices_10 = shelter_prices_10[['10 yr average']] - cpi_10[['10 yr average']]


employment_level_wage_tracker = pd.concat([employment_level], axis=1)
employment_level_wage_tracker.dropna(inplace=True)
wages, wages_10 = smooth_data("CES0500000003", date_start, date_start2, date_end)
wages = wages[["_6m_smoothing_growth"]] - cpi[["_6m_smoothing_growth"]]
wages_10 = wages_10[['10 yr average']] - cpi_10[['10 yr average']]

core_cpi, core_cpi_10 = smooth_data("CPILFESL", date_start, date_start2, date_end)
core_pce, core_pce_10 = smooth_data("DPCCRC1M027SBEA", date_start, date_start2, date_end)

shelter_title = "Shelter Prices"
wages_title = " Fred Wages"

score_table_merged_infla = pd.concat([
                                      score_table("CPI", cpi, cpi_10),
                                      score_table("Core CPI", core_cpi, core_cpi_10),
                                      score_table("PCE", pcec96, pcec96_10),
                                      score_table("Core PCE", core_pce, core_pce_10),
                                      score_table("Shelter Prices", shelter_prices, shelter_prices_10)], axis=0)

score_table_merged_infla = score_table_merged_infla.iloc[:, [4, 0, 1, 2, 3]]
score_table_merged_infla.reset_index(drop=True,inplace=True)
# score_table_merged.set_index("index", inplace=True)



fig_secular_trends = make_subplots(rows=3, cols=2, specs=[[{"secondary_y": True}, {"secondary_y": True}],
                                                          [{"secondary_y": True}, {"secondary_y": True}],
                                                          [{"secondary_y": True}, {"secondary_y": True}]],
                                   subplot_titles=["CPI", "Core CPI"
                                       , "PCE", "Core PCE", "Shelter Prices",
                                                   "Employment (Growth) and Wage tracker levels"])

fig_secular_trends.add_trace(
    go.Scatter(x=cpi.index.to_list(), y=cpi._6m_smoothing_growth/100, name="6m growth average",
               mode="lines", line=dict(width=2, color='white')), secondary_y=False, row=1, col=1)
fig_secular_trends.add_trace(go.Scatter(x=(cpi_10.index.to_list()),
                                        y=(cpi_10['10 yr average']) / 100, mode="lines",
                                        line=dict(width=2, color='green'),
                                        name="10yr average"), secondary_y=False, row=1, col=1)

fig_secular_trends.add_trace(
    go.Scatter(x=core_cpi.index.to_list(), y=core_cpi._6m_smoothing_growth/100, name="6m growth average",
               mode="lines", line=dict(width=2, color='white'), showlegend=False), secondary_y=False, row=1,
    col=2)
fig_secular_trends.add_trace(go.Scatter(x=(core_cpi_10.index.to_list()),
                                        y=core_cpi_10['10 yr average'] / 100,
                                        line=dict(width=2, color='green'), mode="lines",
                                        name="10yr average", showlegend=False), secondary_y=False, row=1, col=2)

fig_secular_trends.add_trace(
    go.Scatter(x=pcec96.index.to_list(), y=pcec96._6m_smoothing_growth/100, name="6m growth average",
               mode="lines", line=dict(width=2, color='white'), showlegend=False), secondary_y=False, row=2,
    col=1)
fig_secular_trends.add_trace(go.Scatter(x=(pcec96_10.index.to_list()),
                                        y=pcec96_10['10 yr average'] / 100,
                                        line=dict(width=2, color='green'), mode="lines",
                                        name="10yr average", showlegend=False), secondary_y=False, row=2, col=1)

fig_secular_trends.add_trace(
    go.Scatter(x=core_pce.index.to_list(), y=core_pce._6m_smoothing_growth/100, name="6m growth average",
               mode="lines", line=dict(width=2, color='white'), showlegend=False), row=2, col=2)
fig_secular_trends.add_trace(go.Scatter(x=(core_pce_10.index.to_list()),
                                        y=core_pce_10['10 yr average'] / 100,
                                        line=dict(width=2, color='green'), mode="lines",
                                        name="10yr average", showlegend=False), secondary_y=False, row=2, col=2)
fig_secular_trends.add_trace(
    go.Scatter(x=shelter_prices.index.to_list(), y=shelter_prices._6m_smoothing_growth/100,
               name="6m growth average",
               mode="lines", line=dict(width=2, color='white'), showlegend=False), secondary_y=False, row=3,
    col=1)
fig_secular_trends.add_trace(go.Scatter(x=(shelter_prices_10.index.to_list()),
                                        y=shelter_prices_10['10 yr average'] / 100,
                                        line=dict(width=2, color='green'), mode="lines",
                                        name="10yr average", showlegend=False), secondary_y=False, row=3, col=1)

fig_secular_trends.add_trace(
    go.Scatter(x=employment_level_wage_tracker.index.to_list(),
               y=employment_level_wage_tracker._6m_smoothing_growth/100,
               name="Employment level 6m annualized growth",
               mode="lines", line=dict(width=2, color='white'), showlegend=True), secondary_y=False, row=3,
    col=2)

fig_secular_trends.update_layout(template="plotly_dark",
                                 height=1000, width=1500)
fig_secular_trends.update_layout(  # customize font and legend orientation & position
    yaxis=dict(tickformat=".1%"),
    title_font_family="Arial Black",
    font=dict(
        family="Rockwell",
        size=15),
    legend=dict(
        title=None, orientation="h", y=1.02, yanchor="bottom", x=0.5, xanchor="center"
    )
)

fig_secular_trends.layout.yaxis.tickformat = ".2%"
fig_secular_trends.layout.yaxis2.tickformat = ".2%"
fig_secular_trends.layout.yaxis3.tickformat = ".2%"
fig_secular_trends.layout.yaxis4.tickformat = ".2%"
fig_secular_trends.layout.yaxis5.tickformat = ".2%"
fig_secular_trends.layout.yaxis6.tickformat = ".2%"
fig_secular_trends.layout.yaxis7.tickformat = ".2%"
fig_secular_trends.layout.yaxis8.tickformat = ".2%"
fig_secular_trends.layout.yaxis9.tickformat = ".2%"
fig_secular_trends.layout.yaxis10.tickformat = ".2%"
fig_secular_trends_2 = make_subplots(rows=2, cols=2)

fig_secular_trends_2.add_trace(go.Scatter(x=wheat.index.to_list(), y=wheat.iloc[:,1], name="Wheat Close Price",
                                          mode="lines", line=dict(width=2, color='white')), row=1, col=1)
fig_secular_trends_2.add_trace(
    go.Scatter(x=wheat.index.to_list(), y=wheat.iloc[:, 0], name="Wheat 100 MA",
               mode="lines", line=dict(width=2, color='green'),showlegend=True), row=1, col=1)
fig_secular_trends_2.add_trace(
    go.Scatter(x=cooper_prices.index.to_list(), y=cooper_prices.iloc[:,1], name="Cooper Close Price",
               mode="lines", line=dict(width=2, color='orange')), row=1, col=2)
fig_secular_trends_2.add_trace(
    go.Scatter(x=cooper_prices.index.to_list(), y=cooper_prices.iloc[:, 0], name="Cooper 100MA",
               mode="lines", line=dict(width=2, color='green'), showlegend=False), row=1, col=2)
fig_secular_trends_2.add_trace(go.Scatter(x=gas.index.to_list(), y=gas.iloc[:,1], name="Gas Close Price",
                                          mode="lines", line=dict(width=2, color='purple')), row=2, col=1)
fig_secular_trends_2.add_trace(
    go.Scatter(x=gas.index.to_list(), y=gas.iloc[:, 0],name = "Gas 100 MA",
               mode="lines", line=dict(width=2, color='green'), showlegend=True), row=2, col=1)

fig_secular_trends_2.add_trace(go.Scatter(x=oil.index.to_list(), y=oil.iloc[:,0], name="Oil 100 MA",
                                          mode="lines", line=dict(width=2, color='blue'), showlegend=False), row=2, col=2)
fig_secular_trends_2.add_trace(
    go.Scatter(x=oil.index.to_list(), y=oil.iloc[:, 1],
               mode="lines", line=dict(width=2, color='green'), showlegend=False), row=2, col=2)

fig_secular_trends_2.update_layout(
    template="plotly_dark",
    title={
        'text': "Inflation Outlook",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig_secular_trends_2.update_layout(  # customize font and legend orientation & position
    title_font_family="Arial Black",
    font=dict(
        family="Rockwell",
        size=16),
    legend=dict(
        title=None, orientation="h", y=0.97, yanchor="bottom", x=0.5, xanchor="center"
    )
)
fig_secular_trends_2.update_layout(height=650, width=1500)


fig_secular_trends_2.layout.xaxis.range = [date_start, date_end]
fig_secular_trends_2.layout.xaxis2.range = [date_start, date_end]
fig_secular_trends_2.layout.xaxis3.range = [date_start, date_end]
fig_secular_trends_2.layout.xaxis4.range = [date_start, date_end]



fig_secular_trends.layout.xaxis.range = [date_start, date_end]
fig_secular_trends.layout.xaxis2.range = [date_start, date_end]
fig_secular_trends.layout.xaxis3.range = [date_start, date_end]
fig_secular_trends.layout.xaxis4.range = [date_start, date_end]
fig_secular_trends.layout.xaxis5.range = [date_start, date_end]
fig_secular_trends.layout.xaxis6.range = [date_start, date_end]



st.table(score_table_merged_infla.style.applymap(filter_color,subset=['Score']))
st.plotly_chart(fig_secular_trends_2, use_container_width=True)
st.plotly_chart(fig_secular_trends, use_container_width=True)
