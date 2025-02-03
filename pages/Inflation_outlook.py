import streamlit as st
import pandas as pd
from fredapi import Fred
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wget
import yfinance as yf
from datetime import datetime,timedelta
import os
from pandas.tseries.offsets import BDay



st.set_page_config(page_title="Inflation",layout="wide",initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    .top-nav {
        display: flex;
        justify-content: center;
        background-color: #333;
        padding: 10px;
    }
    .top-nav a {
        color: white;
        text-decoration: none;
        font-size: 18px;
    }
    .top-nav a:hover {
        color: #ddd;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
    /* Style for the top navigation bar */
    .top-nav {
        background-color: #3B3B3B;
        padding: 15px 0;
        text-align: center;
        display: flex;
        justify-content: center;  /* Center items */
        align-items: center;
    }

    /* Style for the menu links */
    .top-nav a {
        text-decoration: none;
        color: white;
        font-size: 18px;  /* Slightly larger text for better visibility */
        padding: 8px 18px;  /* Adjusted padding to make links tighter */
        margin: 0 10px;  /* Added margin between the links */
        border-radius: 25px;  /* More rounded links */
        transition: background-color 0.3s ease;
    }

    /* Hover effect for the menu items */
    .top-nav a:hover {
        background-color: #4C4C4C;
    }

    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <div class="top-nav">
        <a href="/" target="_self">Home</a>
        <a href="/growth" target="_self">Growth</a>
        <a href="/Inflation_outlook" target="_self">Inflation</a>
        <a href="/Risk_on_off" target="_self">Risk On/Off</a>
        <a href="/Sector_Business_Cycle" target="_self">Business Cycle</a>
        <a href="/Primary_Dealer" target="_self">Primary Dealer</a>
    </div>
""", unsafe_allow_html=True)
frequency = "monthly"
fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')

col1,col2 = st.columns(2,gap="small")
with col1:
    custom_date = st.checkbox("Use Custom Date")
with col2:
    mode = st.checkbox("Smoothen Data")
date_start = pd.Timestamp(datetime.now() +BDay(-3650))
data_displayed = pd.Timestamp(datetime.now()+BDay(-365))
date_start2 = datetime.strptime("2004-01-01","%Y-%m-%d").date()
date_end =pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))
if custom_date:
    date_start_custom = st.date_input("Start date:", pd.Timestamp("2021-01-01"))  
    data_displayed = pd.Timestamp(date_start_custom)
    

def score_table(index, data_, data_10):
    bool_values = {True:1,False:0}
    score_table = pd.DataFrame.from_dict({"trend vs history ": bool_values[data_["_6m_smoothing_growth"][-1] > data_10["10 yr average"][-1]],
                                          "growth": bool_values[data_["_6m_smoothing_growth"][-1] > 0],
                                          "Direction of Trend": bool_values[data_["_6m_smoothing_growth"].diff()[-1]>0]}, orient="index").T
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

def smooth_data(internal_ticker, date_start, date_start2, date_end,mode):
    date_start= (datetime.strptime(date_start,"%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
    #print(date_start)
    data_ = pd.DataFrame(
        fred.get_series(internal_ticker, observation_start=date_start, observation_end=date_end, freq="monthly"))


    data_ = data_.loc[(data_.index > date_start) & (data_.index < date_end)]
    data_.index = pd.to_datetime(data_.index)

    data_2 = pd.DataFrame(
        fred.get_series(internal_ticker, observation_start=date_start2, observation_end=date_end, freq="monthly"))


    data_2 = data_2.loc[(data_2.index > date_start2) & (data_2.index < date_end)]
    data_2.index = pd.to_datetime(data_2.index)
    # creating 6m smoothing growth column and 10 yr average column
    # Calculate the smoothed average
    
    if mode == 0:
    # Calculate the annualized growth rate
        annualized_3m_smoothed_growth_rate = (1+data_.pct_change(3)) ** 4 - 1
        annualized_6m_smoothed_growth_rate = (1+data_.pct_change(6)) ** 2 - 1
        annualized_12m_smoothed_growth_rate = (1+data_.pct_change(12)) ** 1 - 1
        # Multiply the result by 100 and store it in the _6m_smoothing_growth column
        data_['_3m_smoothing_growth'] =  100 * annualized_3m_smoothed_growth_rate
        data_['_6m_smoothing_growth'] = 100 * annualized_6m_smoothed_growth_rate
        data_['_12m_smoothing_growth'] = 100 * annualized_12m_smoothed_growth_rate
        data_2['_3m_smoothing_growth'] =  100 * annualized_3m_smoothed_growth_rate
        data_2['_6m_smoothing_growth'] = 100 * annualized_6m_smoothed_growth_rate
        data_2['_12m_smoothing_growth'] = 100 * annualized_12m_smoothed_growth_rate
        data_2['10 yr average'] = data_2['_6m_smoothing_growth'].rolling(120).mean() 
    else:
        smoothed_3m = data_.iloc[:, 0].rolling(3).mean()
        smoothed_6m = data_.iloc[:, 0].rolling(6).mean()
        smoothed_12m = data_.iloc[:, 0].rolling(12).mean()

        # Calculate the annualized growth rate
        annualized_3m_smoothed_growth_rate = (data_.iloc[:,0] / smoothed_3m) ** 4 - 1
        annualized_6m_smoothed_growth_rate = (data_.iloc[:,0] / smoothed_6m) ** 2 - 1
        annualized_12m_smoothed_growth_rate = (data_.iloc[:,0] / smoothed_12m) - 1
        # Multiply the result by 100 and store it in the _6m_smoothing_growth column
        data_['_3m_smoothing_growth'] =  100 * annualized_3m_smoothed_growth_rate
        data_['_6m_smoothing_growth'] = 100 * annualized_6m_smoothed_growth_rate
        data_['_12m_smoothing_growth'] = 100 * annualized_12m_smoothed_growth_rate
        data_2['_3m_smoothing_growth'] =  100 * annualized_3m_smoothed_growth_rate
        data_2['_6m_smoothing_growth'] = 100 * annualized_6m_smoothed_growth_rate
        data_2['_12m_smoothing_growth'] = 100 * annualized_12m_smoothed_growth_rate
        data_2['10 yr average'] = data_2['_6m_smoothing_growth'].rolling(120).mean() 

    data_.dropna(inplace=True)
    data_2.dropna(inplace=True)
    
    return data_,data_2


def commo_smooth_data(internal_ticker, date_start, date_start2, date_end):
    date_start = (datetime.strptime(date_start, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")

    data_ = yf.download(internal_ticker, start=date_start, end=date_end, interval="1d")[['Close']]

    data_ = data_.loc[(data_.index > date_start) & (data_.index < date_end)]
    data_.index = pd.to_datetime(data_.index).tz_localize(None)

    # creating 6m smoothing growth column and 10 yr average column
    # Calculate the smoothed average
    data_["average"] = data_.rolling(66).mean()

    data_["14d_growth_rate"] = ((data_["Close"].iloc[:,0] / data_["average"]) ** (252 / 66) - 1) * 100
  
    data_["28d_ma"] = data_["Close"].rolling(28).mean()
    data_["100d_ma"] = data_["Close"].rolling(100).mean()

    data_["growth_daily"] = data_["Close"].pct_change(periods=5)


    data_.dropna(inplace=True)
   # data_2.dropna(inplace=True)
    return data_[["100d_ma",'Close',"28d_ma"]]

date_start = date_start.strftime("%Y-%m-%d")
date_start2 = date_start2.strftime("%Y-%m-%d")
date_end = date_end.strftime("%Y-%m-%d")
wheat= commo_smooth_data("ZW=F", date_start,date_start2,date_end)
gas= commo_smooth_data("NG=F", date_start,date_start2,date_end)
oil= commo_smooth_data("CL=F", date_start,date_start2,date_end)
cooper_prices= commo_smooth_data("HG=F", date_start,date_start2,date_end)

# Data importing
employment_level, employment_level_10 = smooth_data("CE16OV", date_start, date_start2, date_end,mode)
employment_level.dropna(inplace=True)

pcec96, pcec96_10 = smooth_data("PCEC96", date_start, date_start2, date_end,0)
cpi, cpi_10 = smooth_data("CPIAUCSL", date_start, date_start2, date_end,0)
# single_search, single_search_10 = smooth_data(ticker_fred, date_start, date_start2, date_end,mode)
shelter_prices, shelter_prices_10 = smooth_data("CUSR0000SAH1", date_start, date_start2, date_end,mode)
shelter_prices["_3m_smoothing_growth"],shelter_prices["_6m_smoothing_growth"],shelter_prices["_12m_smoothing_growth"] = shelter_prices[["_3m_smoothing_growth"]] - cpi[["_3m_smoothing_growth"]],shelter_prices[["_6m_smoothing_growth"]] - cpi[["_6m_smoothing_growth"]],shelter_prices[["_12m_smoothing_growth"]] - cpi[["_12m_smoothing_growth"]]
shelter_prices_10 = shelter_prices_10[['10 yr average']] - cpi_10[['10 yr average']]


wages, wages_10 = smooth_data("CES0500000003", date_start, date_start2, date_end,mode)
wages["_3m_smoothing_growth"] = wages["_3m_smoothing_growth"]- cpi["_3m_smoothing_growth"]
wages["_6m_smoothing_growth"] = wages["_6m_smoothing_growth"]- cpi["_6m_smoothing_growth"]
wages["_12m_smoothing_growth"] = wages["_12m_smoothing_growth"]- cpi["_12m_smoothing_growth"]
wages_10 = wages_10[['10 yr average']] - cpi_10[['10 yr average']]

core_cpi, core_cpi_10 = smooth_data("CPILFESL", date_start, date_start2, date_end,0)
core_pce, core_pce_10 = smooth_data("PCEPILFE", date_start, date_start2, date_end,0)
shelter_title = "Shelter Prices"
wages_title = " Fred Wages"

score_table_merged_infla = pd.concat([
                                      score_table("CPI", cpi, cpi_10),
                                      score_table("Core CPI", core_cpi, core_cpi_10),
                                      score_table("PCE", pcec96, pcec96_10),
                                      score_table("Core PCE", core_pce, core_pce_10),
                                      score_table("Shelter Prices", shelter_prices, shelter_prices_10),
                                      score_table("Wages", wages, wages_10),
                                      score_table("Employment", employment_level, employment_level_10)], axis=0)

score_table_merged_infla = score_table_merged_infla.iloc[:, [4, 0, 1, 2, 3]]
score_table_merged_infla.reset_index(drop=True,inplace=True)




fig_secular_trends = make_subplots(rows=4, cols=2, specs=[[{"secondary_y": True}, {"secondary_y": True}],
                                                          [{"secondary_y": True}, {"secondary_y": True}],
                                                          [{"secondary_y": True}, {"secondary_y": True}],
                                                          [{"secondary_y": True}, {"secondary_y": True}]],
                                   subplot_titles=["CPI", "Core CPI"
                                       , "PCE", "Core PCE", "Shelter Prices",
                                                   "Employment","Wages"])

fig_secular_trends.add_trace(
    go.Scatter(x=cpi.index.to_list(), y=cpi._3m_smoothing_growth/100, legendgroup="3m growth average",name="3m ann growth",
               mode="lines", line=dict(width=2,color='orange')), secondary_y=False, row=1, col=1)
fig_secular_trends.add_trace(
    go.Scatter(x=cpi.index.to_list(), y=cpi._6m_smoothing_growth/100, legendgroup="6m growth average",name="6m ann growth",
               mode="lines", line=dict(width=2,color='#EF553B')), secondary_y=False, row=1, col=1)
fig_secular_trends.add_trace(
    go.Scatter(x=cpi.index.to_list(), y=cpi._12m_smoothing_growth/100, legendgroup="12m growth average",name="12m ann growth",
               mode="lines", line=dict(width=2,color='red')), secondary_y=False, row=1, col=1)
fig_secular_trends.add_trace(go.Scatter(x=(cpi_10.index.to_list()),
                                        y=(cpi_10['10 yr average']) / 100, mode="lines",
                                        line=dict(width=2, color='green'),
                                        legendgroup="10 yr average", name = "10 yr average"), secondary_y=False, row=1, col=1)

fig_secular_trends.add_trace(
    go.Scatter(x=core_cpi.index.to_list(), y=core_cpi._3m_smoothing_growth/100, legendgroup="3m growth average",
               mode="lines", line=dict(width=2, color='orange'), showlegend=False), secondary_y=False, row=1,
    col=2)
fig_secular_trends.add_trace(
    go.Scatter(x=core_cpi.index.to_list(), y=core_cpi._6m_smoothing_growth/100, legendgroup="6m growth average",
               mode="lines", line=dict(width=2, color='#EF553B'), showlegend=False), secondary_y=False, row=1,
    col=2)
fig_secular_trends.add_trace(
    go.Scatter(x=core_cpi.index.to_list(), y=core_cpi._12m_smoothing_growth/100, legendgroup="12m growth average",
               mode="lines", line=dict(width=2, color='red'), showlegend=False), secondary_y=False, row=1,
    col=2)
fig_secular_trends.add_trace(go.Scatter(x=(core_cpi_10.index.to_list()),
                                        y=core_cpi_10['10 yr average'] / 100,
                                        line=dict(width=2, color='green'), mode="lines",
                                        legendgroup="10 yr average", showlegend=False), secondary_y=False, row=1, col=2)

fig_secular_trends.add_trace(
    go.Scatter(x=pcec96.index.to_list(), y=pcec96._3m_smoothing_growth/100, legendgroup="3m growth average",
               mode="lines", line=dict(width=2, color='orange'), showlegend=False), secondary_y=False, row=2,
    col=1)
fig_secular_trends.add_trace(
    go.Scatter(x=pcec96.index.to_list(), y=pcec96._6m_smoothing_growth/100, legendgroup="6m growth average",
               mode="lines", line=dict(width=2, color='#EF553B'), showlegend=False), secondary_y=False, row=2,
    col=1)
fig_secular_trends.add_trace(
    go.Scatter(x=pcec96.index.to_list(), y=pcec96._12m_smoothing_growth/100, legendgroup="12m growth average",
               mode="lines", line=dict(width=2, color='red'), showlegend=False), secondary_y=False, row=2,
    col=1)
fig_secular_trends.add_trace(go.Scatter(x=(pcec96_10.index.to_list()),
                                        y=pcec96_10['10 yr average'] / 100,
                                        line=dict(width=2, color='green'), mode="lines",
                                        legendgroup="10 yr average", showlegend=False), secondary_y=False, row=2, col=1)

fig_secular_trends.add_trace(
    go.Scatter(x=core_pce.index.to_list(), y=core_pce._3m_smoothing_growth/100, legendgroup="3m growth average",
               mode="lines", line=dict(width=2, color='orange'), showlegend=False), row=2, col=2)
fig_secular_trends.add_trace(
    go.Scatter(x=core_pce.index.to_list(), y=core_pce._6m_smoothing_growth/100, legendgroup="6m growth average",
               mode="lines", line=dict(width=2, color='#EF553B'), showlegend=False), row=2, col=2)
fig_secular_trends.add_trace(
    go.Scatter(x=core_pce.index.to_list(), y=core_pce._12m_smoothing_growth/100, legendgroup="12m growth average",
               mode="lines", line=dict(width=2, color='red'), showlegend=False), row=2, col=2)
fig_secular_trends.add_trace(go.Scatter(x=(core_pce_10.index.to_list()),
                                        y=core_pce_10['10 yr average'] / 100,
                                        line=dict(width=2, color='green'), mode="lines",
                                        legendgroup="10 yr average", showlegend=False), secondary_y=False, row=2, col=2)
fig_secular_trends.add_trace(
        go.Scatter(x=shelter_prices.index.to_list(), y=shelter_prices._3m_smoothing_growth/100,
               legendgroup="3m growth average",
               mode="lines", line=dict(width=2, color='orange'), showlegend=False), secondary_y=False, row=3,
    col=1)
fig_secular_trends.add_trace(
    go.Scatter(x=shelter_prices.index.to_list(), y=shelter_prices._6m_smoothing_growth/100,
               legendgroup="6m growth average",
               mode="lines", line=dict(width=2, color='#EF553B'), showlegend=False), secondary_y=False, row=3,
    col=1)
fig_secular_trends.add_trace(
    go.Scatter(x=shelter_prices.index.to_list(), y=shelter_prices._12m_smoothing_growth/100,
               legendgroup="12m growth average",
               mode="lines", line=dict(width=2, color='red'), showlegend=False), secondary_y=False, row=3,
    col=1)

fig_secular_trends.add_trace(go.Scatter(x=(shelter_prices_10.index.to_list()),
                                        y=shelter_prices_10['10 yr average'] / 100,
                                        line=dict(width=2, color='green'), mode="lines",
                                        legendgroup="10 yr average", showlegend=False), secondary_y=False, row=3, col=1)

fig_secular_trends.add_trace(
    go.Scatter(x=employment_level.index.to_list(),
               y=employment_level._3m_smoothing_growth/100,
               legendgroup="3m growth average",
               mode="lines", line=dict(width=2, color='orange'), showlegend=False), secondary_y=False, row=3,
    col=2)
fig_secular_trends.add_trace(
    go.Scatter(x=employment_level.index.to_list(),
               y=employment_level._6m_smoothing_growth/100,
               legendgroup="6m growth average",
               mode="lines", line=dict(width=2, color='#EF553B'), showlegend=False), secondary_y=False, row=3,
    col=2)
fig_secular_trends.add_trace(
    go.Scatter(x=employment_level.index.to_list(),
               y=employment_level._12m_smoothing_growth/100,
               legendgroup="12m growth average",
               mode="lines", line=dict(width=2, color='red'), showlegend=False), secondary_y=False, row=3,
    col=2)
fig_secular_trends.add_trace(
    go.Scatter(x=employment_level_10.index.to_list(),
               y=employment_level_10["10 yr average"]/100,
               legendgroup="10 yr average",
               mode="lines", line=dict(width=2, color='green'), showlegend=False), secondary_y=False, row=3,
    col=2)

fig_secular_trends.add_trace(
    go.Scatter(x=wages.index.to_list(),
               y=wages._3m_smoothing_growth/100,
               legendgroup="3m growth average",
               mode="lines", line=dict(width=2, color='orange'), showlegend=False), secondary_y=False, row=4,
    col=1)
fig_secular_trends.add_trace(
    go.Scatter(x=wages.index.to_list(),
               y=wages._6m_smoothing_growth/100,
               legendgroup="6m growth average",
               mode="lines", line=dict(width=2, color='#EF553B'), showlegend=False), secondary_y=False, row=4,
    col=1)
fig_secular_trends.add_trace(
    go.Scatter(x=wages.index.to_list(),
               y=wages._12m_smoothing_growth/100,
               legendgroup="12m growth average",
               mode="lines", line=dict(width=2, color='red'), showlegend=False), secondary_y=False, row=4,
    col=1)
fig_secular_trends.add_trace(
    go.Scatter(x=wages_10.index.to_list(),
               y=wages_10["10 yr average"]/100,
               legendgroup="10 yr average",
               mode="lines", line=dict(width=2, color='green'), showlegend=False), secondary_y=False, row=4,
    col=1)
fig_secular_trends.update_layout(template="plotly_dark",
                                 height=1000, width=1500)
fig_secular_trends.update_layout(  # customize font and legend orientation & position
    yaxis=dict(tickformat=".1%"),
    font=dict(
        family="Rockwell",
        size=15),
    legend=dict(
        title="Inflation Indicators", orientation="v", y=0.97, yanchor="bottom", x=0.9, xanchor="left"
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
fig_secular_trends.layout.yaxis11.tickformat = ".2%"
fig_secular_trends.layout.yaxis13.tickformat = ".2%"
fig_secular_trends_2 = make_subplots(rows=2, cols=2)

fig_secular_trends_2.add_trace(go.Scatter(x=wheat.index.to_list(), y=wheat.iloc[:,1], name="Wheat",
                                          mode="lines", line=dict(width=2, color='white'), showlegend=True), row=1, col=1)
fig_secular_trends_2.add_trace(
    go.Scatter(x=wheat.index.to_list(), y=wheat.iloc[:, 0], name="100 MA",
               mode="lines", line=dict(width=2, color='green'),showlegend=True), row=1, col=1)
fig_secular_trends_2.add_trace(
    go.Scatter(x=wheat.index.to_list(), y=wheat.iloc[:, 2], name="28 MA",
               mode="lines", line=dict(width=2, color='lightgreen'),showlegend=False), row=1, col=1)

fig_secular_trends_2.add_trace(
    go.Scatter(x=cooper_prices.index.to_list(), y=cooper_prices.iloc[:,1], name="Cooper",
               mode="lines", line=dict(width=2, color='orange'), showlegend=True), row=1, col=2)
fig_secular_trends_2.add_trace(
    go.Scatter(x=cooper_prices.index.to_list(), y=cooper_prices.iloc[:, 0], name="Cooper 100MA",
               mode="lines", line=dict(width=2, color='green'), showlegend=False), row=1, col=2)
fig_secular_trends_2.add_trace(go.Scatter(x=cooper_prices.index.to_list(), y=cooper_prices.iloc[:,2], name="Copper 28 MA",
                                          mode="lines", line=dict(width=2, color='lightgreen'), showlegend=False), row=1, col=2)
fig_secular_trends_2.add_trace(
    go.Scatter(x=gas.index.to_list(), y=gas.iloc[:, 0],name = "Gas 100 MA",
               mode="lines", line=dict(width=2, color='green'), showlegend=False), row=2, col=1)
fig_secular_trends_2.add_trace(
    go.Scatter(x=gas.index.to_list(), y=gas.iloc[:, 2],name = "Oil 28 MA",
               mode="lines", line=dict(width=2, color='lightgreen'), showlegend=False), row=2, col=1)
fig_secular_trends_2.add_trace(go.Scatter(x=gas.index.to_list(), y=gas.iloc[:,1], name="Gas",
                                          mode="lines", line=dict(width=2, color='purple'), showlegend=True), row=2, col=1)

fig_secular_trends_2.add_trace(
    go.Scatter(x=oil.index.to_list(), y=oil.iloc[:, 0],name = "100 MA",
               mode="lines", line=dict(width=2, color='green'), showlegend=False), row=2, col=2)
fig_secular_trends_2.add_trace(
    go.Scatter(x=oil.index.to_list(), y=oil.iloc[:, 1],name = "Oil",
               mode="lines", line=dict(width=2, color='blue'), showlegend=True), row=2, col=2)
fig_secular_trends_2.add_trace(
    go.Scatter(x=oil.index.to_list(), y=oil.iloc[:, 2],name = "28 MA",
               mode="lines", line=dict(width=2, color='lightgreen'), showlegend=True), row=2, col=2)
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


fig_secular_trends_2.layout.xaxis.range = [data_displayed, date_end]
fig_secular_trends_2.layout.xaxis2.range = [data_displayed, date_end]
fig_secular_trends_2.layout.xaxis3.range = [data_displayed, date_end]
fig_secular_trends_2.layout.xaxis4.range = [data_displayed, date_end]


wheat_displayed  = wheat.loc[(wheat.index > data_displayed) & (wheat.index < date_end)]
gas_displayed  = gas.loc[(gas.index > data_displayed) & (gas.index < date_end)]
oil_displayed  = oil.loc[(oil.index > data_displayed) & (oil.index < date_end)]
copper_displayed  = cooper_prices.loc[(cooper_prices.index > data_displayed) & (cooper_prices.index < date_end)]
fig_secular_trends_2.layout.yaxis.range = [min(wheat_displayed.iloc[:,1]), max(wheat_displayed.iloc[:,1])]
fig_secular_trends_2.layout.yaxis2.range = [min(copper_displayed.iloc[:,1]), max(copper_displayed.iloc[:,1])]
fig_secular_trends_2.layout.yaxis3.range = [min(gas_displayed.iloc[:,1]), max(gas_displayed.iloc[:,1])]
fig_secular_trends_2.layout.yaxis4.range = [min(oil_displayed.iloc[:,1]), max(oil_displayed.iloc[:,1])]
fig_secular_trends.layout.xaxis.range = [data_displayed, date_end]
fig_secular_trends.layout.xaxis2.range = [data_displayed, date_end]
fig_secular_trends.layout.xaxis3.range = [data_displayed, date_end]
fig_secular_trends.layout.xaxis4.range = [data_displayed, date_end]
fig_secular_trends.layout.xaxis5.range = [data_displayed, date_end]
fig_secular_trends.layout.xaxis6.range = [data_displayed, date_end]
fig_secular_trends.layout.xaxis7.range = [data_displayed, date_end]
pcec96_displayed  = pcec96.loc[(pcec96.index > data_displayed) & (pcec96.index < date_end)]
pcec96_10_displayed = pcec96_10.loc[(pcec96_10.index > data_displayed) & (pcec96_10.index < date_end)]

cpi_displayed  = cpi.loc[(cpi.index > data_displayed) & (cpi.index < date_end)]
cpi_10_displayed = cpi_10.loc[(cpi_10.index > data_displayed) & (cpi_10.index < date_end)]

core_cpi_displayed  = core_cpi.loc[(core_cpi.index > data_displayed) & (core_cpi.index < date_end)]
core_cpi_10_displayed = core_cpi_10.loc[(core_cpi_10.index > data_displayed) & (core_cpi_10.index < date_end)]

employment_level_displayed  = employment_level.loc[(employment_level.index > data_displayed) & (employment_level.index < date_end)]

core_pce_displayed  = core_pce.loc[(core_pce.index > data_displayed) & (core_pce.index < date_end)]
core_pce_10_displayed = core_pce_10.loc[(core_pce_10.index > data_displayed) & (core_pce_10.index < date_end)]

shelter_prices_displayed = shelter_prices.loc[(shelter_prices.index > data_displayed) & (shelter_prices.index < date_end)]
shelter_prices_10_displayed = shelter_prices_10.loc[(shelter_prices_10.index > data_displayed) & (shelter_prices_10.index < date_end)]

wages_displayed = wages.loc[(wages.index > data_displayed) & (wages.index < date_end)]
wages_10_displayed = wages_10.loc[(wages_10.index > data_displayed) & (wages_10.index < date_end)]


fig_secular_trends.layout.yaxis.range = [min(min(cpi_10_displayed['10 yr average']),min(cpi_displayed._3m_smoothing_growth),
                                                     min(cpi_displayed._6m_smoothing_growth),
                                                     min(cpi_displayed._12m_smoothing_growth))/100, max(max(pcec96_10_displayed['10 yr average']),max(cpi_displayed._3m_smoothing_growth),
                                                     max(cpi_displayed._6m_smoothing_growth),
                                                     max(cpi_displayed._12m_smoothing_growth))/100]


fig_secular_trends.layout.yaxis3.range = [min(min(core_cpi_10_displayed['10 yr average']),min(core_cpi_displayed._3m_smoothing_growth),
                                                     min(core_cpi_displayed._6m_smoothing_growth),
                                                     min(core_cpi_displayed._12m_smoothing_growth))/100, max(max(core_cpi_10_displayed['10 yr average']),max(core_cpi_displayed._3m_smoothing_growth),
                                                     max(core_cpi_displayed._6m_smoothing_growth),
                                                     max(core_cpi_displayed._12m_smoothing_growth))/100]


fig_secular_trends.layout.yaxis5.range = [min(min(pcec96_10_displayed['10 yr average']),min(pcec96_displayed._3m_smoothing_growth),
                                                     min(pcec96_displayed._6m_smoothing_growth),
                                                     min(pcec96_displayed._12m_smoothing_growth))/100, max(max(pcec96_10_displayed['10 yr average']),max(pcec96_displayed._3m_smoothing_growth),
                                                     max(pcec96_displayed._6m_smoothing_growth),
                                                     max(pcec96_displayed._12m_smoothing_growth))/100]

fig_secular_trends.layout.yaxis7.range = [min(min(core_pce_10_displayed['10 yr average']),min(core_pce_displayed._3m_smoothing_growth),
                                                     min(core_pce_displayed._6m_smoothing_growth),
                                                     min(core_pce_displayed._12m_smoothing_growth))/100, max(max(core_pce_10_displayed['10 yr average']),max(core_pce_displayed._3m_smoothing_growth),
                                                     max(core_pce_displayed._6m_smoothing_growth),
                                                     max(core_pce_displayed._12m_smoothing_growth))/100]
fig_secular_trends.layout.yaxis9.range = [min(min(shelter_prices_10_displayed['10 yr average']),min(shelter_prices_displayed._3m_smoothing_growth),
                                                     min(shelter_prices_displayed._6m_smoothing_growth),
                                                     min(shelter_prices_displayed._12m_smoothing_growth))/100, max(max(shelter_prices_10_displayed['10 yr average']),max(shelter_prices_displayed._3m_smoothing_growth),
                                                     max(shelter_prices_displayed._6m_smoothing_growth),
                                                     max(shelter_prices_displayed._12m_smoothing_growth))/100]
fig_secular_trends.layout.yaxis11.range = [min(min(employment_level_displayed._3m_smoothing_growth),
                                                     min(employment_level_displayed._6m_smoothing_growth),
                                                     min(employment_level_displayed._12m_smoothing_growth))/100, max(max(employment_level_displayed._3m_smoothing_growth),
                                                     max(employment_level_displayed._6m_smoothing_growth),
                                                     max(employment_level_displayed._12m_smoothing_growth))/100]
fig_secular_trends.layout.yaxis13.range = [min(min(shelter_prices_10_displayed['10 yr average']),min(wages_displayed._3m_smoothing_growth),
                                                     min(wages_displayed._6m_smoothing_growth),
                                                     min(wages_displayed._12m_smoothing_growth))/100, max(max(wages_10_displayed['10 yr average']),max(wages_displayed._3m_smoothing_growth),
                                                     max(wages_displayed._6m_smoothing_growth),
                                                     max(wages_displayed._12m_smoothing_growth))/100]

st.plotly_chart(fig_secular_trends_2, use_container_width=True)
st.dataframe(score_table_merged_infla.style.applymap(filter_color,subset=['Score']),hide_index=True,width=700)
st.plotly_chart(fig_secular_trends, use_container_width=True)
st.write("Export Data : ")
col1,col2,col3,col4,col5,col6,col7 = st.columns(7,gap="small")
with col1:
    st.download_button("Infla",data=cpi_10.to_csv().encode("utf-8"),
                   file_name="Infla.csv")
with col2:
    st.download_button("Core Infla",data=core_cpi_10.to_csv().encode("utf-8"),
                   file_name="Core_infla.csv")
with col3:
    st.download_button("PCE",data=pcec96_10.to_csv().encode("utf-8"),
                   file_name="pce.csv")
with col4:
    st.download_button("Core PCE",data=core_pce_10.to_csv().encode("utf-8"),
                   file_name="Core_pce.csv")
with col5:
    st.download_button("Wages",data=wages_10.to_csv().encode("utf-8"),
                   file_name="Wages.csv")
with col6:
    st.download_button("Shelter Prices",data=shelter_prices_10.to_csv().encode("utf-8"),
                   file_name="Shelter_Prices.csv")
with col7:
    st.download_button("Employment",data=employment_level.to_csv().encode("utf-8"),
                   file_name="employment.csv")

