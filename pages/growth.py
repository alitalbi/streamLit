import streamlit as st
import pandas as pd
from fredapi import Fred
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime,timedelta
import requests
from ratelimit import limits
from pandas.tseries.offsets import BDay
st.set_page_config(page_title="growth",layout="wide",initial_sidebar_state="collapsed")

# Navigation function to load content for each page

# Set up the top navigation bar with buttons (no new tabs will open)
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

# Top navigation bar with links to different pages
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
@limits(calls=15,period=900)
def get_cli_data():
    cli = pd.read_csv("https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_CLI,/.M.LI...AA...H?startPeriod="+year+"-"+month+"&dimensionAtObservation=AllDimensions&format=csvfilewithlabels",storage_options=option)
    return cli
def data_smooth(data_,date_start,date_end):
    data_ = data_.loc[(data_.index > date_start) & (data_.index < date_end)]
    #data_.index = pd.to_datetime(data_.index).dt.date()
    # creating 6m smoothing growth column and 10 yr average column
    # Calculate the smoothed average
    average = data_.iloc[:, 0].rolling(11).mean()
    shifted = data_.iloc[:, 0].shift(11)
    # Calculate the annualized growth rate
    annualized_6m_smoothed_growth_rate = (data_.iloc[11:, 0] / average) ** 2 - 1

    # Multiply the result by 100 and store it in the _6m_smoothing_growth column
    data_['_6m_smoothing_growth'] = 100 * annualized_6m_smoothed_growth_rate
    data_.dropna(inplace=True)

    print(data_)
    return data_[['_6m_smoothing_growth']]

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
# Set title and description of the app
retail_sales_title = "Real Retail Sales"
employment_level_title = "Employment Level"
pce_title = "PCE"
indpro_title = "Industrial Production"
nonfarm_title = "NonFarm Payroll"
real_personal_income_title = "Real Personal Income"
cpi_title = "CPI"
core_cpi_title = "Core CPI"
core_pce_title = "Core PCE"


date_start_converted = date_start.strftime("%Y-%m-%d")
date_start2_converted = date_start2.strftime("%Y-%m-%d")
date_end_converted = date_end.strftime("%Y-%m-%d")

pce96,pce96_10 = smooth_data("PCEC96", date_start_converted, date_start2_converted, date_end_converted,0)
indpro,indpro_10 = smooth_data("INDPRO", date_start_converted, date_start2_converted, date_end_converted,mode)
nonfarm,nonfarm_10 = smooth_data("PAYEMS", date_start_converted, date_start2_converted, date_end_converted,mode)
real_pers,real_pers_10 = smooth_data("W875RX1", date_start_converted, date_start2_converted, date_end_converted,mode)
retail_sales,retail_sales_10 = smooth_data("RRSFS", date_start_converted, date_start2_converted, date_end_converted,mode)
employment_level,employment_level_10 = smooth_data("CE16OV", date_start_converted, date_start2_converted, date_end_converted,mode)


composite_data = pd.concat(
    [pce96[['_6m_smoothing_growth']], indpro[['_6m_smoothing_growth']], nonfarm[['_6m_smoothing_growth']],
     real_pers[['_6m_smoothing_growth']], retail_sales[['_6m_smoothing_growth']],
     employment_level[['_6m_smoothing_growth']]], axis=1)
composite_data.dropna(inplace=True)
composite_growth = pd.DataFrame(composite_data.mean(axis=1))
composite_growth.columns = ["_6m_smoothing_growth"]
composite_growth_10 = pd.concat(
    [pce96_10[['10 yr average']], indpro_10[['10 yr average']], nonfarm_10[['10 yr average']],
     real_pers_10[['10 yr average']], retail_sales_10[['10 yr average']],
     employment_level_10[['10 yr average']]],
    axis=1)
composite_growth_10.dropna(inplace=True)
composite_growth_10 = pd.DataFrame(composite_growth_10.mean(axis=1))
composite_growth_10.columns = ["10 yr average"]
url = 'https://www.atlantafed.org/-/media/documents/cqer/researchcq/gdpnow/GDPTrackingModelDataAndForecasts.xlsx'
response = requests.get(url)
atlanta_gdp_now = pd.read_excel(response.content, sheet_name="TrackingArchives", usecols=['Forecast Date','GDP Nowcast'])
atlanta_gdp_now.set_index("Forecast Date",inplace=True,drop=True)
year = str(date_start.year)
month = str(date_start.month)
month = month if len(month)==2 else "0"+month

option = {'User-Agent': 'Mozilla/5.0'}
fig_ = go.Figure()


# cli=get_cli_data()
# cli = cli.loc[cli["REF_AREA"]=="USA"][["TIME_PERIOD","OBS_VALUE"]]
# cli.sort_values(by="TIME_PERIOD",ascending=False,inplace=True)
# cli.set_index("TIME_PERIOD",inplace=True,drop=True)
# cli = (cli - cli.mean())/(cli.std()*100)


# fig_.add_trace(go.Scatter(x=cli.index.to_list(), y=cli.iloc[:,0],
#                           name="OECD CLI",
#                           mode="lines", line=dict(width=2, color='orange')))
fig_.add_trace(go.Scatter(x=composite_growth.index.to_list(), y=composite_growth._6m_smoothing_growth / 100,
                          name="6m growth average",
                          mode="lines", line=dict(width=2, color='white')))
fig_.add_trace(go.Scatter(x=atlanta_gdp_now.index.to_list(), y=atlanta_gdp_now.iloc[:,0]/100,
                          name="Atlanta Fed GDP Nowcast",
                          mode="lines", line=dict(width=2, color='blue')))
fig_.add_trace(go.Scatter(x=composite_growth_10.index.to_list(),
                          y=composite_growth_10['10 yr average'] / 100,
                          name="10 yr average",
                          mode="lines", line=dict(width=2, color='green')))

fig_.update_layout(
    template="plotly_dark",
    title={
        'text': "COMPOSITE GROWTH",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig_.update_layout(xaxis_range = [data_displayed,date_end])
composite_displayed  = composite_growth.loc[(composite_growth.index > data_displayed) & (composite_growth.index < date_end)]
composite_10_displayed = composite_growth_10.loc[(composite_growth_10.index > data_displayed) & (composite_growth_10.index < date_end)]
atlanta_displayed = atlanta_gdp_now.loc[(atlanta_gdp_now.index > data_displayed) & (atlanta_gdp_now.index < date_end)]




qs=atlanta_gdp_now.loc[(atlanta_gdp_now.index > np.datetime64(date_start)) & (atlanta_gdp_now.index < np.datetime64(date_end))]

fig_.update_layout(  # customize font and legend orientation & position
    yaxis=dict(tickformat=".1%",range=[min(min(composite_displayed._6m_smoothing_growth),min(atlanta_displayed.iloc[:,0]),min(composite_10_displayed["10 yr average"]))/100,
                           max(max(composite_displayed._6m_smoothing_growth),max(atlanta_displayed.iloc[:,0]),max(composite_10_displayed["10 yr average"]))/100]),

    title_font_family="Arial Black",
    font=dict(
        family="Rockwell",
        size=16),
    legend=dict(
        orientation="h", y=0.97, yanchor="bottom", x=0.5, xanchor="center"
    ),
)
fig_.update_layout(xaxis=dict(rangeselector=dict(font=dict(color="black"))))
fig_cyclical_trends = make_subplots(rows=3, cols=2, subplot_titles=[pce_title, indpro_title
    , nonfarm_title, real_personal_income_title, retail_sales_title, employment_level_title])
fig_cyclical_trends.add_trace(
    go.Scatter(x=pce96.index.to_list(), y=pce96._3m_smoothing_growth / 100,
               mode="lines", line=dict(width=2,color='orange'),legendgroup="3m ann growth",name="3m ann growth"), row=1, col=1)
fig_cyclical_trends.add_trace(
    go.Scatter(x=pce96.index.to_list(), y=pce96._6m_smoothing_growth / 100,
               mode="lines", legendgroup="6m ann growth", line=dict(width=2,color='#EF553B'),name = "6m ann growth"), row=1, col=1)
fig_cyclical_trends.add_trace(
    go.Scatter(x=pce96.index.to_list(), y=pce96._12m_smoothing_growth / 100,
               mode="lines", legendgroup="12m ann growth", line=dict(width=2,color='red'),name="12m ann growth"), row=1, col=1)
fig_cyclical_trends.add_trace(go.Scatter(x=(pce96_10.index.to_list()),
                                         y=(pce96_10['10 yr average']) / 100, mode="lines",
                                         line=dict(width=2, color='green'),
                                         legendgroup="10 yr average",name="10 yr average"), row=1, col=1)
fig_cyclical_trends.add_trace(
    go.Scatter(x=indpro.index.to_list(), y=indpro._3m_smoothing_growth / 100, legendgroup="3m ann growth",
               mode="lines", line=dict(width=2, color='orange'), showlegend=False), row=1, col=2)
fig_cyclical_trends.add_trace(
    go.Scatter(x=indpro.index.to_list(), y=indpro._6m_smoothing_growth / 100, legendgroup="6m ann growth",
               mode="lines", line=dict(width=2, color='#EF553B'), showlegend=False), row=1, col=2)
fig_cyclical_trends.add_trace(
    go.Scatter(x=indpro.index.to_list(), y=indpro._12m_smoothing_growth / 100, legendgroup="12m ann growth",
               mode="lines", line=dict(width=2, color='red'), showlegend=False), row=1, col=2)
fig_cyclical_trends.add_trace(go.Scatter(x=(indpro_10.index.to_list()),
                                         y=indpro_10['10 yr average'] / 100, line=dict(width=2, color='green'),
                                         mode="lines",
                                         legendgroup="10 yr average", showlegend=False), row=1, col=2)
fig_cyclical_trends.add_trace(
    go.Scatter(x=nonfarm.index.to_list(), y=nonfarm._3m_smoothing_growth / 100, legendgroup="3m ann growth",
               mode="lines", line=dict(width=2, color='orange'), showlegend=False), row=2, col=1)
fig_cyclical_trends.add_trace(
    go.Scatter(x=nonfarm.index.to_list(), y=nonfarm._6m_smoothing_growth / 100, legendgroup="6m ann growth",
               mode="lines", line=dict(width=2, color='#EF553B'), showlegend=False), row=2, col=1)
fig_cyclical_trends.add_trace(
    go.Scatter(x=nonfarm.index.to_list(), y=nonfarm._12m_smoothing_growth / 100, legendgroup="12m ann growth",
               mode="lines", line=dict(width=2, color='red'), showlegend=False), row=2, col=1)
fig_cyclical_trends.add_trace(go.Scatter(x=(nonfarm_10.index.to_list()),
                                         y=nonfarm_10['10 yr average'] / 100, line=dict(width=2, color='green'),
                                         mode="lines",
                                         legendgroup="10 yr average", showlegend=False), row=2, col=1)
fig_cyclical_trends.add_trace(
    go.Scatter(x=real_pers.index.to_list(), y=real_pers._3m_smoothing_growth / 100, legendgroup="3m ann growth",
               mode="lines", line=dict(width=2, color='orange'), showlegend=False), row=2, col=2)
fig_cyclical_trends.add_trace(
    go.Scatter(x=real_pers.index.to_list(), y=real_pers._6m_smoothing_growth / 100, legendgroup="6m ann growth",
               mode="lines", line=dict(width=2, color='#EF553B'), showlegend=False), row=2, col=2)
fig_cyclical_trends.add_trace(
    go.Scatter(x=real_pers.index.to_list(), y=real_pers._12m_smoothing_growth / 100, legendgroup="12m ann growth",
               mode="lines", line=dict(width=2, color='red'), showlegend=False), row=2, col=2)
fig_cyclical_trends.add_trace(go.Scatter(x=(real_pers_10.index.to_list()),
                                         y=real_pers_10['10 yr average'] / 100,
                                         line=dict(width=2, color='green'),
                                         mode="lines",
                                         legendgroup="10 yr average", showlegend=False), row=2, col=2)

fig_cyclical_trends.add_trace(
    go.Scatter(x=retail_sales.index.to_list(), y=retail_sales._3m_smoothing_growth / 100,
               legendgroup="3m ann growth",
               mode="lines", line=dict(width=2, color='orange'), showlegend=False), row=3, col=1)
fig_cyclical_trends.add_trace(
    go.Scatter(x=retail_sales.index.to_list(), y=retail_sales._6m_smoothing_growth / 100,
               legendgroup="6m ann growth",
               mode="lines", line=dict(width=2, color='#EF553B'), showlegend=False), row=3, col=1)
fig_cyclical_trends.add_trace(
    go.Scatter(x=retail_sales.index.to_list(), y=retail_sales._12m_smoothing_growth / 100,
               legendgroup="12m ann growth",
               mode="lines", line=dict(width=2, color='red'), showlegend=False), row=3, col=1)
fig_cyclical_trends.add_trace(go.Scatter(x=(retail_sales_10.index.to_list()),
                                         y=retail_sales_10['10 yr average'] / 100,
                                         line=dict(width=2, color='green'), mode="lines",
                                         legendgroup="10 yr average", showlegend=False), row=3, col=1)

fig_cyclical_trends.add_trace(
    go.Scatter(x=employment_level.index.to_list(), y=employment_level._3m_smoothing_growth / 100,
               legendgroup="3m ann growth",
               mode="lines", line=dict(width=2, color='orange'), showlegend=False), row=3, col=2)
fig_cyclical_trends.add_trace(
    go.Scatter(x=employment_level.index.to_list(), y=employment_level._6m_smoothing_growth / 100,
               legendgroup="6m ann growth",
               mode="lines", line=dict(width=2, color='#EF553B'), showlegend=False), row=3, col=2)
fig_cyclical_trends.add_trace(
    go.Scatter(x=employment_level.index.to_list(), y=employment_level._12m_smoothing_growth / 100,
               legendgroup="12m ann growth",
               mode="lines", line=dict(width=2, color='red'), showlegend=False), row=3, col=2)
fig_cyclical_trends.add_trace(go.Scatter(x=(employment_level_10.index.to_list()),
                                         y=employment_level_10['10 yr average'] / 100,
                                         line=dict(width=2, color='green'), mode="lines",
                                         legendgroup="10 yr average", showlegend=False), row=3, col=2)

fig_cyclical_trends.update_layout(template="plotly_dark",
                                  height=1000, width=1500)
fig_cyclical_trends.update_layout(  # customize font and legend orientation & position
    yaxis=dict(tickformat=".1%"),
    title_font_family="Arial Black",
    font=dict(
        family="Rockwell",
        size=18),
        
    legend=dict(
        title="Growth Indicators", orientation="v", y=0.97, yanchor="bottom", x=0.9, xanchor="left"
    )
)
fig_cyclical_trends.layout.sliders = [dict(visible=True)]


fig_cyclical_trends.layout.yaxis2.tickformat = ".2%"
fig_cyclical_trends.layout.yaxis3.tickformat = ".2%"
fig_cyclical_trends.layout.yaxis4.tickformat = ".2%"
fig_cyclical_trends.layout.yaxis5.tickformat = ".2%"
fig_cyclical_trends.layout.yaxis6.tickformat = ".2%"
# date_start = (datetime.datetime.strptime(date_start, "%Y-%m-%d") + timedelta(days=130)).strftime("%Y-%m-%d")
# fig_cyclical_trends.update_layout(xaxis_range=[date_start, date_end])

fig_cyclical_trends.layout.xaxis.range = [data_displayed, date_end]
fig_cyclical_trends.layout.xaxis2.range = [data_displayed, date_end]
fig_cyclical_trends.layout.xaxis3.range = [data_displayed, date_end]
fig_cyclical_trends.layout.xaxis4.range = [data_displayed, date_end]
fig_cyclical_trends.layout.xaxis5.range = [data_displayed, date_end]
fig_cyclical_trends.layout.xaxis6.range = [data_displayed, date_end]
pce_displayed  = pce96.loc[(pce96.index > data_displayed) & (pce96.index < date_end)]
pce_10_displayed = pce96_10.loc[(pce96_10.index > data_displayed) & (pce96_10.index < date_end)]

indpro_displayed  = indpro.loc[(indpro.index > data_displayed) & (indpro.index < date_end)]
indpro_10_displayed = indpro_10.loc[(indpro_10.index > data_displayed) & (indpro_10.index < date_end)]

nonfarm_displayed  = nonfarm.loc[(nonfarm.index > data_displayed) & (nonfarm.index < date_end)]
nonfarm_10_displayed = nonfarm_10.loc[(nonfarm_10.index > data_displayed) & (nonfarm_10.index < date_end)]

real_pers_displayed  = real_pers.loc[(real_pers.index > data_displayed) & (real_pers.index < date_end)]
real_pers_10_displayed = real_pers_10.loc[(real_pers_10.index > data_displayed) & (real_pers_10.index < date_end)]

retail_sales_displayed  = retail_sales.loc[(retail_sales.index > data_displayed) & (retail_sales.index < date_end)]
retail_sales_10_displayed = retail_sales_10.loc[(retail_sales_10.index > data_displayed) & (retail_sales_10.index < date_end)]

employment_level_displayed  = employment_level.loc[(employment_level.index > data_displayed) & (employment_level.index < date_end)]
employment_level_10_displayed = employment_level_10.loc[(employment_level_10.index > data_displayed) & (employment_level_10.index < date_end)]


fig_cyclical_trends.layout.yaxis.range = [min(min(pce_10_displayed['10 yr average']),min(pce_displayed._3m_smoothing_growth),
                                                     min(pce_displayed._6m_smoothing_growth),
                                                     min(pce_displayed._12m_smoothing_growth))/100, max(max(pce_displayed._3m_smoothing_growth),
                                                     max(pce_displayed._6m_smoothing_growth),
                                                     max(pce_displayed._12m_smoothing_growth))/100]
fig_cyclical_trends.layout.yaxis2.range = [min(min(indpro_10_displayed['10 yr average']),min(indpro_displayed._3m_smoothing_growth),
                                                     min(indpro_displayed._6m_smoothing_growth),
                                                     min(indpro_displayed._12m_smoothing_growth))/100, max(max(indpro_displayed._3m_smoothing_growth),
                                                     max(indpro_displayed._6m_smoothing_growth),
                                                     max(indpro_displayed._12m_smoothing_growth))/100]
fig_cyclical_trends.layout.yaxis3.range = [min(min(nonfarm_10_displayed['10 yr average']),min(nonfarm_displayed._3m_smoothing_growth),
                                                     min(nonfarm_displayed._6m_smoothing_growth),
                                                     min(nonfarm_displayed._12m_smoothing_growth))/100, max(max(nonfarm_displayed._3m_smoothing_growth),
                                                     max(nonfarm_displayed._6m_smoothing_growth),
                                                     max(nonfarm_displayed._12m_smoothing_growth))/100]
fig_cyclical_trends.layout.yaxis4.range = [min(min(real_pers_10_displayed['10 yr average']),min(real_pers_displayed._3m_smoothing_growth),
                                                     min(real_pers_displayed._6m_smoothing_growth),
                                                     min(real_pers_displayed._12m_smoothing_growth))/100, max(max(real_pers_displayed._3m_smoothing_growth),
                                                     max(real_pers_displayed._6m_smoothing_growth),
                                                     max(real_pers_displayed._12m_smoothing_growth))/100]
fig_cyclical_trends.layout.yaxis5.range = [min(min(retail_sales_10_displayed['10 yr average']),min(retail_sales_displayed._3m_smoothing_growth),
                                                     min(retail_sales_displayed._6m_smoothing_growth),
                                                     min(retail_sales_displayed._12m_smoothing_growth))/100, max(max(retail_sales_displayed._3m_smoothing_growth),
                                                     max(retail_sales_displayed._6m_smoothing_growth),
                                                     max(retail_sales_displayed._12m_smoothing_growth))/100]
fig_cyclical_trends.layout.yaxis6.range = [min(min(employment_level_10_displayed['10 yr average']),min(employment_level_displayed._3m_smoothing_growth),
                                                     min(employment_level_displayed._6m_smoothing_growth),
                                                     min(employment_level_displayed._12m_smoothing_growth))/100, max(max(employment_level_displayed._3m_smoothing_growth),
                                                     max(employment_level_displayed._6m_smoothing_growth),
                                                     max(employment_level_displayed._12m_smoothing_growth))/100]




#fig_['layout']['yaxis'].update(autorange=True)
score_table_merged = pd.concat(
    [score_table("PCE", pce96, pce96_10), score_table("Industrial Production", indpro, indpro_10),
     score_table("NonFarm Payroll", nonfarm, nonfarm_10),
     score_table("Real Personal Income", real_pers, real_pers_10),
     score_table("Real Retail Sales", retail_sales, retail_sales_10),
     score_table("Employment Level", employment_level, employment_level_10),
     score_table("COMPOSITE GROWTH", composite_growth, composite_growth_10)], axis=0)
score_table_merged = score_table_merged.iloc[:, [4, 0, 1, 2, 3]]
score_table_merged.reset_index(inplace=True,drop=True)

# define the up and down arrow symbols
up_arrow = '\u2191'
down_arrow = '\u2193'

# define a function to format the values in the specified column with arrows
def format_value(column_name, value):
    if column_name in ['trend vs history ', 'growth', 'Direction of Trend']:
        if value == 1:
            return f"{up_arrow}"
        elif value == 0:
            return f"{down_arrow}"
    return value

# format the table data with arrows in columns 1 and 2
for col in ['trend vs history ', 'growth', 'Direction of Trend']:
    score_table_merged[col] = score_table_merged[col].apply(lambda x: format_value(col, x))

# display the formatted table
st.dataframe(score_table_merged.style.applymap(filter_color,subset=['Score']),hide_index=True,width=700)
st.plotly_chart(fig_, use_container_width=True)
st.plotly_chart(fig_cyclical_trends, use_container_width=True)
