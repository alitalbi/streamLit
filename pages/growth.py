import streamlit as st
import pandas as pd
from fredapi import Fred
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime,timedelta
import requests
fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')


st.set_page_config(page_title="growth")

frequency = "monthly"
fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')

date_start = st.date_input("Start date:", pd.Timestamp("2021-01-01"))
print(type(date_start))
date_start2 = datetime.strptime("2004-01-01","%Y-%m-%d").date()
print(type(date_start2))
date_end = st.date_input("End date:", pd.Timestamp(datetime.now().strftime("%Y-%m-%d")))

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
    print(data_)
    return data_[['_6m_smoothing_growth']], data_2[['10 yr average']]

def data_smooth(data_,date_start,date_end):
    data_ = data_.loc[(data_.index > date_start) & (data_.index < date_end)]
    #data_.index = pd.to_datetime(data_.index).dt.date()
    # creating 6m smoothing growth column and 10yr average column
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

st.sidebar.header("Growth")


print("6")

retail_sales_title = "Real Retail Sales"
employment_level_title = "Employment Level"
pce_title = "PCE"
print("7")
indpro_title = "Industrial Production"
print('8')
nonfarm_title = "NonFarm Payroll"
print("9")
real_personal_income_title = "Real Personal Income"
print("3")
cpi_title = "CPI"
core_cpi_title = "Core CPI"
core_pce_title = "Core PCE"
date_start_converted = date_start.strftime("%Y-%m-%d")
date_start2_converted = date_start2.strftime("%Y-%m-%d")
date_end_converted = date_end.strftime("%Y-%m-%d")

pcec96, pcec96_10 = smooth_data("PCEC96", date_start_converted, date_start2_converted, date_end_converted)

print("3")
indpro, indpro_10 = smooth_data("INDPRO", date_start_converted, date_start2_converted, date_end_converted)
print('4')
nonfarm, nonfarm_10 = smooth_data("PAYEMS", date_start_converted, date_start2_converted, date_end_converted)
print("5")

real_pers, real_pers_10 = smooth_data("W875RX1", date_start_converted, date_start2_converted, date_end_converted)

retail_sales, retail_sales_10 = smooth_data("RRSFS", date_start_converted, date_start2_converted, date_end_converted)

employment_level, employment_level_10 = smooth_data("CE16OV", date_start_converted, date_start2_converted, date_end_converted)
employment_level.dropna(inplace=True)

composite_data = pd.concat(
    [pcec96[['_6m_smoothing_growth']], indpro[['_6m_smoothing_growth']], nonfarm[['_6m_smoothing_growth']],
     real_pers[['_6m_smoothing_growth']], retail_sales[['_6m_smoothing_growth']],
     employment_level[['_6m_smoothing_growth']]], axis=1)
composite_data.dropna(inplace=True)
composite_growth = pd.DataFrame(composite_data.mean(axis=1))
composite_growth.columns = ["_6m_smoothing_growth"]
composite_growth_10 = pd.concat(
    [pcec96_10[['10 yr average']], indpro_10[['10 yr average']], nonfarm_10[['10 yr average']],
     real_pers_10[['10 yr average']], retail_sales_10[['10 yr average']],
     employment_level_10[['10 yr average']]],
    axis=1)
composite_growth_10.dropna(inplace=True)
composite_growth_10 = pd.DataFrame(composite_growth_10.mean(axis=1))
composite_growth_10.columns = ["10 yr average"]
url = 'https://www.atlantafed.org/-/media/documents/cqer/researchcq/gdpnow/GDPTrackingModelDataAndForecasts.xlsx'
response = requests.get(url)

# Use pandas to read the downloaded Excel file from memory
atlanta_gdp_now = pd.read_excel(response.content, sheet_name="TrackingArchives", usecols=['Forecast Date','GDP Nowcast'])
print("ali")

#atlanta_gdp_now["Forecast Date"] = atlanta_gdp_now["Forecast Date"].apply(lambda x:datetime.strftime(x,"%Y-%m-%d"))
atlanta_gdp_now.set_index("Forecast Date",inplace=True,drop=True)

#a= data_smooth(atlanta_gdp_now,date_start,date_end)
#print("atlanta index date type : ",type(a.index))

#composite_growth.to_csv("/Users/talbi/Downloads/composite_growth.csv")
fig_ = go.Figure()


# drop the blank values

# ploting the data
# composite_growth_10 = 100 * (composite_growth.iloc[:, 0].rolling(10).mean().pct_change())
fig_.add_trace(go.Scatter(x=composite_growth.index.to_list(), y=composite_growth._6m_smoothing_growth / 100,
                          name="6m growth average",
                          mode="lines", line=dict(width=2, color='white')))
fig_.add_trace(go.Scatter(x=atlanta_gdp_now.index.to_list(), y=atlanta_gdp_now.iloc[:,0]/100,
                          name="Atlanta Fed GDP Nowcast",
                          mode="lines", line=dict(width=2, color='blue')))
fig_.add_trace(go.Scatter(x=composite_growth_10.index.to_list(),
                          y=composite_growth_10['10 yr average'] / 100,
                          name="10 YR average",
                          mode="lines", line=dict(width=2, color='green')))

fig_.update_layout(
    template="plotly_dark",
    title={
        'text': "COMPOSITE GROWTH",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig_.update_layout(xaxis_range = [date_start,date_end])
#           max(composite_growth._6m_smoothing_growth) * 1.1])
fig_.update_layout(  # customize font and legend orientation & position
    yaxis=dict(tickformat=".1%"),

    title_font_family="Arial Black",
    font=dict(
        family="Rockwell",
        size=16),
    legend=dict(
        title=None, orientation="h", y=0.97, yanchor="bottom", x=0.5, xanchor="center"
    ),
)
fig_.update_layout(xaxis=dict(rangeselector=dict(font=dict(color="black"))))
fig_cyclical_trends = make_subplots(rows=3, cols=2, subplot_titles=[pce_title, indpro_title
    , nonfarm_title, real_personal_income_title, retail_sales_title, employment_level_title])

fig_cyclical_trends.add_trace(
    go.Scatter(x=pcec96.index.to_list(), y=pcec96._6m_smoothing_growth / 100, name="6m growth average",
               mode="lines", line=dict(width=2, color='white')), row=1, col=1)
fig_cyclical_trends.add_trace(go.Scatter(x=(pcec96_10.index.to_list()),
                                         y=(pcec96_10['10 yr average']) / 100, mode="lines",
                                         line=dict(width=2, color='green'),
                                         name="10yr average"), row=1, col=1)

fig_cyclical_trends.add_trace(
    go.Scatter(x=indpro.index.to_list(), y=indpro._6m_smoothing_growth / 100, name="6m growth average",
               mode="lines", line=dict(width=2, color='white'), showlegend=False), row=1, col=2)
fig_cyclical_trends.add_trace(go.Scatter(x=(indpro_10.index.to_list()),
                                         y=indpro_10['10 yr average'] / 100, line=dict(width=2, color='green'),
                                         mode="lines",
                                         name="10yr average", showlegend=False), row=1, col=2)

fig_cyclical_trends.add_trace(
    go.Scatter(x=nonfarm.index.to_list(), y=nonfarm._6m_smoothing_growth / 100, name="6m growth average",
               mode="lines", line=dict(width=2, color='white'), showlegend=False), row=2, col=1)
fig_cyclical_trends.add_trace(go.Scatter(x=(nonfarm_10.index.to_list()),
                                         y=nonfarm_10['10 yr average'] / 100, line=dict(width=2, color='green'),
                                         mode="lines",
                                         name="10yr average", showlegend=False), row=2, col=1)

fig_cyclical_trends.add_trace(
    go.Scatter(x=real_pers.index.to_list(), y=real_pers._6m_smoothing_growth / 100, name="6m growth average",
               mode="lines", line=dict(width=2, color='white'), showlegend=False), row=2, col=2)
fig_cyclical_trends.add_trace(go.Scatter(x=(real_pers_10.index.to_list()),
                                         y=real_pers_10['10 yr average'] / 100,
                                         line=dict(width=2, color='green'),
                                         mode="lines",
                                         name="10yr average", showlegend=False), row=2, col=2)

fig_cyclical_trends.add_trace(
    go.Scatter(x=retail_sales.index.to_list(), y=retail_sales._6m_smoothing_growth / 100,
               name="6m growth average",
               mode="lines", line=dict(width=2, color='white'), showlegend=False), row=3, col=1)
fig_cyclical_trends.add_trace(go.Scatter(x=(retail_sales_10.index.to_list()),
                                         y=retail_sales_10['10 yr average'] / 100,
                                         line=dict(width=2, color='green'), mode="lines",
                                         name="10yr average", showlegend=False), row=3, col=1)

fig_cyclical_trends.add_trace(
    go.Scatter(x=employment_level.index.to_list(), y=employment_level._6m_smoothing_growth / 100,
               name="6m growth average",
               mode="lines", line=dict(width=2, color='white'), showlegend=False), row=3, col=2)
fig_cyclical_trends.add_trace(go.Scatter(x=(employment_level_10.index.to_list()),
                                         y=employment_level_10['10 yr average'] / 100,
                                         line=dict(width=2, color='green'), mode="lines",
                                         name="10yr average", showlegend=False), row=3, col=2)

fig_cyclical_trends.update_layout(template="plotly_dark",
                                  height=1000, width=1500)
fig_cyclical_trends.update_layout(  # customize font and legend orientation & position
    yaxis=dict(tickformat=".1%"),
    title_font_family="Arial Black",
    font=dict(
        family="Rockwell",
        size=18),
    legend=dict(
        title=None, orientation="h", y=1.02, yanchor="bottom", x=0.5, xanchor="center"
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

fig_cyclical_trends.layout.xaxis.range = [date_start, date_end]
fig_cyclical_trends.layout.xaxis2.range = [date_start, date_end]
fig_cyclical_trends.layout.xaxis3.range = [date_start, date_end]
fig_cyclical_trends.layout.xaxis4.range = [date_start, date_end]
fig_cyclical_trends.layout.xaxis5.range = [date_start, date_end]
fig_cyclical_trends.layout.xaxis6.range = [date_start, date_end]


fig_['layout']['yaxis'].update(autorange=True)
score_table_merged = pd.concat(
    [score_table("PCE", pcec96, pcec96_10), score_table("Industrial Production", indpro, indpro_10),
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

st.write(score_table_merged.style.applymap(filter_color,subset=['Score']))
st.plotly_chart(fig_, use_container_width=True)
st.plotly_chart(fig_cyclical_trends, use_container_width=True)