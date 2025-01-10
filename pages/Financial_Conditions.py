


import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime,timedelta
import requests
import yfinance as yf
from functools import reduce
import plotly.express as px
import time
from pandas.tseries.offsets import BDay

def agg_zscore(df):
    
    df["mean"] = df.mean(axis=1)
    df["std"] = df.std(axis=1)
    df.iloc[:,:3] =(np.array(df.iloc[:,:3])-np.array(df.iloc[:,3]).reshape(-1,1))/np.array(df.iloc[:,4]).reshape(-1,1)
    #df.loc[:,df.columns[:len(df.columns)-2]] = df.loc[:,df.columns[:len(df.columns)-2]])
    df.drop(["mean","std"],axis=1,inplace=True)
    agg_z_score = df.copy()
    return agg_z_score
    
"""- US Dollar Index (DX-Y.NYB)
- CBOE Interest Rate 10 Year T No (^TNX)
- Gasoline Active Fut (RB=F)"""


import numpy as np
import pandas as pd


def color_scale(val):
    # Compute the global min and max for scaling
    min_val = proxy_return.min().min()
    max_val = proxy_return.max().max()

    # Define RGB color codes for different shades
    deep_red = np.array([295, 200, 200])      # Deep red for strong negative values
    light_red = np.array([255, 0, 0]) # Light red for weak negative values
    deep_green = np.array([0, 255, 0])    # Strong green for strong positive values
    light_green = np.array([100, 205, 100]) # Light green for weak positive values

    # Normalize intensity based on full range of values
    if val < 0:
        intensity = (val - min_val) / (0 - min_val)  # Scale negative values
        color = light_red + intensity * (deep_red - light_red)  # Blend light red to deep red

    elif val > 0:
        intensity = val / max_val  # Scale positive values
        color = light_green + intensity * (deep_green - light_green)  # Blend light green to strong green

    else:
        color = np.array([255, 255, 255])  # White for zero

    # Ensure RGB values are within valid range (0-255)
    color = np.clip(color, 0, 255)

    # Convert RGB values to hexadecimal color code
    hex_code = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
    return f'background-color: {hex_code}'
fin_conditions_tickers = {"US Dollar Index":"DX-Y.NYB",
                          "CBOE 10Y T Note":"^TNX",
                          "RBOB Gasoline Active Fut":"RB=F"}



col1,col2= st.columns(2,gap="small")
with col1:
    dropdown_menu = st.selectbox("Additional Line",options=["","DXY","10Y","Gasoline"])
with col2:
 
    st.markdown("""<h4>Period</h4>""",unsafe_allow_html=True)
    col1,col2,col3,col4,col5,col6 = st.columns(6,gap="small")
    with col1:
        period_1m = st.button("1m",disabled=False)
    with col2:
        period_3m = st.button("3m")
    with col3:
        period_6m = st.button("6m")
    with col4:
        period_12m = st.button("12m")
    with col5:
        period_18m = st.button("18m")    
    with col6:
        period_all = st.button("All")
date_start = "2002-01-01"
date_end = datetime.today().strftime("%Y-%m-%d")

if period_1m:
    date_start = datetime.strptime(date_end,"%Y-%m-%d") +  BDay(-22*2)
    buff = 22
elif period_3m:
    date_start = datetime.strptime(date_end,"%Y-%m-%d") +  BDay(-66*2)
    buff = 66
elif period_6m:
    date_start = datetime.strptime(date_end,"%Y-%m-%d") +  BDay(-132*2)
    buff = 132
elif period_12m:
    date_start = datetime.strptime(date_end,"%Y-%m-%d") +  BDay(-252*2)
    buff = 252
elif period_18m:
    date_start = datetime.strptime(date_end,"%Y-%m-%d") +  BDay(-384*2)
proxy_return = yf.download(list(fin_conditions_tickers.values()), start=date_start, end=date_end, interval="1d")["Adj Close"]
### avg saily dev ###------------------------------------
indicators = ["DXY","10Y","Gasoline"]
proxy_return.columns = ["DXY","10Y","Gasoline"]

col1,col2 = st.columns(2,gap="small")
with col1:
    ret_rolling_window = st.select_slider("Return period ",options=["1d","1w","1m","3m","6m"])
    window_ret = 1 if ret_rolling_window == "1d" else 5 if ret_rolling_window == "1w" else 22 if ret_rolling_window == "1m" else 66 if ret_rolling_window == "3m" else 132
with col2:
    z_rolling_window = st.select_slider("Score Rolling window (in m)",options=["1","2","3","6","12","18","24"])
# rolling_z = proxy_return.copy()
for col in proxy_return.columns :
    proxy_return["return_"+col] = proxy_return[col].pct_change(window_ret)

for col in indicators :
    proxy_return["z"+col] = (proxy_return["return_"+col] - proxy_return["return_"+col].rolling(int(z_rolling_window)*22).mean())/proxy_return["return_"+col].rolling(int(z_rolling_window   )*22).std()
proxy_return["agg_z"] = proxy_return[proxy_return.columns[-3:]].mean(axis=1)
proxy_return.dropna(inplace=True)
agg_table_score = proxy_return[["zDXY","z10Y","zGasoline","agg_z"]][::-1].head(30).style.applymap(color_scale)

if dropdown_menu == "":
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=proxy_return.index.tolist(),y=proxy_return["agg_z"]))
    
else:
    fig = make_subplots(rows=1,
                        cols=1,
                        specs=[[{"secondary_y": True}]],
                        )
    fig.add_trace(go.Scatter(x=proxy_return.index.to_list(),
                               y=proxy_return["agg_z"].to_list(),
                               name="Agg Score"),row=1,col=1,secondary_y=False)
    fig.add_trace(go.Scatter(x=proxy_return.index.to_list(),
                               y=proxy_return[dropdown_menu].to_list(),
                               name=dropdown_menu),row=1,col=1,secondary_y=True)
    

st.plotly_chart(fig,use_container_width=True)
st.dataframe(agg_table_score)
