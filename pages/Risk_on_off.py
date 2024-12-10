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
st.set_page_config(page_title="Risk On/Off Framework")

frequency = "1d"

date_start = "2010-01-01"
date_end = datetime.now().strftime("%Y-%m-%d")

def momentum(period,etf1,etf2):
    period_dict = {"1w":5,"1m":22,"3m":66,"6m":132,"12m":264,"18m":396}
    try:
        data_etf1 = pd.DataFrame(
            yf.download(etf1, date_start, date_end, period=frequency))
        data_etf2 = pd.DataFrame(
            yf.download(etf2, date_start, date_end, period=frequency))
        ratio_etfs = data_etf1['Close'][etf1]/data_etf2['Close'][etf2]
    
        ratio_etfs[etf1+"/"+etf2] = ratio_etfs.diff(period_dict[period])
        ratio_etfs.dropna(inplace=True)
        final_df = ratio_etfs[[etf1+"/"+etf2]][0].to_frame()
        final_df.columns = [etf1+"/"+etf2]
    except KeyError:
        time.sleep(1)
        data_etf1 = pd.DataFrame(
            yf.download(etf1, date_start, date_end, period=frequency))
        data_etf2 = pd.DataFrame(
            yf.download(etf2, date_start, date_end, period=frequency))

        ratio_etfs = data_etf1['Close'][etf1]/data_etf2['Close'][etf2]
    
        ratio_etfs[etf1+"/"+etf2] = ratio_etfs.diff(period_dict[period])

        ratio_etfs.dropna(inplace=True)
        final_df = ratio_etfs[[etf1+"/"+etf2]][0].to_frame()
        final_df.columns = [etf1+"/"+etf2]
    return final_df

#1w
etf_pairs = {"Commo/Stocks":["PDBC","SPY"],
             "Commo/Long-Term Bonds":["PDBC","EDV"],
             "Commo/Gold":["PDBC","GLD"],
             "Stocks/Gold":["SPY","GLD"],
             "Stocks/Long-Term Bonds":["SPY","EDV"],
             "Gold/Bonds":["GLD","EDV"]}
etf_ratio_1w = ""
etf_ratio_1m = ""
etf_ratio_3m = ""
etf_ratio_6m = ""
etf_ratio_12m = ""
etf_ratio_18m = ""

for k,v in enumerate(etf_pairs):
    time.sleep(.1)
    exec("_".join(etf_pairs[v])+"_1w=momentum(\"1w\",\""+etf_pairs[v][0]+"\",\""+etf_pairs[v][1]+"\")") 
    time.sleep(.1)
    exec("_".join(etf_pairs[v])+"_1m=momentum(\"1m\",\""+etf_pairs[v][0]+"\",\""+etf_pairs[v][1]+"\")") 
    time.sleep(.1)
    exec("_".join(etf_pairs[v])+"_3m=momentum(\"3m\",\""+etf_pairs[v][0]+"\",\""+etf_pairs[v][1]+"\")") 
    time.sleep(.1)
    exec("_".join(etf_pairs[v])+"_6m=momentum(\"6m\",\""+etf_pairs[v][0]+"\",\""+etf_pairs[v][1]+"\")") 
    time.sleep(.1)
    exec("_".join(etf_pairs[v])+"_12m=momentum(\"12m\",\""+etf_pairs[v][0]+"\",\""+etf_pairs[v][1]+"\")") 
    time.sleep(.1)
    exec("_".join(etf_pairs[v])+"_18m=momentum(\"18m\",\""+etf_pairs[v][0]+"\",\""+etf_pairs[v][1]+"\")") 
    etf_ratio_1w+="_".join(etf_pairs[v])+"_1w"+','
    etf_ratio_1m+="_".join(etf_pairs[v])+"_1m"+','
    etf_ratio_3m+="_".join(etf_pairs[v])+"_3m"+','
    etf_ratio_6m+="_".join(etf_pairs[v])+"_6m"+','
    etf_ratio_12m+="_".join(etf_pairs[v])+"_12m"+','
    etf_ratio_18m+="_".join(etf_pairs[v])+"_18m"+','


exec("concat_data_1w =reduce(lambda left,right:pd.merge(left,right,left_index=True,right_index=True),["+etf_ratio_1w+"])")
exec("concat_data_1m =reduce(lambda left,right:pd.merge(left,right,left_index=True,right_index=True),["+etf_ratio_1m+"])")
exec("concat_data_3m =reduce(lambda left,right:pd.merge(left,right,left_index=True,right_index=True),["+etf_ratio_3m+"])")
exec("concat_data_6m =reduce(lambda left,right:pd.merge(left,right,left_index=True,right_index=True),["+etf_ratio_6m+"])")
exec("concat_data_12m =reduce(lambda left,right:pd.merge(left,right,left_index=True,right_index=True),["+etf_ratio_12m+"])")
exec("concat_data_18m =reduce(lambda left,right:pd.merge(left,right,left_index=True,right_index=True),["+etf_ratio_18m+"])")
#je veux traduire le risk on par une valeur pos, donc ce qui est plus cyclique et risk on au numérateur

last_momentum_1w = pd.DataFrame(concat_data_1w.iloc[len(concat_data_1w)-1,:])
last_momentum_1m = pd.DataFrame(concat_data_1m.iloc[len(concat_data_1m)-1,:])
last_momentum_3m = pd.DataFrame(concat_data_3m.iloc[len(concat_data_3m)-1,:])
last_momentum_6m = pd.DataFrame(concat_data_6m.iloc[len(concat_data_6m)-1,:])
last_momentum_12m = pd.DataFrame(concat_data_12m.iloc[len(concat_data_12m)-1,:])
last_momentum_18m = pd.DataFrame(concat_data_18m.iloc[len(concat_data_18m)-1,:])
last_momentum_1w.columns = ["1w"]
last_momentum_1m.columns = ["1m"]
last_momentum_3m.columns = ["3m"]
last_momentum_6m.columns = ["6m"]
last_momentum_12m.columns = ["12m"]
last_momentum_18m.columns = ["18m"]


concat_momentum = reduce(lambda left,right : pd.merge(left,right,left_index=True,right_index=True),[last_momentum_1w,last_momentum_1m,last_momentum_3m,
                                                                                         last_momentum_6m,last_momentum_12m,last_momentum_18m])
concat_momentum *= 100
# Add a row with the sum of each column
concat_momentum.loc[''] = concat_momentum.iloc[:len(concat_momentum-1)].mean()
concat_momentum.reset_index(inplace=True)
concat_momentum.columns = ["ETF","1w","1m","3m","6m","12m","18m"]
# Add a column with the sum of each row
concat_momentum['aggregate_periods'] = concat_momentum[["1w","1m","3m","6m","12m","18m"]].mean(axis=1)
concat_momentum["Description"] = list(etf_pairs.keys()) + ["Aggregate"]
concat_momentum.set_index(["Description","ETF"],inplace=True)
#concat_momentum.columns = ["1w","1m","3m","aggregate_period","Description"]

# Define custom color scale
red = np.array([255, 0, 0])  # RGB values for red
dark_green = np.array([43, 126, 31])  # RGB values for dark green
light_green = np.array([144, 238, 144])  # RGB values for light green
white = np.array([0,0,0])
# Define a function to apply cell background color and text color based on values
def color_scale(val):
    if val < 0:
        quantiles = concat_momentum.iloc[:, :3].stack().quantile([0.01, 0.75])
        min_val, max_val = quantiles.iloc[0], quantiles.iloc[1]
        intensity = (val - min_val) / (max_val - min_val)
        color = red + abs(intensity) * (dark_green - red)

    elif val > 0:
        quantiles = concat_momentum.iloc[:, :3].stack().quantile([0.01, 0.99])
        max_val, min_val = quantiles.iloc[0], quantiles.iloc[1]
        intensity = (val - min_val) / (max_val - min_val)

        color = dark_green + intensity * (light_green - dark_green)

    else:
        color = white


    # Ensure RGB values are within valid range (0-255)
    color = np.clip(color, 0, 255)

    # Convert RGB values to hexadecimal color code
    hex_code = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
    return f'background-color: {hex_code}'
# Apply cell background color to the first three columns, excluding the last row

styled_df = concat_momentum.style.applymap(color_scale)
ETF_ratios = concat_momentum.reset_index()
# styled_df = styled_df.set_properties(subset=pd.IndexSlice["aggregate_ratios", :], **{'color': '', 'background-color': ''})
# styled_df = styled_df.set_properties(subset=pd.IndexSlice[:, "aggregate_periods"], **{'color': '', 'background-color': ''})

st.dataframe(styled_df.format({col:"{:.2f}" for col in concat_momentum.columns}),width=1300)
selection_ratios = st.multiselect("Ratios",options=ETF_ratios["Description"][:len(ETF_ratios)-1])
if len(selection_ratios) != 0:
    etf1=etf_pairs[selection_ratios[0]][0]
    etf2 = etf_pairs[selection_ratios[0]][1]
    data_etf1 = pd.DataFrame(
                yf.download(etf1, date_start, date_end, period=frequency))["Close"]
    data_etf2 = pd.DataFrame(
                yf.download(etf2, date_start, date_end, period=frequency))["Close"]
    merged_ratio = data_etf1.pct_change(1).join(data_etf2.pct_change(1))
    merged_ratio[etf1+"/"+etf2] = merged_ratio[etf1]/merged_ratio[etf2]
    #should have insteade of pc_change(1) the ratio for the last 1w + average pct change on a sliding window of 1w,1m,3m,etc..
    fig_ratio = go.Figure()
    fig_ratio.add_trace(
        go.Scatter(x=merged_ratio.index.to_list(), y=merged_ratio[etf1+"/"+etf2], name=etf1+"/"+etf2,
                mode="lines", line=dict(width=2, color='white'), showlegend=True))

    st.plotly_chart(fig_ratio)
# fig_1w = px.line(concat_data_1w)
# fig_1w.update_layout(title_text="Momentum 1w")

# fig_1m = px.line(concat_data_1m)
# fig_1m.update_layout(title_text="Momentum 1m")

# fig_3m = px.line(concat_data_3m)
# fig_3m.update_layout(title_text="Momentum 3m")
# st.plotly_chart(fig_1w)
# st.plotly_chart(fig_1m)
# st.plotly_chart(fig_3m)
