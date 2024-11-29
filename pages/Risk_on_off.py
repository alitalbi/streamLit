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
st.set_page_config(page_title="Risk On/Off Framework")

frequency = "1d"

date_start = "2010-01-01"
date_end = datetime.now().strftime("%Y-%m-%d")

def momentum(period,etf1,etf2):
    data_etf1 = pd.DataFrame(
        yf.download(etf1, date_start, date_end, period=frequency))
    data_etf2 = pd.DataFrame(
        yf.download(etf2, date_start, date_end, period=frequency))

    ratio_etfs = data_etf1['Close']/data_etf2['Close']
  
    if period == "1w":
        ratio_etfs[etf1+"/"+etf2] = ratio_etfs.diff(5)

    elif period == "1m":
        ratio_etfs[etf1+"/"+etf2] = ratio_etfs.diff(22)

    elif period == "3m":
        ratio_etfs[etf1+"/"+etf2] = ratio_etfs.diff(66)

    ratio_etfs.dropna(inplace=True)
    final_df = ratio_etfs[[etf1+"/"+etf2]][0].to_frame()
    final_df.columns = [etf1+"/"+etf2]
    return final_df

#1w

spy_shy_1w = momentum("1w","SPY","SHY")
spy_ief_1w = momentum("1w","SPY","IEF")
spy_tlt_1w = momentum("1w","SPY","TLT")
spy_xly_1w = momentum("1w","XLY","SPY")
qqq_tlt_1w = momentum("1w","QQQ","TLT")
rcd_rhs_1w = momentum("1w","RCD","RHS")
sox_fxy_1w = momentum("1w","SOXL","FXY")
xlp_xly_1w = momentum("1w","XLY","XLP")
spy_pscd_1w = momentum("1w","PSCD","SPY")
spy_iwm_1w = momentum("1w","IWM","SPY")
qqq_iwm_1w = momentum("1w","IWM","QQQ")
itot_govt_1w = momentum("1w","ITOT","GOVT")
gld_tlt_1w = momentum("1w","GLD","TLT")
kre_spy_1w = momentum("1w","KRE","SPY")

#1m
spy_shy_1m = momentum("1m","SPY","SHY")
spy_ief_1m = momentum("1m","SPY","IEF")
spy_tlt_1m = momentum("1m","SPY","TLT")
spy_xly_1m = momentum("1m","XLY","SPY")
qqq_tlt_1m = momentum("1m","QQQ","TLT")
rcd_rhs_1m = momentum("1m","RCD","RHS")
sox_fxy_1m = momentum("1m","SOXL","FXY")
xlp_xly_1m = momentum("1m","XLY","XLP")
spy_pscd_1m = momentum("1m","PSCD","SPY")
spy_iwm_1m = momentum("1m","IWM","SPY")
qqq_iwm_1m = momentum("1m","IWM","QQQ")
itot_govt_1m = momentum("1m","ITOT","GOVT")
gld_tlt_1m = momentum("1m","GLD","TLT")
kre_spy_1m = momentum("1m","KRE","SPY")

#1m
spy_shy_3m = momentum("3m","SPY","SHY")
spy_ief_3m = momentum("3m","SPY","IEF")
spy_tlt_3m = momentum("3m","SPY","TLT")
spy_xly_3m = momentum("3m","XLY","SPY")
qqq_tlt_3m = momentum("3m","QQQ","TLT")
rcd_rhs_3m = momentum("3m","RCD","RHS")
sox_fxy_3m = momentum("3m","SOXX","FXY")
xlp_xly_3m = momentum("3m","XLY","XLP")
spy_pscd_3m = momentum("3m","PSCD","SPY")
spy_iwm_3m = momentum("3m","IWM","SPY")
qqq_iwm_3m = momentum("3m","IWM","QQQ")
itot_govt_3m = momentum("3m","ITOT","GOVT")
gld_tlt_3m = momentum("3m","GLD","TLT")
kre_spy_3m = momentum("3m","KRE","SPY")
#je veux traduire le risk on par une valeur pos, donc ce qui est plus cyclique et risk on au numérateur

concat_data_1w = reduce(lambda left,right:pd.merge(left,right,left_index=True,right_index=True),[spy_shy_1w,spy_ief_1w,spy_tlt_1w,spy_xly_1w,qqq_tlt_1w,
                                                                                rcd_rhs_1w,sox_fxy_1w,xlp_xly_1w,spy_pscd_1w,spy_iwm_1w,
                                                                                qqq_iwm_1w,itot_govt_1w,gld_tlt_1w,kre_spy_1w])
concat_data_1m = reduce(lambda left,right:pd.merge(left,right,left_index=True,right_index=True),[spy_shy_1m,spy_ief_1m,spy_tlt_1m,spy_xly_1m,qqq_tlt_1m,
                                                                                rcd_rhs_1m,sox_fxy_1m,xlp_xly_1m,spy_pscd_1m,spy_iwm_1m,
                                                                                qqq_iwm_1m,itot_govt_1m,gld_tlt_1m,kre_spy_1m])
concat_data_3m = reduce(lambda left,right:pd.merge(left,right,left_index=True,right_index=True),[spy_shy_3m,spy_ief_3m,spy_tlt_3m,spy_xly_3m,qqq_tlt_3m,
                                                                                rcd_rhs_3m,sox_fxy_3m,xlp_xly_3m,spy_pscd_3m,spy_iwm_3m,
                                                                                qqq_iwm_3m,itot_govt_3m,gld_tlt_3m,kre_spy_3m])

last_momentum_1w = pd.DataFrame(concat_data_1w.iloc[len(concat_data_1w)-1,:])
last_momentum_1m = pd.DataFrame(concat_data_1m.iloc[len(concat_data_1m)-1,:])
last_momentum_3m = pd.DataFrame(concat_data_3m.iloc[len(concat_data_3m)-1,:])

concat_momentum = reduce(lambda left,right : pd.merge(left,right,left_index=True,right_index=True),[last_momentum_1w,last_momentum_1m,last_momentum_3m])
st.write(spy_shy_1w,concat_data_1w)
# Add a row with the sum of each column
concat_momentum.loc['aggregate_ratios'] = concat_momentum.mean()

# Add a column with the sum of each row
concat_momentum['aggregate_periods'] = concat_momentum.mean(axis=1)
concat_momentum.columns = ["1w","1m","3m","aggregate_periods"]

# Define custom color scale
red = np.array([255, 0, 0])  # RGB values for red
dark_green = np.array([43, 126, 31])  # RGB values for dark green
light_green = np.array([144, 238, 144])  # RGB values for light green

# Define a function to apply cell background color and text color based on values
def color_scale(val):
    if val < 0:
        quantiles = concat_momentum.iloc[:, :3].stack().quantile([0.25, 0.75])
        min_val, max_val = quantiles.iloc[0], quantiles.iloc[1]
        intensity = (val - min_val) / (max_val - min_val)
        color = red + abs(intensity) * (dark_green - red)

    elif val > 0:
        quantiles = concat_momentum.iloc[:, :3].stack().quantile([0.05, 0.99])
        max_val, min_val = quantiles.iloc[0], quantiles.iloc[1]
        intensity = (val - min_val) / (max_val - min_val)

        color = dark_green + intensity * (light_green - dark_green)

    else:
        color = light_green


    # Ensure RGB values are within valid range (0-255)
    color = np.clip(color, 0, 255)

    # Convert RGB values to hexadecimal color code
    hex_code = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
    return f'background-color: {hex_code}'
# Apply cell background color to the first three columns, excluding the last row
styled_df = concat_momentum.style.applymap(color_scale)

styled_df = styled_df.set_properties(subset=pd.IndexSlice["aggregate_ratios", :], **{'color': '', 'background-color': ''})
styled_df = styled_df.set_properties(subset=pd.IndexSlice[:, "aggregate_periods"], **{'color': '', 'background-color': ''})



st.dataframe(styled_df,width=1100,height=535)
fig_1w = px.line(concat_data_1w)
fig_1w.update_layout(title_text="Momentum 1w")

fig_1m = px.line(concat_data_1m)
fig_1m.update_layout(title_text="Momentum 1m")

fig_3m = px.line(concat_data_3m)
fig_3m.update_layout(title_text="Momentum 3m")
st.plotly_chart(fig_1w)
st.plotly_chart(fig_1m)
st.plotly_chart(fig_3m)
