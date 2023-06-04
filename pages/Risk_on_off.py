import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime,timedelta
import requests
import yfinance as yf
from functools import reduce

st.set_page_config(page_title="Risk On/Off Framework")

frequency = "1d"

date_start = "2010-01-01"
date_end = datetime.now().strftime("%Y-%m-%d")

def momentum(period,etf1,etf2):
    data_etf1 = pd.DataFrame(
        yf.download(etf1, date_start, date_end, period=frequency))
    data_etf2 = pd.DataFrame(
        yf.download(etf2, date_start, date_end, period=frequency))

    ratio_etfs = data_etf1/data_etf2
    if period == "1w":
        ratio_etfs[etf1+"/"+etf2] = ratio_etfs["Close"].diff(5)

    elif period == "1m":
        ratio_etfs[etf1+"/"+etf2] = ratio_etfs["Close"].diff(22)

    elif period == "3m":
        ratio_etfs[etf1+"/"+etf2] = ratio_etfs["Close"].diff(66)

    ratio_etfs.dropna(inplace=True)
    return ratio_etfs[[etf1+"/"+etf2]]

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

# Add a row with the sum of each column
concat_momentum.loc['aggregate_ratios'] = concat_momentum.mean()

# Add a column with the sum of each row
concat_momentum['aggregate_periods'] = concat_momentum.mean(axis=1)
concat_momentum.columns = ["1w","1m","3m","aggregate_periods"]

# Define a function to apply color formatting based on values
def color_negative_red(val):
    color = 'red' if val < 0 else 'green' if val > 0 else ''
    return f'color: {color}'

# Apply color formatting to the first three columns, excluding the last row
styled_df = concat_momentum.style.applymap(color_negative_red)

# Get the last row and last column of the DataFrame
last_row = concat_momentum.iloc[-1]
last_column = concat_momentum.iloc[:, -1]



# Display the styled DataFrame in Streamlit
st.write(styled_df,width=800)

