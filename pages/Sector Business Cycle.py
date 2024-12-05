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
import scipy
#from scipy import stats

st.set_page_config(page_title="Business Cycle",layout="wide")

def hit_ratio(period):
  comparison = etf_returns.loc[sectors,etf_returns.columns[-period:]].sub(etf_returns.loc[broad_market,etf_returns.columns[-period:]])
  positive_ratio = pd.DataFrame(comparison.gt(0).sum(axis=1)/len(etf_returns.columns[-period:]))
  positive_ratio.columns = [str(period)+"m"]
  return positive_ratio
def concat_data(data,period,label_indicator):
    df = data[["sector",str(period)+"m"]].T
    df.columns= df.loc["sector",:]
    df.drop("sector",axis=0,inplace=True)
    df.index = [label_indicator]
    return df

def agg_zscore(df):
    z_1=pd.DataFrame((df.iloc[0,:]-df.iloc[0,:].mean())/df.iloc[0,:].std()).T
    z_2=pd.DataFrame((df.iloc[1,:]-df.iloc[1,:].mean())/df.iloc[1,:].std()).T
    z_3=pd.DataFrame((df.iloc[2,:]-df.iloc[2,:].mean())/df.iloc[2,:].std()).T
    agg_z_score = pd.concat([z_1,z_2,z_3]).mean(axis=0)
    return agg_z_score


def highlight_values(df):
    styled_df = df.copy()  # Make a copy to apply transformations
    numeric_cols = styled_df.select_dtypes(include=[np.number,float]).columns
    # Apply column-wise coloring
    for col in numeric_cols:
        styled_df[col] = styled_df[col].apply(lambda x: 
            f'background-color: rgba(0, {int(255 * min(x, 1))}, 0, 0.8)' if x > 0 else
            (f'background-color: rgba({int(255 * min(-x, 1))}, 0, 0, 0.8)' if x < 0 else '')
        )
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        styled_df[col] = ''
    return styled_df
spdr_sector_etfs = {
    "Consumer Staples": "XLP",
    "Consumer Discretionary": "XLY",
    "Communication": "XLC",
    "Health Care": "XLV",
    "Materials": "XLB",
    "Industrials": "XLI",
    "Technology": "XLK",
    "Energy": "XLE",
    "Financials": "XLF",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Broad US Market":"SCHB"
}
sectors = list(spdr_sector_etfs.values())
sectors.remove("SCHB")
broad_market = "SCHB"
etf_prices = yf.download(list(spdr_sector_etfs.values()), start="2022-01-01", end="2024-12-05", interval="1d")["Adj Close"]
st.write(etf_prices)

