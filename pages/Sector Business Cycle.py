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
etf_returns = round(100*(etf_prices.resample("M").last()/etf_prices.resample("M").first() - 1),2).T
avg_returns = etf_returns.copy()
avg_returns["mtd"] = etf_returns.iloc[:,-1]
avg_returns["2m"] = etf_returns.iloc[:,-2:].mean(axis=1)
avg_returns["3m"] = etf_returns.iloc[:,-3:].mean(axis=1)
avg_returns["6m"] = etf_returns.iloc[:,-6:].mean(axis=1)
avg_returns["9m"] = etf_returns.iloc[:,-9:].mean(axis=1)
avg_returns["12m"] = etf_returns.iloc[:,-12:].mean(axis=1)
avg_returns["18m"] = etf_returns.iloc[:,-18:].mean(axis=1)
avg_returns = avg_returns[["mtd","2m","3m","6m","9m","12m","18m"]]

excess_return = round(avg_returns.loc[sectors,:] - avg_returns.loc[broad_market,:],2)
excess_return.reset_index(inplace=True)
st.write(excess_return)
excess_return["sector"] = excess_return["Ticker"].map({spdr_sector_etfs[v]:v for k,v in enumerate(spdr_sector_etfs)})
excess_return = excess_return[["Ticker","sector","mtd","3m","6m","12m","18m"]]
avg_returns.reset_index(inplace=True)
avg_returns["sector"] = avg_returns["Ticker"].map({spdr_sector_etfs[v]:v for k,v in enumerate(spdr_sector_etfs)})
avg_returns = avg_returns[["Ticker","sector","mtd","3m","6m","12m","18m"]]

# st.markdown("### Average return per Sectors")
# styled_avg_returns = avg_returns.style.apply(highlight_values,axis=None)
# st.dataframe(styled_avg_returns.format({"mtd":"{:.2f}","2m":"{:.2f}","3m":"{:.2f}","6m":"{:.2f}","9m":"{:.2f}",
#                                             "12m":"{:.2f}","18m":"{:.2f}"}),width=1150,height=460,hide_index=True)


styled_excess_returns = excess_return.style.apply(highlight_values,axis=None)

hit_ratios_list= []
for period in [3,6,12,18]:
  sub_df = round(hit_ratio(period),2)*100
  hit_ratios_list.append(sub_df)
hit_ratios = pd.concat(hit_ratios_list,axis=1)
hit_ratios.reset_index(inplace=True)
hit_ratios["sector"] = hit_ratios["Ticker"].map({spdr_sector_etfs[v]:v for k,v in enumerate(spdr_sector_etfs)})
styled_hit_ratios = hit_ratios.style.apply(highlight_values,axis=None)
sector_avg_returns = avg_returns.loc[avg_returns["Ticker"].isin(sectors)]
# st.dataframe(styled_hit_ratios.format({"mtd":"{:.0f}","2m":"{:.0f}","3m":"{:.0f}","6m":"{:.0f}","9m":"{:.0f}",
#                                             "12m":"{:.0f}","18m":"{:.0f}"}))

filtered_col = ["Ticker","sector","3m","6m","12m","18m"]

hit_3m=concat_data(hit_ratios,3,"Hit Rate (% months outperf Market)")
avg_return_3m=concat_data(sector_avg_returns,3,"Avg Monthly Return")
excess_3m=concat_data(excess_return,3,"Avg Monthly Excess Return")
indicator_3m = pd.concat([hit_3m,avg_return_3m,excess_3m])
agg_zscore_3m = agg_zscore(indicator_3m)
indicator_3m.loc["Agg Z-Score 3m"] = agg_zscore_3m
indicator_3m = indicator_3m.astype('float64')
styled_indicator_3m = indicator_3m.style.apply(highlight_values,axis=None)


hit_6m=concat_data(hit_ratios,6,"Hit Rate (% months outperf Market)")
avg_return_6m=concat_data(sector_avg_returns,6,"Avg Monthly Return")
excess_6m=concat_data(excess_return,6,"Avg Monthly Excess Return")
indicator_6m = pd.concat([hit_6m,avg_return_6m,excess_6m])
agg_zscore_6m = agg_zscore(indicator_6m)
indicator_6m.loc["Agg Z-Score 6m"] = agg_zscore_6m
indicator_6m = indicator_6m.astype('float64')


hit_12m=concat_data(hit_ratios,12,"Hit Rate (% months outperf Market)")
avg_return_12m=concat_data(sector_avg_returns,12,"Avg Monthly Return")
excess_12m=concat_data(excess_return,12,"Avg Monthly Excess Return")
indicator_12m = pd.concat([hit_12m,avg_return_12m,excess_12m])
agg_zscore_12m = agg_zscore(indicator_12m)
indicator_12m.loc["Agg Z-Score 12m"] = agg_zscore_12m
indicator_12m = indicator_12m.astype('float64')


hit_18m=concat_data(hit_ratios,18,"Hit Rate (% months outperf Market)")
avg_return_18m=concat_data(sector_avg_returns,18,"Avg Monthly Return")
excess_18m=concat_data(excess_return,18,"Avg Monthly Excess Return")
indicator_18m = pd.concat([hit_18m,avg_return_18m,excess_18m])
agg_zscore_18m = agg_zscore(indicator_18m)
indicator_18m.loc["Agg Z-Score 18m"] = agg_zscore_18m
indicator_18m = indicator_18m.astype('float64')

agg_z = pd.concat([indicator_3m.loc["Agg Z-Score 3m",:],indicator_6m.loc["Agg Z-Score 6m",:],indicator_12m.loc["Agg Z-Score 12m",:],indicator_18m.loc["Agg Z-Score 18m",:]],axis=1).T

st.markdown("### Z-Score")
st.dataframe(agg_z.style.apply(highlight_values,axis=None).format({col:"{:.1f}" for col in indicator_18m.columns}),width=1400)

excess_check = st.checkbox("Display Avg Monthly Excess Return")

if excess_check:
    st.markdown("### Avg Monthly Excess Return ")
    st.dataframe(styled_excess_returns.format({"mtd":"{:.2f}","2m":"{:.2f}","3m":"{:.2f}","6m":"{:.2f}","9m":"{:.2f}",
                                                "12m":"{:.2f}","18m":"{:.2f}"}),width=1150,height=420,hide_index=True)

st.markdown("## Short Term")
st.markdown("### 3m")
st.dataframe(styled_indicator_3m.format({col:"{:.1f}" for col in indicator_3m.columns}),width=1400)

st.markdown("### 6m")
st.dataframe(indicator_6m.style.apply(highlight_values,axis=None).format({col:"{:.1f}" for col in indicator_6m.columns}),width=1400)

st.markdown("## Long Term")
st.markdown("### 12m")
st.dataframe(indicator_12m.style.apply(highlight_values,axis=None).format({col:"{:.1f}" for col in indicator_12m.columns}),width=1400)

st.markdown("### 18m")
st.dataframe(indicator_18m.style.apply(highlight_values,axis=None).format({col:"{:.1f}" for col in indicator_18m.columns}),width=1400)
