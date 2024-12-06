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
st.set_page_config(page_title="Business Cycle",layout="wide")

def hit_ratio(period):
  if period=="MtD":
      period = -1
  comparison = etf_returns.loc[sectors,etf_returns.columns[-period:]].sub(etf_returns.loc[broad_market,etf_returns.columns[-period:]])
  positive_ratio = pd.DataFrame(comparison.gt(0).sum(axis=1)/len(etf_returns.columns[-period:]))
  if period==-1:
    positive_ratio.columns = ["MtD"]
  else:
    positive_ratio.columns = [str(period)+"m"]
  return positive_ratio
def concat_data(data,period,label_indicator):
    if period =="MtD":
        df = data[["sector","MtD"]].T
    else:
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

def style_cycle_column(cycles):
    # Define the mapping of cycle stages to colors
    color_mapping = {
        "Expansion": 'rgba(0, 255, 0, 0.8)',      # Green
        "Recovery": 'rgba(0, 205, 0, 0.8)',       # Lighter green
        "Slowdown": 'rgba(205, 0, 0, 0.4)',       # Light red
        "Recession": 'rgba(255, 0, 0, 0.8)'       # Dark red
    }
    
    # Process each cycle stage in the list
    styled_values = [
        f'background-color: {color_mapping[cycle]}' if cycle in color_mapping else ''
        for cycle in cycles
    ]
    return "; ".join(styled_values)

def ranking_color(df):
    styled_df = df.copy()
    
    for col in styled_df.columns:
        if col=="":
               styled_df[col] = styled_df[col].apply(lambda x: 
            'background-color: rgba(0, 255, 0, 0.8)' if x == "++" else
            'background-color: rgba(0, 205, 0, 0.8)' if x == "+" else 
            'background-color: rgba(205,0, 0, 0.4)' if x == "-" else
            'background-color: rgba(355,0, 0, 0.8)' if x == "--" else ''
        )
        elif col == "Cycle":
            styled_df[col] = styled_df[col].apply(lambda x: style_cycle_column(x) if isinstance(x, list) else '')
        else:
            styled_df[col] = ''
    return styled_df
         
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
sector_roadmap = {"Expansion":{"++":["Financials","Technology"],
                               "+":["Communication"],
                               "-":["Consumer Staples"],
                               "--":["Health Care","Utilities"]},        
                    "Recovery":{"++":["Consumer Discretionary","Real Estate"],
                                "+":["Materials"],
                               "-":["Health Care"],
                               "--":["Consumer Staples","Utilities"]},
                    "Slowdown":{"++":["Consumer Staples","Health Care"],
                                "+":["Industrials"],
                               "-":["Materials"],
                               "--":["Consumer Discretionary","Real Estate"]},
                    "Recession":{"++":["Consumer Staples","Utilities"],
                                 "+":["Health"],
                               "-":["Communication"],
                               "--":["Real Estate","Technology"]}}
sectors = list(spdr_sector_etfs.values())
sectors.remove("SCHB")
broad_market = "SCHB"
etf_prices = yf.download(list(spdr_sector_etfs.values()), start="2022-01-01", end="2024-12-05", interval="1d")["Adj Close"]
etf_returns = round(100*(etf_prices.resample("M").last()/etf_prices.resample("M").first() - 1),2).T
avg_returns = etf_returns.copy()
avg_returns["MtD"] = etf_returns.iloc[:,-1]
avg_returns["2m"] = etf_returns.iloc[:,-2:].mean(axis=1)
avg_returns["3m"] = etf_returns.iloc[:,-3:].mean(axis=1)
avg_returns["6m"] = etf_returns.iloc[:,-6:].mean(axis=1)
avg_returns["9m"] = etf_returns.iloc[:,-9:].mean(axis=1)
avg_returns["12m"] = etf_returns.iloc[:,-12:].mean(axis=1)
avg_returns["18m"] = etf_returns.iloc[:,-18:].mean(axis=1)
avg_returns = avg_returns[["MtD","2m","3m","6m","9m","12m","18m"]]

excess_return = round(avg_returns.loc[sectors,:] - avg_returns.loc[broad_market,:],2)
excess_return.reset_index(inplace=True)
excess_return["sector"] = excess_return["Ticker"].map({spdr_sector_etfs[v]:v for k,v in enumerate(spdr_sector_etfs)})
excess_return = excess_return[["Ticker","sector","MtD","3m","6m","12m","18m"]]
avg_returns.reset_index(inplace=True)
avg_returns["sector"] = avg_returns["Ticker"].map({spdr_sector_etfs[v]:v for k,v in enumerate(spdr_sector_etfs)})
avg_returns = avg_returns[["Ticker","sector","MtD","3m","6m","12m","18m"]]

# st.markdown("### Average return per Sectors")
# styled_avg_returns = avg_returns.style.apply(highlight_values,axis=None)
# st.dataframe(styled_avg_returns.format({"MtD":"{:.2f}","2m":"{:.2f}","3m":"{:.2f}","6m":"{:.2f}","9m":"{:.2f}",
#                                             "12m":"{:.2f}","18m":"{:.2f}"}),width=1150,height=460,hide_index=True)


styled_excess_returns = excess_return.style.apply(highlight_values,axis=None)

hit_ratios_list= []
for period in ["MtD",3,6,12,18]:
  sub_df = round(hit_ratio(period),2)*100
  hit_ratios_list.append(sub_df)
hit_ratios = pd.concat(hit_ratios_list,axis=1)
hit_ratios.reset_index(inplace=True)
hit_ratios["sector"] = hit_ratios["Ticker"].map({spdr_sector_etfs[v]:v for k,v in enumerate(spdr_sector_etfs)})
styled_hit_ratios = hit_ratios.style.apply(highlight_values,axis=None)
sector_avg_returns = avg_returns.loc[avg_returns["Ticker"].isin(sectors)]
# st.dataframe(styled_hit_ratios.format({"MtD":"{:.0f}","2m":"{:.0f}","3m":"{:.0f}","6m":"{:.0f}","9m":"{:.0f}",
#                                             "12m":"{:.0f}","18m":"{:.0f}"}))

filtered_col = ["Ticker","sector","3m","6m","12m","18m"]

hit_mtd=concat_data(hit_ratios,"MtD","Hit Rate (% months outperf Market)")
avg_return_mtd=concat_data(sector_avg_returns,"MtD","Avg Monthly Return")
excess_mtd=concat_data(excess_return,"MtD","Avg Monthly Excess Return")
indicator_mtd = pd.concat([hit_mtd,avg_return_mtd,excess_mtd])
agg_zscore_mtd = agg_zscore(indicator_mtd)
indicator_mtd.loc["Agg Score MtD"] = agg_zscore_mtd
indicator_mtd = indicator_mtd.astype('float64')
styled_indicator_mtd = indicator_mtd.style.apply(highlight_values,axis=None)



hit_3m=concat_data(hit_ratios,3,"Hit Rate (% months outperf Market)")
avg_return_3m=concat_data(sector_avg_returns,3,"Avg Monthly Return")
excess_3m=concat_data(excess_return,3,"Avg Monthly Excess Return")
indicator_3m = pd.concat([hit_3m,avg_return_3m,excess_3m])
agg_zscore_3m = agg_zscore(indicator_3m)
indicator_3m.loc["Agg Score 3m"] = agg_zscore_3m
indicator_3m = indicator_3m.astype('float64')
styled_indicator_3m = indicator_3m.style.apply(highlight_values,axis=None)


hit_6m=concat_data(hit_ratios,6,"Hit Rate (% months outperf Market)")
avg_return_6m=concat_data(sector_avg_returns,6,"Avg Monthly Return")
excess_6m=concat_data(excess_return,6,"Avg Monthly Excess Return")
indicator_6m = pd.concat([hit_6m,avg_return_6m,excess_6m])
agg_zscore_6m = agg_zscore(indicator_6m)
indicator_6m.loc["Agg Score 6m"] = agg_zscore_6m
indicator_6m = indicator_6m.astype('float64')


hit_12m=concat_data(hit_ratios,12,"Hit Rate (% months outperf Market)")
avg_return_12m=concat_data(sector_avg_returns,12,"Avg Monthly Return")
excess_12m=concat_data(excess_return,12,"Avg Monthly Excess Return")
indicator_12m = pd.concat([hit_12m,avg_return_12m,excess_12m])
agg_zscore_12m = agg_zscore(indicator_12m)
indicator_12m.loc["Agg Score 12m"] = agg_zscore_12m
indicator_12m = indicator_12m.astype('float64')


hit_18m=concat_data(hit_ratios,18,"Hit Rate (% months outperf Market)")
avg_return_18m=concat_data(sector_avg_returns,18,"Avg Monthly Return")
excess_18m=concat_data(excess_return,18,"Avg Monthly Excess Return")
indicator_18m = pd.concat([hit_18m,avg_return_18m,excess_18m])
agg_zscore_18m = agg_zscore(indicator_18m)
indicator_18m.loc["Agg Score 18m"] = agg_zscore_18m
indicator_18m = indicator_18m.astype('float64')

agg_z = pd.concat([indicator_mtd.loc["Agg Score MtD",:],indicator_3m.loc["Agg Score 3m",:],indicator_6m.loc["Agg Score 6m",:],indicator_12m.loc["Agg Score 12m",:],indicator_18m.loc["Agg Score 18m",:]],axis=1).T

st.markdown("### Sector Scores for Business Cycles")
col1,col2,col3,col4,col5,col6 = st.columns(6,gap="small")

with col1:
   agg_z_check = st.checkbox("All periods")
with col2:
   agg_z_check_3m = st.checkbox("3m")
with col3:
   agg_z_check_6m = st.checkbox("6m")
with col4:
   agg_z_check_12m = st.checkbox("12m")
with col5:
   agg_z_check_18m = st.checkbox("18m")
periods = ["Agg Score MtD"]
if agg_z_check:
    periods = list(agg_z.index)
if agg_z_check_3m:
   periods.append("Agg Score 3m")
if agg_z_check_6m:
   periods.append("Agg Score 6m")
if agg_z_check_12m:
   periods.append("Agg Score 12m")
if agg_z_check_18m:
   periods.append("Agg Score 18m")
st.dataframe(agg_z.loc[periods,:].style.apply(highlight_values,axis=None).format({col:"{:.2f}" for col in indicator_18m.columns}),width=1400)

roadmap_period = st.selectbox("Period",options=["MtD","3m","6m","12m","18m"])
sorted_zscores = (agg_z.loc[["Agg Score "+roadmap_period],:].T).sort_values(by="Agg Score " +roadmap_period,ascending=False)
sector_ranking = list(sorted_zscores.head(3).index)+list(sorted_zscores.tail(3).index)
# Sample data for sectors and rankings
cycle_mapping = []
all_cycles = []


for index in range(len(sector_ranking)):
    cycle_choice = []
    for cycle in ["Recession","Slowdown","Recovery","Expansion"]:
        if index <=2:
            if sector_ranking[index] in sector_roadmap[cycle]["++"]:
                cycle_choice.append(cycle)
                all_cycles.append(cycle)
            if sector_ranking[index] in sector_roadmap[cycle]["+"]:
                cycle_choice.append(cycle)
                all_cycles.append(cycle)
        else:
            if sector_ranking[index] in sector_roadmap[cycle]["-"]:
                cycle_choice.append(cycle)
                all_cycles.append(cycle)
            if sector_ranking[index] in sector_roadmap[cycle]["--"]:
                cycle_choice.append(cycle)
                all_cycles.append(cycle)
    cycle_mapping.append(cycle_choice)
cycle_count_dict = {cycle:[round(all_cycles.count(cycle)/len(all_cycles),2)] for cycle in list(set(all_cycles))}
all_cycles_count = pd.DataFrame(cycle_count_dict)
all_cycles_count.index = ["Descriptive Proba %"]

ranking = ["++","","+","-","","--"]
roadmap_final = pd.DataFrame({"":ranking,"MtD":sector_ranking,"Cycle":cycle_mapping})
col1,col2 = st.columns(2,gap="small")
with col1:
    st.markdown("### Sector Road Map")
    st.dataframe(roadmap_final.style.apply(ranking_color,axis=None),hide_index=True)
with col2:
    st.markdown("### Market Cycle")
    st.write(all_cycles_count)
# st.markdown("### MtD")
st.dataframe(styled_indicator_mtd.format({col:"{:.1f}" for col in indicator_3m.columns}),width=1400)
excess_check = st.checkbox("Display Avg Monthly Excess Return")
short_term = st.expander("Short Term 3-6m")
long_term = st.expander("Long Term 12-18m")
if excess_check:
    st.markdown("### Avg Monthly Excess Return ")
    st.dataframe(styled_excess_returns.format({"MtD":"{:.2f}","2m":"{:.2f}","3m":"{:.2f}","6m":"{:.2f}","9m":"{:.2f}",
                                                "12m":"{:.2f}","18m":"{:.2f}"}),width=1150,height=420,hide_index=True)
with short_term:
    st.markdown("## Short Term")
    st.markdown("### 3m")
    st.dataframe(styled_indicator_3m.format({col:"{:.1f}" for col in indicator_3m.columns}),width=1400)

    st.markdown("### 6m")
    st.dataframe(indicator_6m.style.apply(highlight_values,axis=None).format({col:"{:.1f}" for col in indicator_6m.columns}),width=1400)

with long_term:
    st.markdown("## Long Term")
    st.markdown("### 12m")
    st.dataframe(indicator_12m.style.apply(highlight_values,axis=None).format({col:"{:.1f}" for col in indicator_12m.columns}),width=1400)

    st.markdown("### 18m")
    st.dataframe(indicator_18m.style.apply(highlight_values,axis=None).format({col:"{:.1f}" for col in indicator_18m.columns}),width=1400)
