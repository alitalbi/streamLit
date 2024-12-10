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

st.set_page_config(page_title="Business Cycle",layout="wide",initial_sidebar_state="collapsed")

# Set up the top navigation bar with buttons (no new tabs will open)
st.markdown("""
    <style>
    .top-nav {
        display: flex;
        justify-content: space-around;
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
        padding: 15px 0;  /* Increased padding to add more space above the navbar */
        text-align: center;
        display: flex;
        justify-content: center;  /* Center items */
        align-items: center;
        margin-bottom: 20px; /* Adds more space below the navbar */
        border-radius: 10px; /* Rounded corners for the navbar */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Adds subtle shadow to lift the navbar */
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

# Render the navigation bar
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

def hit_ratio(etf_returns,period):
    if period=="MtD":
        period = -1
        comparison = etf_returns.loc[sectors,etf_returns.columns[-period:]].sub(etf_returns.loc[broad_market,etf_returns.columns[-period:]])
        positive_ratio = pd.DataFrame(comparison.gt(0).sum(axis=1)/len(etf_returns.columns[-period:]))
        positive_ratio.columns = ["MtD"]
    elif period == "daily":
        comparison = etf_returns.loc[sectors,:]-etf_returns.loc[broad_market,:]
        positive_ratio = pd.DataFrame(np.where(comparison>0,1,0))
        positive_ratio.columns = comparison.columns
        positive_ratio.index = comparison.index
        positive_ratio = positive_ratio.T.rolling(22).mean().T
        positive_ratio.reset_index(inplace=True)
        positive_ratio.dropna(axis=1,inplace=True)
    else:
        comparison = etf_returns.loc[sectors,etf_returns.columns[-period:]].sub(etf_returns.loc[broad_market,etf_returns.columns[-period:]])
        positive_ratio = pd.DataFrame(comparison.gt(0).sum(axis=1)/len(etf_returns.columns[-period:]))
        positive_ratio.columns = [str(period)+"m"]
    return positive_ratio

def agg_market_cycles(subdf_):
    subdf_ = subdf_.T
    
    all_rankings_sector = []
    for col_count in range(len(subdf_.columns)):
        sample = pd.DataFrame(subdf_.iloc[:,col_count])
        sample.sort_values(by=sample.columns[0],ascending=False,inplace=True)
        all_rankings_sector.append(list(sample.head(3).index)+list(sample.tail(3).index))
    dates = []
    all_rankings_df = []
    for subsector_ranking in all_rankings_sector:
        all_cycles = []
        cycle_mapping = []
        for index in range(len(subsector_ranking)):
            cycle_choice = []
            for cycle in ["Recession","Slowdown","Recovery","Expansion"]:
                if index <=2:
                    if subsector_ranking[index] in sector_roadmap[cycle]["++"]:
                        cycle_choice.append(cycle)
                        all_cycles.append(cycle)
                    if subsector_ranking[index] in sector_roadmap[cycle]["+"]:
                        cycle_choice.append(cycle)
                        all_cycles.append(cycle)
                else:
                    if subsector_ranking[index] in sector_roadmap[cycle]["-"]:
                        cycle_choice.append(cycle)
                        all_cycles.append(cycle)
                    if subsector_ranking[index] in sector_roadmap[cycle]["--"]:
                        cycle_choice.append(cycle)
                        all_cycles.append(cycle)
        cycle_mapping.append(cycle_choice)

        cycle_count_dict = {cycle:[round(all_cycles.count(cycle)/len(all_cycles),2)] for cycle in list(set(all_cycles))}
       # cycle_count_dict["Date"] = dates
        all_rankings_df.append(pd.DataFrame(cycle_count_dict))
    
    agg_all_cycles = pd.concat(all_rankings_df)
    agg_all_cycles.index = subdf_.columns
    
    return agg_all_cycles
    # return all_cycles_count

def concat_data(data,period,label_indicator):
    if period =="MtD":
        df = data[["sector","MtD"]].T  
    else:
       df = data[["sector",str(period)+"m"]].T

    df.columns= df.loc["sector",:]
    df.drop("sector",axis=0,inplace=True)
    
    df.index = [label_indicator]
    return df



def agg_zscore(df,type):
    if type==1:

        z_1=pd.DataFrame((df.iloc[0,:]-df.iloc[0,:].mean())/df.iloc[0,:].std()).T
        z_2=pd.DataFrame((df.iloc[1,:]-df.iloc[1,:].mean())/df.iloc[1,:].std()).T
        z_3=pd.DataFrame((df.iloc[2,:]-df.iloc[2,:].mean())/df.iloc[2,:].std()).T
        agg_z_score = pd.concat([z_1,z_2,z_3]).mean(axis=0)
    else:
        df["mean"] = df.mean(axis=1)
        df["std"] = df.std(axis=1)
        df.iloc[:,:11] =(np.array(df.iloc[:,:11])-np.array(df.iloc[:,11]).reshape(-1,1))/np.array(df.iloc[:,12]).reshape(-1,1)
        #df.loc[:,df.columns[:len(df.columns)-2]] = df.loc[:,df.columns[:len(df.columns)-2]])
        df.drop(["mean","std"],axis=1,inplace=True)
        agg_z_score = df.copy()
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
etf_prices = yf.download(list(spdr_sector_etfs.values()), start="2002-01-01", end=datetime.today().strftime("%Y-%m-%d"), interval="1d")["Adj Close"]
### avg saily dev ###------------------------------------
avg_daily = (etf_prices.pct_change(1).rolling(22).mean()*100)

avg_daily.dropna(inplace=True)
avg_daily = avg_daily.T
avg_daily_copy = avg_daily.copy()
excess_avg_daily_return = round(avg_daily.loc[sectors,:] - avg_daily.loc[broad_market,:],2)
avg_daily.reset_index(inplace=True)

excess_avg_daily_return.reset_index(inplace=True)
avg_daily["sector"] = avg_daily["Ticker"].map({spdr_sector_etfs[v]:v for k,v in enumerate(spdr_sector_etfs)})
### avg saily dev ###------------------------------------


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

### avg saily dev ###------------------------------------
daily_avg_hit_ratio = hit_ratio(avg_daily_copy,"daily")
### avg saily dev ###------------------------------------


#daily_avg_hit_ratio["sector"] = daily_avg_hit_ratio["Ticker"].map({spdr_sector_etfs[v]:v for k,v in enumerate(spdr_sector_etfs)})
hit_ratios_list= []
for period in ["MtD",3,6,12,18]:
  sub_df = round(hit_ratio(etf_returns,period),2)*100
  hit_ratios_list.append(sub_df)
hit_ratios = pd.concat(hit_ratios_list,axis=1)
hit_ratios.reset_index(inplace=True)
hit_ratios["sector"] = hit_ratios["Ticker"].map({spdr_sector_etfs[v]:v for k,v in enumerate(spdr_sector_etfs)})
styled_hit_ratios = hit_ratios.style.apply(highlight_values,axis=None)
sector_avg_returns = avg_returns.loc[avg_returns["Ticker"].isin(sectors)]

### avg saily dev ###------------------------------------
avg_daily_copy.reset_index(inplace=True)
for df in ["daily_avg_hit_ratio","excess_avg_daily_return","avg_daily_copy"]:
    exec(df+"[\"sector\"] = "+df+"[\"Ticker\"].map({spdr_sector_etfs[v]:v for k,v in enumerate(spdr_sector_etfs)})")
    exec(df+".drop(\"Ticker\",axis=1,inplace=True)")
    exec(df+".set_index(\"sector\",inplace=True)")
    exec(df+"="+df+".T")

avg_daily_copy.drop("Broad US Market",axis=1,inplace=True)
# st.dataframe(styled_hit_ratios.format({"MtD":"{:.0f}","2m":"{:.0f}","3m":"{:.0f}","6m":"{:.0f}","9m":"{:.0f}",
#                                             "12m":"{:.0f}","18m":"{:.0f}"}))
common_index = daily_avg_hit_ratio.index.intersection(excess_avg_daily_return.index).intersection(avg_daily_copy.index)
### avg saily dev ###------------------------------------------------ 
daily_avg_hit_ratio = daily_avg_hit_ratio.loc[common_index]
excess_avg_daily_return = excess_avg_daily_return.loc[common_index]
avg_daily_copy = avg_daily_copy.loc[common_index]

filtered_col = ["Ticker","sector","3m","6m","12m","18m"]
#hit_avg_daily = concat_data(daily_avg_hit_ratio,"","Hit Rate (% months outperf Market)")
hit_mtd=concat_data(hit_ratios,"MtD","Hit Rate (% months outperf Market)")
avg_return_mtd=concat_data(sector_avg_returns,"MtD","Avg Monthly Return")
excess_mtd=concat_data(excess_return,"MtD","Avg Monthly Excess Return")
indicator_mtd = pd.concat([hit_mtd,avg_return_mtd,excess_mtd])
agg_zscore_daily_avg = agg_zscore(excess_avg_daily_return,2)
agg_zscore_hit_avg = agg_zscore(daily_avg_hit_ratio,2)
agg_zscore_excess_avg = agg_zscore(excess_avg_daily_return,2)

agg_zscores_all_daily_avg = (agg_zscore_excess_avg+agg_zscore_daily_avg+agg_zscore_hit_avg)/3
agg_market_cycle_df = agg_market_cycles(agg_zscores_all_daily_avg)
# st.dataframe(agg_zscores_all_daily_avg.style.apply(highlight_values,axis=None))
agg_zscore_mtd = agg_zscore(indicator_mtd,1)
indicator_mtd.loc["Agg Score MtD"] = agg_zscore_mtd
indicator_mtd = indicator_mtd.astype('float64')
styled_indicator_mtd = indicator_mtd.style.apply(highlight_values,axis=None)



hit_3m=concat_data(hit_ratios,3,"Hit Rate (% months outperf Market)")
avg_return_3m=concat_data(sector_avg_returns,3,"Avg Monthly Return")
excess_3m=concat_data(excess_return,3,"Avg Monthly Excess Return")
indicator_3m = pd.concat([hit_3m,avg_return_3m,excess_3m])
agg_zscore_3m = agg_zscore(indicator_3m,1)
indicator_3m.loc["Agg Score 3m"] = agg_zscore_3m
indicator_3m = indicator_3m.astype('float64')
styled_indicator_3m = indicator_3m.style.apply(highlight_values,axis=None)


hit_6m=concat_data(hit_ratios,6,"Hit Rate (% months outperf Market)")
avg_return_6m=concat_data(sector_avg_returns,6,"Avg Monthly Return")
excess_6m=concat_data(excess_return,6,"Avg Monthly Excess Return")
indicator_6m = pd.concat([hit_6m,avg_return_6m,excess_6m])
agg_zscore_6m = agg_zscore(indicator_6m,1)
indicator_6m.loc["Agg Score 6m"] = agg_zscore_6m
indicator_6m = indicator_6m.astype('float64')


hit_12m=concat_data(hit_ratios,12,"Hit Rate (% months outperf Market)")
avg_return_12m=concat_data(sector_avg_returns,12,"Avg Monthly Return")
excess_12m=concat_data(excess_return,12,"Avg Monthly Excess Return")
indicator_12m = pd.concat([hit_12m,avg_return_12m,excess_12m])
agg_zscore_12m = agg_zscore(indicator_12m,1)
indicator_12m.loc["Agg Score 12m"] = agg_zscore_12m
indicator_12m = indicator_12m.astype('float64')


hit_18m=concat_data(hit_ratios,18,"Hit Rate (% months outperf Market)")
avg_return_18m=concat_data(sector_avg_returns,18,"Avg Monthly Return")
excess_18m=concat_data(excess_return,18,"Avg Monthly Excess Return")
indicator_18m = pd.concat([hit_18m,avg_return_18m,excess_18m])
agg_zscore_18m = agg_zscore(indicator_18m,1)
indicator_18m.loc["Agg Score 18m"] = agg_zscore_18m
indicator_18m = indicator_18m.astype('float64')

agg_z = pd.concat([indicator_mtd.loc["Agg Score MtD",:],indicator_3m.loc["Agg Score 3m",:],indicator_6m.loc["Agg Score 6m",:],indicator_12m.loc["Agg Score 12m",:],indicator_18m.loc["Agg Score 18m",:]],axis=1).T


agg_market_cycle_df = agg_market_cycle_df.loc[:,["Recession","Slowdown","Recovery","Expansion"]]
agg_market_cycle_df["Total %"] = agg_market_cycle_df.sum(axis=1)

agg_market_cycle_df.fillna(0,inplace=True)

st.markdown(
    """
    <h3 style="text-align: center;">Sector Scores for Business Cycles</h3>
    """,
    unsafe_allow_html=True
)
col1,col2,col3 = st.columns(3,gap="small")
with col2:
    use_custom_date = st.expander("Custom Graph",expanded=False)
    with use_custom_date:
        cycle_graph_choice = st.selectbox("Graph type : ",options=["Stacked Bar","Line Areas"])
        start_date = st.date_input("Start date:", pd.Timestamp("2024-01-01"))  
        end_date = st.date_input("End date:", pd.Timestamp(datetime.today()))  
agg_market_cycle_df.index = pd.Series(agg_market_cycle_df.index).apply(lambda x:x.tz_localize(None))
agg_market_cycle_df = agg_market_cycle_df.loc[(agg_market_cycle_df.index > pd.Timestamp(start_date)) & (agg_market_cycle_df.index <= pd.Timestamp(end_date))]
fig = go.Figure()

if cycle_graph_choice == "Stacked Bar": 
# Add traces for each category
    fig.add_trace(go.Bar(
        x=agg_market_cycle_df.index.to_list(),
        y=agg_market_cycle_df["Recession"],
        name="Recession",
        marker=dict(color="red"),
        hoverinfo="x+y+name"
    ))

    fig.add_trace(go.Bar(
        x=agg_market_cycle_df.index.to_list(),
        y=agg_market_cycle_df["Recovery"],
        name="Recovery",
        marker=dict(color="white"),
        hoverinfo="x+y+name"
    ))

    fig.add_trace(go.Bar(
        x=agg_market_cycle_df.index.to_list(),
        y=agg_market_cycle_df["Slowdown"],
        name="Slowdown",
        marker=dict(color="orange"),
        hoverinfo="x+y+name"
    ))

    fig.add_trace(go.Bar(
        x=agg_market_cycle_df.index.to_list(),
        y=agg_market_cycle_df["Expansion"],
        name="Expansion",
        marker=dict(color="green"),
        hoverinfo="x+y+name"
    ))

    # Update layout for stacked bar chart
    fig.update_layout(
        barmode="stack",
        title="Contribution to Total %",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Total Percentage"),
        legend=dict(title="Category"),
        template="plotly_dark"
    )
else:

# Add traces for each category
    categories = ["Recession", "Recovery", "Slowdown", "Expansion"]
    colors = ["red", "white", "orange", "green"]

    for category, color in zip(categories, colors):
        fig.add_trace(go.Scatter(
            x=agg_market_cycle_df.index.to_list(),
            y=agg_market_cycle_df[category],
            mode="lines",
            name=category,
            stackgroup="one",  # Enables stacking
            line=dict(color=color),
            hoverinfo="x+y+name"
        ))
# st.write(agg_market_cycle_df)
fig.update_layout(  # customize font and legend orientation & position
    yaxis=dict(tickformat=".1%",title="Descriptive Proba "),
    title_font_family="Arial Black",
    title={
            'text' : "Aggregate Market Cycle",
            'x':0.5,
            'xanchor': 'center'
        },
    font=dict(
        family="Rockwell",
        size=18),   
    legend=dict(
        title="Cycle", orientation="v", y=0.97, yanchor="bottom", x=0.9, xanchor="left"
    ))

st.plotly_chart(fig)

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
period_options = ["MtD","3m","6m","12m","18m"]
roadmap_period = st.selectbox("Period",options=period_options)

# sub_market_cycle = []
# for period in period_options:
#     sub_market_cycle.append(agg_market_cycles(period))
# agg_market_cycle = pd.concat(sub_market_cycle)[::-1]
# agg_market_cycle.fillna(0,inplace = True)

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


    
