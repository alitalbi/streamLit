import streamlit as st
import pandas as pd
from fredapi import Fred
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime,timedelta
import requests
#fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')


st.set_page_config(page_title="Gavekal Research")

frequency = "monthly"
fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')

date_start = st.date_input("Start date:", pd.Timestamp("2021-01-01"))
print(type(date_start))
date_start2 = datetime.strptime("2004-01-01","%Y-%m-%d").date()
print(type(date_start2))
date_end = st.date_input("End date:", pd.Timestamp(datetime.now().strftime("%Y-%m-%d")))

nominalGDPUS = "NGDPSAXDCUSQ"

def fred_data(internal_ticker,date_start,date_end):
    date_start= (date_start - timedelta(days=365)).strftime("%Y-%m-%d")
    #print(date_start)
    data_ = pd.DataFrame(
        fred.get_series(internal_ticker, observation_start=date_start, observation_end=date_end))


    data_ = data_.loc[(data_.index > date_start) & (data_.index < date_end)]
    data_.index = pd.to_datetime(data_.index)

    return data_

print(fred_data(nominalGDPUS,date_start,date_end))