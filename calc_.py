import plotly.graph_objects as go
import plotly.express as px
from fredapi import Fred
import pandas as pd
import wget
#import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from datetime import timedelta
import socket
import os
import datetime
import numpy as np
import yfinance as yf




cwd = os.getcwd() + "/"

def smooth_data(ticker, date_start, date_start2, date_end):
    frequency = "monthly"
    fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')
    data_ = pd.DataFrame(
        fred.get_series(ticker, observation_start=date_start, observation_end=date_end, freq=frequency))
    data_2 = pd.DataFrame(
        fred.get_series(ticker, observation_start=date_start2, observation_end=date_end, freq=frequency))

    # creating 6m smoothing growth column and 10yr average column
    # Calculate the smoothed average
    average = data_.iloc[:, 0].rolling(11).mean()

    # Calculate the annualized growth rate
    annualized_6m_smoothed_growth_rate = (data_.iloc[:, 0][11:] / average) ** (365 / 180) - 1

    # Multiply the result by 100 and store it in the _6m_smoothing_growth column
    data_['_6m_smoothing_growth'] = 100 * annualized_6m_smoothed_growth_rate
    data_2['mom_average'] = 100 * data_2.iloc[:, 0].pct_change(periods=1)
    data_2['10 yr average'] = data_2['mom_average'].rolling(120).mean()
    data_.dropna(inplace=True)
    data_2.dropna(inplace=True)
    return data_[['_6m_smoothing_growth']], data_2[['10 yr average']]

def replicate_row(pcec96):
    pcec96_index = pcec96.index.to_list()
    index_add = datetime.datetime(2023, 1, 1)
    pcec96_index.append(index_add)
    if pcec96.index.to_list()[-1].month != datetime.datetime.now().month - 1:
        pcec96.loc[len(pcec96), :] = pcec96.iloc[-1, :]
        pcec96.index = pcec96_index
    else:
        pass
    return pcec96

def score_table(index, data_, data_10):
    score_table = pd.DataFrame.from_dict({"trend vs history ": 1 if data_.iloc[-1, 0] > data_10.iloc[-1, 0] else 0,
                                          "growth": 1 if data_.iloc[-1, 0] > 0 else 0,
                                          "Direction of Trend": 1 if (data_.resample("3M").last().diff()).iloc[-1][
                                                                         0] > 0 else 1}, orient="index").T
    score_table['Score'] = score_table.sum(axis=1)
    score_table['Indicators'] = index

    return score_table

def fred_data(ticker):
    date_start = "2017-01-01"
    date_end = "2022-05-27"
    frequency = 'monthly'

    fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')
    # get data as an array and transforming it into a dataframe
    return fred.get_series(ticker, observation_start=date_start, observation_end=date_end, freq=frequency)

date_start = "1960-01-01"
date_start2 = "2004-01-01"

date_end = datetime.datetime.now().strftime("%Y-%m-%d")
fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')

print("1")
# try:
PATH_DATA = "/Users/talbi/Downloads/"

def export_yfinance_data(date_start, date_end):

    # 30y nominal
    _30y = yf.download("^TYX", start=date_start, end=date_end, interval="1d")[['Close']]
    _30y.to_csv(cwd + "_30y.csv")

    # DXY
    dxy = yf.download("DX-Y.NYB", start=date_start, end=date_end, interval="1d")[['Close']]
    dxy.to_csv(cwd + "dxy.csv")

    # 5y nominal
    _5y_nominal = yf.download("^FVX", start=date_start, end=date_end, interval="1d")[['Close']]
    _5y_nominal.to_csv(cwd + "_5y_nominal.csv")

    # Cooper
    cooper = yf.download("HG=F", start=date_start, end=date_end, interval="1d")[['Close']]
    cooper_prices = cooper * 100
    cooper_prices.to_csv(cwd + "cooper_prices.csv")
    try:
        os.remove("wage-growth-data.xlsx")
    except FileNotFoundError:
        pass
    wget.download(
        "http://atlantafed.org/-/media/documents/datafiles/chcs/wage-growth-tracker/wage-growth-data.xlsx")

    # Wheat
    wheat = yf.download("ZW=F", start=date_start, end=datetime.datetime.now(), interval="1d")[['Close']]
    wheat.to_csv("wheat.csv")

    # Oil
    oil = yf.download("CL=F", start=date_start, end=date_end, interval="1d")[['Close']]
    oil.to_csv("oil.csv")

    # Gas
    gas = yf.download("NG=F", start=date_start, end=date_end, interval="1d")[['Close']]
    gas.to_csv("gas.csv")

    # Gas
    gasoline_fund = yf.download("UGA", start=date_start, end=date_end, interval="1d")[['Close']]
    gasoline_fund.to_csv("gasoline_fund.csv")
    return 0

def export_fred_data(date_start, date_end):

    _5y_real = pd.DataFrame(
        fred.get_series("DFII5", observation_start=date_start, observation_end=date_end, freq="daily"))
    _5y_real.columns = ['Close']
    #_5y_real.to_csv(cwd + "_5y_real.csv")

    cpi = fred.get_series("CPIAUCSL", observation_start=date_start, observation_end=date_end, freq="monthly")
    #cpi.to_csv(cwd + "cpi.csv")
    cpi_10 = fred.get_series("CPIAUCSL", observation_start=date_start2, observation_end=date_end, freq="monthly")
    #cpi_10.to_csv(cwd + "cpi10.csv")

    pcec96 = pd.DataFrame(
        fred.get_series("PCEC96", observation_start=date_start, observation_end=date_end, freq="monthly"))
    # pcec96 = replicate_row(pcec96)
    #pcec96.to_csv(cwd + "pce.csv")
    pcec96_10 = pd.DataFrame(
        fred.get_series("PCEC96", observation_start=date_start2, observation_end=date_end, freq="monthly"))
    # pcec96_10 = replicate_row(pcec96_10)
    #pcec96_10.to_csv(cwd + "pce10.csv")

    indpro = fred.get_series("INDPRO", observation_start=date_start, observation_end=date_end, freq="monthly")
    #indpro.to_csv(cwd + "indpro.csv")
    indpro_10 = fred.get_series("INDPRO", observation_start=date_start2, observation_end=date_end, freq="monthly")
    #indpro_10.to_csv(cwd + "indpro10.csv")

    nonfarm = fred.get_series("PAYEMS", observation_start=date_start, observation_end=date_end, freq="monthly")
    #nonfarm.to_csv(cwd + "nonfarm.csv")
    nonfarm_10 = fred.get_series("PAYEMS", observation_start=date_start2, observation_end=date_end, freq="monthly")
    #nonfarm_10.to_csv(cwd + "nonfarm10.csv")

    real_pers = pd.DataFrame(
        fred.get_series("W875RX1", observation_start=date_start, observation_end=date_end, freq="monthly"))
    # real_pers = replicate_row(real_pers)
    #real_pers.to_csv(cwd + "real_pers.csv")
    real_pers_10 = pd.DataFrame(
        fred.get_series("W875RX1", observation_start=date_start2, observation_end=date_end, freq="monthly"))
    # real_pers_10 = replicate_row(real_pers_10)
    #real_pers_10.to_csv(cwd + "real_pers10.csv")

    retail_sales = fred.get_series("RRSFS", observation_start=date_start, observation_end=date_end, freq="monthly")
    #retail_sales.to_csv(cwd + "retail_sales.csv")
    retail_sales_10 = fred.get_series("RRSFS", observation_start=date_start2, observation_end=date_end,
                                      freq="monthly")
    #retail_sales_10.to_csv(cwd + "retail_sales10.csv")

    employment_level = fred.get_series("CE16OV", observation_start=date_start, observation_end=date_end,
                                       freq="monthly")
    #employment_level.to_csv(cwd + 'employment_level.csv')
    employment_level_10 = fred.get_series("CE16OV", observation_start=date_start2, observation_end=date_end,
                                          freq="monthly")
    employment_level_10.to_csv(cwd + 'employment_level10.csv')

    wages = fred.get_series("CES0500000003", observation_start=date_start, observation_end=date_end, freq="monthly")
    #wages.to_csv(cwd + 'wages.csv')
    wages_10 = fred.get_series("CES0500000003", observation_start=date_start2, observation_end=date_end,
                               freq="monthly")
    #wages_10.to_csv(cwd + 'wages10.csv')

    core_cpi = fred.get_series("CPILFESL", observation_start=date_start, observation_end=date_end, freq="monthly")
    #core_cpi.to_csv(cwd + 'core_cpi.csv')
    core_cpi_10 = fred.get_series("CPILFESL", observation_start=date_start2, observation_end=date_end,
                                  freq="monthly")
    #core_cpi_10.to_csv(cwd + 'core_cpi10.csv')

    core_pce = fred.get_series("DPCCRC1M027SBEA", observation_start=date_start, observation_end=date_end,
                               freq="monthly")
    #core_pce.to_csv(cwd + 'core_pce.csv')
    core_pce_10 = fred.get_series("DPCCRC1M027SBEA", observation_start=date_start2, observation_end=date_end,
                                  freq="monthly")
    #core_pce_10.to_csv(cwd + 'core_pce10.csv')

    shelter_prices = fred.get_series("CUSR0000SAH1", observation_start=date_start, observation_end=date_end,
                                     freq="monthly")
    #shelter_prices.to_csv(cwd + 'shelter_prices.csv')
    shelter_prices_10 = fred.get_series("CUSR0000SAH1", observation_start=date_start2, observation_end=date_end,
                                        freq="monthly")
    #shelter_prices_10.to_csv(cwd + 'shelter_prices10.csv')

    return 0

"""
_30y = pd.read_csv(cwd+"_30y.csv")
_5y_real = pd.read_csv(cwd+"_5y_real.csv")
cooper_prices = pd.read_csv(cwd+"cooper_prices.csv")
_5y_nominal = pd.read_csv(cwd+"_5y_nominal.csv")
spread = _30y - _5y_real

merged_data = pd.concat([spread, _5y_nominal, cooper_prices], axis=1)
merged_data.dropna(inplace=True)
merged_data.columns = ["spread 30_5yr", "5y", "cooper", ]

# Data importing
print("2")
cpi, cpi_10 = smooth_data("CPIAUCSL", date_start, date_start2, date_end)
single_search, single_search_10 = smooth_data(ticker_fred, date_start, date_start2, date_end)
pcec96, pcec96_10 = smooth_data("PCEC96", date_start, date_start2, date_end)
# pcec96 = pcec96[["_6m_smoothing_growth"]]-cpi[["_6m_smoothing_growth"]]
# pcec96_10 = pcec96_10[['10 yr average']]-cpi_10[['10 yr average']]
print("3")
indpro, indpro_10 = smooth_data("INDPRO", date_start, date_start2, date_end)
print('4')
nonfarm, nonfarm_10 = smooth_data("PAYEMS", date_start, date_start2, date_end)
print("5")
real_pers, real_pers_10 = smooth_data("W875RX1", date_start, date_start2, date_end)
# real_pers = real_pers[["_6m_smoothing_growth"]] - cpi[["_6m_smoothing_growth"]]
# real_pers_10 = real_pers_10[['10 yr average']] - cpi_10[['10 yr average']]

retail_sales, retail_sales_10 = smooth_data("RRSFS", date_start, date_start2, date_end)
# retail_sales = retail_sales[["_6m_smoothing_growth"]] - cpi[["_6m_smoothing_growth"]]
# retail_sales_10 = retail_sales_10[['10 yr average']] - cpi_10[['10 yr average']]

employment_level, employment_level_10 = smooth_data("CE16OV", date_start, date_start2, date_end)
employment_level.dropna(inplace=True)
shelter_prices, shelter_prices_10 = smooth_data("CUSR0000SAH1", date_start, date_start2, date_end)
shelter_prices = shelter_prices[["_6m_smoothing_growth"]] - cpi[["_6m_smoothing_growth"]]
shelter_prices_10 = shelter_prices_10[['10 yr average']] - cpi_10[['10 yr average']]

cwd = os.getcwd()
# wget.download("http://atlantafed.org/-/media/documents/datafiles/chcs/wage-growth-tracker/wage-growth-data.xlsx")
wage_tracker = pd.DataFrame(pd.read_excel(cwd + "/wage-growth-data.xlsx").iloc[3:, [0, 11]])
wage_tracker.columns = ['date', "wage_tracker"]
wage_tracker.set_index('date', inplace=True)

wheat_ = yf.download("ZW=F", start=date_start, end=datetime.datetime.now(), interval="1d")[['Close']]
oil_ = yf.download("CL=F", start=date_start , end=date_end, interval="1d")[['Close']]
gas_ = yf.download("NG=F", start=date_start, end=date_end, interval="1d")[['Close']]

employment_level_wage_tracker = pd.concat([employment_level, wage_tracker], axis=1)
employment_level_wage_tracker.dropna(inplace=True)
wages, wages_10 = smooth_data("CES0500000003", date_start, date_start2, date_end)
wages = wages[["_6m_smoothing_growth"]] - cpi[["_6m_smoothing_growth"]]
wages_10 = wages_10[['10 yr average']] - cpi_10[['10 yr average']]

core_cpi, core_cpi_10 = smooth_data("CPILFESL", date_start, date_start2, date_end)
core_pce, core_pce_10 = smooth_data("DPCCRC1M027SBEA", date_start, date_start2, date_end)
"""

# if __name__ == "__main__":
#export_yfinance_data(date_start, date_end)
#export_fred_data(date_start, date_end)

