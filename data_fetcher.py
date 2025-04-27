import yfinance as yf
import pandas as pd
import numpy as np
def fetch_yahoo_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)[['Close']]
    df.dropna(inplace=True)
    return df

# You can extend this with investpy or static files for futures/indices
# For example:
# def fetch_static_csv(path): ...
sp = fetch_yahoo_data("^GSPC","2020-01-01","2020-12-31")

