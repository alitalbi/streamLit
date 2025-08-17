import pandas as pd
import urllib

def get_data(ticker,start,end):
    ticker_request = ticker.replace("=", "%3D")

    try:
        endpoint_data = f"https://raw.githubusercontent.com/alitalbi/storage_data_fy/refs/heads/master/{ticker_request}.csv"
        price_df = pd.read_csv(endpoint_data,usecols=["Date","Close"])
        price_df.set_index("Date", inplace=True)
        price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
        price_df = price_df.loc[(price_df.index > start) & (price_df.index < end)]
        return price_df
    except urllib.error.HTTPError as e:
            print(f"HTTP Error: {e.code} {e.reason}")
            print(f"URL: {endpoint_data}")
            raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
