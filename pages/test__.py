import pandas as pd
import matplotlib.pyplot as plt

def import_data(url):
    df = pd.read_csv(url)
    df.set_index('As Of Date', inplace=True)
    df.drop(['Time Series'], axis=1, inplace=True)
    return df
url_tbills_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDPOSGS-B.csv"
url_tbills_2022 ="https://markets.newyorkfed.org/api/pd/get/SBN2022/timeseries/PDPOSGS-B.csv"

url_coupons_2y_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDPOSGSC-L2.csv"
url_coupons_2y_2022 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDPOSGSC-L2.csv"

tbills_2015 = import_data(url_tbills_2015)
tbills_2022 = import_data(url_tbills_2022)

coupons_2y_2015 = import_data(url_coupons_2y_2015)
coupons_2y_2022 = import_data(url_coupons_2y_2022)

tbills_df = pd.concat([tbills_2015,tbills_2022])

coupons_2y_df = pd.concat([coupons_2y_2015,coupons_2y_2022])
tbills_df.plot()
##US Treasury (Excluding TIPS)
plt.title("T-Bills Net positioning")
plt.show()
coupons_2y_df.plot()
plt.title("Net positioning Coupons 2y ")
plt.show()
