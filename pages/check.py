import requests
import pandas as pd
cli = pd.read_csv("https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_CLI,/.M.LI...AA...H?startPeriod=2024-02&dimensionAtObservation=AllDimensions&format=csvfilewithlabels")
cli = cli.loc[cli["REF_AREA"]=="USA"][["TIME_PERIOD","OBS_VALUE"]]
print(cli)