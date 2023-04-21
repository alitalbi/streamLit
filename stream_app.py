import streamlit as st
import pandas as pd
import yfinance as yf
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

#sys.path.insert(1, '/Users/talbi/PycharmProjects/streamLit/venv/lib/python3.10/site-packages')
st.set_page_config(page_title="yahoo finance search",page_icon="📈")

def import_data(url):
    df = pd.read_csv(url)
    df.set_index('As Of Date', inplace=True)
    df.drop(['Time Series'], axis=1, inplace=True)
    return df

#urls
url_tbills_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDPOSGS-B.csv"
url_tbills_2022 ="https://markets.newyorkfed.org/api/pd/get/SBN2022/timeseries/PDPOSGS-B.csv"

url_coupons_2y_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDPOSGSC-L2.csv"
url_coupons_2y_2022 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDPOSGSC-L2.csv"

url_commercial_paper_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDPOSCSCP.csv"
url_commercial_paper_2022 = "https://markets.newyorkfed.org/api/pd/get/SBN2022/timeseries/PDPOSCSCP.csv"

url_ig_bonds_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDPOSCSBND-G13.csv"
url_ig_bonds_2022 = "https://markets.newyorkfed.org/api/pd/get/SBN2022/timeseries/PDPOSCSBND-G13.csv"

url_below_ig_bonds_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDPOSCSBND-BELG13.csv"
url_below_ig_bonds_2022 = "https://markets.newyorkfed.org/api/pd/get/SBN2022/timeseries/PDPOSCSBND-BELG13.csv"

url_transactions_tbill_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDTRGS-EXTB.csv"
url_transactions_tbill_2022 = "https://markets.newyorkfed.org/api/pd/get/SBN2022/timeseries/PDTRGS-EXTB.csv"

url_transactions_coupons_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDTRGSC-L2.csv"
url_transactions_coupons_2022 = "https://markets.newyorkfed.org/api/pd/get/SBN2022/timeseries/PDTRGSC-L2.csv"

url_transactions_IDB_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDGSIDBEXT.csv"
url_transactions_IDB_2022 = "https://markets.newyorkfed.org/api/pd/get/SBN2022/timeseries/PDGSIDBEXT.csv"

url_transactions_others_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2022/timeseries/PDGSWOEXT.csv"
url_transactions_others_2022 = "https://markets.newyorkfed.org/api/pd/get/SBN2022/timeseries/PDGSWOEXT.csv"

#import data
tbills_2015 = import_data(url_tbills_2015)
tbills_2022 = import_data(url_tbills_2022)

coupons_2y_2015 = import_data(url_coupons_2y_2015)
coupons_2y_2022 = import_data(url_coupons_2y_2022)

commercial_paper_2015 = import_data(url_commercial_paper_2015)
commercial_paper_2022 = import_data(url_commercial_paper_2022)

ig_bonds_2015 = import_data(url_ig_bonds_2015)
ig_bonds_2022 = import_data(url_ig_bonds_2022)

below_ig_bonds_2015 = import_data(url_below_ig_bonds_2015)
below_ig_bonds_2022 = import_data(url_below_ig_bonds_2022)

transactions_tbills_2015 = import_data(url_transactions_tbill_2015)
transactions_tbills_2022 = import_data(url_transactions_tbill_2022)

transactions_coupons_2015 = import_data(url_transactions_coupons_2015)
transactions_coupons_2022 = import_data(url_transactions_coupons_2022)

transactions_IDB_2015 = import_data(url_transactions_IDB_2015)
transactions_IDB_2022 = import_data(url_transactions_IDB_2022)

transactions_others_2015 = import_data(url_transactions_others_2015)
transactions_others_2022 = import_data(url_transactions_others_2022)

#dfs

#US treasury
tbills_df = pd.concat([tbills_2015,tbills_2022])

coupons_2y_df = pd.concat([coupons_2y_2015,coupons_2y_2022])

#Corporate
commercial_paper_df = pd.concat([commercial_paper_2015,commercial_paper_2022])

ig_bonds_df = pd.concat([ig_bonds_2015,ig_bonds_2022])

below_ig_bonds_df = pd.concat([below_ig_bonds_2015,below_ig_bonds_2022])

transactions_tbills_df = pd.concat([transactions_tbills_2015,transactions_tbills_2022])

transactions_coupons_df = pd.concat([transactions_coupons_2015,transactions_coupons_2022])

transactions_IDB_df = pd.concat([transactions_IDB_2015,transactions_IDB_2022])
transactions_others_df = pd.concat([transactions_others_2015,transactions_others_2022])
print("2222",transactions_IDB_df,transactions_others_df)
#transactions_cpty_df = pd.concat([transactions_IDB_df,transactions_others_df],axis=1)
#print(transactions_cpty_df)
#transactions_cpty_df.columns = ['Inter-Dealer Brokers','Others']
#print(transactions_cpty_df)
st.title("Net Positions")
# Create two columns
col1, col2 = st.columns(2)



option_treasury = col1.selectbox('US Treasury (Excluding TIPS)', ['T-Bills', 'Coupons'])
if option_treasury == 'T-Bills':
    with col1:
        fig_bills = go.Figure()
        fig_bills.add_trace(go.Scatter(x=tbills_df.index.to_list(), y=tbills_df.iloc[:,0],
                                 mode="lines", line=dict(width=2)))
        fig_bills.update_layout(
            template="plotly_dark",
            title={
                'text': "T-Bills (in millions)",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        st.plotly_chart(fig_bills, use_container_width=True)

elif option_treasury == 'Coupons':
    with col1:
        fig_coupons = go.Figure()
        fig_coupons.add_trace(go.Scatter(x=coupons_2y_df.index.to_list(), y=coupons_2y_df.iloc[:,0],
                                 mode="lines", line=dict(width=2)))
        fig_coupons.update_layout(
            template="plotly_dark",
            title={
                'text': "Coupons <2 year (in millions) ",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        st.plotly_chart(fig_coupons, use_container_width=True)


option_corpo = col2.selectbox('Corporate', ['Commercial Paper', 'IG bonds, notes & debentures','Below IG bonds, notes & debentures'])
if option_corpo == 'Commercial Paper':
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=commercial_paper_df.index.to_list(), y=commercial_paper_df.iloc[:,0],
                                 mode="lines", line=dict(width=2)))
        fig.update_layout(
            template="plotly_dark",
            title={
                'text': "Commercial Paper (in millions)",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        st.plotly_chart(fig, use_container_width=True)

elif option_corpo == 'IG bonds, notes & debentures':
    with col2:
        fig_coupons = go.Figure()
        fig_coupons.add_trace(go.Scatter(x=ig_bonds_df.index.to_list(), y=ig_bonds_df.iloc[:,0],
                                 mode="lines", line=dict(width=2)))
        fig_coupons.update_layout(
            template="plotly_dark",
            title={
                'text': "IG bonds, notes & debentures <= 13 months (in millions)",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        st.plotly_chart(fig_coupons, use_container_width=True)

elif option_corpo == 'Below IG bonds, notes & debentures':
    with col2:
        fig_coupons = go.Figure()
        fig_coupons.add_trace(go.Scatter(x=below_ig_bonds_df.index.to_list(), y=below_ig_bonds_df.iloc[:,0],
                                 mode="lines", line=dict(width=2)))
        fig_coupons.update_layout(
            template="plotly_dark",
            title={
                'text': "Below IG bonds, notes & debentures <= 13 months (in millions)",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        st.plotly_chart(fig_coupons, use_container_width=True)

st.title("Transactions of US Treasury (Excluding TIPS)")

# Create two columns
col1_, col2_ = st.columns(2)

option_security = col1_.selectbox('By Security', ['T-Bills', 'Coupons'])
if option_security == "T-Bills":
    with col1_:
        fig_coupons = go.Figure()
        fig_coupons.add_trace(go.Scatter(x=transactions_tbills_df.index.to_list(), y=transactions_tbills_df.iloc[:, 0],
                                         mode="lines", line=dict(width=2)))
        fig_coupons.update_layout(
            template="plotly_dark",
            title={
                'text': "T-Bills (in millions)",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        st.plotly_chart(fig_coupons, use_container_width=True)
elif option_security == "Coupons":
    with col1_:
        fig_coupons = go.Figure()
        fig_coupons.add_trace(go.Scatter(x=transactions_coupons_df.index.to_list(), y=transactions_coupons_df.iloc[:, 0],
                                         mode="lines", line=dict(width=2)))
        fig_coupons.update_layout(
            template="plotly_dark",
            title={
                'text': "Coupons <= 2years (in millions)",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        st.plotly_chart(fig_coupons, use_container_width=True)

option_tcpty = col2_.selectbox('By Counterparty', ['Inter-Dealer Brokers & Others'])
if option_tcpty == "Inter-Dealer Brokers & Others":
    with col2_:
        fig_coupons = go.Figure()

        fig_coupons.add_trace(go.Scatter(x=transactions_IDB_df.index.to_list(), y=transactions_IDB_df.iloc[:, 0],
                                         mode="lines", line=dict(width=2), name="Inter-Dealer Brokers",
                                         showlegend=True))

        fig_coupons.add_trace(go.Scatter(x=transactions_others_df.index.to_list(), y=transactions_others_df.iloc[:, 0],
                                         mode="lines", line=dict(width=2), name="Others", showlegend=True,
                                         yaxis="y2"))  # Add second y-axis

        fig_coupons.update_layout(
            template="plotly_dark",
            legend=dict(
                title=None, orientation="h", y=0.97, yanchor="bottom", x=0.5, xanchor="center"
            ),
            title={
                'text': "Inter-Dealer Brokers & Others (in millions)",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            yaxis=dict(title="Inter-Dealer Brokers"),
            yaxis2=dict(title="Others", side="right", overlaying="y")  # Add second y-axis configuration
        )

        st.plotly_chart(fig_coupons, use_container_width=True)


# Set title and description of the app
st.title("Yahoo Finance search")
st.write("Talbi & Co Eco Framework (not ESG complaint) ")
st.sidebar.header("Yahoo Finance Price search")
# Set up the search bar and date inputs
search_term = st.text_input("Enter a ticker (e.g. AAPL):")
start_date = st.date_input("Start date:", pd.Timestamp("2021-01-01"))
end_date = st.date_input("End date:", pd.Timestamp(datetime.now().strftime("%Y-%m-%d")))
print(start_date)

# Download the data and plot the close price
if search_term:
    data = yf.download(search_term, start=start_date, end=end_date)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index.to_list(), y=data.Close / 100,
                          name=search_term+ " Close Price",
                          mode="lines", line=dict(width=2)))
    st.plotly_chart(fig, use_container_width=True)
