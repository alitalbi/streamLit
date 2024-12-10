import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime


# Set page configuration
st.set_page_config(page_title="Eco-Framework Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
        }
        .center-title {
            text-align: center;
            font-size: 2.5em;
            font-family: 'Arial', sans-serif;
            color: #333333;
            margin-bottom: 20px;
        }
        .card-container {
            display: flex;
            justify-content: center;
            gap: 2em;
            margin-top: 50px;
        }
        .card {
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 30px;
            width: 250px;
            text-align: center;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        }
        .card-title {
            font-size: 1.5em;
            color: #444444;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .card-description {
            font-size: 1em;
            color: #666666;
        }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown("<div class='center-title'>Welcome to the Eco-Framework Dashboard</div>", unsafe_allow_html=True)

# Card container
st.markdown("""
    <div class="card-container">
        <div class="card" onclick="window.location.href='/Growth'">
            <div class="card-title">Growth</div>
            <div class="card-description">Understand the dynamics of economic growth.</div>
        </div>
        <div class="card" onclick="window.location.href='/Inflation'">
            <div class="card-title">Inflation</div>
            <div class="card-description">Analyze price trends and inflation rates.</div>
        </div>
        <div class="card" onclick="window.location.href='/RiskOnOff'">
            <div class="card-title">Risk On/Off</div>
            <div class="card-description">Monitor market risk sentiment.</div>
        </div>
        <div class="card" onclick="window.location.href='/BusinessCycle'">
            <div class="card-title">Business Cycle</div>
            <div class="card-description">Explore phases of economic cycles.</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# st.set_page_config(layout="wide")

# def import_data(url):
#     df = pd.read_csv(url)
#     df.set_index('As Of Date', inplace=True)
#     df.drop(['Time Series'], axis=1, inplace=True)
#     return df

# #urls
# url_tbills_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDPOSGS-B.csv"
# url_tbills_2022 ="https://markets.newyorkfed.org/api/pd/get/SBN2022/timeseries/PDPOSGS-B.csv"

# url_coupons_2y_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDPOSGSC-L2.csv"
# url_coupons_2y_2022 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDPOSGSC-L2.csv"

# url_commercial_paper_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDPOSCSCP.csv"
# url_commercial_paper_2022 = "https://markets.newyorkfed.org/api/pd/get/SBN2022/timeseries/PDPOSCSCP.csv"

# url_ig_bonds_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDPOSCSBND-G13.csv"
# url_ig_bonds_2022 = "https://markets.newyorkfed.org/api/pd/get/SBN2022/timeseries/PDPOSCSBND-G13.csv"

# url_below_ig_bonds_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDPOSCSBND-BELG13.csv"
# url_below_ig_bonds_2022 = "https://markets.newyorkfed.org/api/pd/get/SBN2022/timeseries/PDPOSCSBND-BELG13.csv"

# url_transactions_tbill_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDTRGS-EXTB.csv"
# url_transactions_tbill_2022 = "https://markets.newyorkfed.org/api/pd/get/SBN2022/timeseries/PDTRGS-EXTB.csv"

# url_transactions_coupons_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDTRGSC-L2.csv"
# url_transactions_coupons_2022 = "https://markets.newyorkfed.org/api/pd/get/SBN2022/timeseries/PDTRGSC-L2.csv"

# url_transactions_IDB_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2015/timeseries/PDGSIDBEXT.csv"
# url_transactions_IDB_2022 = "https://markets.newyorkfed.org/api/pd/get/SBN2022/timeseries/PDGSIDBEXT.csv"

# url_transactions_others_2015 = "https://markets.newyorkfed.org/api/pd/get/SBN2022/timeseries/PDGSWOEXT.csv"
# url_transactions_others_2022 = "https://markets.newyorkfed.org/api/pd/get/SBN2022/timeseries/PDGSWOEXT.csv"

# #import data
# tbills_2015 = import_data(url_tbills_2015)
# tbills_2022 = import_data(url_tbills_2022)

# coupons_2y_2015 = import_data(url_coupons_2y_2015)
# coupons_2y_2022 = import_data(url_coupons_2y_2022)

# commercial_paper_2015 = import_data(url_commercial_paper_2015)
# commercial_paper_2022 = import_data(url_commercial_paper_2022)

# ig_bonds_2015 = import_data(url_ig_bonds_2015)
# ig_bonds_2022 = import_data(url_ig_bonds_2022)

# below_ig_bonds_2015 = import_data(url_below_ig_bonds_2015)
# below_ig_bonds_2022 = import_data(url_below_ig_bonds_2022)

# transactions_tbills_2015 = import_data(url_transactions_tbill_2015)
# transactions_tbills_2022 = import_data(url_transactions_tbill_2022)

# transactions_coupons_2015 = import_data(url_transactions_coupons_2015)
# transactions_coupons_2022 = import_data(url_transactions_coupons_2022)

# transactions_IDB_2015 = import_data(url_transactions_IDB_2015)
# transactions_IDB_2022 = import_data(url_transactions_IDB_2022)

# transactions_others_2015 = import_data(url_transactions_others_2015)
# transactions_others_2022 = import_data(url_transactions_others_2022)

# #dfs

# #US treasury
# tbills_df = pd.concat([tbills_2015,tbills_2022])
# coupons_2y_df = pd.concat([coupons_2y_2015,coupons_2y_2022])
# #Corporate
# commercial_paper_df = pd.concat([commercial_paper_2015,commercial_paper_2022])
# ig_bonds_df = pd.concat([ig_bonds_2015,ig_bonds_2022])
# below_ig_bonds_df = pd.concat([below_ig_bonds_2015,below_ig_bonds_2022])
# transactions_tbills_df = pd.concat([transactions_tbills_2015,transactions_tbills_2022])
# transactions_coupons_df = pd.concat([transactions_coupons_2015,transactions_coupons_2022])
# transactions_IDB_df = pd.concat([transactions_IDB_2015,transactions_IDB_2022])
# transactions_others_df = pd.concat([transactions_others_2015,transactions_others_2022])
# print("2222",transactions_IDB_df,transactions_others_df)

# st.title(" Primary Dealer Statistics from the FED of New York ")
# st.title("Net Positions")
# # Create two columns
# col1, col2 = st.columns(2)


# option_treasury = col1.selectbox('US Treasury (Excluding TIPS)', ['T-Bills', 'Coupons'])
# if option_treasury == 'T-Bills':
#     with col1:
#         fig_bills = go.Figure()
#         fig_bills.add_trace(go.Scatter(x=tbills_df.index.to_list(), y=tbills_df.iloc[:,0],
#                                  mode="lines", line=dict(width=2)))
#         fig_bills.update_layout(
#             template="plotly_dark",
#             title={
#                 'text': "T-Bills (in millions)",
#                 'y': 0.9,
#                 'x': 0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'top'})
#         st.plotly_chart(fig_bills, use_container_width=True)

# elif option_treasury == 'Coupons':
#     with col1:
#         fig_coupons = go.Figure()
#         fig_coupons.add_trace(go.Scatter(x=coupons_2y_df.index.to_list(), y=coupons_2y_df.iloc[:,0],
#                                  mode="lines", line=dict(width=2)))
#         fig_coupons.update_layout(
#             template="plotly_dark",
#             title={
#                 'text': "Coupons <2 year (in millions) ",
#                 'y': 0.9,
#                 'x': 0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'top'})
#         st.plotly_chart(fig_coupons, use_container_width=True)

# option_corpo = col2.selectbox('Corporate', ['Commercial Paper', 'IG bonds, notes & debentures','Below IG bonds, notes & debentures'])
# if option_corpo == 'Commercial Paper':
#     with col2:
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=commercial_paper_df.index.to_list(), y=commercial_paper_df.iloc[:,0],
#                                  mode="lines", line=dict(width=2)))
#         fig.update_layout(
#             template="plotly_dark",
#             title={
#                 'text': "Commercial Paper (in millions)",
#                 'y': 0.9,
#                 'x': 0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'top'})
#         st.plotly_chart(fig, use_container_width=True)

# elif option_corpo == 'IG bonds, notes & debentures':
#     with col2:
#         fig_coupons = go.Figure()
#         fig_coupons.add_trace(go.Scatter(x=ig_bonds_df.index.to_list(), y=ig_bonds_df.iloc[:,0],
#                                  mode="lines", line=dict(width=2)))
#         fig_coupons.update_layout(
#             template="plotly_dark",
#             title={
#                 'text': "IG bonds, notes & debentures <= 13 months (in millions)",
#                 'y': 0.9,
#                 'x': 0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'top'})
#         st.plotly_chart(fig_coupons, use_container_width=True)

# elif option_corpo == 'Below IG bonds, notes & debentures':
#     with col2:
#         fig_coupons = go.Figure()
#         fig_coupons.add_trace(go.Scatter(x=below_ig_bonds_df.index.to_list(), y=below_ig_bonds_df.iloc[:,0],
#                                  mode="lines", line=dict(width=2)))
#         fig_coupons.update_layout(
#             template="plotly_dark",
#             title={
#                 'text': "Below IG bonds, notes & debentures <= 13 months (in millions)",
#                 'y': 0.9,
#                 'x': 0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'top'})
#         st.plotly_chart(fig_coupons, use_container_width=True)

# st.title("Transactions of US Treasury (Excluding TIPS)")

# # Create two columns
# col1_, col2_ = st.columns(2)

# option_security = col1_.selectbox('By Security', ['T-Bills', 'Coupons'])
# if option_security == "T-Bills":
#     with col1_:
#         fig_coupons = go.Figure()
#         fig_coupons.add_trace(go.Scatter(x=transactions_tbills_df.index.to_list(), y=transactions_tbills_df.iloc[:, 0],
#                                          mode="lines", line=dict(width=2)))
#         fig_coupons.update_layout(
#             template="plotly_dark",
#             title={
#                 'text': "T-Bills (in millions)",
#                 'y': 0.9,
#                 'x': 0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'top'})
#         st.plotly_chart(fig_coupons, use_container_width=True)
# elif option_security == "Coupons":
#     with col1_:
#         fig_coupons = go.Figure()
#         fig_coupons.add_trace(go.Scatter(x=transactions_coupons_df.index.to_list(), y=transactions_coupons_df.iloc[:, 0],
#                                          mode="lines", line=dict(width=2)))
#         fig_coupons.update_layout(
#             template="plotly_dark",
#             title={
#                 'text': "Coupons <= 2years (in millions)",
#                 'y': 0.9,
#                 'x': 0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'top'})
#         st.plotly_chart(fig_coupons, use_container_width=True)

# option_tcpty = col2_.selectbox('By Counterparty', ['Inter-Dealer Brokers & Others'])
# if option_tcpty == "Inter-Dealer Brokers & Others":
#     with col2_:
#         fig_coupons = go.Figure()

#         fig_coupons.add_trace(go.Scatter(x=transactions_IDB_df.index.to_list(), y=transactions_IDB_df.iloc[:, 0],
#                                          mode="lines", line=dict(width=2), name="Inter-Dealer Brokers",
#                                          showlegend=True))

#         fig_coupons.add_trace(go.Scatter(x=transactions_others_df.index.to_list(), y=transactions_others_df.iloc[:, 0],
#                                          mode="lines", line=dict(width=2), name="Others", showlegend=True,
#                                          yaxis="y2"))  # Add second y-axis

#         fig_coupons.update_layout(
#             template="plotly_dark",
#             legend=dict(
#                 title=None, orientation="h", y=0.97, yanchor="bottom", x=0.5, xanchor="center"
#             ),
#             title={
#                 'text': "Inter-Dealer Brokers & Others (in millions)",
#                 'y': 0.9,
#                 'x': 0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'top'},
#             yaxis=dict(title="Inter-Dealer Brokers"),
#             yaxis2=dict(title="Others", side="right", overlaying="y")  # Add second y-axis configuration
#         )

#         st.plotly_chart(fig_coupons, use_container_width=True)
