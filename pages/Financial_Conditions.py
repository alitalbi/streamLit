


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
from pandas.tseries.offsets import BDay
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
def agg_zscore(df):
    
    df["mean"] = df.mean(axis=1)
    df["std"] = df.std(axis=1)
    df.iloc[:,:3] =(np.array(df.iloc[:,:3])-np.array(df.iloc[:,3]).reshape(-1,1))/np.array(df.iloc[:,4]).reshape(-1,1)
    #df.loc[:,df.columns[:len(df.columns)-2]] = df.loc[:,df.columns[:len(df.columns)-2]])
    df.drop(["mean","std"],axis=1,inplace=True)
    agg_z_score = df.copy()
    return agg_z_score
    
"""- US Dollar Index (DX-Y.NYB)
- CBOE Interest Rate 10 Year T No (^TNX)
- Gasoline Active Fut (RB=F)"""


import numpy as np
import pandas as pd


def color_scale(val):
    # Compute the global min and max for scaling
    min_val = proxy_return.min().min()
    max_val = proxy_return.max().max()

    # Define RGB color codes for different shades
    deep_red = np.array([295, 200, 200])      # Deep red for strong negative values
    light_red = np.array([255, 0, 0]) # Light red for weak negative values
    deep_green = np.array([0, 255, 0])    # Strong green for strong positive values
    light_green = np.array([100, 205, 100]) # Light green for weak positive values

    # Normalize intensity based on full range of values
    if val < 0:
        intensity = (val - min_val) / (0 - min_val)  # Scale negative values
        color = light_red + intensity * (deep_red - light_red)  # Blend light red to deep red

    elif val > 0:
        intensity = val / max_val  # Scale positive values
        color = light_green + intensity * (deep_green - light_green)  # Blend light green to strong green

    else:
        color = np.array([255, 255, 255])  # White for zero

    # Ensure RGB values are within valid range (0-255)
    color = np.clip(color, 0, 255)

    # Convert RGB values to hexadecimal color code
    hex_code = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
    return f'background-color: {hex_code}'
fin_conditions_tickers = {"US Dollar Index":"DX-Y.NYB",
                          "CBOE 10Y T Note":"^TNX",
                          "RBOB Gasoline Active Fut":"RB=F"}



col1,col2= st.columns(2,gap="small")
with col1:
    dropdown_menu = st.selectbox("Additional Line",options=["","DXY","10Y","Gasoline"])
with col2:
 
    st.markdown("""<h4>Period</h4>""",unsafe_allow_html=True)
    col1,col2,col3,col4,col5,col6 = st.columns(6,gap="small")
    with col1:
        period_1m = st.button("1m",disabled=False)
    with col2:
        period_3m = st.button("3m")
    with col3:
        period_6m = st.button("6m")
    with col4:
        period_12m = st.button("12m")
    with col5:
        period_18m = st.button("18m")    
    with col6:
        period_all = st.button("All")
date_start = "2002-01-01"
date_end = datetime.today().strftime("%Y-%m-%d")

if period_1m:
    date_start = datetime.strptime(date_end,"%Y-%m-%d") +  BDay(-22*2)
    buff = 22
elif period_3m:
    date_start = datetime.strptime(date_end,"%Y-%m-%d") +  BDay(-66*2)
    buff = 66
elif period_6m:
    date_start = datetime.strptime(date_end,"%Y-%m-%d") +  BDay(-132*2)
    buff = 132
elif period_12m:
    date_start = datetime.strptime(date_end,"%Y-%m-%d") +  BDay(-252*2)
    buff = 252
elif period_18m:
    date_start = datetime.strptime(date_end,"%Y-%m-%d") +  BDay(-384*2)
proxy_return = yf.download(list(fin_conditions_tickers.values()), start=date_start, end=date_end, interval="1d")["Adj Close"]
### avg saily dev ###------------------------------------
indicators = ["DXY","10Y","Gasoline"]
proxy_return.columns = ["DXY","10Y","Gasoline"]

col1,col2 = st.columns(2,gap="small")
with col1:
    ret_rolling_window = st.select_slider("Return period ",options=["1d","1w","1m","3m","6m"])
    window_ret = 1 if ret_rolling_window == "1d" else 5 if ret_rolling_window == "1w" else 22 if ret_rolling_window == "1m" else 66 if ret_rolling_window == "3m" else 132
with col2:
    z_rolling_window = st.select_slider("Score Rolling window (in m)",options=["1","2","3","6","12","18","24"])
# rolling_z = proxy_return.copy()
for col in proxy_return.columns :
    proxy_return["return_"+col] = proxy_return[col].pct_change(window_ret)

for col in indicators :
    proxy_return["z"+col] = (proxy_return["return_"+col] - proxy_return["return_"+col].rolling(int(z_rolling_window)*22).mean())/proxy_return["return_"+col].rolling(int(z_rolling_window   )*22).std()

proxy_return["agg_z"] = proxy_return[proxy_return.columns[-3:]].mean(axis=1)
proxy_return.dropna(inplace=True)
agg_table_score = proxy_return[["zDXY","z10Y","zGasoline","agg_z"]][::-1].head(30).style.applymap(color_scale)

if dropdown_menu == "":
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=proxy_return.index.tolist(),y=proxy_return["agg_z"]))
    
else:
    fig = make_subplots(rows=1,
                        cols=1,
                        specs=[[{"secondary_y": True}]],
                        )
    fig.add_trace(go.Scatter(x=proxy_return.index.to_list(),
                               y=proxy_return["agg_z"].to_list(),
                               name="Agg Score"),row=1,col=1,secondary_y=False)
    fig.add_trace(go.Scatter(x=proxy_return.index.to_list(),
                               y=proxy_return[dropdown_menu].to_list(),
                               name=dropdown_menu),row=1,col=1,secondary_y=True)
    

st.plotly_chart(fig,use_container_width=True)
st.dataframe(agg_table_score)



# Generate synthetic data with realistic financial properties
np.random.seed(42)

df= yf.download(list(fin_conditions_tickers.values())+["^GSPC"], start=date_start, end=date_end, interval="1d")["Adj Close"].sort_values(by="Date")
# Features (X) and Target (y)
df = df.pct_change(1)
df
df.dropna(inplace=True)
df.columns = ["DXY","10Y","Gasoline","SPY"]
X = df[["Gasoline", "DXY", "10Y"]]
y = df["SPY"]

# Train-Test Split using TimeSeriesSplit
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

out_of_sample_results = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Regularized Regression using RidgeCV and LassoCV
    alphas = np.logspace(-6, 6, 100)
    ridge = RidgeCV(alphas=alphas, cv=tscv).fit(X_train, y_train)
    lasso = LassoCV(alphas=alphas, cv=tscv).fit(X_train, y_train)

    # Predict on the out-of-sample set
    ridge_pred = ridge.predict(X_test)
    lasso_pred = lasso.predict(X_test)

    # Evaluate models
    ridge_mse = mean_squared_error(y_test, ridge_pred)
    lasso_mse = mean_squared_error(y_test, lasso_pred)

    ridge_r2 = r2_score(y_test, ridge_pred)
    lasso_r2 = r2_score(y_test, lasso_pred)

    # Store results for plotting
    out_of_sample_results.append({
        "ridge_pred": ridge_pred,
        "lasso_pred": lasso_pred,
        "y_test": y_test,
        "ridge_mse": ridge_mse,
        "lasso_mse": lasso_mse,
        "ridge_r2": ridge_r2,
        "lasso_r2": lasso_r2,
        "timestamps": X_test.index
    })

# Statistical tests for assumptions
X_with_const = sm.add_constant(X)  # Add intercept for statsmodels
ols_model = sm.OLS(y, X_with_const).fit()

# Multicollinearity: Variance Inflation Factor (VIF)
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i + 1) for i in range(X.shape[1])]

# Streamlit App
st.title("Enhanced Regression Model Performance")

st.header("Model Coefficients")
st.write("Ridge Regression Coefficients")
st.write({feature: coef for feature, coef in zip(X.columns, ridge.coef_)})

st.write("Lasso Regression Coefficients")
st.write({feature: coef for feature, coef in zip(X.columns, lasso.coef_)})

# st.header("Performance Metrics")


# st.header("Multicollinearity (VIF)")
# st.write(vif_data)

st.header("Out-of-Sample Performance")

# Plot only the out-of-sample performance
for i, result in enumerate(out_of_sample_results):
    fig = go.Figure()
    # Adding prediction for tomorrow based on last data point
last_data_point = X.iloc[-1].values.reshape(1, -1)  # Get the last available data point

# Predict tomorrow's return using Ridge and Lasso
ridge_tomorrow_pred = ridge.predict(last_data_point)[0]
lasso_tomorrow_pred = lasso.predict(last_data_point)[0]

# Append tomorrow's prediction to the respective prediction arrays
ridge_pred_with_tomorrow = np.append(ridge_pred, ridge_tomorrow_pred)
lasso_pred_with_tomorrow = np.append(lasso_pred, lasso_tomorrow_pred)

# Create a plot for Out-of-Sample Performance with dashed lines for tomorrow's prediction
fig = go.Figure()

# Plot actual returns
fig.add_trace(go.Scatter(x=result['timestamps'], y=result['y_test'], mode='lines', name='Actual Returns'))

# Plot Ridge and Lasso predictions
fig.add_trace(go.Scatter(x=result['timestamps'], y=ridge_pred_with_tomorrow, mode='lines', name='Ridge Predictions'))
fig.add_trace(go.Scatter(x=result['timestamps'], y=lasso_pred_with_tomorrow, mode='lines', name='Lasso Predictions'))

# Plot tomorrow's prediction (use a dashed line for distinction)
tomorrow_date = X.index[-1] + timedelta(days=1)
fig.add_trace(go.Scatter(x=[tomorrow_date], y=[ridge_tomorrow_pred], mode='markers+text', 
                         name="Ridge Tomorrow", text=["Ridge Tomorrow"], 
                         marker=dict(symbol='circle', color='red', size=10), 
                         line=dict(dash='dash')))

fig.add_trace(go.Scatter(x=[tomorrow_date], y=[lasso_tomorrow_pred], mode='markers+text', 
                         name="Lasso Tomorrow", text=["Lasso Tomorrow"], 
                         marker=dict(symbol='circle', color='blue', size=10), 
                         line=dict(dash='dash')))

# Update the layout
fig.update_layout(
    title="Out-of-Sample Performance with Tomorrow's Prediction",
    xaxis_title="Time",
    yaxis_title="Returns",
    legend_title="Legend",
    template="plotly_white"
)

# Display the plot in Streamlit
st.plotly_chart(fig)
   
# Save models (for production use)
for i, result in enumerate(out_of_sample_results):
    st.write(f"Fold {i + 1}: Ridge MSE: {result['ridge_mse']:.4f}, R^2: {result['ridge_r2']:.4f}")
    st.write(f"Fold {i + 1}: Lasso MSE: {result['lasso_mse']:.4f}, R^2: {result['lasso_r2']:.4f}")
