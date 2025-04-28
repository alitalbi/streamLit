import streamlit as st
from Fractal import compute_fractal_dimension
from data_fetcher import fetch_yahoo_data
from plot_utils import plot_assets_final_layout
import datetime

scaling_factor = 65

# --- Date Pickers at the top ---
st.subheader("Select Date Range:")

col1, col2 = st.columns(2)  # Two side-by-side columns

with col1:
    start = st.date_input(
        label="Start Date",
        value=datetime.date(2010, 1, 1),
        min_value=datetime.date(1990, 1, 1),
        max_value=datetime.date.today()
    )

with col2:
    end = st.date_input(
        label="End Date",
        value=datetime.date.today(),
        min_value=datetime.date(1990, 1, 1),
        max_value=datetime.date.today()
    )

start = start.strftime("%Y-%m-%d")
end = datetime.date.today().strftime('%Y-%m-%d')

asset_classes = {
    "Equity": {
        "S&P 500": "SPY",
        "EUROSTOXX 600": "^STOXX",
        "MSCI World": "URTH"  # ETF proxy
    },
    "Fixed Income": {
        "Bund Future": None,  # Placeholder
        "10Y US Treasury": "ZN=F"
    },
    "Credit": {
        "IEAC": "IEAC.L",
        "IHYG": "IHYG.L"
    },
    "FX": {
        "EUR/USD": "EURUSD=X",
        "USD/JPY": "JPY=X",
        "EUR/JPY": "EURJPY=X"
    },
    "Commodities": {
        "Gold": "GC=F",
        "Oil": "CL=F",
        "Copper": "HG=F"
    }
}

st.title("📈 Fractal Dimension Dashboard")

# Create tab titles
tab_names = list(asset_classes.keys())
tabs = st.tabs(tab_names)

for tab, asset_group in zip(tabs, tab_names):
    with tab:
        st.header(f"{asset_group}")

        assets = asset_classes[asset_group]
        assets_data = {}  # Collect all assets' data first!

        for name, ticker in assets.items():
            if ticker is None:
                st.warning(f"{name} - Data source not configured.")
                continue
            try:
                price_df = fetch_yahoo_data(ticker, start, end)
                fractal = compute_fractal_dimension(price_df['Close'], scaling_factor)
                price_df['Fractal'] = fractal
                df_clean = price_df[['Close', 'Fractal']].dropna()

                assets_data[name] = df_clean  # <- Save for group plotting!

            except Exception as e:
                st.error(f"Error loading {name} ({ticker}): {e}")

        # 🔥 Plot all products together once collected
        if assets_data:
            fig = plot_assets_final_layout(assets_data)
            st.plotly_chart(fig, use_container_width=True)
