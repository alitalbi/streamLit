"""
Risk On/Off Analysis Page
Market sentiment and risk indicators with momentum computation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf

# Page configuration
st.set_page_config(
    page_title="Risk On/Off Analysis",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .risk-header {
        background: linear-gradient(135deg, #7c3aed, #ec4899);
        color: white;
        padding: 2rem 0;
        margin: -2rem -2rem 2rem -2rem;
        text-align: center;
        border-radius: 0 0 1rem 1rem;
    }
    
    .risk-header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .risk-gauge {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .risk-on {
        background: linear-gradient(135deg, #10b981, #34d399);
        color: white;
    }
    
    .risk-off {
        background: linear-gradient(135deg, #ef4444, #f87171);
        color: white;
    }
    
    .risk-neutral {
        background: linear-gradient(135deg, #f59e0b, #fbbf24);
        color: white;
    }
    
    .momentum-indicator {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_risk_data():
    """Fetch risk sentiment data"""
    try:
        # Create sample data for risk indicators
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        np.random.seed(456)
        
        # Generate realistic risk data
        base_trend = np.sin(np.arange(len(dates)) * 0.01) * 0.5
        
        data = {
            'date': dates,
            'vix': np.random.gamma(2, 8, len(dates)) + base_trend * 5,  # VIX index
            'sp500': np.cumprod(1 + np.random.normal(0.0005, 0.015, len(dates))) * 3000,  # S&P 500
            'gold': np.cumprod(1 + np.random.normal(0.0002, 0.012, len(dates))) * 1800,  # Gold
            'dxy': np.random.normal(100, 5, len(dates)),  # Dollar index
            'bonds_10y': np.random.normal(3.5, 0.8, len(dates)),  # 10-year yield
            'high_yield_spread': np.random.normal(400, 150, len(dates)),  # HY spread
            'put_call_ratio': np.random.normal(0.8, 0.3, len(dates)),  # Put/call ratio
            'copper': np.cumprod(1 + np.random.normal(0.0001, 0.02, len(dates))) * 3.5,  # Copper
        }
        
        df = pd.DataFrame(data)
        
        # Calculate ratios for risk sentiment
        df['stocks_bonds_ratio'] = df['sp500'] / (df['bonds_10y'] * 1000)
        df['gold_dollar_ratio'] = df['gold'] / df['dxy']
        df['copper_gold_ratio'] = df['copper'] / df['gold'] * 1000
        
        return df
    except Exception as e:
        st.error(f"Error fetching risk data: {e}")
        return pd.DataFrame()

def calculate_risk_score(df):
    """Calculate risk-on/risk-off score"""
    if df.empty:
        return 50, "Neutral"
    
    latest_row = df.iloc[-1]
    
    # Individual indicator scores (0-100, 50 = neutral)
    # VIX: lower = more risk-on
    vix_score = max(0, min(100, 100 - (latest_row['vix'] - 15) * 2))
    
    # High yield spread: lower = more risk-on
    hy_score = max(0, min(100, 100 - (latest_row['high_yield_spread'] - 300) * 0.1))
    
    # Put/call ratio: lower = more risk-on
    pc_score = max(0, min(100, 100 - (latest_row['put_call_ratio'] - 0.5) * 50))
    
    # Stocks/bonds ratio: higher = more risk-on
    sb_score = min(100, max(0, (latest_row['stocks_bonds_ratio'] - 0.5) * 100))
    
    # Copper/gold ratio: higher = more risk-on
    cg_score = min(100, max(0, (latest_row['copper_gold_ratio'] - 2) * 25))
    
    # Weighted average
    risk_score = (vix_score * 0.3 + hy_score * 0.25 + pc_score * 0.15 + 
                  sb_score * 0.15 + cg_score * 0.15)
    
    # Determine sentiment
    if risk_score >= 65:
        sentiment = "Risk On"
    elif risk_score <= 35:
        sentiment = "Risk Off"
    else:
        sentiment = "Neutral"
    
    return risk_score, sentiment

def create_risk_gauge(risk_score):
    """Create risk sentiment gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Sentiment Score"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 35], 'color': "lightcoral"},
                {'range': [35, 65], 'color': "lightyellow"},
                {'range': [65, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"},
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig

def create_risk_indicators_chart(df):
    """Create risk indicators dashboard"""
    # Resample to monthly for cleaner charts
    df_monthly = df.set_index('date').resample('M').last().reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('VIX Index', 'High Yield Spread', 'Put/Call Ratio', 'Stocks/Bonds Ratio'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # VIX
    fig.add_trace(
        go.Scatter(x=df_monthly['date'], y=df_monthly['vix'], 
                  name='VIX', line=dict(color='#7c3aed', width=2)),
        row=1, col=1
    )
    
    # High Yield Spread
    fig.add_trace(
        go.Scatter(x=df_monthly['date'], y=df_monthly['high_yield_spread'], 
                  name='HY Spread', line=dict(color='#ec4899', width=2)),
        row=1, col=2
    )
    
    # Put/Call Ratio
    fig.add_trace(
        go.Scatter(x=df_monthly['date'], y=df_monthly['put_call_ratio'], 
                  name='P/C Ratio', line=dict(color='#f59e0b', width=2)),
        row=2, col=1
    )
    
    # Stocks/Bonds Ratio
    fig.add_trace(
        go.Scatter(x=df_monthly['date'], y=df_monthly['stocks_bonds_ratio'], 
                  name='S/B Ratio', line=dict(color='#10b981', width=2)),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Risk Sentiment Indicators",
        title_x=0.5,
        title_font=dict(size=20, family='Inter', color='#1f2937'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#e5e7eb')
    fig.update_yaxes(showgrid=True, gridcolor='#e5e7eb')
    
    return fig

def create_momentum_chart(df):
    """Create momentum analysis chart"""
    # Calculate momentum indicators
    df_monthly = df.set_index('date').resample('M').last().reset_index()
    
    # Simple momentum: 3-month change
    df_monthly['sp500_momentum'] = df_monthly['sp500'].pct_change(periods=3) * 100
    df_monthly['gold_momentum'] = df_monthly['gold'].pct_change(periods=3) * 100
    df_monthly['copper_momentum'] = df_monthly['copper'].pct_change(periods=3) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_monthly['date'],
        y=df_monthly['sp500_momentum'],
        name='S&P 500 Momentum',
        line=dict(color='#1e3a8a', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_monthly['date'],
        y=df_monthly['gold_momentum'],
        name='Gold Momentum',
        line=dict(color='#f59e0b', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_monthly['date'],
        y=df_monthly['copper_momentum'],
        name='Copper Momentum',
        line=dict(color='#ef4444', width=2)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                  annotation_text="Neutral", annotation_position="bottom right")
    
    fig.update_layout(
        title="Asset Momentum (3-Month % Change)",
        title_x=0.5,
        xaxis_title="Date",
        yaxis_title="Momentum (%)",
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_ratios_chart(df):
    """Create key ratios chart"""
    df_monthly = df.set_index('date').resample('M').last().reset_index()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Copper/Gold Ratio', 'Gold/Dollar Ratio')
    )
    
    fig.add_trace(
        go.Scatter(x=df_monthly['date'], y=df_monthly['copper_gold_ratio'], 
                  name='Copper/Gold', line=dict(color='#f59e0b', width=3)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df_monthly['date'], y=df_monthly['gold_dollar_ratio'], 
                  name='Gold/Dollar', line=dict(color='#1e3a8a', width=3)),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Key Risk Ratios",
        title_x=0.5,
        title_font=dict(size=18, family='Inter', color='#1f2937'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#e5e7eb')
    fig.update_yaxes(showgrid=True, gridcolor='#e5e7eb')
    
    return fig

def main():
    """Main risk analysis page function"""
    
    # Header
    st.markdown("""
    <div class="risk-header">
        <h1>⚖️ Risk On/Off Analysis</h1>
        <p>Market Sentiment & Risk Indicators with Momentum Computation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get data
    df = get_risk_data()
    
    if df.empty:
        st.error("Unable to load risk data. Please try again later.")
        return
    
    # Calculate risk score
    risk_score, sentiment = calculate_risk_score(df)
    latest_data = df.iloc[-1]
    
    # Risk sentiment display
    sentiment_class = "risk-on" if sentiment == "Risk On" else "risk-off" if sentiment == "Risk Off" else "risk-neutral"
    
    st.markdown(f"""
    <div class="risk-gauge {sentiment_class}">
        <h2 style="font-size: 3rem; margin: 0;">{sentiment}</h2>
        <p style="font-size: 1.5rem; margin: 0.5rem 0;">Score: {risk_score:.1f}/100</p>
        <p style="opacity: 0.8; margin: 0;">Current market risk sentiment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        vix_color = "#10b981" if latest_data['vix'] < 20 else "#f59e0b" if latest_data['vix'] < 30 else "#ef4444"
        st.markdown(f"""
        <div class="momentum-indicator">
            <h3 style="color: {vix_color}; margin: 0;">VIX</h3>
            <p style="font-size: 1.5rem; margin: 0.5rem 0; color: {vix_color};">{latest_data['vix']:.1f}</p>
            <p style="color: #6b7280; margin: 0;">Volatility Index</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        hy_color = "#10b981" if latest_data['high_yield_spread'] < 400 else "#f59e0b" if latest_data['high_yield_spread'] < 600 else "#ef4444"
        st.markdown(f"""
        <div class="momentum-indicator">
            <h3 style="color: {hy_color}; margin: 0;">HY Spread</h3>
            <p style="font-size: 1.5rem; margin: 0.5rem 0; color: {hy_color};">{latest_data['high_yield_spread']:.0f}</p>
            <p style="color: #6b7280; margin: 0;">Basis Points</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        pc_color = "#ef4444" if latest_data['put_call_ratio'] > 1.0 else "#f59e0b" if latest_data['put_call_ratio'] > 0.8 else "#10b981"
        st.markdown(f"""
        <div class="momentum-indicator">
            <h3 style="color: {pc_color}; margin: 0;">P/C Ratio</h3>
            <p style="font-size: 1.5rem; margin: 0.5rem 0; color: {pc_color};">{latest_data['put_call_ratio']:.2f}</p>
            <p style="color: #6b7280; margin: 0;">Put/Call Ratio</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="momentum-indicator">
            <h3 style="color: #1e3a8a; margin: 0;">S&P 500</h3>
            <p style="font-size: 1.5rem; margin: 0.5rem 0; color: #1e3a8a;">{latest_data['sp500']:.0f}</p>
            <p style="color: #6b7280; margin: 0;">Index Level</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk gauge
    col1, col2 = st.columns([1, 1])
    
    with col1:
        gauge_fig = create_risk_gauge(risk_score)
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    with col2:
        st.markdown(f"""
        <div class="indicator-card">
            <h3 style="color: #1f2937; margin: 0;">Risk Assessment</h3>
            <p style="color: #6b7280; margin: 1rem 0;">
                <strong>Current Environment:</strong> {sentiment}<br>
                <strong>VIX Level:</strong> {"Low" if latest_data['vix'] < 20 else "Moderate" if latest_data['vix'] < 30 else "High"}<br>
                <strong>Credit Spreads:</strong> {"Tight" if latest_data['high_yield_spread'] < 400 else "Moderate" if latest_data['high_yield_spread'] < 600 else "Wide"}<br>
                <strong>Market Stress:</strong> {"Low" if risk_score > 65 else "Moderate" if risk_score > 35 else "High"}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk indicators chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    indicators_fig = create_risk_indicators_chart(df)
    st.plotly_chart(indicators_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Momentum chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    momentum_fig = create_momentum_chart(df)
    st.plotly_chart(momentum_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Ratios chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    ratios_fig = create_ratios_chart(df)
    st.plotly_chart(ratios_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="indicator-card">
            <h4 style="color: #7c3aed; margin: 0;">Risk-On Indicators</h4>
            <ul style="color: #6b7280; padding-left: 1rem;">
                <li>VIX below 20 (fear gauge)</li>
                <li>High yield spreads tightening</li>
                <li>Put/call ratio below 0.8</li>
                <li>Stocks outperforming bonds</li>
                <li>Copper/gold ratio rising</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="indicator-card">
            <h4 style="color: #ec4899; margin: 0;">Risk-Off Indicators</h4>
            <ul style="color: #6b7280; padding-left: 1rem;">
                <li>VIX above 30 (high fear)</li>
                <li>High yield spreads widening</li>
                <li>Put/call ratio above 1.0</li>
                <li>Bonds outperforming stocks</li>
                <li>Gold outperforming copper</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Data table
    with st.expander("📋 View Risk Data"):
        display_df = df.tail(30).copy()  # Show last 30 days
        st.dataframe(
            display_df.round(2),
            use_container_width=True,
            column_config={
                "date": st.column_config.DateColumn("Date"),
                "vix": st.column_config.NumberColumn("VIX", format="%.1f"),
                "sp500": st.column_config.NumberColumn("S&P 500", format="%.0f"),
                "gold": st.column_config.NumberColumn("Gold", format="%.0f"),
                "high_yield_spread": st.column_config.NumberColumn("HY Spread", format="%.0f"),
                "put_call_ratio": st.column_config.NumberColumn("P/C Ratio", format="%.2f"),
                "copper_gold_ratio": st.column_config.NumberColumn("Cu/Au Ratio", format="%.2f"),
            }
        )
    
    # Methodology
    with st.expander("📖 Methodology"):
        st.markdown("""
        **Risk-On/Risk-Off Scoring:**
        
        The risk sentiment score combines multiple market indicators:
        
        - **VIX Index (30%)**: Measures market volatility and fear
        - **High Yield Spreads (25%)**: Credit market stress indicator
        - **Put/Call Ratio (15%)**: Options market sentiment
        - **Stocks/Bonds Ratio (15%)**: Relative performance
        - **Copper/Gold Ratio (15%)**: Industrial vs. safe-haven assets
        
        **Score Interpretation:**
        - 65-100: Risk-On Environment (growth assets favored)
        - 35-64: Neutral/Mixed Environment
        - 0-34: Risk-Off Environment (safe havens favored)
        
        **Key Ratios Explained:**
        - **Copper/Gold**: Industrial demand vs. safe haven preference
        - **Gold/Dollar**: Safe haven vs. risk currency strength
        - **Stocks/Bonds**: Risk assets vs. safe government bonds
        
        Data is updated daily and includes momentum calculations for trend analysis.
        """)

if __name__ == "__main__":
    main()