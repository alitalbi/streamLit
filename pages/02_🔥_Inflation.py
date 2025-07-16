"""
Inflation Analysis Page
Tracks inflation metrics and CPI data with market pricing expectations
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
    page_title="Inflation Analysis",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .inflation-header {
        background: linear-gradient(135deg, #dc2626, #f59e0b);
        color: white;
        padding: 2rem 0;
        margin: -2rem -2rem 2rem -2rem;
        text-align: center;
        border-radius: 0 0 1rem 1rem;
    }
    
    .inflation-header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .inflation-gauge {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .cpi-breakdown {
        background: linear-gradient(135deg, #f3f4f6, #ffffff);
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_inflation_data():
    """Fetch inflation and CPI data"""
    try:
        # Create sample data (replace with real API calls)
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
        np.random.seed(123)
        
        data = {
            'date': dates,
            'cpi': np.random.normal(3.2, 1.5, len(dates)),  # Consumer Price Index
            'core_cpi': np.random.normal(2.8, 1.2, len(dates)),  # Core CPI (ex food & energy)
            'pce': np.random.normal(2.5, 1.0, len(dates)),  # PCE Price Index
            'core_pce': np.random.normal(2.2, 0.8, len(dates)),  # Core PCE
            'shelter_prices': np.random.normal(4.5, 2.0, len(dates)),  # Shelter costs
            'wages': np.random.normal(3.5, 1.5, len(dates)),  # Wage growth
            'employment': np.random.normal(2.0, 1.0, len(dates)),  # Employment cost index
            'breakeven_5y': np.random.normal(2.5, 0.5, len(dates)),  # 5-year breakeven
            'breakeven_10y': np.random.normal(2.3, 0.4, len(dates)),  # 10-year breakeven
        }
        
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error fetching inflation data: {e}")
        return pd.DataFrame()

def calculate_inflation_score(df):
    """Calculate inflation environment score"""
    if df.empty:
        return 0
    
    latest_row = df.iloc[-1]
    
    # Fed target is 2% - calculate deviation
    fed_target = 2.0
    cpi_deviation = abs(latest_row['cpi'] - fed_target)
    core_deviation = abs(latest_row['core_cpi'] - fed_target)
    
    # Score based on proximity to target (higher score = closer to target)
    cpi_score = max(0, 100 - (cpi_deviation * 20))
    core_score = max(0, 100 - (core_deviation * 20))
    
    # Weighted average
    inflation_score = (cpi_score * 0.6) + (core_score * 0.4)
    
    return min(100, max(0, inflation_score))

def create_inflation_gauge(current_inflation, target=2.0):
    """Create inflation gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = current_inflation,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Current CPI Inflation (%)"},
        delta = {'reference': target, 'position': "top"},
        gauge = {
            'axis': {'range': [None, 8]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 2], 'color': "lightgreen"},
                {'range': [2, 4], 'color': "yellow"},
                {'range': [4, 6], 'color': "orange"},
                {'range': [6, 8], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': target
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

def create_cpi_breakdown_chart(df):
    """Create CPI component breakdown"""
    # Sample CPI components
    components = ['Shelter', 'Transportation', 'Food', 'Medical Care', 'Recreation', 'Education', 'Other']
    values = [32.4, 16.8, 13.4, 8.8, 6.1, 3.2, 19.3]  # Sample weights
    colors = ['#1e3a8a', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6', '#06b6d4', '#6b7280']
    
    fig = go.Figure(data=[
        go.Pie(
            labels=components,
            values=values,
            hole=0.4,
            marker=dict(colors=colors),
            textinfo='label+percent',
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="CPI Component Breakdown",
        title_x=0.5,
        height=400,
        font=dict(size=12, family='Inter'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def create_inflation_trends_chart(df):
    """Create inflation trends comparison"""
    fig = go.Figure()
    
    # Add inflation measures
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['cpi'],
        name='CPI',
        line=dict(color='#dc2626', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['core_cpi'],
        name='Core CPI',
        line=dict(color='#f59e0b', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['pce'],
        name='PCE',
        line=dict(color='#1e3a8a', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['core_pce'],
        name='Core PCE',
        line=dict(color='#10b981', width=2)
    ))
    
    # Add Fed target line
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", 
                  annotation_text="Fed Target (2%)", annotation_position="bottom right")
    
    fig.update_layout(
        title="Inflation Measures Comparison",
        title_x=0.5,
        xaxis_title="Date",
        yaxis_title="Inflation Rate (%)",
        height=500,
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

def create_breakeven_chart(df):
    """Create inflation breakeven rates chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['breakeven_5y'],
        name='5-Year Breakeven',
        line=dict(color='#1e3a8a', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['breakeven_10y'],
        name='10-Year Breakeven',
        line=dict(color='#f59e0b', width=3)
    ))
    
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", 
                  annotation_text="Fed Target (2%)", annotation_position="bottom right")
    
    fig.update_layout(
        title="Market Inflation Expectations (Breakeven Rates)",
        title_x=0.5,
        xaxis_title="Date",
        yaxis_title="Expected Inflation Rate (%)",
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

def main():
    """Main inflation page function"""
    
    # Header
    st.markdown("""
    <div class="inflation-header">
        <h1>🔥 Inflation Analysis</h1>
        <p>Consumer Price Index, Core Inflation & Market Expectations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get data
    df = get_inflation_data()
    
    if df.empty:
        st.error("Unable to load inflation data. Please try again later.")
        return
    
    # Calculate inflation score
    inflation_score = calculate_inflation_score(df)
    latest_data = df.iloc[-1]
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Current CPI</div>
            <div class="metric-value">{latest_data['cpi']:.1f}%</div>
            <div class="metric-change">Latest reading</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Core CPI</div>
            <div class="metric-value">{latest_data['core_cpi']:.1f}%</div>
            <div class="metric-change">Ex. food & energy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Fed Target</div>
            <div class="metric-value">2.0%</div>
            <div class="metric-change">Long-term target</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if inflation_score >= 70:
            status = "🟢 On Target"
            color = "#10b981"
        elif inflation_score >= 50:
            status = "🟡 Moderate"
            color = "#f59e0b"
        else:
            status = "🔴 Off Target"
            color = "#ef4444"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Status</div>
            <div class="metric-value" style="color: {color}; font-size: 1.2rem;">{status}</div>
            <div class="metric-change">Score: {inflation_score:.0f}/100</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Inflation gauge
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="inflation-gauge">', unsafe_allow_html=True)
        gauge_fig = create_inflation_gauge(latest_data['cpi'])
        st.plotly_chart(gauge_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        cpi_fig = create_cpi_breakdown_chart(df)
        st.plotly_chart(cpi_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Inflation trends
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    trends_fig = create_inflation_trends_chart(df)
    st.plotly_chart(trends_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Market expectations
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    breakeven_fig = create_breakeven_chart(df)
    st.plotly_chart(breakeven_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed breakdown
    st.markdown("""
    <div class="cpi-breakdown">
        <h3 style="color: #1f2937; margin: 0 0 1rem 0;">📊 Inflation Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="indicator-card">
            <h4 style="color: #1e3a8a; margin: 0;">Current Situation</h4>
            <ul style="color: #6b7280; padding-left: 1rem;">
                <li>CPI: {latest_data['cpi']:.1f}% (vs 2% target)</li>
                <li>Core CPI: {latest_data['core_cpi']:.1f}%</li>
                <li>Shelter: {latest_data['shelter_prices']:.1f}% (largest component)</li>
                <li>Wage Growth: {latest_data['wages']:.1f}%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="indicator-card">
            <h4 style="color: #f59e0b; margin: 0;">Market Expectations</h4>
            <ul style="color: #6b7280; padding-left: 1rem;">
                <li>5-Year Breakeven: {latest_data['breakeven_5y']:.1f}%</li>
                <li>10-Year Breakeven: {latest_data['breakeven_10y']:.1f}%</li>
                <li>Fed Policy: Monitoring closely</li>
                <li>Trend: {"Moderating" if latest_data['cpi'] < 4 else "Elevated"}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Data table
    with st.expander("📋 View Inflation Data"):
        st.dataframe(
            df.round(2),
            use_container_width=True,
            column_config={
                "date": st.column_config.DateColumn("Date"),
                "cpi": st.column_config.NumberColumn("CPI (%)", format="%.1f"),
                "core_cpi": st.column_config.NumberColumn("Core CPI (%)", format="%.1f"),
                "pce": st.column_config.NumberColumn("PCE (%)", format="%.1f"),
                "core_pce": st.column_config.NumberColumn("Core PCE (%)", format="%.1f"),
                "shelter_prices": st.column_config.NumberColumn("Shelter (%)", format="%.1f"),
                "wages": st.column_config.NumberColumn("Wages (%)", format="%.1f"),
                "breakeven_5y": st.column_config.NumberColumn("5Y Breakeven (%)", format="%.1f"),
                "breakeven_10y": st.column_config.NumberColumn("10Y Breakeven (%)", format="%.1f"),
            }
        )
    
    # Methodology
    with st.expander("📖 Methodology"):
        st.markdown("""
        **Inflation Tracking Components:**
        
        - **CPI (Consumer Price Index)**: Broad measure of price changes for goods and services
        - **Core CPI**: CPI excluding volatile food and energy prices
        - **PCE (Personal Consumption Expenditures)**: Fed's preferred inflation measure
        - **Core PCE**: PCE excluding food and energy
        - **Breakeven Rates**: Market-implied inflation expectations from TIPS
        
        **Key Inflation Drivers:**
        - Shelter costs (largest CPI component at ~32%)
        - Transportation and energy prices
        - Food and beverage costs
        - Medical care services
        - Wage growth pressures
        
        **Federal Reserve Target:** 2% annual inflation rate (PCE-based)
        
        The inflation score measures how close current readings are to the Fed's target, 
        with higher scores indicating inflation closer to the 2% target.
        """)

if __name__ == "__main__":
    main()