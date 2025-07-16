"""
Growth Analysis Page
Tracks coincident economic indicators and builds growth composite score
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
    page_title="Growth Analysis",
    page_icon="📈",
    layout="wide"
)

# Custom CSS (inherits from main app)
st.markdown("""
<style>
    .growth-header {
        background: linear-gradient(135deg, #1e3a8a, #10b981);
        color: white;
        padding: 2rem 0;
        margin: -2rem -2rem 2rem -2rem;
        text-align: center;
        border-radius: 0 0 1rem 1rem;
    }
    
    .growth-header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .indicator-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .score-display {
        background: linear-gradient(135deg, #1e3a8a, #f59e0b);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .score-value {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
    }
    
    .score-label {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_economic_data():
    """Fetch economic data for growth analysis"""
    try:
        # Create sample data (in production, replace with real API calls)
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
        np.random.seed(42)
        
        data = {
            'date': dates,
            'pce': np.random.normal(2.5, 1.2, len(dates)),  # Personal Consumption Expenditures
            'industrial_production': np.random.normal(1.8, 2.0, len(dates)),
            'nonfarm_payroll': np.random.normal(200000, 50000, len(dates)),
            'real_personal_income': np.random.normal(2.0, 1.5, len(dates)),
            'real_retail_sales': np.random.normal(3.0, 2.0, len(dates)),
            'employment_level': np.random.normal(0.5, 1.0, len(dates)),
        }
        
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_growth_score(df):
    """Calculate composite growth score"""
    if df.empty:
        return 0
    
    # Normalize indicators (0-100 scale)
    latest_row = df.iloc[-1]
    
    # Weight different indicators
    weights = {
        'pce': 0.25,
        'industrial_production': 0.20,
        'nonfarm_payroll': 0.15,
        'real_personal_income': 0.15,
        'real_retail_sales': 0.15,
        'employment_level': 0.10
    }
    
    # Calculate weighted score
    score = 0
    for indicator, weight in weights.items():
        # Normalize to 0-100 range (simplified)
        normalized_value = max(0, min(100, (latest_row[indicator] + 10) * 5))
        score += normalized_value * weight
    
    return min(100, max(0, score))

def create_growth_indicators_chart(df):
    """Create growth indicators visualization"""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Personal Consumption Expenditures', 'Industrial Production',
            'Non-Farm Payroll', 'Real Personal Income',
            'Real Retail Sales', 'Employment Level'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    indicators = [
        ('pce', 1, 1, '#1e3a8a'),
        ('industrial_production', 1, 2, '#f59e0b'),
        ('nonfarm_payroll', 2, 1, '#10b981'),
        ('real_personal_income', 2, 2, '#ef4444'),
        ('real_retail_sales', 3, 1, '#8b5cf6'),
        ('employment_level', 3, 2, '#06b6d4')
    ]
    
    for indicator, row, col, color in indicators:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df[indicator],
                name=indicator.replace('_', ' ').title(),
                line=dict(color=color, width=2),
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=800,
        title_text="Growth Indicators Dashboard",
        title_x=0.5,
        title_font=dict(size=20, family='Inter', color='#1f2937'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#e5e7eb')
    fig.update_yaxes(showgrid=True, gridcolor='#e5e7eb')
    
    return fig

def create_growth_trend_chart(df):
    """Create growth trend analysis"""
    # Calculate rolling averages
    df['pce_ma'] = df['pce'].rolling(window=6).mean()
    df['industrial_ma'] = df['industrial_production'].rolling(window=6).mean()
    
    fig = go.Figure()
    
    # Add trend lines
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['pce_ma'],
        name='PCE (6-month avg)',
        line=dict(color='#1e3a8a', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['industrial_ma'],
        name='Industrial Production (6-month avg)',
        line=dict(color='#f59e0b', width=3)
    ))
    
    # Add recession bands (sample periods)
    fig.add_vrect(
        x0="2020-03-01", x1="2020-06-01",
        fillcolor="rgba(239, 68, 68, 0.2)",
        layer="below", line_width=0,
        annotation_text="COVID-19 Recession",
        annotation_position="top left"
    )
    
    fig.update_layout(
        title="Growth Trend Analysis",
        title_x=0.5,
        title_font=dict(size=18, family='Inter', color='#1f2937'),
        xaxis_title="Date",
        yaxis_title="Growth Rate (%)",
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
    """Main growth page function"""
    
    # Header
    st.markdown("""
    <div class="growth-header">
        <h1>📈 Growth Analysis</h1>
        <p>Coincident Economic Indicators & Growth Composite Score</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get data
    df = get_economic_data()
    
    if df.empty:
        st.error("Unable to load economic data. Please try again later.")
        return
    
    # Calculate growth score
    growth_score = calculate_growth_score(df)
    
    # Display growth score
    st.markdown(f"""
    <div class="score-display">
        <div class="score-value">{growth_score:.1f}</div>
        <div class="score-label">Growth Composite Score</div>
        <p style="margin: 1rem 0 0 0; opacity: 0.8;">
            Score based on weighted average of key economic indicators
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Interpretation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if growth_score >= 70:
            status = "🟢 Strong Growth"
            color = "#10b981"
        elif growth_score >= 50:
            status = "🟡 Moderate Growth"
            color = "#f59e0b"
        else:
            status = "🔴 Weak Growth"
            color = "#ef4444"
        
        st.markdown(f"""
        <div class="indicator-card">
            <h3 style="color: {color}; margin: 0;">{status}</h3>
            <p style="color: #6b7280; margin: 0.5rem 0;">Current economic momentum</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        latest_data = df.iloc[-1]
        st.markdown(f"""
        <div class="indicator-card">
            <h3 style="color: #1e3a8a; margin: 0;">PCE: {latest_data['pce']:.1f}%</h3>
            <p style="color: #6b7280; margin: 0.5rem 0;">Personal Consumption Expenditures</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="indicator-card">
            <h3 style="color: #f59e0b; margin: 0;">Jobs: {latest_data['nonfarm_payroll']:,.0f}</h3>
            <p style="color: #6b7280; margin: 0.5rem 0;">Non-Farm Payroll (monthly)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Growth trend chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    trend_fig = create_growth_trend_chart(df)
    st.plotly_chart(trend_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed indicators
    st.markdown("""
    <div class="indicator-card">
        <h2 style="color: #1f2937; margin: 0 0 1rem 0;">📊 Detailed Growth Indicators</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # All indicators chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    indicators_fig = create_growth_indicators_chart(df)
    st.plotly_chart(indicators_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Data table
    with st.expander("📋 View Raw Data"):
        st.dataframe(
            df.round(2),
            use_container_width=True,
            column_config={
                "date": st.column_config.DateColumn("Date"),
                "pce": st.column_config.NumberColumn("PCE (%)", format="%.1f"),
                "industrial_production": st.column_config.NumberColumn("Industrial Production (%)", format="%.1f"),
                "nonfarm_payroll": st.column_config.NumberColumn("Non-Farm Payroll", format="%d"),
                "real_personal_income": st.column_config.NumberColumn("Real Personal Income (%)", format="%.1f"),
                "real_retail_sales": st.column_config.NumberColumn("Real Retail Sales (%)", format="%.1f"),
                "employment_level": st.column_config.NumberColumn("Employment Level (%)", format="%.1f"),
            }
        )
    
    # Methodology
    with st.expander("📖 Methodology"):
        st.markdown("""
        **Growth Composite Score Calculation:**
        
        The growth score is calculated using a weighted average of key economic indicators:
        
        - **Personal Consumption Expenditures (25%)**: Consumer spending drives ~70% of GDP
        - **Industrial Production (20%)**: Manufacturing and industrial output
        - **Non-Farm Payroll (15%)**: Employment growth indicator
        - **Real Personal Income (15%)**: Income growth after inflation
        - **Real Retail Sales (15%)**: Consumer spending patterns
        - **Employment Level (10%)**: Overall employment situation
        
        **Score Interpretation:**
        - 70-100: Strong Growth Environment
        - 50-69: Moderate Growth Environment  
        - 0-49: Weak Growth Environment
        
        Data is updated monthly and normalized to provide consistent scoring across different economic cycles.
        """)

if __name__ == "__main__":
    main()