"""
Modern Economic Dashboard
Main entry point for the Streamlit application
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="Economic Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Economic Dashboard\nModern economic indicators and market analysis platform"
    }
)

# Custom CSS for modern white-based theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for consistent theming */
    :root {
        --primary-color: #1e3a8a;      /* Deep Navy Blue */
        --secondary-color: #f59e0b;     /* Amber/Gold */
        --background-color: #ffffff;    /* White */
        --text-primary: #1f2937;       /* Dark Gray */
        --text-secondary: #6b7280;     /* Medium Gray */
        --border-color: #e5e7eb;       /* Light Gray */
        --success-color: #10b981;      /* Emerald */
        --warning-color: #f59e0b;      /* Amber */
        --error-color: #ef4444;        /* Red */
    }
    
    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        padding: 2rem 0;
        margin: -2rem -2rem 2rem -2rem;
        text-align: center;
        border-radius: 0 0 1rem 1rem;
    }
    
    .main-header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .metric-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0.5rem 0;
    }
    
    .metric-change {
        font-size: 0.9rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    .metric-change.positive {
        color: var(--success-color);
    }
    
    .metric-change.negative {
        color: var(--error-color);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--background-color);
        border-right: 1px solid var(--border-color);
    }
    
    .css-1d391kg .css-10trblm {
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    
    /* Navigation styling */
    .nav-section {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .nav-title {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: white;
        border: 1px solid var(--border-color);
        border-radius: 0.5rem;
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def create_sample_data():
    """Create sample economic data for demonstration"""
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
    np.random.seed(42)
    
    data = {
        'date': dates,
        'gdp_growth': np.random.normal(2.5, 1.5, len(dates)),
        'inflation': np.random.normal(3.0, 1.2, len(dates)),
        'unemployment': np.random.normal(5.5, 1.8, len(dates)),
        'interest_rate': np.random.normal(2.0, 1.0, len(dates)),
        'sp500': np.cumprod(1 + np.random.normal(0.008, 0.04, len(dates))) * 3000,
        'vix': np.random.gamma(2, 8, len(dates)),
    }
    
    return pd.DataFrame(data)

def create_metric_card(title, value, change, unit=""):
    """Create a styled metric card"""
    change_class = "positive" if change >= 0 else "negative"
    change_symbol = "↑" if change >= 0 else "↓"
    
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}{unit}</div>
        <div class="metric-change {change_class}">
            {change_symbol} {abs(change):.2f}% from last period
        </div>
    </div>
    """

def create_overview_chart(df):
    """Create an overview chart with key economic indicators"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('GDP Growth Rate', 'Inflation Rate', 'S&P 500 Index', 'VIX Index'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # GDP Growth
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['gdp_growth'], 
                  name='GDP Growth', line=dict(color='#1e3a8a', width=2)),
        row=1, col=1
    )
    
    # Inflation
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['inflation'], 
                  name='Inflation', line=dict(color='#f59e0b', width=2)),
        row=1, col=2
    )
    
    # S&P 500
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['sp500'], 
                  name='S&P 500', line=dict(color='#10b981', width=2)),
        row=2, col=1
    )
    
    # VIX
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['vix'], 
                  name='VIX', line=dict(color='#ef4444', width=2)),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Economic Indicators Overview",
        title_x=0.5,
        title_font=dict(size=20, family='Inter', color='#1f2937'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    # Update axes
    fig.update_xaxes(showgrid=True, gridcolor='#e5e7eb')
    fig.update_yaxes(showgrid=True, gridcolor='#e5e7eb')
    
    return fig

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Economic Dashboard</h1>
        <p>Advanced Economic Indicators & Market Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown("""
    <div class="nav-section">
        <div class="nav-title">📊 Dashboard Pages</div>
        <p style="color: #6b7280; font-size: 0.9rem;">Navigate to different analysis sections</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample data
    df = create_sample_data()
    latest_data = df.iloc[-1]
    previous_data = df.iloc[-2]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        gdp_change = ((latest_data['gdp_growth'] - previous_data['gdp_growth']) / previous_data['gdp_growth']) * 100
        st.markdown(create_metric_card("GDP Growth", f"{latest_data['gdp_growth']:.1f}", gdp_change, "%"), unsafe_allow_html=True)
    
    with col2:
        inflation_change = ((latest_data['inflation'] - previous_data['inflation']) / previous_data['inflation']) * 100
        st.markdown(create_metric_card("Inflation Rate", f"{latest_data['inflation']:.1f}", inflation_change, "%"), unsafe_allow_html=True)
    
    with col3:
        sp500_change = ((latest_data['sp500'] - previous_data['sp500']) / previous_data['sp500']) * 100
        st.markdown(create_metric_card("S&P 500", f"{latest_data['sp500']:.0f}", sp500_change), unsafe_allow_html=True)
    
    with col4:
        vix_change = ((latest_data['vix'] - previous_data['vix']) / previous_data['vix']) * 100
        st.markdown(create_metric_card("VIX Index", f"{latest_data['vix']:.1f}", vix_change), unsafe_allow_html=True)
    
    # Overview chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig = create_overview_chart(df)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick insights
    st.markdown("""
    <div class="nav-section">
        <div class="nav-title">📈 Quick Insights</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">🎯 Market Outlook</div>
            <p style="color: #6b7280; margin: 0.5rem 0;">
                Current economic indicators suggest a balanced market environment with 
                moderate growth and controlled inflation levels.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">⚡ Key Highlights</div>
            <p style="color: #6b7280; margin: 0.5rem 0;">
                • GDP growth remains steady<br>
                • Inflation within target range<br>
                • Market volatility is manageable<br>
                • Economic cycles showing stability
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: #6b7280; border-top: 1px solid #e5e7eb; margin-top: 2rem;">
        <p>Modern Economic Dashboard | Updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()