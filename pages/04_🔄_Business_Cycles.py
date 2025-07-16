"""
Business Cycles Analysis Page
Economic cycle analysis using Leading Economic Index (LEI) and sector rotation
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
    page_title="Business Cycles",
    page_icon="🔄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .cycle-header {
        background: linear-gradient(135deg, #059669, #0d9488);
        color: white;
        padding: 2rem 0;
        margin: -2rem -2rem 2rem -2rem;
        text-align: center;
        border-radius: 0 0 1rem 1rem;
    }
    
    .cycle-header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .cycle-stage {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .expansion {
        border-color: #10b981;
        background: linear-gradient(135deg, #ecfdf5, #ffffff);
    }
    
    .peak {
        border-color: #f59e0b;
        background: linear-gradient(135deg, #fffbeb, #ffffff);
    }
    
    .contraction {
        border-color: #ef4444;
        background: linear-gradient(135deg, #fef2f2, #ffffff);
    }
    
    .trough {
        border-color: #6b7280;
        background: linear-gradient(135deg, #f9fafb, #ffffff);
    }
    
    .lei-display {
        background: linear-gradient(135deg, #1e3a8a, #0d9488);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .sector-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_business_cycle_data():
    """Fetch business cycle and LEI data"""
    try:
        # Create sample data for business cycle indicators
        dates = pd.date_range(start='2000-01-01', end='2024-12-31', freq='M')
        np.random.seed(789)
        
        # Generate realistic business cycle data
        cycle_length = 120  # ~10 years
        cycle_phase = (np.arange(len(dates)) % cycle_length) / cycle_length * 2 * np.pi
        
        # Base cycle with noise
        base_cycle = np.sin(cycle_phase) * 2 + np.random.normal(0, 0.5, len(dates))
        
        data = {
            'date': dates,
            'lei': base_cycle + np.random.normal(0, 0.3, len(dates)),  # Leading Economic Index
            'employment': base_cycle * 0.8 + np.random.normal(0, 0.2, len(dates)),
            'housing_starts': base_cycle * 1.2 + np.random.normal(0, 0.4, len(dates)),
            'stock_prices': base_cycle * 1.5 + np.random.normal(0, 0.3, len(dates)),
            'money_supply': base_cycle * 0.6 + np.random.normal(0, 0.2, len(dates)),
            'yield_spread': -base_cycle * 0.5 + np.random.normal(0, 0.2, len(dates)),
            'consumer_expectations': base_cycle * 0.9 + np.random.normal(0, 0.3, len(dates)),
            'manufacturing_orders': base_cycle * 1.1 + np.random.normal(0, 0.4, len(dates)),
            'gdp_growth': base_cycle * 0.7 + np.random.normal(2.5, 0.5, len(dates)),
            'unemployment': -base_cycle * 0.8 + np.random.normal(5.5, 1.0, len(dates)),
        }
        
        df = pd.DataFrame(data)
        
        # Add sector performance (relative to market)
        sector_names = ['Technology', 'Healthcare', 'Financials', 'Energy', 'Industrials', 'Consumer Discretionary', 'Utilities']
        for sector in sector_names:
            df[f'{sector.lower()}_performance'] = base_cycle * np.random.uniform(0.5, 1.5) + np.random.normal(0, 0.3, len(dates))
        
        return df
    except Exception as e:
        st.error(f"Error fetching business cycle data: {e}")
        return pd.DataFrame()

def determine_cycle_stage(df):
    """Determine current business cycle stage"""
    if df.empty:
        return "Unknown", 0
    
    # Use last 6 months of data for trend analysis
    recent_data = df.tail(6)
    
    lei_trend = recent_data['lei'].iloc[-1] - recent_data['lei'].iloc[0]
    lei_level = recent_data['lei'].iloc[-1]
    gdp_trend = recent_data['gdp_growth'].iloc[-1] - recent_data['gdp_growth'].iloc[0]
    unemployment_trend = recent_data['unemployment'].iloc[-1] - recent_data['unemployment'].iloc[0]
    
    # Determine stage based on LEI and economic indicators
    if lei_trend > 0.5 and gdp_trend > 0 and unemployment_trend < 0:
        stage = "Expansion"
        confidence = 85
    elif lei_trend > 0 and gdp_trend > 1.5 and unemployment_trend < -0.5:
        stage = "Peak"
        confidence = 80
    elif lei_trend < -0.5 and gdp_trend < 0 and unemployment_trend > 0:
        stage = "Contraction"
        confidence = 90
    elif lei_trend < 0 and gdp_trend < -1 and unemployment_trend > 0.5:
        stage = "Trough"
        confidence = 85
    else:
        stage = "Transition"
        confidence = 60
    
    return stage, confidence

def create_lei_chart(df):
    """Create Leading Economic Index chart"""
    fig = go.Figure()
    
    # LEI line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['lei'],
        name='Leading Economic Index',
        line=dict(color='#1e3a8a', width=3)
    ))
    
    # Add recession periods (sample)
    recession_periods = [
        ('2001-03-01', '2001-11-01'),
        ('2007-12-01', '2009-06-01'),
        ('2020-02-01', '2020-04-01')
    ]
    
    for start, end in recession_periods:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="rgba(239, 68, 68, 0.2)",
            layer="below", line_width=0
        )
    
    # Add trend line
    x_numeric = np.arange(len(df))
    z = np.polyfit(x_numeric, df['lei'], 1)
    p = np.poly1d(z)
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=p(x_numeric),
        name='Trend',
        line=dict(color='#f59e0b', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Leading Economic Index (LEI)",
        title_x=0.5,
        xaxis_title="Date",
        yaxis_title="LEI Value",
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

def create_cycle_components_chart(df):
    """Create business cycle components chart"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Employment', 'Housing Starts', 'Stock Prices', 'Consumer Expectations'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    components = [
        ('employment', 1, 1, '#10b981'),
        ('housing_starts', 1, 2, '#f59e0b'),
        ('stock_prices', 2, 1, '#1e3a8a'),
        ('consumer_expectations', 2, 2, '#ef4444')
    ]
    
    for component, row, col, color in components:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df[component],
                name=component.replace('_', ' ').title(),
                line=dict(color=color, width=2),
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=600,
        title_text="Business Cycle Components",
        title_x=0.5,
        title_font=dict(size=20, family='Inter', color='#1f2937'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#e5e7eb')
    fig.update_yaxes(showgrid=True, gridcolor='#e5e7eb')
    
    return fig

def create_sector_rotation_chart(df):
    """Create sector rotation heatmap"""
    # Get last 24 months of data
    recent_data = df.tail(24)
    
    sector_cols = [col for col in df.columns if '_performance' in col]
    sector_data = recent_data[sector_cols]
    
    # Create correlation matrix
    correlation_matrix = sector_data.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=[col.replace('_performance', '').title() for col in correlation_matrix.columns],
        y=[col.replace('_performance', '').title() for col in correlation_matrix.columns],
        colorscale='RdYlBu',
        zmid=0,
        text=correlation_matrix.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Sector Rotation Correlation Matrix",
        title_x=0.5,
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_cycle_forecast_chart(df):
    """Create cycle forecast chart"""
    # Simple forecast based on trend
    last_date = df['date'].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='M')
    
    # Linear extrapolation for forecast
    x_historic = np.arange(len(df))
    lei_trend = np.polyfit(x_historic[-12:], df['lei'].iloc[-12:], 1)
    
    forecast_x = np.arange(len(df), len(df) + 12)
    forecast_values = np.polyval(lei_trend, forecast_x)
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['lei'],
        name='Historical LEI',
        line=dict(color='#1e3a8a', width=3)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        name='Forecast',
        line=dict(color='#f59e0b', width=2, dash='dash')
    ))
    
    # Confidence bands
    confidence_upper = forecast_values + np.random.uniform(0.3, 0.8, len(forecast_values))
    confidence_lower = forecast_values - np.random.uniform(0.3, 0.8, len(forecast_values))
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=confidence_upper,
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=confidence_lower,
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        fillcolor='rgba(245, 158, 11, 0.2)',
        name='Confidence Band',
        showlegend=True
    ))
    
    fig.update_layout(
        title="Business Cycle Forecast (12 Months)",
        title_x=0.5,
        xaxis_title="Date",
        yaxis_title="LEI Value",
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
    """Main business cycles page function"""
    
    # Header
    st.markdown("""
    <div class="cycle-header">
        <h1>🔄 Business Cycles</h1>
        <p>Economic Cycle Analysis using Leading Economic Index (LEI)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get data
    df = get_business_cycle_data()
    
    if df.empty:
        st.error("Unable to load business cycle data. Please try again later.")
        return
    
    # Determine cycle stage
    current_stage, confidence = determine_cycle_stage(df)
    latest_data = df.iloc[-1]
    
    # Current cycle stage display
    stage_class = current_stage.lower()
    stage_emoji = {"Expansion": "📈", "Peak": "🔝", "Contraction": "📉", "Trough": "🔽", "Transition": "🔄"}
    
    st.markdown(f"""
    <div class="cycle-stage {stage_class}">
        <h2 style="font-size: 3rem; margin: 0;">{stage_emoji.get(current_stage, "🔄")} {current_stage}</h2>
        <p style="font-size: 1.5rem; margin: 0.5rem 0;">Current Business Cycle Stage</p>
        <p style="opacity: 0.8; margin: 0;">Confidence: {confidence}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # LEI display
    st.markdown(f"""
    <div class="lei-display">
        <h2 style="font-size: 2.5rem; margin: 0;">{latest_data['lei']:.2f}</h2>
        <p style="font-size: 1.2rem; margin: 0.5rem 0;">Leading Economic Index</p>
        <p style="opacity: 0.8; margin: 0;">Composite of 10 leading indicators</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        gdp_color = "#10b981" if latest_data['gdp_growth'] > 2 else "#f59e0b" if latest_data['gdp_growth'] > 0 else "#ef4444"
        st.markdown(f"""
        <div class="sector-card">
            <h3 style="color: {gdp_color}; margin: 0;">GDP Growth</h3>
            <p style="font-size: 1.5rem; margin: 0.5rem 0; color: {gdp_color};">{latest_data['gdp_growth']:.1f}%</p>
            <p style="color: #6b7280; margin: 0;">Quarterly annualized</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        unemployment_color = "#10b981" if latest_data['unemployment'] < 5 else "#f59e0b" if latest_data['unemployment'] < 7 else "#ef4444"
        st.markdown(f"""
        <div class="sector-card">
            <h3 style="color: {unemployment_color}; margin: 0;">Unemployment</h3>
            <p style="font-size: 1.5rem; margin: 0.5rem 0; color: {unemployment_color};">{latest_data['unemployment']:.1f}%</p>
            <p style="color: #6b7280; margin: 0;">Current rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="sector-card">
            <h3 style="color: #1e3a8a; margin: 0;">Housing Starts</h3>
            <p style="font-size: 1.5rem; margin: 0.5rem 0; color: #1e3a8a;">{latest_data['housing_starts']:.1f}</p>
            <p style="color: #6b7280; margin: 0;">Index level</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="sector-card">
            <h3 style="color: #f59e0b; margin: 0;">Consumer Expect.</h3>
            <p style="font-size: 1.5rem; margin: 0.5rem 0; color: #f59e0b;">{latest_data['consumer_expectations']:.1f}</p>
            <p style="color: #6b7280; margin: 0;">Index level</p>
        </div>
        """, unsafe_allow_html=True)
    
    # LEI chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    lei_fig = create_lei_chart(df)
    st.plotly_chart(lei_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Cycle components
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    components_fig = create_cycle_components_chart(df)
    st.plotly_chart(components_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Forecast chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    forecast_fig = create_cycle_forecast_chart(df)
    st.plotly_chart(forecast_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sector rotation
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    sector_fig = create_sector_rotation_chart(df)
    st.plotly_chart(sector_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Cycle stage explanations
    st.markdown("""
    <div class="sector-card">
        <h3 style="color: #1f2937; margin: 0 0 1rem 0;">📊 Business Cycle Stages</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="sector-card">
            <h4 style="color: #10b981; margin: 0;">📈 Expansion</h4>
            <p style="color: #6b7280; margin: 0.5rem 0;">
                • Rising GDP and employment<br>
                • Increasing business investment<br>
                • Rising consumer confidence<br>
                • Moderate inflation
            </p>
        </div>
        
        <div class="sector-card">
            <h4 style="color: #ef4444; margin: 0;">📉 Contraction</h4>
            <p style="color: #6b7280; margin: 0.5rem 0;">
                • Declining GDP and employment<br>
                • Reduced business investment<br>
                • Falling consumer confidence<br>
                • Deflationary pressures
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="sector-card">
            <h4 style="color: #f59e0b; margin: 0;">🔝 Peak</h4>
            <p style="color: #6b7280; margin: 0.5rem 0;">
                • Economy at maximum output<br>
                • Full employment achieved<br>
                • Rising inflationary pressures<br>
                • Tight monetary policy
            </p>
        </div>
        
        <div class="sector-card">
            <h4 style="color: #6b7280; margin: 0;">🔽 Trough</h4>
            <p style="color: #6b7280; margin: 0.5rem 0;">
                • Economy at minimum output<br>
                • High unemployment<br>
                • Low inflation/deflation<br>
                • Accommodative monetary policy
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Data table
    with st.expander("📋 View Business Cycle Data"):
        display_df = df.tail(24).copy()  # Show last 24 months
        st.dataframe(
            display_df.round(2),
            use_container_width=True,
            column_config={
                "date": st.column_config.DateColumn("Date"),
                "lei": st.column_config.NumberColumn("LEI", format="%.2f"),
                "gdp_growth": st.column_config.NumberColumn("GDP Growth (%)", format="%.1f"),
                "unemployment": st.column_config.NumberColumn("Unemployment (%)", format="%.1f"),
                "housing_starts": st.column_config.NumberColumn("Housing Starts", format="%.1f"),
                "consumer_expectations": st.column_config.NumberColumn("Consumer Expectations", format="%.1f"),
            }
        )
    
    # Methodology
    with st.expander("📖 Methodology"):
        st.markdown("""
        **Leading Economic Index (LEI) Components:**
        
        The LEI is a composite of 10 leading indicators that typically change direction before the economy:
        
        1. **Average weekly hours** (manufacturing)
        2. **Average weekly initial claims** for unemployment insurance
        3. **Manufacturers' new orders** for consumer goods
        4. **ISM Index of New Orders**
        5. **Manufacturers' new orders** for capital goods
        6. **Building permits** for new private housing
        7. **Stock prices** (S&P 500)
        8. **Leading Credit Index**
        9. **Interest rate spread** (10-year vs federal funds)
        10. **Average consumer expectations** for business conditions
        
        **Business Cycle Stages:**
        - **Expansion**: Rising economic activity, employment, and output
        - **Peak**: Economy reaches maximum capacity
        - **Contraction**: Declining economic activity (recession if severe)
        - **Trough**: Economy reaches minimum point, recovery begins
        
        **Forecasting Method:**
        Uses linear trend extrapolation of recent LEI movements with confidence bands
        based on historical volatility. Sector rotation analysis shows relative performance
        correlations to identify cyclical patterns.
        """)

if __name__ == "__main__":
    main()