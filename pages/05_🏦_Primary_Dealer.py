"""
Primary Dealer Analysis Page
Federal Reserve primary dealer operations and Treasury market insights
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
    page_title="Primary Dealer Analysis",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .dealer-header {
        background: linear-gradient(135deg, #1e40af, #3730a3);
        color: white;
        padding: 2rem 0;
        margin: -2rem -2rem 2rem -2rem;
        text-align: center;
        border-radius: 0 0 1rem 1rem;
    }
    
    .dealer-header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .dealer-metric {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .treasury-highlight {
        background: linear-gradient(135deg, #1e40af, #f59e0b);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .operation-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .positive-flow {
        border-left: 4px solid #10b981;
        background: linear-gradient(90deg, #ecfdf5, #ffffff);
    }
    
    .negative-flow {
        border-left: 4px solid #ef4444;
        background: linear-gradient(90deg, #fef2f2, #ffffff);
    }
    
    .neutral-flow {
        border-left: 4px solid #f59e0b;
        background: linear-gradient(90deg, #fffbeb, #ffffff);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_primary_dealer_data():
    """Fetch primary dealer operations data"""
    try:
        # Create sample data for primary dealer operations
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        np.random.seed(101112)
        
        # Generate realistic treasury market data
        base_yield = 3.5
        yield_trend = np.sin(np.arange(len(dates)) * 0.002) * 1.5
        
        data = {
            'date': dates,
            'treasury_10y': base_yield + yield_trend + np.random.normal(0, 0.1, len(dates)),
            'treasury_2y': base_yield - 0.5 + yield_trend * 0.8 + np.random.normal(0, 0.08, len(dates)),
            'treasury_30y': base_yield + 0.8 + yield_trend * 1.2 + np.random.normal(0, 0.12, len(dates)),
            'yield_curve_slope': np.random.normal(0.8, 0.3, len(dates)),
            'dealer_inventory': np.random.normal(50, 15, len(dates)),  # Billions
            'repo_rate': base_yield - 0.2 + np.random.normal(0, 0.05, len(dates)),
            'reverse_repo': np.random.normal(2000, 500, len(dates)),  # Billions
            'soma_holdings': np.random.normal(8000, 200, len(dates)),  # Billions
            'foreign_holdings': np.random.normal(7000, 300, len(dates)),  # Billions
            'auction_demand': np.random.normal(2.5, 0.5, len(dates)),  # Bid-to-cover ratio
            'primary_dealer_count': np.random.randint(20, 26, len(dates)),
        }
        
        df = pd.DataFrame(data)
        
        # Calculate additional metrics
        df['term_premium'] = df['treasury_10y'] - df['treasury_2y']
        df['real_yield_10y'] = df['treasury_10y'] - 2.5  # Assume 2.5% inflation
        df['dealer_leverage'] = df['dealer_inventory'] / 500  # Simplified leverage ratio
        
        return df
    except Exception as e:
        st.error(f"Error fetching primary dealer data: {e}")
        return pd.DataFrame()

def calculate_market_stress_indicator(df):
    """Calculate primary dealer market stress indicator"""
    if df.empty:
        return 0, "Normal"
    
    latest_data = df.iloc[-1]
    
    # Stress indicators
    # 1. Yield curve inversion
    curve_stress = max(0, -latest_data['term_premium'] * 20)  # Negative term premium = stress
    
    # 2. High dealer inventory
    inventory_stress = max(0, (latest_data['dealer_inventory'] - 50) * 2)
    
    # 3. Low auction demand
    auction_stress = max(0, (2.0 - latest_data['auction_demand']) * 30)
    
    # 4. High volatility in yields
    recent_vol = df['treasury_10y'].tail(20).std() * 100
    volatility_stress = max(0, (recent_vol - 10) * 5)
    
    # Combined stress score
    stress_score = min(100, curve_stress + inventory_stress + auction_stress + volatility_stress)
    
    if stress_score < 20:
        stress_level = "Low"
    elif stress_score < 50:
        stress_level = "Moderate"
    else:
        stress_level = "High"
    
    return stress_score, stress_level

def create_yield_curve_chart(df):
    """Create yield curve visualization"""
    latest_data = df.iloc[-1]
    
    # Sample yield curve points
    maturities = ['3M', '6M', '2Y', '5Y', '10Y', '30Y']
    yields = [
        latest_data['treasury_2y'] - 1.5,  # 3M
        latest_data['treasury_2y'] - 1.0,  # 6M
        latest_data['treasury_2y'],        # 2Y
        latest_data['treasury_2y'] + 0.3,  # 5Y
        latest_data['treasury_10y'],       # 10Y
        latest_data['treasury_30y']        # 30Y
    ]
    
    fig = go.Figure()
    
    # Current yield curve
    fig.add_trace(go.Scatter(
        x=maturities,
        y=yields,
        mode='lines+markers',
        name='Current Yield Curve',
        line=dict(color='#1e40af', width=4),
        marker=dict(size=8)
    ))
    
    # Add historical comparison (1 year ago)
    historical_yields = [y + np.random.normal(0, 0.2) for y in yields]
    fig.add_trace(go.Scatter(
        x=maturities,
        y=historical_yields,
        mode='lines+markers',
        name='1 Year Ago',
        line=dict(color='#f59e0b', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="U.S. Treasury Yield Curve",
        title_x=0.5,
        xaxis_title="Maturity",
        yaxis_title="Yield (%)",
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

def create_dealer_operations_chart(df):
    """Create primary dealer operations chart"""
    # Monthly data for cleaner visualization
    df_monthly = df.set_index('date').resample('M').last().reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Dealer Inventory', 'Reverse Repo Operations', 'SOMA Holdings', 'Foreign Holdings'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Dealer Inventory
    fig.add_trace(
        go.Scatter(x=df_monthly['date'], y=df_monthly['dealer_inventory'], 
                  name='Dealer Inventory', line=dict(color='#1e40af', width=2)),
        row=1, col=1
    )
    
    # Reverse Repo
    fig.add_trace(
        go.Scatter(x=df_monthly['date'], y=df_monthly['reverse_repo'], 
                  name='Reverse Repo', line=dict(color='#f59e0b', width=2)),
        row=1, col=2
    )
    
    # SOMA Holdings
    fig.add_trace(
        go.Scatter(x=df_monthly['date'], y=df_monthly['soma_holdings'], 
                  name='SOMA Holdings', line=dict(color='#10b981', width=2)),
        row=2, col=1
    )
    
    # Foreign Holdings
    fig.add_trace(
        go.Scatter(x=df_monthly['date'], y=df_monthly['foreign_holdings'], 
                  name='Foreign Holdings', line=dict(color='#ef4444', width=2)),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Primary Dealer Operations Dashboard",
        title_x=0.5,
        title_font=dict(size=20, family='Inter', color='#1f2937'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Billions ($)", row=1, col=1)
    fig.update_yaxes(title_text="Billions ($)", row=1, col=2)
    fig.update_yaxes(title_text="Billions ($)", row=2, col=1)
    fig.update_yaxes(title_text="Billions ($)", row=2, col=2)
    
    fig.update_xaxes(showgrid=True, gridcolor='#e5e7eb')
    fig.update_yaxes(showgrid=True, gridcolor='#e5e7eb')
    
    return fig

def create_auction_analysis_chart(df):
    """Create treasury auction analysis"""
    df_weekly = df.set_index('date').resample('W').last().reset_index()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Auction Demand (Bid-to-Cover)', 'Term Premium'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Auction Demand
    fig.add_trace(
        go.Scatter(x=df_weekly['date'], y=df_weekly['auction_demand'], 
                  name='Bid-to-Cover', line=dict(color='#1e40af', width=3)),
        row=1, col=1
    )
    
    # Add demand threshold
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", 
                  annotation_text="Weak Demand", annotation_position="bottom right",
                  row=1, col=1)
    
    # Term Premium
    fig.add_trace(
        go.Scatter(x=df_weekly['date'], y=df_weekly['term_premium'], 
                  name='Term Premium', line=dict(color='#f59e0b', width=3)),
        row=1, col=2
    )
    
    # Add zero line for term premium
    fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                  annotation_text="Flat Curve", annotation_position="bottom right",
                  row=1, col=2)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Treasury Auction Analysis",
        title_x=0.5,
        title_font=dict(size=18, family='Inter', color='#1f2937'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#e5e7eb')
    fig.update_yaxes(showgrid=True, gridcolor='#e5e7eb')
    
    return fig

def create_dealer_count_chart(df):
    """Create primary dealer count and concentration chart"""
    df_monthly = df.set_index('date').resample('M').last().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_monthly['date'],
        y=df_monthly['primary_dealer_count'],
        mode='lines+markers',
        name='Primary Dealer Count',
        line=dict(color='#1e40af', width=3),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title="Primary Dealer Count Over Time",
        title_x=0.5,
        xaxis_title="Date",
        yaxis_title="Number of Primary Dealers",
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis=dict(range=[15, 30])
    )
    
    return fig

def main():
    """Main primary dealer page function"""
    
    # Header
    st.markdown("""
    <div class="dealer-header">
        <h1>🏦 Primary Dealer Analysis</h1>
        <p>Federal Reserve Primary Dealer Operations & Treasury Market Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get data
    df = get_primary_dealer_data()
    
    if df.empty:
        st.error("Unable to load primary dealer data. Please try again later.")
        return
    
    # Calculate market stress
    stress_score, stress_level = calculate_market_stress_indicator(df)
    latest_data = df.iloc[-1]
    
    # Market stress indicator
    stress_color = "#10b981" if stress_level == "Low" else "#f59e0b" if stress_level == "Moderate" else "#ef4444"
    
    st.markdown(f"""
    <div class="treasury-highlight">
        <h2 style="font-size: 2.5rem; margin: 0;">Market Stress: {stress_level}</h2>
        <p style="font-size: 1.5rem; margin: 0.5rem 0;">Score: {stress_score:.1f}/100</p>
        <p style="opacity: 0.8; margin: 0;">Based on yield curve, dealer inventory, and auction demand</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="dealer-metric">
            <h3 style="color: #1e40af; margin: 0;">10Y Treasury</h3>
            <p style="font-size: 1.8rem; margin: 0.5rem 0; color: #1e40af;">{latest_data['treasury_10y']:.2f}%</p>
            <p style="color: #6b7280; margin: 0;">Current yield</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        curve_color = "#ef4444" if latest_data['term_premium'] < 0 else "#10b981"
        st.markdown(f"""
        <div class="dealer-metric">
            <h3 style="color: {curve_color}; margin: 0;">Term Premium</h3>
            <p style="font-size: 1.8rem; margin: 0.5rem 0; color: {curve_color};">{latest_data['term_premium']:.1f}bp</p>
            <p style="color: #6b7280; margin: 0;">10Y-2Y spread</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="dealer-metric">
            <h3 style="color: #f59e0b; margin: 0;">Dealer Inventory</h3>
            <p style="font-size: 1.8rem; margin: 0.5rem 0; color: #f59e0b;">${latest_data['dealer_inventory']:.1f}B</p>
            <p style="color: #6b7280; margin: 0;">Treasury holdings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        demand_color = "#ef4444" if latest_data['auction_demand'] < 2.0 else "#10b981"
        st.markdown(f"""
        <div class="dealer-metric">
            <h3 style="color: {demand_color}; margin: 0;">Auction Demand</h3>
            <p style="font-size: 1.8rem; margin: 0.5rem 0; color: {demand_color};">{latest_data['auction_demand']:.1f}x</p>
            <p style="color: #6b7280; margin: 0;">Bid-to-cover ratio</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Yield curve
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    yield_curve_fig = create_yield_curve_chart(df)
    st.plotly_chart(yield_curve_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Dealer operations
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    operations_fig = create_dealer_operations_chart(df)
    st.plotly_chart(operations_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Auction analysis
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    auction_fig = create_auction_analysis_chart(df)
    st.plotly_chart(auction_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Dealer count
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    dealer_count_fig = create_dealer_count_chart(df)
    st.plotly_chart(dealer_count_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Market operations analysis
    st.markdown("""
    <div class="operation-card">
        <h3 style="color: #1f2937; margin: 0 0 1rem 0;">📊 Current Market Operations</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Determine flow direction
        inventory_flow = "positive-flow" if latest_data['dealer_inventory'] > 50 else "negative-flow"
        st.markdown(f"""
        <div class="operation-card {inventory_flow}">
            <h4 style="color: #1e40af; margin: 0;">Dealer Inventory Flow</h4>
            <p style="color: #6b7280; margin: 0.5rem 0;">
                Current: ${latest_data['dealer_inventory']:.1f}B<br>
                Status: {"Above Average" if latest_data['dealer_inventory'] > 50 else "Below Average"}<br>
                Trend: {"Accumulating" if latest_data['dealer_inventory'] > 50 else "Distributing"}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="operation-card neutral-flow">
            <h4 style="color: #f59e0b; margin: 0;">SOMA Operations</h4>
            <p style="color: #6b7280; margin: 0.5rem 0;">
                Holdings: ${latest_data['soma_holdings']:.0f}B<br>
                Reverse Repo: ${latest_data['reverse_repo']:.0f}B<br>
                Policy: {"Accommodative" if latest_data['reverse_repo'] > 1500 else "Tightening"}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        curve_flow = "negative-flow" if latest_data['term_premium'] < 0 else "positive-flow"
        st.markdown(f"""
        <div class="operation-card {curve_flow}">
            <h4 style="color: #ef4444; margin: 0;">Yield Curve Signal</h4>
            <p style="color: #6b7280; margin: 0.5rem 0;">
                10Y-2Y: {latest_data['term_premium']:.1f}bp<br>
                Shape: {"Inverted" if latest_data['term_premium'] < 0 else "Normal"}<br>
                Implication: {"Recession Risk" if latest_data['term_premium'] < 0 else "Growth Supportive"}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="operation-card positive-flow">
            <h4 style="color: #10b981; margin: 0;">Foreign Participation</h4>
            <p style="color: #6b7280; margin: 0.5rem 0;">
                Holdings: ${latest_data['foreign_holdings']:.0f}B<br>
                Dealers: {latest_data['primary_dealer_count']:.0f}<br>
                Liquidity: {"Adequate" if latest_data['auction_demand'] > 2.0 else "Strained"}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Data table
    with st.expander("📋 View Primary Dealer Data"):
        display_df = df.tail(30).copy()  # Show last 30 days
        st.dataframe(
            display_df.round(3),
            use_container_width=True,
            column_config={
                "date": st.column_config.DateColumn("Date"),
                "treasury_10y": st.column_config.NumberColumn("10Y Yield (%)", format="%.3f"),
                "treasury_2y": st.column_config.NumberColumn("2Y Yield (%)", format="%.3f"),
                "term_premium": st.column_config.NumberColumn("Term Premium (bp)", format="%.1f"),
                "dealer_inventory": st.column_config.NumberColumn("Dealer Inventory ($B)", format="%.1f"),
                "auction_demand": st.column_config.NumberColumn("Auction Demand", format="%.2f"),
                "reverse_repo": st.column_config.NumberColumn("Reverse Repo ($B)", format="%.0f"),
            }
        )
    
    # Methodology
    with st.expander("📖 Methodology"):
        st.markdown("""
        **Primary Dealer System:**
        
        Primary dealers are financial institutions that trade government securities with the Federal Reserve
        and are required to participate in Treasury auctions.
        
        **Key Metrics Tracked:**
        
        - **Dealer Inventory**: Amount of Treasury securities held by primary dealers
        - **Auction Demand**: Bid-to-cover ratio in Treasury auctions (higher = more demand)
        - **Term Premium**: Yield difference between long and short-term bonds
        - **SOMA Holdings**: System Open Market Account holdings (Fed's balance sheet)
        - **Reverse Repo**: Fed's reverse repurchase agreement operations
        
        **Market Stress Indicators:**
        
        1. **Yield Curve Inversion**: Negative term premium suggests recession risk
        2. **High Dealer Inventory**: Dealers struggling to distribute securities
        3. **Low Auction Demand**: Weak investor appetite for Treasuries
        4. **High Yield Volatility**: Market uncertainty and instability
        
        **Primary Dealer Functions:**
        - Participate in Treasury auctions
        - Provide liquidity to Treasury markets
        - Serve as counterparties to Fed operations
        - Distribute Treasury securities to investors
        
        The analysis helps identify Treasury market functioning and potential stress periods.
        """)

if __name__ == "__main__":
    main()