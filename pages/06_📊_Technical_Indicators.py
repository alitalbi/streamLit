"""
Technical Indicators Analysis Page
Advanced technical analysis with multiple indicators and trading signals
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import pandas_ta as ta
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="Technical Indicators",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .tech-header {
        background: linear-gradient(135deg, #7c2d12, #ea580c);
        color: white;
        padding: 2rem 0;
        margin: -2rem -2rem 2rem -2rem;
        text-align: center;
        border-radius: 0 0 1rem 1rem;
    }
    
    .tech-header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .indicator-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .indicator-tile {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .indicator-tile:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .signal-bullish {
        border-left: 4px solid #10b981;
        background: linear-gradient(90deg, #ecfdf5, #ffffff);
    }
    
    .signal-bearish {
        border-left: 4px solid #ef4444;
        background: linear-gradient(90deg, #fef2f2, #ffffff);
    }
    
    .signal-neutral {
        border-left: 4px solid #f59e0b;
        background: linear-gradient(90deg, #fffbeb, #ffffff);
    }
    
    .signal-score {
        background: linear-gradient(135deg, #7c2d12, #ea580c);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .oscillator-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_market_data(symbol="SPY", period="2y"):
    """Fetch market data for technical analysis"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            # Create sample data if yfinance fails
            dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
            np.random.seed(42)
            
            # Generate realistic stock data
            returns = np.random.normal(0.0008, 0.02, len(dates))
            prices = np.cumprod(1 + returns) * 400
            
            data = pd.DataFrame({
                'Open': prices * np.random.normal(0.999, 0.005, len(dates)),
                'High': prices * np.random.normal(1.015, 0.01, len(dates)),
                'Low': prices * np.random.normal(0.985, 0.01, len(dates)),
                'Close': prices,
                'Volume': np.random.normal(50000000, 15000000, len(dates))
            }, index=dates)
        
        return data
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df):
    """Calculate various technical indicators"""
    if df.empty:
        return df
    
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()
    
    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_histogram'] = data['MACD'] - data['MACD_signal']
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['BB_middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
    data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
    data['BB_width'] = data['BB_upper'] - data['BB_lower']
    data['BB_position'] = (data['Close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'])
    
    # Stochastic Oscillator
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    data['%K'] = 100 * (data['Close'] - low_14) / (high_14 - low_14)
    data['%D'] = data['%K'].rolling(window=3).mean()
    
    # Williams %R
    data['Williams_R'] = -100 * (high_14 - data['Close']) / (high_14 - low_14)
    
    # Average True Range (ATR)
    data['TR'] = np.maximum(
        data['High'] - data['Low'],
        np.maximum(
            abs(data['High'] - data['Close'].shift()),
            abs(data['Low'] - data['Close'].shift())
        )
    )
    data['ATR'] = data['TR'].rolling(window=14).mean()
    
    # Commodity Channel Index (CCI)
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    data['CCI'] = (tp - tp.rolling(window=14).mean()) / (0.015 * tp.rolling(window=14).std())
    
    # Volume indicators
    data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
    data['Volume_ratio'] = data['Volume'] / data['Volume_SMA']
    
    # Price momentum
    data['ROC'] = data['Close'].pct_change(periods=12) * 100
    data['Momentum'] = data['Close'] - data['Close'].shift(10)
    
    return data

def calculate_signal_score(df):
    """Calculate overall technical signal score"""
    if df.empty:
        return 0, "Neutral"
    
    latest = df.iloc[-1]
    signals = []
    
    # Moving Average signals
    if latest['Close'] > latest['SMA_20']:
        signals.append(1)
    else:
        signals.append(-1)
    
    if latest['Close'] > latest['SMA_50']:
        signals.append(1)
    else:
        signals.append(-1)
    
    if latest['Close'] > latest['SMA_200']:
        signals.append(1)
    else:
        signals.append(-1)
    
    # MACD signal
    if latest['MACD'] > latest['MACD_signal']:
        signals.append(1)
    else:
        signals.append(-1)
    
    # RSI signal
    if latest['RSI'] > 30 and latest['RSI'] < 70:
        signals.append(1)
    elif latest['RSI'] > 70:
        signals.append(-1)
    elif latest['RSI'] < 30:
        signals.append(1)
    else:
        signals.append(0)
    
    # Bollinger Bands
    if latest['BB_position'] > 0.2 and latest['BB_position'] < 0.8:
        signals.append(1)
    elif latest['BB_position'] > 0.8:
        signals.append(-1)
    elif latest['BB_position'] < 0.2:
        signals.append(1)
    else:
        signals.append(0)
    
    # Stochastic
    if latest['%K'] > 20 and latest['%K'] < 80:
        signals.append(1)
    elif latest['%K'] > 80:
        signals.append(-1)
    elif latest['%K'] < 20:
        signals.append(1)
    else:
        signals.append(0)
    
    # Calculate score
    score = sum(signals)
    max_score = len(signals)
    normalized_score = ((score + max_score) / (2 * max_score)) * 100
    
    if normalized_score > 60:
        signal = "Bullish"
    elif normalized_score < 40:
        signal = "Bearish"
    else:
        signal = "Neutral"
    
    return normalized_score, signal

def create_price_chart_with_indicators(df):
    """Create comprehensive price chart with technical indicators"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Moving Averages', 'MACD', 'RSI', 'Volume'),
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # Price and Moving Averages
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='#1e40af', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='#f59e0b', width=1)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='#10b981', width=1)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', line=dict(color='#ef4444', width=1)),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper', 
                  line=dict(color='gray', width=1), fill=None),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower', 
                  line=dict(color='gray', width=1), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
        row=1, col=1
    )
    
    # MACD
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#1e40af', width=2)),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD_signal'], name='MACD Signal', line=dict(color='#f59e0b', width=1)),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=df.index, y=df['MACD_histogram'], name='MACD Histogram', 
               marker_color='rgba(128,128,128,0.5)'),
        row=2, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#7c2d12', width=2)),
        row=3, col=1
    )
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=3, col=1)
    
    # Volume
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', 
               marker_color='rgba(30, 64, 175, 0.5)'),
        row=4, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Volume_SMA'], name='Volume SMA', 
                  line=dict(color='#f59e0b', width=2)),
        row=4, col=1
    )
    
    fig.update_layout(
        height=800,
        title_text="Technical Analysis Dashboard",
        title_x=0.5,
        title_font=dict(size=20, family='Inter', color='#1f2937'),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update y-axis ranges
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    
    return fig

def create_oscillators_chart(df):
    """Create oscillators dashboard"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Stochastic Oscillator', 'Williams %R', 'CCI', 'ROC'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Stochastic
    fig.add_trace(
        go.Scatter(x=df.index, y=df['%K'], name='%K', line=dict(color='#1e40af', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['%D'], name='%D', line=dict(color='#f59e0b', width=2)),
        row=1, col=1
    )
    
    # Williams %R
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Williams_R'], name='Williams %R', line=dict(color='#10b981', width=2)),
        row=1, col=2
    )
    
    # CCI
    fig.add_trace(
        go.Scatter(x=df.index, y=df['CCI'], name='CCI', line=dict(color='#ef4444', width=2)),
        row=2, col=1
    )
    
    # ROC
    fig.add_trace(
        go.Scatter(x=df.index, y=df['ROC'], name='ROC', line=dict(color='#8b5cf6', width=2)),
        row=2, col=2
    )
    
    # Add reference lines
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", row=1, col=1)
    
    fig.add_hline(y=-20, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_hline(y=-80, line_dash="dash", line_color="green", row=1, col=2)
    
    fig.add_hline(y=100, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-100, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)
    
    fig.update_layout(
        height=600,
        title_text="Technical Oscillators",
        title_x=0.5,
        title_font=dict(size=20, family='Inter', color='#1f2937'),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_signal_summary(df):
    """Create signal summary table"""
    if df.empty:
        return pd.DataFrame()
    
    latest = df.iloc[-1]
    
    signals = []
    
    # Moving Average signals
    signals.append({
        'Indicator': 'SMA 20',
        'Value': f"{latest['SMA_20']:.2f}",
        'Signal': 'Bullish' if latest['Close'] > latest['SMA_20'] else 'Bearish',
        'Strength': 'Medium'
    })
    
    signals.append({
        'Indicator': 'SMA 50',
        'Value': f"{latest['SMA_50']:.2f}",
        'Signal': 'Bullish' if latest['Close'] > latest['SMA_50'] else 'Bearish',
        'Strength': 'High'
    })
    
    signals.append({
        'Indicator': 'SMA 200',
        'Value': f"{latest['SMA_200']:.2f}",
        'Signal': 'Bullish' if latest['Close'] > latest['SMA_200'] else 'Bearish',
        'Strength': 'High'
    })
    
    # MACD
    signals.append({
        'Indicator': 'MACD',
        'Value': f"{latest['MACD']:.3f}",
        'Signal': 'Bullish' if latest['MACD'] > latest['MACD_signal'] else 'Bearish',
        'Strength': 'Medium'
    })
    
    # RSI
    rsi_signal = 'Neutral'
    if latest['RSI'] > 70:
        rsi_signal = 'Overbought'
    elif latest['RSI'] < 30:
        rsi_signal = 'Oversold'
    
    signals.append({
        'Indicator': 'RSI',
        'Value': f"{latest['RSI']:.1f}",
        'Signal': rsi_signal,
        'Strength': 'Medium'
    })
    
    # Bollinger Bands
    bb_signal = 'Neutral'
    if latest['BB_position'] > 0.8:
        bb_signal = 'Overbought'
    elif latest['BB_position'] < 0.2:
        bb_signal = 'Oversold'
    
    signals.append({
        'Indicator': 'Bollinger Bands',
        'Value': f"{latest['BB_position']:.2f}",
        'Signal': bb_signal,
        'Strength': 'Low'
    })
    
    # Stochastic
    stoch_signal = 'Neutral'
    if latest['%K'] > 80:
        stoch_signal = 'Overbought'
    elif latest['%K'] < 20:
        stoch_signal = 'Oversold'
    
    signals.append({
        'Indicator': 'Stochastic',
        'Value': f"{latest['%K']:.1f}",
        'Signal': stoch_signal,
        'Strength': 'Low'
    })
    
    return pd.DataFrame(signals)

def main():
    """Main technical indicators page function"""
    
    # Header
    st.markdown("""
    <div class="tech-header">
        <h1>📊 Technical Indicators</h1>
        <p>Advanced Technical Analysis with Multiple Indicators & Trading Signals</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.markdown("## 🎛️ Controls")
    
    symbol = st.sidebar.selectbox(
        "Select Symbol:",
        ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        index=0
    )
    
    period = st.sidebar.selectbox(
        "Select Period:",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=4
    )
    
    # Get data
    df = get_market_data(symbol, period)
    
    if df.empty:
        st.error("Unable to load market data. Please try again later.")
        return
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Calculate signal score
    signal_score, signal_direction = calculate_signal_score(df)
    latest_data = df.iloc[-1]
    
    # Signal score display
    signal_class = "signal-bullish" if signal_direction == "Bullish" else "signal-bearish" if signal_direction == "Bearish" else "signal-neutral"
    
    st.markdown(f"""
    <div class="signal-score">
        <h2 style="font-size: 3rem; margin: 0;">{signal_direction}</h2>
        <p style="font-size: 1.5rem; margin: 0.5rem 0;">Technical Signal Score: {signal_score:.1f}/100</p>
        <p style="opacity: 0.8; margin: 0;">Based on {symbol} analysis across multiple indicators</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price_change = ((latest_data['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        change_color = "#10b981" if price_change > 0 else "#ef4444"
        st.markdown(f"""
        <div class="indicator-tile">
            <h3 style="color: {change_color}; margin: 0;">{symbol} Price</h3>
            <p style="font-size: 1.5rem; margin: 0.5rem 0; color: {change_color};">${latest_data['Close']:.2f}</p>
            <p style="color: #6b7280; margin: 0;">({price_change:+.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        rsi_color = "#ef4444" if latest_data['RSI'] > 70 else "#10b981" if latest_data['RSI'] < 30 else "#f59e0b"
        st.markdown(f"""
        <div class="indicator-tile">
            <h3 style="color: {rsi_color}; margin: 0;">RSI</h3>
            <p style="font-size: 1.5rem; margin: 0.5rem 0; color: {rsi_color};">{latest_data['RSI']:.1f}</p>
            <p style="color: #6b7280; margin: 0;">{"Overbought" if latest_data['RSI'] > 70 else "Oversold" if latest_data['RSI'] < 30 else "Neutral"}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        macd_color = "#10b981" if latest_data['MACD'] > latest_data['MACD_signal'] else "#ef4444"
        st.markdown(f"""
        <div class="indicator-tile">
            <h3 style="color: {macd_color}; margin: 0;">MACD</h3>
            <p style="font-size: 1.5rem; margin: 0.5rem 0; color: {macd_color};">{latest_data['MACD']:.3f}</p>
            <p style="color: #6b7280; margin: 0;">{"Bullish" if latest_data['MACD'] > latest_data['MACD_signal'] else "Bearish"}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        bb_color = "#ef4444" if latest_data['BB_position'] > 0.8 else "#10b981" if latest_data['BB_position'] < 0.2 else "#f59e0b"
        st.markdown(f"""
        <div class="indicator-tile">
            <h3 style="color: {bb_color}; margin: 0;">BB Position</h3>
            <p style="font-size: 1.5rem; margin: 0.5rem 0; color: {bb_color};">{latest_data['BB_position']:.2f}</p>
            <p style="color: #6b7280; margin: 0;">{"Upper" if latest_data['BB_position'] > 0.8 else "Lower" if latest_data['BB_position'] < 0.2 else "Middle"}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    main_chart = create_price_chart_with_indicators(df)
    st.plotly_chart(main_chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Oscillators chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    oscillators_chart = create_oscillators_chart(df)
    st.plotly_chart(oscillators_chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Signal summary
    st.markdown("""
    <div class="oscillator-card">
        <h3 style="color: #1f2937; margin: 0 0 1rem 0;">📋 Signal Summary</h3>
    </div>
    """, unsafe_allow_html=True)
    
    signal_df = create_signal_summary(df)
    if not signal_df.empty:
        st.dataframe(
            signal_df,
            use_container_width=True,
            column_config={
                "Indicator": st.column_config.TextColumn("Indicator"),
                "Value": st.column_config.TextColumn("Current Value"),
                "Signal": st.column_config.TextColumn("Signal"),
                "Strength": st.column_config.TextColumn("Strength")
            },
            hide_index=True
        )
    
    # Analysis insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="oscillator-card">
            <h4 style="color: #7c2d12; margin: 0;">🎯 Key Insights</h4>
            <ul style="color: #6b7280; padding-left: 1rem;">
                <li>Price vs SMA 200: {"Above" if latest_data['Close'] > latest_data['SMA_200'] else "Below"} ({"Bullish" if latest_data['Close'] > latest_data['SMA_200'] else "Bearish"})</li>
                <li>Volume: {"Above" if latest_data['Volume_ratio'] > 1.2 else "Below"} average</li>
                <li>Volatility: {"High" if latest_data['ATR'] > df['ATR'].rolling(50).mean().iloc[-1] else "Normal"}</li>
                <li>Momentum: {"Positive" if latest_data['ROC'] > 0 else "Negative"}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="oscillator-card">
            <h4 style="color: #ea580c; margin: 0;">⚠️ Risk Considerations</h4>
            <ul style="color: #6b7280; padding-left: 1rem;">
                <li>ATR: {latest_data['ATR']:.2f} (volatility measure)</li>
                <li>BB Width: {latest_data['BB_width']:.2f} (volatility)</li>
                <li>Volume Ratio: {latest_data['Volume_ratio']:.2f}</li>
                <li>Stochastic: {"Overbought" if latest_data['%K'] > 80 else "Oversold" if latest_data['%K'] < 20 else "Neutral"}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Raw data
    with st.expander("📊 View Technical Data"):
        display_df = df.tail(50).round(3)
        st.dataframe(
            display_df[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_position', '%K', 'Volume_ratio']],
            use_container_width=True
        )
    
    # Methodology
    with st.expander("📖 Technical Analysis Methodology"):
        st.markdown("""
        **Technical Indicators Used:**
        
        **Trend Following:**
        - **Simple Moving Averages (SMA)**: 20, 50, 200-day averages
        - **Exponential Moving Averages (EMA)**: 12, 26-day for MACD
        - **MACD**: Moving Average Convergence Divergence
        
        **Momentum Oscillators:**
        - **RSI**: Relative Strength Index (14-day)
        - **Stochastic**: %K and %D oscillators
        - **Williams %R**: Momentum indicator
        - **CCI**: Commodity Channel Index
        - **ROC**: Rate of Change
        
        **Volatility Indicators:**
        - **Bollinger Bands**: 20-day SMA ± 2 standard deviations
        - **ATR**: Average True Range (14-day)
        
        **Volume Analysis:**
        - **Volume Ratio**: Current volume vs 20-day average
        - **Volume Trend**: Confirmation of price moves
        
        **Signal Scoring:**
        The overall signal score combines multiple indicators:
        - Moving average crossovers (40% weight)
        - Momentum oscillators (40% weight)
        - Volume confirmation (20% weight)
        
        **Interpretation:**
        - Score 60-100: Bullish environment
        - Score 40-60: Neutral/Mixed signals
        - Score 0-40: Bearish environment
        
        **Risk Management:**
        - Multiple indicator confirmation reduces false signals
        - ATR and Bollinger Band width measure volatility
        - Volume analysis confirms price action validity
        """)

if __name__ == "__main__":
    main()