import streamlit as st
st.set_page_config(initial_sidebar_state="collapsed")
# Custom CSS to style the top menu and the page layout
st.markdown("""
    <style>
    
    /* Home page card styling */
    .stButton > button {
        background-color: #3B3B3B;
        color: white;
        font-size: 14px;
        border-radius: 10px;
        padding: 20px;
        height: auto;
        width: auto;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .stButton:hover > button {
        background-color: #4C4C4C;
    }

    .section-title {
        text-align: center;
        font-size: 24px;
        margin-bottom: 30px;
        font-weight: bold;
        color: #FFFFFF;
    }

    .card-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 30px;
        flex-wrap: wrap;
    }
    .card {
        background-color: #4A4A4A;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        width: 140px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .card:hover {
        background-color: #5A5A5A;
        cursor: pointer;
    }
    .card h4 {
        color: white;
        font-size: 16px;
    }
    .card p {
        color: #9A9A9A;
        font-size: 14px;
    }
    .card-icon {
        font-size: 24px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


# Welcome message with custom title
st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>Tic Tak Macro</h1>", unsafe_allow_html=True)

# Cards section with the navigation links for each page
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div class='card'>
        <a href='/growth' target='_self' style='color: white; text-decoration: none;'>
            <div class="card-icon">📈</div>
            <h4>Growth</h4>
            <p>Understand the dynamics of economic growth.</p>
        </a>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='card'>
        <a href='/Inflation_outlook' target='_self' style='color: white; text-decoration: none;'>
            <div class="card-icon">💸</div>
            <h4>Inflation</h4>
            <p>Analyze price trends and inflation rates.</p>
        </a>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='card'>
        <a href='/Risk_on_off' target='_self' style='color: white; text-decoration: none;'>
            <div class="card-icon">⚖️</div>
            <h4>Risk On/Off</h4>
            <p>Monitor market risk sentiment.</p>
        </a>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class='card'>
        <a href='/Sector_Business_Cycle' target='_self' style='color: white; text-decoration: none;'>
            <div class="card-icon">🔄</div>
            <h4>Business Cycle</h4>
            <p>Explore phases of economic cycles.</p>
        </a>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class='card'>
        <a href='/Primary_Dealer' target='_self' style='color: white; text-decoration: none;'>
            <div class="card-icon">🏦</div>
            <h4>Primary Dealer</h4>
            <p>Explore Federal Reserve data and issuance trends.</p>
        </a>
    </div>
    """, unsafe_allow_html=True)
