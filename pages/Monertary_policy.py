import streamlit as st
import time
import numpy as np

st.set_page_config(page_title="Monetary Policy", page_icon="📈",theme="dark")
st.markdown(
    """
    <style>
    body {
        background-color: #0c0c0d;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Monetary Policy")
st.write(
    """Monetary Policy incoming (brainard curve)"""
)
