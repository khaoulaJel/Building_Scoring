import streamlit as st
from pathlib import Path

def load_css(css_file_path):
    """
    Load CSS from a file and inject it into the Streamlit app
    
    Args:
        css_file_path (str): Path to the CSS file
    """
    with open(css_file_path, "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)