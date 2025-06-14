import streamlit as st
import sys
import os
import pandas as pd
from utils.config import PAGE_CONFIG
from utils.repo_manager import RepoManager
import numpy as np
from pathlib import Path

# src klasörünü Python path'ine ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
  sys.path.insert(0, current_dir)

# Add the src directory to the Python path
src_path = Path(__file__).parent
sys.path.append(str(src_path))

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Real Estate Data Analysis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Repository manager
repo = RepoManager()

# CSS stilleri
st.markdown("""
<style>
  .main-header {
      font-size: 3rem;
      color: #1f77b4;
      text-align: center;
      margin-bottom: 2rem;
  }
  .metric-card {
      background-color: #ffffff;
      padding: 1.5rem;
      border-radius: 0.5rem;
      border: 1px solid #e0e0e0;
      margin-bottom: 1rem;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
  .main {
      padding: 2rem;
  }
  .stTabs [data-baseweb="tab-list"] {
      gap: 2rem;
  }
  .stTabs [data-baseweb="tab"] {
      height: 4rem;
      white-space: pre-wrap;
      background-color: #f0f2f6;
      border-radius: 4px 4px 0 0;
      gap: 1rem;
      padding-top: 10px;
      padding-bottom: 10px;
  }
  .stTabs [aria-selected="true"] {
      background-color: #ffffff;
  }
  .css-1d391kg {
      padding: 1rem;
  }
</style>
""", unsafe_allow_html=True)

# Ana başlık
st.markdown('<h1 class="main-header">🏠 Real Estate Data Analysis</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
  st.markdown("### 📋 Navigation")
  st.markdown("Use the pages on the left to navigate through the application.")
  
  # Repository status
  storage_info = repo.get_storage_info()
  st.markdown("### 📊 Repository Status")
  st.metric("Datasets", storage_info['total_datasets'])
  st.metric("Processed Data", storage_info['total_processed'])
  st.metric("Models", storage_info['total_models'])
  st.metric("Storage Used", f"{storage_info['total_size_mb']:.1f} MB")

# Ana sayfa içeriği
st.markdown("## 📊 Introduction and Summary")
st.markdown("---")

# Introduction Section
st.markdown("""
    This capstone project explores a live Ontario rental dataset consisting of 2,850 listings from Realtor.com. 
    The dataset includes comprehensive features such as property type, building size, rooms, price, city, and crime rate.
    
    Our analysis involves:
    - 📈 Intense data cleaning and preprocessing
    - 🔧 Advanced feature engineering
    - 🤖 Predictive modeling
    - 📊 Various visualizations (box plots, heatmaps, bar charts)
""")

# Problem Description
with st.expander("🎯 Problem Description", expanded=True):
    st.markdown("""
        ### Real Estate Price Prediction Challenge
        
        This project addresses the challenge of predicting rental property values in Ontario, Canada using machine learning techniques.
        Our goal is to understand the complex relationships between various property features and rental prices, providing valuable insights
        for both property owners and potential renters.
    """)

# Research Questions
with st.expander("❓ Research Questions", expanded=True):
    st.markdown("""
        ### Key Research Questions
        
        1. **Property Features Impact**
           - How do building size, number of bedrooms, and bathrooms affect rental prices?
           - What is the relative importance of each feature?
        
        2. **Crime Rate Influence**
           - Does the crime rate in a location significantly impact rental pricing?
           - How does safety perception affect property values?
        
        3. **Prediction Accuracy**
           - Can machine learning models accurately predict rent prices?
           - Which model performs best for this specific use case?
    """)

# Literature Review
with st.expander("📚 Literature Review", expanded=True):
    st.markdown("""
        ### Key References
        
        1. **Canada Crime Report**
           - Crime Severity Index analysis
           - Regional safety metrics
        
        2. **Machine Learning Applications**
           - House Price Prediction using Machine Learning in Python
           - Exploratory Data Analysis for House Rent Prediction
        
        3. **Academic Research**
           - Enhancing House Rental Price Prediction Models
           - Uncertainty Management in Price Prediction
        
        4. **Data Sources**
           - Realtor.com Ontario Rent Listings
           - Current market data analysis
    """)

# Project Contributions
with st.expander("💡 Project Contributions", expanded=True):
    st.markdown("""
        ### Key Contributions
        
        1. **Geographic Relevance**
           - Focused analysis on Ontario, Canada
           - Current data from Realtor.com
        
        2. **Contextual Analysis**
           - Crime rate variations
           - Monthly market trends
        
        3. **Advanced Analytics**
           - K-Means clustering for city grouping
           - Risk and rent level analysis
        
        4. **Model Performance**
           - Multiple model comparison
           - Feature importance analysis
        
        5. **Practical Applications**
           - Urban planning insights
           - Affordable housing analysis
           - Marketing strategy support
    """)

# Navigation
st.markdown("---")
st.markdown("""
    ### 🚀 Get Started
    
    Use the sidebar to navigate through different sections of the analysis:
    
    1. **Data Preprocessing**: Clean and prepare your data
    2. **Model Manager**: Train and evaluate different models
    3. **Model Predictor**: Make predictions with trained models
""")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Created with ❤️ using Streamlit via Sara </p>
    </div>
""", unsafe_allow_html=True)
# Session state initialization
if 'data' not in st.session_state:
  st.session_state.data = None
if 'processed_data' not in st.session_state:
  st.session_state.processed_data = None
if 'processing_steps' not in st.session_state:
  st.session_state.processing_steps = []
