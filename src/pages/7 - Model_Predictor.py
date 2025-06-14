import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io

# src klasörünü Python path'ine ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
  sys.path.insert(0, src_dir)

st.set_page_config(page_title="Real Estate Data Analysis - Model Prediction", layout="wide")

# Create models directory
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(models_dir, exist_ok=True)

st.title("Model Prediction")

# Model check
if 'saved_models' not in st.session_state or not st.session_state.saved_models:
    st.warning("⚠️ Please train a model first in the Model Manager page!")
    if st.button("Go to Model Manager Page"):
        st.switch_page("pages/model_manager.py")
    st.stop()

# Model selection
st.header("1. Model Selection")
selected_model = st.selectbox(
    "Select a model for prediction:",
    list(st.session_state.saved_models.keys())
)

model_data = st.session_state.saved_models[selected_model]

# Load model from file
try:
    model = joblib.load(model_data['model_path'])
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Show model info
st.sidebar.header("Model Information")
st.sidebar.write(f"**Model Type:** {model_data['model_type']}")
st.sidebar.write(f"**Target Variable:** {model_data.get('target', 'N/A')}")
st.sidebar.write(f"**Used Features:** {', '.join(model_data['features'])}")
st.sidebar.write("**Model Metrics:**")
try:
    metrics = model_data.get('metrics', {})
    if 'test_mse' in metrics:
        st.sidebar.write(f"- Test MSE: {metrics['test_mse']:.4f}")
    if 'train_mse' in metrics:
        st.sidebar.write(f"- Train MSE: {metrics['train_mse']:.4f}")
    if 'test_r2' in metrics:
        st.sidebar.write(f"- Test R²: {metrics['test_r2']:.4f}")
    if 'train_r2' in metrics:
        st.sidebar.write(f"- Train R²: {metrics['train_r2']:.4f}")
except Exception as e:
    st.sidebar.warning("Metrics information not available")

# Prediction mode selection
st.header("2. Prediction Mode")
prediction_mode = st.radio(
    "Select prediction mode:",
    ["Single Prediction", "Batch Prediction"]
)

if prediction_mode == "Single Prediction":
    st.header("3. Input Values")
    
    # Get feature values
    input_values = {}
    for feature in model_data['features']:
        if feature in ['Property Type', 'City']:
            # For categorical variables
            unique_values = st.session_state.processed_data[feature].unique()
            input_values[feature] = st.selectbox(f"{feature}:", unique_values)
        elif feature in ['Bathrooms', 'Bedrooms']:
            # For integer variables
            min_val = int(st.session_state.processed_data[feature].min())
            max_val = int(st.session_state.processed_data[feature].max())
            input_values[feature] = st.number_input(
                f"{feature}:",
                min_value=min_val,
                max_value=max_val,
                value=int(st.session_state.processed_data[feature].mean()),
                step=1
            )
        else:
            # For numerical variables
            min_val = st.session_state.processed_data[feature].min()
            max_val = st.session_state.processed_data[feature].max()
            input_values[feature] = st.number_input(
                f"{feature}:",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(st.session_state.processed_data[feature].mean())
            )
    
    # Make prediction
    if st.button("Predict"):
        try:
            # Prepare input data
            input_df = pd.DataFrame([input_values])
            
            # Only use features that were used during training
            input_df = input_df[model_data['features']]
            
            # Use model to predict
            prediction = model.predict(input_df)
            
            # Show result
            st.header("4. Prediction Result")
            st.success(f"Predicted {model_data.get('target', 'value')}: {prediction[0]:.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
else:  # Batch Prediction
    st.header("3. Data Upload")
    uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Read data
            input_df = pd.read_excel(uploaded_file)
            
            # Check required columns
            missing_cols = [col for col in model_data['features'] if col not in input_df.columns]
            if missing_cols:
                st.error(f"Missing columns: {', '.join(missing_cols)}")
            else:
                # Only use required columns
                input_df = input_df[model_data['features']]
                
                # Make predictions
                if st.button("Predict Batch"):
                    try:
                        # Use model to predict
                        predictions = model.predict(input_df)
                        
                        # Show results
                        st.header("4. Prediction Results")
                        results_df = input_df.copy()
                        results_df[f"Predicted {model_data['target']}"] = predictions
                        st.dataframe(results_df)
                        
                        # Download results
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            results_df.to_excel(writer, index=False, sheet_name='Predictions')
                        
                        output.seek(0)
                        st.download_button(
                            label="Download Prediction Results",
                            data=output,
                            file_name=f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
                    except Exception as e:
                        st.error(f"An error occurred during batch prediction: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")

# About the app
st.sidebar.header("About the App")
st.sidebar.info(
    """
    This page allows you to make predictions with your trained models.
    
    Features:
    1. Single prediction: Predict for a single data point
    2. Batch prediction: Predict in bulk from an Excel file
    3. Download results
    4. View model performance metrics
    """
)