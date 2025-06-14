import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import io
import os
import json
from datetime import datetime
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Real Estate Data Analysis - Data Preparation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'preprocessing_steps' not in st.session_state:
    st.session_state.preprocessing_steps = []
if 'data_validation_status' not in st.session_state:
    st.session_state.data_validation_status = {
        'is_valid': False,
        'issues': []
    }

def validate_data(df):
    """Validate data for preprocessing"""
    issues = []
    
    # Check if dataframe is empty
    if df.empty:
        issues.append("DataFrame is empty")
        return False, issues
    
    # Check required columns
    required_columns = ['Property Price']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check for missing values in critical columns
    critical_columns = ['Property Price', 'Building Size', 'Rooms']
    for col in critical_columns:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                issues.append(f"Column '{col}' has {missing_count} missing values")
    
    # Check data types
    if 'Property Price' in df.columns:
        try:
            pd.to_numeric(df['Property Price'], errors='raise')
        except:
            issues.append("Property Price column contains non-numeric values")
    
    return len(issues) == 0, issues

def update_validation_status(is_valid, issues):
    """Update data validation status in session state"""
    st.session_state.data_validation_status = {
        'is_valid': is_valid,
        'issues': issues
    }

st.title("Real Estate Data Preparation and Preprocessing")

# Sidebar with operation options
st.sidebar.header("Operation Options")

# 1. Data Upload
st.header("1. Data Upload")
uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.success(f"File uploaded successfully. Row count: {df.shape[0]}, Column count: {df.shape[1]}")
        
        # Validate data
        is_valid, issues = validate_data(df)
        update_validation_status(is_valid, issues)
        
        if not is_valid:
            st.warning("⚠️ Data validation issues found:")
            for issue in issues:
                st.write(f"- {issue}")
        else:
            st.success("✅ Data validation passed!")
        
        # Store the raw data in session state
        st.session_state.raw_data = df
        
        # Show data
        st.subheader("Uploaded Data")
        st.dataframe(df.head())
        
        # Show data info
        st.subheader("Data Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Data Types:**")
            st.write(df.dtypes)
        with col2:
            st.write("**Missing Values:**")
            st.write(df.isnull().sum())
        
        # 2. Data Cleaning
        st.header("2. Data Cleaning")
        
        # Cleaning options
        cleaning_options = st.multiselect(
            "Select cleaning operations to apply:",
            ["Remove duplicate records", "Remove unnecessary columns", "Filter rows containing Quebec"]
        )
        
        if cleaning_options:
            original_df = df.copy()
            
            if "Remove duplicate records" in cleaning_options and 'URL' in df.columns:
                original_rows = df.shape[0]
                df.drop_duplicates(subset=['URL'], inplace=True)
                df.reset_index(drop=True, inplace=True)
                st.info(f"{original_rows - df.shape[0]} duplicate records removed.")
                st.session_state.preprocessing_steps.append("Removed duplicate records")
            
            if "Remove unnecessary columns" in cleaning_options:
                columns_to_drop = st.multiselect(
                    "Select columns to remove:",
                    df.columns.tolist(),
                    default=['Listing Status', 'Year Built', 'Architecture Style', 'Num Floors', 
                             'Provider', 'Subdivision', 'Brokerage Office Name', 'URL', 'Source Page']
                              if all(col in df.columns for col in ['Listing Status', 'Year Built', 'Architecture Style']) else []
                )
                if columns_to_drop:
                    df = df.drop(columns=columns_to_drop)
                    st.info(f"{len(columns_to_drop)} columns removed.")
                    st.session_state.preprocessing_steps.append(f"Removed columns: {', '.join(columns_to_drop)}")
            
            if "Filter rows containing Quebec" in cleaning_options and 'Address' in df.columns:
                original_rows = df.shape[0]
                df = df[~df['Address'].str.contains('Québec', na=False)]
                
                st.info(f"{original_rows - df.shape[0]} rows containing Quebec filtered.")
                st.session_state.preprocessing_steps.append("Filtered Quebec rows")
            
            # Update processed data in session state
            st.session_state.processed_data = df
            
            # Show changes
            if not df.equals(original_df):
                st.subheader("Changes Made")
                st.write("Original shape:", original_df.shape)
                st.write("New shape:", df.shape)
        
        # 3. Data Type Conversions
        st.header("3. Data Type Conversions")
        
        if st.checkbox("Apply string conversions"):
            string_columns = st.multiselect(
                "Select columns to convert to string:",
                df.columns.tolist(),
                default=['Address', 'Property Type', 'Property Description'] 
                if all(col in df.columns for col in ['Address', 'Property Type', 'Property Description']) else []
            )
            
            if string_columns:
                for col in string_columns:
                    if col in df.columns:
                        df[col] = df[col].astype(str)
                
                st.info(f"{len(string_columns)} columns converted to string.")
                st.session_state.preprocessing_steps.append(f"Converted to string: {', '.join(string_columns)}")
                st.session_state.processed_data = df
        
        if 'Property Price' in df.columns and st.checkbox("Clean price column"):
            df['Currency'] = df['Property Price'].str.extract(r'^(\w+)', expand=False)
            df['Property Price'] = df['Property Price'].str.extract(r'[\$€£]([\d,]+)', expand=False)
            df['Property Price'] = df['Property Price'].str.replace(',', '').astype(float)
            st.info("Property Price column cleaned and converted to numeric values.")
            st.session_state.preprocessing_steps.append("Cleaned Property Price column")
            st.session_state.processed_data = df
        
        # 4. Feature Transformation
        st.header("4. Feature Transformation")
        
        feature_options = []
        if 'Rooms' in df.columns:
            feature_options.append("Extract Bedrooms and Bathrooms columns from Rooms column")
        if 'Building Size' in df.columns:
            feature_options.append("Clean Building Size column")
        if 'Published On' in df.columns and 'Last Updated On' in df.columns:
            feature_options.append("Clean date columns")
            
        selected_transformations = st.multiselect(
            "Select feature transformations to apply:",
            feature_options
        )
        
        if selected_transformations:
            original_df = df.copy()
            
            if "Extract Bedrooms and Bathrooms columns from Rooms column" in selected_transformations and 'Rooms' in df.columns:
                df['Bedrooms'] = df['Rooms'].str.extract(r'(\d+)\s+bedroom', expand=False).astype('Int64')
                df['Bathrooms'] = df['Rooms'].str.extract(r'(\d+)\s+bathroom', expand=False).astype('Int64')
                df.drop(columns=['Rooms'], inplace=True)
                st.info("Bedrooms and Bathrooms columns created, Rooms column removed.")
                st.session_state.preprocessing_steps.append("Extracted Bedrooms and Bathrooms from Rooms")
                
                # Filtering
                if st.checkbox("Apply filtering for number of bedrooms and bathrooms"):
                    original_rows = df.shape[0]
                    df = df[(df['Bedrooms'] <= 6) & (df['Bathrooms'] <= 5)]
                    df.reset_index(drop=True, inplace=True)
                    st.info(f"{original_rows - df.shape[0]} rows filtered.")
                    st.session_state.preprocessing_steps.append("Filtered bedrooms and bathrooms")
            
            if "Clean Building Size column" in selected_transformations and 'Building Size' in df.columns:
                df['Building Size'] = (
                    df['Building Size'].astype(str)
                    .str.extract(r'([\d,.]+)', expand=False)
                    .str.replace(',', '', regex=True)
                    .astype(float)
                )
                st.info("Building Size column cleaned.")
                st.session_state.preprocessing_steps.append("Cleaned Building Size column")
            
            if "Clean date columns" in selected_transformations:
                # Month converter function
                month_map = {
                    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
                }
                
                # Date conversion function
                def convert_date(date_str):
                    if not isinstance(date_str, str) or date_str.strip().lower() == "Unknown":
                        return None
                    parts = date_str.strip().split()
                    if len(parts) == 3:
                        day, month, year = parts
                        month = month_map.get(month[:3], None)
                        if month:
                            try:
                                day = day.zfill(2)  # 1 → 01
                                return f"{day}/{month}/{year}"
                            except:
                                return None
                    return None
                
                for date_col, new_col in [('Published On', 'Published Date'), ('Last Updated On', 'Last Updated Date')]:
                    if date_col in df.columns:
                        df[f'{date_col} Formatted'] = df[date_col].astype(str).str.replace(':', '', regex=False).str.strip().apply(convert_date)
                        df[new_col] = pd.to_datetime(df[f'{date_col} Formatted'], format="%d/%m/%Y", errors='coerce')
                
                # Remove unnecessary columns
                cols_to_drop = ['Published On', 'Published Formatted', 'Last Updated On', 'Last Updated Formatted']
                df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
                df = df.reset_index(drop=True)
                st.info("Date columns cleaned and converted to datetime format.")
                st.session_state.preprocessing_steps.append("Cleaned date columns")
            
            # Update processed data in session state
            st.session_state.processed_data = df
            
            # Show changes
            if not df.equals(original_df):
                st.subheader("Changes Made")
                st.write("Original shape:", original_df.shape)
                st.write("New shape:", df.shape)
        
        # 5. Missing Value Handling
        st.header("5. Missing Value Handling")
        
        # Show missing value info
        st.subheader("Missing Value Count")
        missing_values = df.isnull().sum()
        st.dataframe(missing_values.to_frame(name='Missing Value Count'))
        
        if st.checkbox("Impute missing values with Random Forest"):
            try:
                # Advanced missing value imputation function
                def impute_missing_values(df):
                    # One-hot encoding for Property Type and City
                    columns_to_encode = ['Property Type', 'City']
                    existing_cols = [col for col in columns_to_encode if col in df.columns]
                    
                    if existing_cols:
                        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                        encoded_data = pd.DataFrame(
                            encoder.fit_transform(df[existing_cols])
                        )
                        encoded_data.columns = encoder.get_feature_names_out(existing_cols)
                        encoded_data.index = df.index
                        df = pd.concat([df, encoded_data], axis=1)
                        
                        # Impute missing Building Size values
                        if 'Building Size' in df.columns and 'Bedrooms' in df.columns:
                            df_size_known = df[df['Building Size'].notnull() & df['Bedrooms'].notnull()]
                            if not df_size_known.empty:
                                X_size = df_size_known[['Bedrooms'] + list(encoded_data.columns)]
                                y_size = df_size_known['Building Size']
                                
                                model_size = RandomForestRegressor()
                                model_size.fit(X_size, y_size)
                                
                                missing_size_mask = df['Building Size'].isnull() & df['Bedrooms'].notnull()
                                X_size_missing = df.loc[missing_size_mask, ['Bedrooms'] + list(encoded_data.columns)]
                                
                                if not X_size_missing.empty:
                                    predictions = model_size.predict(X_size_missing)
                                    df.loc[missing_size_mask, 'Building Size'] = predictions
                        
                        # Bedrooms and Bathrooms missing values
                        for col in ['Bedrooms', 'Bathrooms']:
                            if col in df.columns:
                                model_data = df[df[col].notnull() & df['Building Size'].notnull()]
                                if not model_data.empty:
                                    X = model_data[['Building Size'] + list(encoded_data.columns)]
                                    y = model_data[col]
                                    
                                    model = RandomForestRegressor()
                                    model.fit(X, y)
                                    
                                    missing_mask = df[col].isnull() & df['Building Size'].notnull()
                                    X_missing = df.loc[missing_mask, ['Building Size'] + list(encoded_data.columns)]
                                    
                                    if not X_missing.empty:
                                        predictions = model.predict(X_missing)
                                        df.loc[missing_mask, col] = predictions.round().astype('Int64')
                        
                        # Drop encoded columns
                        df = df.drop(columns=list(encoded_data.columns))
                    
                    return df
                
                # Apply imputation
                df = impute_missing_values(df)
                st.success("Missing values imputed successfully!")
                st.session_state.preprocessing_steps.append("Imputed missing values using Random Forest")
                st.session_state.processed_data = df
                
            except Exception as e:
                st.error(f"Error during imputation: {str(e)}")
        
        # 6. Save Processed Data
        st.header("6. Save Processed Data")
        
        if st.button("Save Processed Data"):
            try:
                # Ensure all numeric columns are properly formatted
                numeric_columns = ['Property Price', 'Building Size', 'Bedrooms', 'Bathrooms']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Ensure date columns are properly formatted
                date_columns = ['Published Date', 'Last Updated Date']
                for col in date_columns:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Save processed data to session state
                st.session_state.processed_data = df
                
                # --- Final Data Cleaning Before Saving ---
                # Ensure no missing values remain by applying fallback imputation
                st.info("✨ Applying final cleaning to ensure no missing values remain...")

                # Impute numeric columns with median
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

                # Impute categorical columns with mode
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                for col in categorical_cols:
                    # Use mode()[0] if mode is not empty, otherwise use 'Unknown'
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                         df[col] = df[col].fillna(mode_val[0])
                    else:
                         df[col] = df[col].fillna("Unknown")

                # Verify no missing values remain after final cleaning
                if df.isnull().sum().sum() == 0:
                    st.success("✅ All missing values successfully handled before saving.")
                else:
                    st.error("❌ Warning: Missing values still remain after final cleaning. Please check data.")
                    st.dataframe(df.isnull().sum().to_frame(name='Missing After Final Clean'))
                # -----------------------------------------

                # Save to CSV file for persistence
                output_path = Path('src/processed_data.csv')
                df.to_csv(output_path, index=False)
                st.success(f"✅ Processed data saved to {output_path}!")
                
                # Save to CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Processed Data",
                    data=csv,
                    file_name="processed_data.csv",
                    mime="text/csv"
                )
                
                st.success("Data processed and saved successfully!")
                
                # Show preprocessing steps
                st.subheader("Preprocessing Steps Applied")
                for step in st.session_state.preprocessing_steps:
                    st.write(f"- {step}")
                
            except Exception as e:
                st.error(f"Error saving data: {str(e)}")
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload an Excel file to begin data preprocessing.")

# About the app
st.sidebar.header("About the App")
st.sidebar.info(
    """
    This app is used to prepare and preprocess your real estate dataset.
    
    Steps:
    1. Upload an Excel file
    2. Select data cleaning options
    3. Apply data type conversions
    4. Perform feature transformations
    5. Handle missing values
    6. Download the processed data or proceed to Model Manager
    """
)

# Show current state
st.sidebar.header("Current State")
if st.session_state.raw_data is not None:
    st.sidebar.success("✅ Raw data loaded")
if st.session_state.processed_data is not None:
    st.sidebar.success("✅ Processed data available")
if st.session_state.data_validation_status['is_valid']:
    st.sidebar.success("✅ Data validation passed")
else:
    st.sidebar.warning("⚠️ Data validation issues found")
