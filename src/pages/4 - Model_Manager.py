import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os
from datetime import datetime
from streamlit_echarts import st_echarts
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.advanced_models import AdvancedModelTrainer

# Page configuration
st.set_page_config(
    page_title="Real Estate Data Analysis - Model Manager",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Model Manager")

# Initialize session state for saved models if not exists
if 'saved_models' not in st.session_state:
    st.session_state.saved_models = {}

# Load preprocessed data
try:
    data = pd.read_csv('src/processed_data.csv')
    
    target = 'Property Price'
    
    # Get all columns except the target as potential features
    all_features = [col for col in data.columns if col != target]
    
    # Add feature selection to the sidebar
    selected_features = st.sidebar.multiselect(
        "Select Features for Training",
        all_features,
        default=['Bedrooms', 'Bathrooms', 'Building Size'] if all(item in all_features for item in ['Bedrooms', 'Bathrooms', 'Building Size']) else all_features[:3],
        key="feature_selector"
    )
    
    if not selected_features:
        st.error("Please select at least one feature for training!")
        st.stop()
    
    # Create a copy of the data to avoid SettingWithCopyWarning
    data_copy = data.copy()
    
    # Prepare data using selected features
    X = data_copy[selected_features].copy()
    y = data_copy[target].copy()
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        X.loc[:, col] = label_encoders[col].fit_transform(X[col].astype(str))
    
    # Handle missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_value = X[col].median()
        X.loc[:, col] = X[col].fillna(median_value)
    
    # Get training configuration from sidebar
    st.sidebar.markdown("### Training Configuration")
    test_size = st.sidebar.slider(
        "Test Size",
        min_value=0.10,
        max_value=0.50,
        value=0.20,
        step=0.05,
        help="Proportion of the dataset to include in the test split",
        key="test_size_slider"
    )
    random_state = st.sidebar.number_input(
        "Random State",
        min_value=0,
        value=42,
        step=1,
        help="Seed for random number generator",
        key="random_state_input"
    )
    
    # Model selection
    st.sidebar.markdown("### Model Selection")
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["Linear Regression", "Decision Tree", "Random Forest", "SVR", "XGBoost"],
        help="Choose the type of model to train",
        key="model_selector"
    )
    
    # Model-specific hyperparameters
    st.sidebar.markdown("### Model Hyperparameters")
    if model_type == "Linear Regression":
        # Linear Regression has no hyperparameters to tune
        pass
    elif model_type == "Decision Tree":
        max_depth = st.sidebar.slider("Max Depth", 1, 20, 5, key="dt_max_depth")
        min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2, key="dt_min_samples_split")
        min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, 1, key="dt_min_samples_leaf")
    elif model_type == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 100, key="rf_n_estimators")
        max_depth = st.sidebar.slider("Max Depth", 1, 30, 10, key="rf_max_depth")
        min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2, key="rf_min_samples_split")
    elif model_type == "SVR":
        C = st.sidebar.slider("C", 0.1, 10.0, 1.0, key="svr_c")
        kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear"], key="svr_kernel")
        gamma = st.sidebar.selectbox("Gamma", ["scale", "auto", 0.1, 0.01], key="svr_gamma")
    elif model_type == "XGBoost":
        n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 100, key="xgb_n_estimators")
        max_depth = st.sidebar.slider("Max Depth", 1, 10, 3, key="xgb_max_depth")
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1, key="xgb_learning_rate")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Train New Model", "📈 Model Performance", "💾 Saved Models", "🔍 Advanced Analysis"])
    
    with tab1:
        st.header("Train New Model")
        
        # Model Naming input
        custom_model_name = st.text_input("Enter a name for the model (optional)", key="model_name_input")
        
        if st.button("Train Model", key="train_model_button"):
            with st.spinner("Training model..."):
                # Initialize and train the selected model
                if model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "Decision Tree":
                    model = DecisionTreeRegressor(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state
                    )
                elif model_type == "Random Forest":
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=random_state
                    )
                elif model_type == "SVR":
                    model = SVR(C=C, kernel=kernel, gamma=gamma)
                elif model_type == "XGBoost":
                    model = XGBRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        random_state=random_state
                    )
                
                # Train the model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                # Calculate metrics
                metrics = {
                    'train_r2': r2_score(y_train, y_pred_train),
                    'test_r2': r2_score(y_test, y_pred_test),
                    'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    'train_mae': mean_absolute_error(y_train, y_pred_train),
                    'test_mae': mean_absolute_error(y_test, y_pred_test),
                    'train_mape': np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100 if np.all(y_train != 0) else 0,
                    'test_mape': np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100 if np.all(y_test != 0) else 0,
                    'train_within_10pct': np.mean(np.abs((y_train - y_pred_train) / y_train) <= 0.1) * 100 if np.all(y_train != 0) else 0,
                    'test_within_10pct': np.mean(np.abs((y_test - y_pred_test) / y_test) <= 0.1) * 100 if np.all(y_test != 0) else 0,
                    'train_accuracy': (1 - np.mean(np.abs((y_train - y_pred_train) / y_train))) * 100 if np.all(y_train != 0) else 0,
                    'test_accuracy': (1 - np.mean(np.abs((y_test - y_pred_test) / y_test))) * 100 if np.all(y_test != 0) else 0
                }
                
                # Save model and metrics
                if custom_model_name:
                    model_name = f"{custom_model_name}"
                else:
                    model_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                model_path = f"src/saved_models/{model_name}.joblib"
                
                # Create models directory if it doesn't exist
                os.makedirs("src/saved_models", exist_ok=True)
                
                # Save model
                joblib.dump(model, model_path)
                
                # Save model data
                model_data = {
                    'model_type': model_type,
                    'model_path': model_path,
                    'features': selected_features,
                    'parameters': model.get_params(),
                    'metrics': metrics,
                    'timestamp': datetime.now(),
                    'y_train': y_train,
                    'y_test': y_test,
                    'y_pred_train': y_pred_train,
                    'y_pred_test': y_pred_test,
                    'target': target
                }
                
                st.session_state.saved_models[model_name] = model_data
                
                st.success(f"✅ Model trained and saved successfully as '{model_name}'!")
            
            # Display model information and training results
            st.markdown("### Model Information")
            if model_type == "Linear Regression":
                st.info("Linear Regression is a simple model that assumes a linear relationship between features and target.")
            elif model_type == "Decision Tree":
                st.info("Decision Tree builds a tree-like model of decisions and their possible consequences.")
            elif model_type == "Random Forest":
                st.info("Random Forest is an ensemble model that combines multiple decision trees for better predictions.")
            elif model_type == "SVR":
                st.info("Support Vector Regression finds a hyperplane that best fits the data.")
            elif model_type == "XGBoost":
                st.info("XGBoost is a powerful gradient boosting model that often provides state-of-the-art results.")
            
            # Display metrics in a nice layout
            st.markdown("### Training Results")
            
            # Create metric cards
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Used Features", len(selected_features))
            with col2:
                st.metric("Test R²", f"{metrics['test_r2']:.4f}")
            with col3:
                st.metric("Test RMSE", f"${metrics['test_rmse']:,.2f}")
            with col4:
                st.metric("Test MAE", f"${metrics['test_mae']:,.2f}")
            with col5:
                st.metric("Test Accuracy", f"{metrics['test_accuracy']:.2f}%")
            
            # Create performance visualization using ECharts
            scatter_data = [
                [float(x), float(y)] for x, y in zip(y_test, y_pred_test)
            ]
            
            # Perfect prediction line
            min_val = min(y_test.min(), y_pred_test.min())
            max_val = max(y_test.max(), y_pred_test.max())
            perfect_line = [[float(min_val), float(min_val)], [float(max_val), float(max_val)]]
            
            options = {
                "title": {
                    "text": "Actual vs Predicted Values",
                    "left": "center"
                },
                "tooltip": {
                    "trigger": "item",
                    "formatter": "Actual: ${c[0]:,.2f}<br/>Predicted: ${c[1]:,.2f}"
                },
                "xAxis": {
                    "type": "value",
                    "name": "Actual Values",
                    "nameLocation": "middle",
                    "nameGap": 30
                },
                "yAxis": {
                    "type": "value",
                    "name": "Predicted Values",
                    "nameLocation": "middle",
                    "nameGap": 30
                },
                "series": [
                    {
                        "type": "scatter",
                        "data": scatter_data,
                        "symbolSize": 8,
                        "itemStyle": {
                            "color": "#1f77b4"
                        }
                    },
                    {
                        "type": "line",
                        "data": perfect_line,
                        "lineStyle": {
                            "color": "#ff0000",
                            "type": "dashed"
                        },
                        "name": "Perfect Prediction"
                    }
                ],
                "legend": {
                    "data": ["Predictions", "Perfect Prediction"],
                    "top": 30
                }
            }
            
            st_echarts(options=options, height="400px")
            
            # Show feature importance if available
            if model_type in ['Random Forest', 'XGBoost']:
                st.markdown("### Feature Importance")
                
                if hasattr(model, 'feature_importances_'):
                    importance = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    # Create feature importance chart using ECharts
                    options = {
                        "title": {
                            "text": "Feature Importance",
                            "left": "center"
                        },
                        "tooltip": {
                            "trigger": "axis",
                            "axisPointer": {
                                "type": "shadow"
                            }
                        },
                        "grid": {
                            "left": "3%",
                            "right": "4%",
                            "bottom": "3%",
                            "containLabel": True
                        },
                        "xAxis": {
                            "type": "value",
                            "name": "Importance"
                        },
                        "yAxis": {
                            "type": "category",
                            "data": importance['Feature'].tolist(),
                            "name": "Feature"
                        },
                        "series": [
                            {
                                "name": "Importance",
                                "type": "bar",
                                "data": importance['Importance'].tolist()
                            }
                        ]
                    }
                    
                    st_echarts(options=options, height="400px")
            
            # Show coefficients for Linear Regression
            elif model_type == "Linear Regression":
                st.markdown("### Feature Coefficients")
                
                coefficients = pd.DataFrame({
                    'Feature': selected_features,
                    'Coefficient': model.coef_
                }).sort_values('Coefficient', ascending=False)
                
                # Create coefficients chart using ECharts
                options = {
                    "title": {
                        "text": "Feature Coefficients",
                        "left": "center"
                    },
                    "tooltip": {
                        "trigger": "axis",
                        "axisPointer": {
                            "type": "shadow"
                        }
                    },
                    "grid": {
                        "left": "3%",
                        "right": "4%",
                        "bottom": "3%",
                        "containLabel": True
                    },
                    "xAxis": {
                        "type": "value",
                        "name": "Coefficient"
                    },
                    "yAxis": {
                        "type": "category",
                        "data": coefficients['Feature'].tolist(),
                        "name": "Feature"
                    },
                    "series": [
                        {
                            "name": "Coefficient",
                            "type": "bar",
                            "data": coefficients['Coefficient'].tolist()
                        }
                    ]
                }
                
                st_echarts(options=options, height="400px")
    
    with tab2:
        st.markdown("### Model Performance Analysis")
        
        if not st.session_state.saved_models:
            st.info("No models have been trained yet. Train a model to see performance analysis.")
        else:
            # Model selection for comparison
            selected_models = st.multiselect(
                "Select models to compare",
                list(st.session_state.saved_models.keys()),
                default=list(st.session_state.saved_models.keys())[:2] if len(st.session_state.saved_models) > 1 else [list(st.session_state.saved_models.keys())[0]]
            )
            
            if selected_models:
                # Create comparison metrics
                metrics_df = pd.DataFrame({
                    'Model': selected_models,
                    'R² Score': [st.session_state.saved_models[m]['metrics'].get('test_r2', 0) for m in selected_models],
                    'RMSE': [st.session_state.saved_models[m]['metrics'].get('test_rmse', 0) for m in selected_models],
                    'MAE': [st.session_state.saved_models[m]['metrics'].get('test_mae', 0) for m in selected_models],
                    'MAPE': [st.session_state.saved_models[m]['metrics'].get('test_mape', 0) for m in selected_models],
                    'Accuracy': [st.session_state.saved_models[m]['metrics'].get('test_accuracy', 0) for m in selected_models]
                })
                
                # Display metrics comparison
                st.markdown("#### Performance Metrics Comparison")
                st.dataframe(metrics_df.style.highlight_max(axis=0), use_container_width=True, key="metrics_comparison_df")
                
                # Create comparison chart using ECharts
                options = {
                    "title": {
                        "text": "Model Performance Comparison",
                        "left": "center"
                    },
                    "tooltip": {
                        "trigger": "axis",
                        "axisPointer": {
                            "type": "shadow"
                        }
                    },
                    "legend": {
                        "data": ["R² Score", "RMSE", "MAE", "MAPE", "Accuracy"],
                        "top": 30
                    },
                    "grid": {
                        "left": "3%",
                        "right": "4%",
                        "bottom": "3%",
                        "containLabel": True
                    },
                    "xAxis": {
                        "type": "category",
                        "data": selected_models
                    },
                    "yAxis": {
                        "type": "value"
                    },
                    "series": [
                        {
                            "name": "R² Score",
                            "type": "bar",
                            "data": metrics_df['R² Score'].tolist()
                        },
                        {
                            "name": "RMSE",
                            "type": "bar",
                            "data": metrics_df['RMSE'].tolist()
                        },
                        {
                            "name": "MAE",
                            "type": "bar",
                            "data": metrics_df['MAE'].tolist()
                        },
                        {
                            "name": "MAPE",
                            "type": "bar",
                            "data": metrics_df['MAPE'].tolist()
                        },
                        {
                            "name": "Accuracy",
                            "type": "bar",
                            "data": metrics_df['Accuracy'].tolist()
                        }
                    ]
                }
                
                st_echarts(options=options, height="400px")
    
    with tab3:
        st.markdown("### Saved Models")
        
        if not st.session_state.saved_models:
            st.info("No models have been saved yet. Train a model to see it here!")
        else:
            for model_name, model_data in st.session_state.saved_models.items():
                with st.expander(f"📊 {model_name}", expanded=True):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("#### Model Information")
                        st.write(f"**Type:** {model_data['model_type']}")
                        st.write(f"**Training Date:** {model_data['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"**Number of Features:** {len(model_data['features'])}")
                        
                        st.markdown("#### Performance Metrics")
                        metrics = model_data['metrics']
                        st.metric("Test R²", f"{metrics.get('test_r2', 0):.4f}")
                        st.metric("Test RMSE", f"${metrics.get('test_rmse', 0):,.2f}")
                        st.metric("Test MAE", f"${metrics.get('test_mae', 0):,.2f}")
                        st.metric("Test Accuracy (%10)", f"{metrics.get('test_within_10pct', 0):.2f}%")
                    
                    with col2:
                        st.markdown("#### Feature Importance")
                        if model_data['model_type'] in ['Random Forest', 'XGBoost']:
                            model = joblib.load(model_data['model_path'])
                            if hasattr(model, 'feature_importances_'):
                                importance = pd.DataFrame({
                                    'Feature': model_data['features'],
                                    'Importance': model.feature_importances_
                                }).sort_values('Importance', ascending=False)
                                
                                # Create feature importance chart using ECharts
                                options = {
                                    "title": {
                                        "text": "Top 10 Most Important Features",
                                        "left": "center"
                                    },
                                    "tooltip": {
                                        "trigger": "axis",
                                        "axisPointer": {
                                            "type": "shadow"
                                        }
                                    },
                                    "grid": {
                                        "left": "3%",
                                        "right": "4%",
                                        "bottom": "3%",
                                        "containLabel": True
                                    },
                                    "xAxis": {
                                        "type": "value",
                                        "name": "Importance"
                                    },
                                    "yAxis": {
                                        "type": "category",
                                        "data": importance['Feature'].head(10).tolist(),
                                        "name": "Feature"
                                    },
                                    "series": [
                                        {
                                            "name": "Importance",
                                            "type": "bar",
                                            "data": importance['Importance'].head(10).tolist()
                                        }
                                    ]
                                }
                                
                                st_echarts(options=options, height="400px")
                        else:
                            st.info("Feature importance is not available for this model type.")
                    
                    if st.button("View Detailed Analysis", key=f"view_{model_name}"):
                        st.switch_page("pages/model_evaluation.py")

    with tab4:
        st.header("Advanced Model Analysis")
        
        # Initialize AdvancedModelTrainer
        if 'advanced_trainer' not in st.session_state:
            st.session_state.advanced_trainer = AdvancedModelTrainer()
        trainer = st.session_state.advanced_trainer
        st.info(f"Trainer initialized: {trainer}") # Debug print

        # Check if data is already loaded or load it now
        if 'data_loaded_for_advanced_analysis' not in st.session_state or not st.session_state.data_loaded_for_advanced_analysis:
            st.info("Attempting to load and preprocess data...") # Debug print
            if trainer.load_and_preprocess_data():
                st.session_state.data_loaded_for_advanced_analysis = True
                st.success("Data loaded and preprocessed for advanced analysis.") # Debug print
            else:
                st.error("Failed to load and preprocess data for advanced analysis. Please check your data file.")
                st.exception(e) # Display full traceback here as well
                st.stop()
        else:
            st.info("Data already loaded for advanced analysis.") # Debug print

        if st.session_state.data_loaded_for_advanced_analysis:
            if st.button("Run Advanced Analysis", key="run_advanced_analysis"):
                with st.spinner("Performing advanced analysis..."):
                    # Perform feature selection
                    feature_selection_results = trainer.perform_feature_selection()
                    st.session_state.feature_selection_results = feature_selection_results # Store feature selection results
                    st.success("Feature selection completed and results stored.") # Debug print

            # Display feature selection results if available
            if 'feature_selection_results' in st.session_state:
                st.info("Displaying feature selection results...") # Debug print
                feature_selection_results = st.session_state.feature_selection_results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Feature Selection Results")

                    # Display correlation-based features
                    st.markdown("#### Correlation Analysis")
                    high_corr_df = pd.DataFrame(feature_selection_results['high_correlation'], 
                                              columns=['Feature 1', 'Feature 2', 'Correlation'])
                    st.dataframe(high_corr_df, key="correlation_df")

                    # Display F-regression features
                    st.markdown("#### F-Regression Features")
                    st.write(feature_selection_results['f_regression_features'], key="f_regression_features")

                    # Display Mutual Information features
                    st.markdown("#### Mutual Information Features")
                    st.write(feature_selection_results['mutual_info_features'], key="mutual_info_features")

                with col2:
                    st.subheader("Feature Importance")

                    # Display permutation importance
                    perm_importance = feature_selection_results['permutation_importance']
                    st.dataframe(perm_importance, key="perm_importance_df")

                    # Display RFE features
                    st.markdown("#### Recursive Feature Elimination")
                    st.write(feature_selection_results['rfe_features'], key="rfe_features")

                # Hyperparameter Tuning Section
                st.subheader("Hyperparameter Tuning")

                # Create two columns for model tuning
                col3, col4 = st.columns(2)

                with col3:
                    st.markdown("#### Random Forest Tuning")
                    if st.button("Tune Random Forest", key="tune_rf"):
                        with st.spinner("Tuning Random Forest..."):
                            st.info("Starting Random Forest tuning...") # Debug print
                            rf_model, rf_metrics = trainer.tune_hyperparameters('Random Forest')
                            st.session_state.rf_metrics = rf_metrics # Store in session state
                            st.success("Random Forest tuning completed and results stored.") # Debug print
                            st.rerun() # Trigger a rerun to display results

                    if 'rf_metrics' in st.session_state:
                        st.info("Displaying Random Forest tuning results...") # Debug print
                        rf_metrics = st.session_state.rf_metrics
                        # Display best parameters
                        st.markdown("##### Best Parameters")
                        st.json(rf_metrics['best_params'])

                        # Display performance metrics
                        st.markdown("##### Performance Metrics")
                        metrics_df = pd.DataFrame({
                            'Metric': ['Test RMSE', 'Test R²', 'Test MAE'],
                            'Value': [
                                f"${rf_metrics['test_rmse']:,.2f}",
                                f"{rf_metrics['test_r2']:.4f}",
                                f"${rf_metrics['test_mae']:,.2f}"
                            ]
                        })
                        st.dataframe(metrics_df, key="rf_metrics_df")

                with col4:
                    st.markdown("#### SVR Tuning")
                    if st.button("Tune SVR", key="tune_svr"):
                        with st.spinner("Tuning SVR..."):
                            st.info("Starting SVR tuning...") # Debug print
                            svr_model, svr_metrics = trainer.tune_hyperparameters('SVR')
                            st.session_state.svr_metrics = svr_metrics # Store in session state
                            st.success("SVR tuning completed and results stored.") # Debug print
                            st.rerun() # Trigger a rerun to display results

                    if 'svr_metrics' in st.session_state:
                        st.info("Displaying SVR tuning results...") # Debug print
                        svr_metrics = st.session_state.svr_metrics
                        # Display best parameters
                        st.markdown("##### Best Parameters")
                        st.json(svr_metrics['best_params'])

                        # Display performance metrics
                        st.markdown("##### Performance Metrics")
                        metrics_df = pd.DataFrame({
                            'Metric': ['Test RMSE', 'Test R²', 'Test MAE'],
                            'Value': [
                                f"${svr_metrics['test_rmse']:,.2f}",
                                f"{svr_metrics['test_r2']:.4f}",
                                f"${svr_metrics['test_mae']:,.2f}"
                            ]
                        })
                        st.dataframe(metrics_df, key="svr_metrics_df")

                # Model Comparison
                st.subheader("Model Comparison")
                if 'rf_metrics' in st.session_state and 'svr_metrics' in st.session_state: # Check session state
                    st.info("Displaying model comparison...") # Debug print
                    rf_metrics = st.session_state.rf_metrics
                    svr_metrics = st.session_state.svr_metrics
                    comparison_df = pd.DataFrame({
                        'Model': ['Random Forest', 'SVR'],
                        'Test RMSE': [rf_metrics['test_rmse'], svr_metrics['test_rmse']],
                        'Test R²': [rf_metrics['test_r2'], svr_metrics['test_r2']],
                        'Test MAE': [rf_metrics['test_mae'], svr_metrics['test_mae']]
                    })
                    st.dataframe(comparison_df, key="comparison_df")
        # The `else` block for data loading failure is handled by the `st.exception()` above

except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.exception(e) # Display full traceback here as well
    st.stop()

# Footer
st.markdown("---")
st.markdown("*Model Manager - Real Estate Data Analysis*")
