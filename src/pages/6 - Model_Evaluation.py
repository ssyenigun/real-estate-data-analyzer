import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import os
import pickle
import json

# Page configuration
st.set_page_config(
    page_title="Real Estate Data Analysis - Model Evaluation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Model Evaluation")

# Check if there are any saved models
if 'saved_models' not in st.session_state or not st.session_state.saved_models:
    st.warning("⚠️ Please train and save a model first in the Model Manager page!")
    if st.button("Go to Model Manager Page"):
        st.switch_page("pages/model_manager.py")
    st.stop()

# Model selection
st.header("1. Select Model for Evaluation")
selected_model = st.selectbox(
    "Select a model to evaluate:",
    list(st.session_state.saved_models.keys())
)

if selected_model:
    model_data = st.session_state.saved_models[selected_model]
    
    # Load model from file
    try:
        model = joblib.load(model_data['model_path'])
        model_data['model'] = model  # Add model to model_data for visualization
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # Display model information
    st.header("2. Model Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        st.write(f"**Model Type:** {model_data.get('model_type', 'N/A')}")
        st.write(f"**Target Variable:** {model_data.get('target', 'N/A')}")
        st.write(f"**Number of Features:** {len(model_data.get('features', []))}")
        st.write(f"**Training Date:** {model_data.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M')}")
        
        # Display model parameters
        st.subheader("Model Parameters")
        st.json(model_data.get('parameters', {}))
    
    with col2:
        st.subheader("Feature List")
        features = model_data.get('features', [])
        st.write(", ".join(features))
        
        # Display accuracy metrics
        st.subheader("Model Accuracy Metrics")
        metrics = model_data.get('metrics', {})
        if metrics:
            # Create accuracy metrics cards
            col1, col2 = st.columns(2)
            with col1:
                # Calculate percentage accuracy
                test_r2 = metrics.get('test_r2', 0)
                accuracy_percentage = test_r2 * 100
                
                st.metric(
                    "Model Accuracy", 
                    f"{accuracy_percentage:.1f}%",
                    help="Overall model accuracy based on R² score"
                )
                
                # Calculate prediction accuracy within different thresholds
                y_test = model_data.get('y_test')
                y_pred_test = model_data.get('y_pred_test')
                if y_test is not None and y_pred_test is not None:
                    # Calculate percentage of predictions within 10% of actual value
                    within_10pct = np.mean(np.abs((y_test - y_pred_test) / y_test) <= 0.1) * 100
                    within_20pct = np.mean(np.abs((y_test - y_pred_test) / y_test) <= 0.2) * 100
                    
                    st.metric(
                        "Predictions within 10%", 
                        f"{within_10pct:.1f}%",
                        help="Percentage of predictions within 10% of actual value"
                    )
                    st.metric(
                        "Predictions within 20%", 
                        f"{within_20pct:.1f}%",
                        help="Percentage of predictions within 20% of actual value"
                    )
                
                st.metric(
                    "R² Score (Test)", 
                    f"{test_r2:.4f}",
                    help="R² score indicates how well the model fits the data. Higher is better."
                )
                st.metric(
                    "RMSE (Test)", 
                    f"${metrics.get('test_rmse', 0):,.2f}",
                    help="Root Mean Square Error. Lower is better."
                )
                st.metric(
                    "MAE (Test)", 
                    f"${metrics.get('test_mae', 0):,.2f}",
                    help="Mean Absolute Error. Lower is better."
                )
            with col2:
                st.metric(
                    "Training Time", 
                    f"{metrics.get('training_time', 0):.2f}s",
                    help="Time taken to train the model"
                )
                if 'stability_metrics' in metrics:
                    st.metric(
                        "Stability (RMSE std)", 
                        f"{metrics['stability_metrics'].get('test_rmse_std', 0):.4f}",
                        help="Standard deviation of RMSE across different random seeds. Lower is better."
                    )
                    st.metric(
                        "Stability (R² std)", 
                        f"{metrics['stability_metrics'].get('test_r2_std', 0):.4f}",
                        help="Standard deviation of R² across different random seeds. Lower is better."
                    )
            
            # Add accuracy interpretation
            st.markdown("### Accuracy Interpretation")
            
            # Overall accuracy interpretation
            if accuracy_percentage > 80:
                st.success(f"🎯 **Excellent Accuracy**: {accuracy_percentage:.1f}% - Model explains more than 80% of the variance")
            elif accuracy_percentage > 60:
                st.info(f"👍 **Good Accuracy**: {accuracy_percentage:.1f}% - Model explains more than 60% of the variance")
            elif accuracy_percentage > 40:
                st.warning(f"⚠️ **Moderate Accuracy**: {accuracy_percentage:.1f}% - Model explains more than 40% of the variance")
            else:
                st.error(f"❌ **Poor Accuracy**: {accuracy_percentage:.1f}% - Model explains less than 40% of the variance")
            
            # Prediction accuracy interpretation
            if y_test is not None and y_pred_test is not None:
                st.markdown("### Prediction Accuracy")
                if within_10pct > 70:
                    st.success(f"🎯 **High Precision**: {within_10pct:.1f}% of predictions are within 10% of actual values")
                elif within_10pct > 50:
                    st.info(f"👍 **Good Precision**: {within_10pct:.1f}% of predictions are within 10% of actual values")
                else:
                    st.warning(f"⚠️ **Low Precision**: Only {within_10pct:.1f}% of predictions are within 10% of actual values")
                
                if within_20pct > 90:
                    st.success(f"🎯 **High Coverage**: {within_20pct:.1f}% of predictions are within 20% of actual values")
                elif within_20pct > 70:
                    st.info(f"👍 **Good Coverage**: {within_20pct:.1f}% of predictions are within 20% of actual values")
                else:
                    st.warning(f"⚠️ **Low Coverage**: Only {within_20pct:.1f}% of predictions are within 20% of actual values")
            
            # Error interpretation
            st.markdown("### Error Analysis")
            test_rmse = metrics.get('test_rmse', 0)
            if test_rmse < 10000:  # Assuming price is in dollars
                st.success(f"🎯 **Low Error**: RMSE of ${test_rmse:,.2f} indicates good prediction accuracy")
            elif test_rmse < 50000:
                st.info(f"👍 **Moderate Error**: RMSE of ${test_rmse:,.2f} indicates acceptable prediction accuracy")
            else:
                st.warning(f"⚠️ **High Error**: RMSE of ${test_rmse:,.2f} indicates room for improvement")
            
            # Stability interpretation
            if 'stability_metrics' in metrics:
                st.markdown("### Model Stability")
                rmse_std = metrics['stability_metrics'].get('test_rmse_std', 0)
                r2_std = metrics['stability_metrics'].get('test_r2_std', 0)
                
                if rmse_std < 1000 and r2_std < 0.05:
                    st.success("🎯 **High Stability**: Model shows consistent performance across different random seeds")
                elif rmse_std < 5000 and r2_std < 0.1:
                    st.info("👍 **Good Stability**: Model shows reasonable consistency across different random seeds")
                else:
                    st.warning("⚠️ **Low Stability**: Model performance varies significantly across different random seeds")
    
    # Detailed Performance Analysis
    st.header("3. Detailed Performance Analysis")
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Metrics", "📈 Model Visualizations", "🔍 Feature Analysis", "⚖️ Model Stability"])
    
    with tab1:
        st.subheader("Detailed Metrics")
        if metrics:
            # Create a DataFrame for better visualization
            metrics_df = pd.DataFrame({
                'Metric': ['R² Score', 'RMSE', 'MAE', 'Training Time'],
                'Train': [
                    f"{metrics.get('train_r2', 0):.4f}",
                    f"{metrics.get('train_rmse', 0):.2f}",
                    f"{metrics.get('train_mae', 0):.2f}",
                    f"{metrics.get('training_time', 0):.2f}s"
                ],
                'Test': [
                    f"{metrics.get('test_r2', 0):.4f}",
                    f"{metrics.get('test_rmse', 0):.2f}",
                    f"{metrics.get('test_mae', 0):.2f}",
                    "N/A"
                ]
            })
            st.dataframe(metrics_df, use_container_width=True)
            
            # Display stability metrics if available
            if 'stability_metrics' in metrics:
                st.subheader("Model Stability Metrics")
                stability_df = pd.DataFrame({
                    'Metric': ['RMSE', 'MAE', 'R²', 'Training Time'],
                    'Mean': [
                        f"{metrics['stability_metrics'].get('test_rmse_mean', 0):.4f}",
                        f"{metrics['stability_metrics'].get('test_mae_mean', 0):.4f}",
                        f"{metrics['stability_metrics'].get('test_r2_mean', 0):.4f}",
                        f"{metrics['stability_metrics'].get('training_time_mean', 0):.2f}s"
                    ],
                    'Std Dev': [
                        f"{metrics['stability_metrics'].get('test_rmse_std', 0):.4f}",
                        f"{metrics['stability_metrics'].get('test_mae_std', 0):.4f}",
                        f"{metrics['stability_metrics'].get('test_r2_std', 0):.4f}",
                        f"{metrics['stability_metrics'].get('training_time_std', 0):.2f}s"
                    ]
                })
                st.dataframe(stability_df, use_container_width=True)
    
    with tab2:
        st.subheader("Model Visualizations")
        
        # Create visualizations if we have the necessary data
        try:
            y_train = model_data.get('y_train')
            y_test = model_data.get('y_test')
            y_pred_train = model_data.get('y_pred_train')
            y_pred_test = model_data.get('y_pred_test')
            model = model_data.get('model')
            feature_names = model_data.get('features', [])
            
            if all(x is not None for x in [y_train, y_test, y_pred_train, y_pred_test]):
                # Actual vs Predicted Plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Training set
                ax1.scatter(y_train, y_pred_train, alpha=0.5)
                ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
                ax1.set_xlabel('Actual Values')
                ax1.set_ylabel('Predicted Values')
                ax1.set_title('Training Set: Actual vs Predicted')
                
                # Test set
                ax2.scatter(y_test, y_pred_test, alpha=0.5)
                ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax2.set_xlabel('Actual Values')
                ax2.set_ylabel('Predicted Values')
                ax2.set_title('Test Set: Actual vs Predicted')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                # Residuals Plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Training residuals
                train_residuals = y_train - y_pred_train
                ax1.scatter(y_pred_train, train_residuals, alpha=0.5)
                ax1.axhline(y=0, color='r', linestyle='--')
                ax1.set_xlabel('Predicted Values')
                ax1.set_ylabel('Residuals')
                ax1.set_title('Training Set: Residuals vs Predicted')
                
                # Test residuals
                test_residuals = y_test - y_pred_test
                ax2.scatter(y_pred_test, test_residuals, alpha=0.5)
                ax2.axhline(y=0, color='r', linestyle='--')
                ax2.set_xlabel('Predicted Values')
                ax2.set_ylabel('Residuals')
                ax2.set_title('Test Set: Residuals vs Predicted')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                # Feature Importance Plot (if available)
                if hasattr(model, 'feature_importances_'):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    importances = pd.Series(model.feature_importances_, index=feature_names)
                    importances.sort_values().plot(kind='barh', ax=ax)
                    ax.set_title('Feature Importance')
                    ax.set_xlabel('Importance Score')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
        
        except Exception as e:
            st.error(f"Error creating visualizations: {str(e)}")
    
    with tab3:
        st.subheader("Feature Analysis")
        
        # Create columns for different feature analysis sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Feature Importance")
            
            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                # Create feature importance DataFrame
                feature_importance = pd.DataFrame({
                    'Feature': model_data.get('features', []),
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Plot feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=feature_importance, x='Importance', y='Feature', ax=ax)
                ax.set_title('Feature Importance')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                # Display feature importance table
                st.dataframe(feature_importance, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")
            
            # Feature correlation analysis
            st.markdown("### Feature Correlations")
            try:
                # Get feature data
                X = pd.DataFrame(model_data.get('X_train', []), columns=model_data.get('features', []))
                if not X.empty:
                    # Calculate correlation matrix
                    corr_matrix = X.corr()
                    
                    # Plot correlation heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                    ax.set_title('Feature Correlation Matrix')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Display high correlation pairs
                    high_corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            if abs(corr_matrix.iloc[i, j]) > 0.7:  # Threshold for high correlation
                                high_corr_pairs.append({
                                    'Feature 1': corr_matrix.columns[i],
                                    'Feature 2': corr_matrix.columns[j],
                                    'Correlation': corr_matrix.iloc[i, j]
                                })
                    
                    if high_corr_pairs:
                        st.markdown("#### Highly Correlated Features")
                        high_corr_df = pd.DataFrame(high_corr_pairs)
                        st.dataframe(high_corr_df, use_container_width=True)
                    else:
                        st.info("No highly correlated features found.")
            except Exception as e:
                st.warning(f"Could not generate correlation analysis: {str(e)}")
        
        with col2:
            st.markdown("### Feature Selection Results")
            
            if 'feature_selection_results' in model_data:
                feature_selection = model_data['feature_selection_results']
                
                # Recursive Feature Elimination results
                st.markdown("#### Recursive Feature Elimination (RFE)")
                if 'rfe_features' in feature_selection:
                    st.write("Selected Features:")
                    st.write(feature_selection['rfe_features'])
                else:
                    st.info("RFE results not available.")
                
                # Statistical test results
                st.markdown("#### Statistical Tests")
                if 'f_regression_features' in feature_selection:
                    st.write("F-Regression Selected Features:")
                    for k, features in feature_selection['f_regression_features'].items():
                        st.write(f"Top {k} features:")
                        st.write(features)
                
                if 'mutual_info_features' in feature_selection:
                    st.write("Mutual Information Selected Features:")
                    for k, features in feature_selection['mutual_info_features'].items():
                        st.write(f"Top {k} features:")
                        st.write(features)
                
                # Permutation importance results
                st.markdown("#### Permutation Importance")
                if 'permutation_importance' in feature_selection:
                    perm_importance = feature_selection['permutation_importance']
                    
                    # Plot permutation importance with error bars
                    fig, ax = plt.subplots(figsize=(10, 6))
                    perm_importance.plot(kind='barh', x='feature', y='importance_mean', 
                                       yerr='importance_std', ax=ax)
                    ax.set_title('Feature Importance with Stability')
                    ax.set_xlabel('Importance Score')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Display permutation importance table
                    st.dataframe(perm_importance, use_container_width=True)
            else:
                st.info("Feature selection results not available.")
            
            # Feature statistics
            st.markdown("### Feature Statistics")
            try:
                X = pd.DataFrame(model_data.get('X_train', []), columns=model_data.get('features', []))
                if not X.empty:
                    # Calculate basic statistics
                    stats_df = X.describe().T
                    stats_df['missing_values'] = X.isnull().sum()
                    stats_df['missing_percentage'] = (X.isnull().sum() / len(X)) * 100
                    
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Plot feature distributions
                    st.markdown("#### Feature Distributions")
                    for feature in X.columns:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.histplot(data=X, x=feature, ax=ax)
                        ax.set_title(f'Distribution of {feature}')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
            except Exception as e:
                st.warning(f"Could not generate feature statistics: {str(e)}")
    
    with tab4:
        st.subheader("Model Stability Analysis")
        
        if 'stability_metrics' in metrics:
            # Plot stability metrics
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # RMSE stability
            rmse_mean = metrics['stability_metrics']['test_rmse_mean']
            rmse_std = metrics['stability_metrics']['test_rmse_std']
            ax1.bar(['RMSE'], [rmse_mean], yerr=[rmse_std], capsize=10)
            ax1.set_title('RMSE Stability')
            ax1.set_ylabel('RMSE Score')
            
            # R² stability
            r2_mean = metrics['stability_metrics']['test_r2_mean']
            r2_std = metrics['stability_metrics']['test_r2_std']
            ax2.bar(['R²'], [r2_mean], yerr=[r2_std], capsize=10)
            ax2.set_title('R² Stability')
            ax2.set_ylabel('R² Score')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Training time stability
            fig, ax = plt.subplots(figsize=(8, 6))
            time_mean = metrics['stability_metrics']['training_time_mean']
            time_std = metrics['stability_metrics']['training_time_std']
            ax.bar(['Training Time'], [time_mean], yerr=[time_std], capsize=10)
            ax.set_title('Training Time Stability')
            ax.set_ylabel('Time (seconds)')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Stability metrics not available for this model.")
    
    # Export Options
    st.header("4. Export Options")
    
    export_format = st.selectbox(
        "Select Export Format",
        ["Joblib", "Pickle", "JSON"],
        help="Choose the format to export the model"
    )
    
    if st.button("Export Model"):
        try:
            # Create export directory if it doesn't exist
            export_dir = "src/exports"
            os.makedirs(export_dir, exist_ok=True)
            
            # Generate export filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = selected_model.replace(" ", "_").lower()
            export_filename = f"{model_name}_{timestamp}"
            
            if export_format == "Joblib":
                export_path = os.path.join(export_dir, f"{export_filename}.joblib")
                # Save model and metadata
                export_data = {
                    'model': model,
                    'model_type': model_data.get('model_type'),
                    'features': model_data.get('features'),
                    'parameters': model_data.get('parameters'),
                    'metrics': model_data.get('metrics'),
                    'stability_metrics': model_data.get('metrics', {}).get('stability_metrics'),
                    'feature_selection_results': model_data.get('feature_selection_results'),
                    'timestamp': datetime.now()
                }
                joblib.dump(export_data, export_path)
                
            elif export_format == "Pickle":
                export_path = os.path.join(export_dir, f"{export_filename}.pkl")
                with open(export_path, 'wb') as f:
                    pickle.dump({
                        'model': model,
                        'model_type': model_data.get('model_type'),
                        'features': model_data.get('features'),
                        'parameters': model_data.get('parameters'),
                        'metrics': model_data.get('metrics'),
                        'stability_metrics': model_data.get('metrics', {}).get('stability_metrics'),
                        'feature_selection_results': model_data.get('feature_selection_results'),
                        'timestamp': datetime.now()
                    }, f)
                
            elif export_format == "JSON":
                export_path = os.path.join(export_dir, f"{export_filename}.json")
                # Convert model data to JSON-serializable format
                export_data = {
                    'model_type': model_data.get('model_type'),
                    'features': model_data.get('features'),
                    'parameters': model_data.get('parameters'),
                    'metrics': model_data.get('metrics'),
                    'stability_metrics': model_data.get('metrics', {}).get('stability_metrics'),
                    'feature_selection_results': model_data.get('feature_selection_results'),
                    'timestamp': datetime.now().isoformat()
                }
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=4)
            
            st.success(f"✅ Model exported successfully to: {export_path}")
            
            # Provide download button
            with open(export_path, 'rb') as f:
                st.download_button(
                    label="Download Exported Model",
                    data=f,
                    file_name=os.path.basename(export_path),
                    mime="application/octet-stream"
                )
                
        except Exception as e:
            st.error(f"❌ Error exporting model: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Model Evaluation Module - Analyze and export your model's performance.*") 