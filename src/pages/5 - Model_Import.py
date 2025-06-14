import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

st.set_page_config(page_title="Model Import", page_icon="📥", layout="wide")

st.title("📥 Model Import & Management")
st.markdown("Import previously exported models and manage your model library.")

# Initialize session state
if 'saved_models' not in st.session_state:
  st.session_state.saved_models = {}

# Import Model Section
st.header("📂 Import Model")

uploaded_model = st.file_uploader(
  "Choose a model file (.pkl)",
  type="pkl",
  help="Upload a previously exported model file"
)

if uploaded_model is not None:
  try:
      # Load the model
      model_data = pickle.load(uploaded_model)
      
      # Display model information
      st.subheader("📊 Model Information")
      
      if 'model_metadata' in model_data:
          metadata = model_data['model_metadata']
          
          col1, col2 = st.columns(2)
          
          with col1:
              st.write(f"**Name:** {metadata.get('name', 'Unknown')}")
              st.write(f"**Type:** {metadata.get('type', 'Unknown')}")
              st.write(f"**Created:** {metadata.get('created_at', 'Unknown')}")
          
          with col2:
              if 'performance' in metadata:
                  perf = metadata['performance']
                  st.write(f"**R² Score:** {perf.get('R²', 'N/A')}")
                  st.write(f"**RMSE:** ${perf.get('RMSE', 0):,.0f}")
                  st.write(f"**MAE:** ${perf.get('MAE', 0):,.0f}")
          
          # Model parameters
          if 'parameters' in metadata:
              st.subheader("⚙️ Model Parameters")
              st.json(metadata['parameters'])
          
          # Import button
          import_name = st.text_input(
              "Model Name (for import)", 
              value=metadata.get('name', f"Imported_Model_{datetime.now().strftime('%Y%m%d_%H%M')}")
          )
          
          if st.button("📥 Import Model", type="primary"):
              # Reconstruct model data structure
              imported_model = {
                  'model': model_data['model'],
                  'scaler': model_data.get('scaler'),
                  'feature_columns': model_data['feature_columns'],
                  'model_type': metadata.get('type', 'Unknown'),
                  'model_params': metadata.get('parameters', {}),
                  'training_config': metadata.get('training_config', {}),
                  'performance': metadata.get('performance', {}),
                  'created_at': metadata.get('created_at', datetime.now().isoformat()),
                  'imported_at': datetime.now().isoformat(),
                  'data_info': {
                      'features_used': model_data['feature_columns'],
                      'target_variable': 'Property Price'
                  }
              }
              
              # Save to session state
              st.session_state.saved_models[import_name] = imported_model
              
              st.success(f"✅ Model '{import_name}' imported successfully!")
              st.balloons()
      
      else:
          st.error("❌ Invalid model file format. Missing metadata.")
          
  except Exception as e:
      st.error(f"❌ Error importing model: {str(e)}")

# Model Library Management
st.header("📚 Model Library")

if st.session_state.saved_models:
  st.subheader("📋 Available Models")
  
  # Create model summary table
  model_summary = []
  for name, data in st.session_state.saved_models.items():
      model_summary.append({
          'Name': name,
          'Type': data.get('model_type', 'Unknown'),
          'R² Score': f"{data.get('performance', {}).get('R²', 0):.4f}",
          'RMSE': f"${data.get('performance', {}).get('RMSE', 0):,.0f}",
          'Created': data.get('created_at', 'Unknown')[:16],
          'Imported': data.get('imported_at', 'N/A')[:16] if 'imported_at' in data else 'N/A'
      })
  
  summary_df = pd.DataFrame(model_summary)
  st.dataframe(summary_df, use_container_width=True)
  
  # Model management actions
  st.subheader("🔧 Model Management")
  
  selected_model = st.selectbox("Select Model for Actions", list(st.session_state.saved_models.keys()))
  
  if selected_model:
      col1, col2, col3 = st.columns(3)
      
      with col1:
          if st.button("📊 View Details"):
              model_data = st.session_state.saved_models[selected_model]
              
              st.subheader(f"📊 Details: {selected_model}")
              
              # Performance metrics
              st.write("**Performance Metrics:**")
              perf = model_data.get('performance', {})
              for metric, value in perf.items():
                  if metric in ['RMSE', 'MAE', 'MSE']:
                      st.write(f"- {metric}: ${value:,.0f}")
                  else:
                      st.write(f"- {metric}: {value:.4f}")
              
              # Training configuration
              st.write("**Training Configuration:**")
              train_config = model_data.get('training_config', {})
              for key, value in train_config.items():
                  st.write(f"- {key}: {value}")
              
              # Features
              st.write("**Features Used:**")
              features = model_data.get('feature_columns', [])
              for feature in features:
                  st.write(f"- {feature}")
      
      with col2:
          if st.button("📥 Export Model"):
              model_data = st.session_state.saved_models[selected_model]
              
              # Create export package
              export_data = {
                  'model': model_data['model'],
                  'scaler': model_data.get('scaler'),
                  'feature_columns': model_data['feature_columns'],
                  'model_metadata': {
                      'name': selected_model,
                      'type': model_data.get('model_type'),
                      'parameters': model_data.get('model_params', {}),
                      'performance': model_data.get('performance', {}),
                      'training_config': model_data.get('training_config', {}),
                      'created_at': model_data.get('created_at'),
                      'exported_at': datetime.now().isoformat()
                  }
              }
              
              # Save to pickle file
              filename = f"{selected_model.replace(' ', '_')}.pkl"
              
              try:
                  # Convert to bytes for download
                  import io
                  buffer = io.BytesIO()
                  pickle.dump(export_data, buffer)
                  buffer.seek(0)
                  
                  # Provide download
                  st.download_button(
                      label=f"📁 Download {selected_model}",
                      data=buffer.getvalue(),
                      file_name=filename,
                      mime="application/octet-stream"
                  )
                  
                  st.success(f"✅ {selected_model} ready for download!")
                  
              except Exception as e:
                  st.error(f"❌ Export error: {str(e)}")
      
      with col3:
          if st.button("🗑️ Delete Model"):
              if st.session_state.get('confirm_delete') == selected_model:
                  del st.session_state.saved_models[selected_model]
                  if 'confirm_delete' in st.session_state:
                      del st.session_state.confirm_delete
                  st.success(f"✅ Model '{selected_model}' deleted!")
                  st.rerun()
              else:
                  st.session_state.confirm_delete = selected_model
                  st.warning(f"⚠️ Click again to confirm deletion of '{selected_model}'")

  # Bulk operations
  st.subheader("🔄 Bulk Operations")
  
  col1, col2, col3 = st.columns(3)
  
  with col1:
      if st.button("📥 Export All Models"):
          if st.session_state.saved_models:
              # Create a zip file with all models
              import zipfile
              import io
              
              zip_buffer = io.BytesIO()
              
              with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                  for model_name, model_data in st.session_state.saved_models.items():
                      # Create export package for each model
                      export_data = {
                          'model': model_data['model'],
                          'scaler': model_data.get('scaler'),
                          'feature_columns': model_data['feature_columns'],
                          'model_metadata': {
                              'name': model_name,
                              'type': model_data.get('model_type'),
                              'parameters': model_data.get('model_params', {}),
                              'performance': model_data.get('performance', {}),
                              'training_config': model_data.get('training_config', {}),
                              'created_at': model_data.get('created_at'),
                              'exported_at': datetime.now().isoformat()
                          }
                      }
                      
                      # Save to buffer
                      model_buffer = io.BytesIO()
                      pickle.dump(export_data, model_buffer)
                      model_buffer.seek(0)
                      
                      # Add to zip
                      filename = f"{model_name.replace(' ', '_')}.pkl"
                      zip_file.writestr(filename, model_buffer.getvalue())
              
              zip_buffer.seek(0)
              
              st.download_button(
                  label="📦 Download All Models (ZIP)",
                  data=zip_buffer.getvalue(),
                  file_name=f"all_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                  mime="application/zip"
              )
              
              st.success("✅ All models packaged for download!")
          else:
              st.warning("⚠️ No models to export!")
  
  with col2:
      if st.button("📊 Generate Report"):
          if st.session_state.saved_models:
              # Generate comprehensive report
              report_data = {
                  'report_generated': datetime.now().isoformat(),
                  'total_models': len(st.session_state.saved_models),
                  'models': {}
              }
              
              for name, data in st.session_state.saved_models.items():
                  report_data['models'][name] = {
                      'type': data.get('model_type'),
                      'performance': data.get('performance', {}),
                      'training_config': data.get('training_config', {}),
                      'features': data.get('feature_columns', []),
                      'created_at': data.get('created_at'),
                      'imported_at': data.get('imported_at', 'N/A')
                  }
              
              # Convert to JSON
              report_json = json.dumps(report_data, indent=2)
              
              st.download_button(
                  label="📄 Download Report (JSON)",
                  data=report_json,
                  file_name=f"model_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                  mime="application/json"
              )
              
              st.success("✅ Model report generated!")
          else:
              st.warning("⚠️ No models to report!")
  
  with col3:
      if st.button("🗑️ Clear All Models"):
          if st.session_state.get('confirm_clear_all'):
              st.session_state.saved_models = {}
              if 'confirm_clear_all' in st.session_state:
                  del st.session_state.confirm_clear_all
              st.success("✅ All models cleared!")
              st.rerun()
          else:
              st.session_state.confirm_clear_all = True
              st.warning("⚠️ Click again to confirm clearing ALL models")

else:
  st.info("📝 No models in library. Import or train models to get started!")

# Model Statistics
if st.session_state.saved_models:
  st.header("📈 Model Statistics")
  
  # Calculate statistics
  total_models = len(st.session_state.saved_models)
  model_types = {}
  performance_data = []
  
  for name, data in st.session_state.saved_models.items():
      model_type = data.get('model_type', 'Unknown')
      model_types[model_type] = model_types.get(model_type, 0) + 1
      
      perf = data.get('performance', {})
      if perf:
          performance_data.append({
              'Name': name,
              'Type': model_type,
              'R²': perf.get('R²', 0),
              'RMSE': perf.get('RMSE', 0),
              'MAE': perf.get('MAE', 0)
          })
  
  col1, col2 = st.columns(2)
  
  with col1:
      st.subheader("📊 Model Type Distribution")
      
      if model_types:
          import matplotlib.pyplot as plt
          
          fig, ax = plt.subplots(figsize=(8, 6))
          ax.pie(model_types.values(), labels=model_types.keys(), autopct='%1.1f%%')
          ax.set_title('Model Type Distribution')
          st.pyplot(fig)
  
  with col2:
      st.subheader("🏆 Performance Comparison")
      
      if performance_data:
          perf_df = pd.DataFrame(performance_data)
          
          # Best performing model
          best_r2 = perf_df.loc[perf_df['R²'].idxmax()]
          best_rmse = perf_df.loc[perf_df['RMSE'].idxmin()]
          
          st.write(f"**Best R² Score:** {best_r2['Name']} ({best_r2['R²']:.4f})")
          st.write(f"**Best RMSE:** {best_rmse['Name']} (${best_rmse['RMSE']:,.0f})")
          
          # Performance metrics
          st.write(f"**Average R² Score:** {perf_df['R²'].mean():.4f}")
          st.write(f"**Average RMSE:** ${perf_df['RMSE'].mean():,.0f}")

# Footer
st.markdown("---")
st.markdown("*Model Import & Management - Import, export, and manage your machine learning model library.*")