import os
import pandas as pd
import pickle
import json
from datetime import datetime
import hashlib
from pathlib import Path
from typing import List

class RepoManager:
  """Persistent data repository manager"""
  
  def __init__(self, base_path="repo"):
      self.base_path = base_path
      self.datasets_path = os.path.join(base_path, "datasets")
      self.processed_path = os.path.join(base_path, "processed")
      self.models_path = os.path.join(base_path, "models")
      self.reports_path = os.path.join(base_path, "reports")
      
      # Klasörleri oluştur
      self._create_directories()
      
      # Metadata dosyası
      self.metadata_file = os.path.join(base_path, "metadata.json")
      self.metadata = self._load_metadata()

      self.data_dir = Path("data")
      self.data_dir.mkdir(exist_ok=True)
      self.raw_data_path = self.data_dir / "raw_data.csv"
      self.processed_data_path = self.data_dir / "processed_data.csv"
      self.crime_data_path = self.data_dir / "crime_data.csv"
      self.raw_data = None
      self.processed_data = None
      self.crime_data = None
      self._load_data()
  
  def _create_directories(self):
      """Create necessary directories"""
      for path in [self.datasets_path, self.processed_path, 
                   self.models_path, self.reports_path]:
          os.makedirs(path, exist_ok=True)
  
  def _load_metadata(self):
      """Load metadata file"""
      if os.path.exists(self.metadata_file):
          try:
              with open(self.metadata_file, 'r', encoding='utf-8') as f:
                  return json.load(f)
          except:
              return self._create_empty_metadata()
      return self._create_empty_metadata()
  
  def _create_empty_metadata(self):
      """Create empty metadata structure"""
      return {
          "datasets": {},
          "processed": {},
          "models": {},
          "reports": {},
          "created_at": datetime.now().isoformat(),
          "version": "1.0"
      }
  
  def _save_metadata(self):
      """Save metadata file"""
      self.metadata["last_updated"] = datetime.now().isoformat()
      with open(self.metadata_file, 'w', encoding='utf-8') as f:
          json.dump(self.metadata, f, indent=2, default=str, ensure_ascii=False)
  
  def _generate_id(self, data):
      """Generate unique ID for data"""
      if isinstance(data, pd.DataFrame):
          data_str = f"{data.shape}_{hash(tuple(data.columns))}_{datetime.now().timestamp()}"
      else:
          data_str = f"{str(data)}_{datetime.now().timestamp()}"
      return hashlib.md5(data_str.encode()).hexdigest()[:8]
  
  def save_dataset(self, df, name, description=""):
      """Save dataset"""
      timestamp = datetime.now()
      data_id = self._generate_id(df)
      filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{name}_{data_id}.csv"
      filepath = os.path.join(self.datasets_path, filename)
      
      # Veriyi kaydet
      df.to_csv(filepath, index=False, encoding='utf-8')
      
      # Metadata güncelle
      self.metadata["datasets"][data_id] = {
          "name": name,
          "description": description,
          "filename": filename,
          "filepath": filepath,
          "timestamp": timestamp.isoformat(),
          "rows": len(df),
          "columns": len(df.columns),
          "size_mb": os.path.getsize(filepath) / (1024*1024)
      }
      
      self._save_metadata()
      return data_id
  
  def save_model(self, model, name, model_type, metrics, data_id):
      """Save model"""
      timestamp = datetime.now()
      model_id = hashlib.md5(f"{name}_{timestamp}".encode()).hexdigest()[:8]
      filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{name}_{model_id}.pkl"
      filepath = os.path.join(self.models_path, filename)
      
      # Modeli kaydet
      with open(filepath, 'wb') as f:
          pickle.dump(model, f)
      
      # Metadata güncelle
      self.metadata["models"][model_id] = {
          "name": name,
          "model_type": model_type,
          "filename": filename,
          "filepath": filepath,
          "timestamp": timestamp.isoformat(),
          "metrics": metrics,
          "data_id": data_id,
          "size_mb": os.path.getsize(filepath) / (1024*1024)
      }
      
      self._save_metadata()
      return model_id
  
  def load_dataset(self, data_id):
      """Load dataset"""
      if data_id in self.metadata["datasets"]:
          filepath = self.metadata["datasets"][data_id]["filepath"]
          return pd.read_csv(filepath)
      return None
  
  def load_model(self, model_id):
      """Load model"""
      if model_id in self.metadata["models"]:
          filepath = self.metadata["models"][model_id]["filepath"]
          with open(filepath, 'rb') as f:
              return pickle.load(f)
      return None
  
  def list_datasets(self):
      """List all datasets"""
      return self.metadata["datasets"]
  
  def list_models(self):
      """List all models"""
      return self.metadata["models"]
  
  def get_storage_info(self):
      """Get repository information"""
      total_datasets = len(self.metadata["datasets"])
      total_processed = len(self.metadata.get("processed", {}))
      total_models = len(self.metadata["models"])
      
      # Toplam boyut hesapla
      total_size = 0
      for category in ["datasets", "models"]:
          for item in self.metadata.get(category, {}).values():
              total_size += item.get("size_mb", 0)
      
      return {
          "total_datasets": total_datasets,
          "total_processed": total_processed,
          "total_models": total_models,
          "total_size_mb": round(total_size, 2)
      }
  
  def get_data(self, data_type: str = "processed") -> pd.DataFrame:
      """Get data of specified type"""
      try:
          if data_type == "raw":
              return self.raw_data if self.raw_data is not None else pd.DataFrame()
          elif data_type == "processed":
              return self.processed_data if self.processed_data is not None else pd.DataFrame()
          elif data_type == "crime":
              return self.crime_data if self.crime_data is not None else pd.DataFrame()
          else:
              raise ValueError(f"Invalid data type: {data_type}")
      except Exception as e:
          print(f"Error getting data: {str(e)}")
          return pd.DataFrame()

  def _load_data(self):
      """Load all data files if they exist"""
      try:
          if self.raw_data_path.exists():
              self.raw_data = pd.read_csv(self.raw_data_path)
          if self.processed_data_path.exists():
              self.processed_data = pd.read_csv(self.processed_data_path)
          if self.crime_data_path.exists():
              self.crime_data = pd.read_csv(self.crime_data_path)
      except Exception as e:
          print(f"Error loading data: {str(e)}")

  def get_available_datasets(self) -> List[str]:
      """Get list of available datasets"""
      available = []
      if self.raw_data_path.exists():
          available.append("raw")
      if self.processed_data_path.exists():
          available.append("processed")
      if self.crime_data_path.exists():
          available.append("crime")
      return available

  def clear_data(self, data_type: str = "all"):
      """Clear specified data"""
      try:
          if data_type == "all":
              self.raw_data = None
              self.processed_data = None
              self.crime_data = None
              for file in self.data_dir.glob("*.csv"):
                  file.unlink()
          elif data_type == "raw":
              self.raw_data = None
              if self.raw_data_path.exists():
                  self.raw_data_path.unlink()
          elif data_type == "processed":
              self.processed_data = None
              if self.processed_data_path.exists():
                  self.processed_data_path.unlink()
          elif data_type == "crime":
              self.crime_data = None
              if self.crime_data_path.exists():
                  self.crime_data_path.unlink()
          else:
              raise ValueError(f"Invalid data type: {data_type}")
      except Exception as e:
          print(f"Error clearing data: {str(e)}")