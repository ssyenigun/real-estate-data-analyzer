PAGE_CONFIG = {
  "page_title": "Real Estate Data Analyzer",
  "page_icon": "🏠",
  "layout": "wide",
  "initial_sidebar_state": "expanded"
}

# Veri işleme ayarları
DATA_PROCESSING_CONFIG = {
  "max_file_size": 200,  # MB
  "supported_formats": [".csv", ".xlsx"],
  "required_columns": ["Property Price"],
  "currency_symbols": ["$", "€", "£", "₺"],
  "currency_codes": ["USD", "EUR", "GBP", "TRY", "CAD", "AUD"]
}

# Repository ayarları
REPO_CONFIG = {
  "base_path": "repo",
  "max_storage_mb": 1000,
  "backup_enabled": True
}