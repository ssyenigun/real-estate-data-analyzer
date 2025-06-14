import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest, f_regression, mutual_info_regression
from sklearn.inspection import permutation_importance
import statsmodels.api as sm
from scipy import stats
import time
import joblib
import os
import plotly
import json
from datetime import datetime

class AdvancedModelTrainer:
    def __init__(self, data_path='src/processed_data.csv'):
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.evaluation_metrics = {}
        self.feature_selection_results = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess data with proper handling of missing values"""
        try:
            df = pd.read_csv(self.data_path)
            
            # Separate features and target
            self.X = df.drop('Property Price', axis=1)
            self.y = df['Property Price']
            
            # Handle categorical variables
            categorical_cols = self.X.select_dtypes(include=['object']).columns
            self.label_encoders = {}
            for col in categorical_cols:
                self.label_encoders[col] = LabelEncoder()
                self.X[col] = self.label_encoders[col].fit_transform(self.X[col].astype(str))
            
            # Handle missing values without using target variable
            numeric_cols = self.X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                self.X[col].fillna(self.X[col].median(), inplace=True)
            
            # Check for multicollinearity
            self.check_multicollinearity()
            
            # Split data chronologically if date column exists
            if 'Published_Date' in self.X.columns:
                self.X = self.X.sort_values('Published_Date')
                self.y = self.y[self.X.index]
                split_idx = int(len(self.X) * 0.8)
                self.X_train = self.X.iloc[:split_idx]
                self.X_test = self.X.iloc[split_idx:]
                self.y_train = self.y.iloc[:split_idx]
                self.y_test = self.y.iloc[split_idx:]
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X, self.y, test_size=0.2, random_state=42
                )
            
            # Scale features
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            return True
        except Exception as e:
            print(f"Error in data preprocessing: {str(e)}")
            return False
    
    def check_multicollinearity(self):
        """Check for multicollinearity using VIF"""
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns
        vif_data = pd.DataFrame()
        vif_data["Variable"] = numeric_cols
        vif_data["VIF"] = [self.calculate_vif(self.X[numeric_cols], var) for var in numeric_cols]
        self.vif_data = vif_data.sort_values('VIF', ascending=False)
    
    def calculate_vif(self, X, feature):
        """Calculate VIF for a single feature"""
        X_with_const = sm.add_constant(X.drop(feature, axis=1))
        model = sm.OLS(X[feature], X_with_const).fit()
        return 1 / (1 - model.rsquared)
    
    def perform_feature_selection(self):
        """Perform comprehensive feature selection using multiple methods"""
        # 1. Filter Method - Correlation Threshold
        correlation_matrix = self.X_train.corr()
        high_corr_features = np.where(np.abs(correlation_matrix) > 0.8)
        high_corr_features = [(correlation_matrix.index[x], correlation_matrix.columns[y], correlation_matrix.iloc[x, y]) 
                            for x, y in zip(*high_corr_features) if x != y and x < y]
        
        # Filter Method - Statistical Tests with multiple thresholds
        f_regression_features = {}
        mutual_info_features = {}
        for k in [5, 10, 15]:
            f_regression_features[k] = SelectKBest(f_regression, k=k).fit(self.X_train_scaled, self.y_train)
            mutual_info_features[k] = SelectKBest(mutual_info_regression, k=k).fit(self.X_train_scaled, self.y_train)
        
        # 2. Wrapper Method - Recursive Feature Elimination with cross-validation
        rfe_rf = RFE(
            estimator=RandomForestRegressor(n_estimators=100, random_state=42),
            n_features_to_select=10,
            step=1
        )
        rfe_rf.fit(self.X_train_scaled, self.y_train)
        rfe_features = self.X_train.columns[rfe_rf.support_].tolist()
        
        # 3. Hybrid Method - SelectFromModel + Permutation Importance
        # SelectFromModel with different thresholds
        selector = SelectFromModel(
            RandomForestRegressor(n_estimators=100, random_state=42),
            threshold='median'
        )
        selector.fit(self.X_train_scaled, self.y_train)
        selected_features = self.X_train.columns[selector.get_support()].tolist()
        
        # Permutation Importance with multiple random states for stability
        perm_importance_results = []
        for seed in [42, 123, 456]:
            rf = RandomForestRegressor(n_estimators=100, random_state=seed)
            rf.fit(self.X_train_scaled, self.y_train)
            perm_importance = permutation_importance(
                rf, 
                self.X_train_scaled, 
                self.y_train, 
                n_repeats=10, 
                random_state=seed
            )
            perm_importance_results.append(perm_importance.importances_mean)
        
        # Calculate mean and std of permutation importance across seeds
        mean_importance = np.mean(perm_importance_results, axis=0)
        std_importance = np.std(perm_importance_results, axis=0)
        
        perm_features = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance_mean': mean_importance,
            'importance_std': std_importance
        }).sort_values('importance_mean', ascending=False)
        
        self.feature_selection_results = {
            'high_correlation': high_corr_features,
            'f_regression_features': {k: self.X_train.columns[v.get_support()].tolist() 
                                    for k, v in f_regression_features.items()},
            'mutual_info_features': {k: self.X_train.columns[v.get_support()].tolist() 
                                   for k, v in mutual_info_features.items()},
            'rfe_features': rfe_features,
            'selected_features': selected_features,
            'permutation_importance': perm_features
        }
        
        return self.feature_selection_results
    
    def evaluate_model_stability(self, model_name, n_seeds=5):
        """Evaluate model stability across different random seeds"""
        metrics_list = []
        training_times = []
        
        for seed in range(n_seeds):
            start_time = time.time()
            
            # Train model with current seed
            if model_name == 'Random Forest':
                model = RandomForestRegressor(random_state=seed)
            elif model_name == 'SVR':
                model = SVR()
            else:
                print(f"Model {model_name} not supported for stability analysis")
                return None
            
            model.fit(self.X_train_scaled, self.y_train)
            training_time = time.time() - start_time
            training_times.append(training_time)
            
            # Make predictions
            y_pred_train = model.predict(self.X_train_scaled)
            y_pred_test = model.predict(self.X_test_scaled)
            
            # Calculate metrics
            metrics = {
                'train_mae': mean_absolute_error(self.y_train, y_pred_train),
                'test_mae': mean_absolute_error(self.y_test, y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
                'train_r2': r2_score(self.y_train, y_pred_train),
                'test_r2': r2_score(self.y_test, y_pred_test)
            }
            metrics_list.append(metrics)
        
        # Calculate mean and std of metrics across seeds
        stability_metrics = {}
        for metric in metrics_list[0].keys():
            values = [m[metric] for m in metrics_list]
            stability_metrics[f'{metric}_mean'] = np.mean(values)
            stability_metrics[f'{metric}_std'] = np.std(values)
        
        stability_metrics['training_time_mean'] = np.mean(training_times)
        stability_metrics['training_time_std'] = np.std(training_times)
        
        return stability_metrics
    
    def tune_hyperparameters(self, model_name):
        """Perform hyperparameter tuning for specific models"""
        # Load and preprocess data if not already loaded
        if not hasattr(self, 'X_train_scaled'):
            if not self.load_and_preprocess_data():
                print("Error: Data not loaded for hyperparameter tuning.")
                return None, None

        if model_name == 'Random Forest':
            param_distributions = {
                'n_estimators': [50, 100, 150, 200, 250],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
            model = RandomForestRegressor(random_state=42)
            search = RandomizedSearchCV(
                model,
                param_distributions,
                n_iter=50,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=42
            )
            
        elif model_name == 'SVR':
            param_distributions = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.01, 0.001],
                'kernel': ['rbf', 'linear'],
                'epsilon': [0.1, 0.2]
            }
            model = SVR()
            search = RandomizedSearchCV(
                model,
                param_distributions,
                n_iter=30,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=42
            )
        else:
            print(f"Error: Hyperparameter tuning not supported for model '{model_name}'.")
            return None, None
        
        # Record training time
        start_time = time.time()
        search.fit(self.X_train_scaled, self.y_train)
        training_time = time.time() - start_time
        
        best_model = search.best_estimator_
        best_params = search.best_params_
        
        # Calculate feature importance if applicable
        if hasattr(best_model, 'feature_importances_'):
            self.feature_importance[model_name] = dict(zip(self.X.columns, best_model.feature_importances_))
        
        # Make predictions and calculate metrics
        y_pred_train = best_model.predict(self.X_train_scaled)
        y_pred_test = best_model.predict(self.X_test_scaled)
        
        # Calculate stability metrics
        stability_metrics = self.evaluate_model_stability(model_name)
        
        metrics = {
            'train_mae': mean_absolute_error(self.y_train, y_pred_train),
            'test_mae': mean_absolute_error(self.y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
            'train_r2': r2_score(self.y_train, y_pred_train),
            'test_r2': r2_score(self.y_test, y_pred_test),
            'best_params': best_params,
            'training_time': training_time,
            'stability_metrics': stability_metrics
        }
        
        self.models[model_name] = best_model
        self.evaluation_metrics[model_name] = metrics
        
        return best_model, metrics
    
    def train_models(self):
        """Train models with enhanced feature selection and hyperparameter tuning"""
        # First perform feature selection
        self.perform_feature_selection()
        
        # Train Random Forest with hyperparameter tuning
        rf_model, rf_metrics = self.tune_hyperparameters('Random Forest')
        print("\nRandom Forest Results:")
        print(f"Best Parameters: {rf_metrics['best_params']}")
        print(f"Test RMSE: {rf_metrics['test_rmse']:.2f}")
        print(f"Test R2: {rf_metrics['test_r2']:.2f}")
        
        # Train SVR with hyperparameter tuning
        svr_model, svr_metrics = self.tune_hyperparameters('SVR')
        print("\nSVR Results:")
        print(f"Best Parameters: {svr_metrics['best_params']}")
        print(f"Test RMSE: {svr_metrics['test_rmse']:.2f}")
        print(f"Test R2: {svr_metrics['test_r2']:.2f}")
    
    def save_models(self, directory='src/saved_models'):
        """Save trained models and their metadata"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for name, model in self.models.items():
            # Save model and metadata
            model_data = {
                'model': model,
                'model_type': name,
                'features': self.X.columns.tolist(),
                'parameters': model.get_params(),
                'metrics': self.evaluation_metrics[name],
                'feature_selection_results': self.feature_selection_results,
                'vif_data': self.vif_data.to_dict() if hasattr(self, 'vif_data') else None,
                'timestamp': datetime.now()
            }
            
            model_path = os.path.join(directory, f"{name.lower().replace(' ', '_')}.joblib")
            joblib.dump(model_data, model_path)
            
            # Save additional metadata in JSON format for easier inspection
            metadata_path = os.path.join(directory, f"{name.lower().replace(' ', '_')}_metadata.json")
            metadata = {
                'model_type': name,
                'features': self.X.columns.tolist(),
                'parameters': model.get_params(),
                'metrics': {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                    for k, v in self.evaluation_metrics[name].items() 
                    if k != 'stability_metrics'
                },
                'stability_metrics': {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                    for k, v in self.evaluation_metrics[name].get('stability_metrics', {}).items()
                } if 'stability_metrics' in self.evaluation_metrics[name] else None,
                'feature_selection_results': {
                    k: v.tolist() if isinstance(v, np.ndarray) else v 
                    for k, v in self.feature_selection_results.items()
                },
                'vif_data': self.vif_data.to_dict() if hasattr(self, 'vif_data') else None,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
    
    def load_models(self, directory='src/saved_models'):
        """Load saved models and their metadata"""
        for name in self.models.keys():
            model_path = os.path.join(directory, f"{name.lower().replace(' ', '_')}.joblib")
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.models[name] = model_data['model']
                self.evaluation_metrics[name] = model_data['metrics']
                self.feature_selection_results = model_data.get('feature_selection_results', {})
                if model_data.get('vif_data'):
                    self.vif_data = pd.DataFrame.from_dict(model_data['vif_data'])
    
    def get_model_summary(self):
        """Get summary of all models' performance"""
        summary = pd.DataFrame(self.evaluation_metrics).T
        summary = summary.sort_values('test_rmse')
        return summary
    
    def get_feature_importance_summary(self):
        """Get summary of feature importance across all models"""
        importance_summary = {}
        for model_name, importance_dict in self.feature_importance.items():
            importance_summary[model_name] = pd.Series(importance_dict).sort_values(ascending=False)
        return importance_summary

def main():
    trainer = AdvancedModelTrainer()
    if trainer.load_and_preprocess_data():
        trainer.perform_feature_selection()
        trainer.train_models()
        trainer.save_models()
        
        # Print summary
        print("\nModel Performance Summary:")
        print(trainer.get_model_summary())
        
        print("\nFeature Importance Summary:")
        importance_summary = trainer.get_feature_importance_summary()
        for model_name, importance in importance_summary.items():
            print(f"\n{model_name}:")
            print(importance.head())

if __name__ == "__main__":
    main() 