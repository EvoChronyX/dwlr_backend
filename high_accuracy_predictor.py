"""
High-Accuracy Ensemble Model Predictor
Based on final_high_accuracy_test.py ensemble approach
"""
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import AdvancedFeatureEngineer

class HighAccuracyEnsemblePredictor:
    """
    High-Accuracy Ensemble Model for Groundwater Level Prediction
    Implements the ensemble approach from final_high_accuracy_test.py
    """
    
    def __init__(self):
        self.models = {}
        self.meta_learner = None
        self.scaler = None
        self.feature_engineer = AdvancedFeatureEngineer()
        self.feature_columns = None
        self.is_trained = False
        self.model_metrics = {}
        
    def train_ensemble(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the complete ensemble model
        Returns training metrics
        """
        print("🚀 Training High-Accuracy Ensemble Model...")
        
        # Feature engineering
        df_processed = self.feature_engineer.create_features(df.copy())
        
        # Get feature columns
        self.feature_columns = self.feature_engineer.get_feature_columns(df_processed)
        
        # Clean data
        df_clean = df_processed.dropna()
        print(f"📈 Clean dataset shape: {df_clean.shape}")
        
        X = df_clean[self.feature_columns].values
        y = df_clean[self.feature_engineer.target_col].values
        
        # Time-based split (80% train, 20% validation)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale data
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        print(f"📈 Training set: {X_train_scaled.shape[0]} samples")
        print(f"📈 Validation set: {X_val_scaled.shape[0]} samples")
        print(f"🔢 Total features: {X_train_scaled.shape[1]}")
        
        # Train individual models
        print("🌳 Training XGBoost models...")
        
        # XGBoost 1: High complexity
        self.models['xgb1'] = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.005,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        self.models['xgb1'].fit(X_train_scaled, y_train)
        
        # XGBoost 2: Medium complexity
        self.models['xgb2'] = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.05,
            reg_lambda=0.5,
            random_state=43,
            n_jobs=-1
        )
        self.models['xgb2'].fit(X_train_scaled, y_train)
        
        # XGBoost 3: Low complexity
        self.models['xgb3'] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.02,
            subsample=0.95,
            colsample_bytree=0.95,
            reg_alpha=0.01,
            reg_lambda=0.1,
            random_state=44,
            n_jobs=-1
        )
        self.models['xgb3'].fit(X_train_scaled, y_train)
        
        print("🌲 Training Random Forest...")
        self.models['rf'] = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        self.models['rf'].fit(X_train_scaled, y_train)
        
        print("🌿 Training Gradient Boosting...")
        self.models['gb'] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        self.models['gb'].fit(X_train_scaled, y_train)
        
        # Create ensemble with meta-learner
        print("🎯 Creating ensemble with meta-learner...")
        val_predictions = {}
        for name, model in self.models.items():
            val_predictions[name] = model.predict(X_val_scaled)
        
        meta_features = np.column_stack(list(val_predictions.values()))
        self.meta_learner = Ridge(alpha=0.1)
        self.meta_learner.fit(meta_features, y_val)
        
        # Calculate final ensemble prediction
        final_pred = self.meta_learner.predict(meta_features)
        
        # Calculate metrics
        mse = mean_squared_error(y_val, final_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, final_pred)
        r2 = r2_score(y_val, final_pred)
        mape = np.mean(np.abs((y_val - final_pred) / y_val)) * 100
        
        self.model_metrics = {
            'r2_score': float(r2),
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'n_features': len(self.feature_columns),
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }
        
        self.is_trained = True
        
        print(f"\n🎯 ENSEMBLE MODEL PERFORMANCE:")
        print(f"   R² Score: {r2:.4f}")
        print(f"   MSE: {mse:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        print("✅ Ensemble training completed!")
        
        return self.model_metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained ensemble"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X = self.feature_engineer.prepare_for_prediction(df, self.feature_columns)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_scaled)
        
        # Create meta-features
        meta_features = np.column_stack(list(predictions.values()))
        
        # Final ensemble prediction
        final_predictions = self.meta_learner.predict(meta_features)
        
        return final_predictions
    
    def predict_with_confidence(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals
        Returns: (predictions, lower_bounds, upper_bounds)
        """
        predictions = self.predict(df)
        
        # Calculate confidence intervals based on validation error
        if 'rmse' in self.model_metrics:
            margin = 1.96 * self.model_metrics['rmse']  # 95% confidence interval
            lower_bounds = predictions - margin
            upper_bounds = predictions + margin
        else:
            # Fallback if no metrics available
            margin = predictions * 0.1  # 10% of prediction
            lower_bounds = predictions - margin
            upper_bounds = predictions + margin
        
        return predictions, lower_bounds, upper_bounds
    
    def save_model(self, filepath: str):
        """Save the trained ensemble model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'models': self.models,
            'meta_learner': self.meta_learner,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_metrics': self.model_metrics,
            'feature_engineer': self.feature_engineer
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(model_data, filepath)
        print(f"✅ Ensemble model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained ensemble model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.meta_learner = model_data['meta_learner']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_metrics = model_data.get('model_metrics', {})
        self.feature_engineer = model_data.get('feature_engineer', AdvancedFeatureEngineer())
        self.is_trained = True
        
        print(f"✅ Ensemble model loaded from {filepath}")
        if self.model_metrics:
            print(f"📊 Model metrics: R²={self.model_metrics.get('r2_score', 'N/A'):.4f}, "
                  f"MAE={self.model_metrics.get('mae', 'N/A'):.4f}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        return {
            "model_type": "High-Accuracy Ensemble (XGBoost + RF + GB + Ridge Meta-Learner)",
            "base_models": list(self.models.keys()),
            "n_features": len(self.feature_columns) if self.feature_columns else 0,
            "feature_categories": {
                "original": self.feature_engineer.feature_cols,
                "lag_features": [f for f in (self.feature_columns or []) if '_lag' in f],
                "rolling_features": [f for f in (self.feature_columns or []) if '_rolling' in f],
                "interaction_features": [f for f in (self.feature_columns or []) if any(x in f for x in ['Temp_', 'pH_DO', 'Rainfall_pH'])],
                "polynomial_features": [f for f in (self.feature_columns or []) if '_sq' in f],
                "seasonal_features": [f for f in (self.feature_columns or []) if any(x in f for x in ['_sin', '_cos'])],
            },
            "metrics": self.model_metrics,
            "scaler_type": type(self.scaler).__name__ if self.scaler else None,
            "meta_learner_type": type(self.meta_learner).__name__ if self.meta_learner else None
        }
    
    def predict_future_scenario(self, months_ahead: int = 6, 
                               rainfall_scenario: str = 'normal') -> List[Dict[str, Any]]:
        """
        Predict future water levels based on seasonal patterns and rainfall scenarios
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Generate future dates
        from datetime import datetime, timedelta
        import calendar
        
        today = datetime.now()
        future_predictions = []
        
        # Rainfall scenario multipliers
        rainfall_multipliers = {
            'very_low': 0.3,
            'low': 0.6,
            'normal': 1.0,
            'high': 1.4,
            'very_high': 2.0
        }
        multiplier = rainfall_multipliers.get(rainfall_scenario, 1.0)
        
        for month_offset in range(1, months_ahead + 1):
            future_date = today + timedelta(days=30 * month_offset)
            month = future_date.month
            
            # Create synthetic future data based on seasonal patterns
            # Using typical values for the month
            seasonal_temp = {
                1: 20, 2: 23, 3: 28, 4: 33, 5: 36, 6: 34,
                7: 30, 8: 29, 9: 30, 10: 28, 11: 24, 12: 21
            }
            
            seasonal_rainfall = {
                1: 15, 2: 20, 3: 25, 4: 40, 5: 60, 6: 150,
                7: 200, 8: 180, 9: 120, 10: 80, 11: 30, 12: 20
            }
            
            future_data = pd.DataFrame({
                'Date': [future_date],
                'Temperature_C': [seasonal_temp.get(month, 25)],
                'Rainfall_mm': [seasonal_rainfall.get(month, 50) * multiplier],
                'pH': [7.2],  # Typical pH
                'Dissolved_Oxygen_mg_L': [6.5]  # Typical DO
            })
            
            # Make prediction
            pred, lower, upper = self.predict_with_confidence(future_data)
            
            future_predictions.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'month': month,
                'predicted_level': float(pred[0]),
                'lower_bound': float(lower[0]),
                'upper_bound': float(upper[0]),
                'confidence': 0.95,
                'rainfall_scenario': rainfall_scenario,
                'seasonal_factors': {
                    'temperature': seasonal_temp.get(month, 25),
                    'expected_rainfall': seasonal_rainfall.get(month, 50) * multiplier
                }
            })
        
        return future_predictions
