"""
Production High-Accuracy Predictor
Based EXACTLY on final_high_accuracy_test.py and high_accuracy_model.py
Uses the EXACT same feature engineering and ensemble approach that achieves 98% accuracy
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
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Handle XGBoost import gracefully
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"⚠️ XGBoost not available: {e}")
    XGBOOST_AVAILABLE = False

class ProductionHighAccuracyPredictor:
    """
    Production version of your high-accuracy model
    Uses EXACT same feature engineering and ensemble as final_high_accuracy_test.py
    """
    
    def __init__(self):
        self.models = {}
        self.meta_learner = None
        self.scaler = None
        self.feature_columns = None
        self.is_trained = False
        self.model_metrics = {}
        
        # Original features from your model
        self.feature_cols = ['Temperature_C', 'Rainfall_mm', 'pH', 'Dissolved_Oxygen_mg_L']
        self.target_col = 'Groundwater_Level_m'  # Updated to match training data
        
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        EXACT feature engineering from final_high_accuracy_test.py
        """
        df = df.copy()
        
        # Convert Date to datetime and extract temporal features
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['DayOfYear'] = df['Date'].dt.dayofyear
            df['Quarter'] = df['Date'].dt.quarter
        
        # 1. Lag features
        for col in self.feature_cols:
            if col in df.columns:
                df[f'{col}_lag1'] = df[col].shift(1)
                df[f'{col}_lag7'] = df[col].shift(7)
                df[f'{col}_lag30'] = df[col].shift(30)
        
        # 2. Rolling statistics
        for col in self.feature_cols:
            if col in df.columns:
                df[f'{col}_rolling_mean_7'] = df[col].rolling(window=7).mean()
                df[f'{col}_rolling_std_7'] = df[col].rolling(window=7).std()
                df[f'{col}_rolling_mean_30'] = df[col].rolling(window=30).mean()
                df[f'{col}_rolling_std_30'] = df[col].rolling(window=30).std()
        
        # 3. Seasonal features
        if 'DayOfYear' in df.columns:
            df['Temperature_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
            df['Temperature_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
            df['Rainfall_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
            df['Rainfall_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        
        # 4. Interaction features
        if all(col in df.columns for col in ['Temperature_C', 'Rainfall_mm']):
            df['Temp_Rainfall'] = df['Temperature_C'] * df['Rainfall_mm']
        if all(col in df.columns for col in ['pH', 'Dissolved_Oxygen_mg_L']):
            df['pH_DO'] = df['pH'] * df['Dissolved_Oxygen_mg_L']
        if all(col in df.columns for col in ['Temperature_C', 'pH']):
            df['Temp_pH'] = df['Temperature_C'] * df['pH']
        if all(col in df.columns for col in ['Rainfall_mm', 'pH']):
            df['Rainfall_pH'] = df['Rainfall_mm'] * df['pH']
        
        # 5. Polynomial features
        for col in self.feature_cols:
            if col in df.columns:
                df[f'{col.split("_")[0]}_sq'] = df[col] ** 2
        
        # 6. Water level trend features (only if target exists)
        if self.target_col in df.columns:
            df['Water_Level_lag1'] = df[self.target_col].shift(1)
            df['Water_Level_lag7'] = df[self.target_col].shift(7)
            df['Water_Level_diff'] = df[self.target_col].diff()
            df['Water_Level_rolling_mean_7'] = df[self.target_col].rolling(window=7).mean()
            df['Water_Level_rolling_std_7'] = df[self.target_col].rolling(window=7).std()
        
        # 7. Weather patterns
        if 'Rainfall_mm' in df.columns:
            df['Rainfall_cumulative_7'] = df['Rainfall_mm'].rolling(window=7).sum()
            df['Rainfall_cumulative_30'] = df['Rainfall_mm'].rolling(window=30).sum()
            df['Dry_days'] = (df['Rainfall_mm'] == 0).astype(int).rolling(window=7).sum()
            df['Heavy_rain'] = (df['Rainfall_mm'] > df['Rainfall_mm'].quantile(0.8)).astype(int)
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get all engineered feature columns (excluding Date and target)
        """
        exclude_cols = ['Date', self.target_col]
        feature_columns = [col for col in df.columns 
                          if col not in exclude_cols and not df[col].isna().all()]
        return feature_columns
    
    def train_ensemble(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the EXACT ensemble from final_high_accuracy_test.py
        """
        print("🚀 Training Production High-Accuracy Ensemble Model...")
        print("📋 Using EXACT architecture from final_high_accuracy_test.py")
        
        # Apply feature engineering
        df_processed = self.create_advanced_features(df.copy())
        
        # Get feature columns
        self.feature_columns = self.get_feature_columns(df_processed)
        
        # Clean data
        df_clean = df_processed.dropna()
        print(f"📈 Clean dataset shape: {df_clean.shape}")
        print(f"🔢 Total features: {len(self.feature_columns)}")
        
        X = df_clean[self.feature_columns].values
        y = df_clean[self.target_col].values
        
        # Time-based split (80% train, 20% validation) - EXACT same as your code
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale data with RobustScaler - EXACT same as your code
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        print(f"📈 Training set: {X_train_scaled.shape[0]} samples")
        print(f"📈 Validation set: {X_val_scaled.shape[0]} samples")
        
        # Train individual models - EXACT same parameters as your code
        if XGBOOST_AVAILABLE:
            print("🌳 Training XGBoost models (EXACT parameters from your code)...")
            
            # XGBoost 1: High complexity - EXACT parameters
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
            
            # XGBoost 2: Medium complexity - EXACT parameters
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
            
            # XGBoost 3: Low complexity - EXACT parameters
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
        else:
            print("⚠️ XGBoost not available, using scikit-learn alternatives...")
            # Use similar performing scikit-learn models as fallback
            from sklearn.ensemble import ExtraTreesRegressor
            
            self.models['et1'] = ExtraTreesRegressor(
                n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
            self.models['et1'].fit(X_train_scaled, y_train)
            
            self.models['et2'] = ExtraTreesRegressor(
                n_estimators=300, max_depth=12, random_state=43, n_jobs=-1)
            self.models['et2'].fit(X_train_scaled, y_train)
            
            self.models['et3'] = ExtraTreesRegressor(
                n_estimators=200, max_depth=8, random_state=44, n_jobs=-1)
            self.models['et3'].fit(X_train_scaled, y_train)
        
        # Random Forest - EXACT parameters from your code
        print("🌲 Training Random Forest (EXACT parameters)...")
        self.models['rf'] = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        self.models['rf'].fit(X_train_scaled, y_train)
        
        # Gradient Boosting - EXACT parameters from your code
        print("🌿 Training Gradient Boosting (EXACT parameters)...")
        self.models['gb'] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        self.models['gb'].fit(X_train_scaled, y_train)
        
        # Create ensemble with meta-learner - EXACT same as your code
        print("🎯 Creating ensemble with meta-learner (EXACT approach)...")
        val_predictions = {}
        for name, model in self.models.items():
            val_predictions[name] = model.predict(X_val_scaled)
        
        meta_features = np.column_stack(list(val_predictions.values()))
        self.meta_learner = Ridge(alpha=0.1)  # EXACT same parameters
        self.meta_learner.fit(meta_features, y_val)
        
        # Calculate final ensemble prediction
        final_pred = self.meta_learner.predict(meta_features)
        
        # Calculate metrics - EXACT same as your code
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
            'validation_samples': len(X_val),
            'xgboost_available': XGBOOST_AVAILABLE
        }
        
        self.is_trained = True
        
        print(f"\n🎯 YOUR HIGH-ACCURACY MODEL PERFORMANCE:")
        print(f"   R² Score: {r2:.4f}")
        print(f"   MSE: {mse:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        
        # Accuracy assessment like your code
        if r2 >= 0.9:
            accuracy_level = "EXCELLENT"
            emoji = "🎉"
        elif r2 >= 0.8:
            accuracy_level = "VERY GOOD"
            emoji = "✅"
        elif r2 >= 0.7:
            accuracy_level = "GOOD"
            emoji = "👍"
        else:
            accuracy_level = "NEEDS IMPROVEMENT"
            emoji = "⚠️"
        
        print(f"\n{emoji} ACCURACY LEVEL: {accuracy_level}")
        print("✅ Production ensemble training completed using YOUR exact architecture!")
        
        return self.model_metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained ensemble"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Apply same feature engineering
        df_processed = self.create_advanced_features(df.copy())
        
        # Get features, handle missing columns
        available_features = [col for col in self.feature_columns if col in df_processed.columns]
        missing_features = [col for col in self.feature_columns if col not in df_processed.columns]
        
        if missing_features:
            print(f"⚠️ Missing features for prediction: {missing_features[:5]}...")
            # Create missing features with default values
            for col in missing_features:
                df_processed[col] = 0
        
        X = df_processed[self.feature_columns].fillna(0).values
        
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
        """Make predictions with confidence intervals"""
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
            'feature_cols': self.feature_cols,
            'target_col': self.target_col
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(model_data, filepath)
        print(f"✅ High-accuracy ensemble model saved to {filepath}")
    
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
        self.feature_cols = model_data.get('feature_cols', self.feature_cols)
        self.target_col = model_data.get('target_col', self.target_col)
        self.is_trained = True
        
        print(f"✅ High-accuracy ensemble model loaded from {filepath}")
        if self.model_metrics:
            print(f"📊 Model metrics: R²={self.model_metrics.get('r2_score', 'N/A'):.4f}, "
                  f"MAE={self.model_metrics.get('mae', 'N/A'):.4f}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        return {
            "model_type": "Production High-Accuracy Ensemble (Your Exact Architecture)",
            "based_on": "final_high_accuracy_test.py",
            "base_models": list(self.models.keys()),
            "n_features": len(self.feature_columns) if self.feature_columns else 0,
            "xgboost_available": XGBOOST_AVAILABLE,
            "feature_engineering": {
                "lag_features": "1, 7, 30 day lags",
                "rolling_features": "7, 30 day rolling mean/std",
                "seasonal_features": "sin/cos seasonal encoding",
                "interaction_features": "temp-rainfall, pH-DO, etc.",
                "polynomial_features": "squared terms",
                "weather_patterns": "cumulative rainfall, dry days, heavy rain"
            },
            "metrics": self.model_metrics,
            "scaler_type": type(self.scaler).__name__ if self.scaler else None,
            "meta_learner_type": type(self.meta_learner).__name__ if self.meta_learner else None
        }
    
    def predict_future_scenario(self, months_ahead: int = 6, 
                               rainfall_scenario: str = 'normal') -> List[Dict[str, Any]]:
        """Predict future water levels using YOUR model architecture"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        from datetime import datetime, timedelta
        
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