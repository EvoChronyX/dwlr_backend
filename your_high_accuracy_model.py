"""
Production Backend for YOUR High-Accuracy Groundwater Model
===========================================================

This backend integrates YOUR exact model architecture from final_high_accuracy_test.py:
- ELM (Extreme Learning Machine) with TensorFlow
- 3 XGBoost models with different complexities  
- RandomForest and GradientBoosting
- Advanced feature engineering (98% accuracy)
- All features from groundwater_monitoring_interface.py

Author: Based on your final_high_accuracy_test.py
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Handle XGBoost import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available")

# Set random seeds for reproducibility
np.random.seed(42)
if tf and hasattr(tf, 'random'):
    tf.random.set_seed(42)

class AdvancedELMLayer(tf.keras.layers.Layer):
    """
    Advanced ELM Layer - EXACT same as your final_high_accuracy_test.py
    """
    def __init__(self, n_hidden, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.n_hidden = n_hidden
        self.activation = tf.keras.activations.get(activation)
        self.trainable = False

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.n_hidden),
            initializer='glorot_uniform',
            trainable=False,
            name='W'
        )
        self.b = self.add_weight(
            shape=(self.n_hidden,),
            initializer='zeros',
            trainable=False,
            name='b'
        )

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.W) + self.b)

class YourHighAccuracyModel:
    """
    YOUR EXACT High-Accuracy Model from final_high_accuracy_test.py
    ELM + 3 XGBoost + RandomForest + GradientBoosting = 98% Accuracy
    """
    
    def __init__(self):
        # Original features from YOUR model
        self.feature_cols = ['Temperature_C', 'Rainfall_mm', 'pH', 'Dissolved_Oxygen_mg_L']
        self.target_col = 'Groundwater_Level_m'  # Updated for your training data
        
        # Model components
        self.elm_model = None
        self.ensemble_models = []
        self.meta_learner = None
        self.scaler = None
        self.all_features = None
        
        # Model state
        self.is_trained = False
        self.model_metrics = {}
        
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        EXACT feature engineering from YOUR final_high_accuracy_test.py
        """
        df = df.copy()
        
        # Convert Date to datetime and extract temporal features
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['Quarter'] = df['Date'].dt.quarter
        
        # 1. Lag features (previous day values) - EXACT same as YOUR code
        for col in self.feature_cols:
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_lag7'] = df[col].shift(7)  # Weekly lag
            df[f'{col}_lag30'] = df[col].shift(30)  # Monthly lag
        
        # 2. Rolling statistics - EXACT same as YOUR code
        for col in self.feature_cols:
            df[f'{col}_rolling_mean_7'] = df[col].rolling(window=7).mean()
            df[f'{col}_rolling_std_7'] = df[col].rolling(window=7).std()
            df[f'{col}_rolling_mean_30'] = df[col].rolling(window=30).mean()
            df[f'{col}_rolling_std_30'] = df[col].rolling(window=30).std()
        
        # 3. Seasonal features - EXACT same as YOUR code
        df['Temperature_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['Temperature_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        df['Rainfall_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['Rainfall_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        
        # 4. Interaction features - EXACT same as YOUR code
        df['Temp_Rainfall'] = df['Temperature_C'] * df['Rainfall_mm']
        df['pH_DO'] = df['pH'] * df['Dissolved_Oxygen_mg_L']
        df['Temp_pH'] = df['Temperature_C'] * df['pH']
        df['Rainfall_pH'] = df['Rainfall_mm'] * df['pH']
        
        # 5. Polynomial features - EXACT same as YOUR code
        df['Temperature_sq'] = df['Temperature_C'] ** 2
        df['Rainfall_sq'] = df['Rainfall_mm'] ** 2
        df['pH_sq'] = df['pH'] ** 2
        df['DO_sq'] = df['Dissolved_Oxygen_mg_L'] ** 2
        
        # 6. Water level trend features - EXACT same as YOUR code
        df['Water_Level_lag1'] = df[self.target_col].shift(1)
        df['Water_Level_lag7'] = df[self.target_col].shift(7)
        df['Water_Level_diff'] = df[self.target_col].diff()
        df['Water_Level_rolling_mean_7'] = df[self.target_col].rolling(window=7).mean()
        df['Water_Level_rolling_std_7'] = df[self.target_col].rolling(window=7).std()
        
        # 7. Weather patterns - EXACT same as YOUR code
        df['Rainfall_cumulative_7'] = df['Rainfall_mm'].rolling(window=7).sum()
        df['Rainfall_cumulative_30'] = df['Rainfall_mm'].rolling(window=30).sum()
        df['Dry_days'] = (df['Rainfall_mm'] == 0).astype(int).rolling(window=7).sum()
        df['Heavy_rain'] = (df['Rainfall_mm'] > df['Rainfall_mm'].quantile(0.8)).astype(int)
        
        return df
    
    def build_advanced_elm(self, n_features, n_hidden=128):
        """
        Build Advanced ELM - EXACT same as YOUR final_high_accuracy_test.py
        """
        inputs = tf.keras.Input(shape=(n_features,))
        
        # Multiple ELM layers with different activations - EXACT same
        x1 = AdvancedELMLayer(n_hidden, activation='relu')(inputs)
        x2 = AdvancedELMLayer(n_hidden, activation='tanh')(inputs)
        x3 = AdvancedELMLayer(n_hidden, activation='sigmoid')(inputs)
        
        # Concatenate all ELM outputs - EXACT same
        x = tf.keras.layers.Concatenate()([x1, x2, x3])
        
        # Dense layers with dropout - EXACT same
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        
        # Output layer - EXACT same
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def train_model(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train YOUR EXACT model from final_high_accuracy_test.py
        """
        print("🚀 Training YOUR EXACT High-Accuracy Model...")
        print("📋 Architecture: ELM + 3 XGBoost + RandomForest + GradientBoosting")
        
        # Apply YOUR feature engineering
        df_processed = self.create_advanced_features(df.copy())
        
        # Select all engineered features - EXACT same as YOUR code
        self.all_features = [col for col in df_processed.columns 
                            if col not in ['Date', self.target_col] and not df_processed[col].isna().all()]
        
        # Remove rows with NaN values - EXACT same
        df_clean = df_processed.dropna()
        print(f"📈 Clean dataset shape: {df_clean.shape}")
        
        X = df_clean[self.all_features].values
        y = df_clean[self.target_col].values
        
        print(f"🔢 Total features: {X.shape[1]}")
        print(f"📊 Samples: {X.shape[0]}")
        
        # Time-based split - EXACT same as YOUR code
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Use RobustScaler - EXACT same as YOUR code
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"📈 Training set: {X_train_scaled.shape[0]} samples")
        print(f"📈 Test set: {X_test_scaled.shape[0]} samples")
        
        # Build and train ELM - EXACT same as YOUR code
        print("🧠 Training Advanced ELM...")
        self.elm_model = self.build_advanced_elm(X_train_scaled.shape[1], n_hidden=128)
        
        # Advanced callbacks - EXACT same as YOUR code
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=30, 
                restore_best_weights=True,
                min_delta=1e-6
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=15,
                min_lr=1e-8
            )
        ]
        
        # Train ELM - EXACT same as YOUR code
        history = self.elm_model.fit(
            X_train_scaled, y_train,
            epochs=300,
            batch_size=32,
            validation_data=(X_test_scaled, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Get ELM predictions for ensemble training - EXACT same as YOUR code
        print("🌳 Training Advanced Ensemble...")
        y_pred_elm = self.elm_model.predict(X_train_scaled).flatten()
        residuals = y_train - y_pred_elm
        
        # Multiple models - EXACT same parameters as YOUR code
        self.ensemble_models = []
        
        if XGBOOST_AVAILABLE:
            # XGBoost 1: High complexity - EXACT same as YOUR code
            xgb1 = xgb.XGBRegressor(
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
            xgb1.fit(X_train_scaled, residuals)
            self.ensemble_models.append(('xgb1', xgb1))
            
            # XGBoost 2: Medium complexity - EXACT same as YOUR code
            xgb2 = xgb.XGBRegressor(
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
            xgb2.fit(X_train_scaled, residuals)
            self.ensemble_models.append(('xgb2', xgb2))
            
            # XGBoost 3: Low complexity - EXACT same as YOUR code
            xgb3 = xgb.XGBRegressor(
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
            xgb3.fit(X_train_scaled, residuals)
            self.ensemble_models.append(('xgb3', xgb3))
        
        # RandomForest - EXACT same as YOUR code
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train_scaled, residuals)
        self.ensemble_models.append(('rf', rf))
        
        # GradientBoosting - EXACT same as YOUR code
        gb = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        gb.fit(X_train_scaled, residuals)
        self.ensemble_models.append(('gb', gb))
        
        # Meta-learner - EXACT same as YOUR code
        print("🎯 Training Meta-Learner...")
        meta_features = []
        for name, model in self.ensemble_models:
            pred = model.predict(X_train_scaled)
            meta_features.append(pred)
        
        meta_features = np.column_stack(meta_features)
        self.meta_learner = Ridge(alpha=0.1)
        self.meta_learner.fit(meta_features, residuals)
        
        # Final evaluation - EXACT same as YOUR code
        print("📊 Evaluating YOUR Model...")
        
        # Test predictions
        y_pred_elm_test = self.elm_model.predict(X_test_scaled).flatten()
        
        # Ensemble predictions
        meta_features_test = []
        for name, model in self.ensemble_models:
            pred = model.predict(X_test_scaled)
            meta_features_test.append(pred)
        
        meta_features_test = np.column_stack(meta_features_test)
        y_pred_ensemble = self.meta_learner.predict(meta_features_test)
        
        # Final predictions - EXACT same as YOUR code
        y_pred_final = y_pred_elm_test + y_pred_ensemble
        
        # Calculate metrics - EXACT same as YOUR code
        mse = mean_squared_error(y_test, y_pred_final)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred_final)
        r2 = r2_score(y_test, y_pred_final)
        
        self.model_metrics = {
            'r2_score': float(r2),
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'n_features': len(self.all_features),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'xgboost_available': XGBOOST_AVAILABLE
        }
        
        self.is_trained = True
        
        print(f"\n🎯 YOUR MODEL PERFORMANCE:")
        print(f"   R² Score: {r2:.4f}")
        print(f"   MSE: {mse:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        
        # Success message like YOUR code
        if r2 >= 0.9:
            print("🎉 EXCELLENT! YOUR model achieved R² ≥ 0.9")
        elif r2 >= 0.8:
            print("✅ VERY GOOD! YOUR model achieved R² ≥ 0.8")
        else:
            print(f"👍 YOUR model R² = {r2:.4f}")
            
        print("✅ YOUR exact model training completed!")
        
        return self.model_metrics
    
    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        """Make predictions using YOUR trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # ELM prediction
        elm_pred = self.elm_model.predict(X_scaled).flatten()
        
        # Ensemble predictions
        meta_features = []
        for name, model in self.ensemble_models:
            pred = model.predict(X_scaled)
            meta_features.append(pred)
        
        meta_features = np.column_stack(meta_features)
        ensemble_pred = self.meta_learner.predict(meta_features)
        
        # Final prediction - EXACT same as YOUR code
        final_pred = elm_pred + ensemble_pred
        
        return final_pred
    
    def save_model(self, filepath: str):
        """Save YOUR trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'elm_model_weights': self.elm_model.get_weights() if self.elm_model else None,
            'ensemble_models': self.ensemble_models,
            'meta_learner': self.meta_learner,
            'scaler': self.scaler,
            'all_features': self.all_features,
            'model_metrics': self.model_metrics,
            'feature_cols': self.feature_cols,
            'target_col': self.target_col
        }
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save ELM model separately
        if self.elm_model:
            elm_path = filepath.replace('.pkl', '_elm.h5')
            self.elm_model.save(elm_path)
        
        # Save other components
        joblib.dump(model_data, filepath)
        print(f"✅ YOUR model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load YOUR trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load components
        model_data = joblib.load(filepath)
        
        self.ensemble_models = model_data['ensemble_models']
        self.meta_learner = model_data['meta_learner']
        self.scaler = model_data['scaler']
        self.all_features = model_data['all_features']
        self.model_metrics = model_data.get('model_metrics', {})
        self.feature_cols = model_data.get('feature_cols', self.feature_cols)
        self.target_col = model_data.get('target_col', self.target_col)
        
        # Load ELM model
        elm_path = filepath.replace('.pkl', '_elm.h5')
        if os.path.exists(elm_path):
            try:
                self.elm_model = tf.keras.models.load_model(elm_path, custom_objects={
                    'AdvancedELMLayer': AdvancedELMLayer
                })
                print("✅ ELM model loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load ELM model: {e}")
                self.elm_model = None
        
        self.is_trained = True
        print(f"✅ YOUR model loaded from {filepath}")
        
        if self.model_metrics:
            r2 = self.model_metrics.get('r2_score', 'N/A')
            print(f"📊 YOUR Model R² Score: {r2}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get YOUR model information"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        return {
            "model_name": "YOUR High-Accuracy Ensemble Model",
            "architecture": "ELM + 3 XGBoost + RandomForest + GradientBoosting",
            "based_on": "final_high_accuracy_test.py",
            "elm_layers": "Multi-activation ELM with relu/tanh/sigmoid",
            "ensemble_models": [name for name, _ in self.ensemble_models],
            "meta_learner": "Ridge Regression",
            "feature_engineering": {
                "lag_features": "1, 7, 30 day lags",
                "rolling_features": "7, 30 day rolling mean/std",
                "seasonal_features": "sin/cos encoding",
                "interaction_features": "temp-rainfall, pH-DO, etc.",
                "polynomial_features": "squared terms",
                "weather_patterns": "cumulative rainfall, dry days, heavy rain"
            },
            "performance": self.model_metrics,
            "training_data": "train_dataset.csv",
            "xgboost_available": XGBOOST_AVAILABLE,
            "accuracy_target": "98% (YOUR achievement)"
        }