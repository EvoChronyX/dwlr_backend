"""
Advanced Feature Engineering Module
Based on high_accuracy_model.py and final_high_accuracy_test.py
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for groundwater level prediction
    Implements the same feature engineering as in high_accuracy_model.py
    """
    
    def __init__(self):
        self.feature_cols = ['Temperature_C', 'Rainfall_mm', 'pH', 'Dissolved_Oxygen_mg_L']
        self.target_col = 'Groundwater_Level_m'
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features from input data"""
        df = df.copy()
        
        # Convert Date to datetime and extract temporal features
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['DayOfYear'] = df['Date'].dt.dayofyear
            df['Quarter'] = df['Date'].dt.quarter
        
        # 1. Lag features (previous day values)
        for col in self.feature_cols:
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_lag7'] = df[col].shift(7)
            df[f'{col}_lag30'] = df[col].shift(30)
        
        # 2. Rolling statistics
        for col in self.feature_cols:
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
        df['Temp_Rainfall'] = df['Temperature_C'] * df['Rainfall_mm']
        df['pH_DO'] = df['pH'] * df['Dissolved_Oxygen_mg_L']
        df['Temp_pH'] = df['Temperature_C'] * df['pH']
        df['Rainfall_pH'] = df['Rainfall_mm'] * df['pH']
        
        # 5. Polynomial features
        df['Temperature_sq'] = df['Temperature_C'] ** 2
        df['Rainfall_sq'] = df['Rainfall_mm'] ** 2
        df['pH_sq'] = df['pH'] ** 2
        df['DO_sq'] = df['Dissolved_Oxygen_mg_L'] ** 2
        
        # 6. Water level trend features (if available)
        if self.target_col in df.columns:
            df['Water_Level_lag1'] = df[self.target_col].shift(1)
            df['Water_Level_lag7'] = df[self.target_col].shift(7)
            df['Water_Level_diff'] = df[self.target_col].diff()
            df['Water_Level_rolling_mean_7'] = df[self.target_col].rolling(window=7).mean()
            df['Water_Level_rolling_std_7'] = df[self.target_col].rolling(window=7).std()
        
        # 7. Weather patterns
        df['Rainfall_cumulative_7'] = df['Rainfall_mm'].rolling(window=7).sum()
        df['Rainfall_cumulative_30'] = df['Rainfall_mm'].rolling(window=30).sum()
        df['Dry_days'] = (df['Rainfall_mm'] == 0).astype(int).rolling(window=7).sum()
        df['Heavy_rain'] = (df['Rainfall_mm'] > df['Rainfall_mm'].quantile(0.8)).astype(int)
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of all engineered feature columns"""
        exclude_cols = ['Date', self.target_col]
        feature_columns = [col for col in df.columns 
                          if col not in exclude_cols and not df[col].isna().all()]
        return feature_columns
    
    def prepare_for_prediction(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """Prepare data for prediction with proper feature alignment"""
        # Create features
        df_processed = self.create_features(df)
        
        # Ensure all required features exist
        for col in feature_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0.0  # Default value for missing features
        
        # Select only the required features in correct order
        X = df_processed[feature_columns].values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        return X