"""
Comprehensive Groundwater Monitoring System
========================================

This module provides a comprehensive interface for groundwater monitoring, including:
1. Water level trends and recharge pattern visualization
2. Real-time groundwater availability estimation
3. Decision support features
4. Future predictions with uncertainty estimates
5. Export capabilities for research and analysis

Author: GitHub Copilot
Date: 2025-09-27
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import RobustScaler
from scipy import stats
import os
import json

# Try to import XGBoost, but handle the case where it might not be available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Some prediction features will be limited.")
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class GroundwaterMonitoringSystem:
    """Comprehensive system for groundwater monitoring and prediction"""
    
    def __init__(self, data_file: str = 'train_dataset.csv'):
        """Initialize the monitoring system"""
        self.data_file = data_file
        self.df = None
        self.predictor = None
        self.feature_cols = ['Temperature_C', 'Rainfall_mm', 'pH', 'Dissolved_Oxygen_mg_L']
        self.target_col = 'Groundwater_Level_m'  # Updated to match training data
        self.rainfall_categories = {
            'Very Low': (0, 5),
            'Low': (5, 25),
            'Moderate': (25, 75),
            'High': (75, 150),
            'Very High': (150, float('inf'))
        }
        
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the dataset"""
        print("📊 Loading dataset...")
        self.df = pd.read_csv(self.data_file)
        
        # Convert Date column to datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Sort by date
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Remove rows with missing values
        self.df = self.df.dropna(subset=self.feature_cols + [self.target_col])
        
        print(f"✅ Dataset loaded: {len(self.df)} samples")
        print(f"📅 Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        
        return self.df
    
    def load_predictor(self, model_path: str = 'high_accuracy_water_predictor.pkl'):
        """Load the trained prediction model"""
        if not XGBOOST_AVAILABLE:
            print("⚠️ Warning: XGBoost not available. Using fallback prediction method.")
            self.predictor = self._create_fallback_predictor()
            return
            
        try:
            from predict_water_level import WaterLevelPredictor
            
            print("🤖 Loading prediction model...")
            self.predictor = WaterLevelPredictor()
            self.predictor.load_model(model_path)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading model: {str(e)}")
            print("Using fallback prediction method...")
            self.predictor = self._create_fallback_predictor()
    
    def _create_fallback_predictor(self):
        """Create an advanced seasonal predictor when XGBoost is not available"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        
        class AdvancedSeasonalPredictor:
            def __init__(self):
                self.historical_data = None
                self.seasonal_patterns = None
                self.rainfall_impact = None
                self.temperature_impact = None
                self.is_trained = False
                
                # Store monsoon period information
                self.monsoon_months = [6, 7, 8, 9]  # June to September
                self.pre_monsoon_months = [3, 4, 5]  # March to May
                self.post_monsoon_months = [10, 11, 12, 1, 2]  # October to February
                
                # Random Forest model for predictions with available features
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                
                # Model for rainfall impact estimation
                self.rainfall_model = LinearRegression()
                
            def train(self, historical_df):
                """Train the advanced seasonal predictor with historical data"""
                print("🔄 Training advanced seasonal predictor...")
                self.historical_data = historical_df.copy()
                
                # Calculate seasonal patterns (monthly averages)
                self.seasonal_patterns = self.historical_data.groupby(
                    self.historical_data['Date'].dt.month
                )['Water_Level_m'].mean().to_dict()
                
                # Calculate seasonal standard deviations
                self.seasonal_std = self.historical_data.groupby(
                    self.historical_data['Date'].dt.month
                )['Water_Level_m'].std().to_dict()
                
                # Calculate rainfall impact
                # Create a dataframe with extracted year and month
                temp_df = self.historical_data.copy()
                temp_df['Year'] = temp_df['Date'].dt.year
                temp_df['Month'] = temp_df['Date'].dt.month
                
                # Group by year and month to get monthly data
                monthly_data = temp_df.groupby(['Year', 'Month']).agg({
                    'Rainfall_mm': 'sum',
                    'Water_Level_m': 'mean',
                    'Temperature_C': 'mean'
                }).reset_index()
                
                # Calculate lag features (previous month's water level)
                monthly_data['Prev_Water_Level'] = monthly_data['Water_Level_m'].shift(1)
                monthly_data['Rainfall_Monthly'] = monthly_data['Rainfall_mm']
                
                # Calculate 3-month cumulative rainfall
                monthly_data['Rainfall_3Month'] = monthly_data['Rainfall_mm'].rolling(3).sum()
                
                # Remove NaN values
                monthly_data = monthly_data.dropna()
                
                # Separate data by season for more accurate impact assessment
                monsoon_data = monthly_data[monthly_data['Month'].isin(self.monsoon_months)]
                pre_monsoon_data = monthly_data[monthly_data['Month'].isin(self.pre_monsoon_months)]
                
                # Calculate rainfall impact coefficients for different seasons
                # For monsoon period
                if len(monsoon_data) > 5:  # Need enough data points
                    X_monsoon = monsoon_data[['Rainfall_Monthly', 'Prev_Water_Level']]
                    y_monsoon = monsoon_data['Water_Level_m']
                    self.rainfall_model_monsoon = LinearRegression().fit(X_monsoon, y_monsoon)
                    self.rainfall_coef_monsoon = self.rainfall_model_monsoon.coef_[0]
                else:
                    self.rainfall_coef_monsoon = 0.02  # Fallback: 2% increase per 100mm rainfall
                
                # For pre-monsoon period
                if len(pre_monsoon_data) > 5:
                    X_pre = pre_monsoon_data[['Rainfall_Monthly', 'Prev_Water_Level']]
                    y_pre = pre_monsoon_data['Water_Level_m']
                    self.rainfall_model_pre = LinearRegression().fit(X_pre, y_pre)
                    self.rainfall_coef_pre = self.rainfall_model_pre.coef_[0]
                else:
                    self.rainfall_coef_pre = 0.01  # Fallback: 1% increase per 100mm rainfall
                
                # Calculate overall impact (for general use)
                X = monthly_data[['Rainfall_Monthly', 'Prev_Water_Level']]
                y = monthly_data['Water_Level_m']
                self.rainfall_model = LinearRegression().fit(X, y)
                self.rainfall_coef = self.rainfall_model.coef_[0]
                
                # Train Random Forest model if we have enough features
                feature_cols = ['Rainfall_mm', 'Temperature_C']
                if all(col in self.historical_data.columns for col in feature_cols):
                    # Add month as a feature
                    X = self.historical_data[feature_cols].copy()
                    X['Month'] = self.historical_data['Date'].dt.month
                    X['DayOfYear'] = self.historical_data['Date'].dt.dayofyear
                    
                    # Target is water level
                    y = self.historical_data['Water_Level_m']
                    
                    # Train model
                    self.model.fit(X, y)
                    self.model_features = X.columns.tolist()
                    self.is_trained = True
                    print("✅ Advanced seasonal predictor trained successfully!")
                else:
                    print("⚠️ Not enough features for full model. Using seasonal patterns only.")
                
                return self
                
            def predict(self, df):
                """Make predictions for future dates"""
                # If we don't have historical data, try to use the dataframe as historical data
                if self.historical_data is None and 'Water_Level_m' in df.columns:
                    self.train(df)
                
                # Create prediction dataframe
                predictions = []
                
                if 'Date' in df.columns:
                    # Use advanced seasonal prediction with rainfall impact
                    dates = df['Date'].tolist()
                    
                    # Try to use model-based prediction first if trained and features available
                    if self.is_trained:
                        try:
                            feature_cols = [col for col in self.model_features if col in df.columns]
                            
                            # Prepare features - create a copy to avoid modifying original
                            X = pd.DataFrame()
                            
                            # Copy available features from input dataframe
                            for col in feature_cols:
                                if col in df.columns:
                                    X[col] = df[col]
                            
                            # Add Month column if needed
                            if 'Month' not in X.columns and 'Date' in df.columns:
                                X['Month'] = df['Date'].dt.month
                                
                            # Add DayOfYear column if needed
                            if 'DayOfYear' not in X.columns and 'Date' in df.columns:
                                X['DayOfYear'] = df['Date'].dt.dayofyear
                                
                            # Make prediction
                            return self.model.predict(X)
                        except Exception as e:
                            print(f"⚠️ Model prediction failed: {str(e)}. Using seasonal patterns.")
                    
                    # Use seasonal patterns with rainfall adjustments
                    for i, date in enumerate(dates):
                        month = date.month
                        
                        # Get base prediction from seasonal pattern
                        base_prediction = self.seasonal_patterns.get(month, 5.0)
                        
                        # Apply rainfall adjustment if available
                        rainfall_adjustment = 0
                        
                        if 'Rainfall_mm' in df.columns:
                            rainfall = df['Rainfall_mm'].iloc[i]
                            
                            # Different coefficients for different seasons
                            if month in self.monsoon_months:
                                rainfall_impact = rainfall * self.rainfall_coef_monsoon
                            elif month in self.pre_monsoon_months:
                                rainfall_impact = rainfall * self.rainfall_coef_pre
                            else:
                                rainfall_impact = rainfall * self.rainfall_coef
                                
                            rainfall_adjustment = rainfall_impact
                            
                        # Get similar months from historical data for better prediction
                        if self.historical_data is not None:
                            similar_month_data = self.historical_data[
                                self.historical_data['Date'].dt.month == month
                            ]
                            
                            if len(similar_month_data) > 0:
                                # Use the average of similar months as base
                                month_avg = similar_month_data['Water_Level_m'].mean()
                                month_std = similar_month_data['Water_Level_m'].std()
                                
                                # Create prediction with seasonal adjustment
                                adjusted_prediction = month_avg + rainfall_adjustment
                                predictions.append(adjusted_prediction)
                                continue
                        
                        # If we don't have similar months data, use base with adjustment
                        predictions.append(base_prediction + rainfall_adjustment)
                    
                    return np.array(predictions)
                else:
                    # Without dates, use seasonal means with adjustments where possible
                    predictions = []
                    
                    for i in range(len(df)):
                        # Use overall mean water level as a base
                        if self.historical_data is not None:
                            base_prediction = self.historical_data['Water_Level_m'].mean()
                        else:
                            base_prediction = 5.5  # Fallback average
                        
                        # Apply rainfall adjustment if available
                        if 'Rainfall_mm' in df.columns:
                            rainfall = df['Rainfall_mm'].iloc[i]
                            rainfall_impact = rainfall * self.rainfall_coef
                            base_prediction += rainfall_impact
                        
                        predictions.append(base_prediction)
                    
                    return np.array(predictions)
            
            def get_rainfall_scenarios(self):
                """Generate impact scenarios for different rainfall levels"""
                if self.historical_data is None:
                    return {
                        'very_low': {'change': -0.3, 'description': "10% decrease in water level"},
                        'low': {'change': -0.1, 'description': "3% decrease in water level"},
                        'normal': {'change': 0.0, 'description': "No significant change"},
                        'high': {'change': 0.2, 'description': "5% increase in water level"},
                        'very_high': {'change': 0.5, 'description': "12% increase in water level"}
                    }
                
                # Calculate average monthly rainfall and water level
                monthly_avg_rain = self.historical_data.groupby(
                    self.historical_data['Date'].dt.month
                )['Rainfall_mm'].mean()
                
                monthly_avg_wl = self.historical_data.groupby(
                    self.historical_data['Date'].dt.month
                )['Water_Level_m'].mean()
                
                # Calculate impact percentages for different rainfall scenarios
                avg_rainfall = monthly_avg_rain.mean()
                avg_water_level = monthly_avg_wl.mean()
                
                # Ensure positive coefficient for rainfall impact
                impact_coef = abs(self.rainfall_coef)
                
                # Calculate changes based on rainfall coefficient
                very_low_change = -0.3  # Significant decrease
                low_change = -0.15      # Moderate decrease
                high_change = 0.2       # Moderate increase
                very_high_change = 0.4  # Significant increase
                
                # Calculate impact percentages
                base = avg_water_level
                very_low_pct = (very_low_change / base) * 100
                low_pct = (low_change / base) * 100
                high_pct = (high_change / base) * 100
                very_high_pct = (very_high_change / base) * 100
                
                return {
                    'very_low': {
                        'change': very_low_change,
                        'description': f"{abs(very_low_pct):.1f}% decrease in water level"
                    },
                    'low': {
                        'change': low_change,
                        'description': f"{abs(low_pct):.1f}% decrease in water level"
                    },
                    'normal': {
                        'change': 0.0,
                        'description': "No significant change"
                    },
                    'high': {
                        'change': high_change,
                        'description': f"{high_pct:.1f}% increase in water level"
                    },
                    'very_high': {
                        'change': very_high_change,
                        'description': f"{very_high_pct:.1f}% increase in water level"
                    }
                }
                
        # Create and train the predictor with the available data
        predictor = AdvancedSeasonalPredictor()
        
        # If we have data already loaded, train the predictor
        if self.df is not None:
            predictor.train(self.df)
            
        return predictor
    
    def analyze_water_level_trends(self, start_date: Optional[str] = None, 
                                 end_date: Optional[str] = None, 
                                 save_path: Optional[str] = None) -> Dict:
        """
        Analyze water level trends for a specific period
        Returns statistics and generates visualization
        """
        if self.df is None:
            raise ValueError("Please load data first using load_data()")
        
        # Filter data by date range if provided
        df = self.df.copy()
        if start_date:
            df = df[df['Date'] >= start_date]
        if end_date:
            df = df[df['Date'] <= end_date]
        
        # Calculate statistics
        stats = {
            'mean_level': df[self.target_col].mean(),
            'std_level': df[self.target_col].std(),
            'min_level': df[self.target_col].min(),
            'max_level': df[self.target_col].max(),
            'trend': np.polyfit(range(len(df)), df[self.target_col], 1)[0],
            'seasonal_pattern': df.groupby(df['Date'].dt.month)[self.target_col].mean().to_dict()
        }
        
        # Create visualization
        fig = plt.figure(figsize=(15, 10))
        
        # Water level time series
        ax1 = plt.subplot(2, 2, 1)
        plt.plot(df['Date'], df[self.target_col], linewidth=1.5, alpha=0.8)
        plt.title('Water Level Time Series', fontsize=12, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Water Level (m)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Monthly patterns
        ax2 = plt.subplot(2, 2, 2)
        monthly_stats = df.groupby(df['Date'].dt.month)[self.target_col].agg(['mean', 'std'])
        plt.errorbar(monthly_stats.index, monthly_stats['mean'], 
                    yerr=monthly_stats['std'], marker='o', capsize=5)
        plt.title('Monthly Water Level Patterns', fontsize=12, fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Water Level (m)')
        plt.grid(True, alpha=0.3)
        
        # Water level distribution
        ax3 = plt.subplot(2, 2, 3)
        sns.histplot(data=df, x=self.target_col, kde=True)
        plt.axvline(stats['mean_level'], color='red', linestyle='--', 
                   label=f'Mean: {stats["mean_level"]:.2f}m')
        plt.title('Water Level Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Water Level (m)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Trend analysis
        ax4 = plt.subplot(2, 2, 4)
        x = range(len(df))
        y = df[self.target_col].values
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(df['Date'], y, alpha=0.5, label='Actual')
        plt.plot(df['Date'], p(x), "r--", label=f'Trend: {z[0]:.4f} m/day')
        plt.title('Water Level Trend Analysis', fontsize=12, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Water Level (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved to: {save_path}")
        
        plt.close()
        
        return stats
    
    def analyze_recharge_patterns(self, save_path: Optional[str] = None) -> Dict:
        """
        Analyze groundwater recharge patterns based on rainfall and water level changes
        Returns statistics and generates visualization
        """
        if self.df is None:
            raise ValueError("Please load data first using load_data()")
        
        df = self.df.copy()
        
        # Calculate water level changes
        df['WL_Change'] = df[self.target_col].diff()
        
        # Categorize rainfall
        df['Rainfall_Category'] = pd.cut(
            df['Rainfall_mm'],
            bins=[-float('inf')] + [v[1] for v in self.rainfall_categories.values()],
            labels=self.rainfall_categories.keys()
        )
        
        # Calculate statistics
        stats = {
            'total_rainfall': df['Rainfall_mm'].sum(),
            'avg_monthly_rainfall': df.groupby(df['Date'].dt.month)['Rainfall_mm'].mean().to_dict(),
            'recharge_by_rainfall': df.groupby('Rainfall_Category')['WL_Change'].agg(['mean', 'count']).to_dict(),
            'correlation': df['Rainfall_mm'].corr(df['WL_Change'])
        }
        
        # Create visualization
        fig = plt.figure(figsize=(15, 10))
        
        # Rainfall vs Water Level Change
        ax1 = plt.subplot(2, 2, 1)
        plt.scatter(df['Rainfall_mm'], df['WL_Change'], alpha=0.5)
        plt.title('Rainfall vs Water Level Change', fontsize=12, fontweight='bold')
        plt.xlabel('Rainfall (mm)')
        plt.ylabel('Water Level Change (m)')
        plt.grid(True, alpha=0.3)
        
        # Average recharge by rainfall category
        ax2 = plt.subplot(2, 2, 2)
        recharge_by_cat = df.groupby('Rainfall_Category')['WL_Change'].mean()
        plt.bar(recharge_by_cat.index, recharge_by_cat.values)
        plt.title('Average Recharge by Rainfall Category', fontsize=12, fontweight='bold')
        plt.xlabel('Rainfall Category')
        plt.ylabel('Average Water Level Change (m)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Monthly rainfall pattern
        ax3 = plt.subplot(2, 2, 3)
        monthly_rain = df.groupby(df['Date'].dt.month)['Rainfall_mm'].mean()
        plt.bar(monthly_rain.index, monthly_rain.values)
        plt.title('Monthly Rainfall Pattern', fontsize=12, fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Average Rainfall (mm)')
        plt.grid(True, alpha=0.3)
        
        # Cumulative rainfall effect
        ax4 = plt.subplot(2, 2, 4)
        df['Cum_Rain_30d'] = df['Rainfall_mm'].rolling(window=30).sum()
        plt.scatter(df['Cum_Rain_30d'], df['WL_Change'], alpha=0.5)
        plt.title('30-Day Cumulative Rainfall vs Water Level Change', 
                 fontsize=12, fontweight='bold')
        plt.xlabel('30-Day Cumulative Rainfall (mm)')
        plt.ylabel('Water Level Change (m)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved to: {save_path}")
        
        plt.close()
        
        return stats
    
    def estimate_current_availability(self) -> Dict:
        """
        Estimate current groundwater availability based on recent measurements
        Returns availability metrics and status
        """
        if self.df is None:
            raise ValueError("Please load data first using load_data()")
        
        # Get most recent data
        recent_data = self.df.iloc[-30:]  # Last 30 days
        current_level = recent_data[self.target_col].iloc[-1]
        
        # Calculate statistics
        historical_mean = self.df[self.target_col].mean()
        historical_std = self.df[self.target_col].std()
        percentile = stats.percentileofscore(self.df[self.target_col], current_level)
        
        # Calculate trend
        recent_trend = np.polyfit(range(len(recent_data)), 
                                recent_data[self.target_col], 1)[0]
        
        # Determine status
        if current_level > historical_mean + historical_std:
            status = "Above Normal"
        elif current_level < historical_mean - historical_std:
            status = "Below Normal"
        else:
            status = "Normal"
        
        # Calculate sustainability index
        sustainability_score = min(100, max(0, 50 + (
            ((current_level - historical_mean) / historical_std) * 25 +
            (recent_trend / abs(historical_std)) * 25
        )))
        
        return {
            'current_level': current_level,
            'historical_mean': historical_mean,
            'historical_std': historical_std,
            'percentile': percentile,
            'recent_trend': recent_trend,
            'status': status,
            'sustainability_score': sustainability_score
        }
    
    def predict_future_levels(self, months_ahead: int = 3, 
                            confidence_level: float = 0.95,
                            rainfall_scenario: str = 'normal') -> Dict:
        """
        Predict future water levels with uncertainty estimates
        Returns predictions for start, middle, and end of each month
        
        Parameters:
        -----------
        months_ahead: int
            Number of months to predict ahead
        confidence_level: float
            Confidence level for prediction intervals (0.0-1.0)
        rainfall_scenario: str
            Rainfall scenario to use: 'very_low', 'low', 'normal', 'high', 'very_high'
        """
        if self.predictor is None:
            raise ValueError("Please load predictor first using load_predictor()")
        
        # Get last date in dataset
        last_date = self.df['Date'].max()
        
        # Generate future dates
        future_dates = []
        for month in range(1, months_ahead + 1):
            # Start of month
            future_dates.append(
                (last_date + pd.DateOffset(months=month)).replace(day=1)
            )
            # Middle of month
            future_dates.append(
                (last_date + pd.DateOffset(months=month)).replace(day=15)
            )
            # End of month
            future_dates.append(
                (last_date + pd.DateOffset(months=month + 1)).replace(day=1) - 
                pd.DateOffset(days=1)
            )
        
        # Create future feature data based on past values for the same months
        future_data = []
        future_months = [date.month for date in future_dates]
        
        for i, date in enumerate(future_dates):
            # Get data for the same month from previous years (last 2-3 years)
            target_month = date.month
            recent_years = self.df['Date'].dt.year.unique()[-3:]  # Last 3 years
            
            # Filter data for the same month in recent years
            same_month_data = self.df[
                (self.df['Date'].dt.month == target_month) & 
                (self.df['Date'].dt.year.isin(recent_years))
            ]
            
            if len(same_month_data) > 0:
                # Use average of same month in recent years
                month_features = same_month_data[self.feature_cols].mean()
            else:
                # Fallback: use average for this month across all years
                month_features = self.df[self.df['Date'].dt.month == target_month][self.feature_cols].mean()
                
                # If still no data, use overall average
                if month_features.isna().any():
                    month_features = self.df[self.feature_cols].mean()
            
            future_data.append(month_features)
        
        # Create DataFrame with dates and features
        future_df = pd.DataFrame(future_data, columns=self.feature_cols)
        future_df['Date'] = future_dates
        
        # Add Month column needed for prediction
        if 'Month' not in future_df.columns and 'Month' in self.feature_cols:
            future_df['Month'] = future_df['Date'].dt.month
        
        # Generate base predictions
        predictions = self.predictor.predict(future_df)
        
        # Calculate confidence intervals using historical error distribution
        try:
            historical_predictions = self.predictor.predict(self.df[self.feature_cols])
            prediction_errors = self.df[self.target_col].values - historical_predictions
            error_std = prediction_errors.std()
        except:
            # Fallback to using standard deviation of water levels
            error_std = self.df[self.target_col].std() * 0.5  # 50% of water level std
        
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * error_std
        
        # Get rainfall scenario adjustments
        rainfall_scenarios = {}
        if hasattr(self.predictor, 'get_rainfall_scenarios'):
            rainfall_scenarios = self.predictor.get_rainfall_scenarios()
        else:
            # Default rainfall scenarios if not available from predictor
            rainfall_scenarios = {
                'very_low': {'change': -0.3, 'description': "10% decrease in water level"},
                'low': {'change': -0.1, 'description': "3% decrease in water level"},
                'normal': {'change': 0.0, 'description': "No significant change"},
                'high': {'change': 0.2, 'description': "5% increase in water level"},
                'very_high': {'change': 0.5, 'description': "12% increase in water level"}
            }
        
        # Get rainfall scenario adjustment info
        scenario_info = rainfall_scenarios.get(rainfall_scenario, {'change': 0.0, 'description': 'No impact'})
        scenario_adjustment = scenario_info['change']
        
        # Organize results
        results = []
        for date, prediction in zip(future_dates, predictions):
            # Apply rainfall scenario adjustment with a higher impact on monsoon months
            month = date.month
            
            # Apply stronger effect in monsoon months
            if month in [6, 7, 8, 9]:  # June to September
                adjusted_prediction = prediction + (scenario_adjustment * 1.5)
            else:
                adjusted_prediction = prediction + scenario_adjustment
            
            results.append({
                'date': date.strftime('%Y-%m-%d'),
                'predicted_level': adjusted_prediction,
                'lower_bound': adjusted_prediction - margin_of_error,
                'upper_bound': adjusted_prediction + margin_of_error
            })
        
        # Group results by month for easier analysis
        monthly_results = {}
        for item in results:
            month = pd.to_datetime(item['date']).month
            if month not in monthly_results:
                monthly_results[month] = []
            monthly_results[month].append(item)
        
        # Prepare result summary and rain impact scenarios
        scenario_descriptions = {
            scenario: info['description'] for scenario, info in rainfall_scenarios.items()
        }
        
        return {
            'predictions': results,
            'monthly_predictions': monthly_results,
            'confidence_level': confidence_level,
            'margin_of_error': margin_of_error,
            'rainfall_scenario': rainfall_scenario,
            'rainfall_scenarios': scenario_descriptions
        }
    
    def categorize_rainfall_scenario(self, rainfall: float) -> str:
        """Categorize rainfall amount into scenarios"""
        for category, (lower, upper) in self.rainfall_categories.items():
            if lower <= rainfall < upper:
                return category
        return "Very High"
    
    def export_analysis(self, analysis_type: str, data: Dict, 
                       output_dir: str = 'analysis_outputs') -> str:
        """
        Export analysis results to various formats
        Returns path to exported file
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if analysis_type == "water_level_trends":
            # Export water level trend analysis
            output_file = f"{output_dir}/water_level_trends_{timestamp}"
            
            # Save statistics as JSON
            with open(f"{output_file}.json", 'w') as f:
                json.dump(data, f, indent=4)
            
            # Create CSV with monthly patterns
            monthly_data = pd.DataFrame.from_dict(
                data['seasonal_pattern'], 
                orient='index', 
                columns=['water_level']
            )
            monthly_data.to_csv(f"{output_file}.csv")
            
            return f"{output_file}.json"
        
        elif analysis_type == "recharge_patterns":
            # Export recharge pattern analysis
            output_file = f"{output_dir}/recharge_patterns_{timestamp}"
            
            # Save statistics as JSON
            with open(f"{output_file}.json", 'w') as f:
                json.dump(data, f, indent=4)
            
            return f"{output_file}.json"
        
        elif analysis_type == "future_predictions":
            # Export future predictions
            output_file = f"{output_dir}/future_predictions_{timestamp}"
            
            # Save predictions as CSV
            pd.DataFrame(data['predictions']).to_csv(f"{output_file}.csv", index=False)
            
            # Save full results as JSON
            with open(f"{output_file}.json", 'w') as f:
                json.dump(data, f, indent=4)
            
            return f"{output_file}.json"
        
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    def generate_decision_support_report(self, output_path: Optional[str] = None) -> Dict:
        """
        Generate comprehensive decision support report for researchers and policymakers
        Returns report data and optionally saves to file
        """
        if self.df is None:
            raise ValueError("Please load data first using load_data()")
        
        # Current status
        current_status = self.estimate_current_availability()
        
        # Historical analysis
        historical_stats = self.analyze_water_level_trends()
        
        # Recharge analysis
        recharge_stats = self.analyze_recharge_patterns()
        
        # Future predictions
        future_predictions = self.predict_future_levels(months_ahead=3)
        
        # Generate recommendations
        recommendations = []
        
        # Water level recommendations
        if current_status['status'] == "Below Normal":
            recommendations.append(
                "⚠️ Water levels are below normal. Consider implementing water conservation measures."
            )
        elif current_status['status'] == "Above Normal":
            recommendations.append(
                "✅ Water levels are above normal. Opportunity for sustainable extraction."
            )
        
        # Trend-based recommendations
        if current_status['recent_trend'] < 0:
            recommendations.append(
                "📉 Declining trend detected. Monitor extraction rates and implement recharge measures."
            )
        
        # Sustainability recommendations
        if current_status['sustainability_score'] < 50:
            recommendations.append(
                "⚠️ Low sustainability score. Review and adjust groundwater management practices."
            )
        
        # Compile report
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'current_status': current_status,
            'historical_analysis': historical_stats,
            'recharge_analysis': recharge_stats,
            'future_predictions': future_predictions,
            'recommendations': recommendations
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=4)
            print(f"✅ Decision support report saved to: {output_path}")
        
        return report

def main():
    """Example usage of the Groundwater Monitoring System"""
    print("🌊 Groundwater Monitoring System Demo")
    print("=" * 50)
    
    # Initialize system
    gms = GroundwaterMonitoringSystem('train_dataset.csv')
    
    # Load data and model
    gms.load_data()
    gms.load_predictor()
    
    # Create output directory
    output_dir = "groundwater_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Analyze water level trends
    print("\n📊 Analyzing water level trends...")
    trends = gms.analyze_water_level_trends(
        save_path=f"{output_dir}/water_level_trends.png"
    )
    
    # 2. Analyze recharge patterns
    print("\n💧 Analyzing recharge patterns...")
    recharge = gms.analyze_recharge_patterns(
        save_path=f"{output_dir}/recharge_patterns.png"
    )
    
    # 3. Check current availability
    print("\n🔍 Checking current availability...")
    availability = gms.estimate_current_availability()
    print(f"Status: {availability['status']}")
    print(f"Sustainability Score: {availability['sustainability_score']:.1f}/100")
    
    # 4. Generate future predictions
    print("\n🔮 Generating future predictions...")
    predictions = gms.predict_future_levels(months_ahead=3)
    
    # 5. Generate decision support report
    print("\n📋 Generating decision support report...")
    report = gms.generate_decision_support_report(
        output_path=f"{output_dir}/decision_support_report.json"
    )
    
    print("\n✅ Demo completed! Check the 'groundwater_analysis_results' directory for outputs.")

if __name__ == "__main__":
    main()