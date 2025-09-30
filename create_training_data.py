"""
Sample Training Data Creator
Creates a representative training dataset for the high-accuracy ensemble model
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_sample_training_data(output_path: str = "train_dataset.csv", num_samples: int = 2000):
    """
    Create a realistic training dataset for groundwater level prediction
    """
    print(f"🔄 Creating sample training dataset with {num_samples} samples...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate date range (approximately 8 years of data)
    start_date = datetime(2015, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_samples)]
    
    data = []
    
    # Base parameters for realistic groundwater patterns
    base_level = 15.0  # Base groundwater level in meters
    seasonal_amplitude = 3.0  # Seasonal variation amplitude
    trend_slope = -0.001  # Slight declining trend per day
    
    for i, date in enumerate(dates):
        # Seasonal patterns
        day_of_year = date.timetuple().tm_yday
        seasonal_factor = seasonal_amplitude * np.sin(2 * np.pi * day_of_year / 365.25)
        
        # Long-term trend
        trend_factor = trend_slope * i
        
        # Temperature (seasonal with noise)
        temp_seasonal = 25 + 8 * np.sin(2 * np.pi * day_of_year / 365.25 - np.pi/2)
        temperature = temp_seasonal + np.random.normal(0, 2)
        temperature = max(5, min(45, temperature))  # Realistic bounds
        
        # Rainfall (seasonal with high variability)
        if 150 <= day_of_year <= 300:  # Monsoon season
            rainfall_base = 50 + 100 * np.random.exponential(0.5)
        else:  # Dry season
            rainfall_base = 5 + 20 * np.random.exponential(0.3)
        
        # Add occasional heavy rainfall events
        if np.random.random() < 0.05:  # 5% chance of heavy rain
            rainfall_base += np.random.exponential(100)
        
        rainfall = max(0, rainfall_base)
        
        # pH (relatively stable with slight variations)
        ph = 7.0 + np.random.normal(0, 0.3)
        ph = max(6.0, min(8.5, ph))
        
        # Dissolved Oxygen (temperature dependent with noise)
        do_base = 8.5 - 0.1 * (temperature - 20)  # Higher at lower temps
        dissolved_oxygen = do_base + np.random.normal(0, 0.5)
        dissolved_oxygen = max(3.0, min(12.0, dissolved_oxygen))
        
        # Groundwater level calculation (complex relationships)
        # Base level with seasonal and trend effects
        level = base_level + seasonal_factor + trend_factor
        
        # Rainfall impact (positive with lag effect)
        rainfall_effect = 0.02 * rainfall
        if i > 7:  # 7-day lag
            rainfall_effect += 0.01 * data[i-7]['Rainfall_mm']
        if i > 30:  # 30-day lag
            rainfall_effect += 0.005 * sum(d['Rainfall_mm'] for d in data[i-30:i]) / 30
        
        # Temperature impact (negative - higher temp = lower level)
        temp_effect = -0.05 * (temperature - 25)
        
        # pH impact (optimal around 7.0)
        ph_effect = -0.5 * abs(ph - 7.0)
        
        # DO impact (positive)
        do_effect = 0.1 * (dissolved_oxygen - 6.0)
        
        # Combine all effects
        level += rainfall_effect + temp_effect + ph_effect + do_effect
        
        # Add noise
        level += np.random.normal(0, 0.5)
        
        # Ensure realistic bounds
        level = max(5.0, min(25.0, level))
        
        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Temperature_C': round(temperature, 2),
            'Rainfall_mm': round(rainfall, 2),
            'pH': round(ph, 2),
            'Dissolved_Oxygen_mg_L': round(dissolved_oxygen, 2),
            'Groundwater_Level_m': round(level, 2)
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some data quality issues (realistic)
    # Missing values (1% of data)
    missing_indices = np.random.choice(len(df), size=int(0.01 * len(df)), replace=False)
    for idx in missing_indices:
        col = np.random.choice(['Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L'])
        df.loc[idx, col] = np.nan
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"✅ Training dataset created successfully!")
    print(f"📄 File: {output_path}")
    print(f"📊 Shape: {df.shape}")
    print(f"📈 Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"💧 Groundwater level range: {df['Groundwater_Level_m'].min():.2f} - {df['Groundwater_Level_m'].max():.2f} m")
    print(f"🌡️ Temperature range: {df['Temperature_C'].min():.2f} - {df['Temperature_C'].max():.2f} °C")
    print(f"🌧️ Rainfall range: {df['Rainfall_mm'].min():.2f} - {df['Rainfall_mm'].max():.2f} mm")
    
    return df

if __name__ == "__main__":
    # Create training dataset
    df = create_sample_training_data("train_dataset.csv", 2000)
    
    # Show basic statistics
    print("\n📊 Dataset Statistics:")
    print(df.describe())