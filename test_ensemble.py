"""
Test script for the high-accuracy ensemble model
"""
from high_accuracy_predictor import HighAccuracyEnsemblePredictor
import pandas as pd
import numpy as np

def test_ensemble_model():
    print("🚀 Testing High-Accuracy Ensemble Model...")
    
    # Load training data
    df = pd.read_csv('train_dataset.csv')
    print(f"📊 Loaded dataset with {len(df)} samples")
    
    # Initialize ensemble
    ensemble = HighAccuracyEnsemblePredictor()
    
    # Train the model
    print("🔄 Training ensemble model...")
    metrics = ensemble.train_ensemble(df)
    
    print(f"\n✅ Training completed!")
    print(f"📈 Performance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    # Test prediction
    print("\n🔮 Testing prediction...")
    test_data = pd.DataFrame({
        'Date': ['2024-01-15'],
        'Temperature_C': [25.0],
        'Rainfall_mm': [45.0],
        'pH': [7.2],
        'Dissolved_Oxygen_mg_L': [6.8]
    })
    
    prediction = ensemble.predict(test_data)
    print(f"🎯 Sample prediction: {prediction[0]:.2f} meters")
    
    # Test confidence prediction
    pred, lower, upper = ensemble.predict_with_confidence(test_data)
    print(f"🎯 With confidence: {pred[0]:.2f} m (95% CI: {lower[0]:.2f} - {upper[0]:.2f})")
    
    # Save model
    ensemble.save_model('models/ensemble_model.pkl')
    print("💾 Model saved successfully!")
    
    return metrics

if __name__ == "__main__":
    metrics = test_ensemble_model()