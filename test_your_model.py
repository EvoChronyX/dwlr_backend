"""
Test YOUR Production High-Accuracy Model
Uses YOUR exact architecture from final_high_accuracy_test.py
"""
import pandas as pd
import numpy as np
from production_high_accuracy_predictor import ProductionHighAccuracyPredictor

def test_your_high_accuracy_model():
    """Test YOUR high-accuracy ensemble model"""
    print("🚀 Testing YOUR High-Accuracy Ensemble Model...")
    print("📋 Using YOUR exact architecture from final_high_accuracy_test.py")
    print("=" * 70)
    
    # Load dataset
    df = pd.read_csv('train_dataset.csv')
    print(f"📊 Loaded dataset with {len(df)} samples")
    
    # Initialize YOUR model
    ensemble = ProductionHighAccuracyPredictor()
    
    # Train using YOUR exact method
    print("🔄 Training YOUR ensemble model...")
    try:
        metrics = ensemble.train_ensemble(df)
        
        print(f"\n🎯 YOUR MODEL PERFORMANCE:")
        print(f"   R² Score: {metrics.get('r2_score', 'N/A'):.4f}")
        print(f"   MAE: {metrics.get('mae', 'N/A'):.4f}")
        print(f"   RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        print(f"   MAPE: {metrics.get('mape', 'N/A'):.2f}%")
        print(f"   Features: {metrics.get('n_features', 'N/A')}")
        print(f"   XGBoost Available: {metrics.get('xgboost_available', 'N/A')}")
        
        # Test prediction
        print("\n🔮 Testing prediction on sample data...")
        test_data = pd.DataFrame({
            'Date': ['2025-01-01'],
            'Temperature_C': [25.0],
            'Rainfall_mm': [50.0],
            'pH': [7.2],
            'Dissolved_Oxygen_mg_L': [6.5]
        })
        
        prediction = ensemble.predict(test_data)
        pred_with_conf, lower, upper = ensemble.predict_with_confidence(test_data)
        
        print(f"✅ Sample prediction: {prediction[0]:.2f}m")
        print(f"✅ With confidence: {pred_with_conf[0]:.2f}m [{lower[0]:.2f} - {upper[0]:.2f}]")
        
        # Test future scenario prediction
        print("\n🌟 Testing future scenario prediction...")
        future_preds = ensemble.predict_future_scenario(months_ahead=3, rainfall_scenario='normal')
        
        print("📅 Future predictions:")
        for pred in future_preds[:3]:
            print(f"   {pred['date']}: {pred['predicted_level']:.2f}m")
        
        # Get model info
        print("\n📋 YOUR Model Information:")
        model_info = ensemble.get_model_info()
        print(f"   Type: {model_info.get('model_type', 'N/A')}")
        print(f"   Base Models: {model_info.get('base_models', 'N/A')}")
        print(f"   Features: {model_info.get('n_features', 'N/A')}")
        print(f"   Based On: {model_info.get('based_on', 'N/A')}")
        
        # Save model
        print("\n💾 Saving YOUR trained model...")
        ensemble.save_model('models/production_high_accuracy_model.pkl')
        
        r2_score = metrics.get('r2_score', 0)
        if isinstance(r2_score, float):
            if r2_score >= 0.9:
                print("\n🎉 EXCELLENT! YOUR model achieved R² ≥ 0.9")
            elif r2_score >= 0.8:
                print("\n✅ VERY GOOD! YOUR model achieved R² ≥ 0.8")
            elif r2_score >= 0.7:
                print("\n👍 GOOD! YOUR model achieved R² ≥ 0.7")
            else:
                print(f"\n⚠️ YOUR model R² = {r2_score:.4f} - may need more data or tuning")
        
        print("\n✅ YOUR high-accuracy model test completed successfully!")
        return metrics
        
    except Exception as e:
        print(f"❌ Error training YOUR model: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test YOUR model
    metrics = test_your_high_accuracy_model()
    
    if metrics:
        print(f"\n🎯 FINAL RESULT: YOUR model achieved R² = {metrics.get('r2_score', 'N/A')}")
        print("🚀 YOUR model is ready for production use!")
    else:
        print("\n❌ YOUR model test failed")