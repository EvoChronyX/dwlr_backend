"""
API Testing Script for Backend
Tests all endpoints and verifies model integration
"""
import requests
import json
import time
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: {data}")
            return True
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\n🔍 Testing Model Info Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model/info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model Info:")
            print(f"   Model Type: {data.get('model_type', 'Unknown')}")
            print(f"   Accuracy: {data.get('accuracy', 'N/A')}")
            print(f"   Status: {data.get('status', 'N/A')}")
            return True, data
        else:
            print(f"❌ Model Info Failed: {response.status_code}")
            return False, None
    except Exception as e:
        print(f"❌ Model Info Error: {e}")
        return False, None

def test_predictions():
    """Test predictions endpoint"""
    print("\n🔍 Testing Predictions Endpoint...")
    try:
        payload = {
            "location": "STN001",
            "months_ahead": 3,
            "rainfall_scenario": "normal",
            "confidence": 0.95
        }
        
        response = requests.post(
            f"{BASE_URL}/predictions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Predictions Test:")
            print(f"   Model Type: {data.get('model_type', 'Unknown')}")
            predictions = data.get('predictions', [])
            print(f"   Number of Predictions: {len(predictions)}")
            
            if predictions:
                first_pred = predictions[0]
                print(f"   Sample Prediction:")
                print(f"     Date: {first_pred.get('date', 'N/A')}")
                print(f"     Level: {first_pred.get('predicted_level', 'N/A')}")
                print(f"     Confidence: [{first_pred.get('lower_bound', 'N/A')} - {first_pred.get('upper_bound', 'N/A')}]")
            
            return True, data
        else:
            print(f"❌ Predictions Failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False, None
    except Exception as e:
        print(f"❌ Predictions Error: {e}")
        return False, None

def test_analysis():
    """Test analysis endpoints"""
    print("\n🔍 Testing Analysis Endpoints...")
    
    # Test water level analysis
    try:
        payload = {"location": "STN001", "year": 2020}
        response = requests.post(
            f"{BASE_URL}/analysis/water-level",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Water Level Analysis:")
            summary = data.get('summary', {})
            print(f"   Average Level: {summary.get('avg_level', 'N/A')}")
            print(f"   Min Level: {summary.get('min_level', 'N/A')}")
            print(f"   Max Level: {summary.get('max_level', 'N/A')}")
            return True
        else:
            print(f"❌ Analysis Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Analysis Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 BACKEND API TESTING")
    print("=" * 50)
    
    # Check if server is running
    print("⏳ Waiting for server to start...")
    time.sleep(2)
    
    # Test 1: Health
    health_ok = test_health()
    if not health_ok:
        print("\n❌ Backend server is not running or not responding!")
        print("📝 Start the backend with: python main.py")
        sys.exit(1)
    
    # Test 2: Model Info
    model_ok, model_data = test_model_info()
    
    # Test 3: Predictions
    pred_ok, pred_data = test_predictions()
    
    # Test 4: Analysis
    analysis_ok = test_analysis()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY:")
    print(f"✅ Health Check: {'PASS' if health_ok else 'FAIL'}")
    print(f"✅ Model Info: {'PASS' if model_ok else 'FAIL'}")
    print(f"✅ Predictions: {'PASS' if pred_ok else 'FAIL'}")
    print(f"✅ Analysis: {'PASS' if analysis_ok else 'FAIL'}")
    
    if all([health_ok, model_ok, pred_ok, analysis_ok]):
        print("\n🎉 ALL TESTS PASSED! Backend is working correctly!")
        
        if model_data:
            model_type = model_data.get('model_type', 'Unknown')
            if 'HIGH_ACCURACY' in model_type or 'ENSEMBLE' in model_type:
                print("🎯 YOUR HIGH-ACCURACY MODEL IS ACTIVE!")
            else:
                print("⚠️ Using fallback model - check model loading")
                
        print("\n📱 Your Flutter app can now connect to this backend!")
        print("🔗 Use this URL in MLService: http://127.0.0.1:8000")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
    
    return all([health_ok, model_ok, pred_ok, analysis_ok])

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)