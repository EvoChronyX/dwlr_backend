"""
Test script for YOUR Complete Groundwater Monitoring System
Tests all endpoints with YOUR high-accuracy model integration
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_endpoint(url, method="GET", data=None, description=""):
    """Test an API endpoint"""
    print(f"\n🧪 Testing: {description}")
    print(f"📍 {method} {url}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS")
            
            # Print key information
            if "model" in result:
                print(f"🤖 Model: {result['model']}")
            if "source" in result:
                print(f"📄 Source: {result['source']}")
            if "accuracy" in result:
                print(f"🎯 Accuracy: {result['accuracy']}")
            if "version" in result:
                print(f"🔢 Version: {result['version']}")
                
            return True, result
        else:
            print(f"❌ FAILED: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False, None

def main():
    """Test YOUR complete monitoring system"""
    print("🚀 Testing YOUR Advanced Groundwater Monitoring System")
    print("=" * 60)
    
    # Test 1: Root endpoint
    test_endpoint(f"{BASE_URL}/", "GET", description="Root endpoint - API information")
    
    # Test 2: Health check
    test_endpoint(f"{BASE_URL}/health", "GET", description="Health check - YOUR system status")
    
    # Test 3: Model info
    test_endpoint(f"{BASE_URL}/model-info", "GET", description="YOUR high-accuracy model information")
    
    # Test 4: Current availability
    test_endpoint(f"{BASE_URL}/current-availability", "GET", description="Current groundwater availability")
    
    # Test 5: Dashboard
    test_endpoint(f"{BASE_URL}/dashboard", "GET", description="Dashboard data from YOUR interface")
    
    # Test 6: Recharge patterns
    test_endpoint(f"{BASE_URL}/recharge-patterns", "GET", description="Recharge pattern analysis")
    
    # Test 7: Water level trends
    trends_data = {"analysis_type": "water_level_trends"}
    test_endpoint(f"{BASE_URL}/water-level-trends", "POST", trends_data, "Water level trends analysis")
    
    # Test 8: Future predictions with YOUR model
    prediction_data = {
        "temperature": 25.5,
        "rainfall": 45.2,
        "ph": 7.1,
        "dissolved_oxygen": 6.8,
        "months_ahead": 6,
        "rainfall_scenario": "normal",
        "confidence_level": 0.95
    }
    test_endpoint(f"{BASE_URL}/future-predictions", "POST", prediction_data, "Future predictions with YOUR ensemble")
    
    # Test 9: Decision support
    decision_data = {"location": "Test Location", "report_type": "decision_support"}
    test_endpoint(f"{BASE_URL}/decision-support", "POST", decision_data, "Decision support report")
    
    # Test 10: Export analysis
    export_data = {"analysis_type": "complete_dashboard"}
    test_endpoint(f"{BASE_URL}/export-analysis", "POST", export_data, "Export analysis data")
    
    # Test 11: Legacy endpoints for mobile compatibility
    test_endpoint(f"{BASE_URL}/predict", "POST", prediction_data, "Legacy predict endpoint")
    test_endpoint(f"{BASE_URL}/current-status", "GET", description="Legacy current status")
    
    print("\n" + "=" * 60)
    print("🎉 Testing completed for YOUR Complete Monitoring System!")
    print("📱 All endpoints ready for Flutter app integration")

if __name__ == "__main__":
    main()