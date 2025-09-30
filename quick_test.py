"""
Simple Backend Test
"""
import requests
import json

def quick_test():
    """Quick test of the backend"""
    BASE_URL = "http://127.0.0.1:8000"
    
    print("🔍 Testing Backend...")
    
    # Test health
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"✅ Health: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Model Status: {data.get('model_status', 'unknown')}")
    except Exception as e:
        print(f"❌ Health failed: {e}")
        return False
    
    # Test water level analysis
    try:
        payload = {"location": "STN001", "year": 2020}
        response = requests.post(
            f"{BASE_URL}/analysis/water-level",
            json=payload,
            timeout=10
        )
        print(f"✅ Water Level Analysis: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            summary = data.get('summary', {})
            print(f"   Average Level: {summary.get('avg_level', 'N/A')}")
    except Exception as e:
        print(f"❌ Water level analysis failed: {e}")
    
    return True

if __name__ == "__main__":
    quick_test()