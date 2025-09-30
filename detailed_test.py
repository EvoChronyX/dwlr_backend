"""
Detailed Backend Test
"""
import requests
import json

def detailed_test():
    """Detailed test of the backend endpoints"""
    BASE_URL = "http://127.0.0.1:8000"
    
    print("🔍 Detailed Backend Testing...")
    
    # Test water level analysis
    try:
        payload = {"location": "STN001", "year": 2020}
        response = requests.post(
            f"{BASE_URL}/analysis/water-level",
            json=payload,
            timeout=10
        )
        print(f"\n📊 Water Level Analysis Response:")
        data = response.json()
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"❌ Water level analysis failed: {e}")
    
    # Test recharge analysis
    try:
        payload = {"location": "STN001", "year": 2020}
        response = requests.post(
            f"{BASE_URL}/analysis/recharge",
            json=payload,
            timeout=10
        )
        print(f"\n💧 Recharge Analysis Response:")
        data = response.json()
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"❌ Recharge analysis failed: {e}")
    
    # Test report generation
    try:
        payload = {"location": "STN001", "year": 2020}
        response = requests.post(
            f"{BASE_URL}/analysis/report",
            json=payload,
            timeout=10
        )
        print(f"\n📋 Report Generation Response:")
        data = response.json()
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"❌ Report generation failed: {e}")

if __name__ == "__main__":
    detailed_test()