import requests
import json

def test_worker_directly():
    """Test worker directly"""
    print("Testing worker directly...")
    
    worker_url = "http://localhost:8001"  # worker-1
    
    # Test health
    try:
        response = requests.get(f"{worker_url}/health")
        print(f"Worker health: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Worker health failed: {e}")
        return
    
    # Test inference directly on worker
    test_request = {
        "request_id": "direct-test-123",
        "task_type": "text_classification",
        "data": "I love this product!"
    }
    
    try:
        response = requests.post(
            f"{worker_url}/infer",
            json=test_request,
            timeout=30
        )
        
        print(f"Direct worker inference: {response.status_code}")
        if response.status_code == 200:
            print(f"Success: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Direct worker test failed: {e}")

def test_coordinator_debug():
    """Test coordinator debug endpoint"""
    print("\nTesting coordinator debug...")
    
    try:
        response = requests.get("http://localhost:8000/debug")
        print(f"Debug response: {response.status_code}")
        if response.status_code == 200:
            debug_data = response.json()
            print(f"Debug info: {json.dumps(debug_data, indent=2, default=str)}")
        else:
            print(f"Debug failed: {response.text}")
    except Exception as e:
        print(f"Debug request failed: {e}")

if __name__ == "__main__":
    test_worker_directly()
    test_coordinator_debug()