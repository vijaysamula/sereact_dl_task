import requests
import json
import time

def test_coordinator():
    base_url = "http://localhost:8000"
    
    print("Testing coordinator...")
    
    # Test status endpoint first
    try:
        response = requests.get(f"{base_url}/status")
        print(f"Status response: {response.status_code}")
        if response.status_code == 200:
            status_data = response.json()
            print(f"Workers: {status_data.get('total_workers', 0)}")
            print(f"Healthy workers: {status_data.get('healthy_workers', 0)}")
            print("Worker details:")
            for worker_id, details in status_data.get('workers', {}).items():
                print(f"  {worker_id}: {details['status']} (tasks: {details['current_tasks']})")
        else:
            print(f"Status check failed: {response.text}")
            return
    except Exception as e:
        print(f"Failed to connect to coordinator: {e}")
        return

    # Wait a bit for workers to be healthy
    print("\nWaiting for workers to be healthy...")
    time.sleep(5)
    
    # Test inference
    test_request = {
        "request_id": "test-123",
        "task_type": "text_classification",
        "data": "I love this amazing product!"
    }
    
    print(f"\nSending inference request: {test_request}")
    
    try:
        response = requests.post(
            f"{base_url}/infer",
            json=test_request,
            timeout=30
        )
        
        print(f"Inference response: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Result: {json.dumps(result, indent=2)}")
        else:
            print(f"Inference failed: {response.text}")
            
    except Exception as e:
        print(f"Inference request failed: {e}")

if __name__ == "__main__":
    test_coordinator()