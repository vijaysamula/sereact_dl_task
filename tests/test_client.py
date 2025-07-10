import asyncio
import httpx
import time
from shared.models import InferenceRequest, TaskType

async def test_inference():
    """Simple test client"""
    client = httpx.AsyncClient()
    
    # Test request
    request = InferenceRequest(
        request_id="test-123",
        task_type=TaskType.TEXT_CLASSIFICATION,
        data="I love this movie!"
    )
    
    try:
        response = await client.post(
            "http://localhost:8000/infer",
            json=request.dict()
        )
        print(f"Response: {response.json()}")
        
        # Check status
        status = await client.get("http://localhost:8000/status")
        print(f"Status: {status.json()}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.aclose()

if __name__ == "__main__":
    asyncio.run(test_inference())