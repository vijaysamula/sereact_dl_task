import os
import time
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from worker.worker import Worker
from shared.models import InferenceRequest

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get worker ID from environment or default
WORKER_ID = os.getenv("WORKER_ID", "worker-1")

app = FastAPI(title=f"AI Worker {WORKER_ID}")
worker = Worker(WORKER_ID)

@app.post("/infer")
async def infer(request: InferenceRequest):
    """Process inference request"""
    try:
        logger.info(f"Worker {WORKER_ID}: Received inference request {request.request_id}")
        start_time = time.time()
        result = await worker.process_request(request)
        end_time = time.time()
        
        response_data = {
            "request_id": request.request_id,
            "result": result,
            "processing_time": end_time - start_time,
            "worker_id": WORKER_ID
        }
        
        logger.info(f"Worker {WORKER_ID}: Successfully responded to request {request.request_id}")
        return response_data
        
    except Exception as e:
        logger.error(f"Worker {WORKER_ID}: Failed to process request {request.request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "worker_id": WORKER_ID,
        "models_loaded": len(worker.models) > 0
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)