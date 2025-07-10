import asyncio
import uuid
from fastapi import FastAPI, BackgroundTasks
from coordinator.coordinator import SmartCoordinator
from shared.models import InferenceRequest, TaskType

app = FastAPI(title="Smart AI Inference Coordinator")
coordinator = SmartCoordinator()

@app.on_event("startup")
async def startup_event():
    # Wait a bit for workers to start
    await asyncio.sleep(10)
    
    # Register workers with correct Docker network URLs
    await coordinator.register_worker("worker-1", "http://worker-1:8000")
    await coordinator.register_worker("worker-2", "http://worker-2:8000")
    await coordinator.register_worker("worker-3", "http://worker-3:8000")
    
    # Start background tasks
    asyncio.create_task(coordinator.health_check_workers())
    asyncio.create_task(coordinator.process_queue())

@app.post("/infer")
async def infer(request: InferenceRequest):
    """Main inference endpoint with intelligent queuing"""
    if not request.request_id:
        request.request_id = str(uuid.uuid4())
    
    # Submit to queue and wait for result
    response = await coordinator.submit_and_wait(request, timeout=30.0)
    return response

@app.post("/infer/async")
async def infer_async(request: InferenceRequest):
    """Async inference endpoint - returns immediately with request ID"""
    if not request.request_id:
        request.request_id = str(uuid.uuid4())
    
    # Add to queue without waiting
    await coordinator.queue_task(request, priority=0)
    
    return {
        "request_id": request.request_id,
        "status": "queued",
        "message": "Task queued for processing"
    }

@app.get("/result/{request_id}")
async def get_result(request_id: str):
    """Get result of an async inference request"""
    if request_id in coordinator.active_tasks:
        task_info = coordinator.active_tasks[request_id]
        if "result" in task_info:
            result = task_info["result"]
            # Clean up after retrieval
            del coordinator.active_tasks[request_id]
            return result
        else:
            return {
                "request_id": request_id,
                "status": "processing",
                "worker_id": task_info.get("worker_id"),
                "started_at": task_info.get("started_at")
            }
    else:
        return {
            "request_id": request_id,
            "status": "not_found",
            "message": "Request not found or already completed"
        }

@app.get("/queue/stats")
async def get_queue_stats():
    """Get detailed queue and worker statistics"""
    return coordinator.get_queue_stats()

@app.get("/status")
async def get_status():
    """Get simplified system status"""
    stats = coordinator.get_queue_stats()
    return {
        "queue_size": stats["queue_size"],
        "active_tasks": stats["active_tasks"],
        "workers": {
            worker_id: {
                "status": worker_data["status"].status,
                "current_tasks": worker_data["status"].current_tasks,
                "last_heartbeat": worker_data["status"].last_heartbeat,
                "model_loaded": worker_data["status"].model_loaded,
                "avg_latency": worker_data["status"].avg_latency,
                "total_processed": worker_data["status"].total_processed
            }
            for worker_id, worker_data in coordinator.workers.items()
        },
        "total_workers": stats["total_workers"],
        "healthy_workers": stats["healthy_workers"],
        "busy_workers": stats["busy_workers"],
        "stats": stats["stats"]
    }

@app.get("/workers/performance")
async def get_worker_performance():
    """Get detailed worker performance metrics"""
    return {
        worker_id: {
            "status": worker_data["status"].status,
            "current_tasks": worker_data["status"].current_tasks,
            "max_tasks": worker_data["status"].max_tasks,
            "avg_latency": round(worker_data["status"].avg_latency, 3),
            "total_processed": worker_data["status"].total_processed,
            "performance_score": round(coordinator.calculate_worker_score(worker_data), 2),
            "load_percentage": round((worker_data["status"].current_tasks / worker_data["status"].max_tasks) * 100, 1)
        }
        for worker_id, worker_data in coordinator.workers.items()
    }

@app.post("/priority/infer")
async def priority_infer(request: InferenceRequest, priority: int = 10):
    """High priority inference endpoint"""
    if not request.request_id:
        request.request_id = str(uuid.uuid4())
    
    # Submit with higher priority
    response = await coordinator.submit_and_wait(request, timeout=30.0)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)