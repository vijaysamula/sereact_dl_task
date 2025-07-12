# Updated coordinator/main.py

import asyncio
import uuid
from fastapi import FastAPI, BackgroundTasks
from coordinator.coordinator import Coordinator
from shared.models import InferenceRequest, TaskType

app = FastAPI(title="AI Inference Coordinator - Load Balancing Mode")
coordinator = Coordinator()

@app.on_event("startup")
async def startup_event():
    # Wait for workers to start
    await asyncio.sleep(10)
    
    # Register all workers as universal (capable of all task types)
    await coordinator.register_worker("worker-1", "http://worker-1:8000")
    await coordinator.register_worker("worker-2", "http://worker-2:8000")
    await coordinator.register_worker("worker-3", "http://worker-3:8000")
    
    print("🚀 Load Balancing Mode Enabled")
    print("📋 All workers support all task types:")
    print("   - Text Classification (DistilBERT)")
    print("   - Image Classification (MobileNetV2)")
    print("   - CLIP Text Similarity")
    print("   - CLIP Text-to-Image")
    print("⚖️  Tasks will be distributed based on worker load, not specialization")
    
    # Start background tasks
    asyncio.create_task(coordinator.health_check_workers())
    asyncio.create_task(coordinator.process_queue())

@app.post("/infer")
async def infer(request: InferenceRequest):
    """Main inference endpoint with load balancing"""
    if not request.request_id:
        request.request_id = str(uuid.uuid4())
    
    # Submit to queue and wait for result (load balancing will handle worker selection)
    response = await coordinator.submit_and_wait(request, timeout=30.0)
    return response

@app.post("/infer/async")
async def infer_async(request: InferenceRequest):
    """Async inference endpoint - returns immediately with request ID"""
    if not request.request_id:
        request.request_id = str(uuid.uuid4())
    
    # Add to queue without waiting (load balancing will handle distribution)
    await coordinator.queue_task(request, priority=0)
    
    return {
        "request_id": request.request_id,
        "status": "queued",
        "message": "Task queued for load-balanced processing"
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
                "task_type": str(task_info.get("task_type", "unknown")),
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
    """Get detailed queue and load balancing statistics"""
    return coordinator.get_queue_stats()

@app.get("/status")
async def get_status():
    """Get simplified system status with load balancing info"""
    stats = coordinator.get_queue_stats()
    return {
        "mode": "load_balancing",
        "description": "All workers support all task types",
        "queue_size": stats["queue_size"],
        "active_tasks": stats["active_tasks"],
        "system_load_percentage": stats["system_load_percentage"],
        "total_current_tasks": stats["total_current_tasks"],
        "total_capacity": stats["total_capacity"],
        "workers": {
            worker_id: {
                "status": worker_data["status"].status,
                "current_tasks": worker_data["status"].current_tasks,
                "max_tasks": worker_data["status"].max_tasks,
                "load_percentage": round((worker_data["status"].current_tasks / worker_data["status"].max_tasks) * 100, 1),
                "last_heartbeat": worker_data["status"].last_heartbeat,
                "avg_latency": worker_data["status"].avg_latency,
                "total_processed": worker_data["status"].total_processed,
                "capabilities": ["text_classification", "image_classification", "clip_text_similarity", "clip_text_to_image"]
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
    """Get detailed worker performance metrics for load balancing"""
    return {
        worker_id: {
            "status": worker_data["status"].status,
            "current_tasks": worker_data["status"].current_tasks,
            "max_tasks": worker_data["status"].max_tasks,
            "load_percentage": round((worker_data["status"].current_tasks / worker_data["status"].max_tasks) * 100, 1),
            "avg_latency": round(worker_data["status"].avg_latency, 3),
            "total_processed": worker_data["status"].total_processed,
            "load_score": round(coordinator.calculate_worker_load_score(worker_data), 2),
            "capabilities": ["text_classification", "image_classification", "clip_text_similarity", "clip_text_to_image"],
            "specialization": "universal"
        }
        for worker_id, worker_data in coordinator.workers.items()
    }

@app.get("/load-balancing/info")
async def get_load_balancing_info():
    """Get information about the load balancing strategy"""
    return {
        "strategy": "pure_load_balancing",
        "description": "Tasks are distributed to workers based on current load, not task type specialization",
        "worker_capabilities": {
            "worker-1": ["text_classification", "image_classification", "clip_text_similarity", "clip_text_to_image"],
            "worker-2": ["text_classification", "image_classification", "clip_text_similarity", "clip_text_to_image"],
            "worker-3": ["text_classification", "image_classification", "clip_text_similarity", "clip_text_to_image"]
        },
        "load_balancing_algorithm": {
            "type": "lowest_load_first",
            "factors": [
                "current_task_count",
                "worker_capacity",
                "average_latency",
                "worker_health_status"
            ]
        },
        "benefits": [
            "Better resource utilization",
            "Automatic load distribution",
            "No single point of failure for task types",
            "Elastic scaling capability"
        ]
    }

@app.post("/priority/infer")
async def priority_infer(request: InferenceRequest, priority: int = 10):
    """High priority inference endpoint - still uses load balancing"""
    if not request.request_id:
        request.request_id = str(uuid.uuid4())
    
    # Submit with higher priority (load balancing will still apply)
    response = await coordinator.submit_and_wait(request, timeout=30.0)
    return response

@app.get("/metrics/task-distribution")
async def get_task_distribution():
    """Get statistics on task type distribution across workers"""
    stats = coordinator.get_queue_stats()
    return {
        "task_type_distribution": stats["stats"]["task_type_distribution"],
        "total_requests": stats["stats"]["total_requests"],
        "successful_requests": stats["stats"]["successful_requests"],
        "failed_requests": stats["stats"]["failed_requests"],
        "worker_load_distribution": {
            worker_id: {
                "current_load": worker_detail["current_tasks"],
                "capacity": worker_detail["max_tasks"],
                "utilization_percentage": worker_detail["load_percentage"]
            }
            for worker_id, worker_detail in stats["workers_detail"].items()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)