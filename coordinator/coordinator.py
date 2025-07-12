import asyncio
import time
import uuid
import logging
import heapq
import math
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException
import httpx
from shared.models import InferenceRequest, InferenceResponse, WorkerStatus, TaskType, QueuedTask
from shared.logging_setup import task_logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Coordinator:
    def __init__(self):
        self.workers: Dict[str, dict] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.active_tasks: Dict[str, dict] = {}
        self.client = httpx.AsyncClient(timeout=30.0)
        self.queue_processor_running = False
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "queue_high_watermark": 0,
            "task_type_distribution": {
                "text_classification": 0,
                "image_classification": 0,
                "clip_text_similarity": 0,
                "clip_text_to_image": 0
            }
        }
    
    def make_json_safe(self, obj):
        """Make data structure JSON-safe by replacing inf/nan values"""
        if isinstance(obj, dict):
            return {k: self.make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_json_safe(item) for item in obj]
        elif isinstance(obj, float):
            if math.isinf(obj):
                return 999999.0 if obj > 0 else -999999.0
            elif math.isnan(obj):
                return 0.0
            else:
                return obj
        else:
            return obj
        
    async def register_worker(self, worker_id: str, worker_url: str):
        """Register a new worker - all workers are now universal"""
        self.workers[worker_id] = {
            "status": WorkerStatus(
                worker_id=worker_id,
                status="starting",
                last_heartbeat=time.time(),
                model_loaded=False,
                current_tasks=0,
                max_tasks=5,
                avg_latency=0.0,
                total_processed=0
            ),
            "url": worker_url
        }
        logger.info(f"Worker {worker_id} registered at {worker_url} (Universal: supports all task types)")
        task_logger.log_worker_status(worker_id, "starting", 0, {
            "url": worker_url, 
            "capability": "universal_all_models"
        })
        
    async def health_check_workers(self):
        """Periodically check worker health and update status"""
        while True:
            await asyncio.sleep(5)
            
            for worker_id, worker_data in self.workers.items():
                try:
                    worker_url = worker_data["url"]
                    response = await self.client.get(f"{worker_url}/health", timeout=5.0)
                    
                    if response.status_code == 200:
                        health_data = response.json()
                        status = worker_data["status"]
                        
                        # Update worker status based on load
                        if status.current_tasks >= status.max_tasks:
                            new_status = "busy"
                        elif status.current_tasks > 0:
                            new_status = "active"
                        else:
                            new_status = "healthy"
                        
                        if status.status != new_status:
                            task_logger.log_worker_status(worker_id, new_status, status.current_tasks)
                            
                        status.status = new_status
                        status.last_heartbeat = time.time()
                        status.model_loaded = health_data.get("models_loaded", False)
                        
                        if worker_data["status"].status == "starting" and status.model_loaded:
                            logger.info(f"Worker {worker_id} is now healthy and ready (Universal capability)")
                            status.status = "healthy"
                            task_logger.log_worker_status(worker_id, "healthy", 0, {"all_models_ready": True})
                        
                    else:
                        if worker_data["status"].status != "error":
                            task_logger.log_worker_status(worker_id, "error", worker_data["status"].current_tasks, 
                                                        {"health_check_status": response.status_code})
                        worker_data["status"].status = "error"
                        
                except Exception as e:
                    if worker_data["status"].status != "error":
                        task_logger.log_worker_status(worker_id, "error", worker_data["status"].current_tasks, 
                                                    {"error": str(e)})
                    worker_data["status"].status = "error"
                    
            # Log system stats
            healthy_workers = len([w for w in self.workers.values() if w["status"].status in ["healthy", "active"]])
            task_logger.log_queue_stats(self.task_queue.qsize(), len(self.active_tasks), healthy_workers)
                    
            await asyncio.sleep(10)

    def calculate_worker_load_score(self, worker_data: dict) -> float:
        """Calculate worker load score for pure load balancing (lower = better)"""
        status = worker_data["status"]
        
        if status.status not in ["healthy", "active"]:
            return 999999.0  # ✅ Use large number instead of float('inf')
        
        # Base load score (0-100)
        load_percentage = (status.current_tasks / max(status.max_tasks, 1)) * 100
        
        # Performance factor (lower latency = better score)
        latency_penalty = status.avg_latency * 10 if status.avg_latency > 0 else 0
        
        # Combined score (lower is better)
        total_score = load_percentage + latency_penalty
        
        return total_score
    
    async def get_best_worker_by_load(self) -> Optional[Tuple[str, dict]]:
        """Get the best available worker based purely on current load"""
        available_workers = [
            (worker_id, worker_data) for worker_id, worker_data in self.workers.items() 
            if worker_data["status"].status in ["healthy", "active"] and 
               worker_data["status"].current_tasks < worker_data["status"].max_tasks
        ]
        
        if not available_workers:
            return None
        
        # Sort by load score (lowest first = least loaded)
        scored_workers = [
            (self.calculate_worker_load_score(worker_data), worker_id, worker_data)
            for worker_id, worker_data in available_workers
        ]
        
        scored_workers.sort()  # Lowest score first (least loaded)
        
        if scored_workers[0][0] < 999999.0:  # ✅ Check against large number instead of float('inf')
            worker_id, worker_data = scored_workers[0][1], scored_workers[0][2]
            logger.debug(f"Selected worker {worker_id} for load balancing (score: {scored_workers[0][0]:.2f})")
            return worker_id, worker_data
        
        return None
    
    async def queue_task(self, request: InferenceRequest, priority: int = 0) -> str:
        """Queue a task for processing"""
        task = QueuedTask(
            request=request,
            priority=priority,
            queued_at=time.time(),
            retries=0,
            max_retries=3
        )
        
        # Track task type distribution
        task_type_str = str(request.task_type).replace("TaskType.", "").lower()
        if task_type_str in self.stats["task_type_distribution"]:
            self.stats["task_type_distribution"][task_type_str] += 1
        
        # Use negative priority for max-heap behavior
        await self.task_queue.put((-priority, time.time(), task))
        
        # Update queue stats
        current_queue_size = self.task_queue.qsize()
        if current_queue_size > self.stats["queue_high_watermark"]:
            self.stats["queue_high_watermark"] = current_queue_size
        
        logger.info(f"Queued {request.task_type} task {request.request_id} (priority: {priority}, queue: {current_queue_size})")
        return request.request_id
        
    async def process_queue(self):
        """Main queue processing loop - pure load balancing"""
        if self.queue_processor_running:
            return
            
        self.queue_processor_running = True
        logger.info("Queue processor started (Load Balancing Mode)")
        
        try:
            while True:
                try:
                    # Get task from queue
                    try:
                        _, _, task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    
                    # Find best worker based on load (not task type)
                    worker_result = await self.get_best_worker_by_load()
                    if not worker_result:
                        # No workers available, put task back
                        await asyncio.sleep(0.5)
                        await self.task_queue.put((-task.priority, time.time(), task))
                        continue
                    
                    worker_id, worker_data = worker_result
                    
                    # Log load balancing decision
                    logger.info(f"Load balancing: assigning {task.request.task_type} to {worker_id} "
                              f"(load: {worker_data['status'].current_tasks}/{worker_data['status'].max_tasks})")
                    
                    # Process the task
                    asyncio.create_task(self.execute_task(task, worker_id, worker_data))
                    
                except Exception as e:
                    logger.error(f"Error in queue processor: {str(e)}")
                    await asyncio.sleep(1)
                    
        except asyncio.CancelledError:
            logger.info("Queue processor stopped")
        finally:
            self.queue_processor_running = False
    
    async def execute_task(self, task: QueuedTask, worker_id: str, worker_data: dict):
        """Execute a task on a specific worker"""
        request = task.request
        worker_url = worker_data["url"]
        status = worker_data["status"]
        
        # Log task assignment with load balancing context
        task_logger.log_task_assignment(
            request.request_id, worker_id, str(request.task_type), task.retries
        )
        
        # Mark worker as more loaded
        status.current_tasks += 1
        if status.current_tasks >= status.max_tasks:
            status.status = "busy"
            task_logger.log_worker_status(worker_id, "busy", status.current_tasks)
        elif status.current_tasks > 0:
            status.status = "active"
        
        # Track active task
        self.active_tasks[request.request_id] = {
            "worker_id": worker_id,
            "started_at": time.time(),
            "task": task,
            "task_type": request.task_type
        }
        
        try:
            start_time = time.time()
            
            # Send request to worker
            logger.info(f"Executing {request.task_type} task {request.request_id} on worker {worker_id}")
            response = await self.client.post(
                f"{worker_url}/infer",
                json=request.dict(),
                timeout=20.0
            )
            
            end_time = time.time()
            latency = end_time - start_time
            queue_wait_time = start_time - task.queued_at
            
            if response.status_code == 200:
                result = response.json()
                
                # Update worker performance stats
                status.total_processed += 1
                if status.avg_latency == 0:
                    status.avg_latency = latency
                else:
                    # Exponential moving average
                    status.avg_latency = 0.8 * status.avg_latency + 0.2 * latency
                
                self.stats["successful_requests"] += 1
                
                # Log successful completion
                task_logger.log_task_completion(
                    request.request_id, worker_id, latency, task.retries, True
                )
                
                logger.info(f"Task {request.request_id} ({request.task_type}) completed on {worker_id} "
                          f"(latency: {latency:.3f}s, queue_wait: {queue_wait_time:.3f}s)")
                
                # Store result
                self.active_tasks[request.request_id]["result"] = InferenceResponse(
                    request_id=request.request_id,
                    worker_id=worker_id,
                    result=result,
                    latency=latency,
                    retry_count=task.retries
                )
                
            else:
                try:
                    error_text = await response.text()
                except:
                    error_text = f"HTTP {response.status_code}"
                raise Exception(f"Worker returned {response.status_code}: {error_text}")
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Task {request.request_id} ({request.task_type}) failed on worker {worker_id}: {error_msg}")
            
            # Log failure
            task_logger.log_task_completion(
                request.request_id, worker_id, time.time() - start_time, task.retries, False, error_msg
            )
            
            # Handle retry logic
            task.retries += 1
            if task.retries < task.max_retries:
                task_logger.log_task_retry(request.request_id, worker_id, task.retries, error_msg)
                logger.info(f"Retrying task {request.request_id} (attempt {task.retries + 1}/{task.max_retries})")
                # Put back in queue with higher priority
                await self.task_queue.put((-10, time.time(), task))
            else:
                logger.error(f"Task {request.request_id} failed permanently after {task.retries} retries")
                self.stats["failed_requests"] += 1
                
                # Store error result
                self.active_tasks[request.request_id]["result"] = HTTPException(
                    status_code=500, 
                    detail=f"Task failed permanently: {error_msg}"
                )
        
        finally:
            # Update worker status
            status.current_tasks = max(0, status.current_tasks - 1)
            if status.current_tasks == 0:
                status.status = "healthy"
            elif status.current_tasks < status.max_tasks:
                status.status = "active"
            
            task_logger.log_worker_status(worker_id, status.status, status.current_tasks)
    
    async def submit_and_wait(self, request: InferenceRequest, timeout: float = 30.0) -> InferenceResponse:
        """Submit a task and wait for completion"""
        self.stats["total_requests"] += 1
        
        # Add to queue
        await self.queue_task(request, priority=0)
        
        # Wait for completion
        start_wait = time.time()
        while time.time() - start_wait < timeout:
            if request.request_id in self.active_tasks:
                task_info = self.active_tasks[request.request_id]
                if "result" in task_info:
                    result = task_info["result"]
                    # Clean up
                    del self.active_tasks[request.request_id]
                    
                    if isinstance(result, HTTPException):
                        raise result
                    return result
            
            await asyncio.sleep(0.1)
        
        # Timeout
        if request.request_id in self.active_tasks:
            del self.active_tasks[request.request_id]
        
        self.stats["failed_requests"] += 1
        raise HTTPException(status_code=408, detail="Request timed out")
    
    def get_queue_stats(self) -> dict:
        """Get comprehensive queue and load balancing statistics"""
        healthy_workers = len([w for w in self.workers.values() if w["status"].status in ["healthy", "active"]])
        busy_workers = len([w for w in self.workers.values() if w["status"].status == "busy"])
        error_workers = len([w for w in self.workers.values() if w["status"].status == "error"])
        
        # Calculate load distribution
        total_current_tasks = sum(w["status"].current_tasks for w in self.workers.values())
        total_capacity = sum(w["status"].max_tasks for w in self.workers.values())
        system_load_percentage = (total_current_tasks / total_capacity * 100) if total_capacity > 0 else 0
        
        result = {
            "queue_size": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "total_workers": len(self.workers),
            "healthy_workers": healthy_workers,
            "busy_workers": busy_workers,
            "error_workers": error_workers,
            "system_load_percentage": round(system_load_percentage, 1),
            "total_current_tasks": total_current_tasks,
            "total_capacity": total_capacity,
            "stats": self.stats.copy(),
            "load_balancing_mode": "pure_load_based",
            "workers_detail": {
                worker_id: {
                    "status": worker_data["status"].status,
                    "current_tasks": worker_data["status"].current_tasks,
                    "max_tasks": worker_data["status"].max_tasks,
                    "load_percentage": round((worker_data["status"].current_tasks / worker_data["status"].max_tasks) * 100, 1),
                    "avg_latency": round(worker_data["status"].avg_latency, 3),
                    "total_processed": worker_data["status"].total_processed,
                    "load_score": round(self.calculate_worker_load_score(worker_data), 2)
                }
                for worker_id, worker_data in self.workers.items()
            }
        }
        
        # ✅ Make the entire result JSON-safe
        return self.make_json_safe(result)