import asyncio
import time
import uuid
import logging
import heapq
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException
import httpx
from shared.models import InferenceRequest, InferenceResponse, WorkerStatus, TaskType, QueuedTask
from shared.logging_setup import task_logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Coordinator:
    def __init__(self):
        self.workers: Dict[str, dict] = {}  # worker_id -> {status: WorkerStatus, url: str}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.active_tasks: Dict[str, dict] = {}  # request_id -> task info
        self.client = httpx.AsyncClient(timeout=30.0)
        self.queue_processor_running = False
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "queue_high_watermark": 0
        }
        
    async def register_worker(self, worker_id: str, worker_url: str):
        """Register a new worker"""
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
        logger.info(f"Worker {worker_id} registered at {worker_url}")
        task_logger.log_worker_status(worker_id, "starting", 0, {"url": worker_url})
        
    async def health_check_workers(self):
        """Periodically check worker health and update status"""
        while True:
            await asyncio.sleep(5)  # Wait before starting checks
            
            for worker_id, worker_data in self.workers.items():
                try:
                    worker_url = worker_data["url"]
                    response = await self.client.get(f"{worker_url}/health", timeout=5.0)
                    
                    if response.status_code == 200:
                        health_data = response.json()
                        status = worker_data["status"]
                        
                        # Update worker status based on health and current load
                        if status.current_tasks >= status.max_tasks:
                            new_status = "busy"
                        elif status.current_tasks > 0:
                            new_status = "healthy"
                        else:
                            new_status = "healthy"
                        
                        # Log status changes
                        if status.status != new_status:
                            task_logger.log_worker_status(worker_id, new_status, status.current_tasks)
                            
                        status.status = new_status
                        status.last_heartbeat = time.time()
                        status.model_loaded = health_data.get("models_loaded", False)
                        
                        if worker_data["status"].status == "starting" and status.model_loaded:
                            logger.info(f"Worker {worker_id} is now healthy and ready")
                            status.status = "healthy"
                            task_logger.log_worker_status(worker_id, "healthy", 0, {"models_ready": True})
                        
                    else:
                        if worker_data["status"].status != "error":
                            task_logger.log_worker_status(worker_id, "error", worker_data["status"].current_tasks, 
                                                        {"health_check_status": response.status_code})
                        worker_data["status"].status = "error"
                        logger.warning(f"Worker {worker_id} health check failed: {response.status_code}")
                        
                except Exception as e:
                    if worker_data["status"].status != "error":
                        task_logger.log_worker_status(worker_id, "error", worker_data["status"].current_tasks, 
                                                    {"error": str(e)})
                    worker_data["status"].status = "error"
                    logger.warning(f"Worker {worker_id} health check failed: {str(e)}")
                    
            # Log system stats periodically
            healthy_workers = len([w for w in self.workers.values() if w["status"].status == "healthy"])
            task_logger.log_queue_stats(self.task_queue.qsize(), len(self.active_tasks), healthy_workers)
                    
            await asyncio.sleep(10)  # Check every 10 seconds

    def calculate_worker_score(self, worker_data: dict) -> float:
        """Calculate worker suitability score (higher = better)"""
        status = worker_data["status"]
        
        if status.status != "healthy":
            return 0.0
        
        # Base score
        score = 100.0
        
        # Penalize based on current load
        load_factor = status.current_tasks / max(status.max_tasks, 1)
        score -= load_factor * 50  # Heavy penalty for high load
        
        # Reward based on performance (lower latency = higher score)
        if status.avg_latency > 0:
            latency_penalty = min(status.avg_latency * 10, 30)  # Cap penalty at 30
            score -= latency_penalty
        
        # Small bonus for workers with more experience
        experience_bonus = min(status.total_processed * 0.1, 10)
        score += experience_bonus
        
        return max(score, 0.0)
            
    def get_worker_capabilities(self, worker_id: str) -> List[TaskType]:
        """Get the task types a worker can handle"""
        worker_capabilities = {
            "worker-1": [TaskType.TEXT_CLASSIFICATION],  # DistilBERT specialist
            "worker-2": [TaskType.IMAGE_CLASSIFICATION, TaskType.TEXT_CLASSIFICATION],  # MobileNet + fallback
            "worker-3": [TaskType.CLIP_TEXT_SIMILARITY, TaskType.CLIP_TEXT_TO_IMAGE, TaskType.TEXT_CLASSIFICATION]  # CLIP + fallback
        }
        return worker_capabilities.get(worker_id, [TaskType.TEXT_CLASSIFICATION])
    
    async def get_best_worker_for_task(self, task_type: TaskType) -> Optional[Tuple[str, dict]]:
        """Get the best available worker that can handle the specific task type"""
        # First, get workers that can handle this task type
        capable_workers = []
        for worker_id, worker_data in self.workers.items():
            if (worker_data["status"].status in ["healthy", "busy"] and 
                worker_data["status"].current_tasks < worker_data["status"].max_tasks and
                task_type in self.get_worker_capabilities(worker_id)):
                capable_workers.append((worker_id, worker_data))
        
        if not capable_workers:
            # If no specialist available, try any healthy worker (they all have text classification fallback)
            capable_workers = [
                (worker_id, worker_data) for worker_id, worker_data in self.workers.items() 
                if worker_data["status"].status in ["healthy", "busy"] and 
                   worker_data["status"].current_tasks < worker_data["status"].max_tasks
            ]
        
        if not capable_workers:
            return None
        
        # Sort by worker score (best first)
        scored_workers = [
            (self.calculate_worker_score(worker_data), worker_id, worker_data)
            for worker_id, worker_data in capable_workers
        ]
        
        scored_workers.sort(reverse=True)  # Highest score first
        
        if scored_workers[0][0] > 0:  # Make sure we have a viable worker
            return scored_workers[0][1], scored_workers[0][2]  # worker_id, worker_data
        
        return None
    async def get_best_worker(self) -> Optional[Tuple[str, dict]]:
        """Get the best available worker based on health, load, and performance (legacy method)"""
        available_workers = [
            (worker_id, worker_data) for worker_id, worker_data in self.workers.items() 
            if worker_data["status"].status in ["healthy", "busy"] and 
               worker_data["status"].current_tasks < worker_data["status"].max_tasks
        ]
        
        if not available_workers:
            return None
        
        # Sort by worker score (best first)
        scored_workers = [
            (self.calculate_worker_score(worker_data), worker_id, worker_data)
            for worker_id, worker_data in available_workers
        ]
        
        scored_workers.sort(reverse=True)  # Highest score first
        
        if scored_workers[0][0] > 0:  # Make sure we have a viable worker
            return scored_workers[0][1], scored_workers[0][2]  # worker_id, worker_data
        
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
        
        # Use negative priority for max-heap behavior (higher priority first)
        await self.task_queue.put((-priority, time.time(), task))
        
        # Update queue stats
        current_queue_size = self.task_queue.qsize()
        if current_queue_size > self.stats["queue_high_watermark"]:
            self.stats["queue_high_watermark"] = current_queue_size
        
        logger.info(f"Queued task {request.request_id} with priority {priority} (queue size: {current_queue_size})")
        return request.request_id
        
    async def process_queue(self):
        """Main queue processing loop"""
        if self.queue_processor_running:
            return
            
        self.queue_processor_running = True
        logger.info("Queue processor started")
        
        try:
            while True:
                try:
                    # Get task from queue (wait up to 1 second)
                    try:
                        _, _, task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue  # No tasks available, continue loop
                    
                    # Check if we have available workers for this specific task
                    worker_result = await self.get_best_worker_for_task(task.request.task_type)
                    if not worker_result:
                        # No capable workers available, put task back in queue with delay
                        await asyncio.sleep(0.5)
                        await self.task_queue.put((-task.priority, time.time(), task))
                        continue
                    
                    worker_id, worker_data = worker_result
                    
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
        
        # Log task assignment
        task_logger.log_task_assignment(
            request.request_id, worker_id, request.task_type, task.retries
        )
        
        # Mark worker as busy
        status.current_tasks += 1
        if status.current_tasks >= status.max_tasks:
            status.status = "busy"
            task_logger.log_worker_status(worker_id, "busy", status.current_tasks)
        
        # Track active task
        self.active_tasks[request.request_id] = {
            "worker_id": worker_id,
            "started_at": time.time(),
            "task": task
        }
        
        try:
            start_time = time.time()
            
            # Send request to worker
            logger.info(f"Executing task {request.request_id} on worker {worker_id}")
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
                
                logger.info(f"Task {request.request_id} completed successfully on {worker_id} "
                          f"(latency: {latency:.3f}s, queue_wait: {queue_wait_time:.3f}s)")
                
                # Store result for retrieval
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
            logger.error(f"Task {request.request_id} failed on worker {worker_id}: {error_msg}")
            
            # Log task failure
            task_logger.log_task_completion(
                request.request_id, worker_id, time.time() - start_time, task.retries, False, error_msg
            )
            
            # Handle retry logic
            task.retries += 1
            if task.retries < task.max_retries:
                task_logger.log_task_retry(request.request_id, worker_id, task.retries, error_msg)
                logger.info(f"Retrying task {request.request_id} (attempt {task.retries + 1}/{task.max_retries})")
                # Put back in queue with higher priority (retry)
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
            if status.current_tasks < status.max_tasks and status.status == "busy":
                status.status = "healthy"
                task_logger.log_worker_status(worker_id, "healthy", status.current_tasks)
    
    async def submit_and_wait(self, request: InferenceRequest, timeout: float = 30.0) -> InferenceResponse:
        """Submit a task and wait for its completion"""
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
            
            await asyncio.sleep(0.1)  # Check every 100ms
        
        # Timeout
        if request.request_id in self.active_tasks:
            del self.active_tasks[request.request_id]
        
        self.stats["failed_requests"] += 1
        raise HTTPException(status_code=408, detail="Request timed out")
    
    def get_queue_stats(self) -> dict:
        """Get queue and processing statistics"""
        healthy_workers = len([w for w in self.workers.values() if w["status"].status == "healthy"])
        busy_workers = len([w for w in self.workers.values() if w["status"].status == "busy"])
        error_workers = len([w for w in self.workers.values() if w["status"].status == "error"])
        
        return {
            "queue_size": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "total_workers": len(self.workers),
            "healthy_workers": healthy_workers,
            "busy_workers": busy_workers,
            "error_workers": error_workers,
            "stats": self.stats.copy(),
            "workers_detail": {
                worker_id: {
                    "status": worker_data["status"].status,
                    "current_tasks": worker_data["status"].current_tasks,
                    "avg_latency": worker_data["status"].avg_latency,
                    "total_processed": worker_data["status"].total_processed,
                    "score": self.calculate_worker_score(worker_data)
                }
                for worker_id, worker_data in self.workers.items()
            }
        }