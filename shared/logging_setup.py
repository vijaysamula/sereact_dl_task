# Updated shared/logging_setup.py

import logging
import json
import time
from typing import Dict, Any, Optional

class TaskLogger:
    """Centralized logging for all task operations with metadata"""
    
    def __init__(self, log_file: str = "logs/task_operations.log"):
        self.logger = logging.getLogger("task_logger")
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # File handler for task logs
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter with metadata
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_task_assignment(self, request_id: str, worker_id: str, task_type: str, 
                          retry_count: int = 0, extra_context: Optional[Dict[str, Any]] = None):
        """Log task assignment to worker with load balancing context"""
        metadata = {
            "event": "task_assigned",
            "request_id": request_id,
            "worker_id": worker_id,
            "task_type": task_type,
            "retry_count": retry_count,
            "timestamp": time.time()
        }
        
        if extra_context:
            metadata.update(extra_context)
            
        self.logger.info(f"TASK_ASSIGNED: {json.dumps(metadata)}")
    
    def log_task_completion(self, request_id: str, worker_id: str, latency: float, 
                          retry_count: int, success: bool, error: str = None,
                          extra_context: Optional[Dict[str, Any]] = None):
        """Log task completion with results and context"""
        metadata = {
            "event": "task_completed",
            "request_id": request_id,
            "worker_id": worker_id,
            "latency": round(latency, 3),
            "retry_count": retry_count,
            "success": success,
            "error": error,
            "timestamp": time.time()
        }
        
        if extra_context:
            metadata.update(extra_context)
            
        self.logger.info(f"TASK_COMPLETED: {json.dumps(metadata)}")
    
    def log_task_retry(self, request_id: str, worker_id: str, retry_count: int, reason: str):
        """Log task retry attempt"""
        metadata = {
            "event": "task_retry",
            "request_id": request_id,
            "worker_id": worker_id,
            "retry_count": retry_count,
            "reason": reason,
            "timestamp": time.time()
        }
        self.logger.info(f"TASK_RETRY: {json.dumps(metadata)}")
    
    def log_worker_status(self, worker_id: str, status: str, current_tasks: int, 
                         details: Dict[str, Any] = None):
        """Log worker status changes"""
        metadata = {
            "event": "worker_status",
            "worker_id": worker_id,
            "status": status,
            "current_tasks": current_tasks,
            "details": details or {},
            "timestamp": time.time()
        }
        self.logger.info(f"WORKER_STATUS: {json.dumps(metadata)}")
    
    def log_queue_stats(self, queue_size: int, active_tasks: int, healthy_workers: int):
        """Log queue and system statistics"""
        metadata = {
            "event": "system_stats",
            "queue_size": queue_size,
            "active_tasks": active_tasks,
            "healthy_workers": healthy_workers,
            "timestamp": time.time()
        }
        self.logger.info(f"SYSTEM_STATS: {json.dumps(metadata)}")
    
    def log_load_balancing_decision(self, request_id: str, task_type: str, 
                                  selected_worker: str, available_workers: Dict[str, float]):
        """Log load balancing decisions"""
        metadata = {
            "event": "load_balancing_decision",
            "request_id": request_id,
            "task_type": task_type,
            "selected_worker": selected_worker,
            "available_workers": available_workers,
            "timestamp": time.time()
        }
        self.logger.info(f"LOAD_BALANCING: {json.dumps(metadata)}")

# Global task logger instance
task_logger = TaskLogger()