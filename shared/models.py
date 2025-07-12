# File: shared/models.py
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class TaskType(str, Enum):
    TEXT_CLASSIFICATION = "text_classification"
    IMAGE_CLASSIFICATION = "image_classification"
    CLIP_TEXT_SIMILARITY = "clip_text_similarity"
    CLIP_IMAGE_SIMILARITY = "clip_image_similarity"
    CLIP_TEXT_TO_IMAGE = "clip_text_to_image"

class InferenceRequest(BaseModel):
    request_id: str
    task_type: TaskType
    data: str  # Could be text or base64 encoded image
    
class InferenceResponse(BaseModel):
    request_id: str
    worker_id: str
    result: dict
    latency: float
    retry_count: int = 0

class QueuedTask(BaseModel):
    request: InferenceRequest
    priority: int = 0  # Higher number = higher priority
    queued_at: float
    retries: int = 0
    max_retries: int = 3  # ✅ ADD THIS LINE!

class WorkerStatus(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    worker_id: str
    status: str  # "healthy", "busy", "error", "starting"
    last_heartbeat: float
    model_loaded: bool
    current_tasks: int = 0
    max_tasks: int = 3  # Maximum concurrent tasks per worker
    avg_latency: float = 0.0  # Average response time
    total_processed: int = 0  # Total requests processed