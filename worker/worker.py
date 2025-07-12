# Fixed worker/worker.py - Remove asyncio.create_task from __init__

import time
import random
import asyncio
import logging
import os
from fastapi import FastAPI, HTTPException
import torch
from transformers import pipeline
from shared.models import InferenceRequest, TaskType
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Worker:
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.models = {}
        self.batch_size = int(os.getenv("BATCH_SIZE", "4"))  # Configurable batch size
        self.batch_timeout = float(os.getenv("BATCH_TIMEOUT", "0.1"))  # Max wait for batch
        self.pending_requests = []
        self.batch_lock = asyncio.Lock()
        self.batch_processor_started = False  # ✅ Add flag to track if processor started
        
        self.failure_simulation = {
            "crash_probability": float(os.getenv("CRASH_PROBABILITY", "0.02")),
            "timeout_probability": float(os.getenv("TIMEOUT_PROBABILITY", "0.03")),
            "network_delay_min": float(os.getenv("NETWORK_DELAY_MIN", "0.1")),
            "network_delay_max": float(os.getenv("NETWORK_DELAY_MAX", "0.8")),
        }
        
        # ImageNet classes for image classification
        self.imagenet_classes = [
            "tench", "goldfish", "great_white_shark", "tiger_shark", "hammerhead",
            "electric_ray", "stingray", "rooster", "hen", "ostrich", "brambling",
            "goldfinch", "house_finch", "junco", "indigo_bunting", "robin", "bulbul",
            "jay", "magpie", "chickadee", "water_ouzel", "kite", "bald_eagle", "vulture",
            "great_grey_owl", "European_fire_salamander", "common_newt", "eft", "spotted_salamander",
            "axolotl", "bullfrog", "tree_frog", "tailed_frog", "loggerhead", "leatherback_turtle",
            "mud_turtle", "terrapin", "box_turtle", "banded_gecko", "common_iguana",
            "American_chameleon", "whiptail", "agama", "frilled_lizard", "alligator_lizard",
            "Gila_monster", "green_lizard", "African_chameleon", "Komodo_dragon", "African_crocodile",
            "American_alligator", "triceratops", "thunder_snake", "ringneck_snake",
            "hognose_snake", "green_snake", "king_snake", "garter_snake", "water_snake",
            "vine_snake", "night_snake", "boa_constrictor", "rock_python", "Indian_cobra",
            "green_mamba", "sea_snake", "horned_viper", "diamondback", "sidewinder"
        ]
        
        self.load_all_models()
        
        # ✅ DON'T start batch processor here - will be started when first request comes
        logger.info(f"Worker {self.worker_id}: Initialized with batch processing capability")
        
    def start_batch_processor_if_needed(self):
        """Start batch processor if not already started"""
        if not self.batch_processor_started:
            try:
                asyncio.create_task(self.batch_processor())
                self.batch_processor_started = True
                logger.info(f"Worker {self.worker_id}: Batch processor started")
            except RuntimeError:
                # Event loop not running yet, will start on first request
                pass
        
    def load_all_models(self):
        """Load ALL models on EVERY worker for true load balancing"""
        logger.info(f"Worker {self.worker_id}: Loading all models for universal capability...")
        
        try:
            # 1. Load Text Classification Model (DistilBERT)
            logger.info(f"Worker {self.worker_id}: Loading DistilBERT for text classification...")
            self.models[TaskType.TEXT_CLASSIFICATION] = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1
            )
            logger.info(f"Worker {self.worker_id}: ✅ DistilBERT loaded successfully")
            
            # 2. Load Image Classification Model (MobileNetV2)
            logger.info(f"Worker {self.worker_id}: Loading MobileNetV2 for image classification...")
            try:
                from torchvision import models, transforms
                self.models[TaskType.IMAGE_CLASSIFICATION] = {
                    "model": models.mobilenet_v2(pretrained=True),
                    "transform": transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                }
                self.models[TaskType.IMAGE_CLASSIFICATION]["model"].eval()
                logger.info(f"Worker {self.worker_id}: ✅ MobileNetV2 loaded successfully")
            except Exception as img_error:
                logger.error(f"Worker {self.worker_id}: ❌ Failed to load MobileNetV2: {img_error}")
            
            # 3. Load CLIP Model (for multimodal tasks)
            logger.info(f"Worker {self.worker_id}: Loading CLIP for multimodal tasks...")
            try:
                from transformers import CLIPProcessor, CLIPModel
                clip_model_name = "openai/clip-vit-base-patch32"
                
                clip_model_data = {
                    "model": CLIPModel.from_pretrained(clip_model_name),
                    "processor": CLIPProcessor.from_pretrained(clip_model_name)
                }
                clip_model_data["model"].eval()
                
                # Register CLIP for both similarity tasks
                self.models[TaskType.CLIP_TEXT_SIMILARITY] = clip_model_data
                self.models[TaskType.CLIP_TEXT_TO_IMAGE] = clip_model_data
                
                logger.info(f"Worker {self.worker_id}: ✅ CLIP loaded successfully")
            except Exception as clip_error:
                logger.warning(f"Worker {self.worker_id}: ⚠️ Failed to load CLIP: {clip_error}")
                logger.info(f"Worker {self.worker_id}: CLIP tasks will fall back to text classification")
            
            # Summary of loaded models
            loaded_models = list(self.models.keys())
            logger.info(f"Worker {self.worker_id}: 🎉 Model loading complete!")
            logger.info(f"Worker {self.worker_id}: Available task types: {loaded_models}")
            logger.info(f"Worker {self.worker_id}: Batch processing enabled (batch_size: {self.batch_size})")
            
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: ❌ Critical error during model loading: {str(e)}")
            # Ensure at least text classification works
            if TaskType.TEXT_CLASSIFICATION not in self.models:
                try:
                    self.models[TaskType.TEXT_CLASSIFICATION] = pipeline(
                        "sentiment-analysis",
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        device=-1
                    )
                    logger.info(f"Worker {self.worker_id}: 🔄 Emergency fallback: text classification loaded")
                except Exception as fallback_error:
                    logger.error(f"Worker {self.worker_id}: ❌ Even emergency fallback failed: {fallback_error}")
                    raise

    async def process_request(self, request: InferenceRequest) -> dict:
        """Add request to batch queue for efficient processing"""
        # ✅ Start batch processor when first request arrives
        if not self.batch_processor_started:
            asyncio.create_task(self.batch_processor())
            self.batch_processor_started = True
            logger.info(f"Worker {self.worker_id}: Batch processor started on first request")
        
        logger.info(f"Worker {self.worker_id}: Queuing {request.task_type} request {request.request_id} for batch processing")
        
        # Create a future for this request
        future = asyncio.Future()
        
        async with self.batch_lock:
            self.pending_requests.append({
                "request": request,
                "future": future,
                "queued_at": time.time()
            })
            
            # If batch is full, trigger immediate processing
            if len(self.pending_requests) >= self.batch_size:
                asyncio.create_task(self.process_current_batch())
        
        # Wait for the batch processing to complete
        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            logger.info(f"Worker {self.worker_id}: ✅ Batch-processed request {request.request_id}")
            return result
        except asyncio.TimeoutError:
            logger.error(f"Worker {self.worker_id}: ❌ Batch processing timeout for {request.request_id}")
            raise Exception("Batch processing timeout")

    async def batch_processor(self):
        """Background task to process batches with timeout"""
        logger.info(f"Worker {self.worker_id}: Batch processor loop started")
        
        while True:
            await asyncio.sleep(self.batch_timeout)
            
            async with self.batch_lock:
                if self.pending_requests:
                    # Check if oldest request is getting stale
                    oldest_request = min(self.pending_requests, key=lambda x: x["queued_at"])
                    age = time.time() - oldest_request["queued_at"]
                    
                    if age > self.batch_timeout:
                        asyncio.create_task(self.process_current_batch())

    async def process_current_batch(self):
        """Process the current batch of requests"""
        async with self.batch_lock:
            if not self.pending_requests:
                return
                
            current_batch = self.pending_requests.copy()
            self.pending_requests.clear()
        
        logger.info(f"Worker {self.worker_id}: 🔄 Processing batch of {len(current_batch)} requests")
        
        # Group requests by task type for efficient batch processing
        batches_by_type = {}
        for item in current_batch:
            task_type = item["request"].task_type
            if task_type not in batches_by_type:
                batches_by_type[task_type] = []
            batches_by_type[task_type].append(item)
        
        # Process each task type batch
        for task_type, batch_items in batches_by_type.items():
            try:
                await self._process_batch_by_type(task_type, batch_items)
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: ❌ Batch processing failed for {task_type}: {str(e)}")
                # Set error for all requests in this batch
                for item in batch_items:
                    if not item["future"].done():
                        item["future"].set_exception(Exception(f"Batch processing failed: {str(e)}"))

    async def _process_batch_by_type(self, task_type: TaskType, batch_items: List[Dict[str, Any]]):
        """Process a batch of requests of the same type"""
        batch_size = len(batch_items)
        logger.info(f"Worker {self.worker_id}: Processing {task_type} batch of {batch_size} requests")
        
        # Simulate batch processing time (more efficient than individual)
        batch_processing_time = random.uniform(0.2, 0.6) + (batch_size * 0.05)  # Base time + per-item overhead
        await asyncio.sleep(batch_processing_time)
        
        # Simulate random batch failures
        if random.random() < self.failure_simulation["crash_probability"]:
            raise Exception("Simulated batch processing failure")
        
        # Process requests based on task type
        if task_type == TaskType.TEXT_CLASSIFICATION:
            await self._process_text_batch(batch_items)
        elif task_type == TaskType.IMAGE_CLASSIFICATION:
            await self._process_image_batch(batch_items)
        elif task_type in [TaskType.CLIP_TEXT_SIMILARITY, TaskType.CLIP_TEXT_TO_IMAGE]:
            await self._process_clip_batch(batch_items)
        else:
            raise Exception(f"Unsupported task type for batching: {task_type}")
        
        logger.info(f"Worker {self.worker_id}: ✅ Completed {task_type} batch processing")

    async def _process_text_batch(self, batch_items: List[Dict[str, Any]]):
        """Process a batch of text classification requests"""
        if TaskType.TEXT_CLASSIFICATION not in self.models:
            raise Exception("Text classification model not available")
        
        model = self.models[TaskType.TEXT_CLASSIFICATION]
        
        # Extract text data for batch processing
        texts = [item["request"].data for item in batch_items]
        
        # Process texts in batch (more efficient)
        results = model(texts)
        
        # Set results for each request
        for i, item in enumerate(batch_items):
            result = {
                "prediction": results[i]["label"],
                "confidence": results[i]["score"],
                "worker_id": self.worker_id,
                "task_type": "text_classification",
                "model_info": "DistilBERT",
                "batch_processed": True,
                "batch_size": len(batch_items)
            }
            item["future"].set_result(result)

    async def _process_image_batch(self, batch_items: List[Dict[str, Any]]):
        """Process a batch of image classification requests"""
        if TaskType.IMAGE_CLASSIFICATION not in self.models:
            raise Exception("Image classification model not available")
        
        model_data = self.models[TaskType.IMAGE_CLASSIFICATION]
        model = model_data["model"]
        
        # Create batch of dummy images (in real implementation, decode base64 images)
        batch_size = len(batch_items)
        dummy_batch = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            outputs = model(dummy_batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, 1, dim=1)
        
        # Set results for each request
        for i, item in enumerate(batch_items):
            predicted_class = self.imagenet_classes[top_indices[i].item() % len(self.imagenet_classes)]
            result = {
                "prediction": predicted_class,
                "confidence": top_probs[i].item(),
                "worker_id": self.worker_id,
                "task_type": "image_classification",
                "model_info": "MobileNetV2",
                "batch_processed": True,
                "batch_size": len(batch_items)
            }
            item["future"].set_result(result)

    async def _process_clip_batch(self, batch_items: List[Dict[str, Any]]):
        """Process a batch of CLIP requests"""
        task_type = batch_items[0]["request"].task_type
        
        if task_type not in self.models:
            # Fallback to text classification batch
            await self._process_text_batch_fallback(batch_items)
            return
        
        # CLIP batch processing simulation
        for item in batch_items:
            similarity_scores = {
                "a photo of a cat": random.uniform(0.7, 0.9),
                "a photo of a dog": random.uniform(0.2, 0.4),
                "a car": random.uniform(0.1, 0.3),
                "beautiful sunset": random.uniform(0.3, 0.6),
                "person walking": random.uniform(0.4, 0.7)
            }
            
            best_match = max(similarity_scores.keys(), key=lambda k: similarity_scores[k])
            
            result = {
                "prediction": best_match,
                "confidence": similarity_scores[best_match],
                "similarities": similarity_scores,
                "worker_id": self.worker_id,
                "task_type": str(task_type),
                "model_info": "CLIP-ViT-B/32",
                "batch_processed": True,
                "batch_size": len(batch_items)
            }
            item["future"].set_result(result)

    async def _process_text_batch_fallback(self, batch_items: List[Dict[str, Any]]):
        """Fallback batch processing using text classification"""
        if TaskType.TEXT_CLASSIFICATION not in self.models:
            raise Exception("No fallback model available")
        
        model = self.models[TaskType.TEXT_CLASSIFICATION]
        texts = [item["request"].data for item in batch_items]
        results = model(texts)
        
        for i, item in enumerate(batch_items):
            result = {
                "prediction": results[i]["label"],
                "confidence": results[i]["score"],
                "worker_id": self.worker_id,
                "task_type": "text_classification_fallback",
                "model_info": "DistilBERT (batch fallback)",
                "original_task_type": str(item["request"].task_type),
                "note": "Batch processed with fallback model",
                "batch_processed": True,
                "batch_size": len(batch_items)
            }
            item["future"].set_result(result)