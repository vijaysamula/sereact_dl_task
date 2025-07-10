import time
import random
import asyncio
import logging
import os
from fastapi import FastAPI, HTTPException
import torch
from transformers import pipeline
from shared.models import InferenceRequest, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Worker:
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.models = {}
        self.failure_simulation = {
            "crash_probability": float(os.getenv("CRASH_PROBABILITY", "0.02")),  # 2% chance
            "timeout_probability": float(os.getenv("TIMEOUT_PROBABILITY", "0.03")),  # 3% chance
            "network_delay_min": float(os.getenv("NETWORK_DELAY_MIN", "0.1")),
            "network_delay_max": float(os.getenv("NETWORK_DELAY_MAX", "0.8")),
        }
        self.load_models()
        
    def load_models(self):
        """Load different models per worker to simulate specialization"""
        try:
            # Each worker loads different models to simulate distributed specialization
            if self.worker_id == "worker-1":
                # Worker 1: Text classification specialist
                self.models[TaskType.TEXT_CLASSIFICATION] = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1
                )
                logger.info(f"Worker {self.worker_id}: Loaded DistilBERT for text classification")
                
            elif self.worker_id == "worker-2":
                # Worker 2: Image classification specialist
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
                logger.info(f"Worker {self.worker_id}: Loaded MobileNetV2 for image classification")
                
            elif self.worker_id == "worker-3":
                # Worker 3: CLIP multimodal specialist
                try:
                    from transformers import CLIPProcessor, CLIPModel
                    clip_model_name = "openai/clip-vit-base-patch32"
                    
                    self.models["clip"] = {
                        "model": CLIPModel.from_pretrained(clip_model_name),
                        "processor": CLIPProcessor.from_pretrained(clip_model_name)
                    }
                    self.models["clip"]["model"].eval()
                    
                    # Enable CLIP tasks
                    self.models[TaskType.CLIP_TEXT_SIMILARITY] = self.models["clip"]
                    self.models[TaskType.CLIP_TEXT_TO_IMAGE] = self.models["clip"]
                    
                    logger.info(f"Worker {self.worker_id}: Loaded CLIP for multimodal tasks")
                except Exception as clip_error:
                    logger.warning(f"Worker {self.worker_id}: Failed to load CLIP: {clip_error}")
            
            # All workers can handle basic text classification as fallback
            if TaskType.TEXT_CLASSIFICATION not in self.models:
                self.models[TaskType.TEXT_CLASSIFICATION] = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1
                )
            
            # ImageNet class labels for image classification
            self.imagenet_classes = [
                "tench", "goldfish", "great_white_shark", "tiger_shark", "hammerhead",
                "electric_ray", "stingray", "rooster", "hen", "ostrich", "brambling",
                "goldfinch", "house_finch", "junco", "indigo_bunting", "robin"
            ]
            
            logger.info(f"Worker {self.worker_id}: All models loaded successfully")
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Failed to load models: {str(e)}")
            raise
    
    async def process_request(self, request: InferenceRequest) -> dict:
        """Process inference request"""
        try:
            logger.info(f"Worker {self.worker_id}: Processing request {request.request_id}")
            
            # Simulate some processing time
            await asyncio.sleep(random.uniform(0.1, 0.5))
            
            # Simulate random failures (5% chance)
            if random.random() < 0.05:
                raise Exception("Simulated worker failure")
                
            if request.task_type == TaskType.TEXT_CLASSIFICATION:
                if TaskType.TEXT_CLASSIFICATION not in self.models:
                    raise Exception("Text classification model not loaded")
                    
                result = self.models[TaskType.TEXT_CLASSIFICATION](request.data)
                logger.info(f"Worker {self.worker_id}: Successfully processed text request {request.request_id}")
                
                return {
                    "prediction": result[0]["label"],
                    "confidence": result[0]["score"],
                    "worker_id": self.worker_id,
                    "task_type": "text_classification"
                }
                
            elif request.task_type == TaskType.IMAGE_CLASSIFICATION:
                if TaskType.IMAGE_CLASSIFICATION not in self.models:
                    raise Exception("Image classification model not loaded")
                
                # For demo purposes, simulate image processing with dummy data
                # In real implementation, you'd decode base64 image data
                import torch
                import numpy as np
                
                # Create a dummy image tensor (3, 224, 224)
                dummy_image = torch.randn(1, 3, 224, 224)
                
                model = self.models[TaskType.IMAGE_CLASSIFICATION]["model"]
                with torch.no_grad():
                    outputs = model(dummy_image)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    top_prob, top_idx = torch.topk(probabilities, 1)
                
                predicted_class = self.imagenet_classes[top_idx.item() % len(self.imagenet_classes)]
                confidence = top_prob.item()
                
                logger.info(f"Worker {self.worker_id}: Successfully processed image request {request.request_id}")
                
                return {
                    "prediction": predicted_class,
                    "confidence": confidence,
                    "worker_id": self.worker_id,
                    "task_type": "image_classification"
                }
            else:
                raise Exception(f"Unsupported task type: {request.task_type}")
                
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error processing request {request.request_id}: {str(e)}")
            raise