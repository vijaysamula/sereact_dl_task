# Distributed AI Inference System

A production-ready distributed AI inference system with intelligent task queuing, fault tolerance, and multi-modal AI capabilities.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────────────────────┐
│   Client        │───▶│       Coordinator               │
│   Requests      │    │   - Intelligent Queuing         │
└─────────────────┘    │   - Health Monitoring           │
                       │   - Load Balancing               │
                       │   - Retry Logic                  │
                       └─────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
            ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
            │    Worker 1     │ │    Worker 2     │ │    Worker 3     │
            │   DistilBERT    │ │   MobileNetV2   │ │      CLIP       │
            │ (Text Classify) │ │ (Image Classify)│ │  (Multimodal)   │
            │   Port 8001     │ │   Port 8002     │ │   Port 8003     │
            └─────────────────┘ └─────────────────┘ └─────────────────┘
```

## ✅ Requirements Fulfilled

### 1. **Microservice Inference Architecture**
- ✅ **Coordinator Service**: FastAPI REST service managing inference requests
- ✅ **3 Worker Services**: Each specialized with different models:
  - **Worker 1**: DistilBERT for text classification
  - **Worker 2**: MobileNetV2 for image classification  
  - **Worker 3**: CLIP for multimodal tasks
- ✅ **HTTP Endpoints**: All workers expose `/infer` and `/health` endpoints

### 2. **Simulated Distribution & Fault Tolerance**
- ✅ **Network Delay Simulation**: Configurable delays (0.1-0.8s)
- ✅ **Failure Simulation**: Random crashes (2%) and timeouts (3%)
- ✅ **Health Monitoring**: Heartbeat checks every 10 seconds
- ✅ **Automatic Retries**: Up to 3 retries with intelligent routing
- ✅ **Comprehensive Logging**: All tasks logged with metadata

### 3. **Batching & Queuing**
- ✅ **Priority Queue**: Tasks queued with priority support
- ✅ **Intelligent Dispatch**: Load-aware worker selection
- ✅ **Performance Scoring**: Workers rated on latency and reliability
- ✅ **Batch Processing**: Efficient queue processing loop

### 4. **Logging and Monitoring**
- ✅ **Detailed Task Logs**: Request ID, Worker ID, latency, retry count
- ✅ **CLI Monitor**: Real-time dashboard showing active workers and load
- ✅ **Performance Metrics**: Worker scores, queue stats, system health
- ✅ **Structured Logging**: JSON-formatted logs with metadata

### 5. **Model Usage**
- ✅ **Pre-trained Models**: DistilBERT, MobileNetV2, CLIP
- ✅ **Multi-modal Support**: Text, image, and CLIP multimodal tasks
- ✅ **Realistic Processing**: Actual model inference, not just dummy responses

### 🎯 **Bonus Features Implemented**
- ✅ **Docker Compose**: Complete containerization
- ✅ **Async I/O**: FastAPI with asyncio throughout
- ✅ **Burst Traffic Test**: Comprehensive load testing script
- ✅ **CLI Monitor**: Live system monitoring

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for local development)
- 4GB+ RAM (for model loading)

### 1. Start the System
```bash
# Clone the repository
git clone <your-repo-url>
cd distributed-ai-inference

# Start all services
docker-compose up --build

# Wait for models to load (3-5 minutes)
# You'll see "Models loaded successfully" for each worker
```

### 2. Test Basic Functionality
```bash
# Test text classification
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test-1",
    "task_type": "text_classification",
    "data": "I love this product!"
  }'

# Test image classification
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test-2", 
    "task_type": "image_classification",
    "data": "dummy_image_data"
  }'

# Test CLIP multimodal
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test-3",
    "task_type": "clip_text_similarity", 
    "data": "a photo of a cat"
  }'
```

### 3. Monitor the System
```bash
# Start real-time CLI monitor
python monitor_cli.py

# Or check status via API
curl http://localhost:8000/queue/stats
curl http://localhost:8000/workers/performance
```

### 4. Run Load Tests
```bash
# Simple load test
python simple_load_test.py

# Comprehensive multi-modal test
python enhanced_clip_load_test.py

# Burst traffic simulation
python burst_traffic_test.py --pattern mixed
```

## 📊 API Endpoints

### Coordinator (Port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/infer` | POST | Submit inference request (synchronous) |
| `/infer/async` | POST | Submit async inference request |
| `/result/{request_id}` | GET | Get result of async request |
| `/priority/infer` | POST | High-priority inference request |
| `/status` | GET | System status summary |
| `/queue/stats` | GET | Detailed queue and worker statistics |
| `/workers/performance` | GET | Worker performance metrics |

### Workers (Ports 8001-8003)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/infer` | POST | Process inference task |
| `/health` | GET | Worker health check |

## 🎯 Supported Task Types

| Task Type | Description | Example Input | Specialized Worker |
|-----------|-------------|---------------|-------------------|
| `text_classification` | Sentiment analysis | "I love this!" | Worker 1 (DistilBERT) |
| `image_classification` | Object recognition | `dummy_image_data` | Worker 2 (MobileNetV2) |
| `clip_text_similarity` | Text similarity matching | "a photo of a cat" | Worker 3 (CLIP) |
| `clip_text_to_image` | Text-to-image matching | "find me a sunset" | Worker 3 (CLIP) |

## 🧪 Testing & Simulation

### Load Testing Scripts

1. **simple_load_test.py**: Basic concurrent requests
   ```bash
   python simple_load_test.py  # 10 concurrent requests
   ```

2. **enhanced_clip_load_test.py**: Comprehensive multi-modal testing
   ```bash
   python enhanced_clip_load_test.py  # Multiple test scenarios
   ```

3. **burst_traffic_test.py**: Realistic traffic patterns
   ```bash
   python burst_traffic_test.py --pattern burst    # Burst testing
   python burst_traffic_test.py --pattern sustained # Sustained load
   python burst_traffic_test.py --pattern mixed     # Mixed patterns
   ```

### Failure Simulation

Configure failure rates via environment variables:

```bash
# In docker-compose.yml or environment
CRASH_PROBABILITY=0.05      # 5% crash rate
TIMEOUT_PROBABILITY=0.03    # 3% timeout rate
NETWORK_DELAY_MIN=0.1       # Min network delay
NETWORK_DELAY_MAX=0.8       # Max network delay
```

### Monitoring

**CLI Monitor** provides real-time visibility:
```bash
python monitor_cli.py --interval 1.0  # Update every second
```

Shows:
- 📊 System overview (queue, workers, performance)
- 👷 Worker details (status, load, performance scores)
- 📋 Recent activity
- 🎮 Interactive controls

## 📁 Project Structure

```
distributed-ai-inference/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container definition
├── docker-compose.yml          # Multi-service orchestration
├── .gitignore                   # Git ignore rules
├── shared/
│   ├── __init__.py
│   ├── models.py               # Pydantic data models
│   └── logging_setup.py        # Centralized logging
├── coordinator/
│   ├── __init__.py
│   ├── coordinator.py          # Main coordinator logic
│   └── main.py                 # FastAPI application
├── worker/
│   ├── __init__.py
│   ├── worker.py              # Worker implementation
│   └── main.py                # Worker FastAPI app
├── tests/
│   ├── __init__.py
│   └── test_client.py         # Basic test client
├── monitor_cli.py              # Real-time CLI monitor
├── simple_load_test.py         # Basic load testing
├── enhanced_clip_load_test.py  # Advanced multi-modal tests
├── burst_traffic_test.py       # Burst traffic simulation
└── logs/                       # Log files (created at runtime)
    └── task_operations.log     # Detailed task logging
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKER_ID` | worker-1 | Unique worker identifier |
| `CRASH_PROBABILITY` | 0.02 | Probability of simulated crashes |
| `TIMEOUT_PROBABILITY` | 0.03 | Probability of simulated timeouts |
| `NETWORK_DELAY_MIN` | 0.1 | Minimum network delay (seconds) |
| `NETWORK_DELAY_MAX` | 0.8 | Maximum network delay (seconds) |

### Docker Compose Configuration

The system uses Docker networking for service discovery:
- Coordinator: `http://coordinator:8000`
- Workers: `http://worker-1:8000`, `http://worker-2:8000`, `http://worker-3:8000`

External ports:
- Coordinator: `localhost:8000`
- Worker 1: `localhost:8001`  
- Worker 2: `localhost:8002`
- Worker 3: `localhost:8003`

## 📈 Performance Characteristics

### Expected Latencies
- **Text Classification**: 0.2-0.6s
- **Image Classification**: 0.5-1.0s  
- **CLIP Tasks**: 0.8-1.5s
- **Network Simulation**: +0.1-0.8s

### Fault Tolerance
- **Automatic Retries**: Up to 3 attempts
- **Health Monitoring**: 10-second intervals
- **Failure Recovery**: Automatic worker re-routing
- **Queue Persistence**: In-memory with graceful degradation

### Scalability
- **Worker Specialization**: Each worker optimized for specific tasks
- **Load Balancing**: Performance-based worker selection
- **Queue Management**: Priority-based task scheduling
- **Resource Monitoring**: Real-time capacity tracking

## 🚨 Failure Scenarios & Testing

### 1. Worker Crashes
```bash
# Simulate worker failure
docker-compose stop worker-1

# System automatically routes to other workers
# Monitor recovery in CLI monitor
```

### 2. Network Issues
- Simulated through configurable delays and timeouts
- Automatic retry with exponential backoff
- Health checks detect unresponsive workers

### 3. High Load
```bash
# Test system under heavy load
python burst_traffic_test.py --pattern sustained

# Monitor queue growth and worker distribution
python monitor_cli.py
```

### 4. Model Loading Failures
- Workers gracefully degrade to fallback models
- Health checks report model loading status
- Coordinator routes based on available capabilities

## 🧮 Example Outputs

### Successful Text Classification
```json
{
  "request_id": "test-123",
  "result": {
    "prediction": "POSITIVE",
    "confidence": 0.9998,
    "worker_id": "worker-1",
    "task_type": "text_classification",
    "model_info": "DistilBERT"
  },
  "latency": 0.324,
  "retry_count": 0
}
```

### CLIP Text Similarity
```json
{
  "request_id": "clip-456", 
  "result": {
    "prediction": "a photo of a cat",
    "confidence": 0.8456,
    "similarities": {
      "a photo of a cat": 0.8456,
      "a photo of a dog": 0.2341,
      "a car": 0.1234
    },
    "worker_id": "worker-3",
    "task_type": "clip_text_similarity",
    "model_info": "CLIP-ViT-B/32"
  },
  "latency": 1.123,
  "retry_count": 0
}
```

### Queue Statistics
```json
{
  "queue_size": 5,
  "active_tasks": 3,
  "total_workers": 3,
  "healthy_workers": 3,
  "busy_workers": 0,
  "stats": {
    "total_requests": 142,
    "successful_requests": 137,
    "failed_requests": 5,
    "queue_high_watermark": 12
  }
}
```

## 🤝 Development

### Local Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run coordinator locally
cd coordinator && python main.py

# Run workers locally (separate terminals)
WORKER_ID=worker-1 PORT=8001 python worker/main.py
WORKER_ID=worker-2 PORT=8002 python worker/main.py  
WORKER_ID=worker-3 PORT=8003 python worker/main.py
```

### Adding New Models
1. Update `TaskType` enum in `shared/models.py`
2. Add model loading logic in `worker/worker.py`
3. Implement processing method for new task type
4. Update test scripts to include new task type

### Custom Monitoring
The system provides extensive APIs for custom monitoring:
- `/queue/stats` - Real-time queue metrics
- `/workers/performance` - Individual worker metrics
- Log files in `logs/` directory with structured JSON

## 🛠️ Troubleshooting

### Common Issues

1. **Models not loading**
   ```bash
   # Check worker logs
   docker-compose logs worker-1
   
   # Ensure sufficient memory (4GB+)
   docker stats
   ```

2. **Connection refused errors**
   ```bash
   # Wait for full startup (3-5 minutes)
   # Check service health
   curl http://localhost:8000/status
   ```

3. **High latency**
   ```bash
   # Monitor queue depth
   python monitor_cli.py
   
   # Check worker performance scores
   curl http://localhost:8000/workers/performance
   ```

## 📝 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Transformers**: HuggingFace transformers library
- **FastAPI**: Modern Python web framework
- **Docker**: Containerization platform
- **PyTorch**: Deep learning framework