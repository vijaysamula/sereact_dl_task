# Distributed AI Inference System

A microservice-based AI inference system that simulates distributed processing with multiple worker nodes, fault tolerance, and load balancing.

## 🎯 Assignment Requirements Met

✅ **Microservice Architecture**: Coordinator + 3 Worker services  
✅ **Multiple Models**: DistilBERT, MobileNetV2, CLIP  
✅ **Fault Tolerance**: Simulated failures, retries, health checks  
✅ **Batching & Queuing**: Task queue with batch processing  
✅ **Logging & Monitoring**: Detailed task logs + CLI monitor  
✅ **Docker Compose**: Full containerization  
✅ **Async I/O**: FastAPI with asyncio  
✅ **Burst Traffic Test**: Per-request success/failure reporting  

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- 4GB+ RAM (for model loading)

### 1. Start the System
```bash
git clone https://github.com/vijaysamula/sereact_dl_task.git
cd sereact_dl_task

# Start all services
docker-compose up --build

# Wait 3-5 minutes for models to load
# Look for "Models loaded successfully" messages
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
```

### 3. Monitor the System
```bash
# Real-time CLI monitor
python monitor_cli.py

# Check system status
curl http://localhost:8000/status
```

### 4. Run Burst Traffic Test (Assignment Requirement)
```bash
# Burst traffic test with per-request success/failure reporting
python burst_traffic_test.py --pattern mixed

# Simple load test
python simple_load_test.py
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/infer` | POST | Submit inference request |
| `/status` | GET | System status summary |
| `/queue/stats` | GET | Queue and worker statistics |
| `/workers/performance` | GET | Worker performance metrics |

## 🧪 Supported Task Types

- **text_classification**: Sentiment analysis using DistilBERT
- **image_classification**: Object recognition using MobileNetV2  
- **clip_text_similarity**: Text similarity using CLIP
- **clip_text_to_image**: Text-to-image matching using CLIP

## 🏗️ Architecture

```
Client → Coordinator → Worker Pool
                    ├── Worker 1 (DistilBERT)
                    ├── Worker 2 (MobileNetV2)  
                    └── Worker 3 (CLIP)
```

## 🔧 Failure Simulation

Configure via environment variables:
- `CRASH_PROBABILITY=0.02` (2% crash rate)
- `TIMEOUT_PROBABILITY=0.03` (3% timeout rate)
- `NETWORK_DELAY_MIN=0.1` (Min network delay)
- `NETWORK_DELAY_MAX=0.8` (Max network delay)

## 📁 Key Files

```
├── docker-compose.yml          # Multi-service orchestration
├── coordinator/
│   ├── coordinator.py          # Main coordinator logic
│   └── main.py                 # FastAPI app
├── worker/
│   ├── worker.py              # Worker implementation
│   └── main.py                # Worker FastAPI app
├── shared/
│   ├── models.py              # Data models
│   └── logging_setup.py       # Logging setup
├── monitor_cli.py             # Real-time monitor
├── burst_traffic_test.py      # Assignment requirement
└── simple_load_test.py        # Basic load test
```

## 🚨 Testing Scenarios

### 1. Normal Operation
```bash
python simple_load_test.py
```

### 2. Burst Traffic (Assignment Requirement)
```bash
python burst_traffic_test.py --pattern burst
```

### 3. Failure Simulation
```bash
# Stop a worker to test fault tolerance
docker-compose stop worker-1

# System automatically routes to other workers
```

## 🛠️ Troubleshooting

**Models not loading?**
```bash
# Check worker logs
docker-compose logs worker-1

# Ensure sufficient memory
docker stats
```

**Connection errors?**
```bash
# Wait for full startup (3-5 minutes)
curl http://localhost:8000/status
```

**High latency?**
```bash
# Monitor real-time
python monitor_cli.py
```

## 📈 Expected Performance

- **Text Classification**: 0.2-0.6s
- **Image Classification**: 0.5-1.0s
- **CLIP Tasks**: 0.8-1.5s
- **Success Rate**: >95% under normal load
- **Automatic Retries**: Up to 3 attempts
- **Health Checks**: Every 10 seconds

## 📝 Assignment Compliance

This implementation fulfills all assignment requirements:

1. ✅ **Microservice Architecture**: REST-based coordinator + 3 workers
2. ✅ **Fault Tolerance**: Simulated failures, retries, health monitoring
3. ✅ **Batching & Queuing**: Priority queue with efficient batching
4. ✅ **Logging**: Detailed task logs with metadata
5. ✅ **Model Usage**: DistilBERT, MobileNetV2, CLIP models
6. ✅ **Docker Compose**: Full containerization
7. ✅ **Async I/O**: FastAPI throughout
8. ✅ **Burst Traffic Test**: Per-request success/failure reporting
9. ✅ **Live Monitoring**: CLI dashboard

## 🙏 Notes

- Models load in ~3-5 minutes on first startup
- Each worker supports all task types for better load distribution
- System automatically handles worker failures with retries
- All logs stored in `logs/task_operations.log`
- Monitor provides real-time system visibility