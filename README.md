# Distributed AI Inference System

A microservice-based AI inference system with fault tolerance, load balancing, and batch processing.

## 🎯 Assignment Requirements Met

✅ **Microservice Architecture**: Coordinator + 3 Workers  
✅ **Multiple Models**: DistilBERT, MobileNetV2, CLIP  
✅ **Fault Tolerance**: Failures, retries, health checks  
✅ **Batching & Queuing**: Task queue with batching  
✅ **Logging & Monitoring**: Detailed logs + CLI monitor  
✅ **Docker Compose**: Full containerization  
✅ **Async I/O**: FastAPI with asyncio  
✅ **Burst Traffic Test**: Per-request success/failure reporting  

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/vijaysamula/sereact_dl_task.git
cd sereact_dl_task

# Start system (Docker)
docker compose up --build

# Wait for models to load (3-5 minutes)
# Look for "Model loading complete" messages

# Install monitoring tools locally
pip install -r requirements-local.txt

# Test basic functionality
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{"request_id": "test-1", "task_type": "text_classification", "data": "I love this!"}'

# Monitor system (CLI)
python monitor_cli.py

# Web dashboard
streamlit run streamlit_dashboard.py

# Run burst traffic test (Assignment Requirement)
python burst_traffic_test.py
```

## 🚨 Failure Simulation & Testing

### Method 1: Stop Worker Container
```bash
# Simulate worker failure
docker compose stop worker-1

# Test failover (should work with remaining workers)
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{"request_id": "failover-test", "task_type": "text_classification", "data": "test"}'

# Restart worker
docker compose start worker-1
```

### Method 2: Use Failure Simulation Script
```bash
python simulate_failures.py
```

### Method 3: High Failure Testing
```bash
# Use high failure rate config
docker compose -f docker-compose.yml -f docker-compose.test.yml up
```

## Expected Outputs

### Successful Request
```json
{
  "request_id": "test-1",
  "worker_id": "worker-2",
  "result": {
    "prediction": "POSITIVE",
    "confidence": 0.9998,
    "task_type": "text_classification"
  },
  "latency": 0.234,
  "retry_count": 0
}
```

### During Failure (Automatic Failover)
```json
{
  "request_id": "failover-test",
  "worker_id": "worker-3",  // Auto-routed to healthy worker
  "result": {"prediction": "POSITIVE"},
  "latency": 0.345,
  "retry_count": 0
}
```

### Burst Test Output
```
[14:32:01.234] 🚀 Request burst-001: Sending text_classification task...
[14:32:01.456] ✅ Request burst-001: SUCCESS (0.223s, worker-1, POSITIVE)
[14:32:01.678] ❌ Request burst-002: FAILED (HTTP 500, 1.234s, Worker crash)
[14:32:02.123] ✅ Request burst-003: SUCCESS (0.445s, worker-2, POSITIVE)

📊 SUMMARY: 4/5 successful (80.0%), avg latency: 0.334s
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/infer` | POST | Submit inference request |
| `/status` | GET | System status summary |
| `/queue/stats` | GET | Queue and worker statistics |
| `/workers/performance` | GET | Worker performance metrics |

## 🧪 Supported Task Types

- **text_classification**: Sentiment analysis (DistilBERT)
- **image_classification**: Object recognition (MobileNetV2)  
- **clip_text_similarity**: Text similarity (CLIP)
- **clip_text_to_image**: Text-to-image matching (CLIP)

## 🏗️ Architecture

```
Client → Coordinator → Worker Pool
                    ├── Worker 1 (All Models)
                    ├── Worker 2 (All Models)  
                    └── Worker 3 (All Models)
```

## 📁 Key Files

```
├── docker-compose.yml          # Multi-service orchestration  
├── requirements.txt            # Docker container dependencies
├── requirements-local.txt      # Local monitoring tools
├── coordinator/
│   ├── coordinator.py          # Load balancing & fault tolerance
│   └── main.py                 # FastAPI coordinator
├── worker/
│   ├── worker.py              # Model loading & batch processing
│   └── main.py                # FastAPI worker
├── shared/models.py           # Data models
├── monitor_cli.py             # Enhanced CLI monitor with load balancing insights
├── streamlit_dashboard.py    # Web dashboard with real-time charts (bonus)
├── burst_traffic_test.py      # Assignment requirement
└── simulate_failures.py      # Failure testing
```

## 🛠️ Troubleshooting

**Models not loading?**
```bash
docker compose logs worker-1  # Check for memory issues
```

**Connection errors?**
```bash
docker compose ps            # Check all services running
curl http://localhost:8000/status  # Test coordinator
```

## 📝 Notes

- Models load in ~3-5 minutes on first startup
- Each worker supports all task types for load balancing
- System automatically handles failures with retries
- All logs stored in `logs/task_operations.log`
- **Enhanced CLI monitor** shows load balancing insights and task distribution
- **Streamlit dashboard** provides web interface with real-time charts