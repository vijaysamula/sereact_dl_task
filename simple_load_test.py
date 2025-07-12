#!/usr/bin/env python3
"""
Simple Load Test for Distributed AI Inference System
Basic concurrent request testing with clear output
"""

import asyncio
import aiohttp
import time
import json
import uuid
from typing import List, Dict

class SimpleLoadTester:
    def __init__(self, coordinator_url: str = "http://localhost:8000"):
        self.coordinator_url = coordinator_url
        self.test_requests = [
            {
                "task_type": "text_classification",
                "data": "I love this amazing product!"
            },
            {
                "task_type": "text_classification", 
                "data": "This is really disappointing."
            },
            {
                "task_type": "image_classification",
                "data": "dummy_image_data_sample"
            },
            {
                "task_type": "clip_text_similarity",
                "data": "a photo of a cute cat"
            },
            {
                "task_type": "clip_text_to_image",
                "data": "beautiful sunset landscape"
            }
        ]
    
    async def send_request(self, session: aiohttp.ClientSession, request_data: Dict) -> Dict:
        """Send a single inference request"""
        request_id = f"simple-{uuid.uuid4().hex[:8]}"
        
        payload = {
            "request_id": request_id,
            "task_type": request_data["task_type"],
            "data": request_data["data"]
        }
        
        start_time = time.time()
        
        try:
            async with session.post(
                f"{self.coordinator_url}/infer",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                end_time = time.time()
                latency = end_time - start_time
                
                if response.status == 200:
                    result = await response.json()
                    worker_id = result.get("result", {}).get("worker_id", "unknown")
                    prediction = result.get("result", {}).get("prediction", "unknown")
                    
                    return {
                        "request_id": request_id,
                        "task_type": request_data["task_type"],
                        "status": "success",
                        "latency": latency,
                        "worker_id": worker_id,
                        "prediction": prediction
                    }
                else:
                    error_text = await response.text()
                    return {
                        "request_id": request_id,
                        "task_type": request_data["task_type"],
                        "status": "failed",
                        "latency": latency,
                        "error": f"HTTP {response.status}: {error_text}"
                    }
                    
        except asyncio.TimeoutError:
            return {
                "request_id": request_id,
                "task_type": request_data["task_type"],
                "status": "timeout",
                "latency": time.time() - start_time,
                "error": "Request timeout"
            }
        except Exception as e:
            return {
                "request_id": request_id,
                "task_type": request_data["task_type"],
                "status": "error",
                "latency": time.time() - start_time,
                "error": str(e)
            }
    
    async def run_simple_load_test(self, num_requests: int = 10):
        """Run a simple load test with concurrent requests"""
        print("🧪 SIMPLE LOAD TEST")
        print("=" * 50)
        print(f"📡 Target: {self.coordinator_url}")
        print(f"📊 Requests: {num_requests} concurrent")
        print("=" * 50)
        
        # Check system status first
        async with aiohttp.ClientSession() as session:
            try:
                print("📋 Checking system status...")
                async with session.get(f"{self.coordinator_url}/status") as response:
                    if response.status == 200:
                        status = await response.json()
                        healthy_workers = status.get('healthy_workers', 0)
                        print(f"✅ System online - {healthy_workers} healthy workers")
                    else:
                        print(f"⚠️ System status: HTTP {response.status}")
                        return
            except Exception as e:
                print(f"❌ Cannot connect to system: {str(e)}")
                return
            
            print("\n🚀 Starting load test...")
            
            # Create concurrent requests
            tasks = []
            for i in range(num_requests):
                # Cycle through different request types
                request_data = self.test_requests[i % len(self.test_requests)]
                task = self.send_request(session, request_data)
                tasks.append(task)
            
            # Execute all requests concurrently
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Process results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "request_id": f"simple-{i}",
                        "status": "exception",
                        "error": str(result),
                        "latency": 0
                    })
                else:
                    processed_results.append(result)
            
            # Print results
            self.print_results(processed_results, end_time - start_time)
    
    def print_results(self, results: List[Dict], total_time: float):
        """Print test results summary"""
        print(f"\n📊 LOAD TEST RESULTS")
        print("=" * 50)
        
        # Calculate statistics
        total_requests = len(results)
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] != "success"]
        
        success_rate = len(successful) / total_requests * 100 if total_requests > 0 else 0
        
        print(f"Total Requests:    {total_requests}")
        print(f"Successful:        {len(successful)} ({success_rate:.1f}%)")
        print(f"Failed:            {len(failed)}")
        print(f"Total Time:        {total_time:.3f}s")
        print(f"Requests/sec:      {total_requests / total_time:.1f}")
        
        if successful:
            latencies = [r["latency"] for r in successful]
            avg_latency = sum(latencies) / len(latencies)
            print(f"Avg Latency:       {avg_latency:.3f}s")
            print(f"Min Latency:       {min(latencies):.3f}s")
            print(f"Max Latency:       {max(latencies):.3f}s")
        
        # Worker distribution
        if successful:
            worker_counts = {}
            for result in successful:
                worker_id = result.get("worker_id", "unknown")
                worker_counts[worker_id] = worker_counts.get(worker_id, 0) + 1
            
            print(f"\n🏗️ Worker Distribution:")
            for worker_id, count in sorted(worker_counts.items()):
                percentage = count / len(successful) * 100
                print(f"  {worker_id}: {count} requests ({percentage:.1f}%)")
        
        # Task type distribution
        task_type_counts = {}
        for result in results:
            task_type = result.get("task_type", "unknown")
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
        
        print(f"\n📋 Task Type Distribution:")
        for task_type, count in sorted(task_type_counts.items()):
            percentage = count / total_requests * 100
            print(f"  {task_type}: {count} ({percentage:.1f}%)")
        
        # Show sample successful results
        if successful:
            print(f"\n✅ Sample Successful Results:")
            for i, result in enumerate(successful[:3]):
                print(f"  {i+1}. {result['task_type']} -> {result['prediction']} "
                      f"({result['latency']:.3f}s, {result['worker_id']})")
        
        # Show failures if any
        if failed:
            print(f"\n❌ Failed Requests:")
            for result in failed[:3]:
                error_msg = result.get("error", "unknown error")[:60]
                print(f"  - {result['task_type']}: {error_msg}...")
        
        print("\n" + "=" * 50)
        if success_rate >= 95:
            print("🎉 Excellent performance!")
        elif success_rate >= 80:
            print("👍 Good performance!")
        else:
            print("⚠️ Consider checking system health")

async def main():
    """Main function"""
    print("🧪 Simple Load Tester for Distributed AI Inference System")
    
    tester = SimpleLoadTester()
    
    try:
        # Run basic load test
        await tester.run_simple_load_test(num_requests=10)
        
        print("\n" + "=" * 50)
        print("💡 Next Steps:")
        print("  - Run burst traffic test: python burst_traffic_test.py")
        print("  - Monitor system: python monitor_cli.py")
        print("  - Check status: curl http://localhost:8000/status")
        
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())