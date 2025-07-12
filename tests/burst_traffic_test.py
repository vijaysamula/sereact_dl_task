#!/usr/bin/env python3
"""
Burst Traffic Test for Distributed AI Inference System
Simulates realistic traffic patterns with bursts and valleys
"""

import asyncio
import aiohttp
import time
import json
import random
import uuid
from typing import List, Dict, Tuple
import argparse

class BurstTrafficTester:
    def __init__(self, coordinator_url: str = "http://localhost:8000"):
        self.coordinator_url = coordinator_url
        self.results = []
        self.test_data = {
            "text_samples": [
                "I love this product!",
                "This is disappointing.",
                "Great quality and fast delivery.",
                "Not worth the money.",
                "Amazing customer service!",
                "Poor build quality.",
                "Highly recommended!",
                "Terrible experience.",
                "Perfect for my needs.",
                "Complete waste of time."
            ],
            "clip_queries": [
                "a photo of a cat",
                "beautiful sunset landscape",
                "delicious food plate",
                "busy city street",
                "cute animals playing",
                "modern architecture",
                "nature photography",
                "people working together",
                "colorful artwork",
                "technology devices"
            ],
            "image_requests": [
                "landscape_photo_001",
                "portrait_image_002", 
                "object_detection_003",
                "scene_analysis_004",
                "artistic_image_005"
            ]
        }
    
    def generate_request(self, request_id: str = None) -> Dict:
        """Generate a random inference request"""
        if not request_id:
            request_id = f"burst-{uuid.uuid4().hex[:8]}"
        
        # Random task type selection
        task_types = ["text_classification", "image_classification", "clip_text_similarity", "clip_text_to_image"]
        task_type = random.choice(task_types)
        
        # Select appropriate data
        if task_type == "text_classification":
            data = random.choice(self.test_data["text_samples"])
        elif task_type == "image_classification":
            data = f"dummy_image_{random.choice(self.test_data['image_requests'])}"
        elif task_type in ["clip_text_similarity", "clip_text_to_image"]:
            data = random.choice(self.test_data["clip_queries"])
        else:
            data = random.choice(self.test_data["text_samples"])
        
        return {
            "request_id": request_id,
            "task_type": task_type,
            "data": data
        }
    
    async def send_single_request(self, session: aiohttp.ClientSession, request_id: str = None) -> Dict:
        """Send a single inference request"""
        request_data = self.generate_request(request_id)
        start_time = time.time()
        
        try:
            async with session.post(
                f"{self.coordinator_url}/infer",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                end_time = time.time()
                
                result = {
                    "request_id": request_data["request_id"],
                    "task_type": request_data["task_type"],
                    "latency": end_time - start_time,
                    "timestamp": start_time,
                    "status": "success" if response.status == 200 else "error",
                    "status_code": response.status
                }
                
                if response.status == 200:
                    response_data = await response.json()
                    result["worker_id"] = response_data.get("result", {}).get("worker_id", "unknown")
                    result["prediction"] = response_data.get("result", {}).get("prediction", "unknown")
                else:
                    result["error"] = await response.text()
                
                return result
                
        except Exception as e:
            end_time = time.time()
            return {
                "request_id": request_data["request_id"],
                "task_type": request_data["task_type"],
                "latency": end_time - start_time,
                "timestamp": start_time,
                "status": "error",
                "error": str(e),
                "status_code": 0
            }
    
    async def burst_pattern(self, session: aiohttp.ClientSession, burst_size: int, 
                          burst_duration: float) -> List[Dict]:
        """Execute a burst of requests"""
        print(f"🚀 Starting burst: {burst_size} requests over {burst_duration:.1f}s")
        
        tasks = []
        burst_start = time.time()
        
        for i in range(burst_size):
            # Create request
            task = self.send_single_request(session, f"burst-{int(burst_start)}-{i}")
            tasks.append(task)
            
            # Add small delay between requests in burst
            if i < burst_size - 1:
                await asyncio.sleep(burst_duration / burst_size)
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "status": "error",
                    "error": str(result),
                    "latency": 0,
                    "timestamp": time.time()
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def sustained_load_pattern(self, session: aiohttp.ClientSession, 
                                   requests_per_second: float, duration: float) -> List[Dict]:
        """Execute sustained load pattern"""
        print(f"⚡ Sustained load: {requests_per_second:.1f} req/s for {duration:.1f}s")
        
        interval = 1.0 / requests_per_second
        end_time = time.time() + duration
        tasks = []
        request_count = 0
        
        while time.time() < end_time:
            task = self.send_single_request(session, f"sustained-{int(time.time())}-{request_count}")
            tasks.append(task)
            request_count += 1
            
            await asyncio.sleep(interval)
        
        # Wait for completion
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "status": "error",
                    "error": str(result),
                    "latency": 0,
                    "timestamp": time.time()
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def check_system_status(self, session: aiohttp.ClientSession) -> Dict:
        """Check system status"""
        try:
            async with session.get(f"{self.coordinator_url}/queue/stats") as response:
                if response.status == 200:
                    return await response.json()
                return {"error": f"Status check failed: {response.status}"}
        except Exception as e:
            return {"error": f"Status check error: {str(e)}"}
    
    def print_burst_results(self, results: List[Dict], test_name: str):
        """Print results of a burst test"""
        if not results:
            print(f"❌ No results for {test_name}")
            return
        
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "error"]
        
        success_rate = len(successful) / len(results) * 100
        avg_latency = sum(r["latency"] for r in successful) / len(successful) if successful else 0
        
        print(f"\n📊 {test_name} Results:")
        print(f"   Total requests: {len(results)}")
        print(f"   Successful: {len(successful)} ({success_rate:.1f}%)")
        print(f"   Failed: {len(failed)}")
        print(f"   Average latency: {avg_latency:.3f}s")
        
        if successful:
            latencies = [r["latency"] for r in successful]
            print(f"   Min latency: {min(latencies):.3f}s")
            print(f"   Max latency: {max(latencies):.3f}s")
        
        # Worker distribution
        if successful:
            worker_counts = {}
            for result in successful:
                worker_id = result.get("worker_id", "unknown")
                worker_counts[worker_id] = worker_counts.get(worker_id, 0) + 1
            
            print(f"   Worker distribution:")
            for worker_id, count in sorted(worker_counts.items()):
                percentage = count / len(successful) * 100
                print(f"     {worker_id}: {count} ({percentage:.1f}%)")
        
        # Print failures if any
        if failed:
            print(f"   Failure details:")
            error_counts = {}
            for result in failed:
                error = result.get("error", "unknown error")
                error_counts[error] = error_counts.get(error, 0) + 1
            
            for error, count in error_counts.items():
                print(f"     {error}: {count} times")
    
    async def run_traffic_simulation(self, pattern: str = "mixed"):
        """Run complete traffic simulation"""
        print("🌊 BURST TRAFFIC SIMULATION")
        print("=" * 60)
        print(f"📡 Target: {self.coordinator_url}")
        print(f"🎯 Pattern: {pattern}")
        print("=" * 60)
        
        async with aiohttp.ClientSession() as session:
            # Check initial system status
            print("\n📊 Initial system status:")
            initial_status = await self.check_system_status(session)
            if "error" not in initial_status:
                print(f"   Queue size: {initial_status.get('queue_size', 0)}")
                print(f"   Healthy workers: {initial_status.get('healthy_workers', 0)}")
            else:
                print(f"   ❌ {initial_status['error']}")
            
            all_results = []
            
            if pattern in ["mixed", "burst"]:
                # Test 1: Small burst
                print(f"\n1️⃣ Small Burst Test")
                results1 = await self.burst_pattern(session, burst_size=5, burst_duration=2.0)
                all_results.extend(results1)
                self.print_burst_results(results1, "Small Burst (5 requests)")
                
                await asyncio.sleep(3)  # Rest period
                
                # Test 2: Medium burst
                print(f"\n2️⃣ Medium Burst Test")
                results2 = await self.burst_pattern(session, burst_size=15, burst_duration=5.0)
                all_results.extend(results2)
                self.print_burst_results(results2, "Medium Burst (15 requests)")
                
                await asyncio.sleep(5)  # Rest period
                
                # Test 3: Large burst
                print(f"\n3️⃣ Large Burst Test")
                results3 = await self.burst_pattern(session, burst_size=25, burst_duration=3.0)
                all_results.extend(results3)
                self.print_burst_results(results3, "Large Burst (25 requests)")
            
            if pattern in ["mixed", "sustained"]:
                await asyncio.sleep(5)  # Rest period
                
                # Test 4: Sustained load
                print(f"\n4️⃣ Sustained Load Test")
                results4 = await self.sustained_load_pattern(session, requests_per_second=2.0, duration=10.0)
                all_results.extend(results4)
                self.print_burst_results(results4, "Sustained Load (2 req/s for 10s)")
            
            # Final status check
            print(f"\n📊 Final system status:")
            final_status = await self.check_system_status(session)
            if "error" not in final_status:
                print(f"   Queue size: {final_status.get('queue_size', 0)}")
                print(f"   Active tasks: {final_status.get('active_tasks', 0)}")
                stats = final_status.get('stats', {})
                print(f"   Total processed: {stats.get('successful_requests', 0)}")
                print(f"   Total failed: {stats.get('failed_requests', 0)}")
            else:
                print(f"   ❌ {final_status['error']}")
            
            # Overall summary
            self.print_overall_summary(all_results)
    
    def print_overall_summary(self, all_results: List[Dict]):
        """Print overall test summary"""
        if not all_results:
            print("\n❌ No results to summarize")
            return
        
        print(f"\n{'='*60}")
        print("🎯 OVERALL TRAFFIC SIMULATION SUMMARY")
        print(f"{'='*60}")
        
        successful = [r for r in all_results if r["status"] == "success"]
        failed = [r for r in all_results if r["status"] == "error"]
        
        total_requests = len(all_results)
        success_rate = len(successful) / total_requests * 100 if total_requests > 0 else 0
        
        print(f"📈 Overall Performance:")
        print(f"   Total requests sent: {total_requests}")
        print(f"   Successful: {len(successful)} ({success_rate:.1f}%)")
        print(f"   Failed: {len(failed)}")
        
        if successful:
            latencies = [r["latency"] for r in successful]
            avg_latency = sum(latencies) / len(latencies)
            
            print(f"   Average latency: {avg_latency:.3f}s")
            print(f"   Min latency: {min(latencies):.3f}s")
            print(f"   Max latency: {max(latencies):.3f}s")
            
            # Latency percentiles
            sorted_latencies = sorted(latencies)
            p50 = sorted_latencies[len(sorted_latencies)//2]
            p95 = sorted_latencies[int(len(sorted_latencies)*0.95)]
            p99 = sorted_latencies[int(len(sorted_latencies)*0.99)]
            
            print(f"   50th percentile: {p50:.3f}s")
            print(f"   95th percentile: {p95:.3f}s")
            print(f"   99th percentile: {p99:.3f}s")

def main():
    parser = argparse.ArgumentParser(description="Burst Traffic Tester for Distributed AI System")
    parser.add_argument("--url", default="http://localhost:8000",
                       help="Coordinator URL (default: http://localhost:8000)")
    parser.add_argument("--pattern", choices=["burst", "sustained", "mixed"], default="mixed",
                       help="Traffic pattern to test (default: mixed)")
    
    args = parser.parse_args()
    
    tester = BurstTrafficTester(args.url)
    
    try:
        asyncio.run(tester.run_traffic_simulation(args.pattern))
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")

if __name__ == "__main__":
    main()