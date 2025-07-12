# Updated monitor_cli.py for Load Balancing Mode

#!/usr/bin/env python3
"""
CLI Monitor for Distributed AI Inference System - Load Balancing Mode
Real-time monitoring of load distribution, workers, and system performance
"""

import asyncio
import aiohttp
import time
import json
import os
import argparse
from datetime import datetime
from typing import Dict, Any

class LoadBalancingMonitor:
    def __init__(self, coordinator_url: str = "http://localhost:8000"):
        self.coordinator_url = coordinator_url
        self.running = False
        self.refresh_interval = 2.0
        
    async def fetch_system_status(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Fetch current system status"""
        try:
            async with session.get(f"{self.coordinator_url}/queue/stats") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Failed to fetch status: {response.status}"}
        except Exception as e:
            return {"error": f"Connection failed: {str(e)}"}
    
    async def fetch_worker_performance(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Fetch worker performance metrics"""
        try:
            async with session.get(f"{self.coordinator_url}/workers/performance") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Failed to fetch worker performance: {response.status}"}
        except Exception as e:
            return {"error": f"Worker performance fetch failed: {str(e)}"}
    
    async def fetch_load_balancing_info(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Fetch load balancing information"""
        try:
            async with session.get(f"{self.coordinator_url}/load-balancing/info") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Failed to fetch load balancing info: {response.status}"}
        except Exception as e:
            return {"error": f"Load balancing info fetch failed: {str(e)}"}
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_header(self):
        """Display monitor header"""
        print("=" * 80)
        print("⚖️  DISTRIBUTED AI INFERENCE SYSTEM - LOAD BALANCING MONITOR")
        print("=" * 80)
        print(f"📡 Coordinator: {self.coordinator_url}")
        print(f"🕐 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Mode: Pure Load Balancing (All workers support all task types)")
        print("=" * 80)
    
    def display_system_overview(self, status: Dict[str, Any]):
        """Display system overview with load balancing metrics"""
        if "error" in status:
            print(f"❌ System Status Error: {status['error']}")
            return
        
        print("📊 LOAD BALANCING OVERVIEW")
        print("-" * 50)
        print(f"Queue Size:              {status.get('queue_size', 0):>6}")
        print(f"Active Tasks:            {status.get('active_tasks', 0):>6}")
        print(f"System Load:             {status.get('system_load_percentage', 0):>5.1f}%")
        print(f"Total Current Tasks:     {status.get('total_current_tasks', 0):>6}")
        print(f"Total Capacity:          {status.get('total_capacity', 0):>6}")
        print(f"Available Workers:       {status.get('healthy_workers', 0):>6}")
        print(f"Busy Workers:            {status.get('busy_workers', 0):>6}")
        print(f"Error Workers:           {status.get('error_workers', 0):>6}")
        print()
        
        # Performance statistics
        stats = status.get("stats", {})
        total_requests = stats.get("total_requests", 0)
        successful = stats.get("successful_requests", 0)
        failed = stats.get("failed_requests", 0)
        success_rate = (successful / total_requests * 100) if total_requests > 0 else 0
        
        print("📈 PERFORMANCE METRICS")
        print("-" * 50)
        print(f"Total Requests:          {total_requests:>8}")
        print(f"Successful:              {successful:>8}")
        print(f"Failed:                  {failed:>8}")
        print(f"Success Rate:            {success_rate:>7.1f}%")
        print(f"Queue High Water:        {stats.get('queue_high_watermark', 0):>8}")
        print()
        
        # Task type distribution
        task_dist = stats.get("task_type_distribution", {})
        if any(task_dist.values()):
            print("📋 TASK TYPE DISTRIBUTION")
            print("-" * 50)
            for task_type, count in task_dist.items():
                percentage = (count / total_requests * 100) if total_requests > 0 else 0
                print(f"{task_type.replace('_', ' ').title():<25} {count:>5} ({percentage:>4.1f}%)")
            print()
    
    def display_worker_load_details(self, worker_perf: Dict[str, Any]):
        """Display detailed worker load information"""
        if "error" in worker_perf:
            print(f"❌ Worker Performance Error: {worker_perf['error']}")
            return
        
        print("⚖️  WORKER LOAD BALANCING STATUS")
        print("-" * 85)
        print(f"{'Worker':<10} {'Status':<8} {'Load':<8} {'Cap':<5} {'Util%':<6} {'Score':<6} {'Latency':<8} {'Total':<6}")
        print("-" * 85)
        
        for worker_id, perf in worker_perf.items():
            status_emoji = self.get_status_emoji(perf['status'])
            load_bar = self.get_load_bar(perf['load_percentage'])
            
            print(f"{worker_id:<10} {status_emoji}{perf['status']:<7} "
                  f"{perf['current_tasks']}/{perf['max_tasks']:<6} {perf['max_tasks']:<5} "
                  f"{load_bar}{perf['load_percentage']:>4.0f}% "
                  f"{perf['load_score']:<5.0f} {perf['avg_latency']:<7.3f}s "
                  f"{perf['total_processed']:<6}")
        
        print("-" * 85)
        print("📌 All workers support: Text Classification, Image Classification, CLIP Tasks")
        print("📊 Load Score: Lower is better (combines load percentage + latency)")
        print()
    
    def get_status_emoji(self, status: str) -> str:
        """Get emoji for worker status"""
        status_emojis = {
            "healthy": "🟢",
            "active": "🟡", 
            "busy": "🟠",
            "error": "🔴",
            "starting": "🔵"
        }
        return status_emojis.get(status, "⚪")
    
    def get_load_bar(self, load_percentage: float) -> str:
        """Generate a visual load bar"""
        if load_percentage <= 25:
            return "▁"
        elif load_percentage <= 50:
            return "▃"
        elif load_percentage <= 75:
            return "▅"
        elif load_percentage <= 90:
            return "▇"
        else:
            return "█"
    
    def display_load_balancing_insights(self, status: Dict[str, Any]):
        """Display load balancing insights and recommendations"""
        if "error" in status:
            return
            
        print("🧠 LOAD BALANCING INSIGHTS")
        print("-" * 50)
        
        workers_detail = status.get("workers_detail", {})
        if workers_detail:
            # Calculate load distribution
            loads = [w["load_percentage"] for w in workers_detail.values()]
            avg_load = sum(loads) / len(loads)
            max_load = max(loads)
            min_load = min(loads)
            load_imbalance = max_load - min_load
            
            print(f"Average Load:            {avg_load:>7.1f}%")
            print(f"Load Range:              {min_load:>5.1f}% - {max_load:>5.1f}%")
            print(f"Load Imbalance:          {load_imbalance:>7.1f}%")
            
            # Provide insights
            if load_imbalance > 30:
                print("⚠️  High load imbalance detected")
            elif load_imbalance < 10:
                print("✅ Well-balanced load distribution")
            else:
                print("📊 Moderate load distribution")
            
            # Show most/least loaded workers
            sorted_workers = sorted(workers_detail.items(), 
                                  key=lambda x: x[1]["load_percentage"], reverse=True)
            if sorted_workers:
                most_loaded = sorted_workers[0]
                least_loaded = sorted_workers[-1]
                print(f"Most Loaded:             {most_loaded[0]} ({most_loaded[1]['load_percentage']:.1f}%)")
                print(f"Least Loaded:            {least_loaded[0]} ({least_loaded[1]['load_percentage']:.1f}%)")
        print()
    
    def display_controls(self):
        """Display control instructions"""
        print("🎮 CONTROLS")
        print("-" * 40)
        print("  Ctrl+C : Exit monitor")
        print("  r      : Force refresh")
        print("  q      : Quit")
        print()
    
    async def run_monitor(self):
        """Main monitor loop"""
        self.running = True
        print("⚖️  Starting Load Balancing Monitor...")
        print("Press Ctrl+C to exit")
        
        async with aiohttp.ClientSession() as session:
            try:
                while self.running:
                    # Clear screen and display header
                    self.clear_screen()
                    self.display_header()
                    
                    # Fetch data
                    status = await self.fetch_system_status(session)
                    worker_perf = await self.fetch_worker_performance(session)
                    
                    # Display sections
                    self.display_system_overview(status)
                    self.display_worker_load_details(worker_perf)
                    self.display_load_balancing_insights(status)
                    self.display_controls()
                    
                    # Wait for next refresh
                    await asyncio.sleep(self.refresh_interval)
                    
            except KeyboardInterrupt:
                print("\n👋 Load Balancing Monitor stopped by user")
            except Exception as e:
                print(f"\n❌ Monitor error: {str(e)}")
            finally:
                self.running = False

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description="Load Balancing Monitor for Distributed AI System")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Coordinator URL (default: http://localhost:8000)")
    parser.add_argument("--interval", type=float, default=2.0,
                       help="Refresh interval in seconds (default: 2.0)")
    
    args = parser.parse_args()
    
    monitor = LoadBalancingMonitor(args.url)
    monitor.refresh_interval = args.interval
    
    try:
        asyncio.run(monitor.run_monitor())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()