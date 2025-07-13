#!/usr/bin/env python3
"""
Enhanced CLI Monitor for Distributed AI Inference System
Real-time monitoring with detailed metrics and worker load distribution
"""

import asyncio
import aiohttp
import time
import os
from datetime import datetime
from typing import Dict, Any

class EnhancedMonitor:
    def __init__(self, coordinator_url: str = "http://localhost:8000"):
        self.coordinator_url = coordinator_url
        self.running = False
        self.refresh_interval = 2.0
        
    async def fetch_status(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Fetch system status"""
        try:
            async with session.get(f"{self.coordinator_url}/status") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def fetch_queue_stats(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Fetch detailed queue statistics"""
        try:
            async with session.get(f"{self.coordinator_url}/queue/stats") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}
    
    def clear_screen(self):
        """Clear terminal"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_header(self):
        """Display header"""
        print("=" * 70)
        print("🤖 DISTRIBUTED AI INFERENCE SYSTEM MONITOR")
        print("=" * 70)
        print(f"📡 URL: {self.coordinator_url}")
        print(f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Mode: Load Balancing (All workers support all task types)")
        print("=" * 70)
    
    def display_system_overview(self, status: Dict[str, Any], queue_stats: Dict[str, Any]):
        """Display comprehensive system overview"""
        if "error" in status:
            print(f"❌ Connection Error: {status['error']}")
            return
        
        print("📊 SYSTEM OVERVIEW")
        print("-" * 40)
        print(f"Queue Size:         {status.get('queue_size', 0):>8}")
        print(f"Active Tasks:       {status.get('active_tasks', 0):>8}")
        print(f"System Load:        {status.get('system_load_percentage', 0):>7.1f}%")
        print(f"Total Capacity:     {status.get('total_capacity', 15):>8}")
        print(f"Healthy Workers:    {status.get('healthy_workers', 0):>8}")
        print(f"Busy Workers:       {status.get('busy_workers', 0):>8}")
        print(f"Error Workers:      {status.get('error_workers', 0):>8}")
        print()
        
        # Performance metrics from queue stats
        if "error" not in queue_stats:
            stats = queue_stats.get('stats', {})
            total = stats.get('total_requests', 0)
            success = stats.get('successful_requests', 0)
            failed = stats.get('failed_requests', 0)
            success_rate = (success / total * 100) if total > 0 else 0
            
            print("📈 PERFORMANCE METRICS")
            print("-" * 40)
            print(f"Total Requests:     {total:>8}")
            print(f"Successful:         {success:>8}")
            print(f"Failed:             {failed:>8}")
            print(f"Success Rate:       {success_rate:>7.1f}%")
            print(f"Queue High Water:   {stats.get('queue_high_watermark', 0):>8}")
            print()
    
    def display_worker_details(self, status: Dict[str, Any], queue_stats: Dict[str, Any]):
        """Display detailed worker information"""
        workers = status.get('workers', {})
        if not workers:
            print("❌ No worker data available")
            return
        
        print("🏗️ WORKER STATUS & LOAD BALANCING")
        print("-" * 80)
        print(f"{'Worker':<10} {'Status':<8} {'Load':<6} {'Util%':<6} {'Latency':<8} {'Processed':<10} {'Score':<6}")
        print("-" * 80)
        
        # Get worker details from queue stats if available
        workers_detail = {}
        if "error" not in queue_stats:
            workers_detail = queue_stats.get('workers_detail', {})
        
        for worker_id, worker_info in workers.items():
            status_emoji = self.get_status_emoji(worker_info.get('status', 'unknown'))
            current_tasks = worker_info.get('current_tasks', 0)
            max_tasks = worker_info.get('max_tasks', 5)
            load_percentage = worker_info.get('load_percentage', 0)
            latency = worker_info.get('avg_latency', 0)
            processed = worker_info.get('total_processed', 0)
            
            # Get additional details from queue stats
            load_score = 0
            if worker_id in workers_detail:
                load_score = workers_detail[worker_id].get('load_score', 0)
            
            load_str = f"{current_tasks}/{max_tasks}"
            load_bar = self.get_load_bar(load_percentage)
            
            print(f"{worker_id:<10} {status_emoji}{worker_info.get('status', 'unknown'):<7} "
                  f"{load_str:<6} {load_bar}{load_percentage:>4.0f}% "
                  f"{latency:>7.3f}s {processed:>9} {load_score:>5.0f}")
        
        print("-" * 80)
        print("📌 Load Score: Lower is better (combines load % + latency)")
        print("📊 Load Bar: ▁ Low ▃ Medium ▅ High ▇ Very High █ Max")
        print()
    
    def display_task_distribution(self, queue_stats: Dict[str, Any]):
        """Display task type distribution"""
        if "error" in queue_stats:
            return
        
        stats = queue_stats.get('stats', {})
        task_dist = stats.get('task_type_distribution', {})
        total_requests = stats.get('total_requests', 0)
        
        if not task_dist or total_requests == 0:
            print("📋 TASK DISTRIBUTION: No tasks processed yet")
            print()
            return
        
        print("📋 TASK TYPE DISTRIBUTION")
        print("-" * 50)
        for task_type, count in task_dist.items():
            percentage = (count / total_requests * 100) if total_requests > 0 else 0
            task_name = task_type.replace('_', ' ').title()
            bar = self.get_percentage_bar(percentage)
            print(f"{task_name:<25} {bar} {count:>5} ({percentage:>4.1f}%)")
        print()
    
    def display_load_balancing_insights(self, queue_stats: Dict[str, Any]):
        """Display load balancing insights"""
        if "error" in queue_stats:
            return
        
        workers_detail = queue_stats.get('workers_detail', {})
        if not workers_detail:
            return
        
        print("🧠 LOAD BALANCING INSIGHTS")
        print("-" * 40)
        
        # Calculate load distribution metrics
        loads = [w.get("load_percentage", 0) for w in workers_detail.values()]
        if loads:
            avg_load = sum(loads) / len(loads)
            max_load = max(loads)
            min_load = min(loads)
            load_imbalance = max_load - min_load
            
            print(f"Average Load:       {avg_load:>7.1f}%")
            print(f"Load Range:         {min_load:>5.1f}% - {max_load:>5.1f}%")
            print(f"Load Imbalance:     {load_imbalance:>7.1f}%")
            
            # Load balancing status
            if load_imbalance < 10:
                print("Status:             ✅ Excellent balance")
            elif load_imbalance < 25:
                print("Status:             📊 Good balance")
            elif load_imbalance < 40:
                print("Status:             ⚠️ Moderate imbalance")
            else:
                print("Status:             🔴 High imbalance")
            
            # Show most/least loaded workers
            sorted_workers = sorted(workers_detail.items(), 
                                  key=lambda x: x[1].get("load_percentage", 0), reverse=True)
            if len(sorted_workers) > 1:
                most_loaded = sorted_workers[0]
                least_loaded = sorted_workers[-1]
                print(f"Most Loaded:        {most_loaded[0]} ({most_loaded[1].get('load_percentage', 0):.1f}%)")
                print(f"Least Loaded:       {least_loaded[0]} ({least_loaded[1].get('load_percentage', 0):.1f}%)")
        print()
    
    def get_status_emoji(self, status: str) -> str:
        """Get emoji for status"""
        emojis = {
            "healthy": "🟢",
            "active": "🟡",
            "busy": "🟠", 
            "error": "🔴",
            "starting": "🔵"
        }
        return emojis.get(status, "⚪")
    
    def get_load_bar(self, load_percentage: float) -> str:
        """Generate visual load bar"""
        if load_percentage <= 20:
            return "▁"
        elif load_percentage <= 40:
            return "▃"
        elif load_percentage <= 60:
            return "▅"
        elif load_percentage <= 80:
            return "▇"
        else:
            return "█"
    
    def get_percentage_bar(self, percentage: float) -> str:
        """Generate percentage bar for task distribution"""
        bar_length = 20
        filled = int((percentage / 100) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        return bar
    
    def display_controls(self):
        """Display controls"""
        print("🎮 CONTROLS")
        print("-" * 30)
        print("  Ctrl+C : Exit monitor")
        print("  Auto refresh every 2 seconds")
        print()
    
    async def run_monitor(self):
        """Main monitor loop"""
        self.running = True
        print("🚀 Starting Enhanced Monitor... Press Ctrl+C to exit")
        print("📊 Collecting system metrics...")
        
        async with aiohttp.ClientSession() as session:
            try:
                while self.running:
                    self.clear_screen()
                    self.display_header()
                    
                    # Fetch both status and queue stats
                    status = await self.fetch_status(session)
                    queue_stats = await self.fetch_queue_stats(session)
                    
                    # Display all sections
                    self.display_system_overview(status, queue_stats)
                    self.display_worker_details(status, queue_stats)
                    self.display_task_distribution(queue_stats)
                    self.display_load_balancing_insights(queue_stats)
                    self.display_controls()
                    
                    await asyncio.sleep(self.refresh_interval)
                    
            except KeyboardInterrupt:
                print("\n👋 Monitor stopped by user")
            except Exception as e:
                print(f"\n❌ Monitor error: {str(e)}")
            finally:
                self.running = False

async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced AI System Monitor")
    parser.add_argument("--url", default="http://localhost:8000",
                       help="Coordinator URL")
    parser.add_argument("--interval", type=float, default=2.0,
                       help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    monitor = EnhancedMonitor(args.url)
    monitor.refresh_interval = args.interval
    
    await monitor.run_monitor()

if __name__ == "__main__":
    asyncio.run(main())