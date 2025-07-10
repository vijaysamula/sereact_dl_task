#!/usr/bin/env python3
"""
CLI Monitor for Distributed AI Inference System
Real-time monitoring of workers, queue, and system performance
"""

import asyncio
import aiohttp
import time
import json
import os
import argparse
from datetime import datetime
from typing import Dict, Any

class CLIMonitor:
    def __init__(self, coordinator_url: str = "http://localhost:8000"):
        self.coordinator_url = coordinator_url
        self.running = False
        self.refresh_interval = 2.0  # seconds
        
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
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def format_timestamp(self, timestamp: float) -> str:
        """Format timestamp for display"""
        return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
    
    def display_header(self):
        """Display monitor header"""
        print("=" * 80)
        print("🚀 DISTRIBUTED AI INFERENCE SYSTEM - LIVE MONITOR")
        print("=" * 80)
        print(f"📡 Coordinator: {self.coordinator_url}")
        print(f"🕐 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    def display_system_overview(self, status: Dict[str, Any]):
        """Display system overview section"""
        if "error" in status:
            print(f"❌ System Status Error: {status['error']}")
            return
        
        print("📊 SYSTEM OVERVIEW")
        print("-" * 40)
        print(f"Queue Size:      {status.get('queue_size', 0):>6}")
        print(f"Active Tasks:    {status.get('active_tasks', 0):>6}")
        print(f"Total Workers:   {status.get('total_workers', 0):>6}")
        print(f"Healthy Workers: {status.get('healthy_workers', 0):>6}")
        print(f"Busy Workers:    {status.get('busy_workers', 0):>6}")
        print(f"Error Workers:   {status.get('error_workers', 0):>6}")
        print()
        
        # System statistics
        stats = status.get("stats", {})
        total_requests = stats.get("total_requests", 0)
        successful = stats.get("successful_requests", 0)
        failed = stats.get("failed_requests", 0)
        success_rate = (successful / total_requests * 100) if total_requests > 0 else 0
        
        print("📈 PERFORMANCE METRICS")
        print("-" * 40)
        print(f"Total Requests:     {total_requests:>8}")
        print(f"Successful:         {successful:>8}")
        print(f"Failed:             {failed:>8}")
        print(f"Success Rate:       {success_rate:>7.1f}%")
        print(f"Queue High Water:   {stats.get('queue_high_watermark', 0):>8}")
        print()
    
    def display_worker_details(self, worker_perf: Dict[str, Any]):
        """Display detailed worker information"""
        if "error" in worker_perf:
            print(f"❌ Worker Performance Error: {worker_perf['error']}")
            return
        
        print("👷 WORKER DETAILS")
        print("-" * 80)
        print(f"{'Worker ID':<12} {'Status':<8} {'Tasks':<6} {'Load%':<6} {'Score':<6} {'Avg Lat':<8} {'Total':<6}")
        print("-" * 80)
        
        for worker_id, perf in worker_perf.items():
            status_emoji = self.get_status_emoji(perf['status'])
            print(f"{worker_id:<12} {status_emoji}{perf['status']:<7} "
                  f"{perf['current_tasks']:<6} {perf['load_percentage']:<5.0f}% "
                  f"{perf['performance_score']:<5.0f} {perf['avg_latency']:<7.3f}s "
                  f"{perf['total_processed']:<6}")
        print()
    
    def get_status_emoji(self, status: str) -> str:
        """Get emoji for worker status"""
        status_emojis = {
            "healthy": "🟢",
            "busy": "🟡",
            "error": "🔴",
            "starting": "🔵"
        }
        return status_emojis.get(status, "⚪")
    
    def display_recent_activity(self, status: Dict[str, Any]):
        """Display recent activity summary"""
        print("📋 RECENT ACTIVITY")
        print("-" * 40)
        
        if "workers_detail" in status:
            active_workers = []
            for worker_id, details in status["workers_detail"].items():
                if details["current_tasks"] > 0:
                    active_workers.append(f"{worker_id}: {details['current_tasks']} tasks")
            
            if active_workers:
                print("Currently Processing:")
                for activity in active_workers:
                    print(f"  🔄 {activity}")
            else:
                print("  💤 No active tasks")
        
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
        print("🚀 Starting CLI Monitor...")
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
                    self.display_worker_details(worker_perf)
                    self.display_recent_activity(status)
                    self.display_controls()
                    
                    # Wait for next refresh
                    await asyncio.sleep(self.refresh_interval)
                    
            except KeyboardInterrupt:
                print("\n👋 Monitor stopped by user")
            except Exception as e:
                print(f"\n❌ Monitor error: {str(e)}")
            finally:
                self.running = False

class InteractiveMonitor(CLIMonitor):
    """Enhanced monitor with interactive features"""
    
    def __init__(self, coordinator_url: str = "http://localhost:8000"):
        super().__init__(coordinator_url)
        self.detailed_view = False
        self.selected_worker = None
    
    async def fetch_task_logs(self, session: aiohttp.ClientSession, limit: int = 10) -> list:
        """Fetch recent task logs"""
        try:
            # This would need to be implemented in the coordinator
            # For now, return empty list
            return []
        except Exception:
            return []
    
    def display_detailed_worker_view(self, worker_id: str, worker_perf: Dict[str, Any]):
        """Display detailed view of a specific worker"""
        if worker_id not in worker_perf:
            print(f"❌ Worker {worker_id} not found")
            return
        
        worker = worker_perf[worker_id]
        print(f"🔍 DETAILED VIEW: {worker_id}")
        print("-" * 60)
        print(f"Status:              {self.get_status_emoji(worker['status'])} {worker['status']}")
        print(f"Current Tasks:       {worker['current_tasks']}/{worker['max_tasks']}")
        print(f"Load Percentage:     {worker['load_percentage']:.1f}%")
        print(f"Performance Score:   {worker['performance_score']:.2f}")
        print(f"Average Latency:     {worker['avg_latency']:.3f}s")
        print(f"Total Processed:     {worker['total_processed']}")
        print()

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description="CLI Monitor for Distributed AI Inference System")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Coordinator URL (default: http://localhost:8000)")
    parser.add_argument("--interval", type=float, default=2.0,
                       help="Refresh interval in seconds (default: 2.0)")
    parser.add_argument("--interactive", action="store_true",
                       help="Enable interactive mode")
    
    args = parser.parse_args()
    
    # Create monitor instance
    if args.interactive:
        monitor = InteractiveMonitor(args.url)
    else:
        monitor = CLIMonitor(args.url)
    
    monitor.refresh_interval = args.interval
    
    # Run the monitor
    try:
        asyncio.run(monitor.run_monitor())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()