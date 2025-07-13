#!/usr/bin/env python3
"""
Streamlit Dashboard for Distributed AI Inference System
Real-time web interface with metrics and monitoring
"""

import streamlit as st
import requests
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Page config
st.set_page_config(
    page_title="AI Inference Monitor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
COORDINATOR_URL = st.sidebar.text_input("Coordinator URL", "http://localhost:8000")
REFRESH_INTERVAL = st.sidebar.slider("Refresh Interval (seconds)", 1, 10, 3)
AUTO_REFRESH = st.sidebar.checkbox("Auto Refresh", True)

# Initialize session state
if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

def fetch_system_status():
    """Fetch system status from coordinator"""
    try:
        response = requests.get(f"{COORDINATOR_URL}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def fetch_queue_stats():
    """Fetch detailed queue statistics"""
    try:
        response = requests.get(f"{COORDINATOR_URL}/queue/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def update_metrics_history(status_data):
    """Update metrics history for charts"""
    timestamp = datetime.now()
    metrics = {
        'timestamp': timestamp,
        'queue_size': status_data.get('queue_size', 0),
        'active_tasks': status_data.get('active_tasks', 0),
        'system_load': status_data.get('system_load_percentage', 0),
        'healthy_workers': status_data.get('healthy_workers', 0),
        'total_requests': status_data.get('stats', {}).get('total_requests', 0),
        'successful_requests': status_data.get('stats', {}).get('successful_requests', 0),
        'failed_requests': status_data.get('stats', {}).get('failed_requests', 0)
    }
    
    st.session_state.metrics_history.append(metrics)
    
    # Keep only last 50 points
    if len(st.session_state.metrics_history) > 50:
        st.session_state.metrics_history = st.session_state.metrics_history[-50:]

def display_status_indicators(status_data):
    """Display key status indicators"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        queue_size = status_data.get('queue_size', 0)
        st.metric("Queue Size", queue_size, delta=None)
    
    with col2:
        active_tasks = status_data.get('active_tasks', 0)
        st.metric("Active Tasks", active_tasks, delta=None)
    
    with col3:
        healthy_workers = status_data.get('healthy_workers', 0)
        total_workers = status_data.get('total_workers', 3)
        st.metric("Healthy Workers", f"{healthy_workers}/{total_workers}")
    
    with col4:
        system_load = status_data.get('system_load_percentage', 0)
        st.metric("System Load", f"{system_load:.1f}%")

def display_worker_status(status_data):
    """Display worker status table"""
    workers = status_data.get('workers', {})
    if not workers:
        st.warning("No worker data available")
        return
    
    worker_data = []
    for worker_id, worker_info in workers.items():
        worker_data.append({
            'Worker': worker_id,
            'Status': worker_info.get('status', 'unknown'),
            'Current Tasks': worker_info.get('current_tasks', 0),
            'Max Tasks': worker_info.get('max_tasks', 5),
            'Load %': worker_info.get('load_percentage', 0),
            'Avg Latency': f"{worker_info.get('avg_latency', 0):.3f}s",
            'Total Processed': worker_info.get('total_processed', 0)
        })
    
    df = pd.DataFrame(worker_data)
    
    # Color code status
    def color_status(val):
        if val == 'healthy':
            return 'background-color: #d4edda'
        elif val == 'active':
            return 'background-color: #fff3cd'
        elif val == 'busy':
            return 'background-color: #f8d7da'
        elif val == 'error':
            return 'background-color: #f5c6cb'
        return ''
    
    styled_df = df.style.applymap(color_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True)

def display_performance_charts():
    """Display performance charts"""
    if not st.session_state.metrics_history:
        st.info("No metrics data available yet. Wait for data collection...")
        return
    
    df = pd.DataFrame(st.session_state.metrics_history)
    
    # Queue and Active Tasks Chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig_queue = px.line(df, x='timestamp', y=['queue_size', 'active_tasks'],
                           title='Queue Size & Active Tasks Over Time',
                           labels={'value': 'Count', 'timestamp': 'Time'})
        st.plotly_chart(fig_queue, use_container_width=True)
    
    with col2:
        fig_load = px.line(df, x='timestamp', y='system_load',
                          title='System Load Over Time',
                          labels={'system_load': 'Load %', 'timestamp': 'Time'})
        st.plotly_chart(fig_load, use_container_width=True)
    
    # Success Rate Chart
    df['success_rate'] = (df['successful_requests'] / df['total_requests'] * 100).fillna(0)
    fig_success = px.line(df, x='timestamp', y='success_rate',
                         title='Success Rate Over Time',
                         labels={'success_rate': 'Success Rate %', 'timestamp': 'Time'})
    st.plotly_chart(fig_success, use_container_width=True)

def display_task_distribution(queue_stats):
    """Display task type distribution"""
    task_dist = queue_stats.get('stats', {}).get('task_type_distribution', {})
    if not task_dist or not any(task_dist.values()):
        st.info("No task distribution data available")
        return
    
    # Create pie chart
    fig_pie = px.pie(
        values=list(task_dist.values()),
        names=[name.replace('_', ' ').title() for name in task_dist.keys()],
        title='Task Type Distribution'
    )
    st.plotly_chart(fig_pie, use_container_width=True)

def send_test_request():
    """Send test request"""
    with st.form("test_request_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            task_type = st.selectbox("Task Type", [
                "text_classification",
                "image_classification", 
                "clip_text_similarity",
                "clip_text_to_image"
            ])
        
        with col2:
            test_data = st.text_input("Test Data", "I love this product!")
        
        submit = st.form_submit_button("Send Test Request")
        
        if submit:
            try:
                request_data = {
                    "request_id": f"streamlit-test-{int(time.time())}",
                    "task_type": task_type,
                    "data": test_data
                }
                
                response = requests.post(
                    f"{COORDINATOR_URL}/infer",
                    json=request_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Request successful!")
                    st.json(result)
                else:
                    st.error(f"❌ Request failed: HTTP {response.status_code}")
                    st.text(response.text)
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def main():
    """Main dashboard function"""
    # Header
    st.title("🤖 AI Inference System Dashboard")
    st.markdown("Real-time monitoring of distributed AI inference system")
    
    # Fetch data
    status_data = fetch_system_status()
    queue_stats = fetch_queue_stats()
    
    # Check for errors
    if "error" in status_data:
        st.error(f"❌ Cannot connect to coordinator: {status_data['error']}")
        st.info("Make sure the system is running with: `docker compose up`")
        return
    
    # Update metrics history
    if "error" not in queue_stats:
        update_metrics_history(queue_stats)
    
    st.session_state.last_update = datetime.now()
    
    # Display last update time
    st.sidebar.markdown(f"**Last Update:** {st.session_state.last_update.strftime('%H:%M:%S')}")
    
    # Status indicators
    st.header("📊 System Status")
    display_status_indicators(status_data)
    
    # Worker status
    st.header("🏗️ Worker Status")
    display_worker_status(status_data)
    
    # Performance charts
    st.header("📈 Performance Metrics")
    display_performance_charts()
    
    # Task distribution
    if "error" not in queue_stats:
        st.header("📋 Task Distribution")
        display_task_distribution(queue_stats)
    
    # Test request section
    st.header("🧪 Send Test Request")
    send_test_request()
    
    # Performance stats
    if "error" not in status_data:
        stats = status_data.get('stats', {})
        if stats:
            st.header("📊 Overall Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total = stats.get('total_requests', 0)
                st.metric("Total Requests", total)
            
            with col2:
                success = stats.get('successful_requests', 0)
                st.metric("Successful", success)
            
            with col3:
                failed = stats.get('failed_requests', 0)
                st.metric("Failed", failed)
            
            if total > 0:
                success_rate = (success / total) * 100
                st.progress(success_rate / 100)
                st.caption(f"Success Rate: {success_rate:.1f}%")
    
    # Auto refresh
    if AUTO_REFRESH:
        time.sleep(REFRESH_INTERVAL)
        st.rerun()

if __name__ == "__main__":
    main()