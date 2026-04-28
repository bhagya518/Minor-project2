#!/usr/bin/env python3
"""
Real-time Performance Monitoring Dashboard
Monitor throughput, latency, and system resources in real-time
"""

import streamlit as st
import requests
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import json
import threading
import queue

class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self):
        self.data_queue = queue.Queue()
        self.monitoring = False
        
    def collect_metrics(self, ports: List[int]):
        """Collect metrics from all nodes"""
        while self.monitoring:
            timestamp = datetime.now()
            
            for port in ports:
                try:
                    # Health check
                    health_start = time.time()
                    health_response = requests.get(f"http://localhost:{port}/health", timeout=2)
                    health_latency = (time.time() - health_start) * 1000
                    
                    # Reputation metrics
                    reputation_response = requests.get(f"http://localhost:{port}/reputation", timeout=2)
                    reputation_data = reputation_response.json() if reputation_response.status_code == 200 else {}
                    
                    # Consensus metrics
                    consensus_response = requests.get(f"http://localhost:{port}/consensus/reputations", timeout=2)
                    consensus_data = consensus_response.json() if consensus_response.status_code == 200 else {}
                    
                    metrics = {
                        'timestamp': timestamp,
                        'port': port,
                        'health_latency_ms': health_latency,
                        'health_status': health_response.status_code == 200,
                        'reputation_count': len(reputation_data.get('reputations', {})),
                        'engine_type': reputation_data.get('engine_type', 'unknown'),
                        'shard_distribution': reputation_data.get('shard_distribution', {}),
                        'consensus_nodes': len(consensus_data.get('reputations', {}))
                    }
                    
                    self.data_queue.put(metrics)
                    
                except Exception as e:
                    # Node not responding
                    metrics = {
                        'timestamp': timestamp,
                        'port': port,
                        'health_latency_ms': None,
                        'health_status': False,
                        'reputation_count': 0,
                        'engine_type': 'offline',
                        'shard_distribution': {},
                        'consensus_nodes': 0
                    }
                    self.data_queue.put(metrics)
            
            time.sleep(5)  # Collect every 5 seconds
    
    def start_monitoring(self, ports: List[int]):
        """Start monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.collect_metrics, args=(ports,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False

def main():
    st.set_page_config(
        page_title="Performance Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 Real-time Performance Dashboard")
    st.markdown("Monitor throughput, latency, and system resources across the network")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Node configuration
    node_count = st.sidebar.slider("Number of Nodes", 1, 50, 10)
    ports = [8000 + i for i in range(node_count)]
    
    # Website configuration
    website_count = st.sidebar.slider("Websites per Node", 1, 100, 20)
    
    # Monitoring controls
    if st.sidebar.button("Start Monitoring"):
        if 'monitor' not in st.session_state:
            st.session_state.monitor = PerformanceMonitor()
        
        st.session_state.monitor.start_monitoring(ports)
        st.session_state.monitoring_active = True
        st.success(f"Started monitoring {node_count} nodes")
    
    if st.sidebar.button("Stop Monitoring"):
        if 'monitor' in st.session_state:
            st.session_state.monitor.stop_monitoring()
            st.session_state.monitoring_active = False
            st.success("Stopped monitoring")
    
    # Quick deploy section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Deploy")
    if st.sidebar.button("Deploy Test Network"):
        st.info("Run: `python deploy_test_network.py --nodes 10 --websites 20`")
    
    # Initialize session state
    if 'metrics_data' not in st.session_state:
        st.session_state.metrics_data = []
    
    # Collect data from queue
    if 'monitor' in st.session_state and st.session_state.monitoring_active:
        try:
            while not st.session_state.monitor.data_queue.empty():
                metrics = st.session_state.monitor.data_queue.get_nowait()
                st.session_state.metrics_data.append(metrics)
                
                # Keep only last 1000 data points
                if len(st.session_state.metrics_data) > 1000:
                    st.session_state.metrics_data = st.session_state.metrics_data[-1000:]
        except:
            pass
    
    # Convert to DataFrame
    if st.session_state.metrics_data:
        df = pd.DataFrame(st.session_state.metrics_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        df = pd.DataFrame()
    
    # Main dashboard
    if not df.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        active_nodes = df[df['health_status'] == True]['port'].nunique()
        avg_latency = df[df['health_latency_ms'].notna()]['health_latency_ms'].mean()
        total_reputations = df['reputation_count'].sum()
        engine_types = df['engine_type'].value_counts()
        
        col1.metric("🟢 Active Nodes", active_nodes)
        col2.metric("⚡ Avg Latency", f"{avg_latency:.2f} ms" if not pd.isna(avg_latency) else "N/A")
        col3.metric("📊 Total Reputations", total_reputations)
        col4.metric("🤖 ML Engine", engine_types.index[0] if not engine_types.empty else "N/A")
        
        # Charts
        col1, col2 = st.columns(2)
        
        # Latency over time
        with col1:
            fig_latency = px.line(
                df[df['health_latency_ms'].notna()],
                x='timestamp',
                y='health_latency_ms',
                color='port',
                title="Health Check Latency Over Time",
                labels={'health_latency_ms': 'Latency (ms)', 'timestamp': 'Time'}
            )
            fig_latency.update_layout(height=300)
            st.plotly_chart(fig_latency, use_container_width=True)
        
        # Node status
        with col2:
            latest_status = df.groupby('port').last().reset_index()
            latest_status['status'] = latest_status['health_status'].apply(lambda x: '🟢 Online' if x else '🔴 Offline')
            
            fig_status = px.bar(
                latest_status,
                x='port',
                y='reputation_count',
                color='status',
                title="Node Status and Reputation Count",
                labels={'port': 'Node Port', 'reputation_count': 'Reputation Entries'}
            )
            fig_status.update_layout(height=300)
            st.plotly_chart(fig_status, use_container_width=True)
        
        # Shard distribution
        if 'shard_distribution' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Aggregate shard distribution
                shard_data = {}
                for _, row in df.iterrows():
                    if row['shard_distribution']:
                        for shard, count in row['shard_distribution'].items():
                            shard_data[shard] = shard_data.get(shard, 0) + count
                
                if shard_data:
                    fig_shards = px.pie(
                        values=list(shard_data.values()),
                        names=list(shard_data.keys()),
                        title="Shard Distribution"
                    )
                    st.plotly_chart(fig_shards, use_container_width=True)
            
            with col2:
                # Engine type distribution
                fig_engines = px.pie(
                    values=engine_types.values,
                    names=engine_types.index,
                    title="ML Engine Types"
                )
                st.plotly_chart(fig_engines, use_container_width=True)
        
        # Detailed metrics table
        st.subheader("📊 Detailed Metrics")
        
        # Latest metrics per node
        latest_metrics = df.groupby('port').last().reset_index()
        latest_metrics = latest_metrics[['port', 'health_latency_ms', 'health_status', 'reputation_count', 'engine_type']]
        latest_metrics['health_latency_ms'] = latest_metrics['health_latency_ms'].round(2)
        latest_metrics['health_status'] = latest_metrics['health_status'].apply(lambda x: '✅' if x else '❌')
        
        st.dataframe(latest_metrics, use_container_width=True)
        
        # Raw data export
        if st.button("Export Data"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"performance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    else:
        st.info("📊 No data available. Start monitoring to see real-time metrics.")
        
        # Instructions
        st.markdown("""
        ### Getting Started:
        
        1. **Deploy Nodes**: Use the quick deploy script:
           ```bash
           python deploy_test_network.py --nodes 10 --websites 20
           ```
        
        2. **Start Monitoring**: Click "Start Monitoring" in the sidebar
        
        3. **View Metrics**: Real-time charts and statistics will appear here
        
        4. **Scale Testing**: Adjust node count and website count to test scalability
        
        ### Advanced Testing:
        
        For comprehensive performance testing, run:
        ```bash
        python performance_tester.py
        ```
        
        This will test various combinations of nodes and websites and generate detailed reports.
        """)

if __name__ == "__main__":
    main()
