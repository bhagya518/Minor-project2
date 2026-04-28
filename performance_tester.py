#!/usr/bin/env python3
"""
Performance Testing Framework for Decentralized Monitoring System
Measures throughput, latency, and scalability with variable nodes/websites
"""

import asyncio
import time
import json
import statistics
import concurrent.futures
import subprocess
import psutil
import requests
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    node_count: int
    website_count: int
    throughput_tps: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    consensus_time_ms: float
    ml_inference_time_ms: float
    blockchain_tx_time_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    network_io_mb: float

class PerformanceTester:
    """Comprehensive performance testing framework"""
    
    def __init__(self, base_port: int = 8000, max_nodes: int = 100):
        self.base_port = base_port
        self.max_nodes = max_nodes
        self.active_processes = {}
        self.metrics_history = []
        self.test_results = {}
        
        # Test configurations
        self.node_counts = [4, 8, 16, 32, 50, 100]
        self.website_counts = [3, 10, 25, 50, 100]
        
        # Website pools for testing
        self.website_pools = {
            3: ["https://google.com", "https://github.com", "https://stackoverflow.com"],
            10: [
                "https://google.com", "https://github.com", "https://stackoverflow.com",
                "https://stackoverflow.com", "https://httpbin.org", "https://api.github.com",
                "https://jsonplaceholder.typicode.com", "https://reqres.in",
                "https://publicapis.dev", "https://catfact.ninja/fact"
            ],
            25: [
                "https://google.com", "https://github.com", "https://stackoverflow.com",
                "https://httpbin.org", "https://api.github.com", "https://jsonplaceholder.typicode.com",
                "https://reqres.in", "https://publicapis.dev", "https://catfact.ninja/fact",
                "https://api.coindesk.com/v1/bpi/currentprice.json", "https://api.exchangerate-api.com/v4/latest/USD",
                "https://api.weather.gov/stations/KSFO/observations/latest", "https://api.nasa.gov/planetary/apod",
                "https://api.spacexdata.com/v4/launches/latest", "https://api.github.com/users/octocat",
                "https://api.github.com/repos/microsoft/vscode", "https://api.github.com/repos/facebook/react",
                "https://api.github.com/repos/tensorflow/tensorflow", "https://api.github.com/repos/python/cpython",
                "https://jsonplaceholder.typicode.com/posts/1", "https://jsonplaceholder.typicode.com/users/1",
                "https://reqres.in/api/users/1", "https://reqres.in/api/products/1",
                "https://httpbin.org/get", "https://httpbin.org/post", "https://httpbin.org/put"
            ],
            50: [
                # Extend 25 + 25 more diverse endpoints
                "https://google.com", "https://github.com", "https://stackoverflow.com",
                "https://httpbin.org", "https://api.github.com", "https://jsonplaceholder.typicode.com",
                "https://reqres.in", "https://publicapis.dev", "https://catfact.ninja/fact",
                "https://api.coindesk.com/v1/bpi/currentprice.json", "https://api.exchangerate-api.com/v4/latest/USD",
                "https://api.weather.gov/stations/KSFO/observations/latest", "https://api.nasa.gov/planetary/apod",
                "https://api.spacexdata.com/v4/launches/latest", "https://api.github.com/users/octocat",
                "https://api.github.com/repos/microsoft/vscode", "https://api.github.com/repos/facebook/react",
                "https://api.github.com/repos/tensorflow/tensorflow", "https://api.github.com/repos/python/cpython",
                "https://jsonplaceholder.typicode.com/posts/1", "https://jsonplaceholder.typicode.com/users/1",
                "https://reqres.in/api/users/1", "https://reqres.in/api/products/1",
                "https://httpbin.org/get", "https://httpbin.org/post", "https://httpbin.org/put",
                "https://api.chucknorris.io/jokes/random", "https://api.quotable.io/random",
                "https://api.github.com/zen", "https://api.github.com/octocat",
                "https://api.github.com/emojis", "https://api.github.com/events",
                "https://api.github.com/gists/public", "https://api.github.com/search/repositories?q=stars:>10000",
                "https://api.github.com/rate_limit", "https://api.github.com/user",
                "https://jsonplaceholder.typicode.com/comments", "https://jsonplaceholder.typicode.com/albums",
                "https://jsonplaceholder.typicode.com/photos", "https://jsonplaceholder.typicode.com/todos",
                "https://reqres.in/api/unknown", "https://reqres.in/api/register",
                "https://httpbin.org/patch", "https://httpbin.org/delete",
                "https://httpbin.org/status/200", "https://httpbin.org/status/404",
                "https://httpbin.org/status/500", "https://httpbin.org/delay/1",
                "https://httpbin.org/uuid", "https://httpbin.org/ip"
            ],
            100: [
                # Extend 50 + 50 more endpoints for stress testing
                "https://google.com", "https://github.com", "https://stackoverflow.com",
                "https://httpbin.org", "https://api.github.com", "https://jsonplaceholder.typicode.com",
                "https://reqres.in", "https://publicapis.dev", "https://catfact.ninja/fact",
                "https://api.coindesk.com/v1/bpi/currentprice.json", "https://api.exchangerate-api.com/v4/latest/USD",
                "https://api.weather.gov/stations/KSFO/observations/latest", "https://api.nasa.gov/planetary/apod",
                "https://api.spacexdata.com/v4/launches/latest", "https://api.github.com/users/octocat",
                "https://api.github.com/repos/microsoft/vscode", "https://api.github.com/repos/facebook/react",
                "https://api.github.com/repos/tensorflow/tensorflow", "https://api.github.com/repos/python/cpython",
                "https://jsonplaceholder.typicode.com/posts/1", "https://jsonplaceholder.typicode.com/users/1",
                "https://reqres.in/api/users/1", "https://reqres.in/api/products/1",
                "https://httpbin.org/get", "https://httpbin.org/post", "https://httpbin.org/put",
                "https://api.chucknorris.io/jokes/random", "https://api.quotable.io/random",
                "https://api.github.com/zen", "https://api.github.com/octocat",
                "https://api.github.com/emojis", "https://api.github.com/events",
                "https://api.github.com/gists/public", "https://api.github.com/search/repositories?q=stars:>10000",
                "https://api.github.com/rate_limit", "https://api.github.com/user",
                "https://jsonplaceholder.typicode.com/comments", "https://jsonplaceholder.typicode.com/albums",
                "https://jsonplaceholder.typicode.com/photos", "https://jsonplaceholder.typicode.com/todos",
                "https://reqres.in/api/unknown", "https://reqres.in/api/register",
                "https://httpbin.org/patch", "https://httpbin.org/delete",
                "https://httpbin.org/status/200", "https://httpbin.org/status/404",
                "https://httpbin.org/status/500", "https://httpbin.org/delay/1",
                "https://httpbin.org/uuid", "https://httpbin.org/ip",
                # Additional endpoints for 100 total
                "https://api.github.com/repos/microsoft/TypeScript", "https://api.github.com/repos/facebook/create-react-app",
                "https://api.github.com/repos/vuejs/vue", "https://api.github.com/repos/angular/angular",
                "https://api.github.com/repos/django/django", "https://api.github.com/repos/rails/rails",
                "https://jsonplaceholder.typicode.com/posts", "https://jsonplaceholder.typicode.com/users",
                "https://reqres.in/api/users", "https://reqres.in/api/products",
                "https://httpbin.org/anything", "https://httpbin.org/bearer",
                "https://httpbin.org/cache", "https://httpbin.org/cookies",
                "https://httpbin.org/encoding/utf8", "https://httpbin.org/gzip",
                "https://httpbin.org/html", "https://httpbin.org/json",
                "https://httpbin.org/robots.txt", "https://httpbin.org/route/1",
                "https://httpbin.org/stream/5", "https://httpbin.org/user-agent",
                "https://httpbin.org/xml", "https://httpbin.org/referrer",
                "https://api.github.com/repos/kubernetes/kubernetes", "https://api.github.com/repos/golang/go",
                "https://api.github.com/repos/nodejs/node", "https://api.github.com/repos/rust-lang/rust",
                "https://jsonplaceholder.typicode.com/albums/1", "https://jsonplaceholder.typicode.com/photos/1",
                "https://reqres.in/api/users/2", "https://reqres.in/api/products/2",
                "https://httpbin.org/base64/SFRUUEJJTiBpcyBhd2Vzb21l", "https://httpbin.org/brotli",
                "https://httpbin.org/deflate", "https://httpbin.org/deny",
                "https://httpbin.org/digest-auth/auth/user/passwd", "https://httpbin.org/hidden-basic-auth/user/passwd",
                "https://httpbin.org/image/jpeg", "https://httpbin.org/image/png",
                "https://httpbin.org/image/svg", "https://httpbin.org/image/webp",
                "https://httpbin.org/links/10", "https://httpbin.org/links/5",
                "https://httpbin.org/method", "https://httpbin.org/response-headers?Content-Type=text/plain",
                "https://httpbin.org/stream-bytes/1024", "https://httpbin.org/throttle/0.1",
                "https://httpbin.org/user-agent", "https://httpbin.org/uuid"
            ]
        }
        
    def start_nodes(self, node_count: int, websites: List[str]) -> List[int]:
        """Start multiple monitoring nodes"""
        print(f"🚀 Starting {node_count} nodes with {len(websites)} websites...")
        
        started_ports = []
        
        for i in range(node_count):
            port = self.base_port + i
            node_id = f"node_{chr(97 + i)}" if i < 26 else f"node_{i}"
            
            # Prepare websites string
            websites_str = " ".join([f'"{url}"' for url in websites])
            
            # Start node process
            cmd = [
                sys.executable, "main.py",
                "--port", str(port),
                "--node-id", node_id,
                "--websites"
            ] + websites
            
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd="node_service",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                self.active_processes[port] = process
                started_ports.append(port)
                
                print(f"  ✅ Started {node_id} on port {port}")
                
                # Wait a bit between starts to avoid port conflicts
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  ❌ Failed to start {node_id}: {e}")
                
        return started_ports
    
    def stop_nodes(self):
        """Stop all active nodes"""
        print("🛑 Stopping all nodes...")
        
        for port, process in self.active_processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"  ✅ Stopped node on port {port}")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"  🔴 Force killed node on port {port}")
            except Exception as e:
                print(f"  ❌ Error stopping node on port {port}: {e}")
                
        self.active_processes.clear()
    
    def wait_for_nodes_ready(self, ports: List[int], timeout: int = 60) -> bool:
        """Wait for all nodes to be ready"""
        print(f"⏳ Waiting for nodes to be ready...")
        
        start_time = time.time()
        ready_count = 0
        
        while time.time() - start_time < timeout:
            ready_count = 0
            for port in ports:
                try:
                    response = requests.get(f"http://localhost:{port}/health", timeout=2)
                    if response.status_code == 200:
                        ready_count += 1
                except:
                    pass
            
            print(f"  📊 Ready: {ready_count}/{len(ports)} nodes")
            
            if ready_count == len(ports):
                print("  ✅ All nodes ready!")
                return True
                
            time.sleep(2)
        
        print(f"  ⚠️ Timeout: only {ready_count}/{len(ports)} nodes ready")
        return False
    
    def measure_throughput_latency(self, ports: List[int], duration: int = 60) -> Dict:
        """Measure throughput and latency metrics"""
        print(f"📊 Measuring performance for {duration} seconds...")
        
        metrics = {
            'total_requests': 0,
            'response_times': [],
            'consensus_times': [],
            'ml_times': [],
            'blockchain_times': [],
            'start_time': time.time()
        }
        
        # System resource monitoring
        cpu_usage = []
        memory_usage = []
        network_io = []
        
        def monitor_system():
            """Monitor system resources"""
            while time.time() - metrics['start_time'] < duration:
                cpu_usage.append(psutil.cpu_percent())
                memory = psutil.virtual_memory()
                memory_usage.append(memory.used / 1024 / 1024)  # MB
                net_io = psutil.net_io_counters()
                network_io.append((net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024)  # MB
                time.sleep(1)
        
        # Start system monitoring
        monitor_thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        monitor_future = monitor_thread.submit(monitor_system)
        
        # Measure API performance
        start_time = time.time()
        
        while time.time() - start_time < duration:
            for port in ports:
                try:
                    # Measure health endpoint latency
                    request_start = time.time()
                    response = requests.get(f"http://localhost:{port}/health", timeout=1)
                    request_time = (time.time() - request_start) * 1000
                    
                    if response.status_code == 200:
                        metrics['total_requests'] += 1
                        metrics['response_times'].append(request_time)
                    
                    # Measure consensus endpoint
                    try:
                        consensus_start = time.time()
                        requests.get(f"http://localhost:{port}/consensus/reputations", timeout=1)
                        consensus_time = (time.time() - consensus_start) * 1000
                        metrics['consensus_times'].append(consensus_time)
                    except:
                        pass
                    
                    # Measure ML endpoint
                    try:
                        ml_start = time.time()
                        requests.get(f"http://localhost:{port}/reputation", timeout=1)
                        ml_time = (time.time() - ml_start) * 1000
                        metrics['ml_times'].append(ml_time)
                    except:
                        pass
                    
                except:
                    pass
            
            time.sleep(0.1)  # Small delay between measurement cycles
        
        # Wait for monitoring to finish
        monitor_future.cancel()
        
        # Calculate metrics
        total_time = time.time() - start_time
        throughput = metrics['total_requests'] / total_time if total_time > 0 else 0
        
        result = {
            'throughput_tps': throughput,
            'avg_latency_ms': statistics.mean(metrics['response_times']) if metrics['response_times'] else 0,
            'p95_latency_ms': np.percentile(metrics['response_times'], 95) if metrics['response_times'] else 0,
            'p99_latency_ms': np.percentile(metrics['response_times'], 99) if metrics['response_times'] else 0,
            'avg_consensus_ms': statistics.mean(metrics['consensus_times']) if metrics['consensus_times'] else 0,
            'avg_ml_ms': statistics.mean(metrics['ml_times']) if metrics['ml_times'] else 0,
            'avg_cpu_percent': statistics.mean(cpu_usage) if cpu_usage else 0,
            'avg_memory_mb': statistics.mean(memory_usage) if memory_usage else 0,
            'avg_network_mb': statistics.mean(network_io) if network_io else 0,
            'total_requests': metrics['total_requests'],
            'duration': total_time
        }
        
        return result
    
    def run_scalability_test(self) -> Dict:
        """Run comprehensive scalability test"""
        print("🧪 Starting comprehensive scalability test...")
        
        results = {}
        
        for node_count in self.node_counts:
            for website_count in self.website_counts:
                test_key = f"{node_count}nodes_{website_count}websites"
                print(f"\n📋 Test: {test_key}")
                
                # Get websites for this test
                websites = self.website_pools.get(website_count, self.website_pools[3])
                
                try:
                    # Start nodes
                    ports = self.start_nodes(node_count, websites)
                    
                    if not ports:
                        print(f"  ❌ Failed to start nodes for {test_key}")
                        continue
                    
                    # Wait for nodes to be ready
                    if not self.wait_for_nodes_ready(ports):
                        print(f"  ⚠️ Nodes not ready for {test_key}")
                        self.stop_nodes()
                        continue
                    
                    # Measure performance
                    perf_metrics = self.measure_throughput_latency(ports, duration=30)
                    
                    # Store results
                    results[test_key] = {
                        'node_count': node_count,
                        'website_count': website_count,
                        'metrics': perf_metrics,
                        'timestamp': datetime.now()
                    }
                    
                    print(f"  ✅ Results: {perf_metrics['throughput_tps']:.2f} TPS, {perf_metrics['avg_latency_ms']:.2f}ms avg latency")
                    
                    # Stop nodes
                    self.stop_nodes()
                    
                    # Brief rest between tests
                    time.sleep(5)
                    
                except Exception as e:
                    print(f"  ❌ Error in {test_key}: {e}")
                    self.stop_nodes()
                    continue
        
        return results
    
    def generate_performance_report(self, results: Dict):
        """Generate comprehensive performance report"""
        print("📈 Generating performance report...")
        
        # Create results directory
        os.makedirs("performance_results", exist_ok=True)
        
        # Prepare data for analysis
        data_rows = []
        for test_key, result in results.items():
            metrics = result['metrics']
            data_rows.append({
                'test_name': test_key,
                'node_count': result['node_count'],
                'website_count': result['website_count'],
                'throughput_tps': metrics['throughput_tps'],
                'avg_latency_ms': metrics['avg_latency_ms'],
                'p95_latency_ms': metrics['p95_latency_ms'],
                'p99_latency_ms': metrics['p99_latency_ms'],
                'avg_consensus_ms': metrics['avg_consensus_ms'],
                'avg_ml_ms': metrics['avg_ml_ms'],
                'avg_cpu_percent': metrics['avg_cpu_percent'],
                'avg_memory_mb': metrics['avg_memory_mb'],
                'avg_network_mb': metrics['avg_network_mb'],
                'total_requests': metrics['total_requests']
            })
        
        df = pd.DataFrame(data_rows)
        
        # Save raw data
        df.to_csv("performance_results/scalability_results.csv", index=False)
        
        # Generate visualizations
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Throughput vs Node Count
        for website_count in self.website_counts:
            subset = df[df['website_count'] == website_count]
            axes[0, 0].plot(subset['node_count'], subset['throughput_tps'], 
                           marker='o', label=f'{website_count} websites')
        axes[0, 0].set_xlabel('Node Count')
        axes[0, 0].set_ylabel('Throughput (TPS)')
        axes[0, 0].set_title('Throughput vs Node Count')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Latency vs Node Count
        for website_count in self.website_counts:
            subset = df[df['website_count'] == website_count]
            axes[0, 1].plot(subset['node_count'], subset['avg_latency_ms'], 
                           marker='s', label=f'{website_count} websites')
        axes[0, 1].set_xlabel('Node Count')
        axes[0, 1].set_ylabel('Average Latency (ms)')
        axes[0, 1].set_title('Latency vs Node Count')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Throughput vs Website Count
        for node_count in [4, 16, 50]:
            subset = df[df['node_count'] == node_count]
            axes[0, 2].plot(subset['website_count'], subset['throughput_tps'], 
                           marker='^', label=f'{node_count} nodes')
        axes[0, 2].set_xlabel('Website Count')
        axes[0, 2].set_ylabel('Throughput (TPS)')
        axes[0, 2].set_title('Throughput vs Website Count')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Resource Usage
        axes[1, 0].plot(df['node_count'], df['avg_cpu_percent'], 'r-o', label='CPU %')
        axes[1, 0].set_xlabel('Node Count')
        axes[1, 0].set_ylabel('CPU Usage (%)')
        axes[1, 0].set_title('CPU Usage vs Node Count')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(df['node_count'], df['avg_memory_mb'], 'b-s', label='Memory MB')
        axes[1, 1].set_xlabel('Node Count')
        axes[1, 1].set_ylabel('Memory Usage (MB)')
        axes[1, 1].set_title('Memory Usage vs Node Count')
        axes[1, 1].grid(True)
        
        # Efficiency (TPS per node)
        df['tps_per_node'] = df['throughput_tps'] / df['node_count']
        for website_count in self.website_counts:
            subset = df[df['website_count'] == website_count]
            axes[1, 2].plot(subset['node_count'], subset['tps_per_node'], 
                           marker='d', label=f'{website_count} websites')
        axes[1, 2].set_xlabel('Node Count')
        axes[1, 2].set_ylabel('TPS per Node')
        axes[1, 2].set_title('Efficiency (TPS per Node)')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig("performance_results/scalability_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate summary report
        summary = {
            'test_completed': datetime.now().isoformat(),
            'total_tests': len(results),
            'max_throughput': df['throughput_tps'].max(),
            'min_latency': df['avg_latency_ms'].min(),
            'max_nodes_tested': df['node_count'].max(),
            'max_websites_tested': df['website_count'].max(),
            'best_performance': df.loc[df['throughput_tps'].idxmax()].to_dict(),
            'worst_performance': df.loc[df['throughput_tps'].idxmin()].to_dict()
        }
        
        with open("performance_results/summary_report.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📊 Performance report generated!")
        print(f"  📈 Max Throughput: {summary['max_throughput']:.2f} TPS")
        print(f"  ⚡ Min Latency: {summary['min_latency']:.2f} ms")
        print(f"  🔢 Max Nodes: {summary['max_nodes_tested']}")
        print(f"  🌐 Max Websites: {summary['max_websites_tested']}")
        print(f"  📁 Results saved to: performance_results/")
        
        return summary

def main():
    """Main performance testing function"""
    print("🚀 Performance Testing Framework")
    print("=" * 50)
    
    tester = PerformanceTester(base_port=8000, max_nodes=100)
    
    try:
        # Run comprehensive scalability test
        results = tester.run_scalability_test()
        
        # Generate performance report
        summary = tester.generate_performance_report(results)
        
        print("\n✅ Performance testing completed!")
        print(f"📊 Check performance_results/ for detailed reports and charts.")
        
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
        tester.stop_nodes()
    except Exception as e:
        print(f"\n❌ Testing error: {e}")
        tester.stop_nodes()
    finally:
        tester.stop_nodes()

if __name__ == "__main__":
    main()
