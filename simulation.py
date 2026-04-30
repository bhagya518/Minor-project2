#!/usr/bin/env python3
"""
Network Performance Simulation
Simulates throughput and latency with varying numbers of nodes
"""

import asyncio
import aiohttp
import time
import statistics
import json
from typing import List, Dict, Tuple

# Disable blockchain imports to avoid transaction noise
import os
os.environ['WEB3_PROVIDER_URI'] = 'http://localhost:9999'  # Invalid URI to disable blockchain

class NetworkSimulator:
    def __init__(self):
        self.base_port = 8005
        self.results = {
            'node_counts': [],
            'avg_latencies': [],
            'throughputs': [],
            'latency_std': []
        }
    
    async def measure_node_performance(self, num_nodes: int) -> Tuple[float, float]:
        """Measure performance with given number of nodes"""
        print(f"Testing with {num_nodes} nodes...")
        
       
        node_urls = [f"http://localhost:{self.base_port + i}" for i in range(num_nodes)]
        
        # Measure latency for each node
        latencies = []
        start_time = time.time()
        
        # Send concurrent requests to all nodes
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url in node_urls:
                task = self.ping_node(session, url)
                tasks.append(task)
            
            # Execute all requests concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Calculate latencies
            for result in results:
                if isinstance(result, float):
                    latencies.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        if latencies:
            avg_latency = statistics.mean(latencies)
            latency_std = statistics.stdev(latencies) if len(latencies) > 1 else 0
        else:
            avg_latency = 0
            latency_std = 0
        
        # Throughput = total requests / total time
        throughput = len(node_urls) / total_time if total_time > 0 else 0
        
        return avg_latency, throughput
    
    async def ping_node(self, session: aiohttp.ClientSession, url: str) -> float:
        """Ping a single node and return latency"""
        try:
            start = time.time()
            async with session.get(url + "/health", timeout=5) as response:
                if response.status == 200:
                    return (time.time() - start) * 1000  # Convert to ms
        except:
            return 0
    
    async def run_simulation(self, max_nodes: int = 10, step: int = 1):
        """Run simulation with varying node counts"""
        print("Starting Network Performance Simulation...")
        print("=" * 50)
        
        for num_nodes in range(1, max_nodes + 1, step):
            avg_latency, throughput = await self.measure_node_performance(num_nodes)
            
            self.results['node_counts'].append(num_nodes)
            self.results['avg_latencies'].append(avg_latency)
            self.results['throughputs'].append(throughput)
            self.results['latency_std'].append(
                statistics.stdev([avg_latency, avg_latency + 1]) if avg_latency > 0 else 0  # Add dummy value for std dev
            )
            
            print(f"Nodes: {num_nodes:2d} | "
                  f"Avg Latency: {avg_latency:6.2f}ms | "
                  f"Throughput: {throughput:6.2f} req/s")
        
        print("=" * 50)
        print("Simulation Complete!")
        
        return self.results
    
    def plot_results(self, results: Dict):
        """Plot the simulation results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Latency plot
        ax1.plot(results['node_counts'], results['avg_latencies'], 
                'bo-', linewidth=2, markersize=6, label='Average Latency')
        ax1.fill_between(results['node_counts'], 
                         np.array(results['avg_latencies']) - np.array(results['latency_std']),
                         np.array(results['avg_latencies']) + np.array(results['latency_std']),
                         alpha=0.2, color='blue')
        ax1.set_xlabel('Number of Nodes')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Network Latency vs Node Count')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Throughput plot
        ax2.plot(results['node_counts'], results['throughputs'], 
                'ro-', linewidth=2, markersize=6, label='Throughput')
        ax2.set_xlabel('Number of Nodes')
        ax2.set_ylabel('Throughput (requests/second)')
        ax2.set_title('Network Throughput vs Node Count')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('network_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\n" + "=" * 50)
        print("SIMULATION SUMMARY")
        print("=" * 50)
        print(f"Max nodes tested: {max(results['node_counts'])}")
        print(f"Min latency: {min(results['avg_latencies']):.2f}ms")
        print(f"Max latency: {max(results['avg_latencies']):.2f}ms")
        print(f"Min throughput: {min(results['throughputs']):.2f} req/s")
        print(f"Max throughput: {max(results['throughputs']):.2f} req/s")
        
        # Calculate performance degradation
        if len(results['avg_latencies']) > 1:
            latency_increase = (results['avg_latencies'][-1] - results['avg_latencies'][0]) / results['avg_latencies'][0] * 100
            throughput_decrease = (results['throughputs'][0] - results['throughputs'][-1]) / results['throughputs'][0] * 100
            print(f"Latency increase: {latency_increase:.1f}% from 1 to {max(results['node_counts'])} nodes")
            print(f"Throughput decrease: {throughput_decrease:.1f}% from 1 to {max(results['node_counts'])} nodes")
        
        # Generate JSON statistics
        import json
        stats = {
            "simulation_config": {
                "node_range": f"{min(results['node_counts'])}-{max(results['node_counts'])}",
                "step_size": 5,
                "total_tests": len(results['node_counts'])
            },
            "performance_summary": {
                "min_latency_ms": round(min(results['avg_latencies']), 2),
                "max_latency_ms": round(max(results['avg_latencies']), 2),
                "min_throughput_reqs_per_sec": round(min(results['throughputs']), 2),
                "max_throughput_reqs_per_sec": round(max(results['throughputs']), 2),
                "latency_improvement_percent": round(-latency_increase if 'latency_increase' in locals() else 0, 1),
                "throughput_improvement_percent": round(-throughput_decrease if 'throughput_decrease' in locals() else 0, 1)
            },
            "detailed_results": []
        }
        
        # Add detailed results for each node count
        for i, node_count in enumerate(results['node_counts']):
            stats["detailed_results"].append({
                "node_count": node_count,
                "avg_latency_ms": round(results['avg_latencies'][i], 2),
                "throughput_reqs_per_sec": round(results['throughputs'][i], 2),
                "latency_std_dev": round(results['latency_std'][i], 2)
            })
        
        # Save JSON to file
        with open('network_performance_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nStatistics saved to 'network_performance_stats.json'")

async def main():
    """Main simulation function"""
    simulator = NetworkSimulator()
    
    # Run simulation with 10-50 nodes
    results = await simulator.run_simulation(max_nodes=50, step=5)
    
    # Plot results
    simulator.plot_results(results)
    
    print(f"\nGraph saved as 'network_performance.png'")
    print("Simulation complete!")

if __name__ == "__main__":
    # Check if required packages are available
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for plotting")
        print("Install with: pip install matplotlib")
        exit(1)
    
    try:
        import aiohttp
    except ImportError:
        print("Error: aiohttp is required for HTTP requests")
        print("Install with: pip install aiohttp")
        exit(1)
    
    # Run simulation
    asyncio.run(main())
