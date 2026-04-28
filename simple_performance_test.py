#!/usr/bin/env python3
"""
Simple Performance Testing Script
Test throughput and latency with current running nodes
"""

import requests
import time
import statistics
import json
from datetime import datetime
import concurrent.futures
import sys

def test_node_performance(port: int, duration: int = 30) -> dict:
    """Test individual node performance"""
    print(f"🧪 Testing node on port {port}...")
    
    metrics = {
        'port': port,
        'total_requests': 0,
        'successful_requests': 0,
        'response_times': [],
        'start_time': time.time()
    }
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        try:
            # Test health endpoint
            request_start = time.time()
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            request_time = (time.time() - request_start) * 1000
            
            metrics['total_requests'] += 1
            
            if response.status_code == 200:
                metrics['successful_requests'] += 1
                metrics['response_times'].append(request_time)
                
        except Exception as e:
            metrics['total_requests'] += 1
            # print(f"  ❌ Error: {e}")
        
        time.sleep(0.1)  # Small delay between requests
    
    # Calculate statistics
    if metrics['response_times']:
        metrics['avg_response_time'] = statistics.mean(metrics['response_times'])
        metrics['min_response_time'] = min(metrics['response_times'])
        metrics['max_response_time'] = max(metrics['response_times'])
        metrics['p95_response_time'] = sorted(metrics['response_times'])[int(len(metrics['response_times']) * 0.95)]
    else:
        metrics['avg_response_time'] = 0
        metrics['min_response_time'] = 0
        metrics['max_response_time'] = 0
        metrics['p95_response_time'] = 0
    
    metrics['success_rate'] = metrics['successful_requests'] / metrics['total_requests'] if metrics['total_requests'] > 0 else 0
    metrics['duration'] = time.time() - metrics['start_time']
    metrics['throughput'] = metrics['successful_requests'] / metrics['duration'] if metrics['duration'] > 0 else 0
    
    return metrics

def test_ml_performance(port: int) -> dict:
    """Test ML endpoint performance"""
    print(f"🤖 Testing ML performance on port {port}...")
    
    metrics = {
        'port': port,
        'ml_requests': 0,
        'ml_response_times': [],
        'ml_success': 0
    }
    
    try:
        for i in range(10):  # Test 10 ML requests
            try:
                start_time = time.time()
                response = requests.get(f"http://localhost:{port}/reputation", timeout=5)
                response_time = (time.time() - start_time) * 1000
                
                metrics['ml_requests'] += 1
                
                if response.status_code == 200:
                    metrics['ml_success'] += 1
                    metrics['ml_response_times'].append(response_time)
                    
                    # Parse response to check ML engine
                    data = response.json()
                    if 'engine_type' in data:
                        metrics['engine_type'] = data['engine_type']
                    if 'shard_distribution' in data:
                        metrics['shard_distribution'] = data['shard_distribution']
                    
            except Exception as e:
                # print(f"  ❌ ML request error: {e}")
                pass
            
            time.sleep(0.5)
        
        if metrics['ml_response_times']:
            metrics['avg_ml_time'] = statistics.mean(metrics['ml_response_times'])
        else:
            metrics['avg_ml_time'] = 0
            
        metrics['ml_success_rate'] = metrics['ml_success'] / metrics['ml_requests'] if metrics['ml_requests'] > 0 else 0
        
    except Exception as e:
        print(f"  ❌ ML test failed: {e}")
        metrics['avg_ml_time'] = 0
        metrics['ml_success_rate'] = 0
    
    return metrics

def scan_network(max_ports: int = 20) -> list:
    """Scan for active nodes"""
    print(f"🔍 Scanning for active nodes on ports 8000-{8000 + max_ports - 1}...")
    
    active_ports = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        
        for port in range(8000, 8000 + max_ports):
            future = executor.submit(check_node, port)
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            port, is_active = future.result()
            if is_active:
                active_ports.append(port)
                print(f"  ✅ Node found on port {port}")
    
    return active_ports

def check_node(port: int) -> tuple:
    """Check if node is active"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        return port, response.status_code == 200
    except:
        return port, False

def main():
    print("🚀 Simple Performance Testing")
    print("=" * 40)
    
    # Scan for active nodes
    active_ports = scan_network(20)
    
    if not active_ports:
        print("❌ No active nodes found!")
        print("💡 Make sure nodes are running before testing")
        return
    
    print(f"\n📊 Found {len(active_ports)} active nodes: {active_ports}")
    
    # Test basic performance
    print(f"\n🧪 Running performance tests (30 seconds each)...")
    
    performance_results = []
    ml_results = []
    
    for port in active_ports:
        # Basic performance test
        perf_metrics = test_node_performance(port, duration=30)
        performance_results.append(perf_metrics)
        
        # ML performance test
        ml_metrics = test_ml_performance(port)
        ml_results.append(ml_metrics)
        
        print(f"  ✅ Port {port}: {perf_metrics['throughput']:.2f} TPS, {perf_metrics['avg_response_time']:.2f}ms avg")
    
    # Generate report
    print(f"\n📈 Performance Report")
    print("=" * 40)
    
    total_throughput = sum(r['throughput'] for r in performance_results)
    avg_latency = statistics.mean([r['avg_response_time'] for r in performance_results if r['avg_response_time'] > 0])
    avg_success_rate = statistics.mean([r['success_rate'] for r in performance_results])
    
    print(f"🌐 Network Summary:")
    print(f"  Active Nodes: {len(active_ports)}")
    print(f"  Total Throughput: {total_throughput:.2f} TPS")
    print(f"  Average Latency: {avg_latency:.2f} ms")
    print(f"  Average Success Rate: {avg_success_rate:.2%}")
    
    print(f"\n🤖 ML Engine Summary:")
    engine_types = {}
    shard_counts = {}
    
    for ml_result in ml_results:
        if 'engine_type' in ml_result:
            engine_type = ml_result['engine_type']
            engine_types[engine_type] = engine_types.get(engine_type, 0) + 1
        
        if 'shard_distribution' in ml_result:
            for shard, count in ml_result['shard_distribution'].items():
                shard_counts[shard] = shard_counts.get(shard, 0) + count
    
    print(f"  Engine Types: {engine_types}")
    print(f"  Shard Distribution: {shard_counts}")
    
    print(f"\n📊 Individual Node Results:")
    print(f"{'Port':<6} {'TPS':<8} {'Latency':<10} {'Success':<8} {'Engine':<12}")
    print("-" * 50)
    
    for i, port in enumerate(active_ports):
        perf = performance_results[i]
        ml = ml_results[i]
        engine = ml.get('engine_type', 'unknown')
        
        print(f"{port:<6} {perf['throughput']:<8.2f} {perf['avg_response_time']:<10.2f} {perf['success_rate']:<8.1%} {engine:<12}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'active_ports': active_ports,
        'network_summary': {
            'total_throughput': total_throughput,
            'avg_latency': avg_latency,
            'avg_success_rate': avg_success_rate,
            'node_count': len(active_ports)
        },
        'ml_summary': {
            'engine_types': engine_types,
            'shard_distribution': shard_counts
        },
        'individual_results': [
            {
                'port': performance_results[i]['port'],
                'throughput': performance_results[i]['throughput'],
                'avg_latency': performance_results[i]['avg_response_time'],
                'success_rate': performance_results[i]['success_rate'],
                'engine_type': ml_results[i].get('engine_type', 'unknown')
            }
            for i in range(len(active_ports))
        ]
    }
    
    with open(f"performance_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: performance_test_results_*.json")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if total_throughput > 100:
        print(f"  ✅ High throughput achieved: {total_throughput:.2f} TPS")
    elif total_throughput > 50:
        print(f"  ⚠️ Moderate throughput: {total_throughput:.2f} TPS")
    else:
        print(f"  ❌ Low throughput: {total_throughput:.2f} TPS - consider optimization")
    
    if avg_latency < 100:
        print(f"  ✅ Good latency: {avg_latency:.2f} ms")
    elif avg_latency < 500:
        print(f"  ⚠️ Moderate latency: {avg_latency:.2f} ms")
    else:
        print(f"  ❌ High latency: {avg_latency:.2f} ms - consider optimization")
    
    if 'enhanced' in engine_types:
        print(f"  ✅ Enhanced ML engine active with 4-tier categorization")
    else:
        print(f"  ⚠️ Consider upgrading to enhanced ML engine for better accuracy")
    
    print(f"\n🚀 To test scalability, try:")
    print(f"  python deploy_test_network.py --nodes 20 --websites 50")
    print(f"  python performance_tester.py")

if __name__ == "__main__":
    main()
