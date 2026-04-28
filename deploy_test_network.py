#!/usr/bin/env python3
"""
Quick Test Network Deployment
Deploy multiple nodes with variable websites for performance testing
"""

import subprocess
import time
import requests
import sys
import argparse
from typing import List

def deploy_network(node_count: int = 10, website_count: int = 20):
    """Deploy test network with specified parameters"""
    
    # Website pools
    websites = [
        "https://google.com", "https://github.com", "https://stackoverflow.com",
        "https://httpbin.org", "https://api.github.com", "https://jsonplaceholder.typicode.com",
        "https://reqres.in", "https://publicapis.dev", "https://catfact.ninja/fact",
        "https://api.coindesk.com/v1/bpi/currentprice.json", "https://api.exchangerate-api.com/v4/latest/USD",
        "https://jsonplaceholder.typicode.com/posts", "https://jsonplaceholder.typicode.com/users",
        "https://reqres.in/api/users", "https://reqres.in/api/products",
        "https://httpbin.org/get", "https://httpbin.org/post", "https://httpbin.org/put",
        "https://api.chucknorris.io/jokes/random", "https://api.quotable.io/random"
    ]
    
    # Limit websites to requested count
    websites = websites[:website_count]
    
    print(f"🚀 Deploying {node_count} nodes with {len(websites)} websites...")
    
    processes = []
    ports = []
    
    # Start nodes
    for i in range(node_count):
        port = 8000 + i
        node_id = f"test_node_{i}"
        
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
            
            processes.append(process)
            ports.append(port)
            
            print(f"  ✅ Started {node_id} on port {port}")
            time.sleep(0.5)  # Avoid port conflicts
            
        except Exception as e:
            print(f"  ❌ Failed to start {node_id}: {e}")
    
    # Wait for nodes to be ready
    print(f"\n⏳ Waiting for nodes to be ready...")
    time.sleep(10)
    
    # Check node status
    ready_count = 0
    for port in ports:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                ready_count += 1
                print(f"  ✅ Node on port {port} is ready")
            else:
                print(f"  ⚠️ Node on port {port} returned {response.status_code}")
        except Exception as e:
            print(f"  ❌ Node on port {port} not ready: {e}")
    
    print(f"\n📊 Network Status: {ready_count}/{len(ports)} nodes ready")
    
    # Print network info
    print(f"\n🌐 Network Information:")
    print(f"  Nodes: {ready_count} active")
    print(f"  Websites: {len(websites)}")
    print(f"  Ports: {ports}")
    print(f"  Dashboard: http://localhost:8501")
    print(f"\n📝 Sample URLs to test:")
    for i, port in enumerate(ports[:3]):
        print(f"  Node {i}: http://localhost:{port}/reputation")
    
    print(f"\n⚠️  Press Ctrl+C to stop all nodes")
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n🛑 Stopping all nodes...")
        
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        
        print("✅ All nodes stopped")

def main():
    parser = argparse.ArgumentParser(description='Deploy test network')
    parser.add_argument('--nodes', type=int, default=10, help='Number of nodes to deploy')
    parser.add_argument('--websites', type=int, default=20, help='Number of websites to monitor')
    
    args = parser.parse_args()
    
    deploy_network(args.nodes, args.websites)

if __name__ == "__main__":
    main()
