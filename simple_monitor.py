"""
Simple Ethereum Monitoring Node
No Redis, No Docker - Just Geth + Python
"""

import asyncio
import aiohttp
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
from web3 import Web3
from fastapi import FastAPI, HTTPException
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('monitoring.log')
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Simple Ethereum Monitoring",
    description="Minimal monitoring system without Redis",
    version="1.0.0"
)

# Configuration
class Config:
    GETH_RPC_URL = "http://localhost:8545"
    MONITORING_INTERVAL = 60  # seconds
    RESULTS_FILE = "monitoring_results.json"
    TARGETS = [
        "https://ethereum.org",
        "https://etherscan.io", 
        "https://uniswap.org",
        "https://metamask.io",
        "https://curve.fi",
        "https://aave.com"
    ]

# Initialize Web3 connection
try:
    w3 = Web3(Web3.HTTPProvider(Config.GETH_RPC_URL))
    if w3.is_connected():
        logger.info(f"✅ Connected to Geth - Chain ID: {w3.eth.chain_id}")
        logger.info(f"Current block: {w3.eth.block_number}")
    else:
        logger.error("❌ Failed to connect to Geth")
        w3 = None
except Exception as e:
    logger.error(f"❌ Geth connection error: {e}")
    w3 = None

# Simple in-memory cache (no Redis needed)
class SimpleCache:
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    def get(self, key: str):
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_timeout:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value):
        self.cache[key] = (value, time.time())

cache = SimpleCache()

async def check_website(url: str) -> Dict:
    """Check if website is accessible"""
    try:
        start_time = time.time()
        
        # Configure timeout and headers
        timeout = aiohttp.ClientTimeout(total=10)
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; EthereumMonitor/1.0)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(url, allow_redirects=True) as response:
                response_time = (time.time() - start_time) * 1000
                
                result = {
                    'url': url,
                    'status_code': response.status,
                    'response_time': round(response_time, 2),
                    'accessible': 200 <= response.status < 400,
                    'timestamp': int(time.time()),
                    'datetime': datetime.now().isoformat()
                }
                
                # Add additional info
                result['content_length'] = response.headers.get('content-length', 0)
                result['server'] = response.headers.get('server', 'unknown')
                
                return result
                
    except asyncio.TimeoutError:
        return {
            'url': url,
            'status_code': 0,
            'response_time': 0,
            'accessible': False,
            'error': 'Timeout',
            'timestamp': int(time.time()),
            'datetime': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'url': url,
            'status_code': 0,
            'response_time': 0,
            'accessible': False,
            'error': str(e),
            'timestamp': int(time.time()),
            'datetime': datetime.now().isoformat()
        }

async def monitor_all_targets() -> List[Dict]:
    """Monitor all target websites"""
    logger.info(f"Starting monitoring round for {len(Config.TARGETS)} targets")
    
    # Check all websites concurrently
    tasks = [check_website(url) for url in Config.TARGETS]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions
    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Error monitoring {Config.TARGETS[i]}: {result}")
            valid_results.append({
                'url': Config.TARGETS[i],
                'status_code': 0,
                'response_time': 0,
                'accessible': False,
                'error': str(result),
                'timestamp': int(time.time()),
                'datetime': datetime.now().isoformat()
            })
        else:
            valid_results.append(result)
    
    # Save results to file (simple JSON lines format)
    try:
        with open(Config.RESULTS_FILE, 'a') as f:
            for result in valid_results:
                f.write(json.dumps(result) + '\n')
        logger.info(f"Results saved to {Config.RESULTS_FILE}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    # Log summary
    accessible_count = sum(1 for r in valid_results if r.get('accessible'))
    total_count = len(valid_results)
    logger.info(f"Monitoring complete: {accessible_count}/{total_count} sites accessible")
    
    return valid_results

def get_recent_results(limit: int = 100) -> List[Dict]:
    """Get recent monitoring results from file"""
    try:
        results = []
        with open(Config.RESULTS_FILE, 'r') as f:
            lines = f.readlines()
            # Get last 'limit' lines
            for line in lines[-limit:]:
                if line.strip():
                    results.append(json.loads(line.strip()))
        return results
    except FileNotFoundError:
        return []
    except Exception as e:
        logger.error(f"Error reading results: {e}")
        return []

def calculate_uptime_stats(hours: int = 24) -> Dict:
    """Calculate uptime statistics for the last N hours"""
    try:
        cutoff_time = int(time.time()) - (hours * 3600)
        recent_results = get_recent_results(1000)  # Get more results for accuracy
        
        # Filter by time
        recent_results = [r for r in recent_results if r.get('timestamp', 0) > cutoff_time]
        
        if not recent_results:
            return {}
        
        # Calculate stats per URL
        url_stats = {}
        for result in recent_results:
            url = result['url']
            if url not in url_stats:
                url_stats[url] = {'total': 0, 'accessible': 0}
            
            url_stats[url]['total'] += 1
            if result.get('accessible'):
                url_stats[url]['accessible'] += 1
        
        # Calculate percentages
        stats = {}
        for url, data in url_stats.items():
            if data['total'] > 0:
                stats[url] = {
                    'uptime_percentage': round((data['accessible'] / data['total']) * 100, 2),
                    'total_checks': data['total'],
                    'accessible_checks': data['accessible']
                }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating uptime stats: {e}")
        return {}

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Simple Ethereum Monitoring",
        "status": "running",
        "targets_count": len(Config.TARGETS),
        "geth_connected": w3 is not None and w3.is_connected(),
        "timestamp": int(time.time())
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    geth_status = False
    block_number = None
    
    if w3 and w3.is_connected():
        try:
            block_number = w3.eth.block_number
            geth_status = True
        except Exception as e:
            logger.error(f"Health check Geth error: {e}")
    
    return {
        "status": "healthy" if geth_status else "unhealthy",
        "geth_connected": geth_status,
        "block_number": block_number,
        "timestamp": int(time.time()),
        "datetime": datetime.now().isoformat()
    }

@app.get("/monitor")
async def monitor():
    """Trigger monitoring round"""
    try:
        results = await monitor_all_targets()
        
        # Calculate summary
        accessible_count = sum(1 for r in results if r.get('accessible'))
        total_count = len(results)
        
        return {
            "timestamp": int(time.time()),
            "datetime": datetime.now().isoformat(),
            "summary": {
                "total": total_count,
                "accessible": accessible_count,
                "inaccessible": total_count - accessible_count,
                "success_rate": round((accessible_count / total_count) * 100, 2) if total_count > 0 else 0
            },
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Monitor endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status():
    """Get current system status"""
    try:
        # Get recent results for status
        recent_results = get_recent_results(10)
        
        # Calculate recent uptime
        if recent_results:
            accessible_count = sum(1 for r in recent_results if r.get('accessible'))
            recent_uptime = round((accessible_count / len(recent_results)) * 100, 2)
        else:
            recent_uptime = 0
        
        # Geth info
        geth_info = {}
        if w3 and w3.is_connected():
            try:
                geth_info = {
                    "connected": True,
                    "block_number": w3.eth.block_number,
                    "chain_id": w3.eth.chain_id,
                    "gas_price": w3.eth.gas_price,
                    "syncing": w3.eth.syncing
                }
            except Exception as e:
                geth_info = {"connected": True, "error": str(e)}
        else:
            geth_info = {"connected": False}
        
        return {
            "timestamp": int(time.time()),
            "datetime": datetime.now().isoformat(),
            "monitoring": {
                "targets": len(Config.TARGETS),
                "recent_uptime": recent_uptime,
                "last_check": recent_results[-1]['datetime'] if recent_results else None
            },
            "geth": geth_info,
            "system": {
                "cache_size": len(cache.cache),
                "results_file": Config.RESULTS_FILE
            }
        }
        
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/uptime")
async def uptime(hours: int = 24):
    """Get uptime statistics"""
    try:
        stats = calculate_uptime_stats(hours)
        
        return {
            "period_hours": hours,
            "timestamp": int(time.time()),
            "datetime": datetime.now().isoformat(),
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Uptime endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results")
async def results(limit: int = 50):
    """Get recent monitoring results"""
    try:
        recent_results = get_recent_results(limit)
        
        return {
            "count": len(recent_results),
            "limit": limit,
            "timestamp": int(time.time()),
            "results": recent_results
        }
        
    except Exception as e:
        logger.error(f"Results endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/blockchain")
async def blockchain_info():
    """Get blockchain information"""
    if not w3 or not w3.is_connected():
        raise HTTPException(status_code=503, detail="Geth not connected")
    
    try:
        latest_block = w3.eth.get_block('latest')
        
        return {
            "connected": True,
            "chain_id": w3.eth.chain_id,
            "block_number": latest_block.number,
            "block_timestamp": latest_block.timestamp,
            "block_hash": latest_block.hash.hex(),
            "gas_limit": latest_block.gasLimit,
            "gas_used": latest_block.gasUsed,
            "transaction_count": len(latest_block.transactions),
            "gas_price": w3.eth.gas_price,
            "syncing": w3.eth.syncing,
            "accounts": w3.eth.accounts,
            "timestamp": int(time.time())
        }
        
    except Exception as e:
        logger.error(f"Blockchain info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background monitoring task
async def background_monitoring():
    """Run monitoring in background"""
    while True:
        try:
            await monitor_all_targets()
            await asyncio.sleep(Config.MONITORING_INTERVAL)
        except Exception as e:
            logger.error(f"Background monitoring error: {e}")
            await asyncio.sleep(10)  # Wait before retry

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize background monitoring"""
    logger.info("Starting Simple Ethereum Monitoring Node")
    logger.info(f"Monitoring {len(Config.TARGETS)} targets every {Config.MONITORING_INTERVAL} seconds")
    
    # Start background monitoring
    asyncio.create_task(background_monitoring())
    
    logger.info("Background monitoring started")
    logger.info("API available at: http://localhost:8000")

if __name__ == "__main__":
    print("🚀 Starting Simple Ethereum Monitoring Node")
    print("📊 API will be available at: http://localhost:8000")
    print("🔍 Endpoints:")
    print("  GET /              - Service info")
    print("  GET /health        - Health check")
    print("  GET /monitor       - Trigger monitoring")
    print("  GET /status        - System status")
    print("  GET /uptime        - Uptime statistics")
    print("  GET /results       - Recent results")
    print("  GET /blockchain    - Blockchain info")
    print()
    
    # Run the application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
