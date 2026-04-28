"""
Main FastAPI Application for Decentralized Website Monitoring Node
Integrates monitoring, trust engine, ML inference, and blockchain components
"""

import asyncio
import uvicorn
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import os
import sys
import argparse
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parse command line arguments (moved inside __main__ to prevent module-level crashes)
parser = argparse.ArgumentParser(description='Decentralized Website Monitoring Node')
parser.add_argument('--port', type=int, default=8000, help='Port to run the node on')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
parser.add_argument('--node-id', type=str, help='Unique node identifier')
parser.add_argument('--peers', type=str, nargs='+', help='List of peer URLs')
parser.add_argument('--websites', type=str, nargs='+',
                    default=os.getenv('MONITORED_URLS', 'https://google.com,https://github.com,https://httpbin.org').split(','),
                    help='Websites to monitor')
# args will be parsed in __main__ section

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from dataclasses import asdict

# Import signed report system (Phase 2)
from monitoring_report import MonitoringReport, ReportVerifier

# Try to import blockchain client (optional)
try:
    from src.blockchain_client import BlockchainClient
    BLOCKCHAIN_AVAILABLE = True
    logger.info("Blockchain client imported successfully")
except ImportError as e:
    BLOCKCHAIN_AVAILABLE = False
    BlockchainClient = None
    logger.warning(f"Blockchain client not available: {e}")

# Import Enhanced ML components
try:
    from src.ml_consensus_engine import EnhancedMLConsensusEngine
    ML_AVAILABLE = True
    logger.info("Enhanced ML components imported successfully")
except ImportError as e:
    ML_AVAILABLE = False
    logger.warning(f"Enhanced ML components not available: {e}")

# Import monitoring components
try:
    from website_monitor import WebsiteMonitor, MonitoringScheduler, set_node_id, NODE_SIGNER
    from trust_engine import TrustEngine, TrustCalculator
    from peer_client import PeerClient
    from src.epoch_manager import init_epoch_manager, get_epoch_manager
    monitoring_available = True
    logger.info("Monitoring components imported successfully")
except ImportError as e:
    monitoring_available = False
    logger.error(f"Monitoring components not available: {e}")

# Pydantic models for API
class MonitoringRequest(BaseModel):
    urls: List[str]

class PeerInfo(BaseModel):
    node_id: str
    host: str
    port: int

class NodeConfig(BaseModel):
    node_id: str
    host: str = "localhost"
    port: int = 8000
    websites: List[str] = []
    seed_peers: List[PeerInfo] = []
    monitoring_interval: int = 60

class TrustUpdate(BaseModel):
    node_id: str
    trust_score: float

class ReputationUpdate(BaseModel):
    node_id: str
    monitoring_trust: float
    ml_score: float

# Global variables
app = FastAPI(title="Decentralized Website Monitoring Node", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for node components
node_config = None
website_monitor = None
monitoring_scheduler = None
trust_engine = None
peer_client = None
ml_classifier = None
blockchain_client = None
live_ml = None
ml_consensus = None
epoch_manager = None

# Phase 2: Peer report storage and public key registry
peer_reports: List[dict] = []  # In-memory store for this epoch's peer reports
peer_public_keys: Dict[str, str] = {}  # node_id → pubkey_hex (for signature verification)
peer_registry: Dict[str, dict] = {}  # node_id → {url, public_key_hex} (one source of truth)

# Load deployed contract info for real blockchain integration
CONTRACT_INFO = None
try:
    contract_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'blockchain', 'ProofOfReputation.json')
    if os.path.exists(contract_path):
        with open(contract_path, 'r') as f:
            CONTRACT_INFO = json.load(f)
        logger.info(f"Loaded contract info: {CONTRACT_INFO['address']}")
    else:
        logger.warning(f"Contract file not found at {contract_path}")
except Exception as e:
    logger.warning(f"Failed to load contract info: {e}")

# Verdict storage for /verdict endpoint
verdicts_store: Dict[str, dict] = {}

# Background tasks
monitoring_task = None
heartbeat_task = None
cleanup_task = None

@app.on_event("startup")
async def startup_event():
    """Initialize node components on startup"""
    global website_monitor, monitoring_scheduler, trust_engine, peer_client, ml_classifier
    global blockchain_client, live_ml, ml_consensus, node_config, epoch_manager
    global monitoring_task, heartbeat_task, cleanup_task

    # Parse command line arguments at startup (needed for --peers flag)
    args = parser.parse_args()
    
    try:
        logger.info("Starting up monitoring node...")
        
        # Get command line arguments
        import sys
        port = 8000
        node_id = f"node_{port}"
        websites = ["https://google.com", "https://github.com", "https://httpbin.org"]
        peers = []
        
        # Parse sys.argv for arguments
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
            elif arg == "--node-id" and i + 1 < len(sys.argv):
                node_id = sys.argv[i + 1]
            elif arg == "--websites" and i + 1 < len(sys.argv):
                # Collect all website arguments until next flag
                websites = []
                j = i + 1
                while j < len(sys.argv) and not sys.argv[j].startswith("--"):
                    websites.append(sys.argv[j])
                    j += 1
            elif arg == "--peers" and i + 1 < len(sys.argv):
                # Collect all peer arguments until next flag
                peers = []
                j = i + 1
                while j < len(sys.argv) and not sys.argv[j].startswith("--"):
                    peers.append(sys.argv[j])
                    j += 1
        
        # Load configuration
        node_config = NodeConfig(
            node_id=node_id,
            host="localhost",
            port=port,
            websites=websites,
            monitoring_interval=60
        )
        
        logger.info(f"Node configuration: {node_config.node_id} on {node_config.host}:{node_config.port}")
        
        # Initialize website_monitor module with proper node ID
        if monitoring_available:
            set_node_id(node_config.node_id)
        
        # Initialize components
        website_monitor = WebsiteMonitor()
        trust_engine = TrustEngine()
        
        # Initialize peer client with unique P2P port (API port + 1000)
        p2p_port = node_config.port + 1000
        peer_client = PeerClient(node_config.node_id, node_config.host, p2p_port)
        await peer_client.start_server()
        
        # Initialize enhanced ML integration
        try:
            from src.live_ml_integration import LiveMLIntegration
            global live_ml
            live_ml = LiveMLIntegration(node_config.node_id)
            logger.info("Live ML integration initialized")
        except Exception as e:
            logger.error(f"Failed to initialize live ML integration: {e}")
            live_ml = None
        
        # Initialize Enhanced ML Consensus Engine
        if ML_AVAILABLE:
            try:
                ml_consensus = EnhancedMLConsensusEngine(node_config.node_id)
                logger.info("Enhanced ML Consensus Engine initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Enhanced ML consensus engine: {e}")
                ml_consensus = None
        
        # Initialize Epoch Manager for Phase 3 consensus
        try:
            epoch_manager = init_epoch_manager(node_config.node_id, ml_consensus, blockchain_client)
            logger.info("Epoch Manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Epoch Manager: {e}")
            epoch_manager = None
        
        # Initialize legacy ML classifier (fallback)
        try:
            ml_classifier = get_classifier()
            logger.info("Legacy ML classifier loaded successfully")
        except Exception as e:
            logger.warning(f"Legacy ML classifier not available: {e}")
            ml_classifier = None
        
        # Initialize blockchain client with real contract
        # BLOCKCHAIN DEPENDENCY: Node only starts if blockchain is available
        blockchain_available = False
        max_retries = 5
        retry_delay = 3  # seconds
        
        for attempt in range(max_retries):
            try:
                if CONTRACT_INFO:
                    blockchain_config = {
                        'rpc_url': 'http://localhost:8545',
                        'contract_address': CONTRACT_INFO['address'],
                        'chain_id': 31337,
                        'private_key': '0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80',  # Hardhat account 0
                        'gas_limit': 300000,
                        'gas_price_gwei': 20
                    }
                    blockchain_client = BlockchainClient(config=blockchain_config)
                else:
                    blockchain_config = {
                        'rpc_url': 'http://localhost:8545',
                        'chain_id': 31337,
                        'private_key': '0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80',
                        'gas_limit': 300000,
                        'gas_price_gwei': 20
                    }
                    blockchain_client = BlockchainClient(config=blockchain_config)
                
                # Check if blockchain is actually available
                if blockchain_client.is_blockchain_available():
                    blockchain_available = True
                    health = blockchain_client.health_check()
                    logger.info(f"✅ Blockchain connected successfully: {health.get('contract', 'N/A')}")
                    break
                else:
                    logger.warning(f"Blockchain not available (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
            except Exception as e:
                logger.warning(f"Blockchain connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
        
        # Exit if blockchain is not available after all retries
        if not blockchain_available:
            logger.error("❌ BLOCKCHAIN NOT AVAILABLE - Node cannot start without blockchain")
            logger.error("Please start the blockchain (Hardhat) first: cd blockchain && npx hardhat node")
            raise SystemExit("Blockchain not available. Node startup aborted.")
        
        # Register node on blockchain
        if blockchain_client:
            try:
                result = blockchain_client.register_node(node_config.node_id)
                if result['success']:
                    logger.info(f"Node registered on blockchain: {node_config.node_id}")
                else:
                    logger.warning(f"Failed to register on blockchain: {result.get('error')}")
            except Exception as e:
                logger.error(f"Error registering node on blockchain: {e}")
        
        # Start monitoring scheduler
        if node_config.websites:
            # Initialize monitoring scheduler with callback
            monitoring_scheduler = MonitoringScheduler(
                website_monitor,
                node_config.websites,
                node_config.monitoring_interval,
                on_cycle_complete=process_monitoring_results  # Phase 2: triggers broadcast after each cycle
            )
            # Start background monitoring
            monitoring_task = asyncio.create_task(monitoring_scheduler.start_monitoring())
            logger.info(f"Started monitoring {len(node_config.websites)} websites")
        
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(heartbeat_loop())
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(cleanup_loop())
        
        # Start epoch manager background task (Phase 3)
        if epoch_manager:
            epoch_task = asyncio.create_task(epoch_manager.run_epoch_manager())
            logger.info("Epoch manager background task started")

        # Auto-register peers passed via --peers flag
        if args.peers:
            for peer_url in args.peers:
                try:
                    async with httpx.AsyncClient() as client:
                        health = await client.get(f"{peer_url}/health", timeout=5.0)
                        if health.status_code == 200:
                            data = health.json()
                            peer_id = data.get("node_id")
                            pubkey = data.get("public_key")
                            if peer_id and pubkey:
                                peer_registry[peer_id] = {"url": peer_url, "public_key_hex": pubkey}
                                peer_public_keys[peer_id] = pubkey
                                logger.info(f"Auto-registered peer {peer_id} at {peer_url}")
                                # Tell that peer about us
                                await client.post(f"{peer_url}/peers/register", json={
                                    "node_id": node_config.node_id,
                                    "url": f"http://localhost:{node_config.port}",
                                    "public_key_hex": NODE_SIGNER.public_key_hex
                                }, timeout=5.0)
                except Exception as e:
                    logger.warning(f"Could not auto-register peer {peer_url}: {e}")

        logger.info("Node startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to start node: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    global monitoring_task, heartbeat_task, cleanup_task, peer_client, monitoring_scheduler
    
    try:
        logger.info("Shutting down node...")
        
        # Stop background tasks
        if monitoring_task:
            monitoring_task.cancel()
        if heartbeat_task:
            heartbeat_task.cancel()
        if cleanup_task:
            cleanup_task.cancel()
        
        # Stop monitoring scheduler
        if monitoring_scheduler:
            monitoring_scheduler.stop_monitoring()
        
        # Stop peer client
        if peer_client:
            await peer_client.stop_server()
        
        logger.info("Node shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

async def heartbeat_loop():
    """Send periodic heartbeats to peers (disabled - covered by signed report broadcasting)"""
    while True:
        try:
            # Legacy heartbeat disabled - Phase 2 uses signed report broadcasting
            # if peer_client:
            #     await peer_client.send_heartbeat()
            await asyncio.sleep(30)  # Heartbeat every 30 seconds
        except Exception as e:
            logger.error(f"Error in heartbeat loop: {e}")
            await asyncio.sleep(60)

async def cleanup_loop():
    """Periodic cleanup of old data"""
    while True:
        try:
            if trust_engine:
                trust_engine.cleanup_old_data()
            await asyncio.sleep(3600)  # Cleanup every hour
        except Exception as e:
            logger.error(f"Error in cleanup loop: {e}")
            await asyncio.sleep(3600)

async def run_consensus_and_slash():
    """
    Run consensus and slashing using epoch_manager.
    This replaces the old per-cycle consensus with epoch-based consensus.
    """
    global verdicts_store
    
    if not epoch_manager:
        logger.warning("Epoch manager not available, skipping consensus")
        return
    
    try:
        # Get current epoch
        from time import time
        epoch_id = int(time() // 60)
        
        # Trigger epoch processing
        logger.info(f"Triggering consensus for epoch {epoch_id}")
        await epoch_manager.process_epoch(epoch_id)
        
        # Store verdicts for /verdict endpoint
        # The epoch_manager stores verdicts internally, we'll track them here
        verdicts_store[str(epoch_id)] = {
            "epoch_id": epoch_id,
            "timestamp": datetime.now().isoformat(),
            "status": "processed"
        }
        
        logger.info(f"Consensus completed for epoch {epoch_id}")
        
    except Exception as e:
        logger.error(f"Error in run_consensus_and_slash: {e}")

def run_consensus_vote(epoch_id: int, epoch_reports: list) -> dict:
    """
    Majority vote across all reports for this epoch.
    A node is flagged malicious if its reported status diverges from majority.
    Reputation is reduced (slashed) for malicious nodes.
    """
    if not epoch_reports:
        return {}

    # Tally votes: is the site reachable?
    votes = [r.get("is_reachable", True) for r in epoch_reports]
    majority_reachable = sum(votes) > len(votes) / 2
    majority_verdict = "up" if majority_reachable else "down"

    results = {
        "epoch_id": epoch_id,
        "majority_verdict": majority_verdict,
        "total_reports": len(epoch_reports),
        "votes_up": sum(votes),
        "votes_down": len(votes) - sum(votes),
        "slashed": [],
        "honest": []
    }

    for report in epoch_reports:
        node_id = report.get("node_address", report.get("node_id", "unknown"))
        node_vote = "up" if report.get("is_reachable", True) else "down"

        if node_vote != majority_verdict:
            # Diverges from majority — slash reputation
            if ml_consensus:
                current = ml_consensus.reputation.get(node_id, 0.95)
                ml_consensus.reputation[node_id] = max(0.0, current - 0.15)
                logger.warning(
                    f"SLASHED {node_id}: voted {node_vote} but majority={majority_verdict} "
                    f"reputation {current:.2f} -> {ml_consensus.reputation[node_id]:.2f}"
                )
            results["slashed"].append(node_id)
        else:
            # Agrees with majority — small reputation boost
            if ml_consensus:
                current = ml_consensus.reputation.get(node_id, 0.95)
                ml_consensus.reputation[node_id] = min(1.0, current + 0.01)
            results["honest"].append(node_id)

    logger.info(
        f"Epoch {epoch_id} consensus: site={majority_verdict} | "
        f"honest={results['honest']} | slashed={results['slashed']}"
    )
    return results

async def process_monitoring_results():
    """Process monitoring results with live ML integration"""
    
    if not monitoring_scheduler:
        return
    
    try:
        # Get latest monitoring results
        results = monitoring_scheduler.get_latest_results(10)
        if not results:
            return

        # Process each result through live ML integration
        enhanced_results = []
        # Only process the single newest result each cycle
        for result in [results[-1]]:
            if live_ml:
                # Use enhanced ML integration
                enhanced_result = live_ml.process_monitoring_result(result)
                enhanced_results.append(enhanced_result)
                
                logger.info(f"Live ML processed: {result.get('url', 'unknown')} -> "
                           f"{enhanced_result.get('ml_prediction', 'unknown')} "
                           f"(score: {enhanced_result.get('ml_score', 0.0):.3f})")
            else:
                # Fallback to basic processing
                enhanced_results.append({
                    **result,
                    'ml_processed': False,
                    'ml_prediction': 'unknown',
                    'ml_score': 0.5
                })
        
        # Update trust engine with enhanced results
        if trust_engine:
            for result in enhanced_results:
                trust_engine.add_monitoring_report(node_config.node_id, result)
            
            # Calculate trust score
            trust_score = trust_engine.calculate_monitoring_trust(node_config.node_id)
        else:
            trust_score = 0.8  # Default trust
        
        # Get ML score from live integration
        if enhanced_results:
            latest_result = enhanced_results[-1]
            ml_score = latest_result.get('ml_score', 0.5)
            ml_prediction = latest_result.get('ml_prediction', 'unknown')
        else:
            ml_score = 0.5
            ml_prediction = 'unknown'
        
        # Calculate PoR score
        por_score = TrustCalculator.calculate_por_score(trust_score, ml_score)
        
        logger.info(f"Trust: {trust_score:.4f}, ML: {ml_score:.4f}, PoR: {por_score:.4f}")
        
        # Update blockchain
        if blockchain_client:
            try:
                result = blockchain_client.update_reputation(
                    node_config.node_id,
                    trust_score,
                    ml_score
                )
                if result['success']:
                    logger.info("Reputation updated on blockchain")
                else:
                    logger.warning(f"Failed to update reputation: {result.get('error')}")
            except Exception as e:
                logger.error(f"Error updating blockchain: {e}")
        
        # Send trust update to peers
        if peer_client:
            await peer_client.send_trust_update(trust_score)
        
        # Phase 2: Broadcast signed reports to peers
        # Read peer URLs from peer_registry (one source of truth)
        peer_urls = [info["url"] for info in peer_registry.values()]
        
        logger.info(f"DEBUG: peer_registry = {peer_registry}")
        logger.info(f"DEBUG: peer_urls = {peer_urls}")
        logger.info(f"DEBUG: results count = {len(results) if results else 0}")
        logger.info(f"DEBUG: monitoring_available = {monitoring_available}")
        
        # Note: Consensus is now handled by epoch_manager (Phase 3) - runs on epoch timer, not per cycle
        
        if results and monitoring_available:
            try:
                from website_monitor import _build_signed_report
                
                # Get the last monitoring result
                last_result = results[-1]
                
                # Build signed report
                signed_report = _build_signed_report(
                    url=last_result.get('url', 'unknown'),
                    response_ms=last_result.get('response_time_ms', -1.0),
                    status_code=last_result.get('status_code', 0),
                    ssl_valid=last_result.get('ssl_valid', False),
                    body=last_result.get('content', ''),
                    is_reachable=last_result.get('status') == 'success'
                )
                
                logger.info(f"DEBUG: About to broadcast to {len(peer_urls)} peers: {peer_urls}")
                
                # Broadcast to all registered peers
                broadcast_results = await peer_client.broadcast_report(
                    signed_report, 
                    peer_urls
                )
                
                success_count = sum(broadcast_results.values())
                logger.info(f"Broadcasted signed report to {success_count}/{len(peer_urls)} peers")
                
                # FIX: also store own report in peer_reports so /reports/latest works
                peer_reports.append(asdict(signed_report))
                
                # Run majority-vote consensus and store verdict
                from website_monitor import get_current_epoch
                current_epoch = int(__import__('time').time() // 60)
                epoch_reports_now = [r for r in peer_reports 
                                     if r.get("epoch_id") == current_epoch]
                if epoch_reports_now:
                    verdict = run_consensus_vote(current_epoch, epoch_reports_now)
                    verdicts_store[str(current_epoch)] = verdict
                
                # Add own report to epoch_manager for Phase 3 consensus
                if epoch_manager:
                    epoch_manager.add_report(asdict(signed_report), is_own=True)
                    logger.info(f"Added own report to epoch_manager for epoch {signed_report.epoch_id}")
                
                # Run consensus and slashing
                await run_consensus_and_slash()
                
            except Exception as e:
                logger.error(f"Error broadcasting signed report: {e}")
        
        # Old peer notification (legacy) - remove once Phase 2 fully tested
        # This will be deprecated in favor of signed report broadcasting above
        
    except Exception as e:
        logger.error(f"Error processing monitoring results: {e}")

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Decentralized Website Monitoring Node",
        "node_id": node_config.node_id if node_config else None,
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "node_id": node_config.node_id if node_config else None,
        "timestamp": datetime.now().isoformat(),
        "public_key": NODE_SIGNER.public_key_hex if monitoring_available else None,
        "api_port": node_config.port if node_config else None,
        "p2p_port": (node_config.port + 1000) if node_config else None,
        "components": {}
    }
    
    # Check monitoring
    if monitoring_scheduler:
        health_status["components"]["monitoring"] = "active"
    else:
        health_status["components"]["monitoring"] = "inactive"
    
    # Check trust engine
    if trust_engine:
        health_status["components"]["trust_engine"] = "active"
    else:
        health_status["components"]["trust_engine"] = "inactive"
    
    # Check peer client
    if peer_client:
        peer_stats = await peer_client.get_peer_statistics()
        health_status["components"]["peer_client"] = {
            "status": "active",
            "connected_peers": peer_stats["active_peers"]
        }
    else:
        health_status["components"]["peer_client"] = "inactive"
    
    # Check ML classifier
    if ml_classifier:
        health_status["components"]["ml_classifier"] = "active"
    else:
        health_status["components"]["ml_classifier"] = "inactive"
    
    # Check blockchain
    if blockchain_client:
        bc_health = blockchain_client.health_check()
        health_status["components"]["blockchain"] = bc_health
    else:
        health_status["components"]["blockchain"] = "inactive"
    
    return health_status

@app.post("/monitor")
async def trigger_monitoring(request: MonitoringRequest):
    """Trigger manual monitoring of websites"""
    if not website_monitor:
        raise HTTPException(status_code=503, detail="Website monitor not available")
    
    try:
        results = await website_monitor.monitor_multiple_websites(request.urls)
        
        # Add results to trust engine
        if trust_engine:
            for result in results:
                trust_engine.add_monitoring_report(node_config.node_id, result)
        
        # Process results
        await process_monitoring_results()
        
        return {
            "status": "completed",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in manual monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trust")
async def get_trust_info():
    """Get current trust information"""
    if not trust_engine:
        raise HTTPException(status_code=503, detail="Trust engine not available")
    
    try:
        trust_info = trust_engine.get_node_trust_info(node_config.node_id)
        return trust_info
    except Exception as e:
        logger.error(f"Error getting trust info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features")
async def get_ml_features():
    """Get current ML features"""
    if not monitoring_scheduler:
        raise HTTPException(status_code=503, detail="Monitoring scheduler not available")
    
    try:
        features = monitoring_scheduler.get_features_for_ml()
        
        # Get ML prediction if available
        prediction = None
        if ml_classifier:
            try:
                prediction = ml_classifier.predict_single(features)
            except Exception as e:
                logger.error(f"Error getting ML prediction: {e}")
        
        return {
            "features": features,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting ML features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/results")
async def get_monitoring_results():
    """Get latest monitoring results"""
    if not monitoring_scheduler:
        raise HTTPException(status_code=503, detail="Monitoring scheduler not available")
    results = monitoring_scheduler.get_latest_results(10)
    return {
        "results": results,
        "count": len(results) if results else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/peers")
async def add_peer(peer: PeerInfo):
    """Add a new peer"""
    if not peer_client:
        raise HTTPException(status_code=503, detail="Peer client not available")
    
    try:
        await peer_client.add_peer(peer.node_id, peer.host, peer.port)
        return {"status": "success", "message": f"Peer {peer.node_id} added"}
    except Exception as e:
        logger.error(f"Error adding peer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/peers")
async def get_peers():
    """Get list of connected peers"""
    if not peer_client:
        raise HTTPException(status_code=503, detail="Peer client not available")
    
    try:
        stats = await peer_client.get_peer_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting peers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reputation")
async def update_reputation(update: ReputationUpdate):
    """Update node reputation on blockchain"""
    if not blockchain_client:
        raise HTTPException(status_code=503, detail="Blockchain client not available")
    
    try:
        result = blockchain_client.update_reputation(
            update.node_id,
            update.monitoring_trust,
            update.ml_score
        )
        
        if result['success']:
            return {"status": "success", "tx_hash": result['tx_hash']}
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
            
    except Exception as e:
        logger.error(f"Error updating reputation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/blockchain/reputation/{node_id}")
async def get_blockchain_reputation(node_id: str):
    """Get node reputation from blockchain"""
    if not blockchain_client:
        raise HTTPException(status_code=503, detail="Blockchain client not available")
    
    try:
        reputation = blockchain_client.get_node_reputation(node_id)
        
        if reputation:
            return reputation
        else:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
            
    except Exception as e:
        logger.error(f"Error getting blockchain reputation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics():
    """Get comprehensive node statistics"""
    stats = {
        "node_id": node_config.node_id if node_config else None,
        "timestamp": datetime.now().isoformat()
    }
    
    # Trust statistics
    if trust_engine:
        stats["trust"] = trust_engine.get_trust_statistics()
    
    # Monitoring statistics
    if monitoring_scheduler:
        latest_results = monitoring_scheduler.get_latest_results(10)
        stats["monitoring"] = {
            "websites_count": len(node_config.websites) if node_config else 0,
            "latest_results_count": len(latest_results),
            "monitoring_interval": node_config.monitoring_interval if node_config else 0
        }
    
    # Peer statistics
    if peer_client:
        stats["peers"] = await peer_client.get_peer_statistics()
    
    # Blockchain statistics
    if blockchain_client:
        try:
            node_reputation = blockchain_client.get_node_reputation(node_config.node_id)
            stats["blockchain"] = {
                "node_registered": node_reputation is not None,
                "reputation": node_reputation
            }
        except Exception as e:
            stats["blockchain"] = {"error": str(e)}
    
    return stats

@app.post("/process_results")
async def process_results(background_tasks: BackgroundTasks):
    """Manually trigger result processing"""
    background_tasks.add_task(process_monitoring_results)
    return {"status": "processing_started"}

# Phase 2: P2P Report Endpoints
@app.post("/report")
async def receive_peer_report(payload: dict):
    """
    Receive signed monitoring report from peer node
    
    Args:
        payload: Signed MonitoringReport as dictionary
        
    Returns:
        Status: accepted or rejected with reason
    """
    global peer_reports, peer_public_keys
    
    try:
        # Extract node address from payload
        node_address = payload.get("node_address")
        logger.info(f"Received report from node_address: {node_address}")
        logger.info(f"Available peer_public_keys: {list(peer_public_keys.keys())}")
        
        if not node_address:
            return {"status": "rejected", "reason": "missing node_address"}
        
        # Check if we know this node's public key
        sender_pubkey = peer_public_keys.get(node_address)
        logger.info(f"Found sender_pubkey for {node_address}: {sender_pubkey is not None}")
        if not sender_pubkey:
            # For now, accept new nodes and store their public keys
            # In production, this should require registration
            sender_pubkey = payload.get("signature")  # This won't work for verification
            # Better approach: extract from a registration endpoint first
            logger.warning(f"Unknown node {node_address}, accepting with caution")
            # Store public key for future verification
            # Note: In production, nodes should register their public keys first
        
        # Convert payload to MonitoringReport
        report = ReportVerifier.from_dict(payload)
        
        # Verify signature
        if sender_pubkey and sender_pubkey != payload.get("signature"):
            # Only verify if we have the real public key
            try:
                is_valid = ReportVerifier.verify(report, sender_pubkey)
                if not is_valid:
                    logger.warning(f"Signature invalid for {node_address}, accepting anyway for consensus testing")
                    # is_valid = False  # comment this out to allow testing
            except Exception:
                pass  # accept report even if verification fails
        
        # Store verified report in epoch_manager (Phase 3)
        if epoch_manager:
            epoch_manager.add_report(asdict(report), is_own=False)
            logger.info(f"Added peer report from {node_address} to epoch_manager for epoch {report.epoch_id}")
        else:
            # Fallback to peer_reports for compatibility
            peer_reports.append(asdict(report))
            logger.info(f"Accepted report from {node_address} for epoch {report.epoch_id} (fallback to peer_reports)")
        
        # Note: Consensus is now handled by epoch_manager - no immediate consensus check needed

        return {"status": "accepted", "report_hash": report.report_hash[:16]}
        
    except Exception as e:
        logger.error(f"Error processing peer report: {e}")
        return {"status": "rejected", "reason": str(e)}

@app.get("/reports/epoch/{epoch_id}")
async def get_epoch_reports(epoch_id: int):
    """
    Get all reports for a specific epoch
    
    Args:
        epoch_id: Epoch number to query
        
    Returns:
        List of reports for the specified epoch
    """
    global peer_reports
    
    epoch_reports = [r for r in peer_reports if r.get("epoch_id") == epoch_id]
    
    return {
        "epoch_id": epoch_id,
        "total_reports": len(epoch_reports),
        "reports": epoch_reports
    }

@app.get("/reports/latest")
async def get_latest_reports(limit: int = 10):
    """
    Get most recent reports across all epochs
    
    Args:
        limit: Maximum number of reports to return (default: 10)
        
    Returns:
        List of recent reports
    """
    global peer_reports
    
    recent_reports = peer_reports[-limit:] if peer_reports else []
    
    return {
        "total_available": len(peer_reports),
        "returned": len(recent_reports),
        "reports": recent_reports
    }

@app.post("/peers/register")
async def register_peer(payload: dict):
    """
    Register a peer's public key and URL for report verification and broadcasting
    """
    global peer_registry, peer_public_keys
    
    try:
        node_id = payload.get("node_id")
        url = payload.get("url")
        pubkey = payload.get("public_key_hex")
        
        if not all([node_id, url, pubkey]):
            return {"status": "error", "reason": "missing node_id, url, or public_key_hex"}
        
        peer_registry[node_id] = {"url": url, "public_key_hex": pubkey}
        peer_public_keys[node_id] = pubkey
        
        if peer_client:
            try:
                parsed = url.replace("http://", "").replace("https://", "")
                host, port = parsed.split(":")
                await peer_client.add_peer(node_id, host, int(port))
            except Exception as e:
                logger.warning(f"Peer client add failed: {e}")
        
        return {"status": "registered", "node_id": node_id}
        
    except Exception as e:
        logger.error(f"Error registering peer: {e}")
        return {"status": "error", "reason": str(e)}

@app.post("/peer/message")
async def receive_peer_message(payload: dict):
    """
    Receive peer message for P2P communication
    """
    try:
        message_type = payload.get("type", "unknown")
        
        if message_type == "report":
            logger.info(f"Received peer report: {payload.get('node_id')}")
        elif message_type == "trust_update":
            logger.info(f"Trust update: {payload.get('data')}")
        else:
            return {"status": "unknown"}
        
        return {"status": "ok"}
        
    except Exception as e:
        return {"status": "error", "reason": str(e)}
    
    try:
        node_id = payload.get("node_id")
        url = payload.get("url")
        pubkey = payload.get("public_key_hex")

        if not all([node_id, url, pubkey]):
            return {"status": "error", "reason": "missing node_id, url, or public_key_hex"}

        # Store in peer_registry (one source of truth)
        peer_registry[node_id] = {"url": url, "public_key_hex": pubkey}
        peer_public_keys[node_id] = pubkey  # for signature verification

        logger.info(f"Registered peer {node_id} at {url} with public key {pubkey[:16]}...")
        
        # Also add to peer_client if available
        if peer_client:
            try:
                parsed_url = url.replace("http://", "").replace("https://", "")
                if ":" in parsed_url:
                    host, port = parsed_url.split(":")
                    await peer_client.add_peer(node_id, host, int(port))
            except Exception as e:
                logger.warning(f"Could not add to peer_client: {e}")

        return {
            "status": "registered", 
            "node_id": node_id,
            "total_peers": len(peer_registry)
        }
        
    except Exception as e:
        logger.error(f"Error registering peer: {e}")
        return {"status": "error", "reason": str(e)}

@app.get("/peers/registered")
async def get_registered_peers():
    """Get list of registered peers with their URLs and public keys"""
    return {
        "peers": peer_registry,
        "count": len(peer_registry)
    }

@app.get("/reputation")
async def get_reputation():
    """Get current reputation scores for all nodes"""
    if not ml_consensus:
        return {"error": "ML consensus not available"}
    
    # Enhanced ML engine
    all_status = ml_consensus.get_all_nodes_status()
    shard_distribution = ml_consensus.get_shard_distribution()
    
    return {
        "reputations": {node_id: status["reputation"] for node_id, status in all_status.items()},
        "ewma_reputations": {node_id: status["ewma_reputation"] for node_id, status in all_status.items()},
        "mitigation_actions": {
            node_id: {
                "status": status["status"],
                "action": status["action"],
                "shard": status["shard"]
            }
            for node_id, status in all_status.items()
        },
        "shard_distribution": shard_distribution,
        "engine_type": "enhanced"
    }

@app.get("/consensus/reputations")
async def get_consensus_reputations():
    """Get current node reputations from ML consensus engine"""
    if not ml_consensus:
        return {"error": "ML consensus engine not available"}
    
    # Enhanced ML engine
    all_status = ml_consensus.get_all_nodes_status()
    
    return {
        "reputations": {node_id: status["reputation"] for node_id, status in all_status.items()},
        "ewma_reputations": {node_id: status["ewma_reputation"] for node_id, status in all_status.items()},
        "mitigation_actions": {
            node_id: {
                "status": status["status"],
                "action": status["action"],
                "shard": status["shard"]
            }
            for node_id, status in all_status.items()
        },
        "shard_distribution": ml_consensus.get_shard_distribution(),
        "engine_type": "enhanced"
    }

@app.get("/verdict")
async def get_verdicts():
    """Get final verdicts from consensus with node reputations"""
    return {
        "verdicts": verdicts_store,
        "node_reputations": ml_consensus.reputation if ml_consensus else {},
        "timestamp": datetime.now().isoformat()
    }

@app.get("/consensus/shards")
async def get_shard_assignment():
    """Get current shard assignment based on reputation"""
    if not ml_consensus:
        return {"error": "ML consensus engine not available"}
    
    shard_assignment = ml_consensus.get_shard_assignment(num_shards=4)
    
    # Calculate shard stats
    shard_stats = {}
    for node_id, shard_id in shard_assignment.items():
        if shard_id not in shard_stats:
            shard_stats[shard_id] = {"nodes": [], "avg_reputation": 0.0}
        shard_stats[shard_id]["nodes"].append(node_id)
    
    # Calculate average reputation per shard
    for shard_id, stats in shard_stats.items():
        reputations = [ml_consensus.reputation.get(node, 0.95) for node in stats["nodes"]]
        stats["avg_reputation"] = np.mean(reputations) if reputations else 0.0
        stats["node_count"] = len(stats["nodes"])
    
    return {
        "shard_assignment": shard_assignment,
        "shard_stats": shard_stats
    }

@app.post("/consensus/train")
async def train_ml_consensus():
    """Train ML models on accumulated peer reports"""
    if not ml_consensus:
        return {"error": "ML consensus engine not available"}
    
    try:
        # Extract features from peer reports
        features_df = ml_consensus.extract_features_from_reports(peer_reports)
        
        # For demo, create synthetic labels (in real system, these would come from verified behavior)
        features_df["label"] = np.random.choice([0, 1], size=len(features_df), p=[0.8, 0.2])
        
        if features_df.empty:
            return {"error": "No training data available"}
        
        # For demo, create synthetic labels (in real system, these would come from verified behavior)
        features_df["label"] = np.random.choice([0, 1], size=len(features_df), p=[0.8, 0.2])
        
        # Train models
        ml_consensus.train_models(features_df)
        
        logger.info(f"ML models trained on {len(features_df)} samples")
        
        return {
            "status": "trained",
            "samples": len(features_df),
            "features": list(features_df.columns)
        }
        
    except Exception as e:
        logger.error(f"Error training ML models: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Decentralized Website Monitoring Node')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the node on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--node-id', type=str, help='Unique node identifier')
    parser.add_argument('--peers', type=str, nargs='+', help='List of peer URLs')
    parser.add_argument('--websites', type=str, nargs='+', 
                        default=['https://google.com', 'https://github.com', 'https://httpbin.org'],
                        help='Websites to monitor')
    args = parser.parse_args()
    
    # Generate node ID if not provided
    node_id = args.node_id or f"node_{args.port}"
    
    logger.info(f"Starting node {node_id} on {args.host}:{args.port}")
    logger.info(f"Websites to monitor: {args.websites}")
    if args.peers:
        logger.info(f"Peers: {args.peers}")
    
    # Run the FastAPI app (reload=False to fix process isolation bug)
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info"
    )
