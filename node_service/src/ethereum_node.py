"""
Ethereum-Native Monitoring Node
Optimized for Ethereum L1 + L2 integration with gas efficiency
Supports Arbitrum, Optimism, and Base for high-throughput monitoring
"""

import asyncio
import aiohttp
import ssl
import dns.resolver
import hashlib
import time
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import uvloop
import concurrent.futures
from multiprocessing import cpu_count
import redis
import aioredis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
import pydantic
from pydantic import BaseModel, validator
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account
from web3.exceptions import TransactionNotFound

# Set event loop for high performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/ethereum_monitoring_node.log')
    ]
)
logger = logging.getLogger(__name__)

# ===== ETHEREUM CONFIGURATION =====

@dataclass
class EthereumConfig:
    """Ethereum ecosystem configuration"""
    
    # Layer-1 Configuration
    ETHEREUM_RPC_URL: str = "https://eth-mainnet.alchemyapi.io/v2/YOUR_API_KEY"
    ETHEREUM_CHAIN_ID: int = 1
    ETHEREUM_GAS_MULTIPLIER: float = 1.0
    
    # Layer-2 Configuration
    L2_CONFIGS: Dict[str, Dict] = {
        'arbitrum': {
            'rpc_url': 'https://arb1.arbitrum.io/rpc',
            'chain_id': 42161,
            'gas_multiplier': 0.1,  # 10x cheaper
            'batch_size': 1000,
            'priority': 'high_throughput'
        },
        'optimism': {
            'rpc_url': 'https://mainnet.optimism.io',
            'chain_id': 10,
            'gas_multiplier': 0.05,  # 20x cheaper
            'batch_size': 500,
            'priority': 'low_latency'
        },
        'base': {
            'rpc_url': 'https://mainnet.base.org',
            'chain_id': 8453,
            'gas_multiplier': 0.03,  # 30x cheaper
            'batch_size': 500,
            'priority': 'cost_effective'
        }
    }
    
    # Contract Configuration
    L1_CONTRACT_ADDRESS: str = ""
    MON_TOKEN_ADDRESS: str = ""
    PRIVATE_KEY: str = ""
    
    # Performance Settings
    MAX_CONCURRENT_REQUESTS: int = 5000
    BATCH_TIMEOUT: float = 0.1
    WORKER_PROCESSES: int = cpu_count()
    
    # Gas Optimization
    GAS_PRICE_STRATEGY: str = "eip1559"  # eip1559 or legacy
    MAX_GAS_PRICE: int = 1000 * 10**9  # 1000 gwei
    PRIORITY_FEE_MULTIPLIER: float = 1.2
    
    # Monitoring Targets (Production URLs)
    CRITICAL_TARGETS: List[str] = [
        "https://ethereum.org",
        "https://etherscan.io",
        "https://uniswap.org",
        "https://opensea.io",
        "https://metamask.io"
    ]
    
    HIGH_PRIORITY_TARGETS: List[str] = [
        "https://curve.fi",
        "https://aave.com",
        "https://compound.finance",
        "https://makerdao.com"
    ]
    
    REGULAR_TARGETS: List[str] = [
        "https://sushi.com",
        "https://1inch.io",
        "https://balancer.fi",
        "https://yearn.finance"
    ]

config = EthereumConfig()

# ===== DATA MODELS =====

class EthereumMonitoringReport(BaseModel):
    """Ethereum-optimized monitoring report"""
    
    report_hash: str
    node_id: str
    target_url: str
    response_time: int
    is_reachable: bool
    ssl_valid: bool
    status_code: int
    timestamp: int
    epoch_id: int
    shard_id: int
    priority: str  # critical, high, regular
    
    @validator('priority')
    def validate_priority(cls, v):
        if v not in ['critical', 'high', 'regular']:
            raise ValueError('Priority must be critical, high, or regular')
        return v

class Layer2Batch(BaseModel):
    """Layer-2 batch submission model"""
    
    batch_hash: str
    data_commitment: str  # IPFS hash
    target_layer: str  # arbitrum, optimism, base
    report_count: int
    timestamp: int
    submitter: str

class GasOptimization(BaseModel):
    """Gas optimization parameters"""
    
    base_fee: int
    priority_fee: int
    max_fee: int
    estimated_gas: int
    total_cost: int
    layer: str

# ===== ETHEREUM LAYER MANAGER =====

class EthereumLayerManager:
    """Manages connections to Ethereum L1 and L2 networks"""
    
    def __init__(self, config: EthereumConfig):
        self.config = config
        self.connections = {}
        self.contracts = {}
        self.accounts = {}
        self.gas_optimizers = {}
        
    async def initialize(self):
        """Initialize all layer connections"""
        try:
            # Connect to Ethereum L1
            await self._connect_to_layer('ethereum', self.config.ETHEREUM_RPC_URL)
            
            # Connect to Layer-2 networks
            for layer_name, layer_config in self.config.L2_CONFIGS.items():
                await self._connect_to_layer(layer_name, layer_config['rpc_url'])
            
            # Setup contracts
            await self._setup_contracts()
            
            # Initialize gas optimizers
            await self._initialize_gas_optimizers()
            
            logger.info("Ethereum layer manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize layer manager: {e}")
            raise
    
    async def _connect_to_layer(self, layer_name: str, rpc_url: str):
        """Connect to specific Ethereum layer"""
        try:
            web3 = Web3(Web3.HTTPProvider(rpc_url))
            
            # Add POA middleware for L2 networks
            if layer_name != 'ethereum':
                web3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            # Test connection
            if not web3.is_connected():
                raise Exception(f"Failed to connect to {layer_name}")
            
            # Setup account
            account = Account.from_key(self.config.PRIVATE_KEY)
            
            self.connections[layer_name] = web3
            self.accounts[layer_name] = account
            
            logger.info(f"Connected to {layer_name}: {web3.eth.chain_id}")
            
        except Exception as e:
            logger.error(f"Failed to connect to {layer_name}: {e}")
            raise
    
    async def _setup_contracts(self):
        """Setup smart contract instances"""
        try:
            # Load contract ABI
            with open('blockchain/artifacts/EthereumMonitoring.json', 'r') as f:
                contract_data = json.load(f)
            
            contract_abi = contract_data['abi']
            
            # Setup L1 contract
            self.contracts['ethereum'] = self.connections['ethereum'].eth.contract(
                address=self.config.L1_CONTRACT_ADDRESS,
                abi=contract_abi
            )
            
            # Setup L2 contracts (same ABI, different addresses)
            for layer_name in self.config.L2_CONFIGS.keys():
                # In production, each L2 would have its own contract address
                self.contracts[layer_name] = self.connections[layer_name].eth.contract(
                    address=self.config.L1_CONTRACT_ADDRESS,  # Same address for demo
                    abi=contract_abi
                )
            
            logger.info("Contracts setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup contracts: {e}")
            raise
    
    async def _initialize_gas_optimizers(self):
        """Initialize gas optimizers for each layer"""
        for layer_name in self.connections.keys():
            self.gas_optimizers[layer_name] = EthereumGasOptimizer(
                self.connections[layer_name],
                self.config.L2_CONFIGS.get(layer_name, {}).get('gas_multiplier', 1.0)
            )
    
    def get_optimal_layer(self, priority: str) -> str:
        """Get optimal layer based on priority"""
        if priority == 'critical':
            return 'optimism'  # Fastest finality
        elif priority == 'high':
            return 'arbitrum'   # High throughput
        else:
            return 'base'       # Most cost-effective
    
    async def submit_report(self, report: EthereumMonitoringReport) -> str:
        """Submit report to optimal layer"""
        optimal_layer = self.get_optimal_layer(report.priority)
        
        if optimal_layer == 'ethereum':
            return await self._submit_to_l1(report)
        else:
            return await self._submit_to_l2(report, optimal_layer)
    
    async def _submit_to_l1(self, report: EthereumMonitoringReport) -> str:
        """Submit report directly to Ethereum L1"""
        try:
            web3 = self.connections['ethereum']
            contract = self.contracts['ethereum']
            account = self.accounts['ethereum']
            
            # Get gas optimization
            gas_opt = await self.gas_optimizers['ethereum'].get_optimal_gas()
            
            # Build transaction
            tx_data = contract.functions.submitReport(
                report.report_hash,
                report.target_url,
                report.response_time,
                report.is_reachable,
                report.ssl_valid,
                report.status_code
            ).build_transaction({
                'from': account.address,
                'gas': gas_opt.estimated_gas,
                'maxFeePerGas': gas_opt.max_fee,
                'maxPriorityFeePerGas': gas_opt.priority_fee,
                'nonce': web3.eth.get_transaction_count(account.address),
                'value': gas_opt.total_cost,
                'chainId': self.config.ETHEREUM_CHAIN_ID
            })
            
            # Sign and send transaction
            signed_tx = web3.eth.account.sign_transaction(tx_data, self.config.PRIVATE_KEY)
            tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            logger.info(f"L1 report submitted: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Failed to submit L1 report: {e}")
            raise
    
    async def _submit_to_l2(self, report: EthereumMonitoringReport, layer_name: str) -> str:
        """Submit report to Layer-2 network"""
        try:
            web3 = self.connections[layer_name]
            contract = self.contracts[layer_name]
            account = self.accounts[layer_name]
            
            # Get gas optimization for L2
            gas_opt = await self.gas_optimizers[layer_name].get_optimal_gas()
            
            # Build transaction
            tx_data = contract.functions.submitReport(
                report.report_hash,
                report.target_url,
                report.response_time,
                report.is_reachable,
                report.ssl_valid,
                report.status_code
            ).build_transaction({
                'from': account.address,
                'gas': gas_opt.estimated_gas,
                'gasPrice': gas_opt.max_fee,  # L2 uses legacy gas price
                'nonce': web3.eth.get_transaction_count(account.address),
                'value': gas_opt.total_cost,
                'chainId': self.config.L2_CONFIGS[layer_name]['chain_id']
            })
            
            # Sign and send transaction
            signed_tx = web3.eth.account.sign_transaction(tx_data, self.config.PRIVATE_KEY)
            tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            logger.info(f"L2 ({layer_name}) report submitted: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Failed to submit L2 ({layer_name}) report: {e}")
            raise

# ===== GAS OPTIMIZATION ENGINE =====

class EthereumGasOptimizer:
    """Ethereum gas optimization engine with EIP-1559 support"""
    
    def __init__(self, web3: Web3, gas_multiplier: float = 1.0):
        self.web3 = web3
        self.gas_multiplier = gas_multiplier
        self.gas_history = deque(maxlen=100)
        
    async def get_optimal_gas(self) -> GasOptimization:
        """Get optimal gas parameters"""
        try:
            # Get current gas prices
            latest_block = self.web3.eth.get_block('latest', full_transactions=True)
            base_fee = latest_block.baseFeePerGas
            
            # Calculate priority fee based on network activity
            priority_fee = await self._calculate_priority_fee(latest_block)
            
            # Estimate gas for report submission
            estimated_gas = 50000  # Base estimate for report submission
            
            # Calculate max fee
            max_fee = base_fee + priority_fee
            
            # Apply layer multiplier
            max_fee = int(max_fee * self.gas_multiplier)
            priority_fee = int(priority_fee * self.gas_multiplier)
            
            # Calculate total cost
            total_cost = estimated_gas * max_fee
            
            # Store in history
            self.gas_history.append({
                'timestamp': time.time(),
                'base_fee': base_fee,
                'priority_fee': priority_fee,
                'max_fee': max_fee
            })
            
            return GasOptimization(
                base_fee=base_fee,
                priority_fee=priority_fee,
                max_fee=max_fee,
                estimated_gas=estimated_gas,
                total_cost=total_cost,
                layer='ethereum' if self.gas_multiplier == 1.0 else 'l2'
            )
            
        except Exception as e:
            logger.error(f"Failed to get optimal gas: {e}")
            # Return fallback values
            return GasOptimization(
                base_fee=20 * 10**9,  # 20 gwei
                priority_fee=2 * 10**9,  # 2 gwei
                max_fee=22 * 10**9,   # 22 gwei
                estimated_gas=50000,
                total_cost=1100000000000,  # 0.0011 ETH
                layer='fallback'
            )
    
    async def _calculate_priority_fee(self, latest_block) -> int:
        """Calculate optimal priority fee based on network activity"""
        try:
            # Analyze recent transactions to determine priority fee
            priority_fees = []
            
            for tx in latest_block.transactions[:10]:  # Sample first 10 transactions
                if 'maxPriorityFeePerGas' in tx:
                    priority_fees.append(tx['maxPriorityFeePerGas'])
            
            if priority_fees:
                # Use 80th percentile of priority fees
                priority_fees.sort()
                return priority_fees[int(len(priority_fees) * 0.8)]
            else:
                # Fallback to 2 gwei
                return 2 * 10**9
                
        except Exception:
            return 2 * 10**9  # 2 gwei fallback

# ===== HIGH-PERFORMANCE MONITOR =====

class EthereumWebsiteMonitor:
    """Ethereum-optimized website monitoring with batch processing"""
    
    def __init__(self, config: EthereumConfig):
        self.config = config
        self.session = None
        self.ssl_context = ssl.create_default_context()
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.WORKER_PROCESSES
        )
        self.dns_resolver = dns.resolver.Resolver()
        self.dns_resolver.timeout = 2.0
        self.dns_resolver.lifetime = 2.0
        
        # Priority queues
        self.critical_queue = asyncio.Queue()
        self.high_queue = asyncio.Queue()
        self.regular_queue = asyncio.Queue()
        
    async def __aenter__(self):
        """Initialize async context"""
        connector = aiohttp.TCPConnector(
            limit=1000,
            limit_per_host=100,
            ttl_dns_cache=300,
            use_dns_cache=True,
            ssl=self.ssl_context,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=10.0,
            connect=2.0,
            sock_read=5.0
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'Mozilla/5.0 (compatible; EthereumMonitoring/1.0)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup async context"""
        if self.session:
            await self.session.close()
        self.executor.shutdown(wait=False)
    
    async def monitor_all_targets(self) -> List[EthereumMonitoringReport]:
        """Monitor all targets based on priority"""
        tasks = []
        
        # Critical targets (highest priority)
        for url in self.config.CRITICAL_TARGETS:
            task = asyncio.create_task(self._monitor_single(url, 'critical'))
            tasks.append(task)
        
        # High priority targets
        for url in self.config.HIGH_PRIORITY_TARGETS:
            task = asyncio.create_task(self._monitor_single(url, 'high'))
            tasks.append(task)
        
        # Regular targets
        for url in self.config.REGULAR_TARGETS:
            task = asyncio.create_task(self._monitor_single(url, 'regular'))
            tasks.append(task)
        
        # Wait for all monitoring tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        reports = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error monitoring target: {result}")
            else:
                reports.append(result)
        
        return reports
    
    async def _monitor_single(self, url: str, priority: str) -> EthereumMonitoringReport:
        """Monitor a single URL"""
        start_time = time.time()
        
        report_data = {
            'url': url,
            'timestamp': int(start_time * 1000),
            'response_time': 0,
            'status_code': 0,
            'is_reachable': False,
            'ssl_valid': False,
            'content_hash': '',
            'error': None
        }
        
        try:
            # Parse URL
            parsed = urlparse(url)
            hostname = parsed.hostname
            
            # DNS resolution
            try:
                dns_records = self.dns_resolver.resolve(hostname, 'A')
                dns_ips = [str(record) for record in dns_records]
                report_data['dns_ips'] = dns_ips
            except Exception as e:
                report_data['error'] = f"DNS resolution failed: {e}"
                return self._create_report(report_data, priority)
            
            # HTTP request
            request_start = time.time()
            async with self.session.get(url, allow_redirects=True) as response:
                request_time = (time.time() - request_start) * 1000
                report_data['response_time'] = int(request_time)
                report_data['status_code'] = response.status
                report_data['is_reachable'] = 200 <= response.status_code < 400
                
                # SSL verification
                if url.startswith('https://'):
                    try:
                        ssl_info = response.connection.transport.get_extra_info('ssl_object')
                        if ssl_info:
                            cert = ssl_info.getpeercert()
                            report_data['ssl_valid'] = True
                    except Exception:
                        report_data['ssl_valid'] = False
                
                # Content hash (first 1KB for performance)
                try:
                    content = await response.content.read(1024)
                    report_data['content_hash'] = hashlib.sha256(content).hexdigest()
                except Exception:
                    report_data['content_hash'] = hashlib.sha256(b'').hexdigest()
                
        except asyncio.TimeoutError:
            report_data['error'] = "Request timeout"
        except aiohttp.ClientError as e:
            report_data['error'] = f"HTTP client error: {e}"
        except Exception as e:
            report_data['error'] = f"Unexpected error: {e}"
        
        return self._create_report(report_data, priority)
    
    def _create_report(self, report_data: Dict, priority: str) -> EthereumMonitoringReport:
        """Create monitoring report from data"""
        # Generate report hash
        report_content = json.dumps(report_data, sort_keys=True)
        report_hash = hashlib.sha256(report_content.encode()).hexdigest()
        
        return EthereumMonitoringReport(
            report_hash=report_hash,
            node_id=config.NODE_ID if hasattr(config, 'NODE_ID') else f"node_{int(time.time())}",
            target_url=report_data['url'],
            response_time=report_data['response_time'],
            is_reachable=report_data['is_reachable'],
            ssl_valid=report_data['ssl_valid'],
            status_code=report_data['status_code'],
            timestamp=report_data['timestamp'],
            epoch_id=int(time.time() // 600),  # 10-minute epochs
            shard_id=0,  # Will be assigned by contract
            priority=priority
        )

# ===== MAIN ETHEREUM NODE =====

class EthereumMonitoringNode:
    """Ethereum-native monitoring node with L1/L2 capabilities"""
    
    def __init__(self, config: EthereumConfig):
        self.config = config
        self.node_id = config.NODE_ID if hasattr(config, 'NODE_ID') else f"eth_node_{int(time.time())}"
        self.layer_manager = None
        self.monitor = None
        self.redis_client = None
        self.app = FastAPI(
            title="Ethereum Monitoring Node",
            description="Ethereum-native high-performance monitoring node",
            version="2.0.0"
        )
        self.setup_routes()
        
    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize Redis
            self.redis_client = await aioredis.from_url("redis://localhost:6379")
            
            # Initialize layer manager
            self.layer_manager = EthereumLayerManager(self.config)
            await self.layer_manager.initialize()
            
            # Initialize monitor
            self.monitor = await EthereumWebsiteMonitor(self.config).__aenter__()
            
            logger.info(f"Ethereum monitoring node {self.node_id} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ethereum node: {e}")
            raise
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "node_id": self.node_id,
                "type": "ethereum",
                "timestamp": int(time.time() * 1000),
                "version": "2.0.0"
            }
        
        @self.app.post("/monitor/submit")
        async def submit_report(report: EthereumMonitoringReport):
            """Submit monitoring report to optimal layer"""
            try:
                tx_hash = await self.layer_manager.submit_report(report)
                return {
                    "status": "success",
                    "tx_hash": tx_hash,
                    "layer": self.layer_manager.get_optimal_layer(report.priority),
                    "timestamp": int(time.time() * 1000)
                }
            except Exception as e:
                logger.error(f"Error submitting report: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/monitor/batch")
        async def submit_batch(reports: List[EthereumMonitoringReport]):
            """Submit batch of reports"""
            try:
                results = []
                for report in reports:
                    tx_hash = await self.layer_manager.submit_report(report)
                    results.append({
                        "report_hash": report.report_hash,
                        "tx_hash": tx_hash,
                        "layer": self.layer_manager.get_optimal_layer(report.priority)
                    })
                
                return {
                    "status": "success",
                    "submitted": len(results),
                    "results": results,
                    "timestamp": int(time.time() * 1000)
                }
            except Exception as e:
                logger.error(f"Error submitting batch: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get node metrics"""
            try:
                # Get system metrics from Redis
                cached_metrics = await self.redis_client.get("node_metrics")
                
                if cached_metrics:
                    return json.loads(cached_metrics)
                else:
                    return {
                        "node_id": self.node_id,
                        "type": "ethereum",
                        "timestamp": int(time.time() * 1000),
                        "status": "active"
                    }
            except Exception as e:
                logger.error(f"Error getting metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/monitor/trigger")
        async def trigger_monitoring_round(background_tasks: BackgroundTasks):
            """Trigger a monitoring round"""
            background_tasks.add_task(self._monitoring_round)
            return {
                "status": "triggered",
                "targets_count": len(self.config.CRITICAL_TARGETS + 
                                   self.config.HIGH_PRIORITY_TARGETS + 
                                   self.config.REGULAR_TARGETS),
                "timestamp": int(time.time() * 1000)
            }
    
    async def _monitoring_round(self):
        """Execute a monitoring round"""
        try:
            logger.info(f"Starting Ethereum monitoring round")
            
            # Monitor all targets
            reports = await self.monitor.monitor_all_targets()
            
            # Submit reports to optimal layers
            submission_tasks = []
            for report in reports:
                task = asyncio.create_task(self.layer_manager.submit_report(report))
                submission_tasks.append(task)
            
            # Wait for all submissions
            results = await asyncio.gather(*submission_tasks, return_exceptions=True)
            
            # Count successful submissions
            successful = sum(1 for r in results if not isinstance(r, Exception))
            
            logger.info(f"Monitoring round completed: {successful}/{len(reports)} reports submitted")
            
            # Cache metrics
            metrics = {
                "node_id": self.node_id,
                "type": "ethereum",
                "timestamp": int(time.time() * 1000),
                "reports_monitored": len(reports),
                "reports_submitted": successful,
                "success_rate": (successful / len(reports)) * 100 if reports else 0
            }
            
            await self.redis_client.setex(
                "node_metrics",
                300,  # 5 minutes TTL
                json.dumps(metrics)
            )
            
        except Exception as e:
            logger.error(f"Error in monitoring round: {e}")
    
    async def start_monitoring_loop(self):
        """Start continuous monitoring loop"""
        while True:
            try:
                await self._monitoring_round()
                await asyncio.sleep(60)  # Monitor every minute
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)  # Wait before retry
    
    async def start(self):
        """Start the Ethereum monitoring node"""
        await self.initialize()
        
        # Start background monitoring
        monitoring_task = asyncio.create_task(self.start_monitoring_loop())
        
        logger.info(f"Ethereum monitoring node {self.node_id} started successfully")
        
        return monitoring_task

# ===== MAIN ENTRY POINT =====

async def main():
    """Main entry point for Ethereum monitoring node"""
    import os
    
    # Load configuration from environment
    config.NODE_ID = os.getenv("NODE_ID", f"eth_node_{int(time.time())}")
    config.PRIVATE_KEY = os.getenv("PRIVATE_KEY", "")
    config.L1_CONTRACT_ADDRESS = os.getenv("L1_CONTRACT_ADDRESS", "")
    config.MON_TOKEN_ADDRESS = os.getenv("MON_TOKEN_ADDRESS", "")
    
    # Create and start node
    node = EthereumMonitoringNode(config)
    monitoring_task = await node.start()
    
    try:
        await monitoring_task
    except KeyboardInterrupt:
        logger.info("Shutting down Ethereum node...")
        await node.monitor.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())
