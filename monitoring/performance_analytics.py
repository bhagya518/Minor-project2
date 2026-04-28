"""
Production Performance Analytics & Monitoring System
Real-time metrics collection, analysis, and alerting for high-throughput monitoring
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pandas as pd
import numpy as np
import redis
import aioredis
from fastapi import FastAPI, HTTPException
import prometheus_client as prom
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import psutil
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== METRICS DEFINITIONS =====

# Prometheus metrics
REQUEST_COUNT = Counter('monitoring_requests_total', 'Total monitoring requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('monitoring_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')
SYSTEM_TPS = Gauge('system_tps', 'Current transactions per second')
SYSTEM_LATENCY = Gauge('system_latency_p95', '95th percentile latency (ms)')
NODE_REPUTATION = Gauge('node_reputation', 'Node reputation scores', ['node_id'])
SHARD_TPS = Gauge('shard_tps', 'Transactions per second by shard', ['shard_id'])
VALIDATOR_STAKE = Gauge('validator_stake', 'Validator stake amounts', ['validator_address'])

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: int
    tps: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    success_rate: float
    error_rate: float
    active_nodes: int
    active_validators: int
    total_staked: int
    gas_used: int
    block_time: float
    shard_metrics: Dict[int, Dict]
    node_metrics: Dict[str, Dict]

@dataclass
class AlertConfig:
    """Alert configuration"""
    metric_name: str
    threshold: float
    operator: str  # 'gt', 'lt', 'eq'
    severity: str  # 'critical', 'warning', 'info'
    cooldown: int  # seconds
    enabled: bool = True

class PerformanceAnalyzer:
    """High-performance metrics analyzer"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis_client = redis_client
        self.metrics_buffer = deque(maxlen=10000)
        self.alert_configs = self._init_alert_configs()
        self.alert_history = defaultdict(list)
        self.last_analysis = time.time()
        
    def _init_alert_configs(self) -> Dict[str, AlertConfig]:
        """Initialize default alert configurations"""
        return {
            'low_tps': AlertConfig('tps', 5000, 'lt', 'critical', 300),
            'high_latency': AlertConfig('latency_p95', 200, 'gt', 'critical', 300),
            'low_success_rate': AlertConfig('success_rate', 99.0, 'lt', 'warning', 600),
            'few_nodes': AlertConfig('active_nodes', 900, 'lt', 'warning', 600),
            'high_error_rate': AlertConfig('error_rate', 1.0, 'gt', 'critical', 300),
            'low_stake': AlertConfig('total_staked', 1000000, 'lt', 'warning', 1800)
        }
    
    async def collect_metrics(self, blockchain_client, node_registry) -> PerformanceMetrics:
        """Collect comprehensive performance metrics"""
        try:
            current_time = int(time.time() * 1000)
            
            # Blockchain metrics
            block_number = await blockchain_client.web3.eth.block_number
            gas_price = await blockchain_client.web3.eth.gas_price
            
            # System metrics from blockchain
            system_metrics = await blockchain_client.contract.functions.getSystemMetrics().call()
            tps, total_reports, active_validators, total_staked = system_metrics
            
            # Calculate latency metrics from recent blocks
            latency_metrics = await self._calculate_latency_metrics(blockchain_client)
            
            # Calculate success/error rates
            success_rate, error_rate = await self._calculate_success_rates(blockchain_client)
            
            # Shard metrics
            shard_metrics = await self._collect_shard_metrics(blockchain_client)
            
            # Node metrics
            node_metrics = await self._collect_node_metrics(blockchain_client, node_registry)
            
            # System metrics
            system_load = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            
            metrics = PerformanceMetrics(
                timestamp=current_time,
                tps=tps,
                latency_p50=latency_metrics['p50'],
                latency_p95=latency_metrics['p95'],
                latency_p99=latency_metrics['p99'],
                success_rate=success_rate,
                error_rate=error_rate,
                active_nodes=len(node_metrics),
                active_validators=active_validators,
                total_staked=total_staked,
                gas_used=gas_price,
                block_time=latency_metrics['block_time'],
                shard_metrics=shard_metrics,
                node_metrics=node_metrics
            )
            
            # Store in buffer
            self.metrics_buffer.append(metrics)
            
            # Update Prometheus metrics
            self._update_prometheus_metrics(metrics)
            
            # Store in Redis for historical analysis
            await self._store_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            raise
    
    async def _calculate_latency_metrics(self, blockchain_client) -> Dict[str, float]:
        """Calculate latency metrics from recent blocks"""
        try:
            current_block = await blockchain_client.web3.eth.block_number
            block_times = []
            
            # Get last 100 blocks for latency calculation
            for i in range(max(0, current_block - 100), current_block):
                block = await blockchain_client.web3.eth.get_block(i)
                if i > 0:
                    prev_block = await blockchain_client.web3.eth.get_block(i - 1)
                    block_time = block.timestamp - prev_block.timestamp
                    block_times.append(block_time * 1000)  # Convert to milliseconds
            
            if not block_times:
                return {'p50': 0, 'p95': 0, 'p99': 0, 'block_time': 0}
            
            # Calculate percentiles
            block_times.sort()
            n = len(block_times)
            
            return {
                'p50': block_times[n // 2],
                'p95': block_times[int(n * 0.95)],
                'p99': block_times[int(n * 0.99)],
                'block_time': np.mean(block_times)
            }
            
        except Exception as e:
            logger.error(f"Error calculating latency metrics: {e}")
            return {'p50': 0, 'p95': 0, 'p99': 0, 'block_time': 0}
    
    async def _calculate_success_rates(self, blockchain_client) -> Tuple[float, float]:
        """Calculate success and error rates"""
        try:
            # Get recent batch statistics
            current_batch = await blockchain_client.contract.functions.totalBatches().call()
            
            if current_batch == 0:
                return 100.0, 0.0
            
            # Sample last 10 batches for rate calculation
            success_count = 0
            total_count = 0
            
            for i in range(max(0, current_batch - 10), current_batch):
                batch = await blockchain_client.contract.functions.batches(i).call()
                if batch[3]:  # isValidated
                    success_count += len(batch[0])  # reports length
                total_count += len(batch[0])
            
            success_rate = (success_count / total_count * 100) if total_count > 0 else 100.0
            error_rate = 100.0 - success_rate
            
            return success_rate, error_rate
            
        except Exception as e:
            logger.error(f"Error calculating success rates: {e}")
            return 100.0, 0.0
    
    async def _collect_shard_metrics(self, blockchain_client) -> Dict[int, Dict]:
        """Collect metrics for each shard"""
        shard_metrics = {}
        
        for shard_id in range(3):  # 3 shards
            try:
                node_count, report_count, validator_count = await blockchain_client.contract.functions.getShardStats(shard_id).call()
                
                shard_metrics[shard_id] = {
                    'node_count': node_count,
                    'report_count': report_count,
                    'validator_count': validator_count,
                    'tps': report_count / 300  # Rough TPS calculation
                }
                
                # Update shard-specific Prometheus metrics
                SHARD_TPS.labels(shard_id=str(shard_id)).set(shard_metrics[shard_id]['tps'])
                
            except Exception as e:
                logger.error(f"Error collecting shard {shard_id} metrics: {e}")
                shard_metrics[shard_id] = {'node_count': 0, 'report_count': 0, 'validator_count': 0, 'tps': 0}
        
        return shard_metrics
    
    async def _collect_node_metrics(self, blockchain_client, node_registry) -> Dict[str, Dict]:
        """Collect metrics for individual nodes"""
        node_metrics = {}
        
        for node_id in node_registry.get_registered_nodes():
            try:
                reputation, monitoring_trust, ml_score, stake, is_validator = await blockchain_client.contract.functions.getNodeReputation(node_id).call()
                
                node_metrics[node_id] = {
                    'reputation': reputation / 1000.0,  # Convert to 0-1 scale
                    'monitoring_trust': monitoring_trust / 1000.0,
                    'ml_score': ml_score / 1000.0,
                    'stake': stake,
                    'is_validator': is_validator
                }
                
                # Update node-specific Prometheus metrics
                NODE_REPUTATION.labels(node_id=node_id).set(reputation / 1000.0)
                
                if is_validator:
                    validator_address = node_registry.get_node_address(node_id)
                    VALIDATOR_STAKE.labels(validator_address=validator_address).set(stake)
                
            except Exception as e:
                logger.error(f"Error collecting node {node_id} metrics: {e}")
                node_metrics[node_id] = {
                    'reputation': 0, 'monitoring_trust': 0, 'ml_score': 0,
                    'stake': 0, 'is_validator': False
                }
        
        return node_metrics
    
    def _update_prometheus_metrics(self, metrics: PerformanceMetrics):
        """Update Prometheus metrics"""
        SYSTEM_TPS.set(metrics.tps)
        SYSTEM_LATENCY.set(metrics.latency_p95)
        ACTIVE_CONNECTIONS.set(metrics.active_nodes)
    
    async def _store_metrics(self, metrics: PerformanceMetrics):
        """Store metrics in Redis for historical analysis"""
        try:
            # Store latest metrics
            await self.redis_client.setex(
                "metrics:latest",
                300,  # 5 minutes TTL
                json.dumps(asdict(metrics), default=str)
            )
            
            # Store in time series
            timestamp_key = f"metrics:ts:{metrics.timestamp}"
            await self.redis_client.setex(
                timestamp_key,
                3600,  # 1 hour TTL
                json.dumps(asdict(metrics), default=str)
            )
            
            # Add to time series index
            await self.redis_client.zadd("metrics:timeline", {timestamp_key: metrics.timestamp})
            
            # Keep only last 24 hours
            cutoff = metrics.timestamp - (24 * 60 * 60 * 1000)
            await self.redis_client.zremrangebyscore("metrics:timeline", 0, cutoff)
            
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")
    
    async def check_alerts(self, metrics: PerformanceMetrics) -> List[Dict]:
        """Check alert conditions and return triggered alerts"""
        alerts = []
        current_time = time.time()
        
        for alert_name, config in self.alert_configs.items():
            if not config.enabled:
                continue
            
            # Check cooldown
            last_alert = self.alert_history[alert_name][-1] if self.alert_history[alert_name] else 0
            if current_time - last_alert < config.cooldown:
                continue
            
            # Get metric value
            metric_value = getattr(metrics, config.metric_name, 0)
            
            # Check condition
            triggered = False
            if config.operator == 'gt' and metric_value > config.threshold:
                triggered = True
            elif config.operator == 'lt' and metric_value < config.threshold:
                triggered = True
            elif config.operator == 'eq' and metric_value == config.threshold:
                triggered = True
            
            if triggered:
                alert = {
                    'alert_name': alert_name,
                    'metric': config.metric_name,
                    'value': metric_value,
                    'threshold': config.threshold,
                    'severity': config.severity,
                    'timestamp': current_time,
                    'message': self._generate_alert_message(alert_name, metric_value, config.threshold)
                }
                
                alerts.append(alert)
                self.alert_history[alert_name].append(current_time)
                
                # Log alert
                logger.warning(f"ALERT: {alert['message']}")
                
                # Send notification (implement webhook/slack/email)
                await self._send_alert_notification(alert)
        
        return alerts
    
    def _generate_alert_message(self, alert_name: str, value: float, threshold: float) -> str:
        """Generate alert message"""
        messages = {
            'low_tps': f"Low TPS detected: {value:.2f} < {threshold}",
            'high_latency': f"High latency detected: {value:.2f}ms > {threshold}ms",
            'low_success_rate': f"Low success rate: {value:.2f}% < {threshold}%",
            'few_nodes': f"Few active nodes: {value} < {threshold}",
            'high_error_rate': f"High error rate: {value:.2f}% > {threshold}%",
            'low_stake': f"Low total stake: {value} < {threshold}"
        }
        
        return messages.get(alert_name, f"Alert {alert_name}: {value} vs {threshold}")
    
    async def _send_alert_notification(self, alert: Dict):
        """Send alert notification (implement webhook, Slack, email, etc.)"""
        try:
            # For now, just log the alert
            # In production, implement webhook calls, Slack notifications, etc.
            logger.info(f"Alert notification sent: {alert}")
            
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")
    
    async def get_historical_metrics(self, hours: int = 24) -> List[PerformanceMetrics]:
        """Get historical metrics for analysis"""
        try:
            cutoff_time = int(time.time() * 1000) - (hours * 60 * 60 * 1000)
            
            # Get metric keys from timeline
            metric_keys = await self.redis_client.zrangebyscore(
                "metrics:timeline",
                cutoff_time,
                int(time.time() * 1000)
            )
            
            metrics = []
            for key in metric_keys:
                metric_data = await self.redis_client.get(key)
                if metric_data:
                    metric_dict = json.loads(metric_data)
                    metrics.append(PerformanceMetrics(**metric_dict))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting historical metrics: {e}")
            return []
    
    async def generate_performance_report(self, hours: int = 24) -> Dict:
        """Generate comprehensive performance report"""
        try:
            historical_metrics = await self.get_historical_metrics(hours)
            
            if not historical_metrics:
                return {"error": "No historical data available"}
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame([asdict(m) for m in historical_metrics])
            
            # Calculate statistics
            report = {
                "period_hours": hours,
                "total_data_points": len(historical_metrics),
                "tps": {
                    "avg": df['tps'].mean(),
                    "min": df['tps'].min(),
                    "max": df['tps'].max(),
                    "p95": df['tps'].quantile(0.95)
                },
                "latency": {
                    "p50_avg": df['latency_p50'].mean(),
                    "p95_avg": df['latency_p95'].mean(),
                    "p99_avg": df['latency_p99'].mean(),
                    "p95_max": df['latency_p95'].max()
                },
                "availability": {
                    "avg_success_rate": df['success_rate'].mean(),
                    "min_success_rate": df['success_rate'].min(),
                    "avg_error_rate": df['error_rate'].mean()
                },
                "nodes": {
                    "avg_active_nodes": df['active_nodes'].mean(),
                    "min_active_nodes": df['active_nodes'].min(),
                    "avg_active_validators": df['active_validators'].mean()
                },
                "economics": {
                    "avg_total_staked": df['total_staked'].mean(),
                    "avg_gas_used": df['gas_used'].mean()
                },
                "alerts_summary": {
                    "total_alerts": sum(len(alerts) for alerts in self.alert_history.values()),
                    "critical_alerts": len([a for a in self.alert_history.get('low_tps', []) if time.time() - a < 3600]),
                    "warning_alerts": len([a for a in self.alert_history.get('low_success_rate', []) if time.time() - a < 3600])
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"error": str(e)}

class MonitoringAPI:
    """FastAPI endpoints for performance monitoring"""
    
    def __init__(self, analyzer: PerformanceAnalyzer):
        self.analyzer = analyzer
        self.app = FastAPI(title="Performance Monitoring API", version="1.0.0")
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/metrics")
        async def get_prometheus_metrics():
            """Prometheus metrics endpoint"""
            return generate_latest()
        
        @self.app.get("/performance/current")
        async def get_current_metrics():
            """Get current performance metrics"""
            try:
                # This would be injected with actual blockchain client and node registry
                # For now, return latest from Redis
                latest_data = await self.analyzer.redis_client.get("metrics:latest")
                if latest_data:
                    return json.loads(latest_data)
                else:
                    return {"error": "No current metrics available"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/performance/history")
        async def get_historical_metrics(hours: int = 24):
            """Get historical performance metrics"""
            try:
                metrics = await self.analyzer.get_historical_metrics(hours)
                return [asdict(m) for m in metrics]
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/performance/report")
        async def get_performance_report(hours: int = 24):
            """Get performance report"""
            try:
                report = await self.analyzer.generate_performance_report(hours)
                return report
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/alerts")
        async def get_active_alerts():
            """Get active alerts"""
            try:
                current_metrics = await self.analyzer.redis_client.get("metrics:latest")
                if current_metrics:
                    metrics = PerformanceMetrics(**json.loads(current_metrics))
                    alerts = await self.analyzer.check_alerts(metrics)
                    return alerts
                else:
                    return []
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": int(time.time() * 1000),
                "version": "1.0.0"
            }

# ===== MAIN MONITORING SERVICE =====

class PerformanceMonitoringService:
    """Main performance monitoring service"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = None
        self.analyzer = None
        self.api = None
        self.running = False
        
    async def initialize(self, blockchain_client, node_registry):
        """Initialize the monitoring service"""
        try:
            # Initialize Redis
            self.redis_client = await aioredis.from_url(redis_url)
            
            # Initialize analyzer
            self.analyzer = PerformanceAnalyzer(self.redis_client)
            
            # Initialize API
            self.api = MonitoringAPI(self.analyzer)
            
            logger.info("Performance monitoring service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring service: {e}")
            raise
    
    async def start_monitoring_loop(self, blockchain_client, node_registry):
        """Start continuous monitoring loop"""
        self.running = True
        
        while self.running:
            try:
                # Collect metrics
                metrics = await self.analyzer.collect_metrics(blockchain_client, node_registry)
                
                # Check alerts
                alerts = await self.analyzer.check_alerts(metrics)
                
                # Log summary
                logger.info(f"Metrics - TPS: {metrics.tps:.2f}, Latency P95: {metrics.latency_p95:.2f}ms, "
                           f"Success Rate: {metrics.success_rate:.2f}%, Nodes: {metrics.active_nodes}")
                
                # Wait before next collection
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)  # Wait before retry
    
    def stop(self):
        """Stop the monitoring service"""
        self.running = False
        logger.info("Performance monitoring service stopped")

# ===== USAGE EXAMPLE =====

async def main():
    """Example usage of performance monitoring system"""
    # Initialize monitoring service
    monitoring_service = PerformanceMonitoringService()
    
    # Initialize with blockchain client and node registry
    # await monitoring_service.initialize(blockchain_client, node_registry)
    
    # Start monitoring loop
    # await monitoring_service.start_monitoring_loop(blockchain_client, node_registry)
    
    print("Performance monitoring system ready")

if __name__ == "__main__":
    asyncio.run(main())
