"""
Peer Communication Module
Handles P2P communication between monitoring nodes
Phase 2: Added support for broadcasting signed MonitoringReport objects
"""

import asyncio
import aiohttp
from aiohttp import web
import json
import logging
import time
from typing import Dict, List, Optional, Set, Callable, Tuple
from datetime import datetime
import socket
from dataclasses import dataclass, asdict
import uuid

# Import signed report system (Phase 2)
try:
    from monitoring_report import MonitoringReport
    REPORT_AVAILABLE = True
except ImportError:
    REPORT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Production limits
MAX_PEERS = 50

@dataclass
class PeerNode:
    """Information about a peer node"""
    node_id: str
    host: str
    port: int
    last_seen: datetime
    trust_score: float = 0.5
    is_active: bool = True

class PeerClient:
    """Client for P2P communication between monitoring nodes"""
    
    def __init__(self, node_id: str, host: str = "localhost", port: int = 8000):
        """
        Initialize peer client
        
        Args:
            node_id: Unique identifier for this node
            host: Host address for this node
            port: Port for this node (API port)
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.p2p_port = port + 1000  # Separate P2P port to avoid conflicts
        self.peers: Dict[str, PeerNode] = {}
        self.session = None
        self.server = None
        self.message_handlers = {}
        self.public_key_hex = None  # Will be set when keys are generated
        
        # Message types
        self.MESSAGE_TYPES = {
            'HEARTBEAT': 'heartbeat',
            'MONITORING_RESULT': 'monitoring_result',
            'TRUST_UPDATE': 'trust_update',
            'PEER_DISCOVERY': 'peer_discovery',
            'CONTENT_HASH': 'content_hash',
            'ML_PREDICTION': 'ml_prediction'
        }
        
        logger.info(f"Peer client initialized for node {node_id} on {host}:{port}")
    
    async def start_server(self):
        """Start the P2P server"""
        try:
            # Create aiohttp server
            app = web.Application()
            
            # Add routes
            app.router.add_post('/peer/message', self.handle_message)
            app.router.add_get('/peer/info', self.handle_info_request)
            app.router.add_post('/peer/discovery', self.handle_peer_discovery)
            
            # Start server
            runner = web.AppRunner(app)
            await runner.setup()
            
            site = web.TCPSite(runner, self.host, self.p2p_port)
            await site.start()
            
            self.server = runner
            logger.info(f"P2P server started on {self.host}:{self.p2p_port}")
            
        except Exception as e:
            logger.error(f"Failed to start P2P server: {e}")
            raise
    
    async def close_session(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def stop_server(self):
        """Stop the P2P server"""
        if self.server:
            await self.server.cleanup()
        await self.close_session()
        logger.info("P2P server stopped")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        await self.stop_server()
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """
        Register a handler for specific message types
        
        Args:
            message_type: Type of message
            handler: Async function to handle the message
        """
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for message type: {message_type}")
    
    async def add_peer(self, node_id: str, host: str, port: int):
        """
        Add a peer node
        
        Args:
            node_id: Peer node ID
            host: Peer host address
            port: Peer port
        """
        # Check peer limit
        if len(self.peers) >= MAX_PEERS:
            logger.warning("Peer limit reached, cannot add more peers")
            return
        
        peer = PeerNode(
            node_id=node_id,
            host=host,
            port=port,
            last_seen=datetime.now()
        )
        
        self.peers[node_id] = peer
        logger.info(f"Added peer: {node_id} at {host}:{port}")
    
    async def remove_peer(self, node_id: str):
        """
        Remove a peer node
        
        Args:
            node_id: Peer node ID to remove
        """
        if node_id in self.peers:
            del self.peers[node_id]
            logger.info(f"Removed peer: {node_id}")
    
    async def send_message(self, target_node_id: str, message_type: str, data: Dict) -> bool:
        """
        Send a message to a specific peer
        
        Args:
            target_node_id: Target node ID
            message_type: Type of message
            data: Message data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if target_node_id not in self.peers:
                logger.error(f"Peer {target_node_id} not found")
                return False
            
            peer = self.peers[target_node_id]
            if not peer.is_active:
                logger.warning(f"Peer {target_node_id} is not active")
                return False
            
            # Prepare message
            message = {
                'id': str(uuid.uuid4()),
                'sender_id': self.node_id,
                'type': message_type,
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            
            # Send message
            url = f"http://{peer.host}:{peer.port}/peer/message"
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.post(url, json=message, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    peer.last_seen = datetime.now()
                    logger.debug(f"Message sent to {target_node_id}: {message_type}")
                    return True
                else:
                    logger.error(f"Failed to send message to {target_node_id}: HTTP {response.status}")
                    return False
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout sending message to {target_node_id}")
            return False
        except Exception as e:
            logger.error(f"Error sending message to {target_node_id}: {e}")
            return False
    
    async def broadcast_message(self, message_type: str, data: Dict) -> Dict[str, bool]:
        """
        Broadcast a message to all active peers
        
        Args:
            message_type: Type of message
            data: Message data
            
        Returns:
            Dictionary mapping node_id to success status
        """
        results = {}

        tasks = []
        node_ids = []

        for node_id, peer in self.peers.items():
            if peer.is_active:
                tasks.append(self.send_message(node_id, message_type, data))
                node_ids.append(node_id)

        if tasks:
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for node_id, response in zip(node_ids, responses):
                if isinstance(response, Exception):
                    logger.error(f"Broadcast to {node_id} failed: {response}")
                    results[node_id] = False
                else:
                    results[node_id] = response

        logger.info(f"Broadcast completed: {sum(results.values())}/{len(results)} successful")
        return results
    
    # Phase 2: Broadcast signed monitoring report to all peers
    async def broadcast_report(self, report: 'MonitoringReport', peer_urls: List[str]) -> Dict[str, bool]:
        """
        Broadcast a signed MonitoringReport to all peer nodes
        
        Args:
            report: Signed MonitoringReport to broadcast
            peer_urls: List of peer URLs (e.g., ["http://localhost:8001", "http://localhost:8002"])
            
        Returns:
            Dictionary mapping peer_url to success status
        """
        logger.info(f"broadcast_report called with {len(peer_urls)} peers: {peer_urls}")
        
        if not REPORT_AVAILABLE:
            logger.warning("MonitoringReport not available, cannot broadcast")
            return {}
        
        if not peer_urls:
            logger.warning("No peer URLs provided - nothing to broadcast")
            return {}
        
        # Convert report to dictionary for JSON serialization
        payload = asdict(report)
        
        results = {}
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0)) as session:
            tasks = []
            
            for peer_url in peer_urls:
                try:
                    task = session.post(
                        f"{peer_url}/report",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    tasks.append((peer_url, task))
                except Exception as e:
                    logger.warning(f"Failed to create broadcast task for {peer_url}: {e}")
                    results[peer_url] = False
            
            # Execute all broadcasts concurrently
            if tasks:
                responses = await asyncio.gather(
                    *[task for _, task in tasks],
                    return_exceptions=True
                )
                
                for i, (peer_url, _) in enumerate(tasks):
                    response = responses[i]
                    
                    if isinstance(response, Exception):
                        results[peer_url] = False
                    else:
                        try:
                            if response.status == 200:
                                results[peer_url] = True
                            else:
                                results[peer_url] = False
                        finally:
                            await response.release()
        
        success_count = sum(results.values())
        logger.info(f"Report broadcast completed: {success_count}/{len(peer_urls)} peers received report for {report.url} (epoch {report.epoch_id})")
        
        return results
    
    async def handle_message(self, request):
        """Handle incoming messages from peers"""
        try:
            message = await request.json()
            
            # Validate message
            required_fields = ['id', 'sender_id', 'type', 'timestamp', 'data']
            for field in required_fields:
                if field not in message:
                    return web.json_response(
                        {'error': f'Missing required field: {field}'},
                        status=400
                    )
            
            # Update peer info
            sender_id = message['sender_id']
            if sender_id in self.peers:
                self.peers[sender_id].last_seen = datetime.now()
            
            # Handle message based on type
            message_type = message['type']
            
            if message_type == self.MESSAGE_TYPES['HEARTBEAT']:
                await self._handle_heartbeat(message)
            elif message_type == self.MESSAGE_TYPES['MONITORING_RESULT']:
                await self._handle_monitoring_result(message)
            elif message_type == self.MESSAGE_TYPES['TRUST_UPDATE']:
                await self._handle_trust_update(message)
            elif message_type == self.MESSAGE_TYPES['CONTENT_HASH']:
                await self._handle_content_hash(message)
            elif message_type == self.MESSAGE_TYPES['ML_PREDICTION']:
                await self._handle_ml_prediction(message)
            else:
                # Use custom handler if registered
                if message_type in self.message_handlers:
                    await self.message_handlers[message_type](message)
                else:
                    logger.warning(f"Unknown message type: {message_type}")
            
            return web.json_response({'status': 'received'})
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return web.json_response(
                {'error': str(e)},
                status=500
            )
    
    async def handle_info_request(self, request):
        """Handle peer info requests"""
        try:
            info = {
                'node_id': self.node_id,
                'host': self.host,
                'port': self.port,
                'timestamp': datetime.now().isoformat(),
                'active_peers': len([p for p in self.peers.values() if p.is_active])
            }
            
            return web.json_response(info)
            
        except Exception as e:
            logger.error(f"Error handling info request: {e}")
            return web.json_response(
                {'error': str(e)},
                status=500
            )
    
    async def handle_peer_discovery(self, request):
        """Handle peer discovery requests"""
        try:
            data = await request.json()
            
            # Add the requesting peer if not already known
            peer_info = data.get('peer_info')
            if peer_info:
                await self.add_peer(
                    peer_info['node_id'],
                    peer_info['host'],
                    peer_info['port']
                )
            
            # Return list of known peers
            peers_list = []
            for peer in self.peers.values():
                if peer.is_active and peer.node_id != data.get('requester_id'):
                    peers_list.append({
                        'node_id': peer.node_id,
                        'host': peer.host,
                        'port': peer.port,
                        'trust_score': peer.trust_score
                    })
            
            return web.json_response({'peers': peers_list})
            
        except Exception as e:
            logger.error(f"Error handling peer discovery: {e}")
            return web.json_response(
                {'error': str(e)},
                status=500
            )
    
    async def _handle_heartbeat(self, message: Dict):
        """Handle heartbeat messages"""
        sender_id = message['sender_id']
        logger.debug(f"Received heartbeat from {sender_id}")
    
    async def _handle_monitoring_result(self, message: Dict):
        """Handle monitoring result messages"""
        sender_id = message['sender_id']
        data = message['data']
        
        logger.debug(f"Received monitoring result from {sender_id}")
        
        # Store or process the monitoring result
        # This would typically be handled by the main application
        if 'monitoring_result' in self.message_handlers:
            await self.message_handlers['monitoring_result'](message)
    
    async def _handle_trust_update(self, message: Dict):
        """Handle trust update messages"""
        sender_id = message['sender_id']
        data = message['data']
        
        logger.debug(f"Received trust update from {sender_id}")
        
        # Update peer trust score
        if sender_id in self.peers:
            self.peers[sender_id].trust_score = data.get('trust_score', 0.5)
    
    async def _handle_content_hash(self, message: Dict):
        """Handle content hash messages"""
        sender_id = message['sender_id']
        data = message['data']
        
        logger.debug(f"Received content hash from {sender_id}")
        
        # Process content hash for consistency checking
        if 'content_hash' in self.message_handlers:
            await self.message_handlers['content_hash'](message)
    
    async def _handle_ml_prediction(self, message: Dict):
        """Handle ML prediction messages"""
        sender_id = message['sender_id']
        data = message['data']
        
        logger.debug(f"Received ML prediction from {sender_id}")
        
        # Process ML prediction
        if 'ml_prediction' in self.message_handlers:
            await self.message_handlers['ml_prediction'](message)
    
    async def send_heartbeat(self):
        """Send heartbeat to all peers"""
        await self.broadcast_message(
            self.MESSAGE_TYPES['HEARTBEAT'],
            {'status': 'active'}
        )
    
    async def send_monitoring_result(self, result: Dict):
        """Send monitoring result to peers"""
        await self.broadcast_message(
            self.MESSAGE_TYPES['MONITORING_RESULT'],
            result
        )
    
    async def send_trust_update(self, trust_score: float):
        """Send trust score update to peers"""
        await self.broadcast_message(
            self.MESSAGE_TYPES['TRUST_UPDATE'],
            {'trust_score': trust_score}
        )
    
    async def send_content_hash(self, url: str, content_hash: str):
        """Send content hash to peers"""
        await self.broadcast_message(
            self.MESSAGE_TYPES['CONTENT_HASH'],
            {
                'url': url,
                'content_hash': content_hash,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    async def send_ml_prediction(self, prediction: Dict):
        """Send ML prediction to peers"""
        await self.broadcast_message(
            self.MESSAGE_TYPES['ML_PREDICTION'],
            prediction
        )
    
    async def discover_peers(self, seed_nodes: List[Tuple[str, str, int]]):
        """
        Discover peers from seed nodes
        
        Args:
            seed_nodes: List of (node_id, host, port) tuples
        """
        for node_id, host, port in seed_nodes:
            try:
                if node_id == self.node_id:
                    continue  # Skip self
                
                # Add seed node
                await self.add_peer(node_id, host, port)
                
                # Request peer list
                url = f"http://{host}:{port}/peer/discovery"
                
                if not self.session:
                    self.session = aiohttp.ClientSession()
                
                data = {
                    'requester_id': self.node_id,
                    'peer_info': {
                        'node_id': self.node_id,
                        'host': self.host,
                        'port': self.port
                    }
                }
                
                async with self.session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        result = await response.json()
                        peers = result.get('peers', [])
                        
                        for peer in peers:
                            await self.add_peer(peer['node_id'], peer['host'], peer['port'])
                        
                        logger.info(f"Discovered {len(peers)} peers from {node_id}")
                    
            except Exception as e:
                logger.error(f"Failed to discover peers from {node_id}: {e}")
    
    async def check_peer_health(self):
        """Check health of all peers and update status"""
        for node_id, peer in list(self.peers.items()):
            try:
                url = f"http://{peer.host}:{peer.port}/peer/info"
                
                if not self.session:
                    self.session = aiohttp.ClientSession()
                
                async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        peer.is_active = True
                        peer.last_seen = datetime.now()
                    else:
                        peer.is_active = False
                        logger.warning(f"Peer {node_id} returned status {response.status}")
                        
            except asyncio.TimeoutError:
                peer.is_active = False
                logger.warning(f"Peer {node_id} health check timeout")
            except Exception as e:
                peer.is_active = False
                logger.warning(f"Peer {node_id} health check failed: {e}")
    
    async def get_peer_statistics(self) -> Dict:
        """Get statistics about connected peers"""
        active_peers = [p for p in self.peers.values() if p.is_active]
        
        return {
            'total_peers': len(self.peers),
            'active_peers': len(active_peers),
            'inactive_peers': len(self.peers) - len(active_peers),
            'average_trust_score': sum(p.trust_score for p in active_peers) / len(active_peers) if active_peers else 0,
            'peer_list': [
                {
                    'node_id': p.node_id,
                    'host': p.host,
                    'port': p.port,
                    'trust_score': p.trust_score,
                    'is_active': p.is_active,
                    'last_seen': p.last_seen.isoformat()
                }
                for p in self.peers.values()
            ]
        }

if __name__ == "__main__":
    # Test the peer client
    async def test_peer_communication():
        # Create two peer clients
        node1 = PeerClient("node_1", "localhost", 8001)
        node2 = PeerClient("node_2", "localhost", 8002)
        
        try:
            # Start servers
            await node1.start_server()
            await node2.start_server()
            
            # Add each other as peers
            await node1.add_peer("node_2", "localhost", 8002)
            await node2.add_peer("node_1", "localhost", 8001)
            
            # Test message sending
            success = await node1.send_message("node_2", "test_message", {"data": "hello"})
            print(f"Message sent: {success}")
            
            # Test broadcasting
            results = await node1.broadcast_message("broadcast_test", {"data": "broadcast"})
            print(f"Broadcast results: {results}")
            
            # Get statistics
            stats = await node1.get_peer_statistics()
            print("Node 1 statistics:")
            print(json.dumps(stats, indent=2))
            
            # Wait a bit
            await asyncio.sleep(2)
            
        finally:
            await node1.stop_server()
            await node2.stop_server()
    
    asyncio.run(test_peer_communication())
