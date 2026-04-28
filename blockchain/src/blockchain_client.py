"""
Blockchain Integration Client
Handles interaction with Ethereum blockchain for Proof of Reputation system
"""

import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BlockchainClient:
    """Client for interacting with Proof of Reputation smart contract"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize blockchain client
        
        Args:
            config: Configuration dictionary with blockchain settings
        """
        self.config = config or self._load_config()
        self.w3 = None
        self.contract = None
        self.account = None
        self.contract_address = None
        self.contract_abi = None
        
        self._connect()
        self._load_contract()
    
    def _load_config(self) -> Dict:
        """Load blockchain configuration from environment or config file"""
        config = {
            'rpc_url': os.getenv('ETHEREUM_RPC_URL', 'http://127.0.0.1:8545'),
            'private_key': os.getenv('PRIVATE_KEY'),
            'contract_address': os.getenv('CONTRACT_ADDRESS'),
            'chain_id': int(os.getenv('CHAIN_ID', '31337')),
            'gas_limit': int(os.getenv('GAS_LIMIT', '200000')),
            'gas_price_gwei': int(os.getenv('GAS_PRICE_GWEI', '20'))
        }
        
        # Try to load from deployment file if contract address not in env
        if not config['contract_address']:
            deployment_file = os.path.join(os.path.dirname(__file__), '..', 'deployment.json')
            if os.path.exists(deployment_file):
                with open(deployment_file, 'r') as f:
                    deployment = json.load(f)
                    config['contract_address'] = deployment.get('contractAddress')
        
        return config
    
    def _connect(self):
        """Connect to Ethereum network"""
        try:
            logger.info(f"Connecting to Ethereum network: {self.config['rpc_url']}")
            
            self.w3 = Web3(Web3.HTTPProvider(self.config['rpc_url']))
            
            # Check connection
            if not self.w3.is_connected():
                raise ConnectionError("Failed to connect to Ethereum network")
            
            # Add middleware for POA networks (like Sepolia)
            if self.config['chain_id'] != 1:  # Not mainnet
                self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            # Set up account if private key provided
            if self.config['private_key']:
                self.account = Account.from_key(self.config['private_key'])
                logger.info(f"Using account: {self.account.address}")
            else:
                # Use first account from node
                accounts = self.w3.eth.accounts
                if accounts:
                    self.account = Account.from_key(os.getenv('PRIVATE_KEY', '0x' + '0' * 64))
                    logger.warning("Using default account (set PRIVATE_KEY for proper signing)")
            
            logger.info("Connected to Ethereum network successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to blockchain: {e}")
            raise
    
    def _load_contract(self):
        """Load smart contract ABI and address"""
        try:
            # Load contract ABI
            abi_file = os.path.join(os.path.dirname(__file__), '..', 'ProofOfReputation.json')
            if os.path.exists(abi_file):
                with open(abi_file, 'r') as f:
                    contract_data = json.load(f)
                    self.contract_abi = contract_data['abi']
                    if not self.config['contract_address']:
                        self.contract_address = contract_data.get('address')
            else:
                # Fallback to compiled artifacts
                artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
                contract_artifact = os.path.join(artifacts_dir, 'contracts', 'ProofOfReputation.sol', 'ProofOfReputation.json')
                
                if os.path.exists(contract_artifact):
                    with open(contract_artifact, 'r') as f:
                        artifact = json.load(f)
                        self.contract_abi = artifact['abi']
                else:
                    raise FileNotFoundError("Contract ABI not found")
            
            if not self.config['contract_address']:
                raise ValueError("Contract address not configured")
            
            self.contract_address = self.config['contract_address']
            
            # Create contract instance
            self.contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=self.contract_abi
            )
            
            logger.info(f"Contract loaded at address: {self.contract_address}")
            
        except Exception as e:
            logger.error(f"Failed to load contract: {e}")
            raise
    
    def _send_transaction(self, function_call, value: int = 0) -> Dict:
        """Send transaction to blockchain"""
        try:
            # Build transaction
            transaction = function_call.build_transaction({
                'from': self.account.address,
                'value': value,
                'gas': self.config['gas_limit'],
                'gasPrice': self.w3.to_wei(self.config['gas_price_gwei'], 'gwei'),
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'chainId': self.config['chain_id']
            })
            
            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.config['private_key'])
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
            
            if receipt.status == 1:
                logger.info(f"Transaction successful: {tx_hash.hex()}")
                return {
                    'success': True,
                    'tx_hash': tx_hash.hex(),
                    'block_number': receipt.blockNumber,
                    'gas_used': receipt.gasUsed
                }
            else:
                logger.error(f"Transaction failed: {tx_hash.hex()}")
                return {
                    'success': False,
                    'tx_hash': tx_hash.hex(),
                    'error': 'Transaction reverted'
                }
                
        except Exception as e:
            logger.error(f"Transaction error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def register_node(self, node_id: str) -> Dict:
        """
        Register a new node in the blockchain
        
        Args:
            node_id: Unique identifier for the node
            
        Returns:
            Transaction result
        """
        try:
            logger.info(f"Registering node: {node_id}")
            
            # Check if already registered
            if self.is_node_registered(node_id):
                logger.warning(f"Node {node_id} already registered")
                return {
                    'success': False,
                    'error': 'Node already registered'
                }
            
            # Call registerNode function
            function_call = self.contract.functions.registerNode(node_id)
            result = self._send_transaction(function_call)
            
            if result['success']:
                logger.info(f"Node {node_id} registered successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to register node {node_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_reputation(self, node_id: str, monitoring_trust: float, ml_score: float) -> Dict:
        """
        Update node reputation on blockchain
        
        Args:
            node_id: Node identifier
            monitoring_trust: Monitoring trust score (0-1)
            ml_score: ML confidence score (0-1)
            
        Returns:
            Transaction result
        """
        try:
            logger.info(f"Updating reputation for node: {node_id}")
            
            # Convert to blockchain format (0-1000)
            monitoring_trust_scaled = int(monitoring_trust * 1000)
            ml_score_scaled = int(ml_score * 1000)
            
            # Call updateReputation function
            function_call = self.contract.functions.updateReputation(
                node_id, 
                monitoring_trust_scaled, 
                ml_score_scaled
            )
            result = self._send_transaction(function_call)
            
            if result['success']:
                logger.info(f"Reputation updated for node {node_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to update reputation for node {node_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def submit_report(self, node_id: str, success: bool) -> Dict:
        """
        Submit a report from a node
        
        Args:
            node_id: Node identifier
            success: Whether the report was successful/accurate
            
        Returns:
            Transaction result
        """
        try:
            logger.info(f"Submitting report for node: {node_id}, success: {success}")
            
            # Call submitReport function
            function_call = self.contract.functions.submitReport(node_id, success)
            result = self._send_transaction(function_call)
            
            if result['success']:
                logger.info(f"Report submitted for node {node_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to submit report for node {node_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_node_reputation(self, node_id: str) -> Optional[Dict]:
        """
        Get node reputation from blockchain
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node reputation data or None if not found
        """
        try:
            if not self.is_node_registered(node_id):
                return None
            
            # Call getNodeReputation function
            reputation_data = self.contract.functions.getNodeReputation(node_id).call()
            
            # Convert to readable format
            result = {
                'node_id': node_id,
                'reputation': reputation_data[0] / 1000.0,  # Convert from 0-1000 to 0-1
                'monitoring_trust': reputation_data[1] / 1000.0,
                'ml_score': reputation_data[2] / 1000.0,
                'last_updated': datetime.fromtimestamp(reputation_data[3]).isoformat(),
                'is_registered': True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get reputation for node {node_id}: {e}")
            return None
    
    def get_node_stats(self, node_id: str) -> Optional[Dict]:
        """
        Get node statistics from blockchain
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node statistics or None if not found
        """
        try:
            if not self.is_node_registered(node_id):
                return None
            
            # Call getNodeStats function
            stats_data = self.contract.functions.getNodeStats(node_id).call()
            
            # Convert to readable format
            result = {
                'node_id': node_id,
                'total_reports': stats_data[0],
                'successful_reports': stats_data[1],
                'success_rate': stats_data[2]  # Already in percentage
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get stats for node {node_id}: {e}")
            return None
    
    def is_node_registered(self, node_id: str) -> bool:
        """Check if node is registered on blockchain"""
        try:
            return self.contract.functions.isNodeRegistered(node_id).call()
        except Exception as e:
            logger.error(f"Failed to check registration for node {node_id}: {e}")
            return False
    
    def get_all_nodes(self) -> List[str]:
        """Get all registered node IDs"""
        try:
            return self.contract.functions.getAllNodes().call()
        except Exception as e:
            logger.error(f"Failed to get all nodes: {e}")
            return []
    
    def get_top_nodes(self, n: int = 10) -> List[Dict]:
        """Get top N nodes by reputation"""
        try:
            node_ids, reputations = self.contract.functions.getTopNodes(n).call()
            
            result = []
            for i in range(len(node_ids)):
                result.append({
                    'node_id': node_ids[i],
                    'reputation': reputations[i] / 1000.0  # Convert to 0-1 scale
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get top nodes: {e}")
            return []
    
    def get_node_count(self) -> int:
        """Get total number of registered nodes"""
        try:
            return self.contract.functions.getNodeCount().call()
        except Exception as e:
            logger.error(f"Failed to get node count: {e}")
            return 0
    
    def calculate_por_score(self, monitoring_trust: float, ml_score: float) -> float:
        """
        Calculate Proof of Reputation score
        
        Args:
            monitoring_trust: Monitoring trust score (0-1)
            ml_score: ML confidence score (0-1)
            
        Returns:
            PoR score (0-1)
        """
        return 0.4 * monitoring_trust + 0.6 * ml_score
    
    def health_check(self) -> Dict:
        """Check blockchain connection and contract availability"""
        try:
            # Check web3 connection
            if not self.w3.is_connected():
                return {
                    'status': 'unhealthy',
                    'error': 'Not connected to blockchain'
                }
            
            # Check contract
            if not self.contract:
                return {
                    'status': 'unhealthy',
                    'error': 'Contract not loaded'
                }
            
            # Try to call a view function
            node_count = self.get_node_count()
            latest_block = self.w3.eth.block_number
            
            return {
                'status': 'healthy',
                'blockchain': {
                    'connected': True,
                    'latest_block': latest_block,
                    'chain_id': self.config['chain_id']
                },
                'contract': {
                    'address': self.contract_address,
                    'node_count': node_count
                }
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

# Singleton instance for global use
_blockchain_client = None

def get_blockchain_client(config: Dict = None) -> BlockchainClient:
    """Get or create blockchain client instance"""
    global _blockchain_client
    
    if _blockchain_client is None:
        _blockchain_client = BlockchainClient(config)
    
    return _blockchain_client

if __name__ == "__main__":
    # Test the blockchain client
    import json
    
    try:
        client = get_blockchain_client()
        
        # Health check
        health = client.health_check()
        print("Health Check:")
        print(json.dumps(health, indent=2))
        
        # Get node count
        node_count = client.get_node_count()
        print(f"\nTotal nodes: {node_count}")
        
        # Get all nodes
        all_nodes = client.get_all_nodes()
        print(f"All nodes: {all_nodes}")
        
        # Get top nodes
        top_nodes = client.get_top_nodes(5)
        print("\nTop 5 nodes:")
        for node in top_nodes:
            print(f"  {node['node_id']}: {node['reputation']:.3f}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure blockchain node is running and contract is deployed")
