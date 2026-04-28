"""
Production Deployment Script for Web Monitoring System
Deploys to Polygon mainnet with high-performance configuration
"""

import asyncio
import json
import os
import subprocess
import time
from typing import Dict, List, Optional
import web3
from web3 import Web3
from eth_account import Account
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionDeployer:
    """Production deployment manager for high-throughput monitoring system"""
    
    def __init__(self):
        self.config = self.load_config()
        self.web3 = None
        self.account = None
        self.contract_address = None
        
    def load_config(self) -> Dict:
        """Load production configuration"""
        return {
            # Polygon mainnet RPC
            "rpc_url": "https://polygon-rpc.com",
            "fallback_rpcs": [
                "https://rpc-mainnet.matic.network",
                "https://rpc-mainnet.maticvigil.com",
                "https://rpc-mainnet.matic.quiknode.pro"
            ],
            
            # Contract deployment
            "private_key": os.getenv("PRIVATE_KEY"),
            "gas_limit": 8000000,
            "gas_price_gwei": 30,  # Adjust based on network conditions
            
            # Node configuration
            "num_nodes": 21,  # DPoS validators
            "min_stake": "1000",  # 1000 MON tokens
            "node_regions": {
                "NA": ["us-east-1", "us-west-1", "us-central-1"],
                "EU": ["eu-west-1", "eu-central-1", "eu-north-1"],
                "AP": ["ap-southeast-1", "ap-northeast-1", "ap-south-1"]
            },
            
            # Monitoring targets (production URLs)
            "critical_targets": [
                "https://aws.amazon.com",
                "https://cloud.google.com",
                "https://azure.microsoft.com",
                "https://www.cloudflare.com",
                "https://api.stripe.com",
                "https://api.paypal.com"
            ],
            
            # Infrastructure
            "redis_cluster": [
                "redis-1.monitoring.com:6379",
                "redis-2.monitoring.com:6379",
                "redis-3.monitoring.com:6379"
            ],
            
            # Load balancer
            "load_balancer": "lb.monitoring.com",
            "ssl_cert_path": "/etc/ssl/certs/monitoring.com.crt"
        }
    
    async def connect_to_blockchain(self) -> bool:
        """Connect to Polygon mainnet with fallback RPCs"""
        for rpc_url in [self.config["rpc_url"]] + self.config["fallback_rpcs"]:
            try:
                self.web3 = Web3(Web3.HTTPProvider(rpc_url))
                if self.web3.is_connected():
                    logger.info(f"Connected to Polygon via {rpc_url}")
                    break
            except Exception as e:
                logger.warning(f"Failed to connect to {rpc_url}: {e}")
                continue
        else:
            raise Exception("Failed to connect to any RPC endpoint")
        
        # Setup account
        if not self.config["private_key"]:
            raise Exception("PRIVATE_KEY environment variable not set")
        
        self.account = Account.from_key(self.config["private_key"])
        logger.info(f"Using account: {self.account.address}")
        
        # Check balance
        balance = self.web3.eth.get_balance(self.account.address)
        logger.info(f"Account balance: {self.web3.from_wei(balance, 'ether')} MATIC")
        
        if balance < self.web3.to_wei(1, 'ether'):
            raise Exception("Insufficient MATIC balance for deployment")
        
        return True
    
    async def deploy_production_contract(self) -> str:
        """Deploy production smart contract to Polygon mainnet"""
        try:
            logger.info("Deploying production contract to Polygon mainnet...")
            
            # Load contract artifacts
            with open("blockchain/artifacts/ProductionProofOfReputation.json", "r") as f:
                contract_data = json.load(f)
            
            contract_abi = contract_data["abi"]
            contract_bytecode = contract_data["bytecode"]
            
            # Create contract instance
            contract = self.web3.eth.contract(
                abi=contract_abi,
                bytecode=contract_bytecode
            )
            
            # Get nonce
            nonce = self.web3.eth.get_transaction_count(self.account.address)
            
            # Build deployment transaction
            deploy_tx = contract.constructor().build_transaction({
                'from': self.account.address,
                'gas': self.config["gas_limit"],
                'gasPrice': self.web3.to_wei(self.config["gas_price_gwei"], 'gwei'),
                'nonce': nonce,
                'chainId': 137  # Polygon mainnet
            })
            
            # Sign transaction
            signed_tx = self.web3.eth.account.sign_transaction(
                deploy_tx, 
                self.config["private_key"]
            )
            
            # Send transaction
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            logger.info(f"Deployment transaction sent: {tx_hash.hex()}")
            
            # Wait for confirmation
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
            
            if receipt.status == 1:
                self.contract_address = receipt.contractAddress
                logger.info(f"Contract deployed successfully: {self.contract_address}")
                return self.contract_address
            else:
                raise Exception("Contract deployment failed")
                
        except Exception as e:
            logger.error(f"Contract deployment error: {e}")
            raise
    
    async def setup_validator_nodes(self) -> List[Dict]:
        """Setup and configure validator nodes"""
        logger.info("Setting up validator nodes...")
        
        validators = []
        
        for i in range(self.config["num_nodes"]):
            # Determine region and shard
            region_index = i % len(self.config["node_regions"])
            region = list(self.config["node_regions"].keys())[region_index]
            shard_id = region_index
            
            # Create node configuration
            node_config = {
                "node_id": f"validator_{i:02d}",
                "node_address": self.account.address,  # For demo, use same account
                "stake_amount": int(self.config["min_stake"]) * 10**18,
                "shard_id": shard_id,
                "region": region,
                "endpoint": f"node{i:02d}.monitoring.com",
                "port": 8000 + i,
                "is_validator": True
            }
            
            validators.append(node_config)
            logger.info(f"Configured validator {i}: {node_config['node_id']} in {region}")
        
        return validators
    
    def generate_docker_compose(self, validators: List[Dict]) -> str:
        """Generate Docker Compose configuration for production deployment"""
        
        compose = {
            "version": "3.8",
            "services": {
                "redis-cluster": {
                    "image": "redis:7-alpine",
                    "deploy": {
                        "replicas": 3
                    },
                    "ports": ["6379:6379"],
                    "volumes": ["redis-data:/data"],
                    "command": "redis-server --appendonly yes --cluster-enabled yes"
                },
                
                "load-balancer": {
                    "image": "nginx:alpine",
                    "ports": ["80:80", "443:443"],
                    "volumes": [
                        "./nginx.conf:/etc/nginx/nginx.conf:ro",
                        f"{self.config['ssl_cert_path']}:/etc/ssl/certs/monitoring.com.crt:ro"
                    ],
                    "depends_on": ["monitoring-nodes"]
                }
            }
        }
        
        # Add validator nodes
        for i, validator in enumerate(validators):
            service_name = f"node-{i:02d}"
            
            compose["services"][service_name] = {
                "build": {
                    "context": ".",
                    "dockerfile": "Dockerfile.production"
                },
                "environment": [
                    f"NODE_ID={validator['node_id']}",
                    f"SHARD_ID={validator['shard_id']}",
                    f"REGION={validator['region']}",
                    f"PRIVATE_KEY={self.config['private_key']}",
                    f"CONTRACT_ADDRESS={self.contract_address}",
                    "REDIS_URL=redis://redis-cluster:6379",
                    "BLOCKCHAIN_RPC_URL=https://polygon-rpc.com"
                ],
                "ports": [f"{validator['port']}:8000"],
                "deploy": {
                    "replicas": 1,
                    "resources": {
                        "limits": {
                            "cpus": "2.0",
                            "memory": "4G"
                        }
                    }
                },
                "restart": "unless-stopped",
                "healthcheck": {
                    "test": ["CMD", "curl", "-f", f"http://localhost:8000/health"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3
                }
            }
        
        # Add monitoring-nodes dependency to load balancer
        compose["services"]["load-balancer"]["depends_on"] = [f"node-{i:02d}" for i in range(len(validators))]
        
        # Add volumes
        compose["volumes"] = {
            "redis-data": {}
        }
        
        return json.dumps(compose, indent=2)
    
    async def deploy_production(self) -> bool:
        """Execute full production deployment"""
        try:
            logger.info("Starting production deployment...")
            
            # Step 1: Connect to blockchain
            await self.connect_to_blockchain()
            
            # Step 2: Deploy contract
            contract_address = await self.deploy_production_contract()
            
            # Step 3: Setup validators
            validators = await self.setup_validator_nodes()
            
            # Step 4: Generate configuration files
            docker_compose = self.generate_docker_compose(validators)
            
            # Save configuration files
            with open("docker-compose.production.yml", "w") as f:
                f.write(docker_compose)
            
            logger.info("Production deployment completed successfully!")
            logger.info(f"Contract address: {contract_address}")
            logger.info("Configuration file generated: docker-compose.production.yml")
            
            return True
            
        except Exception as e:
            logger.error(f"Production deployment failed: {e}")
            return False

async def main():
    """Main deployment function"""
    deployer = ProductionDeployer()
    success = await deployer.deploy_production()
    
    if success:
        print("\n🎉 Production deployment completed successfully!")
        print("\nNext steps:")
        print("1. Review generated docker-compose.production.yml")
        print("2. Deploy to production infrastructure:")
        print("   docker-compose -f docker-compose.production.yml up -d")
        print("3. Configure DNS and SSL certificates")
        print("4. Monitor system performance")
    else:
        print("\n❌ Production deployment failed!")
        print("Check logs for details and retry.")

if __name__ == "__main__":
    asyncio.run(main())
