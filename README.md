# Decentralized Website Monitoring System with ML-based Malicious Node Detection and Blockchain-based Proof of Reputation

A comprehensive decentralized system for monitoring websites, detecting malicious nodes using machine learning, and maintaining reputation scores on blockchain.

## System Overview

The system consists of multiple distributed nodes that:
1. Monitor websites (HTTP status, response time, SSL, DNS)
2. Share results with peer nodes (P2P communication)
3. Detect inconsistencies (content hash mismatch, false reports)
4. Generate ML features for malicious node detection
5. Compute Proof of Reputation (PoR) scores
6. Store reputation data on blockchain
7. Provide real-time visualization dashboard

## Architecture

```
WebMonitoring/
|
|-- ml/                          # Machine Learning Module
|   |-- src/
|   |   |-- train_model.py       # ML model training
|   |   |-- predict.py           # ML inference
|   |-- models/                  # Trained models
|   |-- data/                    # Dataset
|   `-- requirements.txt
|
|-- blockchain/                   # Blockchain Module
|   |-- contracts/
|   |   `-- ProofOfReputation.sol # Smart contract
|   |-- scripts/
|   |   `-- deploy.js           # Deployment script
|   |-- test/                    # Contract tests
|   |-- src/
|   |   `-- blockchain_client.py # Blockchain integration
|   |-- hardhat.config.js
|   |-- package.json
|   `-- requirements.txt
|
|-- node_service/                # Node Service
|   |-- src/
|   |   |-- website_monitor.py   # Website monitoring
|   |   |-- trust_engine.py      # Trust calculation
|   |   |-- peer_client.py       # P2P communication
|   |   `-- feature_engineering.py # Feature extraction
|   |-- main.py                  # FastAPI application
|   `-- requirements.txt
|
|-- dashboard/                    # Visualization Dashboard
|   |-- src/
|   |   `-- app.py              # Streamlit dashboard
|   `-- requirements.txt
|
|-- config/                      # Configuration files
|-- logs/                        # Log files
|-- .env.example                 # Environment template
|-- requirements.txt             # Complete dependencies
`-- README.md                    # This file
```

## Features

### Website Monitoring
- HTTP status checking
- Response time measurement
- SSL certificate validation
- DNS resolution monitoring
- Content hash calculation for consistency

### Machine Learning
- Random Forest classifier for malicious node detection
- Real-time feature extraction
- ML confidence scoring
- Model training and evaluation

### Trust Engine
- Multi-factor trust calculation
- Peer feedback integration
- Content consistency checking
- Time-based trust decay

### Blockchain Integration
- Ethereum smart contract for reputation storage
- Proof of Reputation (PoR) calculation
- Sepolia testnet support
- Gas-optimized transactions

### P2P Communication
- Asynchronous peer-to-peer messaging
- Node discovery
- Health monitoring
- Data sharing

### Dashboard
- Real-time visualization
- Trust score monitoring
- ML feature analysis
- Peer network overview
- System statistics

## Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### Setup Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd WebMonitoring
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install blockchain dependencies**
```bash
cd blockchain
npm install
cd ..
```

5. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### 1. Train ML Model

```bash
cd ml/src
python train_model.py
```

This will:
- Load the dataset from `web_monitor_dataset.csv`
- Train a Random Forest classifier
- Save the model to `ml/models/`
- Display evaluation metrics

### 2. Deploy Blockchain Contract

**Local Development:**
```bash
cd blockchain
npm run node  # Start local Hardhat node (in separate terminal)
npm run compile  # Compile contracts
npm run deploy:local  # Deploy to local network
```

**Sepolia Testnet:**
```bash
cd blockchain
npm run compile
npm run deploy:sepolia  # Deploy to Sepolia testnet
```

### 3. Run Node Service

```bash
cd node_service
python main.py
```

The node will:
- Start the FastAPI server on port 8000
- Begin monitoring configured websites
- Connect to peer nodes
- Update blockchain reputation

### 4. Launch Dashboard

```bash
cd dashboard/src
streamlit run app.py --server.port 8501
```

Access the dashboard at `http://localhost:8501`

## Configuration

### Environment Variables

Key environment variables in `.env`:

```env
# Node Configuration
NODE_ID=node_1
NODE_HOST=localhost
NODE_PORT=8000
WEBSITES=https://httpbin.org/status/200,https://google.com
MONITORING_INTERVAL=60

# Blockchain Configuration
ETHEREUM_RPC_URL=http://127.0.0.1:8545
PRIVATE_KEY=0xYourPrivateKeyHere
CONTRACT_ADDRESS=
CHAIN_ID=31337

# ML Configuration
MODEL_PATH=./ml/models
ML_CONFIDENCE_THRESHOLD=0.7
```

### Multiple Nodes

To run multiple nodes:

```bash
# Terminal 1 - Node 1
export NODE_ID=node_1
export NODE_PORT=8000
python node_service/main.py

# Terminal 2 - Node 2
export NODE_ID=node_2
export NODE_PORT=8001
python node_service/main.py

# Terminal 3 - Node 3
export NODE_ID=node_3
export NODE_PORT=8002
python node_service/main.py
```

## API Documentation

### Node Service Endpoints

- `GET /` - Node information
- `GET /health` - Health check
- `GET /trust` - Trust information
- `GET /features` - ML features
- `GET /peers` - Peer information
- `GET /statistics` - System statistics
- `POST /monitor` - Trigger manual monitoring
- `POST /peers` - Add peer
- `POST /reputation` - Update reputation
- `GET /blockchain/reputation/{node_id}` - Get blockchain reputation

### Example API Usage

```python
import requests

# Get node health
response = requests.get("http://localhost:8000/health")
print(response.json())

# Trigger monitoring
response = requests.post(
    "http://localhost:8000/monitor",
    json={"urls": ["https://example.com"]}
)
print(response.json())

# Get trust information
response = requests.get("http://localhost:8000/trust")
print(response.json())
```

## ML Model Details

### Features
- `avg_response_ms`: Average response time
- `ssl_valid_rate`: SSL certificate validity rate
- `content_match_rate`: Content consistency with peers
- `stale_report_rate`: Rate of outdated reports
- `false_report_rate`: Rate of failed reports

### Model Performance
- Algorithm: Random Forest
- Accuracy: ~95% (on synthetic dataset)
- Features: 5 monitoring metrics
- Output: Binary classification (Honest/Malicious)

## Blockchain Integration

### Smart Contract Features
- Node registration
- Reputation updates
- Query functions
- Top nodes retrieval
- Owner management

### PoR Calculation
```
PoR = 0.4 * monitoring_trust + 0.6 * ML_score
```

### Supported Networks
- Local Hardhat network
- Sepolia testnet
- Ethereum mainnet (production)

## Development

### Running Tests

**ML Tests:**
```bash
cd ml/src
python -m pytest
```

**Blockchain Tests:**
```bash
cd blockchain
npm test
```

**Node Service Tests:**
```bash
cd node_service
python -m pytest
```

### Code Structure

- **Modular Design**: Each component is self-contained
- **Async Operations**: All I/O operations are asynchronous
- **Error Handling**: Comprehensive error handling and logging
- **Configuration**: Environment-based configuration
- **Testing**: Unit tests for all major components

## Troubleshooting

### Common Issues

1. **ML Model Not Found**
   - Run `python ml/src/train_model.py` first
   - Check `MODEL_PATH` environment variable

2. **Blockchain Connection Failed**
   - Verify Ethereum RPC URL
   - Check private key format
   - Ensure node is running for local network

3. **Peer Connection Issues**
   - Check firewall settings
   - Verify peer addresses and ports
   - Ensure all nodes are running

4. **Dashboard Not Loading**
   - Verify API base URL in dashboard
   - Check if node service is running
   - Ensure port 8501 is available

### Logs

Check log files in the `logs/` directory:
- `node.log` - Node service logs
- `ml.log` - ML model logs
- `blockchain.log` - Blockchain interaction logs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs
3. Create an issue in the repository

---

**Note**: This is a research/educational project. For production use, additional security measures and optimizations are recommended.
