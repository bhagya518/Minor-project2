#!/usr/bin/env python3
"""
Enhanced ML Consensus Engine integrating ML_MINOR capabilities
Combines RF + IF fusion with EWMA smoothing and 4-tier mitigation
"""

import os
import json
import math
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class MitigationDecision:
    def __init__(self, status: str, action: str, shard: str):
        self.status = status  # HEALTHY/SUSPICIOUS/FAULTY/MALICIOUS
        self.action = action  # ALLOW/WARN/QUARANTINE/SLASHED
        self.shard = shard    # PRIMARY/MONITORING/QUARANTINE/SLASHED

class EnhancedMLConsensusEngine:
    """
    Enhanced ML consensus engine with ML_MINOR integration
    """
    
    def __init__(self, node_id: str, alpha: float = 0.9, iso_contamination: float = 0.15):
        self.node_id = node_id
        self.alpha = alpha  # EWMA smoothing factor
        self.iso_contamination = iso_contamination
        
        # ML Models (ML_MINOR approach)
        self.rf_model = None
        self.iso_model = None
        self.rf_scaler = None
        self.iso_scaler = None
        self.models_loaded = False
        
        # Feature columns
        self.rf_feature_cols = []
        self.behavioral_cols = []
        
        # Load ML_MINOR models
        self.load_enhanced_models()
        
        # Enhanced reputation tracking with EWMA
        self.reputation = {}  # node_id -> current_reputation
        self.reputation_history = defaultdict(list)  # node_id -> [reputation_history]
        self.ewma_reputations = {}  # node_id -> ewma_reputation
        
        # Graph for network analysis
        self.graph = nx.DiGraph()
        
        # Enhanced mitigation thresholds (4-tier)
        self.HEALTHY_T = 0.8
        self.SUSPICIOUS_T = 0.5
        self.FAULTY_T = 0.2
        
        # Consensus tracking
        self.consensus_votes = defaultdict(list)  # epoch_id -> [votes]
        self.consensus_decisions = {}  # epoch_id -> consensus_decision
        
        # Mitigation tracking
        self.mitigation_actions = {}  # node_id -> MitigationDecision
        
        logger.info(f"EnhancedMLConsensusEngine initialized for node {node_id}")
    
    def load_enhanced_models(self):
        """Load ML_MINOR models"""
        try:
            # Path to ML_MINOR models
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ml', 'models')
            
            # Load Random Forest model
            rf_artifact = joblib.load(os.path.join(model_path, 'rf_backbone.joblib'))
            self.rf_model = rf_artifact['model']
            self.rf_feature_cols = list(rf_artifact['feature_cols'])
            rf_scaler_type = rf_artifact.get('scaler_type', 'standard')
            
            # Load Isolation Forest model
            iso_artifact = joblib.load(os.path.join(model_path, 'iso_backbone.joblib'))
            self.iso_model = iso_artifact['model']
            self.behavioral_cols = list(iso_artifact['behavioral_cols'])
            self.iso_scaler = iso_artifact['scaler']
            
            # Initialize RF scaler
            if rf_scaler_type == 'minmax':
                self.rf_scaler = MinMaxScaler()
            else:
                self.rf_scaler = StandardScaler()
            
            self.models_loaded = True
            
            logger.info(f"✅ Loaded ML_MINOR enhanced models")
            logger.info(f"RF features: {len(self.rf_feature_cols)}")
            logger.info(f"Behavioral features: {len(self.behavioral_cols)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load enhanced models: {e}")
            logger.info("Will fall back to basic consensus")
    
    def apply_mitigation_policy(self, reputation_score: float) -> MitigationDecision:
        """Apply 4-tier mitigation policy from ML_MINOR"""
        if reputation_score > self.HEALTHY_T:
            return MitigationDecision(status="HEALTHY", action="ALLOW", shard="PRIMARY")
        elif reputation_score > self.SUSPICIOUS_T:
            return MitigationDecision(status="SUSPICIOUS", action="WARN", shard="MONITORING")
        elif reputation_score > self.FAULTY_T:
            return MitigationDecision(status="FAULTY", action="QUARANTINE", shard="QUARANTINE")
        else:
            return MitigationDecision(status="MALICIOUS", action="SLASHED", shard="SLASHED")
    
    def normalize_0_1(self, arr: np.ndarray, mn: float, mx: float) -> np.ndarray:
        """Normalize array to 0-1 range"""
        return (arr - mn) / (mx - mn + 1e-12)
    
    def calculate_enhanced_reputation(self, features: Dict) -> float:
        """Calculate reputation using ML_MINOR RF + IF fusion"""
        if not self.models_loaded:
            return 0.5  # Default neutral reputation
        
        try:
            # Prepare RF features
            rf_features = []
            for col in self.rf_feature_cols:
                rf_features.append(float(features.get(col, 0.0)))
            
            # Scale RF features
            rf_input = np.array(rf_features).reshape(1, -1)
            rf_scaled = self.rf_scaler.fit_transform(rf_input)
            
            # Get RF probability
            rf_prob = float(self.rf_model.predict_proba(rf_scaled)[:, 1][0])
            
            # Prepare behavioral features for Isolation Forest
            beh_features = []
            for col in self.behavioral_cols:
                beh_features.append(float(features.get(col, 0.0)))
            
            # Scale behavioral features
            beh_input = np.array(beh_features).reshape(1, -1)
            beh_scaled = self.iso_scaler.transform(beh_input)
            
            # Get Isolation Forest score (normalized)
            iso_score = float(-self.iso_model.decision_function(beh_scaled)[0])
            
            # Normalize ISO score (approximate normalization)
            iso_norm = float(np.clip(iso_score / 10.0, 0.0, 1.0))
            
            # Fusion: 70% RF + 30% ISO (ML_MINOR approach)
            risk = (0.7 * rf_prob) + (0.3 * iso_norm)
            reputation = 1.0 - float(np.clip(risk, 0.0, 1.0))
            
            return reputation
            
        except Exception as e:
            logger.error(f"Error calculating enhanced reputation: {e}")
            return 0.5
    
    def apply_ewma_smoothing(self, node_id: str, current_reputation: float) -> float:
        """Apply EWMA smoothing to reputation"""
        if node_id not in self.ewma_reputations:
            # Initialize with current reputation
            self.ewma_reputations[node_id] = current_reputation
            return current_reputation
        
        # Apply EWMA formula
        ewma_rep = self.alpha * self.ewma_reputations[node_id] + (1.0 - self.alpha) * current_reputation
        self.ewma_reputations[node_id] = ewma_rep
        
        return ewma_rep
    
    def evaluate_node(self, node_id: str, features: Dict) -> Tuple[float, MitigationDecision]:
        """Evaluate node and return reputation with mitigation decision"""
        # Calculate raw reputation
        raw_reputation = self.calculate_enhanced_reputation(features)
        
        # Apply EWMA smoothing
        smoothed_reputation = self.apply_ewma_smoothing(node_id, raw_reputation)
        
        # Store reputation
        self.reputation[node_id] = smoothed_reputation
        self.reputation_history[node_id].append(smoothed_reputation)
        
        # Apply mitigation policy
        decision = self.apply_mitigation_policy(smoothed_reputation)
        self.mitigation_actions[node_id] = decision
        
        return smoothed_reputation, decision
    
    def extract_features_from_report(self, report: Dict) -> Dict:
        """Extract features from monitoring report"""
        features = {}
        
        # Basic monitoring features
        features['accuracy'] = report.get('accuracy', 0.0)
        features['false_positive_rate'] = report.get('false_positive_rate', 0.0)
        features['false_negative_rate'] = report.get('false_negative_rate', 0.0)
        features['avg_rt_error'] = report.get('avg_rt_error', 0.0)
        features['max_rt_error'] = report.get('max_rt_error', 0.0)
        features['peer_agreement_rate'] = report.get('peer_agreement_rate', 0.0)
        features['historical_accuracy'] = report.get('historical_accuracy', 0.0)
        features['accuracy_std_dev'] = report.get('accuracy_std_dev', 0.0)
        features['report_consistency'] = report.get('report_consistency', 0.0)
        features['sudden_change_score'] = report.get('sudden_change_score', 0.0)
        features['ssl_accuracy'] = report.get('ssl_accuracy', 0.0)
        features['uptime_deviation'] = report.get('uptime_deviation', 0.0)
        features['rt_consistency'] = report.get('rt_consistency', 0.0)
        
        # Behavioral features for anomaly detection
        features['itt_jitter'] = report.get('itt_jitter', 0.0)
        features['response_time_variance'] = report.get('response_time_variance', 0.0)
        features['report_frequency'] = report.get('report_frequency', 0.0)
        features['timeout_rate'] = report.get('timeout_rate', 0.0)
        features['error_burst_score'] = report.get('error_burst_score', 0.0)
        
        return features
    
    def process_consensus_round(self, epoch_id: str, reports: List[Dict]) -> Dict:
        """Process consensus round with enhanced ML evaluation"""
        results = {
            'epoch_id': epoch_id,
            'evaluations': {},
            'mitigation_actions': {},
            'consensus_decision': None,
            'summary': {}
        }
        
        node_evaluations = {}
        
        # Evaluate each node
        for report in reports:
            node_id = report.get('node_id')
            if not node_id:
                continue
                
            features = self.extract_features_from_report(report)
            reputation, decision = self.evaluate_node(node_id, features)
            
            node_evaluations[node_id] = {
                'reputation': reputation,
                'status': decision.status,
                'action': decision.action,
                'shard': decision.shard,
                'features': features
            }
            
            results['evaluations'][node_id] = node_evaluations[node_id]
            results['mitigation_actions'][node_id] = {
                'status': decision.status,
                'action': decision.action,
                'shard': decision.shard
            }
        
        # Consensus decision based on majority of HEALTHY nodes
        healthy_nodes = [nid for nid, eval in node_evaluations.items() if eval['status'] == 'HEALTHY']
        total_nodes = len(node_evaluations)
        
        if total_nodes > 0:
            healthy_ratio = len(healthy_nodes) / total_nodes
            consensus_decision = {
                'majority_healthy': healthy_ratio > 0.5,
                'healthy_nodes': healthy_nodes,
                'total_nodes': total_nodes,
                'healthy_ratio': healthy_ratio
            }
            results['consensus_decision'] = consensus_decision
        
        # Summary statistics
        status_counts = {}
        for evaluation in node_evaluations.values():
            status = evaluation['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        results['summary'] = {
            'total_evaluated': total_nodes,
            'status_distribution': status_counts,
            'average_reputation': np.mean([eval['reputation'] for eval in node_evaluations.values()]) if node_evaluations else 0.0
        }
        
        # Store consensus results
        self.consensus_decisions[epoch_id] = results
        
        return results
    
    def get_node_status(self, node_id: str) -> Optional[Dict]:
        """Get current status of a node"""
        if node_id not in self.mitigation_actions:
            return None
        
        decision = self.mitigation_actions[node_id]
        return {
            'node_id': node_id,
            'reputation': self.reputation.get(node_id, 0.0),
            'ewma_reputation': self.ewma_reputations.get(node_id, 0.0),
            'status': decision.status,
            'action': decision.action,
            'shard': decision.shard,
            'history_length': len(self.reputation_history.get(node_id, []))
        }
    
    def get_all_nodes_status(self) -> Dict:
        """Get status of all evaluated nodes"""
        return {
            node_id: self.get_node_status(node_id)
            for node_id in self.mitigation_actions.keys()
        }
    
    def get_shard_distribution(self) -> Dict:
        """Get distribution of nodes across shards"""
        shard_counts = defaultdict(int)
        for decision in self.mitigation_actions.values():
            shard_counts[decision.shard] += 1
        
        return dict(shard_counts)
