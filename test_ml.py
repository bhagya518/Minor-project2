#!/usr/bin/env python3
"""Test script to verify ML model is working correctly"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'node_service', 'src'))

from ml_consensus_engine import EnhancedMLConsensusEngine

# Initialize the ML engine
ml_engine = EnhancedMLConsensusEngine("test_node")

if ml_engine.models_loaded:
    print("✅ Models loaded successfully")
    print(f"RF features: {ml_engine.rf_feature_cols}")
    print(f"Behavioral features: {ml_engine.behavioral_cols}")
    
    # Test with sample features
    test_features = {
        "accuracy": 0.9,
        "false_positive_rate": 0.1,
        "false_negative_rate": 0.05,
        "avg_rt_error": 0.5,
        "max_rt_error": 1.0,
        "peer_agreement_rate": 0.8,
        "historical_accuracy": 0.85,
        "accuracy_std_dev": 0.1,
        "report_consistency": 0.9,
        "sudden_change_score": 0.2,
        "ssl_accuracy": 0.95,
        "uptime_deviation": 0.1,
        "rt_consistency": 0.8,
    }
    
    print(f"\nTesting with features: {test_features}")
    
    # Calculate reputation
    reputation = ml_engine.calculate_enhanced_reputation(test_features)
    print(f"\nCalculated reputation: {reputation}")
    
    if reputation != 0.5:
        print("✅ ML model is working correctly!")
    else:
        print("❌ ML model returned default 0.5 value")
else:
    print("❌ Models failed to load")
