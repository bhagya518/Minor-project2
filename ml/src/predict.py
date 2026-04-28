"""
ML Model Inference for Real-time Malicious Node Detection
Loads trained model and provides prediction API
"""

import joblib
import numpy as np
import pandas as pd
import os
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NodeClassifier:
    """ML model for classifying nodes as honest or malicious"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the classifier with trained model
        
        Args:
            model_path: Path to the trained model directory
        """
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.model_metadata = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model, scaler, and metadata"""
        try:
            logger.info(f"Loading model from {model_path}")
            
            # Load model
            model_file = os.path.join(model_path, 'model.pkl')
            self.model = joblib.load(model_file)
            logger.info("Model loaded successfully")
            
            # Load scaler
            scaler_file = os.path.join(model_path, 'scaler.pkl')
            self.scaler = joblib.load(scaler_file)
            logger.info("Scaler loaded successfully")
            
            # Load feature columns
            features_file = os.path.join(model_path, 'features.pkl')
            self.feature_columns = joblib.load(features_file)
            logger.info(f"Feature columns loaded: {self.feature_columns}")
            
            # Load metadata if available
            metadata_file = os.path.join(model_path, 'metadata.json')
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info("Metadata loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def validate_features(self, features: Dict) -> bool:
        """Validate that all required features are present"""
        missing_features = set(self.feature_columns) - set(features.keys())
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return False
        return True
    
    def preprocess_features(self, features: Dict) -> np.ndarray:
        """Preprocess features for prediction"""
        try:
            # Convert to DataFrame with correct column order
            feature_df = pd.DataFrame([features], columns=self.feature_columns)
            
            # Scale features
            scaled_features = self.scaler.transform(feature_df)
            
            return scaled_features
            
        except Exception as e:
            logger.error(f"Error preprocessing features: {e}")
            raise
    
    def predict_single(self, features: Dict) -> Dict:
        """
        Predict if a node is malicious (1) or honest (0)
        
        Args:
            features: Dictionary containing node features
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if not self.model or not self.scaler:
                raise ValueError("Model not loaded. Call load_model() first.")
            
            # Validate features
            if not self.validate_features(features):
                raise ValueError("Invalid or missing features")
            
            # Preprocess features
            processed_features = self.preprocess_features(features)
            
            # Make prediction
            prediction = self.model.predict(processed_features)[0]
            prediction_proba = self.model.predict_proba(processed_features)[0]
            
            # Prepare result
            result = {
                'prediction': int(prediction),  # 0: Honest, 1: Malicious
                'prediction_label': 'Malicious' if prediction == 1 else 'Honest',
                'confidence': float(max(prediction_proba)),
                'honest_probability': float(prediction_proba[0]),
                'malicious_probability': float(prediction_proba[1]),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Prediction completed: {result['prediction_label']} "
                       f"(confidence: {result['confidence']:.4f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def predict_batch(self, features_list: List[Dict]) -> List[Dict]:
        """
        Predict multiple nodes at once
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of prediction results
        """
        try:
            results = []
            
            for i, features in enumerate(features_list):
                try:
                    result = self.predict_single(features)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error predicting node {i}: {e}")
                    # Add error result
                    results.append({
                        'prediction': -1,
                        'prediction_label': 'Error',
                        'confidence': 0.0,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        importance_dict = dict(zip(self.feature_columns, self.model.feature_importances_))
        return importance_dict
    
    def calculate_ml_score(self, features: Dict) -> float:
        """
        Calculate ML score (0-1) for PoR calculation
        Higher score means more trustworthy (honest nodes have higher scores)
        
        Args:
            features: Node features
            
        Returns:
            ML score between 0 and 1
        """
        try:
            prediction_result = self.predict_single(features)
            
            # If predicted as honest, use honest probability as score
            # If predicted as malicious, use low score
            if prediction_result['prediction'] == 0:  # Honest
                ml_score = prediction_result['honest_probability']
            else:  # Malicious
                ml_score = 1.0 - prediction_result['malicious_probability']
            
            return float(ml_score)
            
        except Exception as e:
            logger.error(f"Error calculating ML score: {e}")
            return 0.5  # Default to neutral score

# Singleton instance for global use
_classifier_instance = None

def get_classifier(model_path: str = None) -> NodeClassifier:
    """Get or create classifier instance"""
    global _classifier_instance
    
    if _classifier_instance is None:
        if model_path is None:
            # Default model path
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        
        _classifier_instance = NodeClassifier(model_path)
    
    return _classifier_instance

def predict_node_malicious(features: Dict, model_path: str = None) -> Dict:
    """
    Convenience function for single node prediction
    
    Args:
        features: Node features dictionary
        model_path: Path to model (optional)
        
    Returns:
        Prediction result dictionary
    """
    classifier = get_classifier(model_path)
    return classifier.predict_single(features)

def calculate_ml_score_for_por(features: Dict, model_path: str = None) -> float:
    """
    Convenience function to calculate ML score for PoR
    
    Args:
        features: Node features dictionary
        model_path: Path to model (optional)
        
    Returns:
        ML score (0-1)
    """
    classifier = get_classifier(model_path)
    return classifier.calculate_ml_score(features)

if __name__ == "__main__":
    # Test the classifier with sample data
    import json
    
    # Sample features for testing
    sample_features_honest = {
        'avg_response_ms': 180.0,
        'ssl_valid_rate': 1.0,
        'content_match_rate': 0.95,
        'stale_report_rate': 0.0,
        'false_report_rate': 0.05
    }
    
    sample_features_malicious = {
        'avg_response_ms': 120.0,
        'ssl_valid_rate': 0.7,
        'content_match_rate': 0.5,
        'stale_report_rate': 0.2,
        'false_report_rate': 0.5
    }
    
    try:
        # Initialize classifier
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        classifier = NodeClassifier(model_path)
        
        # Test honest node
        result_honest = classifier.predict_single(sample_features_honest)
        print("Honest Node Prediction:")
        print(json.dumps(result_honest, indent=2))
        
        # Test malicious node
        result_malicious = classifier.predict_single(sample_features_malicious)
        print("\nMalicious Node Prediction:")
        print(json.dumps(result_malicious, indent=2))
        
        # Test ML score calculation
        ml_score_honest = classifier.calculate_ml_score(sample_features_honest)
        ml_score_malicious = classifier.calculate_ml_score(sample_features_malicious)
        
        print(f"\nML Score (Honest): {ml_score_honest:.4f}")
        print(f"ML Score (Malicious): {ml_score_malicious:.4f}")
        
    except Exception as e:
        logger.error(f"Error in testing: {e}")
        print(f"Make sure to train the model first using train_model.py")
