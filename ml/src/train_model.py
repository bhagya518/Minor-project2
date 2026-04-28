"""
ML Model Training Script for Malicious Node Detection
Uses Random Forest classifier to detect malicious vs honest nodes
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NodeClassifierTrainer:
    def __init__(self, data_path, model_output_path):
        self.data_path = data_path
        self.model_output_path = model_output_path
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = [
            'avg_response_ms',
            'ssl_valid_rate', 
            'content_match_rate',
            'stale_report_rate',
            'false_report_rate'
        ]
        
    def load_data(self):
        """Load and preprocess the dataset"""
        try:
            logger.info(f"Loading dataset from {self.data_path}")
            df = pd.read_csv(self.data_path)
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                logger.warning(f"Missing values found: {missing_values}")
                df = df.fillna(df.mean())
            
            # Extract features and target
            X = df[self.feature_columns]
            y = df['label']
            
            logger.info(f"Features shape: {X.shape}")
            logger.info(f"Target distribution: {y.value_counts().to_dict()}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def train_model(self, X_train, y_train):
        """Train Random Forest classifier"""
        logger.info("Training Random Forest model...")
        
        # Initialize Random Forest with optimized parameters
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        logger.info("Model training completed")
        
        # Feature importance
        feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        logger.info(f"Feature importance: {feature_importance}")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        logger.info("Evaluating model performance...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"AUC Score: {auc_score:.4f}")
        
        # Detailed classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=['Honest', 'Malicious']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'classification_report': classification_report(y_test, y_pred, target_names=['Honest', 'Malicious']),
            'confusion_matrix': cm
        }
    
    def save_model(self):
        """Save trained model and scaler"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            
            # Save model
            model_path = os.path.join(self.model_output_path, 'model.pkl')
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save scaler
            scaler_path = os.path.join(self.model_output_path, 'scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
            
            # Save feature columns
            features_path = os.path.join(self.model_output_path, 'features.pkl')
            joblib.dump(self.feature_columns, features_path)
            logger.info(f"Feature columns saved to {features_path}")
            
            # Save metadata
            metadata = {
                'model_type': 'RandomForestClassifier',
                'features': self.feature_columns,
                'training_date': datetime.now().isoformat(),
                'model_params': self.model.get_params()
            }
            
            metadata_path = os.path.join(self.model_output_path, 'metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def run_training(self):
        """Complete training pipeline"""
        try:
            # Load data
            X, y = self.load_data()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logger.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
            
            # Preprocess features
            X_train_scaled, X_test_scaled = self.preprocess_features(X_train, X_test)
            
            # Train model
            self.train_model(X_train_scaled, y_train)
            
            # Evaluate model
            metrics = self.evaluate_model(X_test_scaled, y_test)
            
            # Save model
            self.save_model()
            
            logger.info("Training pipeline completed successfully!")
            return metrics
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise

def main():
    """Main function to run the training"""
    # Paths
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'web_monitor_dataset.csv')
    model_output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    
    logger.info("Starting ML model training for malicious node detection")
    
    # Initialize trainer
    trainer = NodeClassifierTrainer(data_path, model_output_path)
    
    # Run training
    metrics = trainer.run_training()
    
    logger.info("Training completed successfully!")
    logger.info(f"Final model accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Final model AUC: {metrics['auc_score']:.4f}")

if __name__ == "__main__":
    main()
