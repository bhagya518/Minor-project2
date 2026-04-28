"""
Advanced ML Ensemble Training for Malicious Node Detection
Implements 3-model ensemble + Meta Learner as per proposed architecture:
1. Random Forest (supervised)
2. Isolation Forest (unsupervised) 
3. Graph Anomaly (network-based)
4. Gradient Boosting (meta learner)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
import joblib
import os
import logging
from datetime import datetime
import networkx as nx
from scipy.spatial.distance import pdist, squareform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnsembleNodeClassifier:
    """
    3-Model Ensemble + Meta Learner for malicious node detection
    Matches the proposed architecture with 20 engineered features
    """
    
    def __init__(self, data_path, model_output_path):
        self.data_path = data_path
        self.model_output_path = model_output_path
        self.scaler = StandardScaler()
        
        # 20 features from proposed architecture
        self.feature_columns = [
            'accuracy',              # F01 - accuracy rate
            'false_positive_rate',   # F02 - false positive rate
            'false_negative_rate',   # F03 - false negative rate  
            'avg_rt_error',          # F04 - response time deviation
            'max_rt_error',          # F05 - max response time error
            'peer_agreement_rate',   # F06 - agreement with peers
            'historical_accuracy',   # F07 - historical accuracy
            'accuracy_std_dev',      # F08 - accuracy consistency
            'report_consistency',    # F09 - report consistency
            'sudden_change_score',   # F10 - sudden behavior change
            'ssl_accuracy',          # F11 - SSL detection accuracy
            'uptime_deviation',      # F12 - uptime deviation
            'rt_consistency'         # F13 - response time consistency
        ]
        
        # Models
        self.rf_model = None          # Random Forest (supervised)
        self.iso_model = None         # Isolation Forest (unsupervised)
        self.graph_model = None       # Graph Anomaly (network-based)
        self.meta_model = None        # Gradient Boosting (meta learner)
        
        # Training metrics
        self.metrics = {}
        
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
            y = df['is_malicious']
            
            logger.info(f"Features shape: {X.shape}")
            logger.info(f"Target distribution: {y.value_counts().to_dict()}")
            
            return X, y, df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def engineer_graph_features(self, X, df):
        """
        Engineer graph-based features for anomaly detection
        Implements F14-F20 from proposed architecture
        Optimized for large datasets
        """
        logger.info("Engineering graph-based features...")
        
        # Sample data for graph computation to handle large datasets
        max_samples = 2000  # Limit for efficient graph computation
        if len(X) > max_samples:
            logger.info(f"Sampling {max_samples} samples from {len(X)} for graph features")
            sample_indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X.iloc[sample_indices]
            df_sample = df.iloc[sample_indices]
        else:
            X_sample = X
            df_sample = df
            sample_indices = range(len(X))
        
        feature_matrix = X_sample.values
        
        # F18: Local outlier factor (more efficient than full graph)
        logger.info("Computing Local Outlier Factor...")
        lof = LocalOutlierFactor(n_neighbors=min(20, len(feature_matrix)//4), novelty=True)
        lof.fit(feature_matrix)
        lof_scores = lof.decision_function(feature_matrix)
        
        # F14-F17: Simplified graph features using k-nearest neighbors
        logger.info("Computing neighborhood-based features...")
        from sklearn.neighbors import NearestNeighbors
        
        n_neighbors = min(10, len(feature_matrix)//2)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(feature_matrix)
        distances, indices = nbrs.kneighbors(feature_matrix)
        
        # Compute graph-like features
        graph_features_sample = []
        for i in range(len(feature_matrix)):
            # F14: Local clustering (simplified)
            neighbor_indices = indices[i][1:]  # Exclude self
            local_density = np.mean(distances[i][1:])
            clustering_score = 1.0 / (1.0 + local_density)  # Higher = more clustered
            
            # F15: Degree centrality (simplified)
            degree_score = n_neighbors / len(feature_matrix)
            
            # F16: Betweenness (simplified - based on distance to neighbors)
            betweenness_score = np.mean(distances[i][1:])
            
            # F17: Closeness (simplified)
            closeness_score = 1.0 / (1.0 + np.mean(distances[i]))
            
            # F19: Agreement with majority (from peer_agreement_rate)
            majority_agreement = df_sample.iloc[i]['peer_agreement_rate']
            
            # F20: Epoch consistency (from accuracy_std_dev)
            epoch_consistency = 1 - df_sample.iloc[i]['accuracy_std_dev']
            
            graph_features_sample.append([
                clustering_score, degree_score, betweenness_score, closeness_score,
                lof_scores[i], majority_agreement, epoch_consistency
            ])
        
        graph_features_sample = np.array(graph_features_sample)
        
        # If we sampled, map back to original dataset
        if len(X) > max_samples:
            logger.info("Mapping graph features back to full dataset...")
            graph_features = np.zeros((len(X), 7))
            
            # Use k-NN to map features back
            nbrs_full = NearestNeighbors(n_neighbors=1).fit(X_sample)
            _, indices_full = nbrs_full.kneighbors(X)
            
            for i in range(len(X)):
                nearest_sample = indices_full[i][0]
                # Ensure index is within bounds
                nearest_sample = min(nearest_sample, len(graph_features_sample) - 1)
                graph_features[i] = graph_features_sample[nearest_sample]
        else:
            graph_features = graph_features_sample
        
        logger.info(f"Graph features shape: {graph_features.shape}")
        return graph_features
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest (supervised) model"""
        logger.info("Training Random Forest model...")
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.rf_model.fit(X_train, y_train)
        
        # Feature importance
        feature_importance = dict(zip(self.feature_columns, self.rf_model.feature_importances_))
        logger.info("Random Forest feature importances:")
        for feat, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  {feat}: {imp:.4f}")
    
    def train_isolation_forest(self, X_train):
        """Train Isolation Forest (unsupervised) model"""
        logger.info("Training Isolation Forest model...")
        
        self.iso_model = IsolationForest(
            contamination=0.2,  # Expected 20% malicious nodes
            random_state=42,
            n_estimators=100
        )
        
        self.iso_model.fit(X_train)
        logger.info("Isolation Forest trained successfully")
    
    def train_graph_anomaly(self, X_train, df_train):
        """Train Graph Anomaly (network-based) model"""
        logger.info("Training Graph Anomaly model...")
        
        # Use DBSCAN for clustering-based anomaly detection
        self.graph_model = DBSCAN(
            eps=0.5,
            min_samples=5,
            metric='euclidean'
        )
        
        self.graph_model.fit(X_train)
        
        # Count clusters and noise points
        labels = self.graph_model.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"Graph anomaly detected {n_clusters} clusters, {n_noise} noise points")
        
        # Store training data for prediction
        self.graph_training_data = X_train
    
    def get_model_predictions(self, X):
        """Get predictions from all three models"""
        predictions = {}
        
        # Random Forest predictions (probability)
        rf_proba = self.rf_model.predict_proba(X)[:, 1]  # Probability of malicious
        predictions['rf'] = rf_proba
        
        # Isolation Forest predictions (anomaly scores)
        iso_scores = self.iso_model.decision_function(X)
        # Convert to probability (lower score = more anomalous)
        iso_proba = 1 - (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
        predictions['iso'] = iso_proba
        
        # Graph anomaly predictions (based on cluster membership)
        # For new data, we need to predict using the trained model
        try:
            graph_labels = self.graph_model.fit_predict(X)
            graph_proba = np.where(graph_labels == -1, 0.8, 0.2)  # High prob for noise points
        except:
            # If prediction fails, use default values
            graph_proba = np.full(len(X), 0.2)  # Default to low probability
        predictions['graph'] = graph_proba
        
        return predictions
    
    def train_meta_learner(self, X_train, y_train):
        """Train Gradient Boosting meta learner"""
        logger.info("Training Gradient Boosting meta learner...")
        
        # Get predictions from base models
        base_predictions = self.get_model_predictions(X_train)
        
        # Create meta features
        meta_features = np.column_stack([
            base_predictions['rf'],
            base_predictions['iso'], 
            base_predictions['graph']
        ])
        
        # Train meta learner
        self.meta_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        self.meta_model.fit(meta_features, y_train)
        logger.info("Meta learner trained successfully")
    
    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate the complete ensemble"""
        logger.info("Evaluating ensemble performance...")
        
        # Get base model predictions
        base_predictions = self.get_model_predictions(X_test)
        
        # Create meta features for test set
        meta_features = np.column_stack([
            base_predictions['rf'],
            base_predictions['iso'],
            base_predictions['graph']
        ])
        
        # Meta learner predictions
        y_pred = self.meta_model.predict(meta_features)
        y_proba = self.meta_model.predict_proba(meta_features)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_proba)
        
        # Detailed classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        self.metrics = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'base_model_performance': {
                'rf_accuracy': accuracy_score(y_test, (base_predictions['rf'] > 0.5).astype(int)),
                'iso_accuracy': accuracy_score(y_test, (base_predictions['iso'] > 0.5).astype(int)),
                'graph_accuracy': accuracy_score(y_test, (base_predictions['graph'] > 0.5).astype(int))
            }
        }
        
        logger.info(f"Ensemble Accuracy: {accuracy:.4f}")
        logger.info(f"Ensemble AUC: {auc_score:.4f}")
        logger.info(f"Confusion Matrix: {conf_matrix.tolist()}")
        
        return self.metrics
    
    def save_models(self):
        """Save all trained models"""
        os.makedirs(self.model_output_path, exist_ok=True)
        
        # Save models
        joblib.dump(self.rf_model, os.path.join(self.model_output_path, 'random_forest.pkl'))
        joblib.dump(self.iso_model, os.path.join(self.model_output_path, 'isolation_forest.pkl'))
        joblib.dump(self.graph_model, os.path.join(self.model_output_path, 'graph_anomaly.pkl'))
        joblib.dump(self.meta_model, os.path.join(self.model_output_path, 'meta_learner.pkl'))
        joblib.dump(self.scaler, os.path.join(self.model_output_path, 'scaler.pkl'))
        
        # Save feature list and metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'model_type': 'ensemble_3_model',
            'training_date': datetime.now().isoformat(),
            'metrics': self.metrics
        }
        
        joblib.dump(metadata, os.path.join(self.model_output_path, 'metadata.pkl'))
        
        logger.info(f"All models saved to {self.model_output_path}")
    
    def run_training(self):
        """Run complete training pipeline"""
        try:
            logger.info("Starting ensemble training pipeline...")
            
            # Load data
            X, y, df = self.load_data()
            
            # Add graph features
            graph_features = self.engineer_graph_features(X, df)
            
            # Combine original and graph features
            X_enhanced = np.column_stack([X, graph_features])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_enhanced, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train base models
            self.train_random_forest(X_train_scaled, y_train)
            self.train_isolation_forest(X_train_scaled)
            self.train_graph_anomaly(X_train_scaled, df.iloc[:len(X_train)])
            
            # Train meta learner
            self.train_meta_learner(X_train_scaled, y_train)
            
            # Evaluate ensemble
            metrics = self.evaluate_ensemble(X_test_scaled, y_test)
            
            # Save models
            self.save_models()
            
            logger.info("Ensemble training completed successfully!")
            return metrics
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise

def main():
    """Main function to run the ensemble training"""
    # Update dataset path to use your generated dataset
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'dataset (1).csv')
    model_output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    
    logger.info("Starting Advanced ML Ensemble Training for Malicious Node Detection")
    logger.info(f"Using dataset: {data_path}")
    
    # Initialize trainer
    trainer = EnsembleNodeClassifier(data_path, model_output_path)
    
    # Run training
    metrics = trainer.run_training()
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Final Ensemble Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Final Ensemble AUC: {metrics['auc_score']:.4f}")
    logger.info("Base Model Performance:")
    for model, acc in metrics['base_model_performance'].items():
        logger.info(f"  {model}: {acc:.4f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
