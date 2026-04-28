"""
Enhanced ML Ensemble Detector
Implements RandomForest + IsolationForest + Graph Analysis with Bayesian Stacking
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import logging
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class EnsembleDetector:
    """
    Advanced ML Ensemble for malicious node detection
    - RandomForest (supervised)
    - IsolationForest (unsupervised)
    - Graph Analysis (network-based)
    - Bayesian Stacking with GradientBoosting meta-learner
    """
    
    def __init__(self, contamination: float = 0.15, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Base detectors
        self.rf_model = None
        self.iso_model = None
        self.meta_model = None
        
        # Graph analysis components
        self.graph = None
        self.pagerank_scores = {}
        self.clustering_scores = {}
        self.degree_scores = {}
        
        # Training status
        self.is_trained = False
        self.has_labels = False
        
        logger.info("EnsembleDetector initialized")
    
    def _build_graph(self, df: pd.DataFrame, node_col: str = 'node_id') -> nx.DiGraph:
        """Build transaction graph from monitoring data - O(n) complexity"""
        logger.info("Building transaction graph...")
        
        # Create directed graph
        G = nx.DiGraph()
        nodes = df[node_col].tolist()
        G.add_nodes_from(nodes)
        
        # Create node lookup dictionary ONCE (O(n) instead of O(n²))
        node_map = df.set_index(node_col).to_dict('index')
        
        # Identify potential malicious nodes (if labels available)
        if 'label' in df.columns:
            malicious_nodes = set(df[df['label'] == 1][node_col].tolist())
        else:
            # Use heuristic for malicious nodes (high response times, low SSL rates)
            malicious_nodes = set()
            for _, row in df.iterrows():
                if (row.get('avg_response_ms', 0) > 300 or 
                    row.get('ssl_valid_rate', 1.0) < 0.7 or
                    row.get('false_report_rate', 0) > 0.3):
                    malicious_nodes.add(row[node_col])
        
        # Create edges based on monitoring similarity - OPTIMIZED O(n²) with early termination
        edges_added = 0
        similarity_threshold = 0.3
        
        # Pre-calculate node features for faster comparison
        node_features = {}
        for node_id in nodes:
            node_features[node_id] = node_map.get(node_id)
        
        for i, node1 in enumerate(nodes):
            row1 = node_features[node1]
            if row1 is None:
                continue
                
            # Only compare with subset of nodes for efficiency
            max_comparisons = min(50, len(nodes) - i - 1)  # Limit comparisons
            for j in range(i + 1, min(i + 1 + max_comparisons, len(nodes))):
                node2 = nodes[j]
                row2 = node_features[node2]
                if row2 is None:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_similarity(row1, row2)
                
                if similarity > similarity_threshold:
                    # Boost weight if either node is malicious
                    weight = similarity * 10
                    if node1 in malicious_nodes or node2 in malicious_nodes:
                        weight *= 1.5
                    G.add_edge(node1, node2, weight=weight)
                    edges_added += 1
        
        logger.info(f"Graph built: {G.number_of_nodes()} nodes, {edges_added} edges (optimized)")
        return G
    
    def _calculate_similarity(self, row1: pd.Series, row2: pd.Series) -> float:
        """Calculate similarity between two nodes based on monitoring features"""
        features = ['avg_response_ms', 'ssl_valid_rate', 'content_match_rate', 
                   'stale_report_rate', 'false_report_rate']
        
        distances = []
        for feature in features:
            val1 = row1.get(feature, 0)
            val2 = row2.get(feature, 0)
            
            # Normalize and calculate distance
            if feature in ['ssl_valid_rate', 'content_match_rate']:
                # Higher is better for these features
                dist = abs(val1 - val2)
            else:
                # Lower is better for these features
                dist = abs(val1 - val2)
            
            distances.append(dist)
        
        # Convert distances to similarity (inverse)
        avg_distance = np.mean(distances)
        similarity = 1.0 / (1.0 + avg_distance)
        
        return similarity
    
    def _extract_graph_features(self, df: pd.DataFrame, node_col: str = 'node_id') -> pd.DataFrame:
        """Extract graph-based features"""
        logger.info("Extracting graph features...")
        
        if self.graph is None:
            self.graph = self._build_graph(df, node_col)
        
        # Calculate graph metrics
        in_degrees = dict(self.graph.in_degree(weight='weight'))
        out_degrees = dict(self.graph.out_degree(weight='weight'))
        pagerank = nx.pagerank(self.graph, weight='weight')
        
        # Clustering coefficient (on undirected version)
        undirected = self.graph.to_undirected()
        clustering = nx.clustering(undirected, weight='weight')
        
        # Add graph features to dataframe
        df_copy = df.copy()
        df_copy['in_degree'] = df_copy[node_col].map(lambda x: in_degrees.get(x, 0))
        df_copy['out_degree'] = df_copy[node_col].map(lambda x: out_degrees.get(x, 0))
        df_copy['pagerank'] = df_copy[node_col].map(lambda x: pagerank.get(x, 0))
        df_copy['clustering'] = df_copy[node_col].map(lambda x: clustering.get(x, 0))
        
        # Store for later use
        self.pagerank_scores = pagerank
        self.clustering_scores = clustering
        self.degree_scores = {'in': in_degrees, 'out': out_degrees}
        
        logger.info("Graph features extracted")
        return df_copy
    
    def _calculate_graph_anomaly_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate anomaly scores based on graph metrics"""
        logger.info("Calculating graph anomaly scores...")
        
        # Z-score based anomaly detection for graph metrics
        pagerank_z = np.abs((df['pagerank'] - df['pagerank'].mean()) / 
                           (df['pagerank'].std() + 1e-9))
        
        total_degree = df['in_degree'] + df['out_degree']
        degree_z = np.abs((total_degree - total_degree.mean()) / 
                          (total_degree.std() + 1e-9))
        
        clustering_z = np.abs((df['clustering'] - df['clustering'].mean()) / 
                             (df['clustering'].std() + 1e-9))
        
        # Combine graph anomaly scores
        graph_score = (pagerank_z + degree_z + clustering_z) / 3.0
        
        # Normalize to [0, 1]
        graph_score_norm = (graph_score - graph_score.min()) / \
                          (graph_score.max() - graph_score.min() + 1e-9)
        
        logger.info("Graph anomaly scores calculated")
        return graph_score_norm
    
    def fit(self, df: pd.DataFrame, node_col: str = 'node_id') -> Dict:
        """Train the ensemble detector"""
        logger.info("Training ensemble detector...")
        
        # Check for labels
        self.has_labels = 'label' in df.columns and df['label'].nunique() > 1
        
        if self.has_labels:
            logger.info("Labels detected - using supervised training")
        else:
            logger.info("No labels detected - using unsupervised approach")
        
        # Extract graph features
        df_with_graph = self._extract_graph_features(df, node_col)
        
        # Define feature columns
        feature_cols = ['avg_response_ms', 'ssl_valid_rate', 'content_match_rate',
                       'stale_report_rate', 'false_report_rate', 'in_degree',
                       'out_degree', 'pagerank', 'clustering']
        
        # Prepare features
        X = df_with_graph[feature_cols].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)
        
        # Train RandomForest (if labels available)
        if self.has_labels:
            y = df_with_graph['label'].values
            self.rf_model = RandomForestClassifier(
                n_estimators=200, 
                random_state=self.random_state
            )
            self.rf_model.fit(X_scaled, y)
            rf_probs = self.rf_model.predict_proba(X_scaled)[:, 1]
            logger.info("RandomForest trained")
        else:
            rf_probs = np.zeros(len(df))
        
        # Train IsolationForest
        self.iso_model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state
        )
        self.iso_model.fit(X_scaled)
        iso_scores = -self.iso_model.decision_function(X_scaled)
        iso_norm = (iso_scores - iso_scores.min()) / \
                  (iso_scores.max() - iso_scores.min() + 1e-9)
        logger.info("IsolationForest trained")
        
        # Calculate graph anomaly scores
        graph_scores = self._calculate_graph_anomaly_scores(df_with_graph)
        
        # Prepare stacking features
        stack_X = np.column_stack([rf_probs, iso_norm, graph_scores])
        
        # Train meta-learner
        if self.has_labels:
            self.meta_model = GradientBoostingClassifier(
                random_state=self.random_state
            )
            self.meta_model.fit(stack_X, y)
            p_malicious = self.meta_model.predict_proba(stack_X)[:, 1]
            logger.info("Meta-learner trained")
        else:
            # Use weighted fusion for unsupervised case
            w_rf, w_iso, w_graph = 0.0, 0.5, 0.5  # No RF without labels
            p_malicious = np.clip(
                w_rf * rf_probs + w_iso * iso_norm + w_graph * graph_scores,
                1e-6, 1 - 1e-6
            )
        
        # Store training results
        self.training_results = {
            'rf_probs': rf_probs,
            'iso_scores': iso_norm,
            'graph_scores': graph_scores,
            'p_malicious': p_malicious,
            'feature_importance': self._get_feature_importance(),
            'training_timestamp': datetime.now().isoformat()
        }
        
        self.is_trained = True
        logger.info("Ensemble detector training completed")
        
        return self.training_results
    
    def _get_feature_importance(self) -> Dict:
        """Get feature importance from trained models"""
        importance = {}
        
        if self.rf_model:
            feature_names = ['avg_response_ms', 'ssl_valid_rate', 'content_match_rate',
                           'stale_report_rate', 'false_report_rate', 'in_degree',
                           'out_degree', 'pagerank', 'clustering']
            importance['random_forest'] = dict(zip(feature_names, 
                                                 self.rf_model.feature_importances_))
        
        if self.meta_model:
            stack_features = ['rf_prob', 'iso_score', 'graph_score']
            importance['meta_learner'] = dict(zip(stack_features,
                                                 self.meta_model.feature_importances_))
        
        return importance
    
    def predict(self, df: pd.DataFrame, node_col: str = 'node_id') -> Dict:
        """Predict malicious probabilities for new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        logger.info("Making predictions...")
        
        # Extract graph features
        df_with_graph = self._extract_graph_features(df, node_col)
        
        # Prepare features
        feature_cols = ['avg_response_ms', 'ssl_valid_rate', 'content_match_rate',
                       'stale_report_rate', 'false_report_rate', 'in_degree',
                       'out_degree', 'pagerank', 'clustering']
        
        X = df_with_graph[feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)
        
        # Get base model predictions
        if self.rf_model:
            rf_probs = self.rf_model.predict_proba(X_scaled)[:, 1]
        else:
            rf_probs = np.zeros(len(df))
        
        iso_scores = -self.iso_model.decision_function(X_scaled)
        iso_norm = (iso_scores - iso_scores.min()) / \
                  (iso_scores.max() - iso_scores.min() + 1e-9)
        
        graph_scores = self._calculate_graph_anomaly_scores(df_with_graph)
        
        # Stack features
        stack_X = np.column_stack([rf_probs, iso_norm, graph_scores])
        
        # Get final predictions
        if self.meta_model:
            p_malicious = self.meta_model.predict_proba(stack_X)[:, 1]
        else:
            w_rf, w_iso, w_graph = 0.0, 0.5, 0.5
            p_malicious = np.clip(
                w_rf * rf_probs + w_iso * iso_norm + w_graph * graph_scores,
                1e-6, 1 - 1e-6
            )
        
        # Prepare results
        results = {
            'node_ids': df[node_col].tolist(),
            'p_malicious': p_malicious,
            'rf_probs': rf_probs,
            'iso_scores': iso_norm,
            'graph_scores': graph_scores,
            'predictions': (p_malicious >= 0.5).astype(int),
            'prediction_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Predictions completed for {len(df)} nodes")
        return results
    
    def evaluate(self, df: pd.DataFrame, node_col: str = 'node_id') -> Dict:
        """Evaluate model performance if labels are available"""
        if not self.has_labels:
            logger.warning("No labels available for evaluation")
            return {}
        
        predictions = self.predict(df, node_col)
        y_true = df['label'].values
        y_pred = predictions['predictions']
        y_prob = predictions['p_malicious']
        
        metrics = {
            'precision': precision_recall_fscore_support(y_true, y_pred, average='binary')[0],
            'recall': precision_recall_fscore_support(y_true, y_pred, average='binary')[1],
            'f1': precision_recall_fscore_support(y_true, y_pred, average='binary')[2],
            'accuracy': accuracy_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_prob)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'scaler': self.scaler,
            'rf_model': self.rf_model,
            'iso_model': self.iso_model,
            'meta_model': self.meta_model,
            'graph_data': {
                'pagerank': self.pagerank_scores,
                'clustering': self.clustering_scores,
                'degree': self.degree_scores
            },
            'training_results': self.training_results,
            'is_trained': self.is_trained,
            'has_labels': self.has_labels,
            'contamination': self.contamination,
            'random_state': self.random_state
        }
        
        # For simplicity, save as pickle (in production, use joblib)
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        import pickle
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data['scaler']
        self.rf_model = model_data['rf_model']
        self.iso_model = model_data['iso_model']
        self.meta_model = model_data['meta_model']
        self.pagerank_scores = model_data['graph_data']['pagerank']
        self.clustering_scores = model_data['graph_data']['clustering']
        self.degree_scores = model_data['graph_data']['degree']
        self.training_results = model_data['training_results']
        self.is_trained = model_data['is_trained']
        self.has_labels = model_data['has_labels']
        self.contamination = model_data['contamination']
        self.random_state = model_data['random_state']
        
        logger.info(f"Model loaded from {filepath}")
