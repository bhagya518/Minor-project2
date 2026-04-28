"""
Trust Engine Module
Calculates monitoring trust scores based on node behavior and peer feedback
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrustEngine:
    """Engine for calculating and managing node trust scores"""
    
    def __init__(self, window_size: int = 100, trust_decay_rate: float = 0.95):
        """
        Initialize trust engine
        
        Args:
            window_size: Number of recent reports to consider for trust calculation
            trust_decay_rate: Rate at which trust scores decay over time (0-1)
        """
        self.window_size = window_size
        self.trust_decay_rate = trust_decay_rate
        
        # Store node reports and trust data
        self.node_reports = defaultdict(lambda: deque(maxlen=window_size))
        self.node_trust_scores = defaultdict(float)
        self.node_last_update = defaultdict(datetime)
        
        # Peer feedback
        self.peer_feedback = defaultdict(lambda: defaultdict(list))
        
        # Consistency tracking
        self.content_hashes = defaultdict(lambda: defaultdict(list))
        
        logger.info(f"Trust engine initialized: window_size={window_size}, decay_rate={trust_decay_rate}")
    
    def add_monitoring_report(self, node_id: str, report: Dict):
        """
        Add a monitoring report from a node
        
        Args:
            node_id: Node identifier
            report: Monitoring report dictionary
        """
        try:
            # Add timestamp if not present
            if 'timestamp' not in report:
                report['timestamp'] = datetime.now().isoformat()
            
            # Store report
            self.node_reports[node_id].append(report)
            self.node_last_update[node_id] = datetime.now()
            
            # Store content hash for consistency checking
            if 'url' in report and 'content_hash' in report and report['content_hash']:
                url = report['url']
                content_hash = report['content_hash']
                self.content_hashes[url][node_id].append({
                    'hash': content_hash,
                    'timestamp': report['timestamp']
                })
            
            logger.debug(f"Added report from node {node_id}")
            
        except Exception as e:
            logger.error(f"Error adding report from node {node_id}: {e}")
    
    def add_peer_feedback(self, reporter_id: str, target_node_id: str, feedback: Dict):
        """
        Add peer feedback about a node
        
        Args:
            reporter_id: Node providing feedback
            target_node_id: Node being evaluated
            feedback: Feedback dictionary with 'trust_score' (0-1) and 'reason'
        """
        try:
            feedback['timestamp'] = datetime.now().isoformat()
            feedback['reporter_id'] = reporter_id
            
            self.peer_feedback[target_node_id][reporter_id].append(feedback)
            
            logger.debug(f"Added peer feedback from {reporter_id} about {target_node_id}")
            
        except Exception as e:
            logger.error(f"Error adding peer feedback: {e}")
    
    def calculate_monitoring_trust(self, node_id: str) -> float:
        """
        Calculate monitoring trust score for a node
        
        Args:
            node_id: Node identifier
            
        Returns:
            Trust score between 0 and 1
        """
        try:
            reports = list(self.node_reports[node_id])
            
            if not reports:
                return 0.5  # Neutral score for new nodes
            
            # Calculate various trust factors
            
            # 1. Report success rate
            success_rate = self._calculate_success_rate(reports)
            
            # 2. Response time consistency
            response_consistency = self._calculate_response_consistency(reports)
            
            # 3. Content consistency with peers
            content_consistency = self._calculate_content_consistency(node_id)
            
            # 4. Report freshness
            freshness = self._calculate_freshness(node_id)
            
            # 5. Peer feedback score
            peer_score = self._calculate_peer_score(node_id)
            
            # Combine scores with weights
            trust_score = (
                0.3 * success_rate +
                0.2 * response_consistency +
                0.25 * content_consistency +
                0.15 * freshness +
                0.1 * peer_score
            )
            
            # Apply time decay
            last_update = self.node_last_update.get(node_id, datetime.now())
            time_diff = (datetime.now() - last_update).total_seconds()
            decay_factor = self.trust_decay_rate ** (time_diff / 3600)  # Decay per hour
            
            final_score = trust_score * decay_factor
            final_score = max(0.0, min(1.0, final_score))  # Clamp to [0,1]
            
            # Update stored trust score
            self.node_trust_scores[node_id] = final_score
            
            logger.debug(f"Trust score for {node_id}: {final_score:.4f}")
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating trust for node {node_id}: {e}")
            return 0.5
    
    def _calculate_success_rate(self, reports: List[Dict]) -> float:
        """Calculate success rate of monitoring reports"""
        if not reports:
            return 0.0
        
        successful_reports = sum(1 for r in reports if r.get('status') == 'success')
        return successful_reports / len(reports)
    
    def _calculate_response_consistency(self, reports: List[Dict]) -> float:
        """Calculate consistency of response times"""
        response_times = [r.get('response_time_ms', 0) for r in reports if r.get('response_time_ms')]
        
        if len(response_times) < 2:
            return 1.0  # High consistency for single data point
        
        try:
            # Calculate coefficient of variation (lower is better)
            mean_time = statistics.mean(response_times)
            std_time = statistics.stdev(response_times)
            
            if mean_time == 0:
                return 0.0
            
            cv = std_time / mean_time
            # Convert to consistency score (lower CV = higher consistency)
            consistency_score = max(0.0, 1.0 - cv)
            
            return consistency_score
            
        except statistics.StatisticsError:
            return 0.5  # Neutral score on error
    
    def _calculate_content_consistency(self, node_id: str) -> float:
        """Calculate content consistency with peer nodes"""
        try:
            total_consistency = 0.0
            url_count = 0
            
            for url, node_hashes in self.content_hashes.items():
                if node_id not in node_hashes:
                    continue
                
                node_hashes_list = node_hashes[node_id]
                if not node_hashes_list:
                    continue
                
                # Get most recent hash from this node
                recent_node_hash = node_hashes_list[-1]['hash']
                
                # Compare with other nodes
                other_nodes_hashes = []
                for other_node, hashes in node_hashes.items():
                    if other_node != node_id and hashes:
                        other_nodes_hashes.append(hashes[-1]['hash'])
                
                if not other_nodes_hashes:
                    continue
                
                # Calculate consistency (how many nodes agree)
                matching_hashes = sum(1 for h in other_nodes_hashes if h == recent_node_hash)
                consistency_rate = matching_hashes / len(other_nodes_hashes)
                
                total_consistency += consistency_rate
                url_count += 1
            
            if url_count == 0:
                return 1.0  # High consistency if no comparison possible
            
            return total_consistency / url_count
            
        except Exception as e:
            logger.error(f"Error calculating content consistency: {e}")
            return 0.5
    
    def _calculate_freshness(self, node_id: str) -> float:
        """Calculate freshness of reports"""
        last_update = self.node_last_update.get(node_id)
        
        if not last_update:
            return 0.0
        
        time_diff = (datetime.now() - last_update).total_seconds()
        
        # Freshness score decreases with time
        # 1.0 for recent reports (< 1 min), 0.0 for very old (> 1 hour)
        if time_diff < 60:
            return 1.0
        elif time_diff < 3600:
            return 1.0 - (time_diff - 60) / 3540
        else:
            return 0.0
    
    def _calculate_peer_score(self, node_id: str) -> float:
        """Calculate average peer feedback score"""
        try:
            all_feedback = []
            
            for reporter_id, feedback_list in self.peer_feedback[node_id].items():
                for feedback in feedback_list:
                    all_feedback.append(feedback.get('trust_score', 0.5))
            
            if not all_feedback:
                return 0.5  # Neutral score if no feedback
            
            return statistics.mean(all_feedback)
            
        except Exception as e:
            logger.error(f"Error calculating peer score: {e}")
            return 0.5
    
    def get_node_trust_info(self, node_id: str) -> Dict:
        """
        Get detailed trust information for a node
        
        Args:
            node_id: Node identifier
            
        Returns:
            Dictionary with trust details
        """
        try:
            reports = list(self.node_reports[node_id])
            
            # Calculate all components
            success_rate = self._calculate_success_rate(reports)
            response_consistency = self._calculate_response_consistency(reports)
            content_consistency = self._calculate_content_consistency(node_id)
            freshness = self._calculate_freshness(node_id)
            peer_score = self._calculate_peer_score(node_id)
            
            trust_score = self.calculate_monitoring_trust(node_id)
            
            return {
                'node_id': node_id,
                'trust_score': trust_score,
                'components': {
                    'success_rate': success_rate,
                    'response_consistency': response_consistency,
                    'content_consistency': content_consistency,
                    'freshness': freshness,
                    'peer_score': peer_score
                },
                'report_count': len(reports),
                'last_update': self.node_last_update.get(node_id, datetime.now()).isoformat(),
                'peer_feedback_count': sum(len(feedbacks) for feedbacks in self.peer_feedback[node_id].values())
            }
            
        except Exception as e:
            logger.error(f"Error getting trust info for node {node_id}: {e}")
            return {
                'node_id': node_id,
                'trust_score': 0.5,
                'error': str(e)
            }
    
    def get_all_node_trust_scores(self) -> Dict[str, float]:
        """Get trust scores for all nodes"""
        scores = {}
        
        for node_id in self.node_reports.keys():
            scores[node_id] = self.calculate_monitoring_trust(node_id)
        
        return scores
    
    def cleanup(self, max_age_hours: int = 24):
        """
        Clean up old data to prevent memory issues (alias for cleanup_old_data)
        """
        return self.cleanup_old_data(max_age_hours)

    def cleanup_old_data(self, max_age_hours: int = 24):
        """
        Clean up old data to prevent memory issues
        
        Args:
            max_age_hours: Maximum age of data to keep
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            # Clean up old reports
            for node_id in list(self.node_reports.keys()):
                # Filter old reports
                filtered_reports = deque(
                    [r for r in self.node_reports[node_id] 
                     if datetime.fromisoformat(r['timestamp']) > cutoff_time],
                    maxlen=self.window_size
                )
                
                if filtered_reports:
                    self.node_reports[node_id] = filtered_reports
                else:
                    # Remove node if no recent reports
                    del self.node_reports[node_id]
                    if node_id in self.node_trust_scores:
                        del self.node_trust_scores[node_id]
                    if node_id in self.node_last_update:
                        del self.node_last_update[node_id]
            
            # Clean up old peer feedback
            for target_node in list(self.peer_feedback.keys()):
                for reporter_id in list(self.peer_feedback[target_node].keys()):
                    filtered_feedback = [
                        f for f in self.peer_feedback[target_node][reporter_id]
                        if datetime.fromisoformat(f['timestamp']) > cutoff_time
                    ]
                    
                    if filtered_feedback:
                        self.peer_feedback[target_node][reporter_id] = filtered_feedback
                    else:
                        del self.peer_feedback[target_node][reporter_id]
                
                if not self.peer_feedback[target_node]:
                    del self.peer_feedback[target_node]
            
            # Clean up old content hashes
            for url in list(self.content_hashes.keys()):
                for node_id in list(self.content_hashes[url].keys()):
                    filtered_hashes = [
                        h for h in self.content_hashes[url][node_id]
                        if datetime.fromisoformat(h['timestamp']) > cutoff_time
                    ]
                    
                    if filtered_hashes:
                        self.content_hashes[url][node_id] = filtered_hashes
                    else:
                        del self.content_hashes[url][node_id]
                
                if not self.content_hashes[url]:
                    del self.content_hashes[url]
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_trust_statistics(self) -> Dict:
        """Get overall trust statistics"""
        try:
            all_scores = list(self.get_all_node_trust_scores().values())
            
            if not all_scores:
                return {
                    'total_nodes': 0,
                    'average_trust': 0.0,
                    'trust_distribution': {}
                }
            
            avg_trust = statistics.mean(all_scores)
            min_trust = min(all_scores)
            max_trust = max(all_scores)
            
            # Trust distribution
            distribution = {
                'high_trust (0.8-1.0)': sum(1 for s in all_scores if s >= 0.8),
                'medium_trust (0.5-0.8)': sum(1 for s in all_scores if 0.5 <= s < 0.8),
                'low_trust (0.2-0.5)': sum(1 for s in all_scores if 0.2 <= s < 0.5),
                'very_low_trust (0.0-0.2)': sum(1 for s in all_scores if s < 0.2)
            }
            
            return {
                'total_nodes': len(all_scores),
                'average_trust': avg_trust,
                'min_trust': min_trust,
                'max_trust': max_trust,
                'trust_distribution': distribution,
                'total_reports': sum(len(reports) for reports in self.node_reports.values()),
                'peer_feedback_entries': sum(
                    len(feedbacks) 
                    for node_feedbacks in self.peer_feedback.values()
                    for feedbacks in node_feedbacks.values()
                )
            }
            
        except Exception as e:
            logger.error(f"Error calculating trust statistics: {e}")
            return {
                'error': str(e)
            }

class TrustCalculator:
    """Utility class for calculating Proof of Reputation scores"""
    
    @staticmethod
    def calculate_por_score(monitoring_trust: float, ml_score: float) -> float:
        """
        Calculate Proof of Reputation score
        
        Args:
            monitoring_trust: Monitoring trust score (0-1)
            ml_score: ML confidence score (0-1)
            
        Returns:
            PoR score (0-1)
        """
        return 0.4 * monitoring_trust + 0.6 * ml_score
    
    @staticmethod
    def get_trust_level(score: float) -> str:
        """
        Get trust level description based on score
        
        Args:
            score: Trust score (0-1)
            
        Returns:
            Trust level string
        """
        if score >= 0.8:
            return "Very High"
        elif score >= 0.6:
            return "High"
        elif score >= 0.4:
            return "Medium"
        elif score >= 0.2:
            return "Low"
        else:
            return "Very Low"

if __name__ == "__main__":
    # Test the trust engine
    import json
    
    # Create sample data
    trust_engine = TrustEngine()
    
    # Add sample reports
    sample_reports = [
        {
            'url': 'https://example.com',
            'status': 'success',
            'response_time_ms': 150,
            'content_hash': 'abc123',
            'timestamp': datetime.now().isoformat()
        },
        {
            'url': 'https://example.com',
            'status': 'success',
            'response_time_ms': 160,
            'content_hash': 'abc123',
            'timestamp': datetime.now().isoformat()
        },
        {
            'url': 'https://test.com',
            'status': 'error',
            'response_time_ms': None,
            'content_hash': None,
            'timestamp': datetime.now().isoformat()
        }
    ]
    
    node_id = "test_node_1"
    for report in sample_reports:
        trust_engine.add_monitoring_report(node_id, report)
    
    # Calculate trust
    trust_score = trust_engine.calculate_monitoring_trust(node_id)
    trust_info = trust_engine.get_node_trust_info(node_id)
    
    print("Trust Engine Test Results:")
    print(f"Trust Score: {trust_score:.4f}")
    print("Trust Info:")
    print(json.dumps(trust_info, indent=2))
    
    # Test PoR calculation
    ml_score = 0.85
    por_score = TrustCalculator.calculate_por_score(trust_score, ml_score)
    trust_level = TrustCalculator.get_trust_level(por_score)
    
    print(f"\nPoR Score: {por_score:.4f}")
    print(f"Trust Level: {trust_level}")
