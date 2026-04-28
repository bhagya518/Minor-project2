#!/usr/bin/env python3
"""
epoch_manager.py

EPOCH-BASED CONSENSUS MANAGER
Runs independently of monitoring cycle to ensure proper consensus timing

Key Features:
- Runs every 60 seconds (epoch window)
- Waits for peer reports to arrive before consensus
- Requires minimum 2 other node reports for quorum
- Implements 2/3 majority voting
- Executes actual slashing of malicious nodes
- Updates PoR for all nodes
- Clears peer_reports for next epoch
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional
from collections import defaultdict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class EpochManager:
    """
    Manages epoch-based consensus execution
    Runs independently of monitoring cycle to ensure proper timing
    """
    
    def __init__(self, node_id: str, ml_consensus_engine=None, blockchain_client=None):
        self.node_id = node_id
        self.ml_consensus_engine = ml_consensus_engine
        self.blockchain_client = blockchain_client
        
        # Epoch management
        self.current_epoch = 0
        self.epoch_reports = defaultdict(list)  # epoch_id -> list of reports
        self.own_reports = {}  # epoch_id -> own report
        
        # Consensus results
        self.epoch_decisions = {}  # epoch_id -> decision
        self.slash_history = []  # history of slashing actions
        
        # Configuration
        self.epoch_duration = 60  # seconds
        self.min_peers_for_quorum = 2  # need reports from at least 2 other nodes
        self.quorum_threshold = 2/3  # 2/3 majority required
        
    def get_current_epoch(self) -> int:
        """Get current epoch based on time"""
        return int(time.time() // self.epoch_duration)
    
    def add_report(self, report: Dict, is_own: bool = False):
        """
        Add a report to the current epoch
        
        Args:
            report: Monitoring report dictionary
            is_own: Whether this is our own report
        """
        epoch_id = report.get("epoch_id", self.get_current_epoch())
        
        if is_own:
            self.own_reports[epoch_id] = report
        else:
            self.epoch_reports[epoch_id].append(report)
        
        logger.debug(f"Added report for epoch {epoch_id} (own={is_own}), total peer reports: {len(self.epoch_reports[epoch_id])}")
    
    async def run_epoch_manager(self):
        """
        Main epoch manager loop - runs every 60 seconds
        Offset by 10 seconds from monitoring cycle to allow reports to arrive
        """
        logger.info("Starting epoch manager loop...")
        
        # Wait 10 seconds offset to allow reports to arrive
        await asyncio.sleep(10)
        
        while True:
            try:
                current_epoch = self.get_current_epoch()
                
                # Process previous epoch (gives time for all reports to arrive)
                previous_epoch = current_epoch - 1
                
                if previous_epoch in self.epoch_reports or previous_epoch in self.own_reports:
                    await self.process_epoch(previous_epoch)
                
                # Sleep for next epoch cycle
                await asyncio.sleep(self.epoch_duration)
                
            except Exception as e:
                logger.error(f"Error in epoch manager loop: {e}")
                await asyncio.sleep(10)  # Wait before retry
    
    async def process_epoch(self, epoch_id: int):
        """
        Process a single epoch - run consensus, voting, and slashing
        
        Args:
            epoch_id: Epoch to process
        """
        logger.info(f"Processing epoch {epoch_id}")
        
        # Collect all reports for this epoch
        peer_reports = self.epoch_reports.get(epoch_id, [])
        own_report = self.own_reports.get(epoch_id)
        
        all_reports = peer_reports.copy()
        if own_report:
            all_reports.append(own_report)
        
        # Check if we have enough reports for consensus
        if len(peer_reports) < self.min_peers_for_quorum:
            logger.info(f"Epoch {epoch_id}: only {len(peer_reports)} peer reports (need {self.min_peers_for_quorum}), skipping consensus")
            return
        
        logger.info(f"Epoch {epoch_id}: processing {len(all_reports)} total reports ({len(peer_reports)} peer + {1 if own_report else 0} own)")
        
        # Step 1: Build feature matrix from all reports
        try:
            feature_matrix = self.build_feature_matrix(all_reports)
            if feature_matrix.empty:
                logger.warning(f"Epoch {epoch_id}: empty feature matrix, skipping consensus")
                return
        except Exception as e:
            logger.error(f"Epoch {epoch_id}: error building feature matrix: {e}")
            return
        
        # Step 2: Fit ML scaler if not fitted and enough reports available
        if self.ml_consensus_engine and len(all_reports) >= 5:
            try:
                # Check if scaler needs fitting
                if not hasattr(self.ml_consensus_engine, 'scaler_fitted') or not self.ml_consensus_engine.scaler_fitted:
                    # Use ML consensus engine's feature extraction instead
                    features_df = self.ml_consensus_engine.extract_features_from_reports(all_reports)
                    if not features_df.empty and len(features_df) >= 5:
                        # Extract only numeric feature columns
                        if self.ml_consensus_engine.feature_cols:
                            feature_matrix = features_df[self.ml_consensus_engine.feature_cols].fillna(0)
                            self.ml_consensus_engine.scaler.fit(feature_matrix)
                            self.ml_consensus_engine.scaler_fitted = True
                            logger.info(f"Epoch {epoch_id}: ML scaler fitted on live data")
            except Exception as e:
                logger.warning(f"Epoch {epoch_id}: failed to fit scaler: {e}")
        
        # Step 3: Run ML consensus on aggregated reports
        try:
            if self.ml_consensus_engine:
                consensus_results = self.ml_consensus_engine.process_epoch_consensus(epoch_id, all_reports)
                logger.info(f"Epoch {epoch_id}: ML consensus completed")
            else:
                # Fallback: simple majority voting without ML
                consensus_results = self.simple_majority_vote(all_reports)
                logger.info(f"Epoch {epoch_id}: simple majority voting (no ML engine)")
        except Exception as e:
            logger.error(f"Epoch {epoch_id}: error in consensus: {e}")
            return
        
        # Step 3: Quorum voting (2/3 majority)
        malicious_votes = 0
        honest_votes = 0
        node_verdicts = {}  # node_id -> verdict
        
        for report in all_reports:
            node_id = report.get("node_address", report.get("node_id", "unknown"))
            # Get verdict from consensus results or simple heuristic
            if consensus_results and "predictions" in consensus_results:
                # Find prediction for this node
                for pred in consensus_results["predictions"]:
                    if pred.get("node_id") == node_id:
                        # Check for both possible column names
                        p_malicious = pred.get("malicious_probability", pred.get("p_malicious", 0.5))
                        verdict = "malicious" if p_malicious >= 0.5 else "honest"
                        node_verdicts[node_id] = verdict
                        if verdict == "malicious":
                            malicious_votes += 1
                        else:
                            honest_votes += 1
                        break
            else:
                # Fallback: simple heuristic based on report quality
                is_reachable = report.get("is_reachable", True)
                ssl_valid = report.get("ssl_valid", True)
                verdict = "honest" if (is_reachable and ssl_valid) else "malicious"
                node_verdicts[node_id] = verdict
                if verdict == "malicious":
                    malicious_votes += 1
                else:
                    honest_votes += 1
        
        # Step 4: Check quorum - require strict majority with minimum 2 votes
        total_votes = len(all_reports)
        quorum = max(2, (total_votes * 2) // 3 + 1)
        
        logger.info(f"Epoch {epoch_id}: voting results - honest: {honest_votes}, malicious: {malicious_votes}, quorum: {quorum}")
        
        # Step 5: Execute slashing if quorum reached for malicious
        if malicious_votes >= quorum:
            logger.warning(f"Epoch {epoch_id}: quorum reached for malicious ({malicious_votes}/{quorum}), executing slashing")
            
            for report in all_reports:
                node_id = report.get("node_address", report.get("node_id", "unknown"))
                verdict = node_verdicts.get(node_id, "honest")
                
                if verdict == "malicious":
                    try:
                        # Execute actual slash
                        if self.blockchain_client:
                            slash_result = await self.blockchain_client.slash_node(
                                node_id, 
                                amount=0.1,  # Slash 10% reputation
                                reason=f"Epoch {epoch_id} consensus: {malicious_votes}/{total_votes} nodes voted malicious"
                            )
                            logger.warning(f"Epoch {epoch_id}: SLASHED node {node_id} - {slash_result}")
                            self.slash_history.append({
                                "epoch": epoch_id,
                                "node_id": node_id,
                                "reason": f"Consensus: {malicious_votes}/{total_votes} malicious votes",
                                "timestamp": time.time()
                            })
                    except Exception as e:
                        logger.error(f"Epoch {epoch_id}: error slashing node {node_id}: {e}")
        else:
            logger.info(f"Epoch {epoch_id}: no malicious quorum (need {quorum}, got {malicious_votes})")
        
        # Step 6: Update PoR for all nodes with penalty-based calculation
        for report in all_reports:
            node_id = report.get("node_address", report.get("node_id", "unknown"))
            verdict = node_verdicts.get(node_id, "honest")
            
            # Get current PoR from blockchain
            try:
                if self.blockchain_client:
                    current_por = await self.blockchain_client.get_reputation(node_id)
                else:
                    current_por = 0.95  # Default if no blockchain
            except Exception as e:
                current_por = 0.95
                logger.warning(f"Epoch {epoch_id}: could not get current PoR for {node_id}, using default")
            
            # Calculate new PoR with penalty
            penalty = 0.1 if verdict == "malicious" else 0.0
            new_por = current_por * (1 - penalty)
            
            try:
                if self.blockchain_client:
                    await self.blockchain_client.update_reputation(
                        node_id, 
                        new_por, 
                        evidence=f"Epoch {epoch_id} verdict: {verdict} (penalty: {penalty})"
                    )
                    logger.info(f"Epoch {epoch_id}: Updated PoR for {node_id}: {current_por:.3f} → {new_por:.3f} (penalty: {penalty})")
            except Exception as e:
                logger.error(f"Epoch {epoch_id}: error updating PoR for {node_id}: {e}")
        
        # Step 7: Store epoch decision
        self.epoch_decisions[epoch_id] = {
            "malicious_votes": malicious_votes,
            "honest_votes": honest_votes,
            "total_votes": total_votes,
            "quorum_reached": malicious_votes >= quorum,
            "node_verdicts": node_verdicts,
            "consensus_results": consensus_results
        }
        
        # Step 8: Clear reports for this epoch (cleanup)
        if epoch_id in self.epoch_reports:
            del self.epoch_reports[epoch_id]
        if epoch_id in self.own_reports:
            del self.own_reports[epoch_id]
        
        logger.info(f"Epoch {epoch_id}: processing completed, reports cleared")
    
    def build_feature_matrix(self, reports: List[Dict]) -> pd.DataFrame:
        """
        Build feature matrix from reports for ML processing
        
        Args:
            reports: List of monitoring reports
            
        Returns:
            DataFrame with features
        """
        features = []
        
        for report in reports:
            feature = {
                "node_id": report.get("node_address", report.get("node_id", "unknown")),
                "response_ms": report.get("response_ms", 0),
                "status_code": report.get("status_code", 0),
                "ssl_valid": 1 if report.get("ssl_valid", True) else 0,
                "is_reachable": 1 if report.get("is_reachable", True) else 0,
                "epoch_id": report.get("epoch_id", 0)
            }
            features.append(feature)
        
        return pd.DataFrame(features)
    
    def simple_majority_vote(self, reports: List[Dict]) -> Dict:
        """
        Fallback simple majority voting when ML engine is not available
        
        Args:
            reports: List of monitoring reports
            
        Returns:
            Consensus results dictionary
        """
        malicious_count = 0
        honest_count = 0
        
        for report in reports:
            is_reachable = report.get("is_reachable", True)
            ssl_valid = report.get("ssl_valid", True)
            
            # Simple heuristic: if not reachable or SSL invalid, consider malicious
            if not is_reachable or not ssl_valid:
                malicious_count += 1
            else:
                honest_count += 1
        
        total = len(reports)
        consensus_malicious = malicious_count > honest_count
        
        return {
            "consensus": {
                "malicious": consensus_malicious,
                "malicious_votes": malicious_count,
                "honest_votes": honest_count,
                "total_votes": total,
                "confidence": malicious_count / total if total > 0 else 0
            },
            "predictions": [
                {
                    "node_id": r.get("node_address", r.get("node_id", "unknown")),
                    "p_malicious": 0.8 if not r.get("is_reachable", True) else 0.1
                }
                for r in reports
            ]
        }
    
    def get_epoch_status(self, epoch_id: int) -> Optional[Dict]:
        """
        Get status of a processed epoch
        
        Args:
            epoch_id: Epoch to query
            
        Returns:
            Epoch decision dictionary or None
        """
        return self.epoch_decisions.get(epoch_id)
    
    def get_current_epoch_reports(self) -> Dict:
        """
        Get reports for current epoch (for monitoring)
        
        Returns:
            Dictionary with own and peer report counts
        """
        current_epoch = self.get_current_epoch()
        return {
            "epoch_id": current_epoch,
            "own_report": current_epoch in self.own_reports,
            "peer_reports_count": len(self.epoch_reports.get(current_epoch, [])),
            "total_reports": len(self.epoch_reports.get(current_epoch, [])) + (1 if current_epoch in self.own_reports else 0)
        }


# Global epoch manager instance
epoch_manager: Optional[EpochManager] = None


def get_epoch_manager() -> Optional[EpochManager]:
    """Get the global epoch manager instance"""
    return epoch_manager


def init_epoch_manager(node_id: str, ml_consensus_engine=None, blockchain_client=None) -> EpochManager:
    """
    Initialize the global epoch manager
    
    Args:
        node_id: Node identifier
        ml_consensus_engine: ML consensus engine instance
        blockchain_client: Blockchain client instance
        
    Returns:
        Initialized epoch manager
    """
    global epoch_manager
    epoch_manager = EpochManager(node_id, ml_consensus_engine, blockchain_client)
    logger.info(f"Epoch manager initialized for node {node_id}")
    return epoch_manager
