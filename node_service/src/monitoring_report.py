"""
Monitoring Report Schema - Phase 1
Signed report system for decentralized website monitoring

This module defines the canonical data structure that every node produces
and every other component consumes. It serves as the "interface contract"
for the entire system.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

# Cryptography imports
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey
)
from cryptography.hazmat.primitives.serialization import (
    Encoding, PublicFormat, PrivateFormat, NoEncryption
)


@dataclass
class MonitoringReport:
    """
    Canonical monitoring report structure.
    
    Every node produces this format when monitoring websites.
    Every peer verifies this format when receiving reports.
    The blockchain stores the report_hash for immutability.
    """
    
    # What was monitored
    url: str
    epoch_id: int  # which epoch window this belongs to (e.g. unix_time // 60)
    
    # Raw monitoring results
    response_ms: float  # -1.0 if request failed
    status_code: int    # 0 if no response
    ssl_valid: bool
    content_hash: str   # SHA-256 of response body, "" if failed
    is_reachable: bool
    
    # Node identity
    node_address: str  # e.g. "node_a:8001" or a wallet address later
    timestamp: float = field(default_factory=time.time)
    
    # Filled after signing
    report_hash: str = ""   # SHA-256 of the canonical payload
    signature: str = ""     # Ed25519 hex signature of report_hash
    
    def canonical_payload(self) -> bytes:
        """
        Deterministic bytes used for hashing and signing.
        Order is fixed — never use asdict() directly (key order varies).
        """
        payload = {
            "url": self.url,
            "epoch_id": self.epoch_id,
            "response_ms": round(self.response_ms, 2),
            "status_code": self.status_code,
            "ssl_valid": self.ssl_valid,
            "content_hash": self.content_hash,
            "is_reachable": self.is_reachable,
            "node_address": self.node_address,
            "timestamp": round(self.timestamp, 3),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    
    def compute_hash(self) -> str:
        """Compute SHA-256 hash of canonical payload"""
        return hashlib.sha256(self.canonical_payload()).hexdigest()


class NodeSigner:
    """
    Each node has one key pair, signs every report it emits.
    
    Usage:
        signer = NodeSigner()  # generate new key
        signed_report = signer.sign_report(report)
        
        # Save private key for reuse:
        private_hex = signer.export_private_key_hex()
        # Later: signer = NodeSigner(private_key_hex=private_hex)
    """
    
    def __init__(self, private_key_hex: Optional[str] = None):
        if private_key_hex:
            raw = bytes.fromhex(private_key_hex)
            self._private = Ed25519PrivateKey.from_private_bytes(raw)
        else:
            self._private = Ed25519PrivateKey.generate()
        self._public = self._private.public_key()
    
    @property
    def public_key_hex(self) -> str:
        """Get public key as hex string for sharing with peers"""
        return self._public.public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    
    def sign_report(self, report: MonitoringReport) -> MonitoringReport:
        """
        Fills report_hash and signature in-place, returns the report.
        
        Args:
            report: Unsigned monitoring report
            
        Returns:
            The same report object with report_hash and signature filled
        """
        report.report_hash = report.compute_hash()
        sig_bytes = self._private.sign(bytes.fromhex(report.report_hash))
        report.signature = sig_bytes.hex()
        return report
    
    def export_private_key_hex(self) -> str:
        """Export private key for saving to config file"""
        raw = self._private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        return raw.hex()


class ReportVerifier:
    """
    Used by peers when they receive a report over P2P.
    Verifies signatures and hashes without needing the private key.
    """
    
    @staticmethod
    def verify(report: MonitoringReport, sender_public_key_hex: str) -> bool:
        """
        Returns True if signature is valid and hash matches payload.
        
        Args:
            report: Signed monitoring report to verify
            sender_public_key_hex: Public key of the sender node
            
        Returns:
            True if report is cryptographically valid
        """
        try:
            # Re-compute hash independently
            expected_hash = report.compute_hash()
            if expected_hash != report.report_hash:
                return False
            
            # Load sender's public key
            public_key_bytes = bytes.fromhex(sender_public_key_hex)
            public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            
            # Verify signature
            public_key.verify(
                bytes.fromhex(report.signature),
                bytes.fromhex(report.report_hash)
            )
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def to_dict(report: MonitoringReport) -> dict:
        """Convert report to dictionary for serialization"""
        return asdict(report)
    
    @staticmethod
    def from_dict(data: dict) -> MonitoringReport:
        """Create report from dictionary (e.g., after P2P transmission)"""
        return MonitoringReport(**data)


def current_epoch(window_seconds: int = 60) -> int:
    """
    All nodes produce the same epoch_id within the same time window.
    
    Args:
        window_seconds: Duration of each epoch (default: 60 seconds)
        
    Returns:
        Current epoch number (unix_time // window_seconds)
    """
    return int(time.time()) // window_seconds


def create_test_report() -> MonitoringReport:
    """Create a test report for debugging"""
    return MonitoringReport(
        url="https://example.com",
        epoch_id=current_epoch(),
        response_ms=123.45,
        status_code=200,
        ssl_valid=True,
        content_hash="a" * 64,  # 64 hex chars = 256 bits
        is_reachable=True,
        node_address="test_node:8000"
    )


# Example usage and testing
if __name__ == "__main__":
    print("Testing Monitoring Report System...")
    
    # Create a report
    report = create_test_report()
    print(f"Created report for {report.url}")
    print(f"Canonical payload: {report.canonical_payload()}")
    print(f"Computed hash: {report.compute_hash()}")
    
    # Sign the report
    signer = NodeSigner()
    print(f"Node public key: {signer.public_key_hex[:16]}...")
    
    signed_report = signer.sign_report(report)
    print(f"Signed report hash: {signed_report.report_hash[:16]}...")
    print(f"Signature: {signed_report.signature[:16]}...")
    
    # Verify the report
    is_valid = ReportVerifier.verify(signed_report, signer.public_key_hex)
    print(f"Signature verification: {'PASS' if is_valid else 'FAIL'}")
    
    # Test serialization
    report_dict = ReportVerifier.to_dict(signed_report)
    restored_report = ReportVerifier.from_dict(report_dict)
    print(f"Serialization round-trip: {'PASS' if restored_report.report_hash == signed_report.report_hash else 'FAIL'}")
    
    # Test tampering detection
    tampered_report = MonitoringReport(**asdict(signed_report))
    tampered_report.response_ms = 999.99  # Tamper with data
    is_tampered_valid = ReportVerifier.verify(tampered_report, signer.public_key_hex)
    print(f"Tampering detection: {'PASS' if not is_tampered_valid else 'FAIL'}")
    
    print("\nAll tests completed!")
