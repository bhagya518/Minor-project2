"""
Website Monitoring Module
Monitors websites for HTTP status, response time, SSL, DNS, and content
Now emits cryptographically signed MonitoringReport objects
"""

import asyncio
import aiohttp
import ssl
import dns.resolver
import hashlib
import time
import logging
import os
import certifi
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from datetime import datetime
import json

def get_current_epoch():
    """Get current epoch ID"""
    return int(time.time() // 60)  # 1-minute epochs

def get_latest_results():
    """Get latest monitoring results (module-level function)"""
    return {}  # Return empty dict for now

# Import signed report system (Phase 1)
from monitoring_report import MonitoringReport, NodeSigner, current_epoch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create ONE signer at module level (persists across calls)
NODE_SIGNER = NodeSigner()

# MY_NODE_ID will be set from node_config in main.py
MY_NODE_ID = None

def set_node_id(node_id: str):
    """Set the node ID from main.py configuration"""
    global MY_NODE_ID
    MY_NODE_ID = node_id
    logger.info(f"WebsiteMonitor initialized with node_id: {MY_NODE_ID}")
    logger.info(f"Node public key: {NODE_SIGNER.public_key_hex[:16]}...")

def _build_signed_report(url: str, response_ms: float, status_code: int, 
                        ssl_valid: bool, body: str, is_reachable: bool) -> MonitoringReport:
    """
    Build and sign a MonitoringReport from monitoring results
    
    Args:
        url: Monitored URL
        response_ms: Response time in milliseconds
        status_code: HTTP status code
        ssl_valid: Whether SSL certificate is valid
        body: Response body content
        is_reachable: Whether website was reachable
        
    Returns:
        Signed MonitoringReport ready for P2P broadcast
    """
    # Calculate content hash
    content_hash = hashlib.sha256(body.encode()).hexdigest() if body else ""
    
    # Create report
    global MY_NODE_ID
    node_address = MY_NODE_ID or f"node_unknown:{os.getpid()}"
    
    report = MonitoringReport(
        url=url,
        epoch_id=get_current_epoch(),
        response_ms=response_ms,
        status_code=status_code,
        ssl_valid=ssl_valid,
        content_hash=content_hash,
        is_reachable=is_reachable,
        node_address=node_address,
    )
    
    # Sign the report
    return NODE_SIGNER.sign_report(report)


class WebsiteMonitor:
    """Website monitoring service with comprehensive checks"""
    
    def __init__(self, timeout: int = 10, max_retries: int = 3):
        """
        Initialize website monitor
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=10,
            ssl=ssl.create_default_context(cafile=certifi.where())
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def check_website(self, url: str) -> Dict:
        """
        Perform comprehensive website monitoring
        
        Args:
            url: Website URL to monitor
            
        Returns:
            Dictionary with monitoring results
        """
        start_time = time.time()
        
        try:
            # Parse URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError("Invalid URL format")
            
            # Initialize results
            results = {
                'url': url,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'http_status': None,
                'response_time_ms': None,
                'ssl_valid': None,
                'dns_resolution_time_ms': None,
                'content_hash': None,
                'content_length': None,
                'error': None,
                'checks_performed': []
            }
            
            # DNS resolution check
            dns_time = await self._check_dns_resolution(parsed_url.netloc)
            results['dns_resolution_time_ms'] = dns_time
            results['checks_performed'].append('dns')
            
            # HTTP request check
            http_results = await self._check_http_request(url)
            results.update(http_results)
            results['checks_performed'].append('http')
            
            # SSL check (if HTTPS)
            if parsed_url.scheme == 'https':
                ssl_result = await self._check_ssl_certificate(parsed_url.netloc)
                results['ssl_valid'] = ssl_result
                results['checks_performed'].append('ssl')
            
            # Content hash calculation
            if http_results.get('content'):
                content_hash = self._calculate_content_hash(http_results['content'])
                results['content_hash'] = content_hash
                results['content_length'] = len(http_results['content'])
                results['checks_performed'].append('content_hash')
            
            # Calculate total response time
            total_time = (time.time() - start_time) * 1000
            results['total_time_ms'] = total_time
            
            # Set overall status - accept all 2xx and 3xx as success
            if results['http_status'] and 200 <= results['http_status'] < 400:
                results['status'] = 'success'
            
            logger.info(f"Monitoring completed for {url}: {results['status']}")
            return results
            
        except Exception as e:
            logger.error(f"Error monitoring {url}: {e}")
            return {
                'url': url,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e),
                'total_time_ms': (time.time() - start_time) * 1000,
                'checks_performed': []
            }
    
    async def _check_dns_resolution(self, hostname: str) -> Optional[float]:
        """Check DNS resolution time"""
        try:
            start_time = time.time()
            
            # Use asyncio-compatible DNS resolution
            loop = asyncio.get_event_loop()
            answers = await loop.run_in_executor(
                None, 
                lambda: dns.resolver.resolve(hostname, 'A')
            )
            
            resolution_time = (time.time() - start_time) * 1000
            logger.debug(f"DNS resolution for {hostname}: {resolution_time:.2f}ms")
            
            return resolution_time
            
        except Exception as e:
            logger.error(f"DNS resolution failed for {hostname}: {e}")
            return None
    
    async def _check_http_request(self, url: str) -> Dict:
        """Check HTTP request and get response details"""
        results = {
            'http_status': None,
            'response_time_ms': None,
            'content': None,
            'response_headers': {},
            'error': None
        }
        
        # Add proper headers to avoid bot detection
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                async with self.session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    content = await response.text()
                    response_time = (time.time() - start_time) * 1000
                    
                    results.update({
                        'http_status': response.status,
                        'response_time_ms': response_time,
                        'content': content,
                        'response_headers': dict(response.headers),
                        'error': None
                    })
                    
                    logger.debug(f"HTTP request for {url}: {response.status} in {response_time:.2f}ms")
                    return results
                    
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
                if attempt == self.max_retries - 1:
                    results['error'] = 'Request timeout'
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.warning(f"HTTP request error on attempt {attempt + 1} for {url}: {e}")
                if attempt == self.max_retries - 1:
                    results['error'] = str(e)
                await asyncio.sleep(1)
        
        return results
    
    async def _check_ssl_certificate(self, hostname: str) -> bool:
        """Check SSL certificate validity"""
        try:
            # Create SSL context with certifi certificate bundle
            context = ssl.create_default_context(cafile=certifi.where())
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
            
            # Try to establish SSL connection
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(hostname, 443, ssl=context),
                timeout=self.timeout.total
            )
            
            writer.close()
            await writer.wait_closed()
            
            logger.debug(f"SSL certificate valid for {hostname}")
            return True
            
        except Exception as e:
            logger.debug(f"SSL certificate check failed for {hostname}: {e}")
            return False
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate SHA256 hash of content"""
        try:
            # Remove whitespace and normalize content for consistent hashing
            normalized_content = ' '.join(content.split())
            content_hash = hashlib.sha256(normalized_content.encode()).hexdigest()
            return content_hash
        except Exception as e:
            logger.error(f"Error calculating content hash: {e}")
            return ""
    
    async def monitor_websites_async(self, urls: List[str]) -> List[Dict]:
        """
        Monitor multiple websites concurrently (alias for monitor_multiple_websites)
        """
        return await self.monitor_multiple_websites(urls)

    async def monitor_multiple_websites(self, urls: List[str]) -> List[Dict]:
        """
        Monitor multiple websites concurrently
        
        Args:
            urls: List of website URLs to monitor
            
        Returns:
            List of monitoring results
        """
        logger.info(f"Starting monitoring of {len(urls)} websites")
        
        # Ensure session is created
        if not self.session:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=10,
                ssl=ssl.create_default_context()
            )
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.timeout
            )
        
        # Create tasks for concurrent monitoring
        tasks = [self.check_website(url) for url in urls]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error monitoring {urls[i]}: {result}")
                processed_results.append({
                    'url': urls[i],
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error',
                    'error': str(result)
                })
            else:
                processed_results.append(result)
        
        success_count = sum(1 for r in processed_results if r.get('status') == 'success')
        logger.info(f"Monitoring completed: {success_count}/{len(urls)} successful")
        
        return processed_results
    
    def extract_monitoring_features(self, results: List[Dict]) -> Dict:
        """
        Extract features for ML model from monitoring results
        
        Args:
            results: List of monitoring results
            
        Returns:
            Dictionary with extracted features
        """
        if not results:
            return {
                'avg_response_ms': 0,
                'ssl_valid_rate': 0,
                'content_match_rate': 0,
                'stale_report_rate': 0,
                'false_report_rate': 0
            }
        
        # Calculate average response time
        response_times = [r.get('response_time_ms', 0) for r in results if r.get('response_time_ms')]
        avg_response_ms = sum(response_times) / len(response_times) if response_times else 0
        
        # Calculate SSL valid rate
        ssl_checks = [r for r in results if 'ssl_valid' in r]
        ssl_valid_count = sum(1 for r in ssl_checks if r.get('ssl_valid') is True)
        ssl_valid_rate = ssl_valid_count / len(ssl_checks) if ssl_checks else 1.0
        
        # Calculate content match rate (simplified - in real system would compare with peers)
        content_hashes = [r.get('content_hash') for r in results if r.get('content_hash')]
        if len(content_hashes) > 1:
            # Count unique hashes
            unique_hashes = set(content_hashes)
            # Match rate is based on most common hash
            most_common_hash = max(set(content_hashes), key=content_hashes.count)
            match_count = content_hashes.count(most_common_hash)
            content_match_rate = match_count / len(content_hashes)
        else:
            content_match_rate = 1.0
        
        # Calculate stale report rate (old timestamps)
        current_time = datetime.now()
        stale_threshold = 300  # 5 minutes
        stale_count = sum(1 for r in results 
                        if (current_time - datetime.fromisoformat(r['timestamp'])).total_seconds() > stale_threshold)
        stale_report_rate = stale_count / len(results) if results else 0
        
        # Calculate false report rate (error status)
        false_count = sum(1 for r in results if r.get('status') == 'error')
        false_report_rate = false_count / len(results) if results else 0
        
        features = {
            'avg_response_ms': avg_response_ms,
            'ssl_valid_rate': ssl_valid_rate,
            'content_match_rate': content_match_rate,
            'stale_report_rate': stale_report_rate,
            'false_report_rate': false_report_rate
        }
        
        logger.debug(f"Extracted features: {features}")
        return features

class MonitoringScheduler:
    """Scheduler for periodic website monitoring"""
    
    def __init__(self, monitor: WebsiteMonitor, websites: List[str], interval: int = 60, on_cycle_complete=None):
        """
        Initialize monitoring scheduler
        
        Args:
            monitor: WebsiteMonitor instance
            websites: List of websites to monitor
            interval: Monitoring interval in seconds
            on_cycle_complete: Callback function to call after each monitoring cycle
        """
        self.monitor = monitor
        self.websites = websites
        self.interval = interval
        self.running = False
        self.results_history = []
        self.on_cycle_complete = on_cycle_complete  # Callback function
        
    async def start_monitoring(self):
        """Start periodic monitoring"""
        self.running = True
        logger.info(f"Starting periodic monitoring every {self.interval} seconds")
        
        while self.running:
            try:
                # Monitor all websites
                results = await self.monitor.monitor_multiple_websites(self.websites)
                
                # Store results
                self.results_history.extend(results)
                
                # Keep only last 1000 results to prevent memory issues
                if len(self.results_history) > 1000:
                    self.results_history = self.results_history[-1000:]
                
                # Extract features for ML
                features = self.monitor.extract_monitoring_features(results)
                
                logger.info(f"Monitoring cycle completed. Features: {features}")
                
                # Call callback if provided (Phase 2: triggers broadcast)
                if self.on_cycle_complete:
                    try:
                        await self.on_cycle_complete()
                    except Exception as e:
                        logger.error(f"Error in monitoring callback: {e}")
                
                # Wait for next cycle
                await asyncio.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {e}")
                await asyncio.sleep(5)  # Short delay before retry
    
    def stop_monitoring(self):
        """Stop periodic monitoring"""
        self.running = False
        logger.info("Stopping periodic monitoring")
    
    def get_latest_results(self, limit: int = 100) -> List[Dict]:
        """Get latest monitoring results"""
        return self.results_history[-limit:]
    
    def get_features_for_ml(self) -> Dict:
        """Get features for ML model"""
        if not self.results_history:
            return {
                'avg_response_ms': 0,
                'ssl_valid_rate': 0,
                'content_match_rate': 0,
                'stale_report_rate': 0,
                'false_report_rate': 0
            }
        
        # Use recent results (last 100)
        recent_results = self.results_history[-100:]
        return self.monitor.extract_monitoring_features(recent_results)

if __name__ == "__main__":
    # Test the website monitor
    async def test_monitor():
        websites = [
            "https://httpbin.org/status/200",
            "https://httpbin.org/delay/1",
            "https://google.com"
        ]
        
        async with WebsiteMonitor() as monitor:
            results = await monitor.monitor_multiple_websites(websites)
            
            print("Monitoring Results:")
            for result in results:
                print(f"URL: {result['url']}")
                print(f"Status: {result['status']}")
                print(f"HTTP Status: {result.get('http_status')}")
                print(f"Response Time: {result.get('response_time_ms')}ms")
                print(f"SSL Valid: {result.get('ssl_valid')}")
                print(f"Content Hash: {result.get('content_hash')}")
                print("-" * 50)
            
            # Extract features
            features = monitor.extract_monitoring_features(results)
            print(f"ML Features: {features}")
    
    asyncio.run(test_monitor())
