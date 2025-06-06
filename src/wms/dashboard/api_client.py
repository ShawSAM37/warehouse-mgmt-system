"""
WMS API Client for Dashboard Integration
=======================================

Provides comprehensive API client for communicating with the FastAPI backend,
including caching, error handling, retry logic, and health monitoring.
"""

import requests
import streamlit as st
import logging
import time
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .utils.config import DashboardConfig

logger = logging.getLogger(__name__)

class WMSAPIClient:
    """
    API client for WMS backend integration with comprehensive error handling.
    
    Features:
    - Automatic retry with exponential backoff
    - Request/response caching with Streamlit
    - Comprehensive error handling and user feedback
    - Session management with connection pooling
    - Health monitoring and connectivity testing
    - Performance metrics tracking
    """
    
    def __init__(self, config: DashboardConfig):
        """
        Initialize the API client with configuration.
        
        Args:
            config: Dashboard configuration instance
        """
        self.config = config
        self.base_url = config.api_base_url.rstrip('/')
        self.timeout = config.api_timeout
        self.retry_attempts = config.api_retry_attempts
        self.retry_delay = config.api_retry_delay
        
        # Initialize session with retry strategy
        self.session = requests.Session()
        self._setup_session()
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = None
        
        logger.info(f"WMS API Client initialized with base URL: {self.base_url}")
    
    def _setup_session(self) -> None:
        """Setup session with retry strategy and headers."""
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.retry_attempts,
            backoff_factor=self.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.config.request_pool_size,
            pool_maxsize=self.config.max_concurrent_requests
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': f'WMS-Dashboard/1.0.0',
            'X-Client-Version': '1.0.0',
            'X-Request-ID': f'dashboard-{datetime.now().isoformat()}'
        })
        
        if self.config.enable_compression:
            self.session.headers['Accept-Encoding'] = 'gzip, deflate'
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None, method: str = 'GET') -> Optional[Dict]:
        """
        Make HTTP request with comprehensive error handling.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            method: HTTP method (GET, POST, etc.)
            
        Returns:
            Response data or None if error occurred
        """
        url = f"{self.base_url}{endpoint}"
        self.request_count += 1
        self.last_request_time = datetime.now()
        
        try:
            logger.debug(f"Making {method} request to: {url} with params: {params}")
            
            # Make request based on method
            if method.upper() == 'GET':
                response = self.session.get(url, params=params, timeout=self.timeout)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=params, timeout=self.timeout)
            else:
                response = self.session.request(method, url, params=params, timeout=self.timeout)
            
            response.raise_for_status()
            
            # Parse JSON response
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for {endpoint}: {e}")
                st.error("📄 Invalid JSON response from API")
                self.error_count += 1
                return None
            
            logger.debug(f"Successfully received data from {endpoint}")
            return data
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"🔌 Cannot connect to WMS API at {self.base_url}"
            logger.error(f"Connection error for {endpoint}: {e}")
            st.error(error_msg)
            self.error_count += 1
            return None
            
        except requests.exceptions.Timeout as e:
            error_msg = f"⏱️ API request timed out after {self.timeout} seconds"
            logger.error(f"Timeout error for {endpoint}: {e}")
            st.error(error_msg)
            self.error_count += 1
            return None
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else 'Unknown'
            
            if status_code == 404:
                error_msg = f"🚨 API Error: Endpoint not found ({status_code})"
            elif status_code == 500:
                error_msg = f"🚨 API Error: Internal server error ({status_code})"
            elif status_code == 503:
                error_msg = f"🚨 API Error: Service unavailable ({status_code})"
            else:
                error_msg = f"🚨 API Error: {status_code}"
            
            logger.error(f"HTTP error for {endpoint}: {e}")
            st.error(error_msg)
            self.error_count += 1
            return None
            
        except Exception as e:
            error_msg = f"❌ Unexpected error: {str(e)}"
            logger.error(f"Unexpected error for {endpoint}: {e}")
            st.error(error_msg)
            self.error_count += 1
            return None
    
    @st.cache_data(ttl=60, show_spinner=False)
    def get_inventory_summary(_self) -> Optional[Dict]:
        """
        Get inventory summary with 1-minute caching.
        
        Returns:
            Dict containing inventory summary data:
            - total_products: Total number of products
            - total_batches: Total number of batches
            - total_quantity: Total inventory quantity
            - low_stock_count: Number of low stock items
            - expiring_soon_count: Number of expiring items
        """
        return _self._make_request("/reporting/inventory-summary")
    
    @st.cache_data(ttl=30, show_spinner=False)
    def get_stock_levels(_self, limit: int = 100, product_codes: Optional[List[str]] = None) -> Optional[List[Dict]]:
        """
        Get current stock levels with 30-second caching.
        
        Args:
            limit: Maximum number of records to return
            product_codes: Optional list of product codes to filter
            
        Returns:
            List of stock level dictionaries
        """
        params = {"limit": limit}
        if product_codes:
            params["product_codes"] = ",".join(product_codes)
        
        return _self._make_request("/reporting/stock-levels", params)
    
    @st.cache_data(ttl=300, show_spinner=False)
    def get_supplier_performance(_self, days: int = 90, supplier_ids: Optional[List[str]] = None) -> Optional[List[Dict]]:
        """
        Get supplier performance metrics with 5-minute caching.
        
        Args:
            days: Number of days to analyze
            supplier_ids: Optional list of supplier IDs to filter
            
        Returns:
            List of supplier performance dictionaries
        """
        params = {"days": days}
        if supplier_ids:
            params["supplier_ids"] = ",".join(supplier_ids)
        
        return _self._make_request("/reporting/supplier-performance", params)
    
    @st.cache_data(ttl=120, show_spinner=False)
    def get_storage_utilization(_self, zone: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Get storage utilization data with 2-minute caching.
        
        Args:
            zone: Optional storage zone filter
            
        Returns:
            List of storage utilization dictionaries
        """
        params = {}
        if zone and zone != "All Zones":
            params["zone"] = zone
        
        return _self._make_request("/reporting/storage-utilization", params)
    
    @st.cache_data(ttl=60, show_spinner=False)
    def get_expiry_alerts(_self, days_ahead: int = 30, priority: Optional[List[str]] = None) -> Optional[List[Dict]]:
        """
        Get expiry alerts with 1-minute caching.
        
        Args:
            days_ahead: Days ahead to check for expiry
            priority: Optional priority filter list
            
        Returns:
            List of expiry alert dictionaries
        """
        params = {"days_ahead": days_ahead}
        if priority:
            params["priority"] = ",".join(priority)
        
        return _self._make_request("/reporting/expiry-alerts", params)
    
    @st.cache_data(ttl=300, show_spinner=False)
    def get_consumption_trends(_self, days: int = 30) -> Optional[List[Dict]]:
        """
        Get consumption trends with 5-minute caching.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            List of consumption trend data points
        """
        return _self._make_request("/reporting/consumption-trends", {"days": days})
    
    @st.cache_data(ttl=300, show_spinner=False)
    def get_fifo_efficiency(_self, product_codes: Optional[List[str]] = None) -> Optional[List[Dict]]:
        """
        Get FIFO efficiency report with 5-minute caching.
        
        Args:
            product_codes: Optional list of product codes to filter
            
        Returns:
            List of FIFO efficiency dictionaries
        """
        params = {}
        if product_codes:
            params["product_codes"] = ",".join(product_codes)
        
        return _self._make_request("/reporting/fifo-efficiency", params)
    
    def check_api_health(self) -> Dict[str, Any]:
        """
        Check API health and connectivity.
        
        Returns:
            Dictionary with health status information
        """
        start_time = time.time()
        
        try:
            response = self.session.get(
                f"{self.base_url}/health", 
                timeout=5
            )
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            if response.status_code == 200:
                try:
                    health_data = response.json()
                    return {
                        "status": "healthy",
                        "response_time_ms": round(response_time, 2),
                        "api_version": health_data.get("version", "unknown"),
                        "timestamp": datetime.now().isoformat(),
                        "details": health_data
                    }
                except json.JSONDecodeError:
                    return {
                        "status": "healthy",
                        "response_time_ms": round(response_time, 2),
                        "api_version": "unknown",
                        "timestamp": datetime.now().isoformat(),
                        "details": {"message": "Health endpoint accessible"}
                    }
            else:
                return {
                    "status": "unhealthy",
                    "response_time_ms": round(response_time, 2),
                    "error": f"HTTP {response.status_code}",
                    "timestamp": datetime.now().isoformat()
                }
                
        except requests.exceptions.RequestException as e:
            response_time = (time.time() - start_time) * 1000
            return {
                "status": "unreachable",
                "response_time_ms": round(response_time, 2),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get client performance metrics.
        
        Returns:
            Dictionary with performance statistics
        """
        error_rate = (self.error_count / self.request_count * 100) if self.request_count > 0 else 0
        
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate_percent": round(error_rate, 2),
            "last_request_time": self.last_request_time.isoformat() if self.last_request_time else None,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout,
            "retry_attempts": self.retry_attempts
        }
    
    def clear_cache(self) -> None:
        """Clear all cached API responses."""
        try:
            st.cache_data.clear()
            logger.info("API cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def test_all_endpoints(self) -> Dict[str, bool]:
        """
        Test connectivity to all API endpoints.
        
        Returns:
            Dictionary mapping endpoint names to success status
        """
        endpoints = {
            "health": "/health",
            "inventory_summary": "/reporting/inventory-summary",
            "stock_levels": "/reporting/stock-levels",
            "supplier_performance": "/reporting/supplier-performance",
            "storage_utilization": "/reporting/storage-utilization",
            "expiry_alerts": "/reporting/expiry-alerts",
            "consumption_trends": "/reporting/consumption-trends",
            "fifo_efficiency": "/reporting/fifo-efficiency"
        }
        
        results = {}
        for name, endpoint in endpoints.items():
            try:
                response = self.session.get(
                    f"{self.base_url}{endpoint}",
                    timeout=5,
                    params={"limit": 1} if "reporting" in endpoint else None
                )
                results[name] = response.status_code == 200
            except Exception:
                results[name] = False
        
        return results
    
    def __del__(self):
        """Cleanup when client is destroyed."""
        if hasattr(self, 'session'):
            self.session.close()

# Global API client factory
@st.cache_resource
def get_api_client() -> WMSAPIClient:
    """
    Get cached API client instance.
    
    Returns:
        Configured WMS API client
    """
    from .utils.config import config
    return WMSAPIClient(config)

# Health check utility function
def check_api_health(api_url: str = None) -> bool:
    """
    Quick health check utility function.
    
    Args:
        api_url: Optional API URL override
        
    Returns:
        True if API is healthy, False otherwise
    """
    if api_url is None:
        from .utils.config import config
        api_url = config.api_base_url
    
    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False
