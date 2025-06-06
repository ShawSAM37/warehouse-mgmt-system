
"""
Dashboard Configuration Management
=================================

Centralized configuration management for the WMS dashboard with support for
environment variables, validation, and different deployment environments.
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class DashboardConfig:
    """
    Comprehensive configuration management for the WMS dashboard.
    
    Supports environment variables, validation, and different deployment modes.
    All settings can be overridden via environment variables with WMS_ prefix.
    """
    
    # API Configuration
    api_base_url: str = field(default_factory=lambda: os.getenv('WMS_API_URL', 'http://localhost:8000'))
    api_timeout: int = field(default_factory=lambda: int(os.getenv('WMS_API_TIMEOUT', '10')))
    api_retry_attempts: int = field(default_factory=lambda: int(os.getenv('WMS_API_RETRIES', '3')))
    api_retry_delay: float = field(default_factory=lambda: float(os.getenv('WMS_API_RETRY_DELAY', '1.0')))
    
    # Dashboard Settings
    refresh_interval: int = field(default_factory=lambda: int(os.getenv('WMS_REFRESH_INTERVAL', '30')))
    default_theme: str = field(default_factory=lambda: os.getenv('WMS_THEME', 'light'))
    page_title: str = field(default_factory=lambda: os.getenv('WMS_PAGE_TITLE', 'WMS Dashboard'))
    page_icon: str = field(default_factory=lambda: os.getenv('WMS_PAGE_ICON', '📦'))
    
    # Caching Configuration
    cache_ttl_default: int = field(default_factory=lambda: int(os.getenv('WMS_CACHE_TTL', '300')))
    cache_ttl_inventory: int = field(default_factory=lambda: int(os.getenv('WMS_CACHE_TTL_INVENTORY', '60')))
    cache_ttl_alerts: int = field(default_factory=lambda: int(os.getenv('WMS_CACHE_TTL_ALERTS', '30')))
    cache_ttl_charts: int = field(default_factory=lambda: int(os.getenv('WMS_CACHE_TTL_CHARTS', '300')))
    cache_max_size: int = field(default_factory=lambda: int(os.getenv('WMS_CACHE_MAX_SIZE', '100')))
    
    # Performance Settings
    max_concurrent_requests: int = field(default_factory=lambda: int(os.getenv('WMS_MAX_CONCURRENT', '10')))
    request_pool_size: int = field(default_factory=lambda: int(os.getenv('WMS_POOL_SIZE', '20')))
    enable_compression: bool = field(default_factory=lambda: os.getenv('WMS_ENABLE_COMPRESSION', 'true').lower() == 'true')
    
    # Export Configuration
    export_temp_dir: str = field(default_factory=lambda: os.getenv('WMS_EXPORT_DIR', '/tmp/wms_exports'))
    export_max_file_size: int = field(default_factory=lambda: int(os.getenv('WMS_EXPORT_MAX_SIZE', '50')))  # MB
    export_retention_days: int = field(default_factory=lambda: int(os.getenv('WMS_EXPORT_RETENTION', '7')))
    
    # Email Configuration for Reports
    smtp_server: str = field(default_factory=lambda: os.getenv('WMS_SMTP_SERVER', 'smtp.gmail.com'))
    smtp_port: int = field(default_factory=lambda: int(os.getenv('WMS_SMTP_PORT', '587')))
    smtp_use_tls: bool = field(default_factory=lambda: os.getenv('WMS_SMTP_TLS', 'true').lower() == 'true')
    email_user: str = field(default_factory=lambda: os.getenv('WMS_EMAIL_USER', ''))
    email_password: str = field(default_factory=lambda: os.getenv('WMS_EMAIL_PASSWORD', ''))
    email_from_name: str = field(default_factory=lambda: os.getenv('WMS_EMAIL_FROM_NAME', 'WMS Dashboard'))
    
    # Security Settings
    enable_debug: bool = field(default_factory=lambda: os.getenv('WMS_DEBUG', 'false').lower() == 'true')
    log_level: str = field(default_factory=lambda: os.getenv('WMS_LOG_LEVEL', 'INFO'))
    session_timeout: int = field(default_factory=lambda: int(os.getenv('WMS_SESSION_TIMEOUT', '3600')))
    
    # Feature Flags
    enable_real_time: bool = field(default_factory=lambda: os.getenv('WMS_ENABLE_REALTIME', 'true').lower() == 'true')
    enable_exports: bool = field(default_factory=lambda: os.getenv('WMS_ENABLE_EXPORTS', 'true').lower() == 'true')
    enable_email_reports: bool = field(default_factory=lambda: os.getenv('WMS_ENABLE_EMAIL', 'false').lower() == 'true')
    enable_advanced_charts: bool = field(default_factory=lambda: os.getenv('WMS_ENABLE_ADVANCED_CHARTS', 'true').lower() == 'true')
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._setup_logging()
        logger.info("Dashboard configuration initialized successfully")
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Validate API URL
        if not self.api_base_url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid API URL: {self.api_base_url}")
        
        # Validate timeouts and intervals
        if self.api_timeout <= 0:
            raise ValueError("API timeout must be positive")
        
        if self.refresh_interval < 5:
            raise ValueError("Refresh interval must be at least 5 seconds")
        
        # Validate theme
        if self.default_theme not in ['light', 'dark']:
            logger.warning(f"Unknown theme '{self.default_theme}', using 'light'")
            self.default_theme = 'light'
        
        # Validate cache settings
        if self.cache_max_size <= 0:
            raise ValueError("Cache max size must be positive")
        
        # Create export directory if it doesn't exist
        if self.enable_exports:
            os.makedirs(self.export_temp_dir, exist_ok=True)
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def get_api_endpoints(self) -> Dict[str, str]:
        """Get all API endpoint URLs."""
        base = self.api_base_url.rstrip('/')
        return {
            'inventory_summary': f"{base}/reporting/inventory-summary",
            'stock_levels': f"{base}/reporting/stock-levels",
            'supplier_performance': f"{base}/reporting/supplier-performance",
            'storage_utilization': f"{base}/reporting/storage-utilization",
            'expiry_alerts': f"{base}/reporting/expiry-alerts",
            'consumption_trends': f"{base}/reporting/consumption-trends",
            'fifo_efficiency': f"{base}/reporting/fifo-efficiency",
            'health': f"{base}/health"
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration dictionary."""
        return {
            'max_size': self.cache_max_size,
            'default_ttl': self.cache_ttl_default,
            'ttl_inventory': self.cache_ttl_inventory,
            'ttl_alerts': self.cache_ttl_alerts,
            'ttl_charts': self.cache_ttl_charts
        }
    
    def get_email_config(self) -> Dict[str, Any]:
        """Get email configuration for reports."""
        return {
            'smtp_server': self.smtp_server,
            'smtp_port': self.smtp_port,
            'use_tls': self.smtp_use_tls,
            'username': self.email_user,
            'password': self.email_password,
            'from_name': self.email_from_name,
            'enabled': self.enable_email_reports and bool(self.email_user and self.email_password)
        }
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.enable_debug and self.log_level.upper() != 'DEBUG'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)."""
        config_dict = {}
        for key, value in self.__dict__.items():
            # Exclude sensitive information
            if 'password' in key.lower() or 'secret' in key.lower():
                config_dict[key] = '***HIDDEN***'
            else:
                config_dict[key] = value
        return config_dict

# Global configuration instance
config = DashboardConfig()
