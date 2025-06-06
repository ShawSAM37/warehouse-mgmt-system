"""
Warehouse Management System Dashboard Module
============================================

A comprehensive, modular dashboard system for real-time warehouse operations monitoring.

This module provides:
- Real-time inventory tracking and analytics
- FIFO efficiency monitoring
- Supplier performance metrics
- Advanced alert management
- Export functionality (CSV/PDF)
- Responsive design with theme support

Components:
- api_client: FastAPI backend integration
- components: Streamlit UI components
- charts: Data visualization and plotting
- enhancements: Production-grade features
- utils: Shared utilities and configuration

Author: WMS Development Team
Version: 1.0.0
License: MIT
"""

# Version information
__version__ = "1.0.0"
__author__ = "WMS Development Team"
__license__ = "MIT"

# Import core classes and functions
from .api_client import WMSAPIClient
from .components import (
    render_header,
    render_sidebar, 
    render_kpi_cards,
    render_alerts
)
from .charts import (
    render_dynamic_charts,
    render_fifo_efficiency,
    render_alerts_table,
    render_revenue_metrics
)
from .enhancements import (
    DashboardEnhancements,
    ExportManager,
    UserPreferences
)
from .utils.config import DashboardConfig
from .utils.cache import CacheManager
from .utils.helpers import (
    format_number,
    get_priority_color,
    create_gauge_chart,
    safe_get,
    check_api_health
)

# Module-level constants
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_REFRESH_INTERVAL = 30
SUPPORTED_THEMES = ["light", "dark"]
SUPPORTED_EXPORT_FORMATS = ["csv", "pdf", "excel"]

# Color scheme constants
COLOR_SCHEME = {
    'primary': '#1f77b4',
    'success': '#2ca02c',
    'warning': '#ff7f0e', 
    'danger': '#d62728',
    'neutral': '#7f7f7f',
    'background': '#ffffff',
    'surface': '#f8f9fa',
    'text': '#212529'
}

# Export public API
__all__ = [
    # Core classes
    "WMSAPIClient",
    "DashboardConfig", 
    "CacheManager",
    "DashboardEnhancements",
    "ExportManager",
    "UserPreferences",
    
    # UI Components
    "render_header",
    "render_sidebar",
    "render_kpi_cards", 
    "render_alerts",
    
    # Chart functions
    "render_dynamic_charts",
    "render_fifo_efficiency",
    "render_alerts_table",
    "render_revenue_metrics",
    
    # Utility functions
    "format_number",
    "get_priority_color",
    "create_gauge_chart",
    "safe_get",
    "check_api_health",
    
    # Constants
    "COLOR_SCHEME",
    "DEFAULT_API_URL",
    "DEFAULT_REFRESH_INTERVAL",
    "SUPPORTED_THEMES",
    "SUPPORTED_EXPORT_FORMATS",
    
    # Version info
    "__version__",
    "__author__",
    "__license__"
]

# Module initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"WMS Dashboard module v{__version__} initialized")
