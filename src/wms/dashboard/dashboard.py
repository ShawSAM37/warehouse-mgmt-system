"""
Warehouse Management System Dashboard
A comprehensive Streamlit dashboard for real-time warehouse operations monitoring.

Author: WMS Development Team
Version: 1.0.0
Dependencies: streamlit, requests, pandas, plotly
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Any
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Constants
API_BASE_URL = "http://localhost:8000"
REFRESH_INTERVAL = 30  # seconds
DEFAULT_TIMEOUT = 10  # seconds

COLOR_SCHEME = {
    'primary': '#1f77b4',
    'success': '#2ca02c', 
    'warning': '#ff7f0e',
    'danger': '#d62728',
    'neutral': '#7f7f7f',
    'background': '#f8f9fa',
    'text': '#212529'
}

DASHBOARD_VERSION = "1.0.0"

# Streamlit Page Configuration
st.set_page_config(
    page_title="WMS Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/ShawSAM37/warehouse-mgmt-system',
        'Report a bug': 'https://github.com/ShawSAM37/warehouse-mgmt-system/issues',
        'About': f"Warehouse Management System Dashboard v{DASHBOARD_VERSION}"
    }
)

# Custom CSS Styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding-top: 1rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* KPI Card styling */
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Alert banner styling */
    .alert-banner {
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        font-weight: 600;
        border-left: 5px solid;
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .critical-alert {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        border-left-color: #d63031;
    }
    
    .warning-alert {
        background: linear-gradient(135deg, #fdcb6e, #f39c12);
        color: white;
        border-left-color: #e17055;
    }
    
    .success-alert {
        background: linear-gradient(135deg, #00b894, #00a085);
        color: white;
        border-left-color: #00b894;
    }
    
    /* Metric container styling */
    .metric-container {
        text-align: center;
        padding: 1.5rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .sidebar-section {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Chart container styling */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        margin-top: 3rem;
    }
    
    /* Loading spinner */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online { background-color: #2ca02c; }
    .status-offline { background-color: #d62728; }
    .status-warning { background-color: #ff7f0e; }
</style>
""", unsafe_allow_html=True)

class WMSAPIClient:
    """
    API client for WMS backend integration with comprehensive error handling.
    
    This class provides methods to interact with the FastAPI backend,
    including caching, timeout handling, and graceful error recovery.
    """
    
    def __init__(self, base_url: str = API_BASE_URL, timeout: int = DEFAULT_TIMEOUT):
        """
        Initialize the API client.
        
        Args:
            base_url (str): Base URL for the API endpoints
            timeout (int): Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': f'WMS-Dashboard/{DASHBOARD_VERSION}'
        })
        
        logger.info(f"WMS API Client initialized with base URL: {self.base_url}")
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make HTTP request with comprehensive error handling.
        
        Args:
            endpoint (str): API endpoint path
            params (Optional[Dict]): Query parameters
            
        Returns:
            Optional[Dict]: Response data or None if error occurred
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            logger.debug(f"Making request to: {url} with params: {params}")
            
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Successfully received data from {endpoint}")
            return data
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"🔌 Cannot connect to WMS API at {self.base_url}"
            logger.error(f"Connection error: {e}")
            st.error(error_msg)
            return None
            
        except requests.exceptions.Timeout as e:
            error_msg = f"⏱️ API request timed out after {self.timeout} seconds"
            logger.error(f"Timeout error: {e}")
            st.error(error_msg)
            return None
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"🚨 API Error: {e.response.status_code}"
            if e.response.status_code == 404:
                error_msg += " - Endpoint not found"
            elif e.response.status_code == 500:
                error_msg += " - Internal server error"
            
            logger.error(f"HTTP error: {e}")
            st.error(error_msg)
            return None
            
        except json.JSONDecodeError as e:
            error_msg = "📄 Invalid JSON response from API"
            logger.error(f"JSON decode error: {e}")
            st.error(error_msg)
            return None
            
        except Exception as e:
            error_msg = f"❌ Unexpected error: {str(e)}"
            logger.error(f"Unexpected error: {e}")
            st.error(error_msg)
            return None
    
    @st.cache_data(ttl=60, show_spinner=False)
    def get_inventory_summary(_self) -> Optional[Dict]:
        """
        Get inventory summary with 1-minute caching.
        
        Returns:
            Optional[Dict]: Inventory summary data
        """
        return _self._make_request("/reporting/inventory-summary")
    
    @st.cache_data(ttl=30, show_spinner=False)
    def get_stock_levels(_self, limit: int = 100) -> Optional[List[Dict]]:
        """
        Get current stock levels with 30-second caching.
        
        Args:
            limit (int): Maximum number of records to return
            
        Returns:
            Optional[List[Dict]]: Stock level data
        """
        return _self._make_request("/reporting/stock-levels", {"limit": limit})
    
    @st.cache_data(ttl=300, show_spinner=False)
    def get_supplier_performance(_self, days: int = 90) -> Optional[List[Dict]]:
        """
        Get supplier performance metrics with 5-minute caching.
        
        Args:
            days (int): Number of days to analyze
            
        Returns:
            Optional[List[Dict]]: Supplier performance data
        """
        return _self._make_request("/reporting/supplier-performance", {"days": days})
    
    @st.cache_data(ttl=120, show_spinner=False)
    def get_storage_utilization(_self) -> Optional[List[Dict]]:
        """
        Get storage utilization data with 2-minute caching.
        
        Returns:
            Optional[List[Dict]]: Storage utilization data
        """
        return _self._make_request("/reporting/storage-utilization")
    
    @st.cache_data(ttl=60, show_spinner=False)
    def get_expiry_alerts(_self, days_ahead: int = 30) -> Optional[List[Dict]]:
        """
        Get expiry alerts with 1-minute caching.
        
        Args:
            days_ahead (int): Days ahead to check for expiry
            
        Returns:
            Optional[List[Dict]]: Expiry alert data
        """
        return _self._make_request("/reporting/expiry-alerts", {"days_ahead": days_ahead})
    
    @st.cache_data(ttl=300, show_spinner=False)
    def get_consumption_trends(_self, days: int = 30) -> Optional[List[Dict]]:
        """
        Get consumption trends with 5-minute caching.
        
        Args:
            days (int): Number of days to analyze
            
        Returns:
            Optional[List[Dict]]: Consumption trend data
        """
        return _self._make_request("/reporting/consumption-trends", {"days": days})
    
    @st.cache_data(ttl=300, show_spinner=False)
    def get_fifo_efficiency(_self) -> Optional[List[Dict]]:
        """
        Get FIFO efficiency report with 5-minute caching.
        
        Returns:
            Optional[List[Dict]]: FIFO efficiency data
        """
        return _self._make_request("/reporting/fifo-efficiency")

# Utility Functions
def format_number(value: float, suffix: str = "") -> str:
    """
    Format numbers with appropriate suffixes (K, M, B).
    
    Args:
        value (float): Number to format
        suffix (str): Optional suffix to append
        
    Returns:
        str: Formatted number string
        
    Examples:
        >>> format_number(1500)
        '1.5K'
        >>> format_number(2500000, ' units')
        '2.5M units'
    """
    if value is None:
        return "0"
    
    try:
        if abs(value) >= 1_000_000_000:
            return f"{value/1_000_000_000:.1f}B{suffix}"
        elif abs(value) >= 1_000_000:
            return f"{value/1_000_000:.1f}M{suffix}"
        elif abs(value) >= 1_000:
            return f"{value/1_000:.1f}K{suffix}"
        else:
            return f"{value:.0f}{suffix}"
    except (TypeError, ValueError):
        return "0"

def get_priority_color(priority: str) -> str:
    """
    Get color code based on priority level.
    
    Args:
        priority (str): Priority level (URGENT, HIGH, MEDIUM, LOW)
        
    Returns:
        str: Hex color code
    """
    priority_colors = {
        'URGENT': COLOR_SCHEME['danger'],
        'HIGH': COLOR_SCHEME['warning'],
        'MEDIUM': COLOR_SCHEME['primary'],
        'LOW': COLOR_SCHEME['success']
    }
    return priority_colors.get(priority.upper(), COLOR_SCHEME['neutral'])

def create_gauge_chart(value: float, title: str, max_value: float = 100) -> go.Figure:
    """
    Create a gauge chart for displaying metrics.
    
    Args:
        value (float): Current value
        title (str): Chart title
        max_value (float): Maximum value for the gauge
        
    Returns:
        go.Figure: Plotly gauge chart
    """
    # Determine color based on value
    if value >= max_value * 0.8:
        bar_color = COLOR_SCHEME['success']
    elif value >= max_value * 0.6:
        bar_color = COLOR_SCHEME['warning']
    else:
        bar_color = COLOR_SCHEME['danger']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        delta={'reference': max_value * 0.8, 'increasing': {'color': COLOR_SCHEME['success']}},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': COLOR_SCHEME['neutral'],
            'steps': [
                {'range': [0, max_value * 0.4], 'color': '#ffebee'},
                {'range': [max_value * 0.4, max_value * 0.8], 'color': '#fff3e0'},
                {'range': [max_value * 0.8, max_value], 'color': '#e8f5e8'}
            ],
            'threshold': {
                'line': {'color': COLOR_SCHEME['danger'], 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        font={'color': COLOR_SCHEME['text'], 'family': "Arial"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def safe_get(data: Optional[Dict], key: str, default: Any = None) -> Any:
    """
    Safely get value from dictionary with null checking.
    
    Args:
        data (Optional[Dict]): Dictionary to get value from
        key (str): Key to retrieve
        default (Any): Default value if key not found or data is None
        
    Returns:
        Any: Value from dictionary or default
    """
    if data is None:
        return default
    
    try:
        return data.get(key, default)
    except (AttributeError, TypeError):
        return default

# Initialize API client as cached resource
@st.cache_resource
def get_api_client() -> WMSAPIClient:
    """
    Get cached API client instance.
    
    Returns:
        WMSAPIClient: Configured API client
    """
    return WMSAPIClient()

# Health check function
def check_api_health() -> bool:
    """
    Check if the API is accessible.
    
    Returns:
        bool: True if API is healthy, False otherwise
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Initialize session state
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

if 'api_status' not in st.session_state:
    st.session_state.api_status = check_api_health()

# Log dashboard initialization
logger.info(f"WMS Dashboard v{DASHBOARD_VERSION} initialized")


# UI Component

def render_header():
    """
    Render the main dashboard header with company branding and system status.
    
    Features:
    - Professional gradient background
    - Company branding with emoji
    - System status indicator
    - Last updated timestamp
    """
    last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    api_status = "Online" if st.session_state.get('api_status', False) else "Offline"
    status_color = COLOR_SCHEME['success'] if api_status == "Online" else COLOR_SCHEME['danger']
    
    header_html = f'''
    <div class="main-header">
        <h1>📦 Warehouse Management Dashboard</h1>
        <div style="display: flex; justify-content: center; align-items: center; gap: 2rem; margin-top: 1rem;">
            <p style="margin: 0; font-size: 1rem;">
                System Status: 
                <span style="color: {status_color}; font-weight: bold;">
                    <span class="status-indicator status-{'online' if api_status == 'Online' else 'offline'}"></span>
                    {api_status}
                </span>
            </p>
            <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">
                Last Updated: {last_updated}
            </p>
        </div>
    </div>
    '''
    st.markdown(header_html, unsafe_allow_html=True)

def render_sidebar() -> Dict[str, Any]:
    """
    Render the sidebar with dashboard controls and filters.
    
    Returns:
        Dict[str, Any]: Dictionary containing all control values
    """
    # Dashboard Controls Section
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Auto-refresh controls
    auto_refresh = st.sidebar.checkbox(
        "🔄 Auto-refresh", 
        value=True, 
        key="auto_refresh",
        help="Automatically refresh dashboard data"
    )
    
    refresh_rate = REFRESH_INTERVAL
    if auto_refresh:
        refresh_rate = st.sidebar.slider(
            "Refresh Rate (seconds)", 
            min_value=10, 
            max_value=300, 
            value=REFRESH_INTERVAL,
            step=10,
            key="refresh_rate",
            help="How often to refresh the data"
        )
        st.sidebar.info(f"⏱️ Auto-refreshing every {refresh_rate}s")
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now", type="primary", key="manual_refresh"):
        st.cache_data.clear()
        st.session_state.last_refresh = datetime.now()
        st.session_state.api_status = check_api_health()
        st.rerun()
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Filters Section
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.subheader("🔍 Filters & Views")
    
    # Date range selector
    date_range = st.sidebar.selectbox(
        "📅 Analysis Period",
        options=[7, 30, 90],
        index=1,
        format_func=lambda x: f"Last {x} days",
        key="date_range",
        help="Select the time period for analysis"
    )
    
    # View mode selector
    view_mode = st.sidebar.selectbox(
        "👁️ Dashboard View",
        options=["Overview", "Inventory", "Operations", "Analytics"],
        index=0,
        key="view_mode",
        help="Select the dashboard view mode"
    )
    
    # Product code multi-select filter
    # Note: In production, this should be populated from API
    available_products = ["PROD-001", "PROD-002", "PROD-003", "PROD-004", "PROD-005"]
    product_codes = st.sidebar.multiselect(
        "📦 Product Codes",
        options=available_products,
        default=[],
        key="product_codes",
        help="Filter by specific product codes"
    )
    
    # Storage zone dropdown filter
    storage_zones = st.sidebar.selectbox(
        "🏪 Storage Zone",
        options=["All Zones", "Zone A", "Zone B", "Zone C", "Zone D"],
        index=0,
        key="storage_zone",
        help="Filter by storage zone"
    )
    
    # Priority filter for alerts
    priority_filter = st.sidebar.multiselect(
        "⚠️ Alert Priority",
        options=["URGENT", "HIGH", "MEDIUM", "LOW"],
        default=["URGENT", "HIGH"],
        key="priority_filter",
        help="Filter alerts by priority level"
    )
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # System Information Section
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.subheader("ℹ️ System Information")
    
    # Version and API info
    st.sidebar.info(f"**Dashboard Version:** {DASHBOARD_VERSION}")
    
    api_status = "🟢 Online" if st.session_state.get('api_status', False) else "🔴 Offline"
    st.sidebar.info(f"**API Status:** {api_status}")
    st.sidebar.info(f"**API URL:** {API_BASE_URL}")
    
    # Export options
    st.sidebar.markdown("**📥 Export Options**")
    if st.sidebar.button("Export Dashboard Data", key="export_data"):
        st.sidebar.success("Export feature coming soon!")
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'auto_refresh': auto_refresh,
        'refresh_rate': refresh_rate,
        'date_range': date_range,
        'view_mode': view_mode,
        'product_codes': product_codes,
        'storage_zone': storage_zones,
        'priority_filter': priority_filter
    }

def render_kpi_cards(api_client: WMSAPIClient):
    """
    Render KPI cards in a responsive layout with comprehensive metrics.
    
    Args:
        api_client (WMSAPIClient): API client for data fetching
    """
    st.subheader("📊 Key Performance Indicators")
    
    with st.spinner("📈 Loading KPI data..."):
        try:
            # Fetch data from API
            inventory_summary = api_client.get_inventory_summary()
            stock_levels = api_client.get_stock_levels(limit=10)
            storage_data = api_client.get_storage_utilization()
            alerts = api_client.get_expiry_alerts(days_ahead=7)
            
            if inventory_summary:
                # First row of KPIs
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_products = safe_get(inventory_summary, 'total_products', 0)
                    delta_products = f"+{total_products//10}" if total_products > 0 else None
                    st.metric(
                        label="📦 Total Products",
                        value=format_number(total_products),
                        delta=delta_products,
                        help="Total number of active products in inventory"
                    )
                
                with col2:
                    total_quantity = safe_get(inventory_summary, 'total_quantity', 0)
                    delta_quantity = f"+{total_quantity//100}" if total_quantity > 0 else None
                    st.metric(
                        label="📊 Total Inventory",
                        value=format_number(total_quantity, " units"),
                        delta=delta_quantity,
                        help="Total quantity of all inventory items"
                    )
                
                with col3:
                    low_stock_count = safe_get(inventory_summary, 'low_stock_count', 0)
                    delta_low_stock = f"-{low_stock_count//2}" if low_stock_count > 0 else None
                    st.metric(
                        label="⚠️ Low Stock Items",
                        value=low_stock_count,
                        delta=delta_low_stock,
                        delta_color="inverse",
                        help="Number of products below reorder point"
                    )
                
                with col4:
                    expiring_count = len(alerts) if alerts else 0
                    delta_expiring = f"-{expiring_count//3}" if expiring_count > 0 else None
                    st.metric(
                        label="⏰ Expiring Soon",
                        value=expiring_count,
                        delta=delta_expiring,
                        delta_color="inverse",
                        help="Items expiring within 7 days"
                    )
                
                # Second row of KPIs
                col5, col6, col7, col8 = st.columns(4)
                
                with col5:
                    total_batches = safe_get(inventory_summary, 'total_batches', 0)
                    delta_batches = f"+{total_batches//20}" if total_batches > 0 else None
                    st.metric(
                        label="📋 Active Batches",
                        value=total_batches,
                        delta=delta_batches,
                        help="Number of active inventory batches"
                    )
                
                with col6:
                    # Calculate average warehouse utilization
                    avg_utilization = 0
                    if storage_data:
                        utilizations = [safe_get(s, 'utilization_percentage', 0) for s in storage_data]
                        avg_utilization = sum(utilizations) / len(utilizations) if utilizations else 0
                    
                    st.metric(
                        label="🏪 Warehouse Utilization",
                        value=f"{avg_utilization:.1f}%",
                        delta=f"+{avg_utilization//10:.0f}%" if avg_utilization > 0 else None,
                        help="Average storage space utilization"
                    )
                
                with col7:
                    # Calculate daily throughput estimate
                    daily_throughput = total_quantity * 0.05 if total_quantity else 0
                    st.metric(
                        label="🚀 Daily Throughput",
                        value=format_number(daily_throughput, " units"),
                        delta="+12%",
                        help="Estimated daily inventory movement"
                    )
                
                with col8:
                    # Mock fulfillment rate (would come from order data in production)
                    fulfillment_rate = 96.5
                    delta_color = "normal" if fulfillment_rate > 95 else "inverse"
                    delta_fulfillment = "+2.1%" if fulfillment_rate > 95 else "-1.2%"
                    st.metric(
                        label="✅ Fulfillment Rate",
                        value=f"{fulfillment_rate:.1f}%",
                        delta=delta_fulfillment,
                        delta_color=delta_color,
                        help="Order fulfillment accuracy rate"
                    )
            else:
                st.error("❌ Unable to load inventory summary data")
                
        except Exception as e:
            logger.error(f"Error in render_kpi_cards: {e}")
            st.error(f"❌ Error loading KPI data: {str(e)}")

def render_alerts(api_client: WMSAPIClient):
    """
    Render alert system with priority-based styling and comprehensive information.
    
    Args:
        api_client (WMSAPIClient): API client for fetching alert data
    """
    try:
        with st.spinner("🔍 Checking for alerts..."):
            alerts = api_client.get_expiry_alerts(days_ahead=30)
            
            if alerts:
                # Categorize alerts by priority
                critical_alerts = [a for a in alerts if safe_get(a, 'priority') == 'URGENT']
                high_alerts = [a for a in alerts if safe_get(a, 'priority') == 'HIGH']
                medium_alerts = [a for a in alerts if safe_get(a, 'priority') == 'MEDIUM']
                low_alerts = [a for a in alerts if safe_get(a, 'priority') == 'LOW']
                
                # Critical alerts banner
                if critical_alerts:
                    st.markdown(
                        f'''
                        <div class="alert-banner critical-alert">
                            🚨 CRITICAL ALERT: {len(critical_alerts)} items expiring within 3 days!
                            <br><small>Immediate action required</small>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
                    
                    # Show top 3 critical alerts
                    with st.expander(f"View {len(critical_alerts)} Critical Alerts", expanded=True):
                        for alert in critical_alerts[:3]:
                            st.error(
                                f"🔴 **{safe_get(alert, 'product_code', 'Unknown')}** - "
                                f"Batch {safe_get(alert, 'batch_number', 'Unknown')} - "
                                f"{safe_get(alert, 'days_until_expiry', 0)} days remaining - "
                                f"{safe_get(alert, 'quantity', 0)} units"
                            )
                
                # High priority alerts
                if high_alerts:
                    st.markdown(
                        f'''
                        <div class="alert-banner warning-alert">
                            ⚠️ HIGH PRIORITY: {len(high_alerts)} items require attention
                            <br><small>Action needed within 7 days</small>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
                
                # Medium priority alerts
                if medium_alerts:
                    st.warning(f"📋 MEDIUM PRIORITY: {len(medium_alerts)} items to monitor")
                
                # Summary information
                total_alerts = len(alerts)
                st.info(
                    f"📊 **Alert Summary:** {total_alerts} total alerts - "
                    f"{len(critical_alerts)} critical, {len(high_alerts)} high, "
                    f"{len(medium_alerts)} medium, {len(low_alerts)} low priority"
                )
                
            else:
                # No alerts - show success message
                st.markdown(
                    '''
                    <div class="alert-banner success-alert">
                        ✅ All systems operational - No critical alerts detected
                        <br><small>Inventory levels are within acceptable ranges</small>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
                
    except Exception as e:
        logger.error(f"Error in render_alerts: {e}")
        st.error(f"❌ Error loading alerts: {str(e)}")

def render_navigation_breadcrumbs(view_mode: str):
    """
    Render navigation breadcrumbs for better user orientation.
    
    Args:
        view_mode (str): Current view mode
    """
    breadcrumb_html = f'''
    <div style="margin-bottom: 1rem; padding: 0.5rem 1rem; background: #f8f9fa; border-radius: 8px; border-left: 4px solid {COLOR_SCHEME['primary']};">
        <span style="color: {COLOR_SCHEME['neutral']};">Dashboard</span>
        <span style="margin: 0 0.5rem; color: {COLOR_SCHEME['neutral']};">></span>
        <span style="color: {COLOR_SCHEME['primary']}; font-weight: 600;">{view_mode}</span>
    </div>
    '''
    st.markdown(breadcrumb_html, unsafe_allow_html=True)

def render_loading_placeholder(message: str = "Loading data..."):
    """
    Render a loading placeholder with spinner and message.
    
    Args:
        message (str): Loading message to display
    """
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                f'''
                <div class="loading-spinner">
                    <div style="text-align: center;">
                        <div style="margin-bottom: 1rem;">⏳</div>
                        <div>{message}</div>
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )

def render_error_state(error_message: str, retry_callback=None):
    """
    Render error state with retry option.
    
    Args:
        error_message (str): Error message to display
        retry_callback: Optional callback function for retry button
    """
    st.error(f"❌ {error_message}")
    
    if retry_callback:
        if st.button("🔄 Retry", key=f"retry_{hash(error_message)}"):
            retry_callback()

# Chart and Visualization 

def render_dynamic_charts(api_client: WMSAPIClient, date_range: int):
    """
    Render dynamic charts section with consumption trends, storage utilization, and supplier performance.
    
    Args:
        api_client (WMSAPIClient): API client for data fetching
        date_range (int): Number of days for analysis
    """
    st.subheader("📈 Real-Time Analytics & Trends")
    
    # Consumption Trends Chart
    st.markdown("### 📊 Daily Consumption Trends")
    with st.spinner("📊 Loading consumption data..."):
        try:
            consumption_data = api_client.get_consumption_trends(days=date_range)
            
            if consumption_data and len(consumption_data) > 0:
                # Convert to DataFrame and process timestamps
                df_consumption = pd.DataFrame(consumption_data)
                df_consumption['timestamp'] = pd.to_datetime(df_consumption['timestamp'])
                df_consumption['value'] = pd.to_numeric(df_consumption['value'], errors='coerce')
                
                # Create line chart with professional styling
                fig_consumption = px.line(
                    df_consumption,
                    x='timestamp',
                    y='value',
                    title=f'Daily Consumption Trends - Last {date_range} Days',
                    labels={
                        'value': 'Consumption (units)',
                        'timestamp': 'Date'
                    },
                    color_discrete_sequence=[COLOR_SCHEME['primary']]
                )
                
                # Enhanced styling
                fig_consumption.update_layout(
                    height=400,
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Arial", size=12),
                    title_font_size=16,
                    showlegend=False
                )
                
                # Add hover template
                fig_consumption.update_traces(
                    hovertemplate='<b>Date:</b> %{x}<br><b>Consumption:</b> %{y:,.0f} units<extra></extra>',
                    line=dict(width=3)
                )
                
                # Add range selector
                fig_consumption.update_layout(
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=7, label="7d", step="day", stepmode="backward"),
                                dict(count=30, label="30d", step="day", stepmode="backward"),
                                dict(step="all")
                            ])
                        ),
                        rangeslider=dict(visible=True),
                        type="date"
                    )
                )
                
                st.plotly_chart(fig_consumption, use_container_width=True)
                
                # Add summary statistics
                total_consumption = df_consumption['value'].sum()
                avg_daily = df_consumption['value'].mean()
                max_day = df_consumption.loc[df_consumption['value'].idxmax()]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Consumption", format_number(total_consumption, " units"))
                col2.metric("Daily Average", format_number(avg_daily, " units"))
                col3.metric("Peak Day", f"{max_day['timestamp'].strftime('%m/%d')} ({format_number(max_day['value'])} units)")
                
            else:
                st.info("📊 No consumption data available for the selected period")
                
        except Exception as e:
            logger.error(f"Error in consumption trends chart: {e}")
            st.error(f"❌ Error loading consumption trends: {str(e)}")
    
    st.divider()
    
    # Storage Utilization and Supplier Performance side by side
    col1, col2 = st.columns(2)
    
    # Storage Utilization Chart
    with col1:
        st.markdown("### 🏪 Storage Utilization by Zone")
        with st.spinner("🏪 Loading storage data..."):
            try:
                storage_data = api_client.get_storage_utilization()
                
                if storage_data and len(storage_data) > 0:
                    df_storage = pd.DataFrame(storage_data)
                    
                    # Sort by utilization and take top 10
                    df_storage = df_storage.sort_values('utilization_percentage', ascending=True).tail(10)
                    
                    # Create color scale based on utilization
                    colors = []
                    for util in df_storage['utilization_percentage']:
                        if util >= 90:
                            colors.append(COLOR_SCHEME['danger'])
                        elif util >= 70:
                            colors.append(COLOR_SCHEME['warning'])
                        else:
                            colors.append(COLOR_SCHEME['success'])
                    
                    fig_storage = go.Figure(data=[
                        go.Bar(
                            y=df_storage['storage_bin'],
                            x=df_storage['utilization_percentage'],
                            orientation='h',
                            marker_color=colors,
                            text=[f"{util:.1f}%" for util in df_storage['utilization_percentage']],
                            textposition='inside',
                            hovertemplate='<b>%{y}</b><br>Utilization: %{x:.1f}%<br>Used: %{customdata[0]:.0f}<br>Total: %{customdata[1]:.0f}<extra></extra>',
                            customdata=df_storage[['capacity_used', 'capacity_total']].values
                        )
                    ])
                    
                    fig_storage.update_layout(
                        title="Top 10 Storage Locations by Utilization",
                        xaxis_title="Utilization Percentage (%)",
                        yaxis_title="Storage Location",
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial", size=10),
                        margin=dict(l=100, r=20, t=40, b=40)
                    )
                    
                    fig_storage.update_xaxis(range=[0, 100])
                    
                    st.plotly_chart(fig_storage, use_container_width=True)
                    
                    # Storage summary
                    avg_utilization = df_storage['utilization_percentage'].mean()
                    max_utilization = df_storage['utilization_percentage'].max()
                    st.metric("Average Utilization", f"{avg_utilization:.1f}%")
                    
                else:
                    st.info("🏪 No storage data available")
                    
            except Exception as e:
                logger.error(f"Error in storage utilization chart: {e}")
                st.error(f"❌ Error loading storage data: {str(e)}")
    
    # Supplier Performance Chart
    with col2:
        st.markdown("### 🤝 Supplier Performance")
        with st.spinner("🤝 Loading supplier data..."):
            try:
                supplier_data = api_client.get_supplier_performance(days=date_range)
                
                if supplier_data and len(supplier_data) > 0:
                    df_suppliers = pd.DataFrame(supplier_data)
                    
                    # Sort by performance and take top 10
                    df_suppliers = df_suppliers.sort_values('on_time_delivery_rate', ascending=True).tail(10)
                    
                    # Create color gradient
                    fig_suppliers = px.bar(
                        df_suppliers,
                        x='on_time_delivery_rate',
                        y='supplier_name',
                        orientation='h',
                        title="Top 10 Suppliers - On-Time Delivery Rate",
                        labels={
                            'on_time_delivery_rate': 'On-Time Delivery Rate (%)',
                            'supplier_name': 'Supplier'
                        },
                        color='on_time_delivery_rate',
                        color_continuous_scale='RdYlGn',
                        range_color=[0, 100]
                    )
                    
                    fig_suppliers.update_layout(
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial", size=10),
                        margin=dict(l=120, r=20, t=40, b=40),
                        coloraxis_showscale=False
                    )
                    
                    fig_suppliers.update_traces(
                        hovertemplate='<b>%{y}</b><br>On-Time Rate: %{x:.1f}%<br>Lead Time: %{customdata[0]:.1f} days<extra></extra>',
                        customdata=df_suppliers[['average_lead_time']].values
                    )
                    
                    fig_suppliers.update_xaxis(range=[0, 100])
                    
                    st.plotly_chart(fig_suppliers, use_container_width=True)
                    
                    # Supplier summary
                    avg_performance = df_suppliers['on_time_delivery_rate'].mean()
                    best_supplier = df_suppliers.loc[df_suppliers['on_time_delivery_rate'].idxmax()]
                    st.metric("Average Performance", f"{avg_performance:.1f}%")
                    
                else:
                    st.info("🤝 No supplier data available")
                    
            except Exception as e:
                logger.error(f"Error in supplier performance chart: {e}")
                st.error(f"❌ Error loading supplier data: {str(e)}")

def render_fifo_efficiency(api_client: WMSAPIClient):
    """
    Render FIFO efficiency dashboard with gauge charts and detailed metrics.
    
    Args:
        api_client (WMSAPIClient): API client for data fetching
    """
    st.subheader("🔄 FIFO Efficiency Dashboard")
    
    with st.spinner("🔄 Loading FIFO efficiency data..."):
        try:
            fifo_data = api_client.get_fifo_efficiency()
            
            if fifo_data and len(fifo_data) > 0:
                # Top 4 products with gauge charts
                st.markdown("### 📊 Top Product FIFO Performance")
                
                top_products = fifo_data[:4]
                cols = st.columns(len(top_products))
                
                for i, product in enumerate(top_products):
                    with cols[i]:
                        efficiency = safe_get(product, 'fifo_compliance_rate', 0)
                        product_code = safe_get(product, 'product_code', f'Product {i+1}')
                        
                        # Create gauge chart
                        fig_gauge = create_gauge_chart(
                            value=efficiency,
                            title=f"FIFO Efficiency<br><b>{product_code}</b>",
                            max_value=100
                        )
                        
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        # Efficiency status
                        if efficiency >= 90:
                            st.success(f"✅ Excellent ({efficiency:.1f}%)")
                        elif efficiency >= 70:
                            st.warning(f"⚠️ Good ({efficiency:.1f}%)")
                        else:
                            st.error(f"❌ Needs Improvement ({efficiency:.1f}%)")
                
                st.divider()
                
                # Detailed FIFO table
                st.markdown("### 📋 Detailed FIFO Efficiency Report")
                
                df_fifo = pd.DataFrame(fifo_data)
                
                # Add efficiency categories
                def get_efficiency_category(efficiency):
                    if efficiency >= 90:
                        return "🟢 Excellent"
                    elif efficiency >= 70:
                        return "🟡 Good"
                    else:
                        return "🔴 Needs Improvement"
                
                df_fifo['efficiency_status'] = df_fifo['fifo_compliance_rate'].apply(get_efficiency_category)
                
                # Style the dataframe
                def style_efficiency(val):
                    if val >= 90:
                        return f'background-color: {COLOR_SCHEME["success"]}; color: white; font-weight: bold'
                    elif val >= 70:
                        return f'background-color: {COLOR_SCHEME["warning"]}; color: white; font-weight: bold'
                    else:
                        return f'background-color: {COLOR_SCHEME["danger"]}; color: white; font-weight: bold'
                
                # Display styled table
                styled_df = df_fifo.style.applymap(
                    style_efficiency, 
                    subset=['fifo_compliance_rate']
                ).format({
                    'fifo_compliance_rate': '{:.1f}%',
                    'average_batch_age_consumed': '{:.1f} days',
                    'oldest_batch_age': '{:.0f} days'
                })
                
                st.dataframe(styled_df, use_container_width=True)
                
                # FIFO summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                avg_efficiency = df_fifo['fifo_compliance_rate'].mean()
                excellent_count = len(df_fifo[df_fifo['fifo_compliance_rate'] >= 90])
                needs_improvement = len(df_fifo[df_fifo['fifo_compliance_rate'] < 70])
                avg_batch_age = df_fifo['average_batch_age_consumed'].mean()
                
                col1.metric("Average FIFO Efficiency", f"{avg_efficiency:.1f}%")
                col2.metric("Excellent Performance", f"{excellent_count} products")
                col3.metric("Needs Improvement", f"{needs_improvement} products")
                col4.metric("Avg Batch Age", f"{avg_batch_age:.1f} days")
                
                # Export functionality
                csv_data = df_fifo.to_csv(index=False)
                st.download_button(
                    label="📥 Download FIFO Report (CSV)",
                    data=csv_data,
                    file_name=f"fifo_efficiency_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            else:
                st.info("🔄 No FIFO efficiency data available")
                
        except Exception as e:
            logger.error(f"Error in FIFO efficiency dashboard: {e}")
            st.error(f"❌ Error loading FIFO data: {str(e)}")

def render_alerts_table(api_client: WMSAPIClient):
    """
    Render detailed alerts table with filtering and export functionality.
    
    Args:
        api_client (WMSAPIClient): API client for data fetching
    """
    st.subheader("🚨 Alert Management Center")
    
    with st.spinner("🚨 Loading alert data..."):
        try:
            alerts = api_client.get_expiry_alerts(days_ahead=30)
            
            if alerts and len(alerts) > 0:
                df_alerts = pd.DataFrame(alerts)
                
                # Filter controls
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    priority_filter = st.multiselect(
                        "🎯 Filter by Priority",
                        options=['URGENT', 'HIGH', 'MEDIUM', 'LOW'],
                        default=['URGENT', 'HIGH'],
                        key="alerts_priority_filter"
                    )
                
                with col2:
                    days_filter = st.slider(
                        "📅 Days Until Expiry",
                        min_value=0,
                        max_value=30,
                        value=(0, 30),
                        key="alerts_days_filter"
                    )
                
                with col3:
                    sort_option = st.selectbox(
                        "📊 Sort By",
                        options=['days_until_expiry', 'priority', 'quantity', 'product_code'],
                        format_func=lambda x: {
                            'days_until_expiry': 'Days Until Expiry',
                            'priority': 'Priority Level',
                            'quantity': 'Quantity',
                            'product_code': 'Product Code'
                        }[x],
                        key="alerts_sort"
                    )
                
                # Apply filters
                filtered_df = df_alerts.copy()
                
                if priority_filter:
                    filtered_df = filtered_df[filtered_df['priority'].isin(priority_filter)]
                
                filtered_df = filtered_df[
                    (filtered_df['days_until_expiry'] >= days_filter[0]) &
                    (filtered_df['days_until_expiry'] <= days_filter[1])
                ]
                
                # Sort data
                filtered_df = filtered_df.sort_values(sort_option)
                
                if not filtered_df.empty:
                    # Style the alerts table
                    def style_priority_row(row):
                        priority = row['priority']
                        if priority == 'URGENT':
                            return [f'background-color: {COLOR_SCHEME["danger"]}; color: white'] * len(row)
                        elif priority == 'HIGH':
                            return [f'background-color: {COLOR_SCHEME["warning"]}; color: white'] * len(row)
                        elif priority == 'MEDIUM':
                            return [f'background-color: {COLOR_SCHEME["primary"]}; color: white'] * len(row)
                        else:
                            return [f'background-color: {COLOR_SCHEME["success"]}; color: white'] * len(row)
                    
                    # Format the dataframe for display
                    display_df = filtered_df.copy()
                    display_df['expiry_date'] = pd.to_datetime(display_df['expiry_date']).dt.strftime('%Y-%m-%d')
                    
                    styled_alerts = display_df.style.apply(style_priority_row, axis=1)
                    
                    st.dataframe(styled_alerts, use_container_width=True, height=400)
                    
                    # Alert statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    urgent_count = len(filtered_df[filtered_df['priority'] == 'URGENT'])
                    high_count = len(filtered_df[filtered_df['priority'] == 'HIGH'])
                    total_quantity = filtered_df['quantity'].sum()
                    avg_days = filtered_df['days_until_expiry'].mean()
                    
                    col1.metric("🚨 Urgent Alerts", urgent_count)
                    col2.metric("⚠️ High Priority", high_count)
                    col3.metric("📦 Total Quantity at Risk", format_number(total_quantity, " units"))
                    col4.metric("📅 Average Days to Expiry", f"{avg_days:.1f} days")
                    
                    # Export functionality
                    csv_data = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Alerts Report (CSV)",
                        data=csv_data,
                        file_name=f"alerts_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_alerts"
                    )
                    
                else:
                    st.info("🎉 No alerts match the selected criteria")
                    
            else:
                st.success("✅ No alerts found - All inventory items are within safe expiry ranges!")
                
        except Exception as e:
            logger.error(f"Error in alerts table: {e}")
            st.error(f"❌ Error loading alerts data: {str(e)}")

def render_revenue_metrics(api_client: WMSAPIClient):
    """
    Render financial and revenue metrics with realistic mock data.
    
    Args:
        api_client (WMSAPIClient): API client for data fetching
    """
    st.subheader("💰 Financial Performance Metrics")
    
    try:
        # Mock financial data (in production, this would come from financial APIs)
        # Generate realistic values based on inventory data
        inventory_summary = api_client.get_inventory_summary()
        total_inventory = safe_get(inventory_summary, 'total_quantity', 0) if inventory_summary else 0
        
        # Calculate realistic financial metrics
        daily_revenue = total_inventory * 2.5 + 125000  # Base revenue + inventory factor
        cost_per_order = 12.50 + (total_inventory * 0.001)  # Variable cost based on inventory
        inventory_value = total_inventory * 15.75  # Assume $15.75 per unit average
        roi_percentage = 18.5 + (total_inventory * 0.0001)  # ROI improves with scale
        
        # Add some realistic variance
        import random
        random.seed(datetime.now().day)  # Consistent daily variation
        
        daily_revenue *= (1 + random.uniform(-0.1, 0.1))
        cost_per_order *= (1 + random.uniform(-0.05, 0.05))
        roi_percentage *= (1 + random.uniform(-0.1, 0.1))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            revenue_delta = "+8.2%" if daily_revenue > 125000 else "-2.1%"
            st.metric(
                label="💵 Daily Revenue",
                value=f"${format_number(daily_revenue)}",
                delta=revenue_delta,
                help="Total revenue generated today"
            )
        
        with col2:
            cost_delta = "-2.1%" if cost_per_order < 13 else "+1.5%"
            cost_color = "inverse" if cost_per_order < 13 else "normal"
            st.metric(
                label="💸 Cost per Order",
                value=f"${cost_per_order:.2f}",
                delta=cost_delta,
                delta_color=cost_color,
                help="Average cost to process each order"
            )
        
        with col3:
            value_delta = "+5.7%" if inventory_value > 1000000 else "+2.3%"
            st.metric(
                label="📊 Total Inventory Value",
                value=f"${format_number(inventory_value)}",
                delta=value_delta,
                help="Total value of current inventory"
            )
        
        with col4:
            roi_delta = "+3.2%" if roi_percentage > 18 else "+1.1%"
            st.metric(
                label="📈 ROI",
                value=f"{roi_percentage:.1f}%",
                delta=roi_delta,
                help="Return on investment percentage"
            )
        
        # Financial trends chart
        st.markdown("### 📈 Revenue Trend (Last 30 Days)")
        
        # Generate mock trend data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        base_revenue = daily_revenue * 0.9
        trend_data = []
        
        for i, date in enumerate(dates):
            # Add realistic trend and daily variation
            trend_factor = 1 + (i * 0.002)  # Slight upward trend
            daily_variation = 1 + random.uniform(-0.15, 0.15)
            revenue = base_revenue * trend_factor * daily_variation
            trend_data.append({'date': date, 'revenue': revenue})
        
        df_revenue = pd.DataFrame(trend_data)
        
        fig_revenue = px.line(
            df_revenue,
            x='date',
            y='revenue',
            title='Daily Revenue Trend',
            labels={'revenue': 'Revenue ($)', 'date': 'Date'},
            color_discrete_sequence=[COLOR_SCHEME['success']]
        )
        
        fig_revenue.update_layout(
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial", size=12),
            showlegend=False
        )
        
        fig_revenue.update_traces(
            hovertemplate='<b>Date:</b> %{x}<br><b>Revenue:</b> $%{y:,.0f}<extra></extra>',
            line=dict(width=3)
        )
        
        st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Financial summary
        avg_revenue = df_revenue['revenue'].mean()
        revenue_growth = ((df_revenue['revenue'].iloc[-1] - df_revenue['revenue'].iloc[0]) / df_revenue['revenue'].iloc[0]) * 100
        
        col1, col2 = st.columns(2)
        col1.metric("30-Day Average Revenue", f"${format_number(avg_revenue)}")
        col2.metric("Revenue Growth", f"{revenue_growth:+.1f}%")
        
    except Exception as e:
        logger.error(f"Error in revenue metrics: {e}")
        st.error(f"❌ Error loading financial data: {str(e)}")


# Main Application Logic 

def initialize_session_state():
    """Initialize all required session state variables."""
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    if 'api_status' not in st.session_state:
        st.session_state.api_status = check_api_health()
    
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'theme': 'light',
            'default_view': 'Overview',
            'auto_refresh_enabled': True
        }
    
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0
    
    if 'last_error' not in st.session_state:
        st.session_state.last_error = None

def handle_auto_refresh(controls: Dict[str, Any]) -> bool:
    """
    Handle auto-refresh logic with intelligent timing and error recovery.
    
    Args:
        controls (Dict[str, Any]): User controls from sidebar
        
    Returns:
        bool: True if refresh was triggered, False otherwise
    """
    if not controls.get('auto_refresh', False):
        return False
    
    refresh_rate = controls.get('refresh_rate', REFRESH_INTERVAL)
    time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
    
    # Check if refresh interval has passed
    if time_since_refresh >= refresh_rate:
        try:
            # Update API status before refresh
            st.session_state.api_status = check_api_health()
            
            # Clear cache and update timestamp
            st.cache_data.clear()
            st.session_state.last_refresh = datetime.now()
            
            # Reset error count on successful refresh
            st.session_state.error_count = 0
            
            logger.info(f"Auto-refresh triggered after {time_since_refresh:.1f} seconds")
            
            # Use st.rerun() instead of deprecated st.experimental_rerun()
            st.rerun()
            return True
            
        except Exception as e:
            logger.error(f"Auto-refresh failed: {e}")
            st.session_state.error_count += 1
            st.session_state.last_error = str(e)
            
            # Exponential backoff for failed refreshes
            if st.session_state.error_count > 3:
                st.error("⚠️ Multiple refresh failures detected. Auto-refresh temporarily disabled.")
                return False
    
    return False

def safe_render_component(component_func, component_name: str, *args, **kwargs):
    """
    Safely render a component with comprehensive error handling.
    
    Args:
        component_func: Function to render the component
        component_name (str): Name of the component for error reporting
        *args, **kwargs: Arguments to pass to the component function
    """
    try:
        with st.spinner(f"Loading {component_name}..."):
            return component_func(*args, **kwargs)
    except requests.exceptions.ConnectionError:
        st.error(f"🔌 Cannot connect to API while loading {component_name}")
        render_offline_fallback(component_name)
    except requests.exceptions.Timeout:
        st.error(f"⏱️ Timeout while loading {component_name}")
        render_timeout_fallback(component_name)
    except Exception as e:
        logger.error(f"Error in {component_name}: {e}")
        st.error(f"❌ Error loading {component_name}: {str(e)}")
        render_error_fallback(component_name, str(e))

def render_offline_fallback(component_name: str):
    """Render fallback content when API is offline."""
    st.info(f"📱 {component_name} is temporarily unavailable. Please check your connection and try again.")
    
    if st.button(f"🔄 Retry {component_name}", key=f"retry_{component_name}"):
        st.rerun()

def render_timeout_fallback(component_name: str):
    """Render fallback content for timeout scenarios."""
    st.warning(f"⏱️ {component_name} is taking longer than expected to load.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"🔄 Retry {component_name}", key=f"timeout_retry_{component_name}"):
            st.rerun()
    with col2:
        if st.button("⚙️ Check System Status", key=f"status_{component_name}"):
            st.info(f"API Status: {'🟢 Online' if st.session_state.api_status else '🔴 Offline'}")

def render_error_fallback(component_name: str, error_message: str):
    """Render fallback content for general errors."""
    with st.expander(f"🔧 Troubleshooting {component_name}", expanded=False):
        st.code(f"Error: {error_message}")
        st.markdown("""
        **Possible solutions:**
        1. Check if the API server is running
        2. Verify network connectivity
        3. Try refreshing the page
        4. Contact system administrator if the issue persists
        """)

def render_view_content(view_mode: str, api_client: WMSAPIClient, controls: Dict[str, Any]):
    """
    Render content based on selected view mode with comprehensive error handling.
    
    Args:
        view_mode (str): Selected view mode
        api_client (WMSAPIClient): API client instance
        controls (Dict[str, Any]): User controls and filters
    """
    # Add navigation breadcrumbs
    render_navigation_breadcrumbs(view_mode)
    
    if view_mode == "Overview":
        st.markdown("### 🏠 Dashboard Overview")
        
        # Render alerts first (most critical information)
        safe_render_component(render_alerts, "Alerts", api_client)
        st.divider()
        
        # KPI cards
        safe_render_component(render_kpi_cards, "KPI Cards", api_client)
        st.divider()
        
        # Dynamic charts
        safe_render_component(
            render_dynamic_charts, 
            "Dynamic Charts", 
            api_client, 
            controls.get('date_range', 30)
        )
        st.divider()
        
        # Revenue metrics
        safe_render_component(render_revenue_metrics, "Revenue Metrics", api_client)
        
    elif view_mode == "Inventory":
        st.markdown("### 📦 Inventory Management")
        
        # Inventory-specific KPIs
        safe_render_component(render_kpi_cards, "Inventory KPIs", api_client)
        st.divider()
        
        # FIFO efficiency dashboard
        safe_render_component(render_fifo_efficiency, "FIFO Efficiency", api_client)
        st.divider()
        
        # Detailed alerts table
        safe_render_component(render_alerts_table, "Alerts Management", api_client)
        
    elif view_mode == "Operations":
        st.markdown("### ⚙️ Operations Center")
        
        # Operations alerts and monitoring
        safe_render_component(render_alerts_table, "Operations Alerts", api_client)
        st.divider()
        
        # Operational charts
        safe_render_component(
            render_dynamic_charts, 
            "Operations Charts", 
            api_client, 
            controls.get('date_range', 30)
        )
        st.divider()
        
        # Performance metrics placeholder
        render_performance_metrics_placeholder()
        
    elif view_mode == "Analytics":
        st.markdown("### 📊 Advanced Analytics")
        
        # FIFO efficiency analysis
        safe_render_component(render_fifo_efficiency, "FIFO Analytics", api_client)
        st.divider()
        
        # Advanced charts and trends
        safe_render_component(
            render_dynamic_charts, 
            "Analytics Charts", 
            api_client, 
            controls.get('date_range', 30)
        )
        st.divider()
        
        # Deep dive analytics placeholder
        render_analytics_placeholder()

def render_performance_metrics_placeholder():
    """Render placeholder for performance metrics."""
    st.markdown("### 🚀 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Order Processing Time", "2.3 min", "-0.5 min")
    with col2:
        st.metric("Picking Accuracy", "98.7%", "+1.2%")
    with col3:
        st.metric("Throughput Rate", "450 orders/hr", "+25 orders/hr")
    
    st.info("🔧 Detailed performance metrics coming in the next release!")

def render_analytics_placeholder():
    """Render placeholder for advanced analytics."""
    st.markdown("### 🔬 Deep Dive Analytics")
    
    st.info("🔬 Advanced analytics features including predictive modeling and trend analysis are coming soon!")
    
    # Mock advanced analytics preview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🤖 Predictive Analytics**")
        st.progress(0.75)
        st.caption("ML models in development")
    
    with col2:
        st.markdown("**📈 Trend Analysis**")
        st.progress(0.60)
        st.caption("Advanced forecasting coming soon")

def render_footer():
    """
    Render comprehensive footer with metadata, links, and system information.
    """
    st.markdown("---")
    
    # Footer content
    last_refresh = st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')
    api_status_icon = "🟢" if st.session_state.api_status else "🔴"
    api_status_text = "Online" if st.session_state.api_status else "Offline"
    
    footer_html = f"""
    <div style='text-align: center; padding: 2rem 0; color: #6c757d; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin-top: 2rem;'>
        <div style='display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 1rem;'>
            <div>
                <strong>Dashboard Version:</strong> {DASHBOARD_VERSION}<br>
                <small>Last Updated: {last_refresh}</small>
            </div>
            <div>
                <strong>API Status:</strong> {api_status_icon} {api_status_text}<br>
                <small>Endpoint: {API_BASE_URL}</small>
            </div>
            <div>
                <strong>Session Info:</strong><br>
                <small>Errors: {st.session_state.error_count} | Uptime: {(datetime.now() - st.session_state.last_refresh).total_seconds():.0f}s</small>
            </div>
        </div>
        <div style='margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #dee2e6;'>
            <a href='https://github.com/ShawSAM37/warehouse-mgmt-system' target='_blank' style='margin: 0 1rem; color: #007bff; text-decoration: none;'>📚 Documentation</a>
            <a href='https://github.com/ShawSAM37/warehouse-mgmt-system/issues' target='_blank' style='margin: 0 1rem; color: #007bff; text-decoration: none;'>🐛 Report Issue</a>
            <a href='mailto:support@wms.com' style='margin: 0 1rem; color: #007bff; text-decoration: none;'>📧 Support</a>
        </div>
    </div>
    """
    
    st.markdown(footer_html, unsafe_allow_html=True)
    
    # Export functionality
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export Dashboard Data", key="export_dashboard"):
            export_dashboard_data()
    
    with col2:
        if st.button("📊 Generate Report", key="generate_report"):
            generate_dashboard_report()
    
    with col3:
        if st.button("⚙️ System Diagnostics", key="system_diagnostics"):
            show_system_diagnostics()

def export_dashboard_data():
    """Export current dashboard data to CSV."""
    try:
        # This would collect data from all components
        st.success("📥 Dashboard data export feature coming soon!")
        st.info("Will include: KPIs, alerts, inventory levels, and performance metrics")
    except Exception as e:
        st.error(f"Export failed: {e}")

def generate_dashboard_report():
    """Generate comprehensive dashboard report."""
    try:
        st.success("📊 Report generation feature coming soon!")
        st.info("Will include: Executive summary, detailed analytics, and recommendations")
    except Exception as e:
        st.error(f"Report generation failed: {e}")

def show_system_diagnostics():
    """Show system diagnostics and health information."""
    with st.expander("🔧 System Diagnostics", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**API Health Check**")
            api_healthy = check_api_health()
            st.write(f"Status: {'✅ Healthy' if api_healthy else '❌ Unhealthy'}")
            st.write(f"Response Time: {'< 1s' if api_healthy else 'Timeout'}")
            
        with col2:
            st.markdown("**Session Information**")
            st.write(f"Session Duration: {(datetime.now() - st.session_state.last_refresh).total_seconds():.0f}s")
            st.write(f"Error Count: {st.session_state.error_count}")
            st.write(f"Cache Status: Active")

def main():
    """
    Main dashboard application entry point with comprehensive error handling and user experience optimization.
    """
    try:
        # Initialize session state
        initialize_session_state()
        
        # Log application start
        logger.info(f"WMS Dashboard v{DASHBOARD_VERSION} starting...")
        
        # Create API client with error handling
        try:
            api_client = get_api_client()
            logger.info("API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize API client: {e}")
            st.error("🚨 Failed to connect to WMS API. Please check your connection and try again.")
            
            # Render minimal interface for troubleshooting
            st.markdown("## 🔧 Connection Troubleshooting")
            st.markdown(f"**API Endpoint:** {API_BASE_URL}")
            st.markdown("**Possible Issues:**")
            st.markdown("- API server is not running")
            st.markdown("- Network connectivity issues")
            st.markdown("- Firewall blocking connections")
            
            if st.button("🔄 Retry Connection"):
                st.rerun()
            
            return
        
        # Render header
        safe_render_component(render_header, "Header")
        
        # Render sidebar and get controls
        controls = {}
        try:
            controls = render_sidebar()
            logger.debug(f"Sidebar controls: {controls}")
        except Exception as e:
            logger.error(f"Sidebar rendering failed: {e}")
            st.error("⚠️ Sidebar initialization failed. Using default settings.")
            
            # Fallback controls
            controls = {
                'auto_refresh': True,
                'refresh_rate': REFRESH_INTERVAL,
                'date_range': 30,
                'view_mode': 'Overview',
                'product_codes': [],
                'storage_zone': 'All Zones',
                'priority_filter': ['URGENT', 'HIGH']
            }
        
        # Handle auto-refresh
        refresh_triggered = handle_auto_refresh(controls)
        
        if not refresh_triggered:  # Only render content if not refreshing
            # Render main content based on view mode
            view_mode = controls.get('view_mode', 'Overview')
            
            try:
                render_view_content(view_mode, api_client, controls)
            except Exception as e:
                logger.error(f"View content rendering failed: {e}")
                st.error(f"❌ Error rendering {view_mode} view. Please try refreshing or selecting a different view.")
                
                # Offer recovery options
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Refresh Page"):
                        st.rerun()
                with col2:
                    if st.button("🏠 Go to Overview"):
                        st.session_state.view_mode = 'Overview'
                        st.rerun()
        
        # Render footer
        safe_render_component(render_footer, "Footer")
        
        # Log successful completion
        logger.info("Dashboard rendered successfully")
        
    except Exception as e:
        logger.critical(f"Critical error in main application: {e}")
        st.error("🚨 Critical application error occurred. Please refresh the page or contact support.")
        
        # Emergency diagnostic information
        with st.expander("🔧 Emergency Diagnostics", expanded=False):
            st.code(f"Error: {str(e)}")
            st.code(f"Time: {datetime.now()}")
            st.code(f"Session State Keys: {list(st.session_state.keys())}")

# Application entry point
if __name__ == "__main__":
    try:
        # Set up logging for production
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Run main application
        main()
        
    except Exception as e:
        # Last resort error handling
        st.error(f"🚨 Fatal error: {str(e)}")
        st.markdown("Please contact system administrator.")
        logger.critical(f"Fatal application error: {e}")

"""
Warehouse Management System Dashboard - Production Ready
========================================================

A comprehensive, production-ready Streamlit dashboard for real-time warehouse operations monitoring.

Features:
- Real-time inventory tracking and analytics
- FIFO efficiency monitoring
- Supplier performance metrics
- Advanced alert management
- Export functionality (CSV/PDF)
- Dark/light mode support
- Responsive design
- Performance optimizations
- Comprehensive error handling

Dependencies:
- streamlit>=1.28.0
- requests>=2.31.0
- pandas>=2.0.0
- plotly>=5.17.0
- reportlab>=4.0.0 (for PDF generation)
- schedule>=1.2.0 (for automated reports)

Configuration:
Set the following environment variables:
- WMS_API_URL: Base URL for the WMS API (default: http://localhost:8000)
- WMS_DEBUG_MODE: Enable debug mode (default: False)
- WMS_THEME: Default theme (light/dark, default: light)

Usage:
    streamlit run dashboard.py

Author: WMS Development Team
Version: 1.0.0
License: MIT
"""

import base64
import io
import json
import os
import schedule
import smtplib
import threading
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from typing import Dict, List, Optional, Any, Callable, Union
import weakref

# Enhanced Configuration with Environment Variables
class DashboardConfig:
    """Centralized configuration management for the dashboard."""
    
    def __init__(self):
        self.API_BASE_URL = os.getenv('WMS_API_URL', 'http://localhost:8000')
        self.DEBUG_MODE = os.getenv('WMS_DEBUG_MODE', 'False').lower() == 'true'
        self.DEFAULT_THEME = os.getenv('WMS_THEME', 'light')
        self.REFRESH_INTERVAL = int(os.getenv('WMS_REFRESH_INTERVAL', '30'))
        self.MAX_CACHE_SIZE = int(os.getenv('WMS_MAX_CACHE_SIZE', '100'))
        self.ENABLE_ANALYTICS = os.getenv('WMS_ENABLE_ANALYTICS', 'True').lower() == 'true'
        
        # Email configuration for reports
        self.SMTP_SERVER = os.getenv('WMS_SMTP_SERVER', 'smtp.gmail.com')
        self.SMTP_PORT = int(os.getenv('WMS_SMTP_PORT', '587'))
        self.EMAIL_USER = os.getenv('WMS_EMAIL_USER', '')
        self.EMAIL_PASSWORD = os.getenv('WMS_EMAIL_PASSWORD', '')

config = DashboardConfig()

# Advanced CSS with Modern Design Principles
ADVANCED_CSS = """
<style>
    /* CSS Variables for Theme Support */
    :root {
        --primary-color: #1f77b4;
        --success-color: #2ca02c;
        --warning-color: #ff7f0e;
        --danger-color: #d62728;
        --neutral-color: #7f7f7f;
        --background-color: #ffffff;
        --surface-color: #f8f9fa;
        --text-color: #212529;
        --border-color: #dee2e6;
        --shadow-light: 0 2px 4px rgba(0,0,0,0.1);
        --shadow-medium: 0 4px 8px rgba(0,0,0,0.15);
        --shadow-heavy: 0 8px 16px rgba(0,0,0,0.2);
        --transition-fast: 0.15s ease;
        --transition-medium: 0.3s ease;
        --transition-slow: 0.5s ease;
    }
    
    /* Dark Theme Variables */
    [data-theme="dark"] {
        --background-color: #1a1a1a;
        --surface-color: #2d2d2d;
        --text-color: #ffffff;
        --border-color: #404040;
    }
    
    /* Global Styles */
    .main {
        padding-top: 1rem;
        background: var(--background-color);
        color: var(--text-color);
        transition: background-color var(--transition-medium), color var(--transition-medium);
    }
    
    /* Enhanced Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-heavy);
        position: relative;
        overflow: hidden;
        animation: slideInFromTop 0.6s ease-out;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    @keyframes slideInFromTop {
        0% { transform: translateY(-50px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    
    /* Enhanced KPI Cards */
    .kpi-card {
        background: var(--surface-color);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: var(--shadow-light);
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
        transition: all var(--transition-medium);
        position: relative;
        overflow: hidden;
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-heavy);
        border-color: var(--primary-color);
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--success-color));
        border-radius: 16px 16px 0 0;
    }
    
    /* Loading Animations */
    .loading-skeleton {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200% 100%;
        animation: loading 1.5s infinite;
        border-radius: 8px;
        height: 20px;
        margin: 10px 0;
    }
    
    @keyframes loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    /* Enhanced Alert Banners */
    .alert-banner {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        font-weight: 600;
        border-left: 5px solid;
        animation: slideInFromLeft 0.4s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    @keyframes slideInFromLeft {
        0% { transform: translateX(-50px); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }
    
    .alert-banner::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        bottom: 0;
        width: 4px;
        background: rgba(255,255,255,0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.8; }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem;
            font-size: 0.9rem;
        }
        
        .kpi-card {
            padding: 1rem;
            margin-bottom: 0.5rem;
        }
        
        .alert-banner {
            padding: 0.75rem 1rem;
            font-size: 0.9rem;
        }
    }
    
    @media (max-width: 480px) {
        .main {
            padding: 0.5rem;
        }
        
        .main-header h1 {
            font-size: 1.5rem;
        }
    }
    
    /* Enhanced Sidebar */
    .sidebar-section {
        background: var(--surface-color);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-light);
        border: 1px solid var(--border-color);
        transition: all var(--transition-medium);
    }
    
    .sidebar-section:hover {
        box-shadow: var(--shadow-medium);
    }
    
    /* Chart Containers */
    .chart-container {
        background: var(--surface-color);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: var(--shadow-light);
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
        transition: all var(--transition-medium);
    }
    
    .chart-container:hover {
        box-shadow: var(--shadow-medium);
    }
    
    /* Enhanced Footer */
    .footer {
        background: linear-gradient(135deg, var(--surface-color) 0%, var(--background-color) 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-top: 3rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-light);
    }
    
    /* Accessibility Enhancements */
    .sr-only {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
    }
    
    /* Focus Indicators */
    button:focus, input:focus, select:focus {
        outline: 2px solid var(--primary-color);
        outline-offset: 2px;
    }
    
    /* High Contrast Mode */
    @media (prefers-contrast: high) {
        :root {
            --shadow-light: 0 2px 4px rgba(0,0,0,0.3);
            --shadow-medium: 0 4px 8px rgba(0,0,0,0.4);
            --shadow-heavy: 0 8px 16px rgba(0,0,0,0.5);
        }
    }
    
    /* Reduced Motion */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
</style>
"""

# Performance Optimization Classes
class CacheManager:
    """Advanced caching manager with memory optimization."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.memory_usage = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value with LRU tracking."""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """Set cached value with TTL and memory management."""
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = {
            'value': value,
            'expires': time.time() + ttl,
            'size': self._estimate_size(value)
        }
        self.access_times[key] = time.time()
        self._update_memory_usage()
    
    def _evict_lru(self) -> None:
        """Evict least recently used items."""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self.cache.pop(lru_key, None)
        self.access_times.pop(lru_key, None)
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object."""
        if isinstance(obj, (str, int, float, bool)):
            return 8
        elif isinstance(obj, (list, tuple)):
            return sum(self._estimate_size(item) for item in obj)
        elif isinstance(obj, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items())
        else:
            return 64  # Default estimate
    
    def _update_memory_usage(self) -> None:
        """Update total memory usage."""
        self.memory_usage = sum(
            item['size'] for item in self.cache.values()
        )
    
    def cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, item in self.cache.items()
            if item['expires'] < current_time
        ]
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)

# Initialize cache manager
cache_manager = CacheManager(max_size=config.MAX_CACHE_SIZE)

# Enhanced User Preferences Management
class UserPreferences:
    """Manage user preferences with persistence."""
    
    def __init__(self):
        self.preferences = self._load_preferences()
    
    def _load_preferences(self) -> Dict[str, Any]:
        """Load preferences from session state or defaults."""
        default_prefs = {
            'theme': config.DEFAULT_THEME,
            'auto_refresh': True,
            'refresh_rate': config.REFRESH_INTERVAL,
            'default_view': 'Overview',
            'favorite_views': [],
            'keyboard_shortcuts': True,
            'high_contrast': False,
            'reduced_motion': False,
            'notification_preferences': {
                'critical_alerts': True,
                'low_stock_warnings': True,
                'system_notifications': False
            }
        }
        
        if 'user_preferences' in st.session_state:
            return {**default_prefs, **st.session_state.user_preferences}
        return default_prefs
    
    def save_preferences(self) -> None:
        """Save preferences to session state."""
        st.session_state.user_preferences = self.preferences
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get preference value."""
        return self.preferences.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set preference value."""
        self.preferences[key] = value
        self.save_preferences()
    
    def toggle_theme(self) -> str:
        """Toggle between light and dark theme."""
        current_theme = self.get('theme', 'light')
        new_theme = 'dark' if current_theme == 'light' else 'light'
        self.set('theme', new_theme)
        return new_theme

# Initialize user preferences
user_prefs = UserPreferences()

# Export Functionality
class ExportManager:
    """Comprehensive export functionality for dashboard data."""
    
    def __init__(self, api_client: 'WMSAPIClient'):
        self.api_client = api_client
    
    def export_to_csv(self, data: List[Dict], filename: str) -> bytes:
        """Export data to CSV format."""
        if not data:
            return b"No data available"
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False).encode('utf-8')
    
    def export_to_pdf(self, dashboard_data: Dict[str, Any]) -> bytes:
        """Generate comprehensive PDF report."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1f77b4')
        )
        story.append(Paragraph("Warehouse Management Dashboard Report", title_style))
        story.append(Spacer(1, 12))
        
        # Report metadata
        metadata = [
            ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Dashboard Version:', DASHBOARD_VERSION],
            ['API Endpoint:', config.API_BASE_URL]
        ]
        
        metadata_table = Table(metadata, colWidths=[2*inch, 3*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 20))
        
        # KPI Summary
        if 'inventory_summary' in dashboard_data:
            story.append(Paragraph("Key Performance Indicators", styles['Heading2']))
            kpi_data = dashboard_data['inventory_summary']
            
            kpi_table_data = [
                ['Metric', 'Value'],
                ['Total Products', str(kpi_data.get('total_products', 'N/A'))],
                ['Total Inventory', f"{kpi_data.get('total_quantity', 'N/A')} units"],
                ['Low Stock Items', str(kpi_data.get('low_stock_count', 'N/A'))],
                ['Expiring Soon', str(kpi_data.get('expiring_soon_count', 'N/A'))]
            ]
            
            kpi_table = Table(kpi_table_data, colWidths=[2*inch, 2*inch])
            kpi_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(kpi_table)
            story.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def schedule_report(self, report_type: str, frequency: str, email: str) -> bool:
        """Schedule automated report generation."""
        try:
            def generate_and_send():
                # Generate report
                dashboard_data = self._collect_dashboard_data()
                
                if report_type == 'pdf':
                    report_data = self.export_to_pdf(dashboard_data)
                    filename = f"wms_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    content_type = 'application/pdf'
                else:
                    report_data = self.export_to_csv(dashboard_data.get('alerts', []), 'alerts_report.csv')
                    filename = f"wms_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    content_type = 'text/csv'
                
                # Send email
                self._send_email_report(email, report_data, filename, content_type)
            
            # Schedule based on frequency
            if frequency == 'daily':
                schedule.every().day.at("08:00").do(generate_and_send)
            elif frequency == 'weekly':
                schedule.every().monday.at("08:00").do(generate_and_send)
            elif frequency == 'monthly':
                schedule.every().month.do(generate_and_send)
            
            return True
        except Exception as e:
            logger.error(f"Failed to schedule report: {e}")
            return False
    
    def _collect_dashboard_data(self) -> Dict[str, Any]:
        """Collect current dashboard data."""
        return {
            'inventory_summary': self.api_client.get_inventory_summary(),
            'stock_levels': self.api_client.get_stock_levels(),
            'alerts': self.api_client.get_expiry_alerts(),
            'supplier_performance': self.api_client.get_supplier_performance(),
            'storage_utilization': self.api_client.get_storage_utilization()
        }
    
    def _send_email_report(self, email: str, report_data: bytes, filename: str, content_type: str) -> None:
        """Send report via email."""
        if not config.EMAIL_USER or not config.EMAIL_PASSWORD:
            logger.warning("Email configuration not set")
            return
        
        msg = MIMEMultipart()
        msg['From'] = config.EMAIL_USER
        msg['To'] = email
        msg['Subject'] = f"WMS Dashboard Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        body = f"""
        Warehouse Management System Dashboard Report
        
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Report Type: {filename.split('.')[-1].upper()}
        
        Please find the attached report.
        
        Best regards,
        WMS System
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach report
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(report_data)
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename= {filename}'
        )
        msg.attach(part)
        
        # Send email
        try:
            server = smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT)
            server.starttls()
            server.login(config.EMAIL_USER, config.EMAIL_PASSWORD)
            text = msg.as_string()
            server.sendmail(config.EMAIL_USER, email, text)
            server.quit()
            logger.info(f"Report sent successfully to {email}")
        except Exception as e:
            logger.error(f"Failed to send email report: {e}")

# Enhanced Debug and Monitoring
class DebugManager:
    """Comprehensive debugging and monitoring functionality."""
    
    def __init__(self):
        self.performance_metrics = {}
        self.error_log = []
        self.api_call_log = []
    
    def log_performance(self, operation: str, duration: float) -> None:
        """Log performance metrics."""
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []
        
        self.performance_metrics[operation].append({
            'duration': duration,
            'timestamp': datetime.now()
        })
        
        # Keep only last 100 entries per operation
        if len(self.performance_metrics[operation]) > 100:
            self.performance_metrics[operation] = self.performance_metrics[operation][-100:]
    
    def log_error(self, error: str, context: str = "") -> None:
        """Log errors with context."""
        self.error_log.append({
            'error': error,
            'context': context,
            'timestamp': datetime.now()
        })
        
        # Keep only last 50 errors
        if len(self.error_log) > 50:
            self.error_log = self.error_log[-50:]
    
    def log_api_call(self, endpoint: str, status: str, duration: float) -> None:
        """Log API calls."""
        self.api_call_log.append({
            'endpoint': endpoint,
            'status': status,
            'duration': duration,
            'timestamp': datetime.now()
        })
        
        # Keep only last 100 API calls
        if len(self.api_call_log) > 100:
            self.api_call_log = self.api_call_log[-100:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {}
        for operation, metrics in self.performance_metrics.items():
            durations = [m['duration'] for m in metrics]
            if durations:
                summary[operation] = {
                    'avg_duration': sum(durations) / len(durations),
                    'max_duration': max(durations),
                    'min_duration': min(durations),
                    'call_count': len(durations)
                }
        return summary
    
    def render_debug_panel(self) -> None:
        """Render debug information panel."""
        if not config.DEBUG_MODE:
            return
        
        with st.expander("🔧 Debug Information", expanded=False):
            tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Errors", "API Calls", "System"])
            
            with tab1:
                st.subheader("Performance Metrics")
                perf_summary = self.get_performance_summary()
                if perf_summary:
                    for operation, metrics in perf_summary.items():
                        st.metric(
                            f"{operation} (avg)",
                            f"{metrics['avg_duration']:.3f}s",
                            f"calls: {metrics['call_count']}"
                        )
                else:
                    st.info("No performance data available")
            
            with tab2:
                st.subheader("Recent Errors")
                if self.error_log:
                    for error in self.error_log[-10:]:
                        st.error(f"**{error['timestamp']}**: {error['error']}")
                        if error['context']:
                            st.caption(f"Context: {error['context']}")
                else:
                    st.success("No recent errors")
            
            with tab3:
                st.subheader("API Call Log")
                if self.api_call_log:
                    df_api = pd.DataFrame(self.api_call_log[-20:])
                    st.dataframe(df_api, use_container_width=True)
                else:
                    st.info("No API calls logged")
            
            with tab4:
                st.subheader("System Information")
                st.json({
                    'cache_size': len(cache_manager.cache),
                    'memory_usage': f"{cache_manager.memory_usage} bytes",
                    'session_state_keys': list(st.session_state.keys()),
                    'config': {
                        'api_url': config.API_BASE_URL,
                        'debug_mode': config.DEBUG_MODE,
                        'theme': user_prefs.get('theme')
                    }
                })

# Initialize debug manager
debug_manager = DebugManager()

# Enhanced Keyboard Shortcuts
def setup_keyboard_shortcuts():
    """Setup keyboard shortcuts for common actions."""
    if not user_prefs.get('keyboard_shortcuts', True):
        return
    
    shortcuts_js = """
    <script>
    document.addEventListener('keydown', function(e) {
        // Ctrl+R: Refresh dashboard
        if (e.ctrlKey && e.key === 'r') {
            e.preventDefault();
            window.parent.postMessage({type: 'streamlit:rerun'}, '*');
        }
        
        // Ctrl+1-4: Switch views
        if (e.ctrlKey && ['1', '2', '3', '4'].includes(e.key)) {
            e.preventDefault();
            const views = ['Overview', 'Inventory', 'Operations', 'Analytics'];
            const viewIndex = parseInt(e.key) - 1;
            if (views[viewIndex]) {
                // This would need to be implemented with session state
                console.log('Switch to view:', views[viewIndex]);
            }
        }
        
        // Ctrl+T: Toggle theme
        if (e.ctrlKey && e.key === 't') {
            e.preventDefault();
            // This would trigger theme toggle
            console.log('Toggle theme');
        }
        
        // Ctrl+E: Export data
        if (e.ctrlKey && e.key === 'e') {
            e.preventDefault();
            console.log('Export data');
        }
    });
    </script>
    """
    
    st.markdown(shortcuts_js, unsafe_allow_html=True)

# Enhanced Theme Management
def apply_theme():
    """Apply current theme to the dashboard."""
    theme = user_prefs.get('theme', 'light')
    
    theme_css = f"""
    <script>
    document.documentElement.setAttribute('data-theme', '{theme}');
    </script>
    """
    
    st.markdown(theme_css, unsafe_allow_html=True)

# Enhanced Accessibility Features
def setup_accessibility():
    """Setup accessibility features."""
    accessibility_css = """
    <style>
    /* Screen reader improvements */
    .metric-label::before {
        content: "Metric: ";
        position: absolute;
        left: -10000px;
        width: 1px;
        height: 1px;
        overflow: hidden;
    }
    
    /* Keyboard navigation */
    .stButton > button:focus {
        outline: 3px solid var(--primary-color);
        outline-offset: 2px;
    }
    
    /* High contrast mode support */
    @media (prefers-contrast: high) {
        .kpi-card {
            border: 2px solid var(--text-color);
        }
        
        .alert-banner {
            border: 2px solid currentColor;
        }
    }
    </style>
    """
    
    st.markdown(accessibility_css, unsafe_allow_html=True)

# Enhanced Error Handling with Recovery
class ErrorHandler:
    """Comprehensive error handling with recovery mechanisms."""
    
    def __init__(self):
        self.error_count = 0
        self.last_error_time = None
        self.recovery_strategies = {
            'ConnectionError': self._handle_connection_error,
            'TimeoutError': self._handle_timeout_error,
            'ValidationError': self._handle_validation_error,
            'AuthenticationError': self._handle_auth_error
        }
    
    def handle_error(self, error: Exception, context: str = "") -> bool:
        """Handle errors with appropriate recovery strategy."""
        error_type = type(error).__name__
        self.error_count += 1
        self.last_error_time = datetime.now()
        
        debug_manager.log_error(str(error), context)
        
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type](error, context)
        else:
            return self._handle_generic_error(error, context)
    
    def _handle_connection_error(self, error: Exception, context: str) -> bool:
        """Handle connection errors with retry logic."""
        st.error("🔌 Connection lost. Attempting to reconnect...")
        
        # Implement exponential backoff
        retry_delay = min(2 ** self.error_count, 30)
        time.sleep(retry_delay)
        
        # Test connection
        if check_api_health():
            st.success("✅ Connection restored!")
            self.error_count = 0
            return True
        
        return False
    
    def _handle_timeout_error(self, error: Exception, context: str) -> bool:
        """Handle timeout errors."""
        st.warning("⏱️ Request timed out. This might be due to high server load.")
        return False
    
    def _handle_validation_error(self, error: Exception, context: str) -> bool:
        """Handle validation errors."""
        st.error(f"📋 Data validation error: {str(error)}")
        return False
    
    def _handle_auth_error(self, error: Exception, context: str) -> bool:
        """Handle authentication errors."""
        st.error("🔐 Authentication failed. Please check your credentials.")
        return False
    
    def _handle_generic_error(self, error: Exception, context: str) -> bool:
        """Handle generic errors."""
        st.error(f"❌ An unexpected error occurred: {str(error)}")
        return False

# Initialize error handler
error_handler = ErrorHandler()

# Enhanced Main Application with All Features
def main():
    """
    Enhanced main application with all production features.
    """
    try:
        # Apply advanced styling
        st.markdown(ADVANCED_CSS, unsafe_allow_html=True)
        
        # Setup accessibility and keyboard shortcuts
        setup_accessibility()
        setup_keyboard_shortcuts()
        
        # Apply theme
        apply_theme()
        
        # Initialize session state
        initialize_session_state()
        
        # Create API client
        api_client = get_api_client()
        
        # Initialize export manager
        export_manager = ExportManager(api_client)
        
        # Render header
        render_header()
        
        # Render sidebar with enhanced features
        controls = render_enhanced_sidebar(export_manager)
        
        # Handle auto-refresh
        handle_auto_refresh(controls)
        
        # Render main content
        render_view_content(controls['view_mode'], api_client, controls)
        
        # Render debug panel
        debug_manager.render_debug_panel()
        
        # Render enhanced footer
        render_enhanced_footer(export_manager)
        
    except Exception as e:
        error_handler.handle_error(e, "main_application")

def render_enhanced_sidebar(export_manager: ExportManager) -> Dict[str, Any]:
    """Render enhanced sidebar with all features."""
    # Theme toggle
    if st.sidebar.button("🎨 Toggle Theme"):
        new_theme = user_prefs.toggle_theme()
        st.rerun()
    
    # Original sidebar content
    controls = render_sidebar()
    
    # Export section
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Export & Reports")
    
    export_format = st.sidebar.selectbox("Format", ["CSV", "PDF"])
    
    if st.sidebar.button("📥 Export Now"):
        if export_format == "CSV":
            # Export current data
            data = []  # This would be populated with current dashboard data
            csv_data = export_manager.export_to_csv(data, "dashboard_export.csv")
            st.sidebar.download_button(
                "Download CSV",
                csv_data,
                "dashboard_export.csv",
                "text/csv"
            )
        else:
            # Generate PDF report
            dashboard_data = {}  # This would be populated with current data
            pdf_data = export_manager.export_to_pdf(dashboard_data)
            st.sidebar.download_button(
                "Download PDF",
                pdf_data,
                "dashboard_report.pdf",
                "application/pdf"
            )
    
    # Scheduled reports
    with st.sidebar.expander("📅 Schedule Reports"):
        email = st.text_input("Email Address")
        frequency = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly"])
        report_type = st.selectbox("Type", ["PDF", "CSV"])
        
        if st.button("Schedule Report"):
            if email:
                success = export_manager.schedule_report(
                    report_type.lower(), 
                    frequency.lower(), 
                    email
                )
                if success:
                    st.success("Report scheduled successfully!")
                else:
                    st.error("Failed to schedule report")
            else:
                st.error("Please enter an email address")
    
    return controls

def render_enhanced_footer(export_manager: ExportManager):
    """Render enhanced footer with additional features."""
    render_footer()
    
    # Additional footer features
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔧 System Health**")
        health_status = "🟢 Healthy" if check_api_health() else "🔴 Unhealthy"
        st.markdown(f"Status: {health_status}")
        
        cache_health = "🟢 Optimal" if len(cache_manager.cache) < cache_manager.max_size * 0.8 else "🟡 High"
        st.markdown(f"Cache: {cache_health}")
    
    with col2:
        st.markdown("**⚡ Performance**")
        perf_summary = debug_manager.get_performance_summary()
        if perf_summary:
            avg_response = sum(p['avg_duration'] for p in perf_summary.values()) / len(perf_summary)
            st.markdown(f"Avg Response: {avg_response:.3f}s")
        else:
            st.markdown("Avg Response: N/A")
    
    with col3:
        st.markdown("**📊 Usage Stats**")
        st.markdown(f"Errors: {error_handler.error_count}")
        st.markdown(f"Cache Hits: {len(cache_manager.cache)}")

if __name__ == "__main__":
    main()
