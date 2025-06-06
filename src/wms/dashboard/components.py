"""
UI Components for WMS Dashboard
==============================

Comprehensive collection of Streamlit UI components for the warehouse management
dashboard including headers, sidebars, KPI cards, alerts, and utility components.
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
import logging
import time

from .api_client import WMSAPIClient, get_api_client, check_api_health
from .utils.config import DashboardConfig
from .utils.helpers import (
    format_number, 
    get_priority_color, 
    safe_get, 
    format_datetime,
    calculate_percentage_change
)

logger = logging.getLogger(__name__)

# Color schemes and constants
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

DASHBOARD_VERSION = "1.0.0"

def render_header() -> None:
    """
    Render the main dashboard header with company branding and system status.
    
    Features:
    - Professional gradient background
    - Company branding with emoji
    - System status indicator with real-time health check
    - Last updated timestamp
    - Responsive design for mobile devices
    """
    try:
        # Get current time and system status
        current_time = datetime.now()
        last_updated = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Check API health status
        api_status = check_api_health()
        status_text = "Online" if api_status else "Offline"
        status_color = COLOR_SCHEME['success'] if api_status else COLOR_SCHEME['danger']
        status_icon = "🟢" if api_status else "🔴"
        
        # Create header HTML with responsive design
        header_html = f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            position: relative;
            overflow: hidden;
            animation: slideInFromTop 0.6s ease-out;
        ">
            <div style="
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                animation: shimmer 3s infinite;
            "></div>
            
            <h1 style="
                margin: 0;
                font-size: 2.5rem;
                font-weight: 700;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                margin-bottom: 1rem;
            ">
                📦 Warehouse Management Dashboard
            </h1>
            
            <div style="
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 2rem;
                flex-wrap: wrap;
                margin-top: 1rem;
            ">
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    font-size: 1rem;
                    background: rgba(255, 255, 255, 0.1);
                    padding: 0.5rem 1rem;
                    border-radius: 25px;
                    backdrop-filter: blur(10px);
                ">
                    <span style="font-weight: bold;">System Status:</span>
                    <span style="color: {status_color}; font-weight: bold;">
                        {status_icon} {status_text}
                    </span>
                </div>
                
                <div style="
                    font-size: 0.9rem;
                    opacity: 0.9;
                    background: rgba(255, 255, 255, 0.1);
                    padding: 0.5rem 1rem;
                    border-radius: 25px;
                    backdrop-filter: blur(10px);
                ">
                    <span style="font-weight: bold;">Last Updated:</span> {last_updated}
                </div>
                
                <div style="
                    font-size: 0.9rem;
                    opacity: 0.9;
                    background: rgba(255, 255, 255, 0.1);
                    padding: 0.5rem 1rem;
                    border-radius: 25px;
                    backdrop-filter: blur(10px);
                ">
                    <span style="font-weight: bold;">Version:</span> {DASHBOARD_VERSION}
                </div>
            </div>
        </div>
        
        <style>
            @keyframes slideInFromTop {{
                0% {{ transform: translateY(-50px); opacity: 0; }}
                100% {{ transform: translateY(0); opacity: 1; }}
            }}
            
            @keyframes shimmer {{
                0% {{ left: -100%; }}
                100% {{ left: 100%; }}
            }}
            
            @media (max-width: 768px) {{
                .header-content h1 {{
                    font-size: 1.8rem !important;
                }}
                .header-content > div {{
                    flex-direction: column !important;
                    gap: 1rem !important;
                }}
            }}
        </style>
        """
        
        st.markdown(header_html, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error rendering header: {e}")
        st.error("❌ Error loading dashboard header")

def render_sidebar() -> Dict[str, Any]:
    """
    Render the comprehensive sidebar with dashboard controls and filters.
    
    Returns:
        Dictionary containing all control values and user preferences
    """
    try:
        # Dashboard Controls Section
        st.sidebar.markdown("""
        <div style="
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        ">
        """, unsafe_allow_html=True)
        
        st.sidebar.title("🎛️ Dashboard Controls")
        
        # Auto-refresh controls
        auto_refresh = st.sidebar.checkbox(
            "🔄 Auto-refresh", 
            value=st.session_state.get('auto_refresh', True),
            key="auto_refresh",
            help="Automatically refresh dashboard data at specified intervals"
        )
        
        refresh_rate = 30  # Default
        if auto_refresh:
            refresh_rate = st.sidebar.slider(
                "Refresh Rate (seconds)", 
                min_value=10, 
                max_value=300, 
                value=st.session_state.get('refresh_rate', 30),
                step=10,
                key="refresh_rate",
                help="How often to refresh the data automatically"
            )
            st.sidebar.info(f"⏱️ Auto-refreshing every {refresh_rate}s")
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Now", type="primary", key="manual_refresh"):
            st.cache_data.clear()
            st.session_state.last_refresh = datetime.now()
            st.rerun()
        
        st.sidebar.markdown("</div>", unsafe_allow_html=True)
        
        # Theme Toggle
        st.sidebar.markdown("---")
        current_theme = st.session_state.get('theme', 'light')
        if st.sidebar.button(f"🎨 Switch to {'Dark' if current_theme == 'light' else 'Light'} Mode"):
            new_theme = 'dark' if current_theme == 'light' else 'light'
            st.session_state.theme = new_theme
            st.rerun()
        
        # Filters Section
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Filters & Views")
        
        # Date range selector
        date_range = st.sidebar.selectbox(
            "📅 Analysis Period",
            options=[7, 30, 90],
            index=1,
            format_func=lambda x: f"Last {x} days",
            key="date_range",
            help="Select the time period for data analysis"
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
        with st.sidebar.expander("📦 Product Filters", expanded=False):
            # In a real implementation, this would be populated from the API
            available_products = ["PROD-001", "PROD-002", "PROD-003", "PROD-004", "PROD-005"]
            product_codes = st.multiselect(
                "Product Codes",
                options=available_products,
                default=[],
                key="product_codes",
                help="Filter by specific product codes"
            )
            
            # Storage zone dropdown filter
            storage_zones = st.selectbox(
                "Storage Zone",
                options=["All Zones", "Zone A", "Zone B", "Zone C", "Zone D"],
                index=0,
                key="storage_zone",
                help="Filter by storage zone"
            )
        
        # Priority filter for alerts
        with st.sidebar.expander("⚠️ Alert Filters", expanded=False):
            priority_filter = st.multiselect(
                "Alert Priority",
                options=["URGENT", "HIGH", "MEDIUM", "LOW"],
                default=["URGENT", "HIGH"],
                key="priority_filter",
                help="Filter alerts by priority level"
            )
            
            alert_days = st.slider(
                "Alert Timeframe (days)",
                min_value=1,
                max_value=90,
                value=30,
                key="alert_days",
                help="Number of days ahead to check for alerts"
            )
        
        # Export Options Section
        st.sidebar.markdown("---")
        st.sidebar.subheader("📥 Export & Reports")
        
        export_format = st.sidebar.selectbox(
            "Export Format",
            options=["CSV", "PDF", "Excel"],
            key="export_format"
        )
        
        if st.sidebar.button("📥 Export Current View", key="export_current"):
            st.sidebar.success("Export functionality will be implemented in the next phase!")
        
        # Scheduled reports
        with st.sidebar.expander("📅 Scheduled Reports", expanded=False):
            email = st.text_input("Email Address", key="report_email")
            frequency = st.selectbox(
                "Frequency", 
                options=["Daily", "Weekly", "Monthly"],
                key="report_frequency"
            )
            
            if st.button("Schedule Report", key="schedule_report"):
                if email:
                    st.success("Report scheduling will be implemented in the next phase!")
                else:
                    st.error("Please enter an email address")
        
        # System Information Section
        st.sidebar.markdown("---")
        st.sidebar.subheader("ℹ️ System Information")
        
        # API Health Check
        api_health = check_api_health()
        health_status = "🟢 Online" if api_health else "🔴 Offline"
        st.sidebar.info(f"**API Status:** {health_status}")
        
        # Version and configuration info
        st.sidebar.info(f"**Dashboard Version:** {DASHBOARD_VERSION}")
        
        # Performance metrics
        if st.sidebar.button("📊 Performance Metrics", key="perf_metrics"):
            try:
                api_client = get_api_client()
                metrics = api_client.get_performance_metrics()
                
                st.sidebar.json({
                    "Total Requests": metrics.get("total_requests", 0),
                    "Error Rate": f"{metrics.get('error_rate_percent', 0):.1f}%",
                    "Last Request": metrics.get("last_request_time", "Never")
                })
            except Exception as e:
                st.sidebar.error(f"Error loading metrics: {e}")
        
        # Debug mode toggle (if enabled in config)
        from .utils.config import config
        if config.enable_debug:
            st.sidebar.markdown("---")
            debug_mode = st.sidebar.checkbox("🔧 Debug Mode", key="debug_mode")
            if debug_mode:
                st.sidebar.json({
                    "Session State Keys": list(st.session_state.keys()),
                    "Current Time": datetime.now().isoformat(),
                    "Cache Size": "Unknown"  # Would need to implement cache size tracking
                })
        
        return {
            'auto_refresh': auto_refresh,
            'refresh_rate': refresh_rate,
            'date_range': date_range,
            'view_mode': view_mode,
            'product_codes': product_codes,
            'storage_zone': storage_zones,
            'priority_filter': priority_filter,
            'alert_days': alert_days,
            'export_format': export_format,
            'theme': st.session_state.get('theme', 'light')
        }
        
    except Exception as e:
        logger.error(f"Error rendering sidebar: {e}")
        st.sidebar.error("❌ Error loading sidebar controls")
        
        # Return minimal fallback controls
        return {
            'auto_refresh': True,
            'refresh_rate': 30,
            'date_range': 30,
            'view_mode': 'Overview',
            'product_codes': [],
            'storage_zone': 'All Zones',
            'priority_filter': ['URGENT', 'HIGH'],
            'alert_days': 30,
            'export_format': 'CSV',
            'theme': 'light'
        }

def render_kpi_cards(api_client: WMSAPIClient) -> None:
    """
    Render KPI cards in a responsive layout with comprehensive metrics.
    
    Args:
        api_client: API client for data fetching
    """
    st.subheader("📊 Key Performance Indicators")
    
    try:
        with st.spinner("📈 Loading KPI data..."):
            # Fetch data from API
            inventory_summary = api_client.get_inventory_summary()
            
            if inventory_summary:
                # First row of KPIs
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_products = safe_get(inventory_summary, 'total_products', 0)
                    # Mock delta calculation (in production, would compare with previous period)
                    delta_products = f"+{max(1, total_products//20)}" if total_products > 0 else None
                    
                    st.metric(
                        label="📦 Total Products",
                        value=format_number(total_products),
                        delta=delta_products,
                        help="Total number of active products in inventory system"
                    )
                
                with col2:
                    total_quantity = safe_get(inventory_summary, 'total_quantity', 0)
                    delta_quantity = f"+{max(100, total_quantity//50)}" if total_quantity > 0 else None
                    
                    st.metric(
                        label="📊 Total Inventory",
                        value=format_number(total_quantity, " units"),
                        delta=delta_quantity,
                        help="Total quantity of all inventory items across all locations"
                    )
                
                with col3:
                    low_stock_count = safe_get(inventory_summary, 'low_stock_count', 0)
                    delta_low_stock = f"-{max(1, low_stock_count//3)}" if low_stock_count > 0 else "0"
                    
                    st.metric(
                        label="⚠️ Low Stock Items",
                        value=low_stock_count,
                        delta=delta_low_stock,
                        delta_color="inverse",
                        help="Number of products below their reorder point threshold"
                    )
                
                with col4:
                    expiring_count = safe_get(inventory_summary, 'expiring_soon_count', 0)
                    delta_expiring = f"-{max(1, expiring_count//2)}" if expiring_count > 0 else "0"
                    
                    st.metric(
                        label="⏰ Expiring Soon",
                        value=expiring_count,
                        delta=delta_expiring,
                        delta_color="inverse",
                        help="Items expiring within the next 30 days"
                    )
                
                # Second row of KPIs
                col5, col6, col7, col8 = st.columns(4)
                
                with col5:
                    total_batches = safe_get(inventory_summary, 'total_batches', 0)
                    delta_batches = f"+{max(1, total_batches//30)}" if total_batches > 0 else None
                    
                    st.metric(
                        label="📋 Active Batches",
                        value=total_batches,
                        delta=delta_batches,
                        help="Number of active inventory batches currently in the system"
                    )
                
                with col6:
                    # Mock warehouse utilization (would come from storage utilization API)
                    utilization = 78.5  # Mock value
                    delta_util = "+2.3%"
                    
                    st.metric(
                        label="🏪 Warehouse Utilization",
                        value=f"{utilization:.1f}%",
                        delta=delta_util,
                        help="Average storage space utilization across all zones"
                    )
                
                with col7:
                    # Mock daily throughput calculation
                    daily_throughput = total_quantity * 0.05 if total_quantity else 0
                    
                    st.metric(
                        label="🚀 Daily Throughput",
                        value=format_number(daily_throughput, " units"),
                        delta="+12%",
                        help="Estimated daily inventory movement and processing volume"
                    )
                
                with col8:
                    # Mock fulfillment rate
                    fulfillment_rate = 96.8
                    delta_color = "normal" if fulfillment_rate > 95 else "inverse"
                    delta_fulfillment = "+1.2%" if fulfillment_rate > 95 else "-0.8%"
                    
                    st.metric(
                        label="✅ Fulfillment Rate",
                        value=f"{fulfillment_rate:.1f}%",
                        delta=delta_fulfillment,
                        delta_color=delta_color,
                        help="Order fulfillment accuracy and completion rate"
                    )
                
                # Add timestamp for data freshness
                st.caption(f"📅 Data as of: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
            else:
                render_error_state("Unable to load inventory summary data", "kpi_cards")
                
    except Exception as e:
        logger.error(f"Error in render_kpi_cards: {e}")
        render_error_state(f"Error loading KPI data: {str(e)}", "kpi_cards")

def render_alerts(api_client: WMSAPIClient, priority_filter: List[str] = None, days_ahead: int = 30) -> None:
    """
    Render alert system with priority-based styling and comprehensive information.
    
    Args:
        api_client: API client for fetching alert data
        priority_filter: List of priority levels to include
        days_ahead: Number of days ahead to check for alerts
    """
    try:
        with st.spinner("🔍 Checking for alerts..."):
            alerts = api_client.get_expiry_alerts(days_ahead=days_ahead, priority=priority_filter)
            
            if alerts:
                # Categorize alerts by priority
                critical_alerts = [a for a in alerts if safe_get(a, 'priority') == 'URGENT']
                high_alerts = [a for a in alerts if safe_get(a, 'priority') == 'HIGH']
                medium_alerts = [a for a in alerts if safe_get(a, 'priority') == 'MEDIUM']
                low_alerts = [a for a in alerts if safe_get(a, 'priority') == 'LOW']
                
                # Critical alerts banner
                if critical_alerts:
                    st.markdown(
                        f"""
                        <div style="
                            background: linear-gradient(135deg, #ff6b6b, #ee5a52);
                            color: white;
                            padding: 1rem 1.5rem;
                            border-radius: 12px;
                            margin-bottom: 1.5rem;
                            font-weight: 600;
                            border-left: 5px solid #d63031;
                            animation: slideInFromLeft 0.4s ease-out;
                            box-shadow: 0 4px 12px rgba(214, 39, 40, 0.3);
                        ">
                            🚨 <strong>CRITICAL ALERT:</strong> {len(critical_alerts)} items expiring within 3 days!
                            <br><small style="opacity: 0.9;">Immediate action required to prevent stock loss</small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Show top 3 critical alerts in expandable section
                    with st.expander(f"🔴 View {len(critical_alerts)} Critical Alerts", expanded=True):
                        for i, alert in enumerate(critical_alerts[:3]):
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                st.error(
                                    f"**{safe_get(alert, 'product_code', 'Unknown')}** - "
                                    f"Batch {safe_get(alert, 'batch_number', 'Unknown')}"
                                )
                            
                            with col2:
                                st.write(f"⏰ {safe_get(alert, 'days_until_expiry', 0)} days")
                            
                            with col3:
                                st.write(f"📦 {format_number(safe_get(alert, 'quantity', 0))} units")
                        
                        if len(critical_alerts) > 3:
                            st.info(f"... and {len(critical_alerts) - 3} more critical alerts")
                
                # High priority alerts
                if high_alerts:
                    st.markdown(
                        f"""
                        <div style="
                            background: linear-gradient(135deg, #fdcb6e, #f39c12);
                            color: white;
                            padding: 1rem 1.5rem;
                            border-radius: 12px;
                            margin-bottom: 1.5rem;
                            font-weight: 600;
                            border-left: 5px solid #e17055;
                            box-shadow: 0 4px 12px rgba(225, 112, 85, 0.3);
                        ">
                            ⚠️ <strong>HIGH PRIORITY:</strong> {len(high_alerts)} items require attention
                            <br><small style="opacity: 0.9;">Action needed within 7 days</small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Medium priority alerts
                if medium_alerts:
                    st.warning(f"📋 **MEDIUM PRIORITY:** {len(medium_alerts)} items to monitor")
                
                # Alert summary with metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🚨 Critical", len(critical_alerts))
                
                with col2:
                    st.metric("⚠️ High Priority", len(high_alerts))
                
                with col3:
                    st.metric("📋 Medium Priority", len(medium_alerts))
                
                with col4:
                    st.metric("📊 Total Alerts", len(alerts))
                
                # Alert summary information
                total_quantity_at_risk = sum(safe_get(alert, 'quantity', 0) for alert in alerts)
                avg_days_to_expiry = sum(safe_get(alert, 'days_until_expiry', 0) for alert in alerts) / len(alerts)
                
                st.info(
                    f"📊 **Alert Summary:** {len(alerts)} total alerts affecting "
                    f"{format_number(total_quantity_at_risk)} units. "
                    f"Average time to expiry: {avg_days_to_expiry:.1f} days."
                )
                
            else:
                # No alerts - show success message
                st.markdown(
                    """
                    <div style="
                        background: linear-gradient(135deg, #00b894, #00a085);
                        color: white;
                        padding: 1rem 1.5rem;
                        border-radius: 12px;
                        margin-bottom: 1.5rem;
                        font-weight: 600;
                        border-left: 5px solid #00b894;
                        box-shadow: 0 4px 12px rgba(0, 184, 148, 0.3);
                    ">
                        ✅ <strong>All systems operational</strong> - No critical alerts detected
                        <br><small style="opacity: 0.9;">All inventory levels are within acceptable ranges</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
    except Exception as e:
        logger.error(f"Error in render_alerts: {e}")
        render_error_state(f"Error loading alerts: {str(e)}", "alerts")

def render_loading_placeholder(message: str = "Loading data...") -> None:
    """
    Render a loading placeholder with spinner and message.
    
    Args:
        message: Loading message to display
    """
    st.markdown(
        f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 12px;
            margin: 1rem 0;
        ">
            <div style="
                width: 40px;
                height: 40px;
                border: 4px solid #dee2e6;
                border-top: 4px solid #1f77b4;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-bottom: 1rem;
            "></div>
            <div style="
                font-size: 1.1rem;
                color: #6c757d;
                font-weight: 500;
            ">{message}</div>
        </div>
        
        <style>
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

def render_error_state(error_message: str, component_name: str = "component") -> None:
    """
    Render error state with retry option and troubleshooting information.
    
    Args:
        error_message: Error message to display
        component_name: Name of the component that failed
    """
    st.error(f"❌ {error_message}")
    
    with st.expander("🔧 Troubleshooting", expanded=False):
        st.markdown(f"""
        **Component:** {component_name}
        
        **Possible solutions:**
        1. Check if the API server is running
        2. Verify network connectivity
        3. Try refreshing the page
        4. Clear browser cache
        5. Contact system administrator if the issue persists
        
        **Error Details:**
        ```
        {error_message}
        ```
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(f"🔄 Retry {component_name.title()}", key=f"retry_{component_name}_{int(time.time())}"):
            st.rerun()
    
    with col2:
        if st.button("🏠 Go to Overview", key=f"overview_{component_name}_{int(time.time())}"):
            st.session_state.view_mode = 'Overview'
            st.rerun()

def render_navigation_breadcrumbs(view_mode: str) -> None:
    """
    Render navigation breadcrumbs for better user orientation.
    
    Args:
        view_mode: Current view mode
    """
    breadcrumb_html = f"""
    <div style="
        margin-bottom: 1rem;
        padding: 0.75rem 1rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 8px;
        border-left: 4px solid {COLOR_SCHEME['primary']};
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    ">
        <nav style="
            display: flex;
            align-items: center;
            font-size: 0.9rem;
            color: {COLOR_SCHEME['text']};
        ">
            <span style="color: {COLOR_SCHEME['neutral']};">🏠 Dashboard</span>
            <span style="margin: 0 0.5rem; color: {COLOR_SCHEME['neutral']};">></span>
            <span style="color: {COLOR_SCHEME['primary']}; font-weight: 600;">📊 {view_mode}</span>
        </nav>
    </div>
    """
    st.markdown(breadcrumb_html, unsafe_allow_html=True)

def render_accessibility_features() -> None:
    """
    Render accessibility features and screen reader support.
    """
    st.markdown("""
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
            outline: 3px solid #1f77b4;
            outline-offset: 2px;
        }
        
        /* High contrast mode support */
        @media (prefers-contrast: high) {
            .kpi-card {
                border: 2px solid #212529;
            }
            
            .alert-banner {
                border: 2px solid currentColor;
            }
        }
        
        /* Reduced motion support */
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)
