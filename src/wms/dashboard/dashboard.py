"""
WMS Dashboard - Main Application
===============================

Comprehensive warehouse management dashboard integrating all components,
charts, API client, and enhancements for a complete production-ready solution.
"""

import streamlit as st
import logging
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
import traceback
import sys
import os

# Import all dashboard components
from .components import (
    render_header,
    render_sidebar,
    render_kpi_cards,
    render_alerts,
    render_navigation_breadcrumbs,
    render_loading_placeholder,
    render_error_state,
    render_accessibility_features
)

from .charts import (
    render_dynamic_charts,
    render_fifo_efficiency,
    render_alerts_table,
    render_revenue_metrics
)

from .api_client import WMSAPIClient, get_api_client, check_api_health
from .enhancements import (
    DashboardEnhancements,
    get_enhancements,
    UserPreferences,
    PerformanceMonitor,
    DebugManager,
    ErrorHandler
)

from .utils.config import DashboardConfig
from .utils.helpers import format_number, safe_get, format_datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('wms_dashboard.log') if os.access('.', os.W_OK) else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

# Dashboard version and metadata
DASHBOARD_VERSION = "1.0.0"
DASHBOARD_TITLE = "Warehouse Management Dashboard"

def initialize_session_state() -> None:
    """
    Initialize all required session state variables for the dashboard.
    
    This function sets up default values for all session state variables
    used throughout the dashboard application.
    """
    try:
        # Core application state
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        
        if 'api_status' not in st.session_state:
            st.session_state.api_status = check_api_health()
        
        if 'error_count' not in st.session_state:
            st.session_state.error_count = 0
        
        if 'last_error' not in st.session_state:
            st.session_state.last_error = None
        
        # User preferences and settings
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = UserPreferences().__dict__
        
        # View and filter state
        if 'current_view' not in st.session_state:
            st.session_state.current_view = 'Overview'
        
        if 'filter_state' not in st.session_state:
            st.session_state.filter_state = {
                'date_range': 30,
                'product_codes': [],
                'storage_zone': 'All Zones',
                'priority_filter': ['URGENT', 'HIGH'],
                'alert_days': 30
            }
        
        # Performance and monitoring
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = {
                'page_loads': 0,
                'api_calls': 0,
                'errors': 0,
                'cache_hits': 0
            }
        
        # Theme and UI state
        if 'theme' not in st.session_state:
            st.session_state.theme = 'light'
        
        if 'sidebar_expanded' not in st.session_state:
            st.session_state.sidebar_expanded = True
        
        # Cache management
        if 'cache_last_cleared' not in st.session_state:
            st.session_state.cache_last_cleared = datetime.now()
        
        # Auto-refresh settings
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        
        if 'refresh_rate' not in st.session_state:
            st.session_state.refresh_rate = 30
        
        # Debug and development
        if 'debug_mode' not in st.session_state:
            st.session_state.debug_mode = False
        
        # Application lifecycle
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = False
        
        logger.debug("Session state initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing session state: {e}")
        st.error("❌ Error initializing application state")

def handle_auto_refresh(controls: Dict[str, Any], enhancements: DashboardEnhancements) -> bool:
    """
    Handle auto-refresh logic with intelligent timing and error recovery.
    
    Args:
        controls: User controls from sidebar
        enhancements: Dashboard enhancements instance
        
    Returns:
        True if refresh was triggered, False otherwise
    """
    try:
        if not controls.get('auto_refresh', False):
            return False
        
        refresh_rate = controls.get('refresh_rate', 30)
        time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
        
        # Check if refresh interval has passed
        if time_since_refresh >= refresh_rate:
            # Update API status before refresh
            st.session_state.api_status = check_api_health()
            
            # Exponential backoff for errors
            if st.session_state.error_count > 3:
                backoff_multiplier = min(2 ** (st.session_state.error_count - 3), 8)
                actual_refresh_rate = refresh_rate * backoff_multiplier
                
                if time_since_refresh < actual_refresh_rate:
                    return False
                
                st.warning(f"⚠️ Reduced refresh rate due to errors (every {actual_refresh_rate:.0f}s)")
            
            # Clear cache and update timestamp
            st.cache_data.clear()
            st.session_state.last_refresh = datetime.now()
            st.session_state.cache_last_cleared = datetime.now()
            
            # Reset error count on successful refresh
            if st.session_state.api_status:
                st.session_state.error_count = 0
            
            # Record performance metrics
            enhancements.get_performance_monitor().record_page_load()
            
            logger.info(f"Auto-refresh triggered after {time_since_refresh:.1f} seconds")
            
            # Show refresh indicator
            enhancements.show_toast("🔄 Dashboard refreshed", "info", 2000)
            
            st.rerun()
            return True
            
    except Exception as e:
        logger.error(f"Auto-refresh failed: {e}")
        st.session_state.error_count += 1
        st.session_state.last_error = str(e)
        enhancements.get_error_handler().handle_error(e, "auto_refresh")
    
    return False

def safe_render_component(
    component_func: Callable, 
    component_name: str, 
    enhancements: DashboardEnhancements,
    *args, **kwargs
) -> bool:
    """
    Safely render a component with comprehensive error handling.
    
    Args:
        component_func: Function to render the component
        component_name: Name of the component for error reporting
        enhancements: Dashboard enhancements instance
        *args, **kwargs: Arguments to pass to the component function
        
    Returns:
        True if component rendered successfully, False otherwise
    """
    try:
        start_time = time.time()
        
        with st.spinner(f"Loading {component_name}..."):
            result = component_func(*args, **kwargs)
        
        # Record performance metrics
        execution_time = time.time() - start_time
        enhancements.get_performance_monitor().record_api_call(
            component_name, execution_time, True
        )
        
        logger.debug(f"Component {component_name} rendered successfully in {execution_time:.3f}s")
        return True
        
    except Exception as e:
        # Record error metrics
        execution_time = time.time() - start_time
        enhancements.get_performance_monitor().record_api_call(
            component_name, execution_time, False
        )
        
        # Handle the error
        recovery_action = lambda: st.rerun()
        success = enhancements.get_error_handler().handle_error(e, component_name, recovery_action)
        
        if not success:
            render_error_state(f"Error loading {component_name}: {str(e)}", component_name)
        
        logger.error(f"Error in component {component_name}: {e}")
        return False

def render_view_content(
    view_mode: str, 
    api_client: WMSAPIClient, 
    controls: Dict[str, Any],
    enhancements: DashboardEnhancements
) -> None:
    """
    Render content based on selected view mode with comprehensive error handling.
    
    Args:
        view_mode: Selected view mode
        api_client: API client instance
        controls: User controls and filters
        enhancements: Dashboard enhancements instance
    """
    try:
        # Add navigation breadcrumbs
        render_navigation_breadcrumbs(view_mode)
        
        # Update current view in session state
        st.session_state.current_view = view_mode
        
        if view_mode == "Overview":
            st.markdown("### 🏠 Dashboard Overview")
            st.caption("Complete warehouse operations overview with key metrics and alerts")
            
            # Render alerts first (most critical information)
            safe_render_component(
                render_alerts, "Alerts", enhancements,
                api_client, controls.get('priority_filter', []), controls.get('alert_days', 30)
            )
            st.divider()
            
            # KPI cards
            safe_render_component(render_kpi_cards, "KPI Cards", enhancements, api_client)
            st.divider()
            
            # Dynamic charts
            safe_render_component(
                render_dynamic_charts, "Dynamic Charts", enhancements,
                api_client, controls.get('date_range', 30), controls.get('storage_zone', 'All Zones')
            )
            st.divider()
            
            # Revenue metrics
            safe_render_component(render_revenue_metrics, "Revenue Metrics", enhancements, api_client)
            
        elif view_mode == "Inventory":
            st.markdown("### 📦 Inventory Management")
            st.caption("Comprehensive inventory tracking, FIFO efficiency, and stock management")
            
            # Inventory-specific KPIs
            safe_render_component(render_kpi_cards, "Inventory KPIs", enhancements, api_client)
            st.divider()
            
            # FIFO efficiency dashboard
            safe_render_component(
                render_fifo_efficiency, "FIFO Efficiency", enhancements,
                api_client, controls.get('product_codes')
            )
            st.divider()
            
            # Detailed alerts table
            safe_render_component(
                render_alerts_table, "Alerts Management", enhancements,
                api_client, controls.get('priority_filter', []), controls.get('alert_days', 30)
            )
            
        elif view_mode == "Operations":
            st.markdown("### ⚙️ Operations Center")
            st.caption("Real-time operations monitoring, alerts management, and performance tracking")
            
            # Operations alerts and monitoring
            safe_render_component(
                render_alerts_table, "Operations Alerts", enhancements,
                api_client, controls.get('priority_filter', []), controls.get('alert_days', 30)
            )
            st.divider()
            
            # Operational charts
            safe_render_component(
                render_dynamic_charts, "Operations Charts", enhancements,
                api_client, controls.get('date_range', 30), controls.get('storage_zone', 'All Zones')
            )
            st.divider()
            
            # Performance metrics
            render_performance_metrics_section(enhancements)
            
        elif view_mode == "Analytics":
            st.markdown("### 📊 Advanced Analytics")
            st.caption("Deep dive analytics, FIFO efficiency analysis, and predictive insights")
            
            # FIFO efficiency analysis
            safe_render_component(
                render_fifo_efficiency, "FIFO Analytics", enhancements,
                api_client, controls.get('product_codes')
            )
            st.divider()
            
            # Advanced charts and trends
            safe_render_component(
                render_dynamic_charts, "Analytics Charts", enhancements,
                api_client, controls.get('date_range', 30), controls.get('storage_zone', 'All Zones')
            )
            st.divider()
            
            # Analytics insights
            render_analytics_insights_section(api_client, controls, enhancements)
        
        else:
            st.error(f"❌ Unknown view mode: {view_mode}")
            
    except Exception as e:
        logger.error(f"Error rendering view content for {view_mode}: {e}")
        enhancements.get_error_handler().handle_error(e, f"view_content_{view_mode}")

def render_performance_metrics_section(enhancements: DashboardEnhancements) -> None:
    """Render performance metrics section for operations view."""
    st.markdown("### 🚀 System Performance Metrics")
    
    try:
        perf_data = enhancements.get_performance_monitor().get_performance_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("API Response Time", f"{perf_data['avg_response_time']:.3f}s")
        
        with col2:
            st.metric("Error Rate", f"{perf_data['error_rate']:.1f}%")
        
        with col3:
            st.metric("Cache Hit Rate", f"{perf_data['cache_hit_rate']:.1f}%")
        
        with col4:
            st.metric("Uptime", f"{perf_data['uptime_seconds']:.0f}s")
        
        # Performance trends (mock data for now)
        st.info("📈 Detailed performance analytics and trends coming in the next release!")
        
    except Exception as e:
        logger.error(f"Error rendering performance metrics: {e}")
        st.error("❌ Error loading performance metrics")

def render_analytics_insights_section(
    api_client: WMSAPIClient, 
    controls: Dict[str, Any],
    enhancements: DashboardEnhancements
) -> None:
    """Render analytics insights section for analytics view."""
    st.markdown("### 🔬 Analytics Insights")
    
    try:
        # Predictive analytics placeholder
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🤖 Predictive Analytics**")
            st.progress(0.75)
            st.caption("ML models in development for demand forecasting")
            
            st.markdown("**📈 Trend Analysis**")
            st.progress(0.60)
            st.caption("Advanced trend analysis and pattern recognition")
        
        with col2:
            st.markdown("**🎯 Optimization Recommendations**")
            st.info("Based on current data patterns:")
            st.write("• Consider increasing safety stock for PROD-001")
            st.write("• Zone A utilization is approaching capacity")
            st.write("• Supplier performance trending upward")
            
            st.markdown("**📊 Data Quality Score**")
            st.metric("Overall Score", "94.2%", "+2.1%")
        
        st.info("🔬 Advanced analytics features including ML-powered insights are coming soon!")
        
    except Exception as e:
        logger.error(f"Error rendering analytics insights: {e}")
        st.error("❌ Error loading analytics insights")

def render_enhanced_footer(enhancements: DashboardEnhancements) -> None:
    """
    Render enhanced footer with system information and additional features.
    
    Args:
        enhancements: Dashboard enhancements instance
    """
    try:
        st.markdown("---")
        
        # Main footer content
        footer_col1, footer_col2, footer_col3 = st.columns(3)
        
        with footer_col1:
            st.markdown("**🔧 System Health**")
            health_status = "🟢 Healthy" if st.session_state.api_status else "🔴 Unhealthy"
            st.markdown(f"API Status: {health_status}")
            
            cache_age = (datetime.now() - st.session_state.cache_last_cleared).total_seconds()
            cache_status = "🟢 Fresh" if cache_age < 300 else "🟡 Aging"
            st.markdown(f"Cache Status: {cache_status}")
        
        with footer_col2:
            st.markdown("**⚡ Performance**")
            perf_summary = enhancements.get_performance_monitor().get_performance_summary()
            st.markdown(f"Avg Response: {perf_summary['avg_response_time']:.3f}s")
            st.markdown(f"Error Rate: {perf_summary['error_rate']:.1f}%")
        
        with footer_col3:
            st.markdown("**📊 Session Info**")
            st.markdown(f"Page Loads: {perf_summary['page_loads']}")
            st.markdown(f"Errors: {st.session_state.error_count}")
        
        # Footer metadata
        st.markdown("---")
        
        footer_metadata = f"""
        <div style='text-align: center; color: #6c757d; padding: 1rem;'>
            <p><strong>{DASHBOARD_TITLE}</strong> v{DASHBOARD_VERSION}</p>
            <p>Last Updated: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')} | 
               Session Duration: {(datetime.now() - st.session_state.last_refresh).total_seconds():.0f}s</p>
            <p>
                <a href='https://github.com/ShawSAM37/warehouse-mgmt-system' target='_blank'>📚 Documentation</a> | 
                <a href='https://github.com/ShawSAM37/warehouse-mgmt-system/issues' target='_blank'>🐛 Report Issue</a> | 
                <a href='mailto:support@wms.com'>📧 Support</a>
            </p>
        </div>
        """
        
        st.markdown(footer_metadata, unsafe_allow_html=True)
        
        # Export and utility buttons
        footer_actions_col1, footer_actions_col2, footer_actions_col3 = st.columns(3)
        
        with footer_actions_col1:
            if st.button("📥 Export Dashboard Data", key="export_dashboard_footer"):
                enhancements.show_toast("📥 Export functionality coming soon!", "info")
        
        with footer_actions_col2:
            if st.button("📊 Generate Report", key="generate_report_footer"):
                enhancements.show_toast("📊 Report generation coming soon!", "info")
        
        with footer_actions_col3:
            if st.button("⚙️ System Diagnostics", key="system_diagnostics_footer"):
                show_system_diagnostics(enhancements)
        
    except Exception as e:
        logger.error(f"Error rendering enhanced footer: {e}")
        st.error("❌ Error loading footer")

def show_system_diagnostics(enhancements: DashboardEnhancements) -> None:
    """Show system diagnostics modal."""
    with st.expander("🔧 System Diagnostics", expanded=True):
        diag_col1, diag_col2 = st.columns(2)
        
        with diag_col1:
            st.markdown("**API Health Check**")
            api_healthy = check_api_health()
            st.write(f"Status: {'✅ Healthy' if api_healthy else '❌ Unhealthy'}")
            st.write(f"Response Time: {'< 1s' if api_healthy else 'Timeout'}")
            
            st.markdown("**Cache Information**")
            cache_age = (datetime.now() - st.session_state.cache_last_cleared).total_seconds()
            st.write(f"Last Cleared: {cache_age:.0f}s ago")
            st.write(f"Status: {'Fresh' if cache_age < 300 else 'Aging'}")
        
        with diag_col2:
            st.markdown("**Session Information**")
            st.write(f"Session Duration: {(datetime.now() - st.session_state.last_refresh).total_seconds():.0f}s")
            st.write(f"Error Count: {st.session_state.error_count}")
            st.write(f"Current View: {st.session_state.current_view}")
            
            st.markdown("**Performance Metrics**")
            perf_data = enhancements.get_performance_monitor().get_performance_summary()
            st.write(f"API Calls: {perf_data['total_api_calls']}")
            st.write(f"Cache Hit Rate: {perf_data['cache_hit_rate']:.1f}%")

def main() -> None:
    """
    Main dashboard application entry point with comprehensive initialization and error handling.
    """
    try:
        # Initialize session state first
        initialize_session_state()
        
        # Load configuration
        config = DashboardConfig()
        
        # Initialize enhancements
        enhancements = get_enhancements(config)
        
        # Apply initial styling and setup
        if not st.session_state.app_initialized:
            enhancements.initialize()
            st.session_state.app_initialized = True
            logger.info(f"{DASHBOARD_TITLE} v{DASHBOARD_VERSION} initialized")
        
        # Create API client with error handling
        try:
            api_client = get_api_client()
            logger.debug("API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize API client: {e}")
            st.error("🚨 Failed to connect to WMS API. Please check your connection and try again.")
            
            # Render minimal troubleshooting interface
            st.markdown("## 🔧 Connection Troubleshooting")
            st.markdown(f"**API Endpoint:** {config.api_base_url}")
            st.markdown("**Possible Issues:**")
            st.markdown("- API server is not running")
            st.markdown("- Network connectivity issues")
            st.markdown("- Firewall blocking connections")
            
            if st.button("🔄 Retry Connection"):
                st.cache_resource.clear()
                st.rerun()
            
            return
        
        # Render header
        safe_render_component(render_header, "Header", enhancements)
        
        # Render sidebar and get controls
        controls = {}
        try:
            controls = render_sidebar()
            logger.debug(f"Sidebar controls: {controls}")
        except Exception as e:
            logger.error(f"Sidebar rendering failed: {e}")
            enhancements.get_error_handler().handle_error(e, "sidebar")
            
            # Fallback controls
            controls = {
                'auto_refresh': True,
                'refresh_rate': 30,
                'date_range': 30,
                'view_mode': 'Overview',
                'product_codes': [],
                'storage_zone': 'All Zones',
                'priority_filter': ['URGENT', 'HIGH'],
                'alert_days': 30,
                'theme': 'light'
            }
        
        # Apply theme if changed
        if controls.get('theme') != st.session_state.theme:
            st.session_state.theme = controls['theme']
            enhancements.styling.apply_theme(controls['theme'])
        
        # Handle auto-refresh
        refresh_triggered = handle_auto_refresh(controls, enhancements)
        
        if not refresh_triggered:  # Only render content if not refreshing
            # Render main content based on view mode
            view_mode = controls.get('view_mode', 'Overview')
            
            try:
                render_view_content(view_mode, api_client, controls, enhancements)
            except Exception as e:
                logger.error(f"View content rendering failed: {e}")
                enhancements.get_error_handler().handle_error(e, f"view_content_{view_mode}")
                
                # Offer recovery options
                recovery_col1, recovery_col2 = st.columns(2)
                with recovery_col1:
                    if st.button("🔄 Refresh Page"):
                        st.rerun()
                with recovery_col2:
                    if st.button("🏠 Go to Overview"):
                        st.session_state.current_view = 'Overview'
                        st.rerun()
        
        # Render debug panel if enabled
        if config.enable_debug or st.session_state.debug_mode:
            enhancements.get_debug_manager().render_debug_panel()
        
        # Render enhanced footer
        safe_render_component(render_enhanced_footer, "Footer", enhancements, enhancements)
        
        # Setup accessibility features
        render_accessibility_features()
        
        # Log successful completion
        logger.debug("Dashboard rendered successfully")
        
    except Exception as e:
        logger.critical(f"Critical error in main application: {e}")
        logger.critical(f"Traceback: {traceback.format_exc()}")
        
        st.error("🚨 Critical application error occurred. Please refresh the page or contact support.")
        
        # Emergency diagnostic information
        with st.expander("🔧 Emergency Diagnostics", expanded=False):
            st.code(f"Error: {str(e)}")
            st.code(f"Time: {datetime.now()}")
            st.code(f"Session State Keys: {list(st.session_state.keys())}")
            st.code(f"Traceback: {traceback.format_exc()}")

# Application entry point
if __name__ == "__main__":
    try:
        # Set page config first (must be first Streamlit command)
        st.set_page_config(
            page_title=DASHBOARD_TITLE,
            page_icon="📦",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/ShawSAM37/warehouse-mgmt-system',
                'Report a bug': 'https://github.com/ShawSAM37/warehouse-mgmt-system/issues',
                'About': f"{DASHBOARD_TITLE} v{DASHBOARD_VERSION}"
            }
        )
        
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
        logger.critical(f"Traceback: {traceback.format_exc()}")
