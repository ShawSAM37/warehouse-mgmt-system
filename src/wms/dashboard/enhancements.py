"""
Production Enhancements for WMS Dashboard
=========================================

Comprehensive enhancements including advanced styling, export functionality,
performance optimizations, user experience improvements, monitoring, debugging,
security features, and automation capabilities.
"""

import streamlit as st
import base64
import io
import json
import os
import logging
import smtplib
import schedule
import time
import threading
import hashlib
import uuid
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional, List, Dict, Any, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path

# Import reportlab for PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logging.warning("ReportLab not available. PDF export functionality will be disabled.")

# Import utilities
from .utils.cache import CacheManager
from .utils.config import DashboardConfig
from .utils.helpers import format_number, get_priority_color, safe_get, sanitize_filename

logger = logging.getLogger(__name__)

class DashboardStyling:
    """
    Advanced styling system for the dashboard with modern CSS,
    responsive design, and accessibility features.
    """
    
    CSS = """
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
        --border-radius: 12px;
        --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Dark Theme Variables */
    [data-theme="dark"] {
        --background-color: #1a1a1a;
        --surface-color: #2d2d2d;
        --text-color: #ffffff;
        --border-color: #404040;
        --shadow-light: 0 2px 4px rgba(0,0,0,0.3);
        --shadow-medium: 0 4px 8px rgba(0,0,0,0.4);
        --shadow-heavy: 0 8px 16px rgba(0,0,0,0.5);
    }
    
    /* Global Styles */
    .main {
        padding-top: 1rem;
        background: var(--background-color);
        color: var(--text-color);
        font-family: var(--font-family);
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
    
    /* Enhanced KPI Cards */
    .kpi-card {
        background: var(--surface-color);
        padding: 1.5rem;
        border-radius: var(--border-radius);
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
        border-radius: var(--border-radius) var(--border-radius) 0 0;
    }
    
    /* Loading Animations */
    .loading-skeleton {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200% 100%;
        animation: loading 1.5s infinite;
        border-radius: var(--border-radius);
        height: 20px;
        margin: 10px 0;
    }
    
    .spinner {
        width: 40px;
        height: 40px;
        border: 4px solid var(--border-color);
        border-top: 4px solid var(--primary-color);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    /* Alert Banners */
    .alert-banner {
        padding: 1rem 1.5rem;
        border-radius: var(--border-radius);
        margin-bottom: 1.5rem;
        font-weight: 600;
        border-left: 5px solid;
        animation: slideInFromLeft 0.4s ease-out;
        position: relative;
        overflow: hidden;
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
    
    /* Toast Notifications */
    .toast {
        position: fixed;
        top: 20px;
        right: 20px;
        background: var(--surface-color);
        color: var(--text-color);
        padding: 1rem 1.5rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-heavy);
        border-left: 4px solid var(--success-color);
        animation: slideInFromRight 0.3s ease-out;
        z-index: 1000;
        max-width: 400px;
    }
    
    .toast.error {
        border-left-color: var(--danger-color);
    }
    
    .toast.warning {
        border-left-color: var(--warning-color);
    }
    
    /* Responsive Design */
    @media (max-width: 1200px) {
        .main {
            padding: 0.5rem;
        }
    }
    
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
            padding: 0.25rem;
        }
        
        .main-header h1 {
            font-size: 1.5rem;
        }
        
        .toast {
            position: fixed;
            top: 10px;
            left: 10px;
            right: 10px;
            max-width: none;
        }
    }
    
    /* Accessibility Features */
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
    button:focus, input:focus, select:focus, textarea:focus {
        outline: 3px solid var(--primary-color);
        outline-offset: 2px;
    }
    
    /* High Contrast Mode */
    @media (prefers-contrast: high) {
        :root {
            --shadow-light: 0 2px 4px rgba(0,0,0,0.4);
            --shadow-medium: 0 4px 8px rgba(0,0,0,0.5);
            --shadow-heavy: 0 8px 16px rgba(0,0,0,0.6);
        }
        
        .kpi-card, .alert-banner {
            border-width: 2px;
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
    
    /* Animations */
    @keyframes slideInFromTop {
        0% { transform: translateY(-50px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes slideInFromLeft {
        0% { transform: translateX(-50px); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInFromRight {
        0% { transform: translateX(50px); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    @keyframes loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.8; }
    }
    </style>
    """
    
    @staticmethod
    def inject_css():
        """Inject the advanced CSS into the Streamlit app."""
        st.markdown(DashboardStyling.CSS, unsafe_allow_html=True)
    
    @staticmethod
    def apply_theme(theme: str = "light"):
        """Apply theme to the dashboard."""
        theme_script = f"""
        <script>
        document.documentElement.setAttribute('data-theme', '{theme}');
        localStorage.setItem('dashboard-theme', '{theme}');
        </script>
        """
        st.markdown(theme_script, unsafe_allow_html=True)

@dataclass
class UserPreferences:
    """User preferences and settings management."""
    theme: str = "light"
    auto_refresh: bool = True
    refresh_rate: int = 30
    default_view: str = "Overview"
    favorite_views: List[str] = field(default_factory=list)
    keyboard_shortcuts: bool = True
    high_contrast: bool = False
    reduced_motion: bool = False
    notification_preferences: Dict[str, bool] = field(default_factory=lambda: {
        'critical_alerts': True,
        'low_stock_warnings': True,
        'system_notifications': False
    })
    
    def save_to_session(self):
        """Save preferences to Streamlit session state."""
        st.session_state.user_preferences = self.__dict__
    
    @classmethod
    def load_from_session(cls) -> 'UserPreferences':
        """Load preferences from Streamlit session state."""
        if 'user_preferences' in st.session_state:
            return cls(**st.session_state.user_preferences)
        return cls()
    
    def toggle_theme(self) -> str:
        """Toggle between light and dark theme."""
        self.theme = 'dark' if self.theme == 'light' else 'light'
        self.save_to_session()
        return self.theme

class ExportManager:
    """
    Comprehensive export functionality for dashboard data including
    CSV, PDF, and automated report generation with email integration.
    """
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.export_dir = Path(config.export_temp_dir)
        self.export_dir.mkdir(exist_ok=True)
    
    def export_to_csv(self, data: List[Dict], filename: str) -> bytes:
        """
        Export data to CSV format with proper formatting.
        
        Args:
            data: List of dictionaries to export
            filename: Filename for the export
            
        Returns:
            CSV data as bytes
        """
        try:
            if not data:
                return b"No data available for export"
            
            import pandas as pd
            df = pd.DataFrame(data)
            
            # Format datetime columns
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        pass
            
            # Add metadata
            metadata = [
                ['Export Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Dashboard Version', '1.0.0'],
                ['Total Records', len(df)],
                ['', '']  # Empty row separator
            ]
            
            # Create CSV with metadata
            csv_buffer = io.StringIO()
            
            # Write metadata
            for row in metadata:
                csv_buffer.write(','.join(str(item) for item in row) + '\n')
            
            # Write data
            df.to_csv(csv_buffer, index=False)
            
            return csv_buffer.getvalue().encode('utf-8')
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return f"Error exporting data: {str(e)}".encode('utf-8')
    
    def export_to_pdf(self, dashboard_data: Dict[str, Any], title: str = "WMS Dashboard Report") -> bytes:
        """
        Generate comprehensive PDF report with professional layout.
        
        Args:
            dashboard_data: Dictionary containing dashboard data
            title: Report title
            
        Returns:
            PDF data as bytes
        """
        if not REPORTLAB_AVAILABLE:
            return b"PDF export not available. Please install reportlab."
        
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch)
            styles = getSampleStyleSheet()
            story = []
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.HexColor('#1f77b4'),
                alignment=1  # Center alignment
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=12,
                textColor=colors.HexColor('#2e3338')
            )
            
            # Title
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 12))
            
            # Report metadata
            metadata = [
                ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Dashboard Version:', '1.0.0'],
                ['Data Source:', self.config.api_base_url]
            ]
            
            metadata_table = Table(metadata, colWidths=[2*inch, 3*inch])
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(metadata_table)
            story.append(Spacer(1, 20))
            
            # KPI Summary
            if 'inventory_summary' in dashboard_data:
                story.append(Paragraph("Key Performance Indicators", heading_style))
                kpi_data = dashboard_data['inventory_summary']
                
                kpi_table_data = [
                    ['Metric', 'Value', 'Status'],
                    ['Total Products', str(safe_get(kpi_data, 'total_products', 'N/A')), '✓'],
                    ['Total Inventory', f"{safe_get(kpi_data, 'total_quantity', 'N/A')} units", '✓'],
                    ['Low Stock Items', str(safe_get(kpi_data, 'low_stock_count', 'N/A')), 
                     '⚠️' if safe_get(kpi_data, 'low_stock_count', 0) > 0 else '✓'],
                    ['Expiring Soon', str(safe_get(kpi_data, 'expiring_soon_count', 'N/A')),
                     '⚠️' if safe_get(kpi_data, 'expiring_soon_count', 0) > 0 else '✓']
                ]
                
                kpi_table = Table(kpi_table_data, colWidths=[2*inch, 1.5*inch, 1*inch])
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
            
            # Alerts Summary
            if 'alerts' in dashboard_data and dashboard_data['alerts']:
                story.append(Paragraph("Critical Alerts", heading_style))
                alerts = dashboard_data['alerts'][:10]  # Top 10 alerts
                
                alert_data = [['Product Code', 'Days to Expiry', 'Quantity', 'Priority']]
                for alert in alerts:
                    alert_data.append([
                        safe_get(alert, 'product_code', 'Unknown'),
                        str(safe_get(alert, 'days_until_expiry', 'N/A')),
                        str(safe_get(alert, 'quantity', 'N/A')),
                        safe_get(alert, 'priority', 'Unknown')
                    ])
                
                alert_table = Table(alert_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch])
                alert_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.red),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightpink),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(alert_table)
            
            # Build PDF
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            return f"Error generating PDF: {str(e)}".encode('utf-8')
    
    def schedule_report(self, report_type: str, frequency: str, email: str, data_callback: Callable) -> bool:
        """
        Schedule automated report generation and delivery.
        
        Args:
            report_type: Type of report ('pdf' or 'csv')
            frequency: Report frequency ('daily', 'weekly', 'monthly')
            email: Email address for delivery
            data_callback: Function to get current data
            
        Returns:
            True if scheduling was successful
        """
        try:
            def generate_and_send():
                try:
                    # Get current data
                    dashboard_data = data_callback()
                    
                    # Generate report
                    if report_type.lower() == 'pdf':
                        report_data = self.export_to_pdf(dashboard_data)
                        filename = f"wms_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        content_type = 'application/pdf'
                    else:
                        # Default to CSV
                        alerts_data = dashboard_data.get('alerts', [])
                        report_data = self.export_to_csv(alerts_data, 'dashboard_export.csv')
                        filename = f"wms_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        content_type = 'text/csv'
                    
                    # Send email
                    self._send_email_report(email, report_data, filename, content_type)
                    logger.info(f"Scheduled {report_type} report sent to {email}")
                    
                except Exception as e:
                    logger.error(f"Error in scheduled report generation: {e}")
            
            # Schedule based on frequency
            if frequency.lower() == 'daily':
                schedule.every().day.at("08:00").do(generate_and_send)
            elif frequency.lower() == 'weekly':
                schedule.every().monday.at("08:00").do(generate_and_send)
            elif frequency.lower() == 'monthly':
                schedule.every().month.do(generate_and_send)
            else:
                logger.error(f"Invalid frequency: {frequency}")
                return False
            
            # Start scheduler thread if not already running
            if not hasattr(self, '_scheduler_thread') or not self._scheduler_thread.is_alive():
                self._start_scheduler_thread()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule report: {e}")
            return False
    
    def _send_email_report(self, email: str, report_data: bytes, filename: str, content_type: str) -> None:
        """Send report via email with proper formatting."""
        email_config = self.config.get_email_config()
        
        if not email_config['enabled']:
            logger.warning("Email configuration not available")
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config['username']
            msg['To'] = email
            msg['Subject'] = f"WMS Dashboard Report - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Email body
            body = f"""
            Warehouse Management System Dashboard Report
            
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            Report Type: {filename.split('.')[-1].upper()}
            File Size: {len(report_data)} bytes
            
            This automated report contains the latest warehouse metrics and analytics.
            
            Best regards,
            WMS Dashboard System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach report
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(report_data)
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename= {filename}')
            msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            if email_config['use_tls']:
                server.starttls()
            server.login(email_config['username'], email_config['password'])
            text = msg.as_string()
            server.sendmail(email_config['username'], email, text)
            server.quit()
            
            logger.info(f"Report sent successfully to {email}")
            
        except Exception as e:
            logger.error(f"Failed to send email report: {e}")
    
    def _start_scheduler_thread(self):
        """Start background scheduler thread."""
        def scheduler_worker():
            while True:
                try:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Error in scheduler thread: {e}")
                    time.sleep(60)
        
        self._scheduler_thread = threading.Thread(target=scheduler_worker, daemon=True)
        self._scheduler_thread.start()
        logger.info("Email scheduler thread started")

class PerformanceMonitor:
    """
    Performance monitoring and optimization system for the dashboard.
    """
    
    def __init__(self):
        self.metrics = {
            'api_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'page_loads': 0,
            'errors': 0,
            'response_times': []
        }
        self.start_time = time.time()
    
    def record_api_call(self, endpoint: str, response_time: float, success: bool):
        """Record API call metrics."""
        self.metrics['api_calls'] += 1
        self.metrics['response_times'].append(response_time)
        
        if not success:
            self.metrics['errors'] += 1
        
        # Keep only last 100 response times
        if len(self.metrics['response_times']) > 100:
            self.metrics['response_times'] = self.metrics['response_times'][-100:]
    
    def record_cache_event(self, hit: bool):
        """Record cache hit/miss."""
        if hit:
            self.metrics['cache_hits'] += 1
        else:
            self.metrics['cache_misses'] += 1
    
    def record_page_load(self):
        """Record page load event."""
        self.metrics['page_loads'] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        uptime = time.time() - self.start_time
        response_times = self.metrics['response_times']
        
        return {
            'uptime_seconds': uptime,
            'total_api_calls': self.metrics['api_calls'],
            'total_errors': self.metrics['errors'],
            'error_rate': (self.metrics['errors'] / max(self.metrics['api_calls'], 1)) * 100,
            'cache_hit_rate': (self.metrics['cache_hits'] / max(self.metrics['cache_hits'] + self.metrics['cache_misses'], 1)) * 100,
            'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0,
            'page_loads': self.metrics['page_loads']
        }

class DebugManager:
    """
    Comprehensive debugging and monitoring functionality for development and production.
    """
    
    def __init__(self):
        self.debug_enabled = False
        self.logs = []
        self.performance_monitor = PerformanceMonitor()
        self.system_info = self._collect_system_info()
    
    def enable_debug_mode(self):
        """Enable debug mode with enhanced logging."""
        self.debug_enabled = True
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    
    def disable_debug_mode(self):
        """Disable debug mode."""
        self.debug_enabled = False
        logging.getLogger().setLevel(logging.INFO)
        logger.info("Debug mode disabled")
    
    def log_event(self, level: str, message: str, context: Dict[str, Any] = None):
        """Log debug event with context."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'context': context or {}
        }
        
        self.logs.append(event)
        
        # Keep only last 1000 logs
        if len(self.logs) > 1000:
            self.logs = self.logs[-1000:]
        
        if self.debug_enabled:
            logger.log(getattr(logging, level.upper(), logging.INFO), f"{message} | Context: {context}")
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for debugging."""
        try:
            import platform
            import psutil
            
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
            }
        except ImportError:
            return {'note': 'psutil not available for system monitoring'}
        except Exception as e:
            return {'error': str(e)}
    
    def render_debug_panel(self):
        """Render comprehensive debug panel in Streamlit."""
        if not self.debug_enabled:
            return
        
        with st.expander("🔧 Debug Panel", expanded=False):
            tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Logs", "System", "Cache"])
            
            with tab1:
                st.subheader("Performance Metrics")
                perf_data = self.performance_monitor.get_performance_summary()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("API Calls", perf_data['total_api_calls'])
                col2.metric("Error Rate", f"{perf_data['error_rate']:.1f}%")
                col3.metric("Cache Hit Rate", f"{perf_data['cache_hit_rate']:.1f}%")
                
                col4, col5, col6 = st.columns(3)
                col4.metric("Avg Response Time", f"{perf_data['avg_response_time']:.3f}s")
                col5.metric("Page Loads", perf_data['page_loads'])
                col6.metric("Uptime", f"{perf_data['uptime_seconds']:.0f}s")
            
            with tab2:
                st.subheader("Recent Logs")
                if self.logs:
                    for log in self.logs[-20:]:  # Show last 20 logs
                        level_color = {
                            'ERROR': '🔴',
                            'WARNING': '🟡',
                            'INFO': '🔵',
                            'DEBUG': '⚪'
                        }.get(log['level'], '⚪')
                        
                        st.text(f"{level_color} {log['timestamp']} - {log['message']}")
                        if log['context']:
                            st.json(log['context'])
                else:
                    st.info("No logs available")
            
            with tab3:
                st.subheader("System Information")
                st.json(self.system_info)
            
            with tab4:
                st.subheader("Cache Status")
                # This would integrate with the actual cache manager
                st.info("Cache monitoring integration pending")

class ErrorHandler:
    """
    Comprehensive error handling with recovery strategies and user feedback.
    """
    
    def __init__(self):
        self.error_count = 0
        self.last_error_time = None
        self.error_history = []
    
    def handle_error(self, error: Exception, context: str = "", recovery_action: Callable = None) -> bool:
        """
        Handle errors with appropriate recovery strategy and user feedback.
        
        Args:
            error: The exception that occurred
            context: Context information about where the error occurred
            recovery_action: Optional recovery action to attempt
            
        Returns:
            True if error was handled successfully, False otherwise
        """
        self.error_count += 1
        self.last_error_time = datetime.now()
        
        error_info = {
            'timestamp': self.last_error_time.isoformat(),
            'error_type': type(error).__name__,
            'message': str(error),
            'context': context
        }
        
        self.error_history.append(error_info)
        
        # Keep only last 50 errors
        if len(self.error_history) > 50:
            self.error_history = self.error_history[-50:]
        
        logger.error(f"Error in {context}: {error}")
        
        # Attempt recovery if provided
        if recovery_action:
            try:
                recovery_action()
                st.success("✅ Error resolved automatically")
                return True
            except Exception as recovery_error:
                logger.error(f"Recovery action failed: {recovery_error}")
        
        # Display user-friendly error message
        self._display_error_message(error, context)
        return False
    
    def _display_error_message(self, error: Exception, context: str):
        """Display user-friendly error message with troubleshooting tips."""
        error_type = type(error).__name__
        
        if "Connection" in error_type:
            st.error("🔌 Connection Error: Unable to connect to the server. Please check your internet connection and try again.")
        elif "Timeout" in error_type:
            st.error("⏱️ Timeout Error: The request took too long to complete. Please try again or contact support if the issue persists.")
        elif "Permission" in error_type or "Unauthorized" in error_type:
            st.error("🔒 Permission Error: You don't have permission to access this resource. Please contact your administrator.")
        else:
            st.error(f"❌ An error occurred in {context}: {str(error)}")
        
        with st.expander("🔧 Troubleshooting Tips"):
            st.markdown("""
            **Common solutions:**
            1. Refresh the page (Ctrl+R or Cmd+R)
            2. Clear your browser cache
            3. Check your internet connection
            4. Try again in a few minutes
            5. Contact support if the issue persists
            """)

class DashboardEnhancements:
    """
    Main enhancements coordinator that brings together all enhancement features.
    """
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.styling = DashboardStyling()
        self.user_prefs = UserPreferences.load_from_session()
        self.export_manager = ExportManager(config)
        self.debug_manager = DebugManager()
        self.error_handler = ErrorHandler()
        self.performance_monitor = PerformanceMonitor()
        
        # Enable debug mode if configured
        if config.enable_debug:
            self.debug_manager.enable_debug_mode()
    
    def initialize(self):
        """Initialize all enhancement features."""
        # Apply styling
        self.styling.inject_css()
        
        # Apply theme
        self.styling.apply_theme(self.user_prefs.theme)
        
        # Setup keyboard shortcuts
        self._setup_keyboard_shortcuts()
        
        # Record page load
        self.performance_monitor.record_page_load()
        
        logger.info("Dashboard enhancements initialized")
    
    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for common actions."""
        if not self.user_prefs.keyboard_shortcuts:
            return
        
        shortcuts_js = """
        <script>
        document.addEventListener('keydown', function(e) {
            // Ctrl+R: Refresh dashboard
            if (e.ctrlKey && e.key === 'r') {
                e.preventDefault();
                window.location.reload();
            }
            
            // Ctrl+1-4: Switch views
            if (e.ctrlKey && ['1', '2', '3', '4'].includes(e.key)) {
                e.preventDefault();
                const views = ['Overview', 'Inventory', 'Operations', 'Analytics'];
                const viewIndex = parseInt(e.key) - 1;
                if (views[viewIndex]) {
                    console.log('Switch to view:', views[viewIndex]);
                    // This would need integration with session state
                }
            }
            
            // Ctrl+T: Toggle theme
            if (e.ctrlKey && e.key === 't') {
                e.preventDefault();
                console.log('Toggle theme shortcut');
            }
            
            // Ctrl+E: Export data
            if (e.ctrlKey && e.key === 'e') {
                e.preventDefault();
                console.log('Export data shortcut');
            }
            
            // Escape: Close modals/expanders
            if (e.key === 'Escape') {
                console.log('Escape pressed');
            }
        });
        </script>
        """
        
        st.markdown(shortcuts_js, unsafe_allow_html=True)
    
    def show_toast(self, message: str, toast_type: str = "success", duration: int = 3000):
        """
        Show toast notification to user.
        
        Args:
            message: Message to display
            toast_type: Type of toast ('success', 'error', 'warning', 'info')
            duration: Duration in milliseconds
        """
        toast_html = f"""
        <div class="toast {toast_type}" id="toast-{int(time.time())}">
            {message}
        </div>
        <script>
        setTimeout(function() {{
            const toast = document.getElementById('toast-{int(time.time())}');
            if (toast) {{
                toast.style.opacity = '0';
                setTimeout(() => toast.remove(), 300);
            }}
        }}, {duration});
        </script>
        """
        
        st.markdown(toast_html, unsafe_allow_html=True)
    
    def get_export_manager(self) -> ExportManager:
        """Get the export manager instance."""
        return self.export_manager
    
    def get_debug_manager(self) -> DebugManager:
        """Get the debug manager instance."""
        return self.debug_manager
    
    def get_error_handler(self) -> ErrorHandler:
        """Get the error handler instance."""
        return self.error_handler
    
    def get_performance_monitor(self) -> PerformanceMonitor:
        """Get the performance monitor instance."""
        return self.performance_monitor

# Global enhancements instance
_enhancements_instance = None

def get_enhancements(config: DashboardConfig = None) -> DashboardEnhancements:
    """Get or create global enhancements instance."""
    global _enhancements_instance
    
    if _enhancements_instance is None:
        if config is None:
            from .utils.config import config as default_config
            config = default_config
        _enhancements_instance = DashboardEnhancements(config)
    
    return _enhancements_instance
