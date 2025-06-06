"""
Helper Utilities for WMS Dashboard
==================================

Collection of utility functions for formatting, validation, chart creation,
and API health checks used throughout the dashboard.
"""

import re
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Optional, Dict, Union
import logging

logger = logging.getLogger(__name__)

# Color scheme for consistent styling
PRIORITY_COLORS = {
    'URGENT': '#d62728',
    'HIGH': '#ff7f0e',
    'MEDIUM': '#1f77b4',
    'LOW': '#2ca02c',
    'DEFAULT': '#7f7f7f'
}

CHART_COLORS = {
    'primary': '#1f77b4',
    'success': '#2ca02c',
    'warning': '#ff7f0e',
    'danger': '#d62728',
    'neutral': '#7f7f7f'
}

def format_number(value: Union[int, float, Decimal], suffix: str = "", precision: int = 1) -> str:
    """
    Format numbers with appropriate suffixes (K, M, B, T).
    
    Args:
        value: Number to format
        suffix: Optional suffix to append (e.g., " units", " $")
        precision: Decimal places for formatted number
        
    Returns:
        Formatted number string
        
    Examples:
        >>> format_number(1500)
        '1.5K'
        >>> format_number(2500000, ' units')
        '2.5M units'
        >>> format_number(1234.56, '$', 2)
        '$1.23K'
    """
    if value is None:
        return "0"
    
    try:
        num_value = float(value)
        abs_value = abs(num_value)
        
        if abs_value >= 1_000_000_000_000:  # Trillion
            formatted = f"{num_value/1_000_000_000_000:.{precision}f}T"
        elif abs_value >= 1_000_000_000:  # Billion
            formatted = f"{num_value/1_000_000_000:.{precision}f}B"
        elif abs_value >= 1_000_000:  # Million
            formatted = f"{num_value/1_000_000:.{precision}f}M"
        elif abs_value >= 1_000:  # Thousand
            formatted = f"{num_value/1_000:.{precision}f}K"
        else:
            # For small numbers, use appropriate precision
            if abs_value < 1 and abs_value > 0:
                formatted = f"{num_value:.{precision+2}f}"
            else:
                formatted = f"{num_value:.{precision}f}" if precision > 0 else f"{num_value:.0f}"
        
        return f"{formatted}{suffix}"
        
    except (TypeError, ValueError, OverflowError):
        logger.warning(f"Could not format number: {value}")
        return f"0{suffix}"

def get_priority_color(priority: str) -> str:
    """
    Get color code based on priority level.
    
    Args:
        priority: Priority level (URGENT, HIGH, MEDIUM, LOW)
        
    Returns:
        Hex color code
        
    Examples:
        >>> get_priority_color('URGENT')
        '#d62728'
        >>> get_priority_color('unknown')
        '#7f7f7f'
    """
    return PRIORITY_COLORS.get(priority.upper(), PRIORITY_COLORS['DEFAULT'])

def create_gauge_chart(
    value: float, 
    title: str, 
    max_value: float = 100,
    min_value: float = 0,
    thresholds: Optional[Dict[str, float]] = None,
    unit: str = "%"
) -> go.Figure:
    """
    Create a gauge chart for displaying metrics.
    
    Args:
        value: Current value
        title: Chart title
        max_value: Maximum value for the gauge
        min_value: Minimum value for the gauge
        thresholds: Dict with 'good', 'warning', 'critical' thresholds
        unit: Unit to display with the value
        
    Returns:
        Plotly gauge chart figure
        
    Examples:
        >>> fig = create_gauge_chart(85, "FIFO Efficiency", 100)
        >>> fig = create_gauge_chart(2.5, "Response Time", 5, 0, unit="s")
    """
    if thresholds is None:
        thresholds = {
            'good': max_value * 0.8,
            'warning': max_value * 0.6,
            'critical': max_value * 0.4
        }
    
    # Determine gauge color based on value
    if value >= thresholds['good']:
        gauge_color = CHART_COLORS['success']
    elif value >= thresholds['warning']:
        gauge_color = CHART_COLORS['warning']
    else:
        gauge_color = CHART_COLORS['danger']
    
    # Create gauge steps
    steps = [
        {'range': [min_value, thresholds['critical']], 'color': '#ffebee'},
        {'range': [thresholds['critical'], thresholds['warning']], 'color': '#fff3e0'},
        {'range': [thresholds['warning'], thresholds['good']], 'color': '#f3e5f5'},
        {'range': [thresholds['good'], max_value], 'color': '#e8f5e8'}
    ]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': '#212529'}},
        number={'suffix': unit, 'font': {'size': 20}},
        delta={
            'reference': thresholds['good'], 
            'increasing': {'color': CHART_COLORS['success']},
            'decreasing': {'color': CHART_COLORS['danger']}
        },
        gauge={
            'axis': {
                'range': [min_value, max_value], 
                'tickwidth': 1,
                'tickcolor': '#212529'
            },
            'bar': {'color': gauge_color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': '#dee2e6',
            'steps': steps,
            'threshold': {
                'line': {'color': CHART_COLORS['danger'], 'width': 4},
                'thickness': 0.75,
                'value': thresholds['critical']
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        font={'color': '#212529', 'family': "Arial, sans-serif"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def safe_get(data: Optional[Dict], key: str, default: Any = None) -> Any:
    """
    Safely get value from dictionary with null checking.
    
    Args:
        data: Dictionary to get value from (can be None)
        key: Key to retrieve
        default: Default value if key not found or data is None
        
    Returns:
        Value from dictionary or default
        
    Examples:
        >>> safe_get({'a': 1}, 'a')
        1
        >>> safe_get(None, 'a', 'default')
        'default'
        >>> safe_get({'a': 1}, 'b', 0)
        0
    """
    if data is None:
        return default
    
    try:
        return data.get(key, default)
    except (AttributeError, TypeError):
        logger.warning(f"safe_get failed for key '{key}' on data type {type(data)}")
        return default

def check_api_health(api_url: str = "http://localhost:8000", timeout: int = 5) -> bool:
    """
    Check if the API is accessible and healthy.
    
    Args:
        api_url: Base URL of the API
        timeout: Request timeout in seconds
        
    Returns:
        True if API is healthy, False otherwise
        
    Examples:
        >>> check_api_health("http://localhost:8000")
        True
        >>> check_api_health("http://invalid-url")
        False
    """
    try:
        health_url = f"{api_url.rstrip('/')}/health"
        response = requests.get(health_url, timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking API health: {e}")
        return False

def validate_api_response(response: Dict, required_fields: Optional[list] = None) -> bool:
    """
    Validate API response structure.
    
    Args:
        response: API response dictionary
        required_fields: List of required field names
        
    Returns:
        True if response is valid, False otherwise
    """
    if not isinstance(response, dict):
        return False
    
    if required_fields:
        return all(field in response for field in required_fields)
    
    return True

def calculate_percentage_change(current: float, previous: float) -> Optional[float]:
    """
    Calculate percentage change between two values.
    
    Args:
        current: Current value
        previous: Previous value
        
    Returns:
        Percentage change or None if calculation not possible
        
    Examples:
        >>> calculate_percentage_change(110, 100)
        10.0
        >>> calculate_percentage_change(90, 100)
        -10.0
    """
    try:
        if previous == 0:
            return None
        return ((current - previous) / previous) * 100
    except (TypeError, ZeroDivisionError):
        return None

def format_datetime(dt: datetime, format_type: str = "short") -> str:
    """
    Format datetime for display.
    
    Args:
        dt: Datetime object to format
        format_type: Format type ('short', 'long', 'time_only', 'date_only')
        
    Returns:
        Formatted datetime string
    """
    if not isinstance(dt, datetime):
        return str(dt)
    
    formats = {
        'short': '%m/%d %H:%M',
        'long': '%Y-%m-%d %H:%M:%S',
        'time_only': '%H:%M:%S',
        'date_only': '%Y-%m-%d',
        'relative': None  # Special case for relative time
    }
    
    if format_type == 'relative':
        now = datetime.now()
        diff = now - dt
        
        if diff.days > 0:
            return f"{diff.days} days ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hours ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minutes ago"
        else:
            return "Just now"
    
    format_str = formats.get(format_type, formats['short'])
    return dt.strftime(format_str)

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system usage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        max_name_length = 255 - len(ext) - 1 if ext else 255
        sanitized = f"{name[:max_name_length]}.{ext}" if ext else name[:255]
    
    return sanitized or 'untitled'

def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate text to specified length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to append when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def parse_size_string(size_str: str) -> int:
    """
    Parse size string (e.g., "10MB", "1.5GB") to bytes.
    
    Args:
        size_str: Size string with unit
        
    Returns:
        Size in bytes
    """
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3,
        'TB': 1024**4
    }
    
    size_str = size_str.upper().strip()
    
    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            try:
                number = float(size_str[:-len(unit)])
                return int(number * multiplier)
            except ValueError:
                pass
    
    # Try to parse as plain number (assume bytes)
    try:
        return int(float(size_str))
    except ValueError:
        return 0
