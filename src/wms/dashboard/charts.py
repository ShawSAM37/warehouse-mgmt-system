"""
Chart and Visualization Components for WMS Dashboard
===================================================

Comprehensive collection of interactive charts and visualizations using Plotly
for the warehouse management dashboard including trends, performance metrics,
FIFO efficiency, and financial analytics.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
import logging
import random
import io
import base64

from .api_client import WMSAPIClient
from .utils.helpers import (
    format_number, 
    get_priority_color, 
    safe_get, 
    format_datetime,
    create_gauge_chart
)
from .utils.config import DashboardConfig

logger = logging.getLogger(__name__)

# Chart color schemes
CHART_COLORS = {
    'primary': '#1f77b4',
    'success': '#2ca02c',
    'warning': '#ff7f0e',
    'danger': '#d62728',
    'neutral': '#7f7f7f',
    'background': '#ffffff',
    'grid': '#e1e5e9',
    'text': '#2e3338'
}

# Plotly theme configuration
PLOTLY_THEME = {
    'layout': {
        'font': {'family': 'Inter, Arial, sans-serif', 'size': 12, 'color': CHART_COLORS['text']},
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'colorway': [CHART_COLORS['primary'], CHART_COLORS['success'], CHART_COLORS['warning'], 
                    CHART_COLORS['danger'], CHART_COLORS['neutral']],
        'hovermode': 'x unified',
        'showlegend': True,
        'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1}
    }
}

def apply_chart_styling(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """
    Apply consistent styling to Plotly charts.
    
    Args:
        fig: Plotly figure object
        title: Chart title
        height: Chart height in pixels
        
    Returns:
        Styled Plotly figure
    """
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'weight': 'bold'}
        },
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        **PLOTLY_THEME['layout']
    )
    
    # Update axes styling
    fig.update_xaxes(
        gridcolor=CHART_COLORS['grid'],
        gridwidth=1,
        showgrid=True,
        zeroline=False
    )
    
    fig.update_yaxes(
        gridcolor=CHART_COLORS['grid'],
        gridwidth=1,
        showgrid=True,
        zeroline=False
    )
    
    return fig

def format_chart_data(data: List[Dict], x_col: str, y_col: str) -> pd.DataFrame:
    """
    Format and validate chart data.
    
    Args:
        data: Raw data from API
        x_col: X-axis column name
        y_col: Y-axis column name
        
    Returns:
        Formatted pandas DataFrame
    """
    if not data:
        return pd.DataFrame()
    
    try:
        df = pd.DataFrame(data)
        
        # Handle datetime conversion for time series
        if x_col in df.columns and 'date' in x_col.lower() or 'time' in x_col.lower():
            df[x_col] = pd.to_datetime(df[x_col])
        
        # Ensure numeric columns are properly typed
        if y_col in df.columns:
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        
        # Remove rows with NaN values
        df = df.dropna(subset=[x_col, y_col])
        
        return df
        
    except Exception as e:
        logger.error(f"Error formatting chart data: {e}")
        return pd.DataFrame()

def export_chart_data(data: pd.DataFrame, filename: str) -> bytes:
    """
    Export chart data to CSV format.
    
    Args:
        data: DataFrame to export
        filename: Filename for the export
        
    Returns:
        CSV data as bytes
    """
    try:
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue().encode('utf-8')
    except Exception as e:
        logger.error(f"Error exporting chart data: {e}")
        return b"Error exporting data"

def render_dynamic_charts(api_client: WMSAPIClient, date_range: int, storage_zone: str = "All Zones") -> None:
    """
    Render dynamic charts section with consumption trends, storage utilization, and supplier performance.
    
    Args:
        api_client: API client for data fetching
        date_range: Number of days for analysis
        storage_zone: Storage zone filter
    """
    st.subheader("📈 Real-Time Analytics & Trends")
    
    # Consumption Trends Chart
    st.markdown("### 📊 Daily Consumption Trends")
    
    try:
        with st.spinner("📊 Loading consumption data..."):
            consumption_data = api_client.get_consumption_trends(days=date_range)
            
            if consumption_data and len(consumption_data) > 0:
                # Format data
                df_consumption = format_chart_data(consumption_data, 'timestamp', 'value')
                
                if not df_consumption.empty:
                    # Create line chart
                    fig_consumption = px.line(
                        df_consumption,
                        x='timestamp',
                        y='value',
                        title=f'Daily Consumption Trends - Last {date_range} Days',
                        labels={'value': 'Consumption (units)', 'timestamp': 'Date'}
                    )
                    
                    # Enhanced styling
                    fig_consumption.update_traces(
                        line=dict(width=3, color=CHART_COLORS['primary']),
                        hovertemplate='<b>Date:</b> %{x}<br><b>Consumption:</b> %{y:,.0f} units<extra></extra>',
                        fill='tonexty' if len(df_consumption) > 1 else None,
                        fillcolor='rgba(31, 119, 180, 0.1)'
                    )
                    
                    # Add range selector buttons
                    fig_consumption.update_layout(
                        xaxis=dict(
                            rangeselector=dict(
                                buttons=list([
                                    dict(count=7, label="7d", step="day", stepmode="backward"),
                                    dict(count=30, label="30d", step="day", stepmode="backward"),
                                    dict(step="all", label="All")
                                ]),
                                bgcolor='rgba(0,0,0,0)',
                                bordercolor='rgba(0,0,0,0.1)',
                                borderwidth=1
                            ),
                            rangeslider=dict(visible=True, bgcolor='rgba(0,0,0,0.05)'),
                            type="date"
                        )
                    )
                    
                    fig_consumption = apply_chart_styling(fig_consumption, height=450)
                    st.plotly_chart(fig_consumption, use_container_width=True)
                    
                    # Summary statistics
                    total_consumption = df_consumption['value'].sum()
                    avg_daily = df_consumption['value'].mean()
                    max_day = df_consumption.loc[df_consumption['value'].idxmax()]
                    min_day = df_consumption.loc[df_consumption['value'].idxmin()]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Consumption", format_number(total_consumption, " units"))
                    col2.metric("Daily Average", format_number(avg_daily, " units"))
                    col3.metric("Peak Day", f"{max_day['timestamp'].strftime('%m/%d')}: {format_number(max_day['value'])}")
                    col4.metric("Lowest Day", f"{min_day['timestamp'].strftime('%m/%d')}: {format_number(min_day['value'])}")
                    
                    # Export functionality
                    csv_data = export_chart_data(df_consumption, f"consumption_trends_{date_range}d.csv")
                    st.download_button(
                        label="📥 Download Consumption Data",
                        data=csv_data,
                        file_name=f"consumption_trends_{date_range}d_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("📊 No consumption data available for the selected period")
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
        
        try:
            with st.spinner("🏪 Loading storage data..."):
                storage_data = api_client.get_storage_utilization(zone=storage_zone if storage_zone != "All Zones" else None)
                
                if storage_data and len(storage_data) > 0:
                    df_storage = pd.DataFrame(storage_data)
                    
                    # Sort by utilization and take top 10
                    df_storage = df_storage.sort_values('utilization_percentage', ascending=True).tail(10)
                    
                    # Create color mapping based on utilization
                    def get_utilization_color(util):
                        if util >= 90:
                            return CHART_COLORS['danger']
                        elif util >= 70:
                            return CHART_COLORS['warning']
                        else:
                            return CHART_COLORS['success']
                    
                    colors = [get_utilization_color(util) for util in df_storage['utilization_percentage']]
                    
                    fig_storage = go.Figure(data=[
                        go.Bar(
                            y=df_storage['storage_bin'],
                            x=df_storage['utilization_percentage'],
                            orientation='h',
                            marker_color=colors,
                            text=[f"{util:.1f}%" for util in df_storage['utilization_percentage']],
                            textposition='inside',
                            hovertemplate='<b>%{y}</b><br>' +
                                        'Utilization: %{x:.1f}%<br>' +
                                        'Used: %{customdata[0]:.0f}<br>' +
                                        'Total: %{customdata[1]:.0f}<extra></extra>',
                            customdata=df_storage[['capacity_used', 'capacity_total']].values
                        )
                    ])
                    
                    fig_storage = apply_chart_styling(
                        fig_storage, 
                        "Top 10 Storage Locations by Utilization",
                        height=400
                    )
                    
                    fig_storage.update_layout(
                        xaxis_title="Utilization Percentage (%)",
                        yaxis_title="Storage Location",
                        margin=dict(l=100, r=20, t=60, b=40)
                    )
                    
                    fig_storage.update_xaxes(range=[0, 100])
                    
                    st.plotly_chart(fig_storage, use_container_width=True)
                    
                    # Storage summary metrics
                    avg_utilization = df_storage['utilization_percentage'].mean()
                    max_utilization = df_storage['utilization_percentage'].max()
                    over_capacity = len(df_storage[df_storage['utilization_percentage'] > 90])
                    
                    st.metric("Average Utilization", f"{avg_utilization:.1f}%")
                    if over_capacity > 0:
                        st.warning(f"⚠️ {over_capacity} locations over 90% capacity")
                    
                else:
                    st.info("🏪 No storage data available")
                    
        except Exception as e:
            logger.error(f"Error in storage utilization chart: {e}")
            st.error(f"❌ Error loading storage data: {str(e)}")
    
    # Supplier Performance Chart
    with col2:
        st.markdown("### 🤝 Supplier Performance")
        
        try:
            with st.spinner("🤝 Loading supplier data..."):
                supplier_data = api_client.get_supplier_performance(days=date_range)
                
                if supplier_data and len(supplier_data) > 0:
                    df_suppliers = pd.DataFrame(supplier_data)
                    
                    # Sort by performance and take top 10
                    df_suppliers = df_suppliers.sort_values('on_time_delivery_rate', ascending=True).tail(10)
                    
                    # Create color gradient based on performance
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
                    
                    fig_suppliers.update_traces(
                        hovertemplate='<b>%{y}</b><br>' +
                                    'On-Time Rate: %{x:.1f}%<br>' +
                                    'Lead Time: %{customdata[0]:.1f} days<br>' +
                                    'Total Orders: %{customdata[1]}<extra></extra>',
                        customdata=df_suppliers[['average_lead_time', 'total_orders_last_30_days']].values
                    )
                    
                    fig_suppliers = apply_chart_styling(
                        fig_suppliers,
                        height=400
                    )
                    
                    fig_suppliers.update_layout(
                        margin=dict(l=120, r=20, t=60, b=40),
                        coloraxis_showscale=False
                    )
                    
                    fig_suppliers.update_xaxes(range=[0, 100])
                    
                    st.plotly_chart(fig_suppliers, use_container_width=True)
                    
                    # Supplier summary metrics
                    avg_performance = df_suppliers['on_time_delivery_rate'].mean()
                    best_supplier = df_suppliers.loc[df_suppliers['on_time_delivery_rate'].idxmax()]
                    
                    st.metric("Average Performance", f"{avg_performance:.1f}%")
                    st.success(f"🏆 Best: {best_supplier['supplier_name']} ({best_supplier['on_time_delivery_rate']:.1f}%)")
                    
                else:
                    st.info("🤝 No supplier data available")
                    
        except Exception as e:
            logger.error(f"Error in supplier performance chart: {e}")
            st.error(f"❌ Error loading supplier data: {str(e)}")

def render_fifo_efficiency(api_client: WMSAPIClient, product_codes: Optional[List[str]] = None) -> None:
    """
    Render FIFO efficiency dashboard with gauge charts and detailed metrics.
    
    Args:
        api_client: API client for data fetching
        product_codes: Optional list of product codes to filter
    """
    st.subheader("🔄 FIFO Efficiency Dashboard")
    
    try:
        with st.spinner("🔄 Loading FIFO efficiency data..."):
            fifo_data = api_client.get_fifo_efficiency(product_codes=product_codes)
            
            if fifo_data and len(fifo_data) > 0:
                # Top 4 products with gauge charts
                st.markdown("### 📊 Top Product FIFO Performance")
                
                top_products = fifo_data[:4]
                cols = st.columns(len(top_products))
                
                for i, product in enumerate(top_products):
                    with cols[i]:
                        efficiency = safe_get(product, 'fifo_compliance_rate', 0)
                        product_code = safe_get(product, 'product_code', f'Product {i+1}')
                        
                        # Create gauge chart using helper function
                        fig_gauge = create_gauge_chart(
                            value=efficiency,
                            title=f"FIFO Efficiency<br><b>{product_code}</b>",
                            max_value=100,
                            thresholds={'good': 90, 'warning': 70, 'critical': 50},
                            unit="%"
                        )
                        
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        # Efficiency status with color coding
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
                
                # Interactive filters
                col1, col2 = st.columns(2)
                
                with col1:
                    efficiency_filter = st.selectbox(
                        "Filter by Efficiency",
                        options=["All", "Excellent (90%+)", "Good (70-89%)", "Needs Improvement (<70%)"],
                        key="fifo_efficiency_filter"
                    )
                
                with col2:
                    sort_by = st.selectbox(
                        "Sort by",
                        options=["FIFO Compliance Rate", "Product Code", "Average Batch Age"],
                        key="fifo_sort"
                    )
                
                # Apply filters
                filtered_df = df_fifo.copy()
                
                if efficiency_filter == "Excellent (90%+)":
                    filtered_df = filtered_df[filtered_df['fifo_compliance_rate'] >= 90]
                elif efficiency_filter == "Good (70-89%)":
                    filtered_df = filtered_df[
                        (filtered_df['fifo_compliance_rate'] >= 70) & 
                        (filtered_df['fifo_compliance_rate'] < 90)
                    ]
                elif efficiency_filter == "Needs Improvement (<70%)":
                    filtered_df = filtered_df[filtered_df['fifo_compliance_rate'] < 70]
                
                # Apply sorting
                sort_column_map = {
                    "FIFO Compliance Rate": "fifo_compliance_rate",
                    "Product Code": "product_code",
                    "Average Batch Age": "average_batch_age_consumed"
                }
                
                sort_column = sort_column_map[sort_by]
                filtered_df = filtered_df.sort_values(sort_column, ascending=False)
                
                # Style the dataframe
                def style_efficiency_row(row):
                    efficiency = row['fifo_compliance_rate']
                    if efficiency >= 90:
                        return [f'background-color: {CHART_COLORS["success"]}; color: white'] * len(row)
                    elif efficiency >= 70:
                        return [f'background-color: {CHART_COLORS["warning"]}; color: white'] * len(row)
                    else:
                        return [f'background-color: {CHART_COLORS["danger"]}; color: white'] * len(row)
                
                # Format columns for display
                display_columns = ['product_code', 'fifo_compliance_rate', 'average_batch_age_consumed', 'oldest_batch_age', 'efficiency_status']
                display_df = filtered_df[display_columns].copy()
                
                # Rename columns for better display
                display_df.columns = ['Product Code', 'FIFO Compliance (%)', 'Avg Batch Age (days)', 'Oldest Batch (days)', 'Status']
                
                styled_df = display_df.style.apply(style_efficiency_row, axis=1).format({
                    'FIFO Compliance (%)': '{:.1f}%',
                    'Avg Batch Age (days)': '{:.1f}',
                    'Oldest Batch (days)': '{:.0f}'
                })
                
                st.dataframe(styled_df, use_container_width=True, height=400)
                
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
                csv_data = export_chart_data(df_fifo, "fifo_efficiency_report.csv")
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

def render_alerts_table(api_client: WMSAPIClient, priority_filter: List[str] = None, days_ahead: int = 30) -> None:
    """
    Render detailed alerts table with filtering and export functionality.
    
    Args:
        api_client: API client for data fetching
        priority_filter: List of priority levels to include
        days_ahead: Number of days ahead to check for alerts
    """
    st.subheader("🚨 Alert Management Center")
    
    try:
        with st.spinner("🚨 Loading alert data..."):
            alerts = api_client.get_expiry_alerts(days_ahead=days_ahead, priority=priority_filter)
            
            if alerts and len(alerts) > 0:
                df_alerts = pd.DataFrame(alerts)
                
                # Interactive filters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    priority_options = ["All"] + list(df_alerts['priority'].unique())
                    selected_priority = st.selectbox(
                        "🎯 Filter by Priority",
                        options=priority_options,
                        key="alerts_priority_filter"
                    )
                
                with col2:
                    days_filter = st.slider(
                        "📅 Days Until Expiry",
                        min_value=0,
                        max_value=int(df_alerts['days_until_expiry'].max()) if not df_alerts.empty else 30,
                        value=(0, days_ahead),
                        key="alerts_days_filter"
                    )
                
                with col3:
                    sort_options = ["Days Until Expiry", "Priority", "Quantity", "Product Code"]
                    sort_by = st.selectbox(
                        "📊 Sort By",
                        options=sort_options,
                        key="alerts_sort"
                    )
                
                # Apply filters
                filtered_df = df_alerts.copy()
                
                if selected_priority != "All":
                    filtered_df = filtered_df[filtered_df['priority'] == selected_priority]
                
                filtered_df = filtered_df[
                    (filtered_df['days_until_expiry'] >= days_filter[0]) &
                    (filtered_df['days_until_expiry'] <= days_filter[1])
                ]
                
                # Apply sorting
                sort_column_map = {
                    "Days Until Expiry": "days_until_expiry",
                    "Priority": "priority",
                    "Quantity": "quantity",
                    "Product Code": "product_code"
                }
                
                sort_column = sort_column_map[sort_by]
                filtered_df = filtered_df.sort_values(sort_column)
                
                if not filtered_df.empty:
                    # Style the alerts table
                    def style_priority_row(row):
                        priority = row['priority']
                        color = get_priority_color(priority)
                        return [f'background-color: {color}; color: white'] * len(row)
                    
                    # Format the dataframe for display
                    display_df = filtered_df.copy()
                    display_df['expiry_date'] = pd.to_datetime(display_df['expiry_date']).dt.strftime('%Y-%m-%d')
                    
                    # Select and rename columns for display
                    display_columns = ['product_code', 'batch_number', 'quantity', 'expiry_date', 'days_until_expiry', 'priority']
                    display_df = display_df[display_columns]
                    display_df.columns = ['Product Code', 'Batch Number', 'Quantity', 'Expiry Date', 'Days Until Expiry', 'Priority']
                    
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
                    csv_data = export_chart_data(filtered_df, "alerts_report.csv")
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

def render_revenue_metrics(api_client: WMSAPIClient) -> None:
    """
    Render financial and revenue metrics with realistic mock data.
    
    Args:
        api_client: API client for data fetching
    """
    st.subheader("💰 Financial Performance Metrics")
    
    try:
        # Get inventory data to base financial calculations on
        inventory_summary = api_client.get_inventory_summary()
        total_inventory = safe_get(inventory_summary, 'total_quantity', 0) if inventory_summary else 0
        
        # Generate realistic financial metrics based on inventory
        base_revenue = 125000
        daily_revenue = base_revenue + (total_inventory * 0.5)  # Revenue scales with inventory
        cost_per_order = 12.50 + (total_inventory * 0.0001)  # Slight increase with scale
        inventory_value = total_inventory * 15.75  # Assume $15.75 per unit average
        roi_percentage = 18.5 + (total_inventory * 0.00005)  # ROI improves with scale
        
        # Add realistic daily variation
        random.seed(datetime.now().day)  # Consistent daily variation
        daily_revenue *= (1 + random.uniform(-0.08, 0.12))
        cost_per_order *= (1 + random.uniform(-0.05, 0.05))
        roi_percentage *= (1 + random.uniform(-0.1, 0.1))
        
        # Display financial KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            revenue_delta = "+8.2%" if daily_revenue > base_revenue else "-2.1%"
            st.metric(
                label="💵 Daily Revenue",
                value=f"${format_number(daily_revenue)}",
                delta=revenue_delta,
                help="Total revenue generated today from warehouse operations"
            )
        
        with col2:
            cost_delta = "-2.1%" if cost_per_order < 13 else "+1.5%"
            cost_color = "inverse" if cost_per_order < 13 else "normal"
            st.metric(
                label="💸 Cost per Order",
                value=f"${cost_per_order:.2f}",
                delta=cost_delta,
                delta_color=cost_color,
                help="Average cost to process and fulfill each order"
            )
        
        with col3:
            value_delta = "+5.7%" if inventory_value > 1000000 else "+2.3%"
            st.metric(
                label="📊 Total Inventory Value",
                value=f"${format_number(inventory_value)}",
                delta=value_delta,
                help="Total monetary value of current inventory holdings"
            )
        
        with col4:
            roi_delta = "+3.2%" if roi_percentage > 18 else "+1.1%"
            st.metric(
                label="📈 ROI",
                value=f"{roi_percentage:.1f}%",
                delta=roi_delta,
                help="Return on investment percentage for warehouse operations"
            )
        
        # Revenue trend chart (last 30 days)
        st.markdown("### 📈 Revenue Trend Analysis")
        
        # Generate mock trend data with realistic patterns
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        base_daily_revenue = daily_revenue * 0.9
        
        trend_data = []
        for i, date in enumerate(dates):
            # Add realistic patterns: weekday/weekend variation, growth trend
            weekday_factor = 0.8 if date.weekday() >= 5 else 1.0  # Lower on weekends
            trend_factor = 1 + (i * 0.001)  # Slight upward trend
            daily_variation = 1 + random.uniform(-0.15, 0.15)
            
            revenue = base_daily_revenue * weekday_factor * trend_factor * daily_variation
            trend_data.append({
                'date': date,
                'revenue': revenue,
                'weekday': date.strftime('%A')
            })
        
        df_revenue = pd.DataFrame(trend_data)
        
        # Create revenue trend chart
        fig_revenue = px.line(
            df_revenue,
            x='date',
            y='revenue',
            title='Daily Revenue Trend - Last 30 Days',
            labels={'revenue': 'Revenue ($)', 'date': 'Date'},
            hover_data=['weekday']
        )
        
        fig_revenue.update_traces(
            line=dict(width=3, color=CHART_COLORS['success']),
            hovertemplate='<b>Date:</b> %{x}<br><b>Revenue:</b> $%{y:,.0f}<br><b>Day:</b> %{customdata[0]}<extra></extra>',
            fill='tonexty',
            fillcolor='rgba(44, 160, 44, 0.1)'
        )
        
        fig_revenue = apply_chart_styling(fig_revenue, height=350)
        st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Financial summary metrics
        col1, col2, col3 = st.columns(3)
        
        avg_revenue = df_revenue['revenue'].mean()
        revenue_growth = ((df_revenue['revenue'].iloc[-1] - df_revenue['revenue'].iloc[0]) / df_revenue['revenue'].iloc[0]) * 100
        total_month_revenue = df_revenue['revenue'].sum()
        
        col1.metric("30-Day Average Revenue", f"${format_number(avg_revenue)}")
        col2.metric("Revenue Growth", f"{revenue_growth:+.1f}%")
        col3.metric("Monthly Total", f"${format_number(total_month_revenue)}")
        
        # Export revenue data
        csv_data = export_chart_data(df_revenue, "revenue_trends.csv")
        st.download_button(
            label="📥 Download Revenue Data",
            data=csv_data,
            file_name=f"revenue_trends_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        logger.error(f"Error in revenue metrics: {e}")
        st.error(f"❌ Error loading financial data: {str(e)}")

