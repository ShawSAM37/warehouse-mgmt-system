"""
WMS ML Dashboard Integration
===========================

Production-grade Streamlit dashboard for ML forecasting and anomaly detection.
Includes real-time monitoring, interactive visualizations, and model management.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio
import json

# Configure page
st.set_page_config(
    page_title="WMS ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-critical { border-left-color: #dc3545; }
    .alert-high { border-left-color: #fd7e14; }
    .alert-medium { border-left-color: #ffc107; }
    .alert-low { border-left-color: #28a745; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

class MLDashboardAPI:
    """API client for ML dashboard integration."""
    
    def __init__(self):
        self.base_url = st.secrets.get("API_BASE_URL", "http://localhost:8000")
        self.token = st.session_state.get("access_token")
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
    
    def get_forecast(self, product_code: str, horizon: int = 30, model_type: str = None) -> Dict:
        """Get forecast for a specific product."""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/forecast/product/{product_code}",
                headers=self.headers,
                json={"horizon": horizon, "model_type": model_type},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to get forecast: {str(e)}")
            return {}
    
    def get_anomalies(self, product_codes: List[str] = None) -> Dict:
        """Get anomaly detection results."""
        try:
            payload = {"product_codes": product_codes} if product_codes else {}
            response = requests.post(
                f"{self.base_url}/api/v1/anomaly/detect",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to get anomalies: {str(e)}")
            return {}
    
    def get_model_health(self) -> Dict:
        """Get model health status."""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/models/health",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to get model health: {str(e)}")
            return {}
    
    def get_active_alerts(self) -> Dict:
        """Get active anomaly alerts."""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/anomaly/alerts/active",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to get alerts: {str(e)}")
            return {}

def render_forecasting_dashboard():
    """Render the forecasting dashboard with interactive controls."""
    st.header("📈 Demand Forecasting Dashboard")
    
    api = MLDashboardAPI()
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("Forecast Configuration")
        
        # Product selection
        product_code = st.selectbox(
            "Select Product",
            options=["PROD-001", "PROD-002", "PROD-003", "PROD-004"],
            key="forecast_product"
        )
        
        # Forecast horizon
        horizon = st.slider(
            "Forecast Horizon (days)",
            min_value=7,
            max_value=90,
            value=30,
            step=7,
            key="forecast_horizon"
        )
        
        # Model selection
        model_type = st.selectbox(
            "Model Type",
            options=["Auto-Select", "Prophet", "XGBoost", "Linear"],
            key="forecast_model"
        )
        
        # Generate forecast button
        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Generating forecast..."):
                forecast_data = api.get_forecast(
                    product_code, 
                    horizon, 
                    None if model_type == "Auto-Select" else model_type.lower()
                )
                st.session_state.forecast_data = forecast_data
    
    # Main dashboard content
    if 'forecast_data' in st.session_state and st.session_state.forecast_data:
        forecast_data = st.session_state.forecast_data
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Model Used",
                forecast_data.get('model_used', 'N/A').title(),
                help="Best performing model selected"
            )
        
        with col2:
            rmse = forecast_data.get('model_metrics', {}).get('rmse', 0)
            st.metric(
                "RMSE",
                f"{rmse:.2f}",
                help="Root Mean Square Error"
            )
        
        with col3:
            mae = forecast_data.get('model_metrics', {}).get('mae', 0)
            st.metric(
                "MAE",
                f"{mae:.2f}",
                help="Mean Absolute Error"
            )
        
        with col4:
            mape = forecast_data.get('model_metrics', {}).get('mape', 0)
            st.metric(
                "MAPE",
                f"{mape:.1f}%",
                help="Mean Absolute Percentage Error"
            )
        
        # Forecast visualization
        st.subheader("Demand Forecast Visualization")
        
        if 'forecast' in forecast_data and forecast_data['forecast']:
            # Convert forecast data to DataFrame
            forecast_df = pd.DataFrame(forecast_data['forecast'])
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
            
            # Create interactive forecast chart
            fig = go.Figure()
            
            # Add forecast line
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['forecast'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=6)
            ))
            
            # Add confidence intervals if available
            if 'confidence_intervals' in forecast_data and forecast_data['confidence_intervals']:
                ci_df = pd.DataFrame(forecast_data['confidence_intervals'])
                ci_df['date'] = pd.to_datetime(ci_df['date'])
                
                fig.add_trace(go.Scatter(
                    x=ci_df['date'],
                    y=ci_df['upper'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=ci_df['date'],
                    y=ci_df['lower'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    name='Confidence Interval',
                    fillcolor='rgba(31,119,180,0.2)'
                ))
            
            fig.update_layout(
                title=f"Demand Forecast for {product_code}",
                xaxis_title="Date",
                yaxis_title="Quantity",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast summary table
            st.subheader("Forecast Summary")
            summary_df = forecast_df.copy()
            summary_df['date'] = summary_df['date'].dt.strftime('%Y-%m-%d')
            summary_df['forecast'] = summary_df['forecast'].round(2)
            
            st.dataframe(
                summary_df[['date', 'forecast']].head(14),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("Select a product and click 'Generate Forecast' to view predictions.")

def render_anomaly_dashboard():
    """Render the anomaly monitoring dashboard."""
    st.header("🚨 Anomaly Detection Dashboard")
    
    api = MLDashboardAPI()
    
    # Auto-refresh toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Real-time Anomaly Monitoring")
    with col2:
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
    
    # Get anomaly data
    if auto_refresh and 'last_anomaly_refresh' not in st.session_state:
        st.session_state.last_anomaly_refresh = time.time()
    
    if (auto_refresh and 
        time.time() - st.session_state.get('last_anomaly_refresh', 0) > 30) or \
       'anomaly_data' not in st.session_state:
        
        with st.spinner("Loading anomaly data..."):
            anomaly_data = api.get_anomalies()
            st.session_state.anomaly_data = anomaly_data
            st.session_state.last_anomaly_refresh = time.time()
    
    anomaly_data = st.session_state.get('anomaly_data', {})
    
    if anomaly_data and 'anomalies' in anomaly_data:
        anomalies = anomaly_data['anomalies']
        summary = anomaly_data.get('summary', {})
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Anomalies",
                summary.get('anomalies_detected', 0),
                help="Total anomalies detected"
            )
        
        with col2:
            critical_count = summary.get('severity_distribution', {}).get('critical', 0)
            st.metric(
                "Critical Alerts",
                critical_count,
                delta=f"+{critical_count}" if critical_count > 0 else None,
                delta_color="inverse"
            )
        
        with col3:
            high_count = summary.get('severity_distribution', {}).get('high', 0)
            st.metric(
                "High Priority",
                high_count,
                help="High priority anomalies"
            )
        
        with col4:
            detection_rate = summary.get('detection_rate', 0) * 100
            st.metric(
                "Detection Rate",
                f"{detection_rate:.1f}%",
                help="Percentage of records flagged as anomalous"
            )
        
        # Severity distribution chart
        if summary.get('severity_distribution'):
            st.subheader("Anomaly Severity Distribution")
            
            severity_data = summary['severity_distribution']
            severity_df = pd.DataFrame([
                {'Severity': k.title(), 'Count': v} 
                for k, v in severity_data.items() if v > 0
            ])
            
            if not severity_df.empty:
                fig = px.pie(
                    severity_df,
                    values='Count',
                    names='Severity',
                    color='Severity',
                    color_discrete_map={
                        'Critical': '#dc3545',
                        'High': '#fd7e14',
                        'Medium': '#ffc107',
                        'Low': '#28a745'
                    }
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly alerts feed
        st.subheader("Recent Anomaly Alerts")
        
        # Severity filter
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=['critical', 'high', 'medium', 'low'],
            default=['critical', 'high'],
            key="severity_filter"
        )
        
        # Filter anomalies
        filtered_anomalies = [
            anomaly for anomaly in anomalies
            if anomaly.get('severity') in severity_filter
        ]
        
        # Display anomalies
        for i, anomaly in enumerate(filtered_anomalies[:10]):
            severity = anomaly.get('severity', 'unknown')
            severity_class = f"alert-{severity}"
            
            with st.container():
                st.markdown(f"""
                <div class="metric-card {severity_class}">
                    <h4>{anomaly.get('product_code', 'Unknown Product')} - {severity.title()} Anomaly</h4>
                    <p><strong>Score:</strong> {anomaly.get('anomaly_score', 0):.2f}</p>
                    <p><strong>Time:</strong> {anomaly.get('timestamp', 'Unknown')}</p>
                    <p><strong>Quantity:</strong> {anomaly.get('data_point', {}).get('total_quantity', 0):.1f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Expandable explanation
                with st.expander("View Explanation"):
                    explanation = anomaly.get('explanation', {})
                    
                    if explanation.get('contributing_factors'):
                        st.write("**Contributing Factors:**")
                        for factor in explanation['contributing_factors']:
                            st.write(f"• {factor}")
                    
                    if explanation.get('recommendations'):
                        st.write("**Recommendations:**")
                        for rec in explanation['recommendations']:
                            st.write(f"• {rec}")
    else:
        st.info("No anomaly data available. Check your API connection.")
    
    # Auto-refresh mechanism
    if auto_refresh:
        time.sleep(1)
        st.rerun()

def render_model_management():
    """Render the model management interface."""
    st.header("⚙️ Model Management Dashboard")
    
    api = MLDashboardAPI()
    
    # Get model health data
    model_health = api.get_model_health()
    
    if model_health:
        # System status overview
        st.subheader("System Health Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_status = model_health.get('forecast_models', {}).get('status', 'unknown')
            status_color = "🟢" if forecast_status == "healthy" else "🔴"
            st.metric(
                "Forecasting Models",
                f"{status_color} {forecast_status.title()}",
                help="Status of forecasting models"
            )
        
        with col2:
            anomaly_status = model_health.get('anomaly_models', {}).get('status', 'unknown')
            status_color = "🟢" if anomaly_status == "healthy" else "🔴"
            st.metric(
                "Anomaly Detection",
                f"{status_color} {anomaly_status.title()}",
                help="Status of anomaly detection models"
            )
        
        with col3:
            system_status = model_health.get('system_status', 'unknown')
            status_color = "🟢" if system_status == "healthy" else "🔴"
            st.metric(
                "Overall System",
                f"{status_color} {system_status.title()}",
                help="Overall system health"
            )
        
        # Model details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Forecasting Models")
            forecast_models = model_health.get('forecast_models', {})
            
            if forecast_models.get('available'):
                st.success(f"Available Models: {', '.join(forecast_models['available'])}")
            
            dependencies = forecast_models.get('dependencies', {})
            for dep, status in dependencies.items():
                status_icon = "✅" if status else "❌"
                st.write(f"{status_icon} {dep.title()}: {'Available' if status else 'Not Available'}")
        
        with col2:
            st.subheader("Anomaly Detection")
            anomaly_models = model_health.get('anomaly_models', {})
            
            if anomaly_models.get('available_methods'):
                st.success(f"Available Methods: {len(anomaly_models['available_methods'])}")
                for method in anomaly_models['available_methods']:
                    st.write(f"• {method.replace('_', ' ').title()}")
        
        # Model retraining controls
        st.subheader("Model Retraining")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Retrain Forecasting Models", type="secondary"):
                with st.spinner("Initiating forecast model retraining..."):
                    # This would call the retrain API
                    st.success("Forecast model retraining initiated!")
        
        with col2:
            if st.button("Retrain Anomaly Models", type="secondary"):
                with st.spinner("Initiating anomaly model retraining..."):
                    # This would call the retrain API
                    st.success("Anomaly model retraining initiated!")
        
        # Performance metrics (mock data for demonstration)
        st.subheader("Model Performance Trends")
        
        # Generate sample performance data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
        performance_data = pd.DataFrame({
            'Date': dates,
            'Forecast_RMSE': 5 + np.random.normal(0, 0.5, len(dates)),
            'Anomaly_Accuracy': 0.95 + np.random.normal(0, 0.02, len(dates))
        })
        
        # Performance charts
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Forecast Model RMSE', 'Anomaly Detection Accuracy'),
            vertical_spacing=0.1
        )
        
        # Forecast RMSE trend
        fig.add_trace(
            go.Scatter(
                x=performance_data['Date'],
                y=performance_data['Forecast_RMSE'],
                mode='lines+markers',
                name='RMSE',
                line=dict(color='#1f77b4')
            ),
            row=1, col=1
        )
        
        # Anomaly accuracy trend
        fig.add_trace(
            go.Scatter(
                x=performance_data['Date'],
                y=performance_data['Anomaly_Accuracy'],
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='#2ca02c')
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=False)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="RMSE", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("Unable to retrieve model health information.")

def main():
    """Main dashboard application."""
    st.title("🤖 WMS ML Dashboard")
    
    # Check authentication
    if 'authenticated' not in st.session_state or not st.session_state.authenticated:
        st.error("Please log in to access the ML dashboard.")
        st.stop()
    
    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["📈 Forecasting", "🚨 Anomaly Detection", "⚙️ Model Management"])
    
    with tab1:
        render_forecasting_dashboard()
    
    with tab2:
        render_anomaly_dashboard()
    
    with tab3:
        render_model_management()
    
    # Footer
    st.markdown("---")
    st.markdown("*WMS ML Dashboard - Real-time Machine Learning Monitoring*")

if __name__ == "__main__":
    main()
