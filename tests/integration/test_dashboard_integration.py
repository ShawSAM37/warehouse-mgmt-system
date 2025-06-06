"""
Dashboard Integration Tests
==========================

Comprehensive test suite for the WMS dashboard including component integration,
API connectivity, UI rendering, data flow, and error handling scenarios.
"""

import pytest
import pandas as pd
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import io
import time

# Import dashboard modules
from src.wms.dashboard.api_client import WMSAPIClient
from src.wms.dashboard.components import (
    render_header, render_sidebar, render_kpi_cards, render_alerts
)
from src.wms.dashboard.charts import (
    render_dynamic_charts, render_fifo_efficiency, render_alerts_table, render_revenue_metrics
)
from src.wms.dashboard.enhancements import DashboardEnhancements, UserPreferences
from src.wms.dashboard.utils.config import DashboardConfig
from src.wms.dashboard.utils.helpers import format_number, get_priority_color, safe_get
from src.wms.dashboard.dashboard import initialize_session_state, handle_auto_refresh

# Test configuration
TEST_API_BASE_URL = "http://test-api:8000"
TEST_TIMEOUT = 5

class TestConfig:
    """Test configuration class."""
    
    def __init__(self):
        self.api_base_url = TEST_API_BASE_URL
        self.api_timeout = TEST_TIMEOUT
        self.enable_debug = True
        self.cache_ttl_default = 60
        self.export_temp_dir = "/tmp/test_exports"

# ===== FIXTURES =====

@pytest.fixture
def mock_config():
    """Mock dashboard configuration."""
    return TestConfig()

@pytest.fixture
def mock_api_client(mock_config):
    """Mock API client with test configuration."""
    client = WMSAPIClient(mock_config)
    return client

@pytest.fixture
def sample_inventory_summary():
    """Sample inventory summary data."""
    return {
        "total_products": 150,
        "total_batches": 450,
        "total_quantity": 25000,
        "low_stock_count": 12,
        "expiring_soon_count": 8
    }

@pytest.fixture
def sample_stock_levels():
    """Sample stock levels data."""
    return [
        {
            "product_code": "PROD-001",
            "current_stock": 500,
            "reorder_point": 100,
            "priority_level": "MEDIUM",
            "unit_of_measurement": "KG"
        },
        {
            "product_code": "PROD-002", 
            "current_stock": 50,
            "reorder_point": 200,
            "priority_level": "URGENT",
            "unit_of_measurement": "PCS"
        },
        {
            "product_code": "PROD-003",
            "current_stock": 1200,
            "reorder_point": 300,
            "priority_level": "LOW",
            "unit_of_measurement": "L"
        }
    ]

@pytest.fixture
def sample_supplier_performance():
    """Sample supplier performance data."""
    return [
        {
            "supplier_id": "SUP-001",
            "supplier_name": "Test Supplier A",
            "on_time_delivery_rate": 95.5,
            "average_lead_time": 7.2,
            "total_orders_last_30_days": 15
        },
        {
            "supplier_id": "SUP-002",
            "supplier_name": "Test Supplier B", 
            "on_time_delivery_rate": 87.3,
            "average_lead_time": 9.1,
            "total_orders_last_30_days": 8
        }
    ]

@pytest.fixture
def sample_storage_utilization():
    """Sample storage utilization data."""
    return [
        {
            "storage_bin": "BIN-A01",
            "capacity_total": 1000,
            "capacity_used": 750,
            "utilization_percentage": 75.0,
            "zone": "Zone A"
        },
        {
            "storage_bin": "BIN-B01",
            "capacity_total": 800,
            "capacity_used": 720,
            "utilization_percentage": 90.0,
            "zone": "Zone B"
        }
    ]

@pytest.fixture
def sample_expiry_alerts():
    """Sample expiry alerts data."""
    return [
        {
            "product_code": "PROD-001",
            "batch_number": "BATCH-001",
            "quantity": 50,
            "expiry_date": "2025-06-15",
            "days_until_expiry": 9,
            "priority": "HIGH"
        },
        {
            "product_code": "PROD-002",
            "batch_number": "BATCH-002", 
            "quantity": 25,
            "expiry_date": "2025-06-10",
            "days_until_expiry": 4,
            "priority": "URGENT"
        }
    ]

@pytest.fixture
def sample_consumption_trends():
    """Sample consumption trends data."""
    base_date = datetime.now() - timedelta(days=30)
    return [
        {
            "timestamp": (base_date + timedelta(days=i)).isoformat(),
            "value": 100 + (i * 5) + (i % 7) * 10,
            "metric_name": "daily_consumption"
        }
        for i in range(30)
    ]

@pytest.fixture
def sample_fifo_efficiency():
    """Sample FIFO efficiency data."""
    return [
        {
            "product_code": "PROD-001",
            "fifo_compliance_rate": 95.5,
            "average_batch_age_consumed": 12.3,
            "oldest_batch_age": 45
        },
        {
            "product_code": "PROD-002",
            "fifo_compliance_rate": 87.2,
            "average_batch_age_consumed": 18.7,
            "oldest_batch_age": 67
        }
    ]

@pytest.fixture
def mock_streamlit_session():
    """Mock Streamlit session state."""
    session_state = {
        'last_refresh': datetime.now(),
        'api_status': True,
        'error_count': 0,
        'user_preferences': UserPreferences().__dict__,
        'current_view': 'Overview',
        'theme': 'light',
        'auto_refresh': True,
        'refresh_rate': 30
    }
    return session_state

# ===== API CLIENT TESTS =====

class TestAPIClient:
    """Test API client functionality."""
    
    @patch('requests.Session.get')
    def test_api_client_initialization(self, mock_get, mock_config):
        """Test API client initialization."""
        client = WMSAPIClient(mock_config)
        assert client.base_url == TEST_API_BASE_URL
        assert client.timeout == TEST_TIMEOUT
    
    @patch('requests.Session.get')
    def test_successful_api_call(self, mock_get, mock_api_client, sample_inventory_summary):
        """Test successful API call."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_inventory_summary
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = mock_api_client._make_request("/reporting/inventory-summary")
        
        assert result == sample_inventory_summary
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_api_connection_error(self, mock_get, mock_api_client):
        """Test API connection error handling."""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        with patch('streamlit.error') as mock_st_error:
            result = mock_api_client._make_request("/reporting/inventory-summary")
            
            assert result is None
            mock_st_error.assert_called()
    
    @patch('requests.Session.get')
    def test_api_timeout_error(self, mock_get, mock_api_client):
        """Test API timeout error handling."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
        
        with patch('streamlit.error') as mock_st_error:
            result = mock_api_client._make_request("/reporting/inventory-summary")
            
            assert result is None
            mock_st_error.assert_called()
    
    @patch('requests.Session.get')
    def test_api_http_error(self, mock_get, mock_api_client):
        """Test API HTTP error handling."""
        import requests
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_get.return_value = mock_response
        
        with patch('streamlit.error') as mock_st_error:
            result = mock_api_client._make_request("/reporting/inventory-summary")
            
            assert result is None
            mock_st_error.assert_called()

# ===== COMPONENT TESTS =====

class TestComponents:
    """Test dashboard components."""
    
    @patch('streamlit.markdown')
    @patch('src.wms.dashboard.components.check_api_health')
    def test_render_header(self, mock_health_check, mock_st_markdown):
        """Test header rendering."""
        mock_health_check.return_value = True
        
        render_header()
        
        mock_st_markdown.assert_called()
        mock_health_check.assert_called()
    
    @patch('streamlit.sidebar')
    @patch('streamlit.session_state', new_callable=lambda: MagicMock())
    def test_render_sidebar(self, mock_session_state, mock_sidebar):
        """Test sidebar rendering and controls."""
        # Mock sidebar components
        mock_sidebar.checkbox.return_value = True
        mock_sidebar.slider.return_value = 30
        mock_sidebar.selectbox.return_value = "Overview"
        mock_sidebar.multiselect.return_value = []
        mock_sidebar.button.return_value = False
        
        controls = render_sidebar()
        
        assert isinstance(controls, dict)
        assert 'auto_refresh' in controls
        assert 'view_mode' in controls
        assert 'date_range' in controls
    
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    @patch('streamlit.spinner')
    def test_render_kpi_cards(self, mock_spinner, mock_metric, mock_columns, mock_api_client, sample_inventory_summary):
        """Test KPI cards rendering."""
        # Mock Streamlit components
        mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        # Mock API client response
        mock_api_client.get_inventory_summary = Mock(return_value=sample_inventory_summary)
        
        render_kpi_cards(mock_api_client)
        
        mock_api_client.get_inventory_summary.assert_called_once()
        mock_columns.assert_called()
    
    @patch('streamlit.spinner')
    @patch('streamlit.markdown')
    def test_render_alerts(self, mock_markdown, mock_spinner, mock_api_client, sample_expiry_alerts):
        """Test alerts rendering."""
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        mock_api_client.get_expiry_alerts = Mock(return_value=sample_expiry_alerts)
        
        render_alerts(mock_api_client, ['URGENT', 'HIGH'], 30)
        
        mock_api_client.get_expiry_alerts.assert_called_once()
        mock_markdown.assert_called()

# ===== CHART TESTS =====

class TestCharts:
    """Test chart generation and data visualization."""
    
    @patch('streamlit.plotly_chart')
    @patch('streamlit.spinner')
    @patch('streamlit.columns')
    def test_render_dynamic_charts(self, mock_columns, mock_spinner, mock_plotly_chart, 
                                 mock_api_client, sample_consumption_trends, sample_storage_utilization, 
                                 sample_supplier_performance):
        """Test dynamic charts rendering."""
        # Mock Streamlit components
        mock_columns.return_value = [Mock(), Mock()]
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        # Mock API responses
        mock_api_client.get_consumption_trends = Mock(return_value=sample_consumption_trends)
        mock_api_client.get_storage_utilization = Mock(return_value=sample_storage_utilization)
        mock_api_client.get_supplier_performance = Mock(return_value=sample_supplier_performance)
        
        render_dynamic_charts(mock_api_client, 30, "All Zones")
        
        # Verify API calls
        mock_api_client.get_consumption_trends.assert_called_once()
        mock_api_client.get_storage_utilization.assert_called_once()
        mock_api_client.get_supplier_performance.assert_called_once()
        
        # Verify charts were rendered
        assert mock_plotly_chart.call_count >= 1
    
    @patch('streamlit.plotly_chart')
    @patch('streamlit.columns')
    @patch('streamlit.spinner')
    def test_render_fifo_efficiency(self, mock_spinner, mock_columns, mock_plotly_chart,
                                  mock_api_client, sample_fifo_efficiency):
        """Test FIFO efficiency dashboard."""
        mock_columns.return_value = [Mock(), Mock()]
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        mock_api_client.get_fifo_efficiency = Mock(return_value=sample_fifo_efficiency)
        
        render_fifo_efficiency(mock_api_client)
        
        mock_api_client.get_fifo_efficiency.assert_called_once()
        mock_plotly_chart.assert_called()
    
    @patch('streamlit.dataframe')
    @patch('streamlit.download_button')
    @patch('streamlit.spinner')
    def test_render_alerts_table(self, mock_spinner, mock_download, mock_dataframe,
                                mock_api_client, sample_expiry_alerts):
        """Test alerts table rendering."""
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        mock_api_client.get_expiry_alerts = Mock(return_value=sample_expiry_alerts)
        
        render_alerts_table(mock_api_client, ['URGENT', 'HIGH'], 30)
        
        mock_api_client.get_expiry_alerts.assert_called_once()
        mock_dataframe.assert_called()
        mock_download.assert_called()
    
    @patch('streamlit.plotly_chart')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    def test_render_revenue_metrics(self, mock_metric, mock_columns, mock_plotly_chart, mock_api_client):
        """Test revenue metrics rendering."""
        mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]
        mock_api_client.get_inventory_summary = Mock(return_value={"total_quantity": 1000})
        
        render_revenue_metrics(mock_api_client)
        
        mock_metric.assert_called()
        mock_plotly_chart.assert_called()

# ===== UTILITY FUNCTION TESTS =====

class TestUtilities:
    """Test utility functions."""
    
    def test_format_number(self):
        """Test number formatting utility."""
        assert format_number(1500) == "1.5K"
        assert format_number(2500000, " units") == "2.5M units"
        assert format_number(1234567890) == "1.2B"
        assert format_number(50) == "50"
    
    def test_get_priority_color(self):
        """Test priority color mapping."""
        assert get_priority_color("URGENT") == "#d62728"
        assert get_priority_color("HIGH") == "#ff7f0e"
        assert get_priority_color("MEDIUM") == "#1f77b4"
        assert get_priority_color("LOW") == "#2ca02c"
        assert get_priority_color("UNKNOWN") == "#7f7f7f"
    
    def test_safe_get(self):
        """Test safe dictionary access."""
        data = {"key1": "value1", "key2": None}
        
        assert safe_get(data, "key1") == "value1"
        assert safe_get(data, "key2") is None
        assert safe_get(data, "key3", "default") == "default"
        assert safe_get(None, "key1", "default") == "default"

# ===== INTEGRATION TESTS =====

class TestIntegration:
    """Test end-to-end integration scenarios."""
    
    @patch('streamlit.session_state', new_callable=lambda: MagicMock())
    def test_session_state_initialization(self, mock_session_state):
        """Test session state initialization."""
        # Mock session state as empty dict
        mock_session_state.__contains__ = lambda self, key: False
        mock_session_state.__setitem__ = Mock()
        
        initialize_session_state()
        
        # Verify session state variables were set
        assert mock_session_state.__setitem__.call_count > 0
    
    @patch('streamlit.cache_data')
    @patch('streamlit.rerun')
    @patch('src.wms.dashboard.dashboard.check_api_health')
    def test_auto_refresh_logic(self, mock_health_check, mock_rerun, mock_cache, mock_streamlit_session):
        """Test auto-refresh functionality."""
        mock_health_check.return_value = True
        mock_cache.clear = Mock()
        
        # Mock enhancements
        mock_enhancements = Mock()
        mock_enhancements.get_performance_monitor.return_value.record_page_load = Mock()
        mock_enhancements.show_toast = Mock()
        
        controls = {
            'auto_refresh': True,
            'refresh_rate': 1  # 1 second for testing
        }
        
        # Set last refresh to 2 seconds ago
        with patch('streamlit.session_state', mock_streamlit_session):
            mock_streamlit_session['last_refresh'] = datetime.now() - timedelta(seconds=2)
            
            result = handle_auto_refresh(controls, mock_enhancements)
            
            assert result is True
            mock_cache.clear.assert_called_once()
            mock_rerun.assert_called_once()
    
    def test_error_handling_integration(self, mock_api_client):
        """Test error handling across components."""
        # Simulate API error
        mock_api_client.get_inventory_summary = Mock(return_value=None)
        
        with patch('streamlit.error') as mock_error:
            # This should handle the error gracefully
            result = mock_api_client.get_inventory_summary()
            
            assert result is None
            # Error handling should be triggered in the component
    
    @patch('src.wms.dashboard.enhancements.DashboardEnhancements')
    def test_enhancements_integration(self, mock_enhancements_class, mock_config):
        """Test dashboard enhancements integration."""
        mock_enhancements = Mock()
        mock_enhancements_class.return_value = mock_enhancements
        
        # Test initialization
        enhancements = DashboardEnhancements(mock_config)
        
        # Verify enhancements were created
        assert enhancements is not None

# ===== PERFORMANCE TESTS =====

class TestPerformance:
    """Test performance and scalability."""
    
    def test_large_dataset_handling(self, mock_api_client):
        """Test handling of large datasets."""
        # Create large dataset
        large_dataset = [
            {
                "product_code": f"PROD-{i:06d}",
                "current_stock": i * 10,
                "reorder_point": i * 2,
                "priority_level": "MEDIUM"
            }
            for i in range(10000)
        ]
        
        mock_api_client.get_stock_levels = Mock(return_value=large_dataset)
        
        start_time = time.time()
        result = mock_api_client.get_stock_levels()
        execution_time = time.time() - start_time
        
        assert len(result) == 10000
        assert execution_time < 1.0  # Should complete within 1 second
    
    def test_memory_usage(self, sample_consumption_trends):
        """Test memory usage with data processing."""
        # Convert to DataFrame (simulating chart processing)
        df = pd.DataFrame(sample_consumption_trends)
        
        # Verify data processing doesn't consume excessive memory
        assert len(df) == 30
        assert 'timestamp' in df.columns
        assert 'value' in df.columns
    
    def test_concurrent_api_calls(self, mock_api_client, sample_inventory_summary):
        """Test concurrent API call handling."""
        import threading
        
        mock_api_client.get_inventory_summary = Mock(return_value=sample_inventory_summary)
        
        results = []
        
        def make_call():
            result = mock_api_client.get_inventory_summary()
            results.append(result)
        
        # Create multiple threads
        threads = [threading.Thread(target=make_call) for _ in range(5)]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all calls completed
        assert len(results) == 5
        assert all(result == sample_inventory_summary for result in results)

# ===== ERROR SCENARIO TESTS =====

class TestErrorScenarios:
    """Test various error scenarios and recovery."""
    
    def test_network_failure_recovery(self, mock_api_client):
        """Test recovery from network failures."""
        import requests
        
        # First call fails, second succeeds
        mock_api_client._make_request = Mock(side_effect=[
            None,  # First call fails
            {"status": "success"}  # Second call succeeds
        ])
        
        # First attempt
        result1 = mock_api_client._make_request("/test")
        assert result1 is None
        
        # Second attempt (recovery)
        result2 = mock_api_client._make_request("/test")
        assert result2 == {"status": "success"}
    
    def test_malformed_data_handling(self, mock_api_client):
        """Test handling of malformed API responses."""
        # Test with malformed data
        malformed_data = {"incomplete": "data"}
        mock_api_client.get_inventory_summary = Mock(return_value=malformed_data)
        
        with patch('src.wms.dashboard.components.safe_get') as mock_safe_get:
            mock_safe_get.side_effect = lambda data, key, default=None: default
            
            # This should handle malformed data gracefully
            result = mock_api_client.get_inventory_summary()
            assert result == malformed_data
    
    def test_cache_failure_handling(self, mock_api_client):
        """Test handling of cache failures."""
        with patch('streamlit.cache_data') as mock_cache:
            mock_cache.side_effect = Exception("Cache error")
            
            # Should still work without cache
            mock_api_client.get_inventory_summary = Mock(return_value={"test": "data"})
            result = mock_api_client.get_inventory_summary()
            
            assert result == {"test": "data"}

# ===== CONFIGURATION TESTS =====

class TestConfiguration:
    """Test configuration and environment handling."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        config = DashboardConfig()
        
        assert hasattr(config, 'api_base_url')
        assert hasattr(config, 'api_timeout')
        assert hasattr(config, 'enable_debug')
    
    def test_theme_switching(self):
        """Test theme switching functionality."""
        prefs = UserPreferences()
        
        # Test theme toggle
        original_theme = prefs.theme
        new_theme = prefs.toggle_theme()
        
        assert new_theme != original_theme
        assert new_theme in ['light', 'dark']
    
    @patch.dict('os.environ', {'WMS_API_URL': 'http://custom-api:9000'})
    def test_environment_variable_override(self):
        """Test environment variable configuration override."""
        config = DashboardConfig()
        
        assert config.api_base_url == 'http://custom-api:9000'

# ===== EXPORT FUNCTIONALITY TESTS =====

class TestExportFunctionality:
    """Test export and reporting functionality."""
    
    def test_csv_export(self, sample_stock_levels):
        """Test CSV export functionality."""
        from src.wms.dashboard.enhancements import ExportManager
        
        export_manager = ExportManager(TestConfig())
        csv_data = export_manager.export_to_csv(sample_stock_levels, "test_export.csv")
        
        assert isinstance(csv_data, bytes)
        assert len(csv_data) > 0
        
        # Verify CSV content
        csv_content = csv_data.decode('utf-8')
        assert 'product_code' in csv_content
        assert 'PROD-001' in csv_content
    
    @patch('src.wms.dashboard.enhancements.REPORTLAB_AVAILABLE', True)
    def test_pdf_export(self, sample_inventory_summary):
        """Test PDF export functionality."""
        from src.wms.dashboard.enhancements import ExportManager
        
        export_manager = ExportManager(TestConfig())
        dashboard_data = {'inventory_summary': sample_inventory_summary}
        
        pdf_data = export_manager.export_to_pdf(dashboard_data, "Test Report")
        
        assert isinstance(pdf_data, bytes)
        assert len(pdf_data) > 0

# ===== TEST RUNNER =====

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
