"""
Unit tests for reporting calculations and dashboard functionality.
Comprehensive test suite covering calculation accuracy, API endpoints, and edge cases.
"""
import pytest
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Generator, Any
from unittest.mock import Mock, patch
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Project imports
from src.wms.reporting.calculations import (
    get_current_stock_levels, calculate_inventory_turnover, get_low_stock_products,
    calculate_days_of_supply, calculate_supplier_performance, get_delivery_accuracy_metrics,
    calculate_storage_utilization, get_batch_expiry_alerts, get_consumption_trends,
    calculate_fifo_efficiency
)
from src.wms.reporting.schemas import (
    ProductStockLevel, SupplierPerformanceMetric, StorageUtilizationReport,
    BatchExpiryAlert, TimeSeriesDataPoint, PriorityLevel
)
from src.wms.inventory.models import (
    InventoryBatch, InventoryTransaction, Product, Supplier, 
    PurchaseOrder, PurchaseOrderLineItem, StorageLocation
)
from src.wms.enums import TransactionType, POStatus, UnitOfMeasurement, StorageType, StockType
from src.wms.utils.db import Base

# Test constants
TEST_PRODUCT_CODE = "TEST-PROD-001"
TEST_SUPPLIER_ID = "TEST-SUP-001"
TEST_STORAGE_BIN = "TEST-BIN-A01"
BASE_DATE = datetime(2025, 1, 1, 10, 0, 0)

# Test database configuration
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
Base.metadata.create_all(bind=test_engine)

@pytest.fixture(scope="function")
def db_session() -> Generator[Session, Any, None]:
    """Create database session with rollback isolation."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()

@pytest.fixture(scope="function")
def test_product(db_session: Session) -> Product:
    """Create a test product."""
    product = Product(
        product_code=TEST_PRODUCT_CODE,
        unit_of_measurement=UnitOfMeasurement.KG,
        description="Test Product for Reporting",
        reorder_point=50.0,
        is_active=True,
        created_at=BASE_DATE,
        updated_at=BASE_DATE
    )
    db_session.add(product)
    db_session.commit()
    return product

@pytest.fixture(scope="function")
def test_supplier(db_session: Session) -> Supplier:
    """Create a test supplier."""
    supplier = Supplier(
        supplier_id=TEST_SUPPLIER_ID,
        supplier_name="Test Supplier Ltd.",
        partner_number="P12345",
        contact_person="John Doe",
        email="john@testsupplier.com",
        phone="+1-555-0123",
        average_lead_time_days=7,
        is_active=True,
        created_at=BASE_DATE,
        updated_at=BASE_DATE
    )
    db_session.add(supplier)
    db_session.commit()
    return supplier

@pytest.fixture(scope="function")
def test_storage_location(db_session: Session) -> StorageLocation:
    """Create a test storage location."""
    location = StorageLocation(
        storage_bin=TEST_STORAGE_BIN,
        storage_type=StorageType.AMBIENT,
        capacity_total=1000.0,
        capacity_available=800.0,
        zone="Test Zone",
        is_active=True,
        created_at=BASE_DATE,
        updated_at=BASE_DATE
    )
    db_session.add(location)
    db_session.commit()
    return location

@pytest.fixture(scope="function")
def test_batches(db_session: Session, test_product: Product, test_storage_location: StorageLocation) -> List[InventoryBatch]:
    """Create test inventory batches with varying quantities and dates."""
    batches = []
    quantities = [100.0, 150.0, 75.0]
    
    for i, qty in enumerate(quantities):
        batch = InventoryBatch(
            batch_number=f"TEST-BATCH-{i+1:03d}",
            product_code=TEST_PRODUCT_CODE,
            storage_type=StorageType.AMBIENT,
            storage_bin=TEST_STORAGE_BIN,
            quantity=qty,
            unit_of_measurement=UnitOfMeasurement.KG,
            stock_type=StockType.UNRESTRICTED,
            goods_receipt_date=BASE_DATE + timedelta(days=i),
            goods_receipt_time=BASE_DATE + timedelta(days=i),
            expiry_date=BASE_DATE + timedelta(days=365 + i),
            is_active=True,
            created_at=BASE_DATE + timedelta(days=i),
            updated_at=BASE_DATE + timedelta(days=i)
        )
        batches.append(batch)
        db_session.add(batch)
    
    db_session.commit()
    return batches

@pytest.fixture(scope="function")
def test_transactions(db_session: Session, test_batches: List[InventoryBatch]) -> List[InventoryTransaction]:
    """Create test transactions for consumption analysis."""
    transactions = []
    
    for i, batch in enumerate(test_batches):
        # Create consumption transactions
        for day in range(5):
            transaction = InventoryTransaction(
                transaction_id=f"TRX-TEST-{i:03d}-{day:03d}",
                batch_number=batch.batch_number,
                transaction_type=TransactionType.CONSUMPTION,
                quantity=10.0,
                transaction_date=BASE_DATE + timedelta(days=day + 10),
                performed_by="test_user",
                notes=f"Test consumption {day}",
                created_at=BASE_DATE + timedelta(days=day + 10),
                updated_at=BASE_DATE + timedelta(days=day + 10)
            )
            transactions.append(transaction)
            db_session.add(transaction)
    
    db_session.commit()
    return transactions

class TestInventoryCalculations:
    """Test suite for inventory calculation functions."""
    
    def test_get_current_stock_levels_with_data(self, db_session: Session, test_batches: List[InventoryBatch]):
        """Test getting current stock levels with existing data."""
        stock_levels = get_current_stock_levels(db_session, [TEST_PRODUCT_CODE])
        
        assert len(stock_levels) == 1
        assert stock_levels[0].product_code == TEST_PRODUCT_CODE
        assert stock_levels[0].current_stock == Decimal('325.0')  # 100 + 150 + 75
        assert stock_levels[0].reorder_point == Decimal('50.0')
    
    def test_get_current_stock_levels_no_data(self, db_session: Session, test_product: Product):
        """Test getting stock levels when no batches exist."""
        stock_levels = get_current_stock_levels(db_session, [TEST_PRODUCT_CODE])
        assert len(stock_levels) == 0
    
    def test_get_current_stock_levels_all_products(self, db_session: Session, test_batches: List[InventoryBatch]):
        """Test getting stock levels for all products."""
        stock_levels = get_current_stock_levels(db_session)
        assert len(stock_levels) >= 1
        assert any(level.product_code == TEST_PRODUCT_CODE for level in stock_levels)
    
    @pytest.mark.parametrize("consumption_qty,expected_turnover", [
        (150.0, 0.4615),  # 150 / (325/1) = 0.4615
        (300.0, 0.9231),  # 300 / (325/1) = 0.9231
        (0.0, 0.0)        # No consumption
    ])
    def test_calculate_inventory_turnover(
        self, 
        db_session: Session, 
        test_batches: List[InventoryBatch],
        consumption_qty: float,
        expected_turnover: float
    ):
        """Test inventory turnover calculation with various consumption levels."""
        # Create consumption transaction
        if consumption_qty > 0:
            transaction = InventoryTransaction(
                transaction_id="TRX-TURNOVER-TEST",
                batch_number=test_batches[0].batch_number,
                transaction_type=TransactionType.CONSUMPTION,
                quantity=consumption_qty,
                transaction_date=datetime.now() - timedelta(days=15),
                performed_by="test_user"
            )
            db_session.add(transaction)
            db_session.commit()
        
        turnover = calculate_inventory_turnover(db_session, TEST_PRODUCT_CODE, 30)
        assert abs(turnover - expected_turnover) < 0.01
    
    def test_get_low_stock_products(self, db_session: Session, test_product: Product):
        """Test identification of low stock products."""
        # Create a batch with low quantity (below 20% of reorder point)
        low_stock_batch = InventoryBatch(
            batch_number="LOW-STOCK-BATCH",
            product_code=TEST_PRODUCT_CODE,
            storage_type=StorageType.AMBIENT,
            storage_bin=TEST_STORAGE_BIN,
            quantity=5.0,  # Below 20% of reorder point (50 * 0.2 = 10)
            unit_of_measurement=UnitOfMeasurement.KG,
            stock_type=StockType.UNRESTRICTED,
            goods_receipt_date=BASE_DATE,
            is_active=True
        )
        db_session.add(low_stock_batch)
        db_session.commit()
        
        low_stock_products = get_low_stock_products(db_session, threshold_percentage=0.2)
        
        assert len(low_stock_products) == 1
        assert low_stock_products[0].product_code == TEST_PRODUCT_CODE
        assert low_stock_products[0].priority_level == PriorityLevel.URGENT
    
    def test_calculate_days_of_supply_with_consumption(
        self, 
        db_session: Session, 
        test_batches: List[InventoryBatch],
        test_transactions: List[InventoryTransaction]
    ):
        """Test days of supply calculation with consumption history."""
        days_of_supply = calculate_days_of_supply(db_session, TEST_PRODUCT_CODE)
        
        # With 325 total stock and consumption pattern, should have reasonable days of supply
        assert days_of_supply > 0
        assert isinstance(days_of_supply, int)
    
    def test_calculate_days_of_supply_no_consumption(self, db_session: Session, test_batches: List[InventoryBatch]):
        """Test days of supply calculation without consumption history."""
        days_of_supply = calculate_days_of_supply(db_session, TEST_PRODUCT_CODE)
        assert days_of_supply == -1  # Cannot calculate without consumption

class TestSupplierCalculations:
    """Test suite for supplier performance calculations."""
    
    def test_calculate_supplier_performance_with_orders(
        self, 
        db_session: Session, 
        test_supplier: Supplier
    ):
        """Test supplier performance calculation with order history."""
        # Create test purchase orders
        orders = []
        for i in range(3):
            order = PurchaseOrder(
                po_number=f"PO-TEST-{i:03d}",
                supplier_id=TEST_SUPPLIER_ID,
                order_date=BASE_DATE + timedelta(days=i * 10),
                expected_delivery_date=BASE_DATE + timedelta(days=i * 10 + 7),
                actual_delivery_date=BASE_DATE + timedelta(days=i * 10 + 6),  # On time
                status=POStatus.RECEIVED,
                created_at=BASE_DATE + timedelta(days=i * 10),
                updated_at=BASE_DATE + timedelta(days=i * 10)
            )
            orders.append(order)
            db_session.add(order)
        
        db_session.commit()
        
        performance = calculate_supplier_performance(db_session, TEST_SUPPLIER_ID, 90)
        
        assert performance.supplier_id == TEST_SUPPLIER_ID
        assert performance.on_time_delivery_rate == 100.0  # All orders on time
        assert performance.average_lead_time == 6.0  # Average 6 days
        assert performance.total_orders_last_30_days >= 0
    
    def test_calculate_supplier_performance_no_orders(self, db_session: Session, test_supplier: Supplier):
        """Test supplier performance calculation without order history."""
        performance = calculate_supplier_performance(db_session, TEST_SUPPLIER_ID, 90)
        
        assert performance.supplier_id == TEST_SUPPLIER_ID
        assert performance.on_time_delivery_rate == 0.0
        assert performance.average_lead_time == 0.0
        assert performance.total_orders_last_30_days == 0
    
    def test_get_delivery_accuracy_metrics(self, db_session: Session, test_supplier: Supplier):
        """Test overall delivery accuracy metrics calculation."""
        # Create mixed delivery performance orders
        orders_data = [
            (BASE_DATE, BASE_DATE + timedelta(days=7), BASE_DATE + timedelta(days=6)),  # On time
            (BASE_DATE + timedelta(days=5), BASE_DATE + timedelta(days=12), BASE_DATE + timedelta(days=14)),  # Late
            (BASE_DATE + timedelta(days=10), BASE_DATE + timedelta(days=17), BASE_DATE + timedelta(days=17))  # On time
        ]
        
        for i, (order_date, expected, actual) in enumerate(orders_data):
            order = PurchaseOrder(
                po_number=f"PO-ACCURACY-{i:03d}",
                supplier_id=TEST_SUPPLIER_ID,
                order_date=order_date,
                expected_delivery_date=expected,
                actual_delivery_date=actual,
                status=POStatus.RECEIVED
            )
            db_session.add(order)
        
        db_session.commit()
        
        metrics = get_delivery_accuracy_metrics(db_session, 30)
        
        assert metrics["overall_on_time_rate"] == 66.67  # 2 out of 3 on time
        assert metrics["average_delay_days"] == 2.0  # One order 2 days late
        assert metrics["total_orders_analyzed"] == 3

class TestStorageCalculations:
    """Test suite for storage utilization calculations."""
    
    def test_calculate_storage_utilization(
        self, 
        db_session: Session, 
        test_storage_location: StorageLocation,
        test_batches: List[InventoryBatch]
    ):
        """Test storage utilization calculation."""
        utilization_reports = calculate_storage_utilization(db_session)
        
        assert len(utilization_reports) >= 1
        
        # Find our test storage location
        test_report = next(
            (r for r in utilization_reports if r.storage_bin == TEST_STORAGE_BIN), 
            None
        )
        
        assert test_report is not None
        assert test_report.capacity_total == Decimal('1000.0')
        assert test_report.capacity_used == Decimal('325.0')  # Total from test batches
        assert test_report.utilization_percentage == 32.5  # 325/1000 * 100
    
    def test_calculate_storage_utilization_empty_location(
        self, 
        db_session: Session, 
        test_storage_location: StorageLocation
    ):
        """Test storage utilization for empty location."""
        utilization_reports = calculate_storage_utilization(db_session)
        
        test_report = next(
            (r for r in utilization_reports if r.storage_bin == TEST_STORAGE_BIN), 
            None
        )
        
        assert test_report is not None
        assert test_report.utilization_percentage == 0.0
    
    def test_get_batch_expiry_alerts(self, db_session: Session, test_product: Product, test_storage_location: StorageLocation):
        """Test batch expiry alert generation."""
        # Create batches with different expiry dates
        expiry_dates = [
            datetime.now() + timedelta(days=5),   # Urgent
            datetime.now() + timedelta(days=15),  # Medium
            datetime.now() + timedelta(days=45)   # Low
        ]
        
        for i, expiry_date in enumerate(expiry_dates):
            batch = InventoryBatch(
                batch_number=f"EXPIRY-BATCH-{i:03d}",
                product_code=TEST_PRODUCT_CODE,
                storage_type=StorageType.AMBIENT,
                storage_bin=TEST_STORAGE_BIN,
                quantity=50.0,
                unit_of_measurement=UnitOfMeasurement.KG,
                stock_type=StockType.UNRESTRICTED,
                goods_receipt_date=BASE_DATE,
                expiry_date=expiry_date,
                is_active=True
            )
            db_session.add(batch)
        
        db_session.commit()
        
        alerts = get_batch_expiry_alerts(db_session, days_ahead=30)
        
        # Should get alerts for first two batches (within 30 days)
        assert len(alerts) == 2
        assert alerts[0].priority == PriorityLevel.HIGH  # 5 days
        assert alerts[1].priority == PriorityLevel.MEDIUM  # 15 days

class TestTransactionAnalytics:
    """Test suite for transaction analytics functions."""
    
    def test_get_consumption_trends(
        self, 
        db_session: Session, 
        test_batches: List[InventoryBatch],
        test_transactions: List[InventoryTransaction]
    ):
        """Test consumption trends calculation."""
        trends = get_consumption_trends(db_session, days=30)
        
        assert len(trends) > 0
        assert all(isinstance(point, TimeSeriesDataPoint) for point in trends)
        assert all(point.metric_name == "daily_consumption" for point in trends)
    
    def test_calculate_fifo_efficiency(
        self, 
        db_session: Session, 
        test_batches: List[InventoryBatch],
        test_transactions: List[InventoryTransaction]
    ):
        """Test FIFO efficiency calculation."""
        efficiency = calculate_fifo_efficiency(db_session, TEST_PRODUCT_CODE)
        
        assert 0.0 <= efficiency <= 100.0
        assert isinstance(efficiency, float)
    
    def test_calculate_fifo_efficiency_no_transactions(self, db_session: Session, test_batches: List[InventoryBatch]):
        """Test FIFO efficiency with no consumption transactions."""
        efficiency = calculate_fifo_efficiency(db_session, TEST_PRODUCT_CODE)
        assert efficiency == 0.0

class TestEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    def test_calculations_with_empty_database(self, db_session: Session):
        """Test all calculations with empty database."""
        # Test that functions handle empty database gracefully
        assert get_current_stock_levels(db_session) == []
        assert calculate_inventory_turnover(db_session, "NONEXISTENT", 30) == 0.0
        assert get_low_stock_products(db_session) == []
        assert calculate_days_of_supply(db_session, "NONEXISTENT") == -1
        assert calculate_storage_utilization(db_session) == []
        assert get_batch_expiry_alerts(db_session) == []
        assert get_consumption_trends(db_session) == []
        assert calculate_fifo_efficiency(db_session, "NONEXISTENT") == 0.0
    
    def test_calculations_with_invalid_product_code(self, db_session: Session, test_batches: List[InventoryBatch]):
        """Test calculations with non-existent product code."""
        invalid_code = "INVALID-PRODUCT"
        
        assert get_current_stock_levels(db_session, [invalid_code]) == []
        assert calculate_inventory_turnover(db_session, invalid_code, 30) == 0.0
        assert calculate_days_of_supply(db_session, invalid_code) == -1
        assert calculate_fifo_efficiency(db_session, invalid_code) == 0.0
    
    @pytest.mark.parametrize("days_param", [0, -1, 1000])
    def test_calculations_with_edge_case_days(self, db_session: Session, test_batches: List[InventoryBatch], days_param: int):
        """Test calculations with edge case day parameters."""
        # Functions should handle edge cases gracefully
        if days_param <= 0:
            # Should handle gracefully or use default
            result = calculate_inventory_turnover(db_session, TEST_PRODUCT_CODE, days_param)
            assert isinstance(result, float)
        else:
            # Should work normally with large values
            result = calculate_inventory_turnover(db_session, TEST_PRODUCT_CODE, days_param)
            assert isinstance(result, float)

class TestPerformance:
    """Test suite for performance benchmarks."""
    
    def test_large_dataset_performance(self, db_session: Session, test_product: Product, test_storage_location: StorageLocation):
        """Test calculation performance with large datasets."""
        # Create large number of batches and transactions
        batch_count = 1000
        transaction_count = 5000
        
        # Create batches
        batches = []
        for i in range(batch_count):
            batch = InventoryBatch(
                batch_number=f"PERF-BATCH-{i:06d}",
                product_code=TEST_PRODUCT_CODE,
                storage_type=StorageType.AMBIENT,
                storage_bin=TEST_STORAGE_BIN,
                quantity=100.0,
                unit_of_measurement=UnitOfMeasurement.KG,
                stock_type=StockType.UNRESTRICTED,
                goods_receipt_date=BASE_DATE + timedelta(days=i % 365),
                is_active=True
            )
            batches.append(batch)
            if i % 100 == 0:  # Commit in batches
                db_session.add_all(batches)
                db_session.commit()
                batches = []
        
        if batches:
            db_session.add_all(batches)
            db_session.commit()
        
        # Test performance of stock level calculation
        start_time = time.time()
        stock_levels = get_current_stock_levels(db_session, [TEST_PRODUCT_CODE])
        execution_time = time.time() - start_time
        
        assert len(stock_levels) == 1
        assert execution_time < 5.0  # Should complete within 5 seconds
        
        # Test performance of storage utilization
        start_time = time.time()
        utilization = calculate_storage_utilization(db_session)
        execution_time = time.time() - start_time
        
        assert len(utilization) >= 1
        assert execution_time < 3.0  # Should complete within 3 seconds
