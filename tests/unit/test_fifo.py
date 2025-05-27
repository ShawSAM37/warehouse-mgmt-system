"""
Unit tests for FIFO inventory consumption logic.
Comprehensive test suite covering core functionality, edge cases, and database integrity.
"""
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Generator, Any
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Project imports
from src.wms.inventory.fifo import (
    consume_fifo,
    get_available_stock,
    validate_consumption_request,
    get_fifo_batch_order,
    get_consumption_history
)
from src.wms.inventory.models import InventoryBatch, InventoryTransaction, Product, StorageLocation
from src.wms.enums import TransactionType, UnitOfMeasurement, StorageType, StockType
from src.wms.utils.db import Base

# Test constants
TEST_PRODUCT_CODE = "TEST-PROD-001"
TEST_PRODUCT_CODE_2 = "TEST-PROD-002"
TEST_STORAGE_BIN = "TEST-BIN-A01"
BASE_DATE = datetime(2025, 1, 1, 10, 0, 0)

# Test database configuration
@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine with proper SQLite configuration."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={
            "check_same_thread": False,
        },
        poolclass=StaticPool,
        echo=False
    )
    
    # Configure SQLite for proper transaction handling
    @event.listens_for(engine, "connect")
    def do_connect(dbapi_connection, connection_record):
        # Disable pysqlite's emitting of the BEGIN statement entirely
        dbapi_connection.isolation_level = None

    @event.listens_for(engine, "begin")
    def do_begin(conn):
        # Emit our own BEGIN
        conn.exec_driver_sql("BEGIN")
    
    # Create all tables
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)

@pytest.fixture(scope="function")
def db_session(test_engine) -> Generator[Session, Any, None]:
    """Create database session with proper rollback between tests."""
    connection = test_engine.connect()
    transaction = connection.begin()
    
    # Create session bound to the connection
    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()
    
    yield session
    
    # Cleanup
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(scope="function")
def test_product(db_session: Session) -> Product:
    """Create a test product."""
    product = Product(
        product_code=TEST_PRODUCT_CODE,
        unit_of_measurement=UnitOfMeasurement.KG,
        description="Test Product for FIFO Testing",
        is_active=True,
        created_at=BASE_DATE,
        updated_at=BASE_DATE
    )
    db_session.add(product)
    db_session.commit()
    return product

@pytest.fixture(scope="function")
def test_storage_location(db_session: Session) -> StorageLocation:
    """Create a test storage location."""
    location = StorageLocation(
        storage_bin=TEST_STORAGE_BIN,
        storage_type=StorageType.AMBIENT,
        capacity_total=1000.0,
        capacity_available=800.0,
        is_active=True,
        zone="Test Zone",
        created_at=BASE_DATE,
        updated_at=BASE_DATE
    )
    db_session.add(location)
    db_session.commit()
    return location

@pytest.fixture(scope="function")
def single_batch(db_session: Session, test_product: Product, test_storage_location: StorageLocation) -> InventoryBatch:
    """Create a single test batch with 150.0 quantity."""
    batch = InventoryBatch(
        batch_number="TEST-BATCH-001",
        product_code=TEST_PRODUCT_CODE,
        storage_type=StorageType.AMBIENT,
        storage_bin=TEST_STORAGE_BIN,
        quantity=150.0,
        unit_of_measurement=UnitOfMeasurement.KG,
        stock_type=StockType.UNRESTRICTED,
        goods_receipt_date=BASE_DATE,
        goods_receipt_time=BASE_DATE,
        is_active=True,
        created_at=BASE_DATE,
        updated_at=BASE_DATE
    )
    db_session.add(batch)
    db_session.commit()
    return batch

@pytest.fixture(scope="function")
def fifo_batches(db_session: Session, test_product: Product, test_storage_location: StorageLocation) -> List[InventoryBatch]:
    """Create multiple batches with FIFO-ordered receipt dates."""
    batches = []
    quantities = [100.0, 150.0, 200.0]
    
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
            is_active=True,
            created_at=BASE_DATE + timedelta(days=i),
            updated_at=BASE_DATE + timedelta(days=i)
        )
        batches.append(batch)
        db_session.add(batch)
    
    db_session.commit()
    return batches

@pytest.fixture(scope="function")
def empty_inventory(db_session: Session, test_product: Product) -> None:
    """Ensure clean inventory state for testing."""
    # Clean slate - no batches exist
    pass

class TestFIFOConsumption:
    """Test suite for FIFO inventory consumption logic."""
    
    def test_single_batch_full_consumption(self, db_session: Session, single_batch: InventoryBatch):
        """Test consuming exact quantity from single batch."""
        # Act
        result = consume_fifo(
            db=db_session,
            product_code=TEST_PRODUCT_CODE,
            quantity=150.0,
            performed_by="test_user"
        )
        
        # Assert
        assert len(result) == 1
        assert result[0] == ("TEST-BATCH-001", 150.0)
        
        # Verify batch state
        db_session.refresh(single_batch)
        assert single_batch.quantity == 0.0
        assert single_batch.is_active == False
        
        # Verify transaction created
        transactions = db_session.query(InventoryTransaction).all()
        assert len(transactions) == 1
        assert transactions[0].transaction_type == TransactionType.CONSUMPTION
        assert transactions[0].quantity == 150.0
        assert transactions[0].batch_number == "TEST-BATCH-001"
    
    def test_single_batch_partial_consumption(self, db_session: Session, single_batch: InventoryBatch):
        """Test consuming partial quantity from single batch."""
        # Act
        result = consume_fifo(
            db=db_session,
            product_code=TEST_PRODUCT_CODE,
            quantity=75.0,
            performed_by="test_user"
        )
        
        # Assert
        assert len(result) == 1
        assert result[0] == ("TEST-BATCH-001", 75.0)
        
        # Verify batch state
        db_session.refresh(single_batch)
        assert single_batch.quantity == 75.0
        assert single_batch.is_active == True
    
    def test_multi_batch_fifo_order(self, db_session: Session, fifo_batches: List[InventoryBatch]):
        """Test consumption follows FIFO order across multiple batches."""
        # Act - consume 175.0 (should take all of batch 1 + 75 from batch 2)
        result = consume_fifo(
            db=db_session,
            product_code=TEST_PRODUCT_CODE,
            quantity=175.0,
            performed_by="test_user"
        )
        
        # Assert
        assert len(result) == 2
        assert result[0] == ("TEST-BATCH-001", 100.0)  # First batch fully consumed
        assert result[1] == ("TEST-BATCH-002", 75.0)   # Second batch partially consumed
        
        # Verify batch states
        db_session.refresh(fifo_batches[0])
        db_session.refresh(fifo_batches[1])
        db_session.refresh(fifo_batches[2])
        
        assert fifo_batches[0].quantity == 0.0
        assert fifo_batches[0].is_active == False
        assert fifo_batches[1].quantity == 75.0
        assert fifo_batches[1].is_active == True
        assert fifo_batches[2].quantity == 200.0
        assert fifo_batches[2].is_active == True
    
    def test_insufficient_stock_error(self, db_session: Session, single_batch: InventoryBatch):
        """Test error handling when requesting more than available stock."""
        initial_quantity = single_batch.quantity
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            consume_fifo(
                db=db_session,
                product_code=TEST_PRODUCT_CODE,
                quantity=200.0,
                performed_by="test_user"
            )
        
        assert "Insufficient stock" in str(exc_info.value)
        
        # Verify no changes occurred
        db_session.refresh(single_batch)
        assert single_batch.quantity == initial_quantity
        assert single_batch.is_active == True
        
        # Verify no transactions created
        transactions = db_session.query(InventoryTransaction).all()
        assert len(transactions) == 0
    
    def test_zero_quantity_validation(self, db_session: Session, single_batch: InventoryBatch):
        """Test validation of zero consumption quantity."""
        with pytest.raises(ValueError) as exc_info:
            consume_fifo(
                db=db_session,
                product_code=TEST_PRODUCT_CODE,
                quantity=0.0,
                performed_by="test_user"
            )
        
        assert "must be positive" in str(exc_info.value)
    
    def test_negative_quantity_validation(self, db_session: Session, single_batch: InventoryBatch):
        """Test validation of negative consumption quantity."""
        with pytest.raises(ValueError) as exc_info:
            consume_fifo(
                db=db_session,
                product_code=TEST_PRODUCT_CODE,
                quantity=-10.0,
                performed_by="test_user"
            )
        
        assert "must be positive" in str(exc_info.value)
    
    def test_empty_inventory_error(self, db_session: Session, empty_inventory):
        """Test error handling when no inventory exists."""
        with pytest.raises(ValueError) as exc_info:
            consume_fifo(
                db=db_session,
                product_code=TEST_PRODUCT_CODE,
                quantity=50.0,
                performed_by="test_user"
            )
        
        assert "No available inventory" in str(exc_info.value)
    
    def test_decimal_precision_handling(self, db_session: Session, test_product: Product, test_storage_location: StorageLocation):
        """Test handling of decimal quantities with high precision."""
        # Create batch with precise decimal quantity
        batch = InventoryBatch(
            batch_number="DECIMAL-BATCH",
            product_code=TEST_PRODUCT_CODE,
            storage_type=StorageType.AMBIENT,
            storage_bin=TEST_STORAGE_BIN,
            quantity=123.4567,
            unit_of_measurement=UnitOfMeasurement.KG,
            stock_type=StockType.UNRESTRICTED,
            goods_receipt_date=BASE_DATE,
            goods_receipt_time=BASE_DATE,
            is_active=True,
            created_at=BASE_DATE,
            updated_at=BASE_DATE
        )
        db_session.add(batch)
        db_session.commit()
        
        # Act
        result = consume_fifo(
            db=db_session,
            product_code=TEST_PRODUCT_CODE,
            quantity=23.4567,
            performed_by="test_user"
        )
        
        # Assert
        assert len(result) == 1
        assert result[0] == ("DECIMAL-BATCH", 23.4567)
        
        # Verify remaining quantity with precision
        db_session.refresh(batch)
        assert abs(batch.quantity - 100.0) < 0.0001
    
    def test_transaction_metadata_recording(self, db_session: Session, single_batch: InventoryBatch):
        """Test that transaction metadata is properly recorded."""
        test_date = datetime(2025, 5, 27, 14, 30, 0)
        
        # Act
        consume_fifo(
            db=db_session,
            product_code=TEST_PRODUCT_CODE,
            quantity=50.0,
            transaction_date=test_date,
            reference_document="ORDER-12345",
            performed_by="test_user",
            from_location=TEST_STORAGE_BIN,
            notes="Test consumption for order fulfillment"
        )
        
        # Assert
        transaction = db_session.query(InventoryTransaction).first()
        assert transaction.transaction_date == test_date
        assert transaction.reference_document == "ORDER-12345"
        assert transaction.performed_by == "test_user"
        assert transaction.from_location == TEST_STORAGE_BIN
        assert transaction.notes == "Test consumption for order fulfillment"
    
    @pytest.mark.parametrize("consumption_qty,expected_result", [
        (50.0, [("TEST-BATCH-001", 50.0)]),
        (100.0, [("TEST-BATCH-001", 100.0)]),
        (175.0, [("TEST-BATCH-001", 100.0), ("TEST-BATCH-002", 75.0)]),
        (300.0, [("TEST-BATCH-001", 100.0), ("TEST-BATCH-002", 150.0), ("TEST-BATCH-003", 50.0)]),
        (450.0, [("TEST-BATCH-001", 100.0), ("TEST-BATCH-002", 150.0), ("TEST-BATCH-003", 200.0)])
    ])
    def test_various_consumption_scenarios(self, db_session: Session, fifo_batches: List[InventoryBatch], consumption_qty: float, expected_result: List[tuple]):
        """Test multiple consumption scenarios with parameterized inputs."""
        # Act
        result = consume_fifo(
            db=db_session,
            product_code=TEST_PRODUCT_CODE,
            quantity=consumption_qty,
            performed_by="test_user"
        )
        
        # Assert
        assert result == expected_result
        assert sum(qty for _, qty in result) == consumption_qty

class TestHelperFunctions:
    """Test suite for FIFO helper functions."""
    
    def test_get_available_stock_with_inventory(self, db_session: Session, fifo_batches: List[InventoryBatch]):
        """Test get_available_stock with existing inventory."""
        available = get_available_stock(db_session, TEST_PRODUCT_CODE)
        assert available == 450.0  # 100 + 150 + 200
    
    def test_get_available_stock_empty_inventory(self, db_session: Session, empty_inventory):
        """Test get_available_stock with no inventory."""
        available = get_available_stock(db_session, TEST_PRODUCT_CODE)
        assert available == 0.0
    
    def test_validate_consumption_request_valid(self, db_session: Session, fifo_batches: List[InventoryBatch]):
        """Test validation helper for valid consumption request."""
        valid, message = validate_consumption_request(db_session, TEST_PRODUCT_CODE, 200.0)
        assert valid == True
        assert "valid" in message.lower()
    
    def test_validate_consumption_request_insufficient_stock(self, db_session: Session, fifo_batches: List[InventoryBatch]):
        """Test validation helper for insufficient stock."""
        valid, message = validate_consumption_request(db_session, TEST_PRODUCT_CODE, 500.0)
        assert valid == False
        assert "insufficient" in message.lower()
    
    def test_validate_consumption_request_invalid_quantity(self, db_session: Session, fifo_batches: List[InventoryBatch]):
        """Test validation helper for invalid quantity."""
        valid, message = validate_consumption_request(db_session, TEST_PRODUCT_CODE, -10.0)
        assert valid == False
        assert "positive" in message.lower()
    
    def test_get_fifo_batch_order(self, db_session: Session, fifo_batches: List[InventoryBatch]):
        """Test FIFO batch order retrieval."""
        batch_order = get_fifo_batch_order(db_session, TEST_PRODUCT_CODE)
        
        assert len(batch_order) == 3
        assert batch_order[0]["batch_number"] == "TEST-BATCH-001"
        assert batch_order[1]["batch_number"] == "TEST-BATCH-002"
        assert batch_order[2]["batch_number"] == "TEST-BATCH-003"
        
        # Verify quantities
        assert batch_order[0]["quantity"] == 100.0
        assert batch_order[1]["quantity"] == 150.0
        assert batch_order[2]["quantity"] == 200.0
    
    def test_get_consumption_history(self, db_session: Session, fifo_batches: List[InventoryBatch]):
        """Test consumption history retrieval."""
        # Perform some consumptions
        consume_fifo(db_session, TEST_PRODUCT_CODE, 75.0, performed_by="user1")
        consume_fifo(db_session, TEST_PRODUCT_CODE, 50.0, performed_by="user2")
        
        # Get history
        history = get_consumption_history(db_session, product_code=TEST_PRODUCT_CODE)
        
        assert len(history) == 2
        # History should be ordered by date descending (most recent first)
        assert history[0]["performed_by"] == "user2"
        assert history[1]["performed_by"] == "user1"

class TestDatabaseIntegrity:
    """Test suite for database integrity and transaction safety."""
    
    def test_database_rollback_on_error(self, db_session: Session, fifo_batches: List[InventoryBatch]):
        """Test that database state is preserved when consumption fails."""
        initial_stock = get_available_stock(db_session, TEST_PRODUCT_CODE)
        initial_transaction_count = db_session.query(InventoryTransaction).count()
        
        # Attempt consumption that should fail
        with pytest.raises(ValueError):
            consume_fifo(db_session, TEST_PRODUCT_CODE, 999.0, performed_by="test_user")
        
        # Verify no changes occurred
        assert get_available_stock(db_session, TEST_PRODUCT_CODE) == initial_stock
        assert db_session.query(InventoryTransaction).count() == initial_transaction_count
        
        # Verify batch states unchanged
        for batch in fifo_batches:
            db_session.refresh(batch)
            assert batch.is_active == True
    
    def test_transaction_id_generation(self, db_session: Session, single_batch: InventoryBatch):
        """Test that transaction IDs follow the established pattern."""
        test_date = datetime(2025, 5, 27, 10, 0, 0)
        
        consume_fifo(
            db=db_session,
            product_code=TEST_PRODUCT_CODE,
            quantity=50.0,
            transaction_date=test_date,
            performed_by="test_user"
        )
        
        transaction = db_session.query(InventoryTransaction).first()
        # Should follow pattern: TRX-YYYYMMDD-###
        assert transaction.transaction_id.startswith("TRX-20250527-")
        assert len(transaction.transaction_id) == 16  # TRX-YYYYMMDD-### format
    
    def test_batch_updated_at_timestamp(self, db_session: Session, single_batch: InventoryBatch):
        """Test that batch updated_at timestamp is properly set."""
        original_updated_at = single_batch.updated_at
        test_date = datetime(2025, 5, 27, 15, 30, 0)
        
        consume_fifo(
            db=db_session,
            product_code=TEST_PRODUCT_CODE,
            quantity=50.0,
            transaction_date=test_date,
            performed_by="test_user"
        )
        
        db_session.refresh(single_batch)
        assert single_batch.updated_at == test_date
        assert single_batch.updated_at != original_updated_at

class TestPerformanceAndIntegration:
    """Test suite for performance and integration scenarios."""
    
    def test_large_dataset_consumption(self, db_session: Session, test_product: Product, test_storage_location: StorageLocation):
        """Test FIFO logic with large number of batches."""
        # Create 50 batches
        batches = []
        for i in range(50):
            batch = InventoryBatch(
                batch_number=f"LARGE-BATCH-{i+1:03d}",
                product_code=TEST_PRODUCT_CODE,
                storage_type=StorageType.AMBIENT,
                storage_bin=TEST_STORAGE_BIN,
                quantity=10.0,
                unit_of_measurement=UnitOfMeasurement.KG,
                stock_type=StockType.UNRESTRICTED,
                goods_receipt_date=BASE_DATE + timedelta(hours=i),
                goods_receipt_time=BASE_DATE + timedelta(hours=i),
                is_active=True,
                created_at=BASE_DATE + timedelta(hours=i),
                updated_at=BASE_DATE + timedelta(hours=i)
            )
            batches.append(batch)
            db_session.add(batch)
        
        db_session.commit()
        
        # Consume across many batches
        result = consume_fifo(db_session, TEST_PRODUCT_CODE, 250.0, performed_by="test_user")
        
        # Should consume from first 25 batches
        assert len(result) == 25
        assert sum(qty for _, qty in result) == 250.0
        
        # Verify first 25 batches are deactivated
        for i in range(25):
            db_session.refresh(batches[i])
            assert batches[i].is_active == False
            assert batches[i].quantity == 0.0
    
    def test_multiple_product_isolation(self, db_session: Session, test_storage_location: StorageLocation):
        """Test that consumption is properly isolated by product."""
        # Create products
        product1 = Product(
            product_code=TEST_PRODUCT_CODE,
            unit_of_measurement=UnitOfMeasurement.KG,
            description="Test Product 1",
            is_active=True,
            created_at=BASE_DATE,
            updated_at=BASE_DATE
        )
        product2 = Product(
            product_code=TEST_PRODUCT_CODE_2,
            unit_of_measurement=UnitOfMeasurement.KG,
            description="Test Product 2",
            is_active=True,
            created_at=BASE_DATE,
            updated_at=BASE_DATE
        )
        db_session.add_all([product1, product2])
        
        # Create batches for each product
        batch1 = InventoryBatch(
            batch_number="PROD1-BATCH-001",
            product_code=TEST_PRODUCT_CODE,
            storage_type=StorageType.AMBIENT,
            storage_bin=TEST_STORAGE_BIN,
            quantity=100.0,
            unit_of_measurement=UnitOfMeasurement.KG,
            stock_type=StockType.UNRESTRICTED,
            goods_receipt_date=BASE_DATE,
            goods_receipt_time=BASE_DATE,
            is_active=True,
            created_at=BASE_DATE,
            updated_at=BASE_DATE
        )
        batch2 = InventoryBatch(
            batch_number="PROD2-BATCH-001",
            product_code=TEST_PRODUCT_CODE_2,
            storage_type=StorageType.AMBIENT,
            storage_bin=TEST_STORAGE_BIN,
            quantity=100.0,
            unit_of_measurement=UnitOfMeasurement.KG,
            stock_type=StockType.UNRESTRICTED,
            goods_receipt_date=BASE_DATE,
            goods_receipt_time=BASE_DATE,
            is_active=True,
            created_at=BASE_DATE,
            updated_at=BASE_DATE
        )
        db_session.add_all([batch1, batch2])
        db_session.commit()
        
        # Consume from product 1
        result = consume_fifo(db_session, TEST_PRODUCT_CODE, 50.0, performed_by="test_user")
        
        # Verify only product 1 batch affected
        assert len(result) == 1
        assert result[0][0] == "PROD1-BATCH-001"
        
        db_session.refresh(batch1)
        db_session.refresh(batch2)
        assert batch1.quantity == 50.0
        assert batch2.quantity == 100.0  # Unchanged
