"""
Integration tests for Warehouse Management System API endpoints.
Comprehensive test suite covering CRUD operations, business logic, error handling, and performance.
"""
import pytest
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# Project imports
from src.wms.main import app
from src.wms.utils.db import Base, get_db
from src.wms.enums import TransactionType, UnitOfMeasurement, StorageType, StockType

# Test database configuration
TEST_DATABASE_URL = "sqlite:///:memory:"

# Create test engine with proper SQLite configuration
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False
)

# Configure SQLite for proper transaction handling
@event.listens_for(test_engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()

# Create session factory
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Create all tables
Base.metadata.create_all(bind=test_engine)

# Test data factory
class TestDataFactory:
    """Factory for generating consistent test data across integration tests."""
    
    @staticmethod
    def product_payload(product_code: str = None) -> Dict[str, Any]:
        """Generate valid product creation payload."""
        return {
            "product_code": product_code or f"TEST-PROD-{uuid.uuid4().hex[:8].upper()}",
            "unit_of_measurement": UnitOfMeasurement.KG.value,
            "description": "Test Product for Integration Testing",
            "hsn_sac_code": "1234",
            "default_shelf_life_days": 365,
            "weight_per_unit": 1.0,
            "volume_per_unit": 0.001
        }
    
    @staticmethod
    def storage_location_payload(storage_bin: str = None) -> Dict[str, Any]:
        """Generate valid storage location creation payload."""
        return {
            "storage_bin": storage_bin or f"TEST-BIN-{uuid.uuid4().hex[:4].upper()}",
            "storage_type": StorageType.AMBIENT.value,
            "capacity_total": 100.0,
            "capacity_available": 80.0,
            "zone": "Test Zone",
            "aisle": "A",
            "rack": "01",
            "level": "1"
        }
    
    @staticmethod
    def supplier_payload(supplier_id: str = None) -> Dict[str, Any]:
        """Generate valid supplier creation payload."""
        return {
            "supplier_id": supplier_id or f"SUP-{uuid.uuid4().hex[:6].upper()}",
            "supplier_name": "Test Supplier Ltd.",
            "partner_number": f"P{uuid.uuid4().hex[:8].upper()}",
            "contact_person": "John Doe",
            "email": "john.doe@testsupplier.com",
            "phone": "+1-555-0123",
            "address": "123 Test Street, Test City"
        }
    
    @staticmethod
    def batch_payload(product_code: str, storage_bin: str, batch_number: str = None) -> Dict[str, Any]:
        """Generate valid inventory batch creation payload."""
        return {
            "batch_number": batch_number or f"{datetime.now().strftime('%Y%m%d')}-RAW-{uuid.uuid4().hex[:4].upper()}",
            "product_code": product_code,
            "storage_type": StorageType.AMBIENT.value,
            "storage_bin": storage_bin,
            "quantity": 100.0,
            "unit_of_measurement": UnitOfMeasurement.KG.value,
            "stock_type": StockType.UNRESTRICTED.value,
            "goods_receipt_date": datetime.now().isoformat(),
            "goods_receipt_time": datetime.now().isoformat(),
            "country_of_origin": "US",
            "expiry_date": (datetime.now() + timedelta(days=365)).isoformat()
        }
    
    @staticmethod
    def transaction_payload(batch_number: str, quantity: float = 10.0) -> Dict[str, Any]:
        """Generate valid inventory transaction creation payload."""
        return {
            "transaction_id": f"TRX-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:3].upper()}",
            "batch_number": batch_number,
            "transaction_type": TransactionType.CONSUMPTION.value,
            "quantity": quantity,
            "transaction_date": datetime.now().isoformat(),
            "performed_by": "test_user",
            "notes": "Integration test transaction"
        }
    
    @staticmethod
    def purchase_order_payload(supplier_id: str, po_number: str = None) -> Dict[str, Any]:
        """Generate valid purchase order creation payload."""
        return {
            "po_number": po_number or f"PO-{uuid.uuid4().hex[:8].upper()}",
            "supplier_id": supplier_id,
            "order_date": datetime.now().isoformat(),
            "expected_delivery_date": (datetime.now() + timedelta(days=7)).isoformat(),
            "status": "Open"
        }

# Pytest fixtures
@pytest.fixture(scope="function")
def db_session() -> Session:
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
def client(db_session: Session) -> TestClient:
    """Create FastAPI test client with database dependency override."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()

@pytest.fixture(scope="function")
def sample_product(client: TestClient) -> Dict[str, Any]:
    """Create a sample product for testing."""
    product_data = TestDataFactory.product_payload("SAMPLE-PROD-001")
    response = client.post("/api/v1/products/", json=product_data)
    assert response.status_code == 201
    return response.json()

@pytest.fixture(scope="function")
def sample_storage_location(client: TestClient) -> Dict[str, Any]:
    """Create a sample storage location for testing."""
    location_data = TestDataFactory.storage_location_payload("SAMPLE-BIN-A01")
    response = client.post("/api/v1/storage/", json=location_data)
    assert response.status_code == 201
    return response.json()

@pytest.fixture(scope="function")
def sample_supplier(client: TestClient) -> Dict[str, Any]:
    """Create a sample supplier for testing."""
    supplier_data = TestDataFactory.supplier_payload("SAMPLE-SUP-001")
    response = client.post("/api/v1/suppliers/", json=supplier_data)
    assert response.status_code == 201
    return response.json()

# Test classes organized by API entity
class TestProductAPI:
    """Integration tests for Product API endpoints."""
    
    def test_create_product_success(self, client: TestClient):
        """Test successful product creation."""
        product_data = TestDataFactory.product_payload()
        response = client.post("/api/v1/products/", json=product_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["product_code"] == product_data["product_code"]
        assert data["unit_of_measurement"] == product_data["unit_of_measurement"]
        assert data["description"] == product_data["description"]
        assert data["is_active"] == True
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_create_product_validation_error(self, client: TestClient):
        """Test product creation with invalid data."""
        invalid_data = {"unit_of_measurement": "KG"}  # Missing required product_code
        response = client.post("/api/v1/products/", json=invalid_data)
        
        assert response.status_code == 422
        error_detail = response.json()
        assert "product_code" in str(error_detail)
    
    def test_create_product_duplicate_error(self, client: TestClient):
        """Test product creation with duplicate product code."""
        product_data = TestDataFactory.product_payload("DUPLICATE-PROD")
        
        # First creation should succeed
        response1 = client.post("/api/v1/products/", json=product_data)
        assert response1.status_code == 201
        
        # Second creation should fail
        response2 = client.post("/api/v1/products/", json=product_data)
        assert response2.status_code == 409
        assert "already exists" in response2.json()["detail"]
    
    def test_get_product_success(self, client: TestClient, sample_product: Dict[str, Any]):
        """Test retrieving existing product."""
        product_code = sample_product["product_code"]
        response = client.get(f"/api/v1/products/{product_code}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["product_code"] == product_code
        assert data["description"] == sample_product["description"]
    
    def test_get_product_not_found(self, client: TestClient):
        """Test 404 for non-existent product."""
        response = client.get("/api/v1/products/NON-EXISTENT-PROD")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_update_product_success(self, client: TestClient, sample_product: Dict[str, Any]):
        """Test product update."""
        product_code = sample_product["product_code"]
        update_data = {"description": "Updated Test Product"}
        
        response = client.put(f"/api/v1/products/{product_code}", json=update_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["description"] == update_data["description"]
        assert data["updated_at"] != sample_product["updated_at"]
    
    def test_delete_product_success(self, client: TestClient, sample_product: Dict[str, Any]):
        """Test product deletion (soft delete)."""
        product_code = sample_product["product_code"]
        
        # Delete product
        response = client.delete(f"/api/v1/products/{product_code}")
        assert response.status_code == 204
        
        # Verify product is no longer accessible
        response = client.get(f"/api/v1/products/{product_code}")
        assert response.status_code == 404
    
    def test_list_products_pagination(self, client: TestClient):
        """Test product listing with pagination."""
        # Create multiple products
        for i in range(5):
            product_data = TestDataFactory.product_payload(f"LIST-PROD-{i:03d}")
            client.post("/api/v1/products/", json=product_data)
        
        # Test pagination
        response = client.get("/api/v1/products/?skip=2&limit=2")
        assert response.status_code == 200
        
        products = response.json()
        assert len(products) <= 2

class TestInventoryBatchAPI:
    """Integration tests for Inventory Batch API endpoints."""
    
    def test_create_batch_success(self, client: TestClient, sample_product: Dict[str, Any], sample_storage_location: Dict[str, Any]):
        """Test successful batch creation."""
        batch_data = TestDataFactory.batch_payload(
            sample_product["product_code"],
            sample_storage_location["storage_bin"]
        )
        
        response = client.post("/api/v1/inventory/batches/", json=batch_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["batch_number"] == batch_data["batch_number"]
        assert data["product_code"] == batch_data["product_code"]
        assert data["quantity"] == batch_data["quantity"]
        assert data["is_active"] == True
    
    def test_create_batch_invalid_product(self, client: TestClient, sample_storage_location: Dict[str, Any]):
        """Test batch creation with non-existent product."""
        batch_data = TestDataFactory.batch_payload(
            "NON-EXISTENT-PROD",
            sample_storage_location["storage_bin"]
        )
        
        response = client.post("/api/v1/inventory/batches/", json=batch_data)
        assert response.status_code == 400
        assert "not found" in response.json()["detail"]
    
    def test_get_batch_with_relationships(self, client: TestClient, sample_product: Dict[str, Any], sample_storage_location: Dict[str, Any]):
        """Test retrieving batch with related transactions."""
        batch_data = TestDataFactory.batch_payload(
            sample_product["product_code"],
            sample_storage_location["storage_bin"]
        )
        
        # Create batch
        create_response = client.post("/api/v1/inventory/batches/", json=batch_data)
        assert create_response.status_code == 201
        
        # Get batch with relationships
        batch_number = batch_data["batch_number"]
        response = client.get(f"/api/v1/inventory/batches/{batch_number}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["batch_number"] == batch_number
        assert "transactions" in data
        assert "quality_checks" in data
    
    def test_update_batch_quantity(self, client: TestClient, sample_product: Dict[str, Any], sample_storage_location: Dict[str, Any]):
        """Test updating batch quantity."""
        batch_data = TestDataFactory.batch_payload(
            sample_product["product_code"],
            sample_storage_location["storage_bin"]
        )
        
        # Create batch
        create_response = client.post("/api/v1/inventory/batches/", json=batch_data)
        assert create_response.status_code == 201
        
        # Update quantity
        batch_number = batch_data["batch_number"]
        update_data = {"quantity": 75.0}
        
        response = client.put(f"/api/v1/inventory/batches/{batch_number}", json=update_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["quantity"] == 75.0

class TestInventoryTransactionAPI:
    """Integration tests for Inventory Transaction API endpoints."""
    
    def test_create_transaction_success(self, client: TestClient, sample_product: Dict[str, Any], sample_storage_location: Dict[str, Any]):
        """Test successful transaction creation."""
        # Create batch first
        batch_data = TestDataFactory.batch_payload(
            sample_product["product_code"],
            sample_storage_location["storage_bin"]
        )
        batch_response = client.post("/api/v1/inventory/batches/", json=batch_data)
        assert batch_response.status_code == 201
        
        # Create transaction
        transaction_data = TestDataFactory.transaction_payload(batch_data["batch_number"], 25.0)
        response = client.post("/api/v1/inventory/transactions/", json=transaction_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["batch_number"] == transaction_data["batch_number"]
        assert data["quantity"] == transaction_data["quantity"]
        assert data["transaction_type"] == transaction_data["transaction_type"]
    
    def test_list_transactions_with_filtering(self, client: TestClient, sample_product: Dict[str, Any], sample_storage_location: Dict[str, Any]):
        """Test transaction listing with filters."""
        # Create batch and transactions
        batch_data = TestDataFactory.batch_payload(
            sample_product["product_code"],
            sample_storage_location["storage_bin"]
        )
        client.post("/api/v1/inventory/batches/", json=batch_data)
        
        # Create multiple transactions
        for i in range(3):
            transaction_data = TestDataFactory.transaction_payload(batch_data["batch_number"], 10.0 + i)
            client.post("/api/v1/inventory/transactions/", json=transaction_data)
        
        # Test filtering by batch number
        response = client.get(f"/api/v1/inventory/transactions/?batch_number={batch_data['batch_number']}")
        assert response.status_code == 200
        
        transactions = response.json()
        assert len(transactions) == 3
        for txn in transactions:
            assert txn["batch_number"] == batch_data["batch_number"]

class TestStorageLocationAPI:
    """Integration tests for Storage Location API endpoints."""
    
    def test_create_storage_location_success(self, client: TestClient):
        """Test successful storage location creation."""
        location_data = TestDataFactory.storage_location_payload()
        response = client.post("/api/v1/storage/", json=location_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["storage_bin"] == location_data["storage_bin"]
        assert data["storage_type"] == location_data["storage_type"]
        assert data["capacity_total"] == location_data["capacity_total"]
    
    def test_get_storage_location_with_batches(self, client: TestClient, sample_product: Dict[str, Any]):
        """Test retrieving storage location with current batches."""
        # Create storage location
        location_data = TestDataFactory.storage_location_payload("STORAGE-TEST-BIN")
        location_response = client.post("/api/v1/storage/", json=location_data)
        assert location_response.status_code == 201
        
        # Create batch in this location
        batch_data = TestDataFactory.batch_payload(
            sample_product["product_code"],
            "STORAGE-TEST-BIN"
        )
        client.post("/api/v1/inventory/batches/", json=batch_data)
        
        # Get location with batches
        response = client.get("/api/v1/storage/STORAGE-TEST-BIN")
        
        assert response.status_code == 200
        data = response.json()
        assert data["storage_bin"] == "STORAGE-TEST-BIN"
        assert "batches" in data
        assert len(data["batches"]) == 1

class TestSupplierAPI:
    """Integration tests for Supplier API endpoints."""
    
    def test_create_supplier_success(self, client: TestClient):
        """Test successful supplier creation."""
        supplier_data = TestDataFactory.supplier_payload()
        response = client.post("/api/v1/suppliers/", json=supplier_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["supplier_id"] == supplier_data["supplier_id"]
        assert data["supplier_name"] == supplier_data["supplier_name"]
        assert data["email"] == supplier_data["email"]
    
    def test_get_supplier_with_orders(self, client: TestClient, sample_supplier: Dict[str, Any]):
        """Test retrieving supplier with purchase orders."""
        # Create purchase order for supplier
        po_data = TestDataFactory.purchase_order_payload(sample_supplier["supplier_id"])
        client.post("/api/v1/purchase-orders/", json=po_data)
        
        # Get supplier with orders
        supplier_id = sample_supplier["supplier_id"]
        response = client.get(f"/api/v1/suppliers/{supplier_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["supplier_id"] == supplier_id
        assert "purchase_orders" in data

class TestPurchaseOrderAPI:
    """Integration tests for Purchase Order API endpoints."""
    
    def test_create_purchase_order_success(self, client: TestClient, sample_supplier: Dict[str, Any]):
        """Test successful purchase order creation."""
        po_data = TestDataFactory.purchase_order_payload(sample_supplier["supplier_id"])
        response = client.post("/api/v1/purchase-orders/", json=po_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["po_number"] == po_data["po_number"]
        assert data["supplier_id"] == po_data["supplier_id"]
        assert data["status"] == po_data["status"]
    
    def test_get_purchase_order_with_line_items(self, client: TestClient, sample_supplier: Dict[str, Any]):
        """Test retrieving purchase order with line items."""
        # Create purchase order
        po_data = TestDataFactory.purchase_order_payload(sample_supplier["supplier_id"])
        po_response = client.post("/api/v1/purchase-orders/", json=po_data)
        assert po_response.status_code == 201
        
        # Get PO with line items
        po_number = po_data["po_number"]
        response = client.get(f"/api/v1/purchase-orders/{po_number}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["po_number"] == po_number
        assert "line_items" in data
        assert "supplier" in data

class TestBusinessLogicIntegration:
    """Integration tests for business logic workflows."""
    
    def test_full_inventory_lifecycle(self, client: TestClient):
        """Test complete inventory lifecycle from product creation to consumption."""
        # Create product
        product_data = TestDataFactory.product_payload("LIFECYCLE-PROD")
        product_response = client.post("/api/v1/products/", json=product_data)
        assert product_response.status_code == 201
        
        # Create storage location
        location_data = TestDataFactory.storage_location_payload("LIFECYCLE-BIN")
        location_response = client.post("/api/v1/storage/", json=location_data)
        assert location_response.status_code == 201
        
        # Create multiple batches with FIFO dates
        batch_numbers = []
        for i in range(3):
            batch_data = TestDataFactory.batch_payload(
                "LIFECYCLE-PROD",
                "LIFECYCLE-BIN",
                f"LIFECYCLE-BATCH-{i+1:03d}"
            )
            # Adjust receipt dates for FIFO testing
            receipt_date = datetime.now() - timedelta(days=2-i)
            batch_data["goods_receipt_date"] = receipt_date.isoformat()
            batch_data["quantity"] = 100.0 + (i * 50)  # 100, 150, 200
            
            batch_response = client.post("/api/v1/inventory/batches/", json=batch_data)
            assert batch_response.status_code == 201
            batch_numbers.append(batch_data["batch_number"])
        
        # Verify total available stock
        # Note: This would require a stock availability endpoint
        # For now, we'll verify by checking individual batches
        total_stock = 0
        for batch_number in batch_numbers:
            batch_response = client.get(f"/api/v1/inventory/batches/{batch_number}")
            total_stock += batch_response.json()["quantity"]
        
        assert total_stock == 450.0  # 100 + 150 + 200

class TestErrorHandlingAndValidation:
    """Integration tests for error handling and validation scenarios."""
    
    def test_validation_error_responses(self, client: TestClient):
        """Test API validation error handling."""
        # Test missing required fields
        invalid_product = {"unit_of_measurement": "KG"}  # Missing product_code
        response = client.post("/api/v1/products/", json=invalid_product)
        
        assert response.status_code == 422
        error_detail = response.json()
        assert "product_code" in str(error_detail)
        
        # Test invalid enum values
        invalid_enum_product = {
            "product_code": "TEST-INVALID-ENUM",
            "unit_of_measurement": "INVALID_UNIT"
        }
        response = client.post("/api/v1/products/", json=invalid_enum_product)
        assert response.status_code == 422
    
    def test_business_logic_errors(self, client: TestClient):
        """Test business logic error scenarios."""
        # Test creating batch for non-existent product
        invalid_batch = TestDataFactory.batch_payload("NON_EXISTENT_PRODUCT", "TEST-BIN")
        response = client.post("/api/v1/inventory/batches/", json=invalid_batch)
        
        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()
    
    def test_database_constraint_violations(self, client: TestClient):
        """Test database constraint handling."""
        # Test duplicate product creation
        product_data = TestDataFactory.product_payload("CONSTRAINT-TEST")
        
        # First creation should succeed
        response1 = client.post("/api/v1/products/", json=product_data)
        assert response1.status_code == 201
        
        # Second creation should fail
        response2 = client.post("/api/v1/products/", json=product_data)
        assert response2.status_code == 409

class TestPerformanceAndScalability:
    """Integration tests for performance and scalability scenarios."""
    
    def test_large_dataset_creation_performance(self, client: TestClient):
        """Test API performance with large dataset creation."""
        start_time = time.time()
        
        # Create 100 products
        for i in range(100):
            product_data = TestDataFactory.product_payload(f"PERF-PROD-{i:03d}")
            response = client.post("/api/v1/products/", json=product_data)
            assert response.status_code == 201
        
        creation_time = time.time() - start_time
        assert creation_time < 15.0  # Should complete within 15 seconds
    
    def test_listing_performance_with_pagination(self, client: TestClient):
        """Test listing performance with pagination."""
        # Create test data
        for i in range(50):
            product_data = TestDataFactory.product_payload(f"LIST-PERF-{i:03d}")
            client.post("/api/v1/products/", json=product_data)
        
        # Test listing performance
        start_time = time.time()
        response = client.get("/api/v1/products/?limit=50")
        listing_time = time.time() - start_time
        
        assert response.status_code == 200
        assert listing_time < 2.0  # Should complete within 2 seconds
    
    def test_concurrent_operations_simulation(self, client: TestClient):
        """Test API behavior under simulated concurrent operations."""
        # Simulate concurrent product creation
        products = []
        for i in range(20):
            product_data = TestDataFactory.product_payload(f"CONCURRENT-{i:03d}")
            response = client.post("/api/v1/products/", json=product_data)
            assert response.status_code == 201
            products.append(response.json())
        
        # Verify all products were created successfully
        assert len(products) == 20
        product_codes = [p["product_code"] for p in products]
        assert len(set(product_codes)) == 20  # All unique

class TestHealthAndMonitoring:
    """Integration tests for health check and monitoring endpoints."""
    
    def test_health_check_endpoint(self, client: TestClient):
        """Test application health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert "timestamp" in data
        assert "checks" in data
    
    def test_api_info_endpoint(self, client: TestClient):
        """Test API information endpoint."""
        response = client.get("/api/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "api_version" in data
        assert "service" in data
        assert "features" in data
        assert "endpoints" in data

# Test execution order and dependencies
class TestExecutionOrder:
    """Tests that verify proper execution order and dependencies."""
    
    def test_dependency_chain_creation(self, client: TestClient):
        """Test creating entities in proper dependency order."""
        # 1. Create product (no dependencies)
        product_data = TestDataFactory.product_payload("CHAIN-PROD")
        product_response = client.post("/api/v1/products/", json=product_data)
        assert product_response.status_code == 201
        
        # 2. Create storage location (no dependencies)
        location_data = TestDataFactory.storage_location_payload("CHAIN-BIN")
        location_response = client.post("/api/v1/storage/", json=location_data)
        assert location_response.status_code == 201
        
        # 3. Create supplier (no dependencies)
        supplier_data = TestDataFactory.supplier_payload("CHAIN-SUP")
        supplier_response = client.post("/api/v1/suppliers/", json=supplier_data)
        assert supplier_response.status_code == 201
        
        # 4. Create batch (depends on product and storage location)
        batch_data = TestDataFactory.batch_payload("CHAIN-PROD", "CHAIN-BIN", "CHAIN-BATCH")
        batch_response = client.post("/api/v1/inventory/batches/", json=batch_data)
        assert batch_response.status_code == 201
        
        # 5. Create transaction (depends on batch)
        transaction_data = TestDataFactory.transaction_payload("CHAIN-BATCH", 25.0)
        transaction_response = client.post("/api/v1/inventory/transactions/", json=transaction_data)
        assert transaction_response.status_code == 201
        
        # 6. Create purchase order (depends on supplier)
        po_data = TestDataFactory.purchase_order_payload("CHAIN-SUP", "CHAIN-PO")
        po_response = client.post("/api/v1/purchase-orders/", json=po_data)
        assert po_response.status_code == 201
        
        # Verify all entities were created successfully
        assert all([
            product_response.status_code == 201,
            location_response.status_code == 201,
            supplier_response.status_code == 201,
            batch_response.status_code == 201,
            transaction_response.status_code == 201,
            po_response.status_code == 201
        ])
