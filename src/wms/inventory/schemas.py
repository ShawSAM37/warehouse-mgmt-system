"""
Pydantic schemas for API request/response validation and serialization.
These schemas define the data models for the warehouse management system API.
"""
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from decimal import Decimal

# Import enums from the existing enums.py file
from ..enums import (
    StorageType, StockType, UnitOfMeasurement, DocumentCategory,
    TransactionType, POStatus, QualityCheckResult, AnomalyStatus,
    SeverityLevel, AnomalyType, InspectionType
)

# ===== BASE SCHEMAS =====

class ProductBase(BaseModel):
    """Base schema for Product."""
    product_code: str = Field(..., max_length=20, description="Unique product identifier")
    handling_unit: Optional[str] = Field(None, max_length=20, description="Handling unit ID")
    unit_of_measurement: UnitOfMeasurement = Field(..., description="Unit of measurement")
    description: Optional[str] = Field(None, max_length=255, description="Product description")
    hsn_sac_code: Optional[str] = Field(None, pattern=r'^\d{4,8}$', description="HSN/SAC code")
    default_shelf_life_days: Optional[int] = Field(None, gt=0, description="Default shelf life in days")
    weight_per_unit: Optional[float] = Field(None, gt=0, description="Weight per unit in kg")
    volume_per_unit: Optional[float] = Field(None, gt=0, description="Volume per unit in liters")


class StorageLocationBase(BaseModel):
    """Base schema for Storage Location."""
    storage_bin: str = Field(..., max_length=10, description="Storage bin identifier")
    storage_type: StorageType = Field(..., description="Type of storage")
    capacity_total: float = Field(..., gt=0, description="Total capacity in cubic meters")
    capacity_available: float = Field(..., gt=0, description="Available capacity in cubic meters")
    zone: Optional[str] = Field(None, max_length=50, description="Storage zone")
    aisle: Optional[str] = Field(None, max_length=10, description="Aisle identifier")
    rack: Optional[str] = Field(None, max_length=10, description="Rack identifier")
    level: Optional[str] = Field(None, max_length=10, description="Level identifier")

    @field_validator('capacity_available')
    @classmethod
    def validate_capacity(cls, v, info):
        """Ensure available capacity doesn't exceed total capacity."""
        if 'capacity_total' in info.data and v > info.data['capacity_total']:
            raise ValueError("Available capacity cannot exceed total capacity")
        return v


class SupplierBase(BaseModel):
    """Base schema for Supplier."""
    supplier_id: str = Field(..., max_length=20, description="Unique supplier identifier")
    supplier_name: str = Field(..., max_length=100, description="Supplier name")
    partner_number: Optional[str] = Field(None, max_length=20, description="Partner number")
    contact_person: Optional[str] = Field(None, max_length=100, description="Contact person name")
    email: Optional[str] = Field(None, pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', description="Email address")
    phone: Optional[str] = Field(None, pattern=r'^\+?[0-9\-\s]{8,15}$', description="Phone number")
    address: Optional[str] = Field(None, max_length=255, description="Address")


class PurchaseOrderBase(BaseModel):
    """Base schema for Purchase Order."""
    po_number: str = Field(..., max_length=20, pattern=r'^PO-\d{5,6}$', description="Purchase order number")
    supplier_id: str = Field(..., max_length=20, description="Supplier identifier")
    order_date: datetime = Field(..., description="Order date")
    expected_delivery_date: Optional[datetime] = Field(None, description="Expected delivery date")
    status: POStatus = Field(default=POStatus.OPEN, description="Purchase order status")


class PurchaseOrderLineItemBase(BaseModel):
    """Base schema for Purchase Order Line Item."""
    po_line_item_id: str = Field(..., max_length=30, description="Line item identifier")
    po_number: str = Field(..., max_length=20, description="Purchase order number")
    product_code: str = Field(..., max_length=20, description="Product code")
    ordered_quantity: float = Field(..., gt=0, description="Ordered quantity")
    unit_price: Optional[float] = Field(None, gt=0, description="Unit price")
    received_quantity: float = Field(default=0, ge=0, description="Received quantity")
    status: POStatus = Field(default=POStatus.OPEN, description="Line item status")


class InventoryBatchBase(BaseModel):
    """Base schema for Inventory Batch."""
    batch_number: str = Field(..., max_length=20, pattern=r'^\d{8}-[A-Z]{3}-\d{4}$', description="Batch number")
    product_code: str = Field(..., max_length=20, description="Product code")
    storage_type: StorageType = Field(..., description="Storage type")
    storage_bin: str = Field(..., max_length=10, description="Storage bin")
    quantity: float = Field(..., gt=0, description="Quantity")
    document_number: Optional[str] = Field(None, max_length=20, description="Document number")
    unit_of_measurement: UnitOfMeasurement = Field(..., description="Unit of measurement")
    stock_type: StockType = Field(default=StockType.UNRESTRICTED, description="Stock type")
    goods_receipt_date: datetime = Field(..., description="Goods receipt date")
    goods_receipt_time: Optional[datetime] = Field(None, description="Goods receipt time")
    restricted_use: bool = Field(default=False, description="Restricted use flag")
    country_of_origin: Optional[str] = Field(None, pattern=r'^[A-Z]{2}$', description="Country of origin (ISO 3166-1 alpha-2)")
    sale_order_number: Optional[str] = Field(None, max_length=20, description="Sale order number")
    sale_order_line_number: Optional[str] = Field(None, max_length=20, description="Sale order line number")
    inspection_id_type: Optional[str] = Field(None, max_length=20, description="Inspection ID type")
    quality_inspection_number: Optional[str] = Field(None, max_length=20, description="Quality inspection number")
    expiry_date: Optional[datetime] = Field(None, description="Expiry date")
    handling_unit_id: Optional[str] = Field(None, max_length=20, description="Handling unit ID")
    higher_level_hu: Optional[str] = Field(None, max_length=20, description="Higher level handling unit")
    highest_level_hu: Optional[str] = Field(None, max_length=20, description="Highest level handling unit")
    packet_quantity: Optional[float] = Field(None, description="Packet quantity")
    quantity_as_shipped: Optional[float] = Field(None, description="Quantity as shipped")
    document_category: Optional[DocumentCategory] = Field(None, description="Document category")
    loading_weight: Optional[float] = Field(None, description="Loading weight")
    loading_volume: Optional[float] = Field(None, description="Loading volume")
    capacity_consumed: Optional[float] = Field(None, description="Capacity consumed")
    capacity_left: Optional[float] = Field(None, description="Capacity left")
    serial_number: Optional[str] = Field(None, max_length=50, description="Serial number")

    @field_validator('expiry_date')
    @classmethod
    def validate_expiry_date(cls, v, info):
        """Ensure expiry date is after goods receipt date."""
        if v is not None and 'goods_receipt_date' in info.data:
            if v < info.data['goods_receipt_date']:
                raise ValueError("Expiry date must be after goods receipt date")
        return v


class InventoryTransactionBase(BaseModel):
    """Base schema for Inventory Transaction."""
    transaction_id: str = Field(..., max_length=20, pattern=r'^TRX-\d{8}-\d{3}$', description="Transaction ID")
    batch_number: str = Field(..., max_length=20, description="Batch number")
    transaction_type: TransactionType = Field(..., description="Transaction type")
    quantity: float = Field(..., gt=0, description="Transaction quantity")
    transaction_date: datetime = Field(..., description="Transaction date")
    reference_document: Optional[str] = Field(None, max_length=20, description="Reference document")
    from_location: Optional[str] = Field(None, max_length=10, description="From location")
    to_location: Optional[str] = Field(None, max_length=10, description="To location")
    performed_by: Optional[str] = Field(None, max_length=50, description="Performed by user")
    notes: Optional[str] = Field(None, description="Transaction notes")


class QualityCheckBase(BaseModel):
    """Base schema for Quality Check."""
    inspection_id: str = Field(..., max_length=20, pattern=r'^QC-\d{5,6}$', description="Inspection ID")
    batch_number: str = Field(..., max_length=20, description="Batch number")
    inspection_type: InspectionType = Field(..., description="Inspection type")
    inspection_date: datetime = Field(..., description="Inspection date")
    inspector: Optional[str] = Field(None, max_length=50, description="Inspector name")
    result: QualityCheckResult = Field(default=QualityCheckResult.PENDING, description="Inspection result")
    notes: Optional[str] = Field(None, description="Inspection notes")


class AnomalyDetectionBase(BaseModel):
    """Base schema for Anomaly Detection."""
    anomaly_id: str = Field(..., max_length=20, pattern=r'^ANM-\d{5,6}$', description="Anomaly ID")
    detection_timestamp: datetime = Field(..., description="Detection timestamp")
    anomaly_type: AnomalyType = Field(..., description="Anomaly type")
    related_entity_id: Optional[str] = Field(None, max_length=20, description="Related entity ID")
    severity: SeverityLevel = Field(..., description="Severity level")
    description: str = Field(..., description="Anomaly description")
    status: AnomalyStatus = Field(default=AnomalyStatus.NEW, description="Anomaly status")


class ForecastResultBase(BaseModel):
    """Base schema for Forecast Result."""
    forecast_id: str = Field(..., max_length=20, pattern=r'^FC-\d{8}-\d{3}$', description="Forecast ID")
    product_code: str = Field(..., max_length=20, description="Product code")
    forecast_date: datetime = Field(..., description="Forecast date")
    forecast_period: str = Field(..., max_length=20, description="Forecast period")
    predicted_quantity: float = Field(..., ge=0, description="Predicted quantity")
    confidence_interval_lower: float = Field(..., ge=0, description="Lower confidence interval")
    confidence_interval_upper: float = Field(..., ge=0, description="Upper confidence interval")
    model_version: str = Field(..., max_length=50, description="Model version")
    model_accuracy: Optional[float] = Field(None, ge=0, le=1, description="Model accuracy")


class InventorySnapshotBase(BaseModel):
    """Base schema for Inventory Snapshot."""
    snapshot_id: str = Field(..., max_length=30, description="Snapshot ID")
    snapshot_date: date = Field(..., description="Snapshot date")
    product_code: str = Field(..., max_length=20, description="Product code")
    total_quantity: float = Field(..., ge=0, description="Total quantity")
    bins_count: int = Field(..., ge=0, description="Number of bins")
    storage_types: Dict[str, float] = Field(..., description="Storage type distribution")
    oldest_batch_date: datetime = Field(..., description="Oldest batch date")
    nearest_expiry_date: Optional[datetime] = Field(None, description="Nearest expiry date")
    days_of_supply: Optional[float] = Field(None, ge=0, description="Days of supply")


# ===== CREATE SCHEMAS =====

class ProductCreate(ProductBase):
    """Schema for creating a product."""
    pass


class StorageLocationCreate(StorageLocationBase):
    """Schema for creating a storage location."""
    pass


class SupplierCreate(SupplierBase):
    """Schema for creating a supplier."""
    pass


class PurchaseOrderCreate(PurchaseOrderBase):
    """Schema for creating a purchase order."""
    pass


class PurchaseOrderLineItemCreate(PurchaseOrderLineItemBase):
    """Schema for creating a purchase order line item."""
    pass


class InventoryBatchCreate(InventoryBatchBase):
    """Schema for creating an inventory batch."""
    pass


class InventoryTransactionCreate(InventoryTransactionBase):
    """Schema for creating an inventory transaction."""
    pass


class QualityCheckCreate(QualityCheckBase):
    """Schema for creating a quality check."""
    pass


class AnomalyDetectionCreate(AnomalyDetectionBase):
    """Schema for creating an anomaly detection."""
    pass


class ForecastResultCreate(ForecastResultBase):
    """Schema for creating a forecast result."""
    pass


class InventorySnapshotCreate(InventorySnapshotBase):
    """Schema for creating an inventory snapshot."""
    pass


# ===== UPDATE SCHEMAS =====

class ProductUpdate(BaseModel):
    """Schema for updating a product."""
    handling_unit: Optional[str] = Field(None, max_length=20)
    description: Optional[str] = Field(None, max_length=255)
    hsn_sac_code: Optional[str] = Field(None, pattern=r'^\d{4,8}$')
    default_shelf_life_days: Optional[int] = Field(None, gt=0)
    weight_per_unit: Optional[float] = Field(None, gt=0)
    volume_per_unit: Optional[float] = Field(None, gt=0)
    is_active: Optional[bool] = None


class InventoryBatchUpdate(BaseModel):
    """Schema for updating an inventory batch."""
    quantity: Optional[float] = Field(None, gt=0)
    stock_type: Optional[StockType] = None
    restricted_use: Optional[bool] = None
    quality_inspection_number: Optional[str] = Field(None, max_length=20)
    serial_number: Optional[str] = Field(None, max_length=50)
    is_active: Optional[bool] = None


class QualityCheckUpdate(BaseModel):
    """Schema for updating a quality check."""
    result: Optional[QualityCheckResult] = None
    notes: Optional[str] = None


class AnomalyDetectionUpdate(BaseModel):
    """Schema for updating an anomaly detection."""
    status: Optional[AnomalyStatus] = None
    description: Optional[str] = None


# ===== RESPONSE SCHEMAS =====

class Product(ProductBase):
    """Schema for product response."""
    is_active: bool = True
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class StorageLocation(StorageLocationBase):
    """Schema for storage location response."""
    is_active: bool = True
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class Supplier(SupplierBase):
    """Schema for supplier response."""
    is_active: bool = True
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PurchaseOrder(PurchaseOrderBase):
    """Schema for purchase order response."""
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PurchaseOrderLineItem(PurchaseOrderLineItemBase):
    """Schema for purchase order line item response."""
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class InventoryBatch(InventoryBatchBase):
    """Schema for inventory batch response."""
    is_active: bool = True
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class InventoryTransaction(InventoryTransactionBase):
    """Schema for inventory transaction response."""
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class QualityCheck(QualityCheckBase):
    """Schema for quality check response."""
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AnomalyDetection(AnomalyDetectionBase):
    """Schema for anomaly detection response."""
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ForecastResult(ForecastResultBase):
    """Schema for forecast result response."""
    created_at: datetime

    model_config = {"from_attributes": True}


class InventorySnapshot(InventorySnapshotBase):
    """Schema for inventory snapshot response."""
    created_at: datetime

    model_config = {"from_attributes": True}


# ===== NESTED RESPONSE SCHEMAS =====

class ProductWithBatches(Product):
    """Product schema with related batches."""
    batches: List[InventoryBatch] = []


class InventoryBatchWithTransactions(InventoryBatch):
    """Inventory batch schema with related transactions."""
    transactions: List[InventoryTransaction] = []
    quality_checks: List[QualityCheck] = []


class PurchaseOrderWithLineItems(PurchaseOrder):
    """Purchase order schema with line items."""
    line_items: List[PurchaseOrderLineItem] = []
    supplier: Optional[Supplier] = None


class StorageLocationWithBatches(StorageLocation):
    """Storage location schema with batches."""
    batches: List[InventoryBatch] = []


class SupplierWithOrders(Supplier):
    """Supplier schema with purchase orders."""
    purchase_orders: List[PurchaseOrder] = []


# ===== UTILITY SCHEMAS =====

class PaginationMeta(BaseModel):
    """Pagination metadata."""
    page: int = Field(..., ge=1, description="Current page number")
    per_page: int = Field(..., ge=1, le=100, description="Items per page")
    total: int = Field(..., ge=0, description="Total number of items")
    pages: int = Field(..., ge=0, description="Total number of pages")


class PaginatedResponse(BaseModel):
    """Generic paginated response."""
    data: List[Any] = Field(..., description="List of items")
    meta: PaginationMeta = Field(..., description="Pagination metadata")


class HealthCheck(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy", description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    service: str = Field(default="warehouse-management-system", description="Service name")
    version: str = Field(default="1.0.0", description="Service version")


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    path: Optional[str] = Field(None, description="Request path")


class BulkOperationResponse(BaseModel):
    """Response for bulk operations."""
    success_count: int = Field(..., ge=0, description="Number of successful operations")
    error_count: int = Field(..., ge=0, description="Number of failed operations")
    errors: List[str] = Field(default_factory=list, description="List of error messages")


class InventorySummary(BaseModel):
    """Summary of inventory statistics."""
    total_products: int = Field(..., ge=0, description="Total number of products")
    total_batches: int = Field(..., ge=0, description="Total number of batches")
    total_quantity: float = Field(..., ge=0, description="Total quantity in stock")
    expiring_soon: int = Field(..., ge=0, description="Number of batches expiring within 30 days")
    low_stock_products: int = Field(..., ge=0, description="Number of products with low stock")


class DashboardStats(BaseModel):
    """Dashboard statistics."""
    inventory_summary: InventorySummary
    recent_transactions: List[InventoryTransaction] = []
    pending_quality_checks: List[QualityCheck] = []
    active_anomalies: List[AnomalyDetection] = []
    open_purchase_orders: int = Field(..., ge=0, description="Number of open purchase orders")


# ===== FILTER SCHEMAS =====

class ProductFilter(BaseModel):
    """Filter schema for products."""
    product_code: Optional[str] = Field(None, description="Filter by product code")
    unit_of_measurement: Optional[UnitOfMeasurement] = Field(None, description="Filter by unit of measurement")
    is_active: Optional[bool] = Field(None, description="Filter by active status")
    created_after: Optional[datetime] = Field(None, description="Filter by creation date")
    created_before: Optional[datetime] = Field(None, description="Filter by creation date")


class InventoryBatchFilter(BaseModel):
    """Filter schema for inventory batches."""
    product_code: Optional[str] = Field(None, description="Filter by product code")
    storage_type: Optional[StorageType] = Field(None, description="Filter by storage type")
    storage_bin: Optional[str] = Field(None, description="Filter by storage bin")
    stock_type: Optional[StockType] = Field(None, description="Filter by stock type")
    expiring_before: Optional[datetime] = Field(None, description="Filter by expiry date")
    is_active: Optional[bool] = Field(None, description="Filter by active status")


class InventoryTransactionFilter(BaseModel):
    """Filter schema for inventory transactions."""
    batch_number: Optional[str] = Field(None, description="Filter by batch number")
    transaction_type: Optional[TransactionType] = Field(None, description="Filter by transaction type")
    date_from: Optional[datetime] = Field(None, description="Filter by transaction date")
    date_to: Optional[datetime] = Field(None, description="Filter by transaction date")
    performed_by: Optional[str] = Field(None, description="Filter by user")


class QualityCheckFilter(BaseModel):
    """Filter schema for quality checks."""
    batch_number: Optional[str] = Field(None, description="Filter by batch number")
    inspection_type: Optional[InspectionType] = Field(None, description="Filter by inspection type")
    result: Optional[QualityCheckResult] = Field(None, description="Filter by result")
    inspector: Optional[str] = Field(None, description="Filter by inspector")
    date_from: Optional[datetime] = Field(None, description="Filter by inspection date")
    date_to: Optional[datetime] = Field(None, description="Filter by inspection date")


class AnomalyDetectionFilter(BaseModel):
    """Filter schema for anomaly detections."""
    anomaly_type: Optional[AnomalyType] = Field(None, description="Filter by anomaly type")
    severity: Optional[SeverityLevel] = Field(None, description="Filter by severity")
    status: Optional[AnomalyStatus] = Field(None, description="Filter by status")
    related_entity_id: Optional[str] = Field(None, description="Filter by related entity")
    date_from: Optional[datetime] = Field(None, description="Filter by detection date")
    date_to: Optional[datetime] = Field(None, description="Filter by detection date")
