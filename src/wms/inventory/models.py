"""
Core data models for the warehouse management system.
These models define the fundamental entities tracked in the system.
"""
import re
from pydantic import BaseModel, Field, field_validator, EmailStr
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class StorageType(str, Enum):
    """Valid storage types in the warehouse."""
    COLD_STORAGE = "Cold Storage"
    AMBIENT = "Ambient"
    RACK = "Rack"
    BULK = "Bulk"


class StockType(str, Enum):
    """Valid stock types."""
    UNRESTRICTED = "Unrestricted"
    QUALITY_INSPECTION = "Quality Inspection"
    BLOCKED = "Blocked"


class UnitOfMeasurement(str, Enum):
    """Valid units of measurement."""
    KG = "KG"
    L = "L"
    UNITS = "Units"
    BOX = "Box"


class DocumentCategory(str, Enum):
    """Valid document categories."""
    PDI = "PDI"
    PWR = "PWR"


class TransactionType(str, Enum):
    """Valid inventory transaction types."""
    RECEIPT = "RECEIPT"
    CONSUMPTION = "CONSUMPTION"
    TRANSFER = "TRANSFER"
    ADJUSTMENT = "ADJUSTMENT"
    RETURN = "RETURN"


class POStatus(str, Enum):
    """Valid purchase order statuses."""
    OPEN = "Open"
    PARTIALLY_RECEIVED = "Partially Received"
    CLOSED = "Closed"


class QualityCheckResult(str, Enum):
    """Valid quality check results."""
    PASS = "Pass"
    FAIL = "Fail"
    PENDING = "Pending"


class AnomalyStatus(str, Enum):
    """Valid anomaly statuses."""
    NEW = "New"
    INVESTIGATING = "Investigating"
    RESOLVED = "Resolved"


class SeverityLevel(str, Enum):
    """Valid severity levels for anomalies and alerts."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class AnomalyType(str, Enum):
    """Valid anomaly types."""
    STOCKOUT_RISK = "Stockout Risk"
    EXPIRY_RISK = "Expiry Risk"
    CONSUMPTION_SPIKE = "Consumption Spike"
    CONSUMPTION_DROP = "Consumption Drop"
    QUALITY_ISSUE = "Quality Issue"
    SPACE_CONSTRAINT = "Space Constraint"
    DELIVERY_DELAY = "Delivery Delay"
    UNEXPECTED_RECEIPT = "Unexpected Receipt"


class InspectionType(str, Enum):
    """Valid inspection types."""
    RECEIPT = "Receipt Inspection"
    RANDOM = "Random Sampling"
    COMPLAINT = "Complaint Investigation"
    EXPIRY = "Expiry Check"


class ActiveStatus(str, Enum):
    """Standardized active status values."""
    ACTIVE = "Active"
    INACTIVE = "Inactive"


class Product(BaseModel):
    """
    Product master data.
    
    Relationships:
    - product_code: Referenced by InventoryBatch.product_code
    - product_code: Referenced by PurchaseOrderLineItem.product_code
    """
    product_code: str = Field(description="Unique product identifier")
    handling_unit: Optional[str] = Field(None, description="ID if Raw material, else OEM consumable items")
    unit_of_measurement: UnitOfMeasurement
    description: Optional[str] = None
    hsn_sac_code: Optional[str] = None
    default_shelf_life_days: Optional[int] = Field(None, gt=0)
    weight_per_unit: Optional[float] = Field(None, gt=0)
    volume_per_unit: Optional[float] = Field(None, gt=0)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @field_validator('hsn_sac_code')
    @classmethod
    def validate_hsn_code(cls, v: Optional[str]) -> Optional[str]:
        """Validate HSN/SAC code format."""
        if v is not None and not re.match(r'^\d{4,8}$', v):
            raise ValueError("HSN/SAC code must be 4-8 digits")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "product_code": "PROD-0001",
                    "handling_unit": "RM-12345",
                    "unit_of_measurement": "KG",
                    "description": "Raw Steel Sheet",
                    "hsn_sac_code": "7208",
                    "default_shelf_life_days": 365
                }
            ]
        }
    }


class InventoryBatch(BaseModel):
    """
    Represents a batch of inventory in the warehouse.
    
    Relationships:
    - product_code: Foreign key to Product.product_code
    - storage_bin: Foreign key to StorageLocation.storage_bin
    - batch_number: Referenced by InventoryTransaction.batch_number
    - batch_number: Referenced by QualityCheck.batch_number
    """
    batch_number: str = Field(description="Unique batch identifier")
    product_code: str = Field(description="Product code this batch contains")
    storage_type: StorageType
    storage_bin: str = Field(description="Physical location in warehouse")
    quantity: float = Field(gt=0)
    document_number: Optional[str] = Field(None, description="IBD/OBD number")
    unit_of_measurement: UnitOfMeasurement
    stock_type: StockType = StockType.UNRESTRICTED
    goods_receipt_date: datetime
    goods_receipt_time: Optional[datetime] = None
    restricted_use: bool = False
    country_of_origin: Optional[str] = None
    sale_order_number: Optional[str] = None
    sale_order_line_number: Optional[str] = None
    inspection_id_type: Optional[str] = None
    quality_inspection_number: Optional[str] = None
    expiry_date: Optional[datetime] = None
    is_active: bool = True  # Standardized field name
    handling_unit_id: Optional[str] = None
    higher_level_hu: Optional[str] = None
    highest_level_hu: Optional[str] = None
    packet_quantity: Optional[float] = None
    quantity_as_shipped: Optional[float] = None
    document_category: Optional[DocumentCategory] = None
    loading_weight: Optional[float] = None
    loading_volume: Optional[float] = None
    capacity_consumed: Optional[float] = None
    capacity_left: Optional[float] = None
    serial_number: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @field_validator('expiry_date')
    @classmethod
    def expiry_date_must_be_future(cls, v: Optional[datetime], info) -> Optional[datetime]:
        """Validate that expiry date is in the future."""
        if v is not None:
            goods_receipt_date = info.data.get('goods_receipt_date')
            if goods_receipt_date and v < goods_receipt_date:
                raise ValueError("Expiry date must be after goods receipt date")
        return v

    @field_validator('country_of_origin')
    @classmethod
    def validate_country_code(cls, v: Optional[str]) -> Optional[str]:
        """Validate country code format (ISO 3166-1 alpha-2)."""
        if v is not None and not re.match(r'^[A-Z]{2}$', v):
            raise ValueError("Country code must be a valid ISO 3166-1 alpha-2 code")
        return v
        
    @field_validator('batch_number')
    @classmethod
    def validate_batch_number(cls, v: str) -> str:
        """Validate batch number format."""
        # Example format: 20250522-RAW-0001
        if not re.match(r'^\d{8}-[A-Z]{3}-\d{4}$', v):
            raise ValueError("Batch number must be in format YYYYMMDD-XXX-####")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "batch_number": "20250522-RAW-0001",
                    "product_code": "PROD-0001",
                    "storage_type": "Ambient",
                    "storage_bin": "A01",
                    "quantity": 100.0,
                    "document_number": "IBD-12345",
                    "unit_of_measurement": "KG",
                    "stock_type": "Unrestricted",
                    "goods_receipt_date": "2025-05-22T10:30:00",
                    "country_of_origin": "IN",
                    "expiry_date": "2026-05-22T00:00:00"
                }
            ]
        }
    }


class InventoryTransaction(BaseModel):
    """
    Records movement of inventory (receipt, consumption, transfer).
    
    Relationships:
    - batch_number: Foreign key to InventoryBatch.batch_number
    - from_location: Foreign key to StorageLocation.storage_bin (optional)
    - to_location: Foreign key to StorageLocation.storage_bin (optional)
    """
    transaction_id: str = Field(description="Unique transaction identifier")
    batch_number: str
    transaction_type: TransactionType
    quantity: float = Field(gt=0)
    transaction_date: datetime = Field(default_factory=datetime.now)
    reference_document: Optional[str] = None
    from_location: Optional[str] = None
    to_location: Optional[str] = None
    performed_by: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @field_validator('transaction_id')
    @classmethod
    def validate_transaction_id(cls, v: str) -> str:
        """Validate transaction ID format."""
        # Example format: TRX-20250522-001
        if not re.match(r'^TRX-\d{8}-\d{3}$', v):
            raise ValueError("Transaction ID must be in format TRX-YYYYMMDD-###")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "transaction_id": "TRX-20250522-001",
                    "batch_number": "20250522-RAW-0001",
                    "transaction_type": "RECEIPT",
                    "quantity": 100.0,
                    "reference_document": "GR-12345",
                    "to_location": "A01"
                }
            ]
        }
    }


class StorageLocation(BaseModel):
    """
    Represents a physical storage location in the warehouse.
    
    Relationships:
    - storage_bin: Referenced by InventoryBatch.storage_bin
    - storage_bin: Referenced by InventoryTransaction.from_location/to_location
    """
    storage_type: StorageType
    storage_bin: str = Field(description="Unique bin identifier")
    capacity_total: float = Field(gt=0, description="Total capacity in cubic meters")
    capacity_available: float = Field(gt=0, description="Available capacity in cubic meters")
    is_active: bool = True
    zone: Optional[str] = None
    aisle: Optional[str] = None
    rack: Optional[str] = None
    level: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @field_validator('capacity_available')
    @classmethod
    def capacity_available_must_be_valid(cls, v: float, info) -> float:
        """Validate that available capacity is not greater than total."""
        capacity_total = info.data.get('capacity_total')
        if capacity_total and v > capacity_total:
            raise ValueError("Available capacity cannot exceed total capacity")
        return v
        
    @field_validator('storage_bin')
    @classmethod
    def validate_storage_bin(cls, v: str) -> str:
        """Validate storage bin format."""
        # Example format: A01-1
        if not re.match(r'^[A-Z]\d{2}(-\d)?$', v):
            raise ValueError("Storage bin must be in format X##(-#)")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "storage_type": "Rack",
                    "storage_bin": "A01",
                    "capacity_total": 10.0,
                    "capacity_available": 8.5,
                    "zone": "Inbound",
                    "aisle": "A",
                    "rack": "01",
                    "level": "1"
                }
            ]
        }
    }


class Supplier(BaseModel):
    """
    Supplier master data.
    
    Relationships:
    - supplier_id: Referenced by PurchaseOrder.supplier_id
    """
    supplier_id: str = Field(description="Unique supplier identifier")
    supplier_name: str
    partner_number: Optional[str] = None
    contact_person: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        """Validate email format."""
        if v is not None and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError("Invalid email format")
        return v
    
    @field_validator('phone')
    @classmethod
    def validate_phone(cls, v: Optional[str]) -> Optional[str]:
        """Validate phone number format."""
        if v is not None and not re.match(r'^\+?[0-9\-\s]{8,15}$', v):
            raise ValueError("Invalid phone number format")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "supplier_id": "SUP-001",
                    "supplier_name": "ABC Materials Ltd.",
                    "partner_number": "P12345",
                    "contact_person": "John Doe",
                    "email": "john.doe@abcmaterials.com"
                }
            ]
        }
    }


class PurchaseOrder(BaseModel):
    """
    Purchase order information.
    
    Relationships:
    - supplier_id: Foreign key to Supplier.supplier_id
    - po_number: Referenced by PurchaseOrderLineItem.po_number
    """
    po_number: str = Field(description="Unique purchase order number")
    supplier_id: str
    order_date: datetime
    expected_delivery_date: Optional[datetime] = None
    status: POStatus = POStatus.OPEN
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @field_validator('po_number')
    @classmethod
    def validate_po_number(cls, v: str) -> str:
        """Validate PO number format."""
        # Example format: PO-12345
        if not re.match(r'^PO-\d{5,6}$', v):
            raise ValueError("PO number must be in format PO-#####")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "po_number": "PO-12345",
                    "supplier_id": "SUP-001",
                    "order_date": "2025-05-15T09:00:00",
                    "expected_delivery_date": "2025-05-22T00:00:00",
                    "status": "Open"
                }
            ]
        }
    }


class PurchaseOrderLineItem(BaseModel):
    """
    Line item in a purchase order.
    
    Relationships:
    - po_number: Foreign key to PurchaseOrder.po_number
    - product_code: Foreign key to Product.product_code
    """
    po_line_item_id: str = Field(description="Unique line item identifier")
    po_number: str
    product_code: str
    ordered_quantity: float = Field(gt=0)
    unit_price: Optional[float] = None
    received_quantity: float = 0
    status: POStatus = POStatus.OPEN
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @field_validator('unit_price')
    @classmethod
    def validate_unit_price(cls, v: Optional[float], info) -> Optional[float]:
        """Validate unit price."""
        if v is not None and v <= 0:
            raise ValueError("Unit price must be positive")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "po_line_item_id": "PO-12345-1",
                    "po_number": "PO-12345",
                    "product_code": "PROD-0001",
                    "ordered_quantity": 500.0,
                    "unit_price": 125.50,
                    "received_quantity": 0.0,
                    "status": "Open"
                }
            ]
        }
    }


class QualityCheck(BaseModel):
    """
    Quality inspection record.
    
    Relationships:
    - batch_number: Foreign key to InventoryBatch.batch_number
    """
    inspection_id: str = Field(description="Unique inspection identifier")
    batch_number: str
    inspection_type: InspectionType
    inspection_date: datetime = Field(default_factory=datetime.now)
    inspector: Optional[str] = None
    result: QualityCheckResult = QualityCheckResult.PENDING
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @field_validator('inspection_id')
    @classmethod
    def validate_inspection_id(cls, v: str) -> str:
        """Validate inspection ID format."""
        # Example format: QC-12345
        if not re.match(r'^QC-\d{5,6}$', v):
            raise ValueError("Inspection ID must be in format QC-#####")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "inspection_id": "QC-12345",
                    "batch_number": "20250522-RAW-0001",
                    "inspection_type": "Receipt Inspection",
                    "inspection_date": "2025-05-22T11:30:00",
                    "inspector": "Jane Smith",
                    "result": "Pass",
                    "notes": "All parameters within acceptable range"
                }
            ]
        }
    }


class AnomalyDetection(BaseModel):
    """
    Record of detected anomalies in inventory or transactions.
    """
    anomaly_id: str = Field(description="Unique anomaly identifier")
    detection_timestamp: datetime = Field(default_factory=datetime.now)
    anomaly_type: AnomalyType
    related_entity_id: Optional[str] = None  # Product code, batch number, etc.
    severity: SeverityLevel
    description: str
    status: AnomalyStatus = AnomalyStatus.NEW
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @field_validator('anomaly_id')
    @classmethod
    def validate_anomaly_id(cls, v: str) -> str:
        """Validate anomaly ID format."""
        # Example format: ANM-12345
        if not re.match(r'^ANM-\d{5,6}$', v):
            raise ValueError("Anomaly ID must be in format ANM-#####")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "anomaly_id": "ANM-12345",
                    "anomaly_type": "Consumption Spike",
                    "related_entity_id": "PROD-0001",
                    "severity": "Medium",
                    "description": "Unusual 45% increase in daily consumption rate",
                    "status": "New"
                }
            ]
        }
    }


class ForecastResult(BaseModel):
    """
    Results from demand forecasting.
    
    Relationships:
    - product_code: Foreign key to Product.product_code
    """
    forecast_id: str = Field(description="Unique forecast identifier")
    product_code: str
    forecast_date: datetime
    forecast_period: str  # Daily, Weekly, Monthly
    predicted_quantity: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    model_version: str
    model_accuracy: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    @field_validator('forecast_id')
    @classmethod
    def validate_forecast_id(cls, v: str) -> str:
        """Validate forecast ID format."""
        # Example format: FC-20250522-001
        if not re.match(r'^FC-\d{8}-\d{3}$', v):
            raise ValueError("Forecast ID must be in format FC-YYYYMMDD-###")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "forecast_id": "FC-20250522-001",
                    "product_code": "PROD-0001",
                    "forecast_date": "2025-05-22T00:00:00",
                    "forecast_period": "Weekly",
                    "predicted_quantity": 750.0,
                    "confidence_interval_lower": 680.0,
                    "confidence_interval_upper": 820.0,
                    "model_version": "XGBoost-v1.2",
                    "model_accuracy": 0.92
                }
            ]
        }
    }


class InventorySnapshot(BaseModel):
    """
    Daily snapshot of inventory levels for analytics and historical reporting.
    """
    snapshot_id: str = Field(description="Unique snapshot identifier")
    snapshot_date: date
    product_code: str
    total_quantity: float
    bins_count: int
    storage_types: Dict[str, float]  # StorageType: quantity
    oldest_batch_date: datetime
    nearest_expiry_date: Optional[datetime] = None
    days_of_supply: Optional[float] = None  # Based on forecast
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "snapshot_id": "SNP-20250522-PROD0001",
                    "snapshot_date": "2025-05-22",
                    "product_code": "PROD-0001",
                    "total_quantity": 2500.0,
                    "bins_count": 5,
                    "storage_types": {"Ambient": 1500.0, "Rack": 1000.0},
                    "oldest_batch_date": "2025-03-15T00:00:00",
                    "nearest_expiry_date": "2025-09-22T00:00:00",
                    "days_of_supply": 12.5
                }
            ]
        }
    }
