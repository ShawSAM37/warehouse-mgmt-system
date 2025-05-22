# src/wms/shared/enums.py
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
