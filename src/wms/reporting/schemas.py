"""
Pydantic schemas for reporting and dashboard API endpoints.
Defines response models, filters, and aggregation schemas for warehouse analytics.
"""
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum

# Priority level enum for reporting
class PriorityLevel(str, Enum):
    """Priority levels for alerts and actions."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"

class TrendDirection(str, Enum):
    """Direction of metric trends."""
    UP = "UP"
    DOWN = "DOWN"
    STABLE = "STABLE"

class MetricPeriod(str, Enum):
    """Time periods for metric aggregation."""
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"

# ===== RESPONSE SCHEMAS =====

class InventorySummaryResponse(BaseModel):
    """Summary of overall inventory status."""
    total_products: int = Field(..., ge=0, description="Total number of active products")
    total_batches: int = Field(..., ge=0, description="Total number of active batches")
    total_quantity: Decimal = Field(..., ge=0, description="Total quantity in stock")
    low_stock_count: int = Field(..., ge=0, description="Number of products with low stock")
    expiring_soon_count: int = Field(..., ge=0, description="Number of batches expiring within 30 days")
    
    model_config = {"from_attributes": True}

class ProductStockLevel(BaseModel):
    """Current stock level information for a product."""
    product_code: str = Field(..., description="Product identifier")
    current_stock: Decimal = Field(..., ge=0, description="Current stock quantity")
    reorder_point: Decimal = Field(..., ge=0, description="Reorder point quantity")
    days_until_stockout: Optional[int] = Field(None, ge=0, description="Estimated days until stockout")
    priority_level: PriorityLevel = Field(..., description="Replenishment priority level")
    unit_of_measurement: Optional[str] = Field(None, description="Unit of measurement")
    
    model_config = {"from_attributes": True}

class SupplierPerformanceMetric(BaseModel):
    """Supplier performance metrics and KPIs."""
    supplier_id: str = Field(..., description="Supplier identifier")
    supplier_name: Optional[str] = Field(None, description="Supplier name")
    on_time_delivery_rate: float = Field(..., ge=0, le=100, description="On-time delivery rate percentage")
    average_lead_time: float = Field(..., ge=0, description="Average lead time in days")
    total_orders_last_30_days: int = Field(..., ge=0, description="Total orders in last 30 days")
    quality_rating: Optional[float] = Field(None, ge=0, le=5, description="Quality rating out of 5")
    
    model_config = {"from_attributes": True}

class StorageUtilizationReport(BaseModel):
    """Storage location utilization metrics."""
    storage_bin: str = Field(..., description="Storage bin identifier")
    capacity_total: Decimal = Field(..., gt=0, description="Total capacity")
    capacity_used: Decimal = Field(..., ge=0, description="Used capacity")
    utilization_percentage: float = Field(..., ge=0, le=100, description="Utilization percentage")
    zone: Optional[str] = Field(None, description="Storage zone")
    storage_type: Optional[str] = Field(None, description="Type of storage")
    
    model_config = {"from_attributes": True}

class BatchExpiryAlert(BaseModel):
    """Alert for batches approaching expiry."""
    batch_number: str = Field(..., description="Batch identifier")
    product_code: str = Field(..., description="Product identifier")
    quantity: Decimal = Field(..., ge=0, description="Quantity in batch")
    expiry_date: datetime = Field(..., description="Expiry date")
    days_until_expiry: int = Field(..., ge=0, description="Days until expiry")
    priority: PriorityLevel = Field(..., description="Priority level for action")
    storage_bin: Optional[str] = Field(None, description="Current storage location")
    
    model_config = {"from_attributes": True}

class TimeSeriesDataPoint(BaseModel):
    """Single data point in a time series."""
    timestamp: datetime = Field(..., description="Timestamp of the data point")
    value: Decimal = Field(..., description="Value at the timestamp")
    metric_name: str = Field(..., description="Name of the metric")
    
    model_config = {"from_attributes": True}

class TrendAnalysis(BaseModel):
    """Trend analysis comparing current and previous values."""
    current_value: Decimal = Field(..., description="Current value of the metric")
    previous_value: Optional[Decimal] = Field(None, description="Previous value for comparison")
    percentage_change: Optional[float] = Field(None, description="Percentage change from previous value")
    trend_direction: TrendDirection = Field(..., description="Direction of the trend")
    
    model_config = {"from_attributes": True}

class InventoryTurnoverMetric(BaseModel):
    """Inventory turnover analysis for products."""
    product_code: str = Field(..., description="Product identifier")
    turnover_rate: Decimal = Field(..., ge=0, description="Inventory turnover rate")
    days_of_supply: Optional[int] = Field(None, ge=0, description="Days of supply remaining")
    consumption_velocity: Decimal = Field(..., ge=0, description="Average daily consumption")
    
    model_config = {"from_attributes": True}

class FIFOEfficiencyReport(BaseModel):
    """FIFO consumption efficiency metrics."""
    product_code: str = Field(..., description="Product identifier")
    fifo_compliance_rate: float = Field(..., ge=0, le=100, description="FIFO compliance percentage")
    average_batch_age_consumed: float = Field(..., ge=0, description="Average age of consumed batches in days")
    oldest_batch_age: Optional[int] = Field(None, ge=0, description="Age of oldest remaining batch in days")
    
    model_config = {"from_attributes": True}

# ===== FILTER SCHEMAS =====

class ReportDateFilter(BaseModel):
    """Date range filter for reports."""
    date_from: Optional[datetime] = Field(None, description="Start date for the report")
    date_to: Optional[datetime] = Field(None, description="End date for the report")
    
    @validator('date_from', pre=True, always=True)
    def set_default_date_from(cls, v):
        """Set default start date to 30 days ago if not provided."""
        if v is None:
            return datetime.now() - timedelta(days=30)
        return v
    
    @validator('date_to', pre=True, always=True)
    def set_default_date_to(cls, v):
        """Set default end date to now if not provided."""
        if v is None:
            return datetime.now()
        return v
    
    @validator('date_to')
    def validate_date_range(cls, v, values):
        """Ensure end date is after start date."""
        if 'date_from' in values and v < values['date_from']:
            raise ValueError('date_to must be after date_from')
        return v

class ProductFilterOptions(BaseModel):
    """Filter options for product-specific reports."""
    product_codes: Optional[List[str]] = Field(None, description="List of product codes to include")
    supplier_ids: Optional[List[str]] = Field(None, description="List of supplier IDs to include")
    storage_zones: Optional[List[str]] = Field(None, description="List of storage zones to include")
    low_stock_only: bool = Field(False, description="Include only low stock products")
    
    @validator('product_codes')
    def validate_product_codes(cls, v):
        """Ensure product codes are not empty if provided."""
        if v is not None and len(v) == 0:
            return None
        return v

class MetricPeriodFilter(BaseModel):
    """Filter for metric aggregation periods."""
    period: MetricPeriod = Field(MetricPeriod.DAILY, description="Aggregation period")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of data points")
    
class AdvancedReportFilter(BaseModel):
    """Advanced filtering options for complex reports."""
    date_filter: ReportDateFilter = Field(default_factory=ReportDateFilter)
    product_filter: ProductFilterOptions = Field(default_factory=ProductFilterOptions)
    period_filter: MetricPeriodFilter = Field(default_factory=MetricPeriodFilter)
    include_inactive: bool = Field(False, description="Include inactive products/batches")
    minimum_quantity: Optional[Decimal] = Field(None, ge=0, description="Minimum quantity threshold")

# ===== AGGREGATION SCHEMAS =====

class DashboardSummary(BaseModel):
    """Complete dashboard summary with all key metrics."""
    inventory_summary: InventorySummaryResponse
    top_suppliers: List[SupplierPerformanceMetric] = Field(default_factory=list)
    storage_utilization: List[StorageUtilizationReport] = Field(default_factory=list)
    urgent_alerts: List[BatchExpiryAlert] = Field(default_factory=list)
    low_stock_products: List[ProductStockLevel] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)
    
    model_config = {"from_attributes": True}

class PerformanceMetrics(BaseModel):
    """System performance and efficiency metrics."""
    total_transactions_today: int = Field(..., ge=0, description="Total transactions processed today")
    average_processing_time: float = Field(..., ge=0, description="Average transaction processing time in seconds")
    fifo_compliance_rate: float = Field(..., ge=0, le=100, description="Overall FIFO compliance rate")
    inventory_accuracy: float = Field(..., ge=0, le=100, description="Inventory accuracy percentage")
    
    model_config = {"from_attributes": True}

class AlertSummary(BaseModel):
    """Summary of all active alerts."""
    total_alerts: int = Field(..., ge=0, description="Total number of active alerts")
    urgent_count: int = Field(..., ge=0, description="Number of urgent alerts")
    high_priority_count: int = Field(..., ge=0, description="Number of high priority alerts")
    expiry_alerts: List[BatchExpiryAlert] = Field(default_factory=list)
    low_stock_alerts: List[ProductStockLevel] = Field(default_factory=list)
    
    model_config = {"from_attributes": True}

# ===== UTILITY SCHEMAS =====

class ReportMetadata(BaseModel):
    """Metadata for generated reports."""
    report_type: str = Field(..., description="Type of report generated")
    generated_at: datetime = Field(default_factory=datetime.now)
    generated_by: Optional[str] = Field(None, description="User who generated the report")
    parameters: dict = Field(default_factory=dict, description="Parameters used for report generation")
    execution_time_ms: Optional[int] = Field(None, ge=0, description="Report generation time in milliseconds")

class HealthCheckResponse(BaseModel):
    """Health check response for reporting module."""
    status: str = Field(default="healthy", description="Health status")
    database_connected: bool = Field(..., description="Database connection status")
    cache_available: bool = Field(default=True, description="Cache availability status")
    last_update: datetime = Field(default_factory=datetime.now)
    
    model_config = {"from_attributes": True}

# Export all schemas for easy importing
__all__ = [
    "PriorityLevel",
    "TrendDirection", 
    "MetricPeriod",
    "InventorySummaryResponse",
    "ProductStockLevel",
    "SupplierPerformanceMetric",
    "StorageUtilizationReport",
    "BatchExpiryAlert",
    "TimeSeriesDataPoint",
    "TrendAnalysis",
    "InventoryTurnoverMetric",
    "FIFOEfficiencyReport",
    "ReportDateFilter",
    "ProductFilterOptions",
    "MetricPeriodFilter",
    "AdvancedReportFilter",
    "DashboardSummary",
    "PerformanceMetrics",
    "AlertSummary",
    "ReportMetadata",
    "HealthCheckResponse"
]
