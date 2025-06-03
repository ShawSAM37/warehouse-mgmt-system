"""
Calculation functions for warehouse reporting and analytics.
Provides comprehensive metrics for inventory, supplier performance, storage, and transactions.
"""
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Dict
import logging

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc, case

from ..inventory.models import (
    InventoryBatch, InventoryTransaction, Product, 
    Supplier, PurchaseOrder, PurchaseOrderLineItem, StorageLocation
)
from ..enums import TransactionType, POStatus, StockType
from .schemas import (
    ProductStockLevel, SupplierPerformanceMetric, StorageUtilizationReport,
    BatchExpiryAlert, TimeSeriesDataPoint, PriorityLevel
)

logger = logging.getLogger(__name__)

# ===== INVENTORY CALCULATIONS =====

def get_current_stock_levels(db: Session, product_codes: Optional[List[str]] = None) -> List[ProductStockLevel]:
    """
    Get current stock levels for all or specified products.
    
    Args:
        db: Database session
        product_codes: Optional list of product codes to filter
        
    Returns:
        List of ProductStockLevel objects with current inventory data
        
    Example:
        >>> stock_levels = get_current_stock_levels(db, ["PROD-001", "PROD-002"])
        >>> for level in stock_levels:
        ...     print(f"{level.product_code}: {level.current_stock}")
    """
    try:
        # Base query for current stock aggregation
        query = db.query(
            InventoryBatch.product_code,
            func.sum(InventoryBatch.quantity).label('total_quantity')
        ).filter(
            InventoryBatch.is_active == True,
            InventoryBatch.stock_type == StockType.UNRESTRICTED
        )
        
        if product_codes:
            query = query.filter(InventoryBatch.product_code.in_(product_codes))
        
        query = query.group_by(InventoryBatch.product_code)
        results = query.all()
        
        stock_levels = []
        for product_code, total_quantity in results:
            # Get product details for reorder point
            product = db.query(Product).filter(Product.product_code == product_code).first()
            reorder_point = Decimal(str(product.reorder_point or 0)) if product else Decimal('0')
            
            # Calculate days until stockout
            days_until_stockout = calculate_days_of_supply(db, product_code)
            if days_until_stockout == -1:
                days_until_stockout = None
            
            # Determine priority level
            current_stock = Decimal(str(total_quantity or 0))
            priority = _determine_stock_priority(current_stock, reorder_point, days_until_stockout)
            
            stock_levels.append(
                ProductStockLevel(
                    product_code=product_code,
                    current_stock=current_stock,
                    reorder_point=reorder_point,
                    days_until_stockout=days_until_stockout,
                    priority_level=priority,
                    unit_of_measurement=product.unit_of_measurement.value if product else None
                )
            )
        
        return stock_levels
        
    except Exception as e:
        logger.error(f"Error fetching current stock levels: {e}")
        return []

def calculate_inventory_turnover(db: Session, product_code: str, days: int = 30) -> float:
    """
    Calculate inventory turnover rate for a product over specified period.
    
    Args:
        db: Database session
        product_code: Product identifier
        days: Number of days to analyze
        
    Returns:
        Inventory turnover rate (consumption / average inventory)
        
    Example:
        >>> turnover = calculate_inventory_turnover(db, "PROD-001", 30)
        >>> print(f"Turnover rate: {turnover}")
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Calculate total consumption
        total_consumed = db.query(
            func.sum(InventoryTransaction.quantity)
        ).join(InventoryBatch).filter(
            and_(
                InventoryBatch.product_code == product_code,
                InventoryTransaction.transaction_type == TransactionType.CONSUMPTION,
                InventoryTransaction.transaction_date >= cutoff_date
            )
        ).scalar() or 0
        
        # Calculate average inventory level
        avg_inventory = db.query(
            func.avg(InventoryBatch.quantity)
        ).filter(
            and_(
                InventoryBatch.product_code == product_code,
                InventoryBatch.is_active == True
            )
        ).scalar() or 1  # Prevent division by zero
        
        if avg_inventory == 0:
            return 0.0
        
        turnover = float(total_consumed) / float(avg_inventory)
        return round(turnover, 4)
        
    except Exception as e:
        logger.error(f"Error calculating inventory turnover for {product_code}: {e}")
        return 0.0

def get_low_stock_products(db: Session, threshold_percentage: float = 0.2) -> List[ProductStockLevel]:
    """
    Get products with stock below threshold percentage of reorder point.
    
    Args:
        db: Database session
        threshold_percentage: Threshold as fraction of reorder point (0.2 = 20%)
        
    Returns:
        List of ProductStockLevel objects for low stock products
        
    Example:
        >>> low_stock = get_low_stock_products(db, 0.3)
        >>> print(f"Found {len(low_stock)} low stock products")
    """
    try:
        # Query products with their current stock and reorder points
        query = db.query(
            Product.product_code,
            Product.reorder_point,
            Product.unit_of_measurement,
            func.sum(InventoryBatch.quantity).label('total_quantity')
        ).outerjoin(
            InventoryBatch, 
            and_(
                Product.product_code == InventoryBatch.product_code,
                InventoryBatch.is_active == True
            )
        ).filter(
            Product.is_active == True
        ).group_by(
            Product.product_code, 
            Product.reorder_point, 
            Product.unit_of_measurement
        )
        
        results = query.all()
        low_stock_products = []
        
        for product_code, reorder_point, unit_of_measurement, total_quantity in results:
            current_stock = Decimal(str(total_quantity or 0))
            reorder_decimal = Decimal(str(reorder_point or 0))
            
            # Skip products without reorder point
            if reorder_decimal == 0:
                continue
                
            # Check if below threshold
            threshold_level = reorder_decimal * Decimal(str(threshold_percentage))
            if current_stock <= threshold_level:
                days_until_stockout = calculate_days_of_supply(db, product_code)
                priority = _determine_stock_priority(current_stock, reorder_decimal, days_until_stockout)
                
                low_stock_products.append(
                    ProductStockLevel(
                        product_code=product_code,
                        current_stock=current_stock,
                        reorder_point=reorder_decimal,
                        days_until_stockout=days_until_stockout if days_until_stockout != -1 else None,
                        priority_level=priority,
                        unit_of_measurement=unit_of_measurement.value if unit_of_measurement else None
                    )
                )
        
        return low_stock_products
        
    except Exception as e:
        logger.error(f"Error fetching low stock products: {e}")
        return []

def calculate_days_of_supply(db: Session, product_code: str) -> int:
    """
    Calculate estimated days of supply remaining for a product.
    
    Args:
        db: Database session
        product_code: Product identifier
        
    Returns:
        Days of supply remaining (-1 if cannot be calculated)
        
    Example:
        >>> days = calculate_days_of_supply(db, "PROD-001")
        >>> print(f"Days of supply: {days}")
    """
    try:
        # Get current stock
        current_stock = db.query(
            func.sum(InventoryBatch.quantity)
        ).filter(
            and_(
                InventoryBatch.product_code == product_code,
                InventoryBatch.is_active == True
            )
        ).scalar() or 0
        
        if current_stock == 0:
            return 0
        
        # Calculate average daily consumption over last 30 days
        days_back = 30
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        total_consumed = db.query(
            func.sum(InventoryTransaction.quantity)
        ).join(InventoryBatch).filter(
            and_(
                InventoryBatch.product_code == product_code,
                InventoryTransaction.transaction_type == TransactionType.CONSUMPTION,
                InventoryTransaction.transaction_date >= cutoff_date
            )
        ).scalar() or 0
        
        if total_consumed == 0:
            return -1  # Cannot calculate without consumption history
        
        avg_daily_consumption = total_consumed / days_back
        days_of_supply = int(current_stock / avg_daily_consumption)
        
        return max(0, days_of_supply)
        
    except Exception as e:
        logger.error(f"Error calculating days of supply for {product_code}: {e}")
        return -1

# ===== SUPPLIER PERFORMANCE =====

def calculate_supplier_performance(db: Session, supplier_id: str, days: int = 90) -> SupplierPerformanceMetric:
    """
    Calculate comprehensive supplier performance metrics.
    
    Args:
        db: Database session
        supplier_id: Supplier identifier
        days: Number of days to analyze
        
    Returns:
        SupplierPerformanceMetric object with performance data
        
    Example:
        >>> performance = calculate_supplier_performance(db, "SUP-001", 90)
        >>> print(f"On-time rate: {performance.on_time_delivery_rate}%")
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get supplier details
        supplier = db.query(Supplier).filter(Supplier.supplier_id == supplier_id).first()
        supplier_name = supplier.supplier_name if supplier else None
        
        # Get orders in the period
        orders = db.query(PurchaseOrder).filter(
            and_(
                PurchaseOrder.supplier_id == supplier_id,
                PurchaseOrder.order_date >= cutoff_date
            )
        ).all()
        
        if not orders:
            return SupplierPerformanceMetric(
                supplier_id=supplier_id,
                supplier_name=supplier_name,
                on_time_delivery_rate=0.0,
                average_lead_time=0.0,
                total_orders_last_30_days=0,
                quality_rating=0.0
            )
        
        # Calculate on-time delivery rate
        delivered_orders = [o for o in orders if o.actual_delivery_date]
        on_time_orders = [
            o for o in delivered_orders 
            if o.actual_delivery_date <= o.expected_delivery_date
        ]
        
        on_time_rate = (len(on_time_orders) / len(delivered_orders) * 100) if delivered_orders else 0.0
        
        # Calculate average lead time
        lead_times = [
            (o.actual_delivery_date - o.order_date).days 
            for o in delivered_orders
        ]
        avg_lead_time = sum(lead_times) / len(lead_times) if lead_times else 0.0
        
        # Count recent orders (last 30 days)
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_orders = len([o for o in orders if o.order_date >= recent_cutoff])
        
        return SupplierPerformanceMetric(
            supplier_id=supplier_id,
            supplier_name=supplier_name,
            on_time_delivery_rate=round(on_time_rate, 2),
            average_lead_time=round(avg_lead_time, 2),
            total_orders_last_30_days=recent_orders,
            quality_rating=4.5  # Placeholder - would need quality data
        )
        
    except Exception as e:
        logger.error(f"Error calculating supplier performance for {supplier_id}: {e}")
        return SupplierPerformanceMetric(
            supplier_id=supplier_id,
            supplier_name=None,
            on_time_delivery_rate=0.0,
            average_lead_time=0.0,
            total_orders_last_30_days=0,
            quality_rating=0.0
        )

def get_delivery_accuracy_metrics(db: Session, days: int = 30) -> Dict[str, float]:
    """
    Get overall delivery accuracy metrics across all suppliers.
    
    Args:
        db: Database session
        days: Number of days to analyze
        
    Returns:
        Dictionary with delivery accuracy metrics
        
    Example:
        >>> metrics = get_delivery_accuracy_metrics(db, 30)
        >>> print(f"Overall on-time rate: {metrics['overall_on_time_rate']}%")
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get all delivered orders in period
        delivered_orders = db.query(PurchaseOrder).filter(
            and_(
                PurchaseOrder.order_date >= cutoff_date,
                PurchaseOrder.actual_delivery_date.isnot(None)
            )
        ).all()
        
        if not delivered_orders:
            return {
                "overall_on_time_rate": 0.0,
                "average_delay_days": 0.0,
                "total_orders_analyzed": 0
            }
        
        # Calculate metrics
        on_time_orders = [
            o for o in delivered_orders 
            if o.actual_delivery_date <= o.expected_delivery_date
        ]
        
        delayed_orders = [
            o for o in delivered_orders 
            if o.actual_delivery_date > o.expected_delivery_date
        ]
        
        on_time_rate = len(on_time_orders) / len(delivered_orders) * 100
        
        # Calculate average delay for delayed orders
        delays = [
            (o.actual_delivery_date - o.expected_delivery_date).days 
            for o in delayed_orders
        ]
        avg_delay = sum(delays) / len(delays) if delays else 0.0
        
        return {
            "overall_on_time_rate": round(on_time_rate, 2),
            "average_delay_days": round(avg_delay, 2),
            "total_orders_analyzed": len(delivered_orders)
        }
        
    except Exception as e:
        logger.error(f"Error calculating delivery accuracy metrics: {e}")
        return {
            "overall_on_time_rate": 0.0,
            "average_delay_days": 0.0,
            "total_orders_analyzed": 0
        }

# ===== STORAGE ANALYTICS =====

def calculate_storage_utilization(db: Session) -> List[StorageUtilizationReport]:
    """
    Calculate storage utilization for all storage locations.
    
    Args:
        db: Database session
        
    Returns:
        List of StorageUtilizationReport objects
        
    Example:
        >>> utilization = calculate_storage_utilization(db)
        >>> for location in utilization:
        ...     print(f"{location.storage_bin}: {location.utilization_percentage}%")
    """
    try:
        # Get all storage locations with their current usage
        query = db.query(
            StorageLocation.storage_bin,
            StorageLocation.capacity_total,
            StorageLocation.zone,
            StorageLocation.storage_type,
            func.coalesce(func.sum(InventoryBatch.quantity), 0).label('used_quantity')
        ).outerjoin(
            InventoryBatch,
            and_(
                StorageLocation.storage_bin == InventoryBatch.storage_bin,
                InventoryBatch.is_active == True
            )
        ).filter(
            StorageLocation.is_active == True
        ).group_by(
            StorageLocation.storage_bin,
            StorageLocation.capacity_total,
            StorageLocation.zone,
            StorageLocation.storage_type
        )
        
        results = query.all()
        utilization_reports = []
        
        for storage_bin, capacity_total, zone, storage_type, used_quantity in results:
            capacity_total_decimal = Decimal(str(capacity_total or 0))
            used_quantity_decimal = Decimal(str(used_quantity or 0))
            
            if capacity_total_decimal > 0:
                utilization_pct = float(used_quantity_decimal / capacity_total_decimal * 100)
            else:
                utilization_pct = 0.0
            
            utilization_reports.append(
                StorageUtilizationReport(
                    storage_bin=storage_bin,
                    capacity_total=capacity_total_decimal,
                    capacity_used=used_quantity_decimal,
                    utilization_percentage=round(utilization_pct, 2),
                    zone=zone,
                    storage_type=storage_type.value if storage_type else None
                )
            )
        
        return utilization_reports
        
    except Exception as e:
        logger.error(f"Error calculating storage utilization: {e}")
        return []

def get_batch_expiry_alerts(db: Session, days_ahead: int = 30) -> List[BatchExpiryAlert]:
    """
    Get alerts for batches expiring within specified days.
    
    Args:
        db: Database session
        days_ahead: Number of days ahead to check for expiry
        
    Returns:
        List of BatchExpiryAlert objects
        
    Example:
        >>> alerts = get_batch_expiry_alerts(db, 30)
        >>> print(f"Found {len(alerts)} expiring batches")
    """
    try:
        cutoff_date = datetime.now() + timedelta(days=days_ahead)
        
        # Query batches expiring within the timeframe
        expiring_batches = db.query(InventoryBatch).filter(
            and_(
                InventoryBatch.is_active == True,
                InventoryBatch.expiry_date.isnot(None),
                InventoryBatch.expiry_date <= cutoff_date,
                InventoryBatch.quantity > 0
            )
        ).order_by(InventoryBatch.expiry_date).all()
        
        alerts = []
        for batch in expiring_batches:
            days_until_expiry = (batch.expiry_date - datetime.now()).days
            
            # Determine priority based on days until expiry
            if days_until_expiry <= 3:
                priority = PriorityLevel.URGENT
            elif days_until_expiry <= 7:
                priority = PriorityLevel.HIGH
            elif days_until_expiry <= 14:
                priority = PriorityLevel.MEDIUM
            else:
                priority = PriorityLevel.LOW
            
            alerts.append(
                BatchExpiryAlert(
                    batch_number=batch.batch_number,
                    product_code=batch.product_code,
                    quantity=batch.quantity,
                    expiry_date=batch.expiry_date,
                    days_until_expiry=max(0, days_until_expiry),
                    priority=priority,
                    storage_bin=batch.storage_bin
                )
            )
        
        return alerts
        
    except Exception as e:
        logger.error(f"Error getting batch expiry alerts: {e}")
        return []

# ===== TRANSACTION ANALYTICS =====

def get_consumption_trends(db: Session, days: int = 30) -> List[TimeSeriesDataPoint]:
    """
    Get daily consumption trends over specified period.
    
    Args:
        db: Database session
        days: Number of days to analyze
        
    Returns:
        List of TimeSeriesDataPoint objects with daily consumption data
        
    Example:
        >>> trends = get_consumption_trends(db, 30)
        >>> for point in trends:
        ...     print(f"{point.timestamp}: {point.value}")
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Query daily consumption totals
        daily_consumption = db.query(
            func.date(InventoryTransaction.transaction_date).label('date'),
            func.sum(InventoryTransaction.quantity).label('total_consumed')
        ).join(InventoryBatch).filter(
            and_(
                InventoryTransaction.transaction_type == TransactionType.CONSUMPTION,
                InventoryTransaction.transaction_date >= cutoff_date
            )
        ).group_by(
            func.date(InventoryTransaction.transaction_date)
        ).order_by(
            func.date(InventoryTransaction.transaction_date)
        ).all()
        
        trends = []
        for date, total_consumed in daily_consumption:
            trends.append(
                TimeSeriesDataPoint(
                    timestamp=datetime.combine(date, datetime.min.time()),
                    value=Decimal(str(total_consumed or 0)),
                    metric_name="daily_consumption"
                )
            )
        
        return trends
        
    except Exception as e:
        logger.error(f"Error getting consumption trends: {e}")
        return []

def calculate_fifo_efficiency(db: Session, product_code: str) -> float:
    """
    Calculate FIFO compliance efficiency for a product.
    
    Args:
        db: Database session
        product_code: Product identifier
        
    Returns:
        FIFO efficiency percentage (0-100)
        
    Example:
        >>> efficiency = calculate_fifo_efficiency(db, "PROD-001")
        >>> print(f"FIFO efficiency: {efficiency}%")
    """
    try:
        # Get consumption transactions for the product (last 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        
        consumption_transactions = db.query(InventoryTransaction).join(InventoryBatch).filter(
            and_(
                InventoryBatch.product_code == product_code,
                InventoryTransaction.transaction_type == TransactionType.CONSUMPTION,
                InventoryTransaction.transaction_date >= cutoff_date
            )
        ).order_by(InventoryTransaction.transaction_date).all()
        
        if not consumption_transactions:
            return 0.0
        
        # Analyze FIFO compliance
        fifo_compliant_transactions = 0
        
        for transaction in consumption_transactions:
            batch = db.query(InventoryBatch).filter(
                InventoryBatch.batch_number == transaction.batch_number
            ).first()
            
            if not batch:
                continue
            
            # Check if this was the oldest available batch at time of consumption
            older_batches = db.query(InventoryBatch).filter(
                and_(
                    InventoryBatch.product_code == product_code,
                    InventoryBatch.goods_receipt_date < batch.goods_receipt_date,
                    InventoryBatch.is_active == True,
                    InventoryBatch.quantity > 0
                )
            ).count()
            
            # If no older batches were available, this was FIFO compliant
            if older_batches == 0:
                fifo_compliant_transactions += 1
        
        efficiency = (fifo_compliant_transactions / len(consumption_transactions)) * 100
        return round(efficiency, 2)
        
    except Exception as e:
        logger.error(f"Error calculating FIFO efficiency for {product_code}: {e}")
        return 0.0

# ===== HELPER FUNCTIONS =====

def _determine_stock_priority(
    current_stock: Decimal, 
    reorder_point: Decimal, 
    days_until_stockout: Optional[int]
) -> PriorityLevel:
    """Determine priority level based on stock levels and stockout timeline."""
    if current_stock == 0:
        return PriorityLevel.URGENT
    
    if days_until_stockout is not None:
        if days_until_stockout <= 3:
            return PriorityLevel.URGENT
        elif days_until_stockout <= 7:
            return PriorityLevel.HIGH
    
    if reorder_point > 0:
        stock_ratio = current_stock / reorder_point
        if stock_ratio <= 0.2:
            return PriorityLevel.URGENT
        elif stock_ratio <= 0.5:
            return PriorityLevel.HIGH
        elif stock_ratio <= 0.8:
            return PriorityLevel.MEDIUM
    
    return PriorityLevel.LOW
