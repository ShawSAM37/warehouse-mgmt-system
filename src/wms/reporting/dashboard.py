"""
Dashboard API endpoints for warehouse management reporting and analytics.
Provides comprehensive metrics, real-time updates, and performance monitoring.
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from functools import wraps
import json

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

# Project imports
from ..utils.db import get_db
from ..inventory.models import Product, InventoryBatch, InventoryTransaction
from ..enums import TransactionType, StockType
from .schemas import (
    InventorySummaryResponse, ProductStockLevel, SupplierPerformanceMetric,
    StorageUtilizationReport, BatchExpiryAlert, TimeSeriesDataPoint,
    ReportDateFilter, ProductFilterOptions, MetricPeriodFilter,
    DashboardSummary, PerformanceMetrics, AlertSummary, HealthCheckResponse,
    PriorityLevel, InventoryTurnoverMetric, FIFOEfficiencyReport
)
from .calculations import (
    get_current_stock_levels, calculate_inventory_turnover, get_low_stock_products,
    calculate_days_of_supply, calculate_supplier_performance, get_delivery_accuracy_metrics,
    calculate_storage_utilization, get_batch_expiry_alerts, get_consumption_trends,
    calculate_fifo_efficiency
)

# Configure logging
logger = logging.getLogger(__name__)

# Simple in-memory cache for expensive calculations
cache_store: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 300  # 5 minutes

def cache_result(ttl: int = CACHE_TTL):
    """Simple caching decorator for expensive calculations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Check if cached result exists and is still valid
            if cache_key in cache_store:
                cached_data = cache_store[cache_key]
                if time.time() - cached_data['timestamp'] < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_data['result']
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache_store[cache_key] = {
                'result': result,
                'timestamp': time.time()
            }
            
            logger.debug(f"Cache miss for {func.__name__}, result cached")
            return result
        return wrapper
    return decorator

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)

# Global connection manager
manager = ConnectionManager()

# Create router
router = APIRouter(prefix="/reporting", tags=["Reporting & Analytics"])

# ===== CORE ENDPOINTS =====

@router.get("/inventory-summary", response_model=InventorySummaryResponse)
async def get_inventory_summary(
    db: Session = Depends(get_db)
):
    """
    Get comprehensive inventory summary with key metrics.
    
    Returns:
        InventorySummaryResponse: Summary of inventory status including total products,
        batches, quantities, low stock count, and expiring batches.
    """
    try:
        start_time = time.time()
        
        # Get total active products
        total_products = db.query(Product).filter(Product.is_active == True).count()
        
        # Get total active batches
        total_batches = db.query(InventoryBatch).filter(
            InventoryBatch.is_active == True
        ).count()
        
        # Get total quantity across all batches
        total_quantity = db.query(
            db.func.sum(InventoryBatch.quantity)
        ).filter(
            InventoryBatch.is_active == True,
            InventoryBatch.stock_type == StockType.UNRESTRICTED
        ).scalar() or 0
        
        # Get low stock count (products below 20% of reorder point)
        low_stock_products = get_low_stock_products(db, threshold_percentage=0.2)
        low_stock_count = len(low_stock_products)
        
        # Get expiring soon count (within 30 days)
        expiring_alerts = get_batch_expiry_alerts(db, days_ahead=30)
        expiring_soon_count = len(expiring_alerts)
        
        execution_time = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Inventory summary generated in {execution_time}ms")
        
        return InventorySummaryResponse(
            total_products=total_products,
            total_batches=total_batches,
            total_quantity=total_quantity,
            low_stock_count=low_stock_count,
            expiring_soon_count=expiring_soon_count
        )
        
    except SQLAlchemyError as e:
        logger.error(f"Database error in inventory summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred while fetching inventory summary"
        )
    except Exception as e:
        logger.error(f"Unexpected error in inventory summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )

@router.get("/stock-levels", response_model=List[ProductStockLevel])
async def get_stock_levels(
    product_codes: Optional[str] = Query(None, description="Comma-separated product codes"),
    low_stock_only: bool = Query(False, description="Return only low stock products"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    db: Session = Depends(get_db)
):
    """
    Get current stock levels for products with optional filtering.
    
    Args:
        product_codes: Optional comma-separated list of product codes
        low_stock_only: If True, return only products with low stock
        skip: Number of records to skip for pagination
        limit: Maximum number of records to return
        
    Returns:
        List[ProductStockLevel]: Current stock levels with priority indicators
    """
    try:
        # Parse product codes if provided
        product_code_list = None
        if product_codes:
            product_code_list = [code.strip() for code in product_codes.split(",")]
        
        if low_stock_only:
            stock_levels = get_low_stock_products(db, threshold_percentage=0.2)
        else:
            stock_levels = get_current_stock_levels(db, product_code_list)
        
        # Apply pagination
        total_count = len(stock_levels)
        paginated_results = stock_levels[skip:skip + limit]
        
        logger.info(f"Retrieved {len(paginated_results)} stock levels (total: {total_count})")
        return paginated_results
        
    except Exception as e:
        logger.error(f"Error fetching stock levels: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching stock levels"
        )

@router.get("/supplier-performance", response_model=List[SupplierPerformanceMetric])
@cache_result(ttl=600)  # Cache for 10 minutes
async def get_supplier_performance(
    days: int = Query(90, ge=1, le=365, description="Number of days to analyze"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Get supplier performance metrics for all active suppliers.
    
    Args:
        days: Number of days to analyze for performance calculation
        skip: Number of records to skip for pagination
        limit: Maximum number of records to return
        
    Returns:
        List[SupplierPerformanceMetric]: Performance metrics for each supplier
    """
    try:
        # Get all active suppliers
        from ..inventory.models import Supplier
        suppliers = db.query(Supplier).filter(
            Supplier.is_active == True
        ).offset(skip).limit(limit).all()
        
        performance_metrics = []
        for supplier in suppliers:
            metrics = calculate_supplier_performance(db, supplier.supplier_id, days)
            performance_metrics.append(metrics)
        
        logger.info(f"Generated performance metrics for {len(performance_metrics)} suppliers")
        return performance_metrics
        
    except Exception as e:
        logger.error(f"Error fetching supplier performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching supplier performance metrics"
        )

@router.get("/storage-utilization", response_model=List[StorageUtilizationReport])
@cache_result(ttl=300)  # Cache for 5 minutes
async def get_storage_utilization(
    zone: Optional[str] = Query(None, description="Filter by storage zone"),
    min_utilization: Optional[float] = Query(None, ge=0, le=100, description="Minimum utilization percentage"),
    db: Session = Depends(get_db)
):
    """
    Get storage utilization metrics for all storage locations.
    
    Args:
        zone: Optional zone filter
        min_utilization: Optional minimum utilization percentage filter
        
    Returns:
        List[StorageUtilizationReport]: Utilization metrics for storage locations
    """
    try:
        utilization_reports = calculate_storage_utilization(db)
        
        # Apply filters
        if zone:
            utilization_reports = [r for r in utilization_reports if r.zone == zone]
        
        if min_utilization is not None:
            utilization_reports = [r for r in utilization_reports if r.utilization_percentage >= min_utilization]
        
        logger.info(f"Retrieved utilization for {len(utilization_reports)} storage locations")
        return utilization_reports
        
    except Exception as e:
        logger.error(f"Error fetching storage utilization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching storage utilization metrics"
        )

@router.get("/expiry-alerts", response_model=List[BatchExpiryAlert])
async def get_expiry_alerts(
    days_ahead: int = Query(30, ge=1, le=365, description="Days ahead to check for expiry"),
    priority: Optional[PriorityLevel] = Query(None, description="Filter by priority level"),
    db: Session = Depends(get_db)
):
    """
    Get alerts for batches expiring within specified timeframe.
    
    Args:
        days_ahead: Number of days ahead to check for expiring batches
        priority: Optional priority level filter
        
    Returns:
        List[BatchExpiryAlert]: Expiry alerts sorted by expiry date
    """
    try:
        alerts = get_batch_expiry_alerts(db, days_ahead)
        
        # Apply priority filter
        if priority:
            alerts = [alert for alert in alerts if alert.priority == priority]
        
        logger.info(f"Retrieved {len(alerts)} expiry alerts")
        return alerts
        
    except Exception as e:
        logger.error(f"Error fetching expiry alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching expiry alerts"
        )

@router.get("/consumption-trends", response_model=List[TimeSeriesDataPoint])
@cache_result(ttl=600)  # Cache for 10 minutes
async def get_consumption_trends(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get daily consumption trends over specified period.
    
    Args:
        days: Number of days to analyze for consumption trends
        
    Returns:
        List[TimeSeriesDataPoint]: Daily consumption data points
    """
    try:
        trends = get_consumption_trends(db, days)
        logger.info(f"Retrieved consumption trends for {len(trends)} days")
        return trends
        
    except Exception as e:
        logger.error(f"Error fetching consumption trends: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching consumption trends"
        )

# ===== ADVANCED ENDPOINTS =====

@router.get("/products/{product_code}/analytics", response_model=Dict[str, Any])
@cache_result(ttl=300)
async def get_product_analytics(
    product_code: str,
    days: int = Query(30, ge=1, le=365, description="Analysis period in days"),
    db: Session = Depends(get_db)
):
    """
    Get detailed analytics for a specific product.
    
    Args:
        product_code: Product identifier
        days: Number of days to analyze
        
    Returns:
        Dict containing comprehensive product analytics
    """
    try:
        # Verify product exists
        product = db.query(Product).filter(Product.product_code == product_code).first()
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product {product_code} not found"
            )
        
        # Get current stock levels
        stock_levels = get_current_stock_levels(db, [product_code])
        current_stock = stock_levels[0] if stock_levels else None
        
        # Calculate metrics
        turnover_rate = calculate_inventory_turnover(db, product_code, days)
        days_of_supply = calculate_days_of_supply(db, product_code)
        fifo_efficiency = calculate_fifo_efficiency(db, product_code)
        
        # Get recent transactions
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_transactions = db.query(InventoryTransaction).join(InventoryBatch).filter(
            InventoryBatch.product_code == product_code,
            InventoryTransaction.transaction_date >= cutoff_date
        ).count()
        
        analytics = {
            "product_code": product_code,
            "current_stock": current_stock.model_dump() if current_stock else None,
            "turnover_rate": turnover_rate,
            "days_of_supply": days_of_supply,
            "fifo_efficiency": fifo_efficiency,
            "recent_transactions": recent_transactions,
            "analysis_period_days": days,
            "generated_at": datetime.now().isoformat()
        }
        
        logger.info(f"Generated analytics for product {product_code}")
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating product analytics for {product_code}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating product analytics"
        )

@router.get("/suppliers/{supplier_id}/scorecard", response_model=Dict[str, Any])
@cache_result(ttl=600)
async def get_supplier_scorecard(
    supplier_id: str,
    days: int = Query(90, ge=1, le=365, description="Analysis period in days"),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive supplier scorecard with detailed metrics.
    
    Args:
        supplier_id: Supplier identifier
        days: Number of days to analyze
        
    Returns:
        Dict containing comprehensive supplier scorecard
    """
    try:
        # Verify supplier exists
        from ..inventory.models import Supplier
        supplier = db.query(Supplier).filter(Supplier.supplier_id == supplier_id).first()
        if not supplier:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Supplier {supplier_id} not found"
            )
        
        # Get performance metrics
        performance = calculate_supplier_performance(db, supplier_id, days)
        
        # Get additional metrics
        from ..inventory.models import PurchaseOrder
        total_orders = db.query(PurchaseOrder).filter(
            PurchaseOrder.supplier_id == supplier_id
        ).count()
        
        scorecard = {
            "supplier_info": {
                "supplier_id": supplier_id,
                "supplier_name": supplier.supplier_name,
                "contact_person": supplier.contact_person,
                "email": supplier.email
            },
            "performance_metrics": performance.model_dump(),
            "historical_data": {
                "total_orders_all_time": total_orders,
                "analysis_period_days": days
            },
            "generated_at": datetime.now().isoformat()
        }
        
        logger.info(f"Generated scorecard for supplier {supplier_id}")
        return scorecard
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating supplier scorecard for {supplier_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating supplier scorecard"
        )

@router.get("/inventory-health", response_model=Dict[str, Any])
@cache_result(ttl=300)
async def get_inventory_health(
    db: Session = Depends(get_db)
):
    """
    Get overall inventory health metrics and system status.
    
    Returns:
        Dict containing comprehensive inventory health indicators
    """
    try:
        # Get basic inventory summary
        summary = await get_inventory_summary(db)
        
        # Get additional health metrics
        delivery_metrics = get_delivery_accuracy_metrics(db, 30)
        
        # Calculate overall health score
        health_indicators = {
            "low_stock_ratio": summary.low_stock_count / max(summary.total_products, 1),
            "expiring_ratio": summary.expiring_soon_count / max(summary.total_batches, 1),
            "delivery_performance": delivery_metrics.get("overall_on_time_rate", 0) / 100
        }
        
        # Calculate weighted health score (0-100)
        health_score = (
            (1 - health_indicators["low_stock_ratio"]) * 0.4 +
            (1 - health_indicators["expiring_ratio"]) * 0.3 +
            health_indicators["delivery_performance"] * 0.3
        ) * 100
        
        health_status = {
            "overall_health_score": round(health_score, 2),
            "status": "excellent" if health_score >= 90 else 
                     "good" if health_score >= 75 else 
                     "fair" if health_score >= 60 else "poor",
            "inventory_summary": summary.model_dump(),
            "delivery_metrics": delivery_metrics,
            "health_indicators": health_indicators,
            "generated_at": datetime.now().isoformat()
        }
        
        logger.info(f"Generated inventory health report with score: {health_score}")
        return health_status
        
    except Exception as e:
        logger.error(f"Error generating inventory health report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating inventory health report"
        )

@router.get("/fifo-efficiency", response_model=List[FIFOEfficiencyReport])
@cache_result(ttl=600)
async def get_fifo_efficiency_report(
    product_codes: Optional[str] = Query(None, description="Comma-separated product codes"),
    min_efficiency: Optional[float] = Query(None, ge=0, le=100, description="Minimum efficiency percentage"),
    db: Session = Depends(get_db)
):
    """
    Get FIFO efficiency report for products.
    
    Args:
        product_codes: Optional comma-separated list of product codes
        min_efficiency: Optional minimum efficiency percentage filter
        
    Returns:
        List[FIFOEfficiencyReport]: FIFO efficiency metrics for products
    """
    try:
        # Parse product codes if provided
        product_code_list = None
        if product_codes:
            product_code_list = [code.strip() for code in product_codes.split(",")]
        
        # Get products to analyze
        query = db.query(Product).filter(Product.is_active == True)
        if product_code_list
