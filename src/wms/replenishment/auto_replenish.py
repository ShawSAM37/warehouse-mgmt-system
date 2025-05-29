"""
Automatic inventory replenishment system for warehouse management.
Implements multiple replenishment strategies with real-time monitoring and automated order generation.
"""
import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy import and_, or_, func, desc

# Project imports
from ..utils.db import get_db
from ..inventory.models import (
    Product, Supplier, PurchaseOrder, PurchaseOrderLineItem, 
    InventoryBatch, InventoryTransaction, StorageLocation
)
from ..inventory.schemas import (
    PurchaseOrder as PurchaseOrderSchema,
    PurchaseOrderCreate, PurchaseOrderLineItemCreate,
    ErrorResponse
)
from ..enums import TransactionType, POStatus, UnitOfMeasurement, StockType

# Configure logging
logger = logging.getLogger(__name__)

# Configuration and Data Models
class ReplenishmentStrategy(str, Enum):
    """Available replenishment strategies."""
    REORDER_POINT = "REORDER_POINT"
    TOP_OFF = "TOP_OFF"
    PERIODIC = "PERIODIC"
    DEMAND_FORECAST = "DEMAND_FORECAST"

class ReplenishmentPriority(str, Enum):
    """Priority levels for replenishment orders."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"

@dataclass
class ReplenishmentConfig:
    """Configuration for replenishment parameters."""
    default_safety_stock_days: int = 7
    default_lead_time_buffer: float = 1.2
    max_order_frequency_days: int = 3
    demand_forecast_days: int = 90
    minimum_order_value: float = 100.0
    urgent_threshold_days: int = 2
    enable_automatic_ordering: bool = True

class ReplenishmentRequest(BaseModel):
    """Request model for manual replenishment trigger."""
    strategy: ReplenishmentStrategy = Field(..., description="Replenishment strategy to use")
    product_codes: Optional[List[str]] = Field(None, description="Specific products to replenish")
    supplier_ids: Optional[List[str]] = Field(None, description="Specific suppliers to use")
    force_generation: bool = Field(False, description="Force order generation even if not needed")
    dry_run: bool = Field(False, description="Preview orders without creating them")

class ReplenishmentResult(BaseModel):
    """Result of replenishment cycle."""
    strategy_used: ReplenishmentStrategy
    products_analyzed: int
    orders_generated: int
    total_order_value: float
    warnings: List[str] = []
    orders_created: List[str] = []
    execution_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.now)

class ProductReplenishmentInfo(BaseModel):
    """Product-specific replenishment information."""
    product_code: str
    current_stock: float
    reorder_point: float
    target_stock: float
    recommended_order_qty: float
    priority: ReplenishmentPriority
    supplier_id: Optional[str]
    estimated_stockout_date: Optional[datetime]

class ReplenishmentAlert(BaseModel):
    """Alert for urgent replenishment needs."""
    alert_id: str
    product_code: str
    alert_type: str
    severity: str
    message: str
    current_stock: float
    days_until_stockout: int
    recommended_action: str
    created_at: datetime = Field(default_factory=datetime.now)

# Core Calculation Classes
class DemandCalculator:
    """Calculates demand patterns and forecasts."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def calculate_average_daily_sales(
        self, 
        product_code: str, 
        days_back: int = 90
    ) -> Decimal:
        """Calculate average daily sales from transaction history."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Get consumption transactions
        consumption_total = self.db.query(
            func.sum(InventoryTransaction.quantity)
        ).join(InventoryBatch).filter(
            and_(
                InventoryBatch.product_code == product_code,
                InventoryTransaction.transaction_type == TransactionType.CONSUMPTION,
                InventoryTransaction.transaction_date >= cutoff_date
            )
        ).scalar() or Decimal('0')
        
        if consumption_total > 0:
            return consumption_total / days_back
        
        # Fallback to estimated demand if no consumption history
        return self._estimate_demand_from_receipts(product_code, days_back)
    
    def _estimate_demand_from_receipts(
        self, 
        product_code: str, 
        days_back: int
    ) -> Decimal:
        """Estimate demand from receipt patterns when consumption data is limited."""
        receipt_total = self.db.query(
            func.sum(InventoryTransaction.quantity)
        ).join(InventoryBatch).filter(
            and_(
                InventoryBatch.product_code == product_code,
                InventoryTransaction.transaction_type == TransactionType.RECEIPT,
                InventoryTransaction.transaction_date >= datetime.now() - timedelta(days=days_back)
            )
        ).scalar() or Decimal('0')
        
        # Assume 70% of receipts represent demand
        return (receipt_total * Decimal('0.7')) / days_back
    
    def calculate_demand_variability(
        self, 
        product_code: str, 
        days_back: int = 90
    ) -> Decimal:
        """Calculate demand variability for safety stock calculation."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Get daily consumption amounts
        daily_consumption = self.db.query(
            func.date(InventoryTransaction.transaction_date).label('date'),
            func.sum(InventoryTransaction.quantity).label('daily_total')
        ).join(InventoryBatch).filter(
            and_(
                InventoryBatch.product_code == product_code,
                InventoryTransaction.transaction_type == TransactionType.CONSUMPTION,
                InventoryTransaction.transaction_date >= cutoff_date
            )
        ).group_by(
            func.date(InventoryTransaction.transaction_date)
        ).all()
        
        if len(daily_consumption) < 7:  # Need at least a week of data
            return Decimal('0')
        
        # Calculate standard deviation
        amounts = [Decimal(str(day.daily_total)) for day in daily_consumption]
        mean = sum(amounts) / len(amounts)
        variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
        
        return variance.sqrt()
    
    def forecast_demand(
        self, 
        product_code: str, 
        forecast_days: int = 30
    ) -> Decimal:
        """Simple moving average demand forecast."""
        avg_daily = self.calculate_average_daily_sales(product_code)
        return avg_daily * forecast_days

class SafetyStockCalculator:
    """Calculates safety stock requirements."""
    
    def __init__(self, db: Session, config: ReplenishmentConfig):
        self.db = db
        self.config = config
    
    def calculate_safety_stock(
        self, 
        product: Product, 
        supplier: Supplier
    ) -> Decimal:
        """Calculate safety stock using demand variability and lead time."""
        demand_calc = DemandCalculator(self.db)
        
        # Get demand variability
        demand_variability = demand_calc.calculate_demand_variability(product.product_code)
        
        # Get effective lead time (with buffer)
        lead_time_days = supplier.average_lead_time_days or 7
        effective_lead_time = lead_time_days * self.config.default_lead_time_buffer
        
        # Safety stock formula: Z-score × √(lead time) × demand variability
        # Using Z-score of 1.65 for 95% service level
        z_score = Decimal('1.65')
        
        safety_stock = z_score * (Decimal(str(effective_lead_time)).sqrt()) * demand_variability
        
        # Minimum safety stock based on configuration
        min_safety_stock = demand_calc.calculate_average_daily_sales(
            product.product_code
        ) * self.config.default_safety_stock_days
        
        return max(safety_stock, min_safety_stock)

class ReorderPointCalculator:
    """Calculates reorder points for products."""
    
    def __init__(self, db: Session, config: ReplenishmentConfig):
        self.db = db
        self.config = config
        self.demand_calc = DemandCalculator(db)
        self.safety_calc = SafetyStockCalculator(db, config)
    
    def calculate_reorder_point(
        self, 
        product: Product, 
        supplier: Supplier
    ) -> Decimal:
        """Calculate reorder point: (avg daily sales × lead time) + safety stock."""
        avg_daily_sales = self.demand_calc.calculate_average_daily_sales(product.product_code)
        lead_time_days = supplier.average_lead_time_days or 7
        safety_stock = self.safety_calc.calculate_safety_stock(product, supplier)
        
        reorder_point = (avg_daily_sales * lead_time_days) + safety_stock
        
        return reorder_point.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

# Supplier Management
class SupplierCapacityManager:
    """Manages supplier capacity and performance analysis."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def validate_supplier_capacity(
        self, 
        supplier: Supplier, 
        additional_quantity: float,
        timeframe_days: int = 30
    ) -> bool:
        """Validate if supplier can handle additional order quantity."""
        if not supplier.monthly_capacity_limit:
            return True  # No capacity limit defined
        
        # Calculate current month's orders
        current_month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        current_orders = self.db.query(
            func.sum(PurchaseOrderLineItem.ordered_quantity)
        ).join(PurchaseOrder).filter(
            and_(
                PurchaseOrder.supplier_id == supplier.supplier_id,
                PurchaseOrder.order_date >= current_month_start,
                PurchaseOrder.status.in_([POStatus.OPEN, POStatus.PARTIALLY_RECEIVED])
            )
        ).scalar() or 0
        
        return (current_orders + additional_quantity) <= supplier.monthly_capacity_limit
    
    def assess_supplier_performance(self, supplier: Supplier) -> Dict[str, Any]:
        """Assess supplier performance metrics."""
        # Get recent orders (last 6 months)
        six_months_ago = datetime.now() - timedelta(days=180)
        
        recent_orders = self.db.query(PurchaseOrder).filter(
            and_(
                PurchaseOrder.supplier_id == supplier.supplier_id,
                PurchaseOrder.order_date >= six_months_ago
            )
        ).all()
        
        if not recent_orders:
            return {"performance_score": 0.5, "orders_analyzed": 0}
        
        # Calculate on-time delivery rate
        on_time_deliveries = sum(
            1 for order in recent_orders 
            if order.actual_delivery_date and order.expected_delivery_date
            and order.actual_delivery_date <= order.expected_delivery_date
        )
        
        delivered_orders = sum(
            1 for order in recent_orders 
            if order.actual_delivery_date
        )
        
        on_time_rate = on_time_deliveries / delivered_orders if delivered_orders > 0 else 0
        
        # Calculate average lead time accuracy
        lead_time_accuracy = self._calculate_lead_time_accuracy(recent_orders)
        
        # Overall performance score (weighted average)
        performance_score = (on_time_rate * 0.6) + (lead_time_accuracy * 0.4)
        
        return {
            "performance_score": performance_score,
            "on_time_delivery_rate": on_time_rate,
            "lead_time_accuracy": lead_time_accuracy,
            "orders_analyzed": len(recent_orders),
            "total_delivered": delivered_orders
        }
    
    def _calculate_lead_time_accuracy(self, orders: List[PurchaseOrder]) -> float:
        """Calculate lead time accuracy from order history."""
        accurate_predictions = 0
        total_predictions = 0
        
        for order in orders:
            if order.actual_delivery_date and order.expected_delivery_date:
                total_predictions += 1
                # Consider accurate if within 2 days of expected
                if abs((order.actual_delivery_date - order.expected_delivery_date).days) <= 2:
                    accurate_predictions += 1
        
        return accurate_predictions / total_predictions if total_predictions > 0 else 0

# Main Replenishment Engine
class ReplenishmentEngine:
    """Core engine for automated inventory replenishment."""
    
    def __init__(self, db: Session, config: Optional[ReplenishmentConfig] = None):
        self.db = db
        self.config = config or ReplenishmentConfig()
        self.reorder_calc = ReorderPointCalculator(db, self.config)
        self.supplier_mgr = SupplierCapacityManager(db)
        self.demand_calc = DemandCalculator(db)
        
        # Strategy mapping
        self.strategies = {
            ReplenishmentStrategy.REORDER_POINT: self._reorder_point_strategy,
            ReplenishmentStrategy.TOP_OFF: self._top_off_strategy,
            ReplenishmentStrategy.PERIODIC: self._periodic_strategy,
            ReplenishmentStrategy.DEMAND_FORECAST: self._demand_forecast_strategy
        }
    
    async def run_replenishment_cycle(
        self, 
        strategy: ReplenishmentStrategy,
        product_codes: Optional[List[str]] = None,
        supplier_ids: Optional[List[str]] = None,
        dry_run: bool = False
    ) -> ReplenishmentResult:
        """Execute replenishment cycle with specified strategy."""
        start_time = datetime.now()
        warnings = []
        orders_created = []
        
        try:
            logger.info(f"Starting replenishment cycle with strategy: {strategy}")
            
            if strategy not in self.strategies:
                raise ValueError(f"Unsupported replenishment strategy: {strategy}")
            
            # Execute strategy
            replenishment_needs = await self.strategies[strategy](
                product_codes, supplier_ids
            )
            
            if not replenishment_needs:
                logger.info("No replenishment needs identified")
                return ReplenishmentResult(
                    strategy_used=strategy,
                    products_analyzed=0,
                    orders_generated=0,
                    total_order_value=0.0,
                    warnings=["No products require replenishment"],
                    execution_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
                )
            
            # Group by supplier and generate orders
            if not dry_run:
                generated_orders = await self._generate_purchase_orders(replenishment_needs)
                orders_created = [order.po_number for order in generated_orders]
                total_value = sum(
                    sum(item.ordered_quantity * (item.unit_price or 0) for item in order.line_items)
                    for order in generated_orders
                )
            else:
                generated_orders = []
                total_value = sum(
                    need['recommended_quantity'] * (need['estimated_unit_price'] or 0)
                    for need in replenishment_needs
                )
                warnings.append("Dry run mode - no orders were actually created")
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return ReplenishmentResult(
                strategy_used=strategy,
                products_analyzed=len(replenishment_needs),
                orders_generated=len(generated_orders),
                total_order_value=total_value,
                warnings=warnings,
                orders_created=orders_created,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"Replenishment cycle failed: {e}")
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Replenishment cycle failed: {str(e)}"
            )
    
    async def _reorder_point_strategy(
        self, 
        product_codes: Optional[List[str]] = None,
        supplier_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Reorder point strategy implementation."""
        logger.info("Executing reorder point strategy")
        
        # Build query for products needing replenishment
        query = self.db.query(Product).filter(Product.is_active == True)
        
        if product_codes:
            query = query.filter(Product.product_code.in_(product_codes))
        
        products = query.all()
        replenishment_needs = []
        
        for product in products:
            # Get current stock level
            current_stock = self._get_current_stock_level(product.product_code)
            
            # Get preferred supplier
            supplier = self._get_preferred_supplier(product)
            if not supplier:
                continue
            
            if supplier_ids and supplier.supplier_id not in supplier_ids:
                continue
            
            # Calculate reorder point
            reorder_point = self.reorder_calc.calculate_reorder_point(product, supplier)
            
            # Check if replenishment is needed
            if current_stock <= reorder_point:
                # Calculate recommended order quantity
                target_stock = reorder_point * Decimal('2')  # 2x reorder point as target
                recommended_qty = max(
                    target_stock - current_stock,
                    Decimal(str(product.minimum_order_quantity or 1))
                )
                
                # Validate supplier capacity
                if not self.supplier_mgr.validate_supplier_capacity(
                    supplier, float(recommended_qty)
                ):
                    logger.warning(f"Supplier {supplier.supplier_id} capacity exceeded for {product.product_code}")
                    continue
                
                replenishment_needs.append({
                    'product': product,
                    'supplier': supplier,
                    'current_stock': current_stock,
                    'reorder_point': reorder_point,
                    'target_stock': target_stock,
                    'recommended_quantity': recommended_qty,
                    'estimated_unit_price': product.last_purchase_price or 0,
                    'priority': self._calculate_priority(current_stock, reorder_point)
                })
        
        return replenishment_needs
    
    async def _top_off_strategy(
        self, 
        product_codes: Optional[List[str]] = None,
        supplier_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Top-off strategy for fast-moving SKUs."""
        logger.info("Executing top-off strategy")
        
        # Identify fast-moving products (high consumption rate)
        fast_movers = self._identify_fast_moving_products(product_codes)
        replenishment_needs = []
        
        for product_info in fast_movers:
            product = product_info['product']
            current_stock = product_info['current_stock']
            consumption_rate = product_info['daily_consumption']
            
            supplier = self._get_preferred_supplier(product)
            if not supplier or (supplier_ids and supplier.supplier_id not in supplier_ids):
                continue
            
            # Calculate maximum stock level (e.g., 30 days of supply)
            max_stock_level = consumption_rate * 30
            
            # Top off if below 80% of maximum
            if current_stock < (max_stock_level * Decimal('0.8')):
                recommended_qty = max_stock_level - current_stock
                
                if self.supplier_mgr.validate_supplier_capacity(supplier, float(recommended_qty)):
                    replenishment_needs.append({
                        'product': product,
                        'supplier': supplier,
                        'current_stock': current_stock,
                        'recommended_quantity': recommended_qty,
                        'estimated_unit_price': product.last_purchase_price or 0,
                        'priority': ReplenishmentPriority.MEDIUM
                    })
        
        return replenishment_needs
    
    async def _periodic_strategy(
        self, 
        product_codes: Optional[List[str]] = None,
        supplier_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Periodic replenishment strategy."""
        logger.info("Executing periodic replenishment strategy")
        
        # Get products that haven't been ordered recently
        cutoff_date = datetime.now() - timedelta(days=self.config.max_order_frequency_days)
        
        recent_orders_subquery = self.db.query(
            PurchaseOrderLineItem.product_code
        ).join(PurchaseOrder).filter(
            PurchaseOrder.order_date >= cutoff_date
        ).subquery()
        
        query = self.db.query(Product).filter(
            and_(
                Product.is_active == True,
                ~Product.product_code.in_(recent_orders_subquery)
            )
        )
        
        if product_codes:
            query = query.filter(Product.product_code.in_(product_codes))
        
        products = query.all()
        replenishment_needs = []
        
        for product in products:
            current_stock = self._get_current_stock_level(product.product_code)
            supplier = self._get_preferred_supplier(product)
            
            if not supplier or (supplier_ids and supplier.supplier_id not in supplier_ids):
                continue
            
            # Calculate periodic order quantity based on consumption forecast
            forecast_demand = self.demand_calc.forecast_demand(
                product.product_code, 
                self.config.max_order_frequency_days
            )
            
            if forecast_demand > current_stock:
                recommended_qty = forecast_demand - current_stock
                
                if self.supplier_mgr.validate_supplier_capacity(supplier, float(recommended_qty)):
                    replenishment_needs.append({
                        'product': product,
                        'supplier': supplier,
                        'current_stock': current_stock,
                        'recommended_quantity': recommended_qty,
                        'estimated_unit_price': product.last_purchase_price or 0,
                        'priority': ReplenishmentPriority.LOW
                    })
        
        return replenishment_needs
    
    async def _demand_forecast_strategy(
        self, 
        product_codes: Optional[List[str]] = None,
        supplier_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Demand forecast-based replenishment strategy."""
        logger.info("Executing demand forecast strategy")
        
        query = self.db.query(Product).filter(Product.is_active == True)
        if product_codes:
            query = query.filter(Product.product_code.in_(product_codes))
        
        products = query.all()
        replenishment_needs = []
        
        for product in products:
            current_stock = self._get_current_stock_level(product.product_code)
            supplier = self._get_preferred_supplier(product)
            
            if not supplier or (supplier_ids and supplier.supplier_id not in supplier_ids):
                continue
            
            # Forecast demand for next period
            forecast_demand = self.demand_calc.forecast_demand(
                product.product_code, 
                self.config.demand_forecast_days
            )
            
            # Include safety stock in calculation
            safety_stock = SafetyStockCalculator(self.db, self.config).calculate_safety_stock(
                product, supplier
            )
            
            required_stock = forecast_demand + safety_stock
            
            if current_stock < required_stock:
                recommended_qty = required_stock - current_stock
                
                if self.supplier_mgr.validate_supplier_capacity(supplier, float(recommended_qty)):
                    replenishment_needs.append({
                        'product': product,
                        'supplier': supplier,
                        'current_stock': current_stock,
                        'forecast_demand': forecast_demand,
                        'safety_stock': safety_stock,
                        'recommended_quantity': recommended_qty,
                        'estimated_unit_price': product.last_purchase_price or 0,
                        'priority': self._calculate_priority(current_stock, required_stock)
                    })
        
        return replenishment_needs
    
    def _get_current_stock_level(self, product_code: str) -> Decimal:
        """Get current stock level for a product."""
        total_stock = self.db.query(
            func.sum(InventoryBatch.quantity)
        ).filter(
            and_(
                InventoryBatch.product_code == product_code,
                InventoryBatch.is_active == True,
                InventoryBatch.stock_type == StockType.UNRESTRICTED
            )
        ).scalar()
        
        return Decimal(str(total_stock or 0))
    
    def _get_preferred_supplier(self, product: Product) -> Optional[Supplier]:
        """Get preferred supplier for a product."""
        if product.preferred_supplier_id:
            return self.db.query(Supplier).filter(
                Supplier.supplier_id == product.preferred_supplier_id
            ).first()
        
        # Fallback to any active supplier
        return self.db.query(Supplier).filter(
            Supplier.is_active == True
        ).first()
    
    def _identify_fast_moving_products(
        self, 
        product_codes: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Identify fast-moving products based on consumption patterns."""
        # Get products with high consumption rates
        thirty_days_ago = datetime.now() - timedelta(days=30)
        
        consumption_query = self.db.query(
            InventoryBatch.product_code,
            func.sum(InventoryTransaction.quantity).label('total_consumed')
        ).join(InventoryTransaction).filter(
            and_(
                InventoryTransaction.transaction_type == TransactionType.CONSUMPTION,
                InventoryTransaction.transaction_date >= thirty_days_ago
            )
        ).group_by(InventoryBatch.product_code)
        
        if product_codes:
            consumption_query = consumption_query.filter(
                InventoryBatch.product_code.in_(product_codes)
            )
        
        consumption_data = consumption_query.all()
        
        # Calculate daily consumption and identify fast movers
        fast_movers = []
        for item in consumption_data:
            daily_consumption = Decimal(str(item.total_consumed)) / 30
            if daily_consumption > Decimal('5'):  # Threshold for fast-moving
                product = self.db.query(Product).filter(
                    Product.product_code == item.product_code
                ).first()
                
                if product:
                    current_stock = self._get_current_stock_level(product.product_code)
                    fast_movers.append({
                        'product': product,
                        'current_stock': current_stock,
                        'daily_consumption': daily_consumption
                    })
        
        return fast_movers
    
    def _calculate_priority(
        self, 
        current_stock: Decimal, 
        threshold: Decimal
    ) -> ReplenishmentPriority:
        """Calculate replenishment priority based on stock levels."""
        if current_stock <= 0:
            return ReplenishmentPriority.URGENT
        elif current_stock <= (threshold * Decimal('0.5')):
            return ReplenishmentPriority.HIGH
        elif current_stock <= (threshold * Decimal('0.8')):
            return ReplenishmentPriority.MEDIUM
        else:
            return ReplenishmentPriority.LOW
    
    async def _generate_purchase_orders(
        self, 
        replenishment_needs: List[Dict[str, Any]]
    ) -> List[PurchaseOrder]:
        """Generate purchase orders from replenishment needs."""
        # Group by supplier
        supplier_groups = {}
        for need in replenishment_needs:
            supplier = need['supplier']
            if supplier.supplier_id not in supplier_groups:
                supplier_groups[supplier.supplier_id] = {
                    'supplier': supplier,
                    'items': []
                }
            supplier_groups[supplier.supplier_id]['items'].append(need)
        
        generated_orders = []
        
        for supplier_id, group in supplier_groups.items():
            supplier = group['supplier']
            items = group['items']
            
            # Calculate total order value
            total_value = sum(
                float(item['recommended_quantity']) * (item['estimated_unit_price'] or 0)
                for item in items
            )
            
            # Skip if below minimum order value
            if total_value < self.config.minimum_order_value:
                logger.info(f"Skipping order for {supplier_id} - below minimum value")
                continue
            
            try:
                # Create purchase order
                po_number = f"PO-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
                
                purchase_order = PurchaseOrder(
                    po_number=po_number,
                    supplier_id=supplier.supplier_id,
                    order_date=datetime.now(),
                    expected_delivery_date=datetime.now() + timedelta(
                        days=supplier.average_lead_time_days or 7
                    ),
                    status=POStatus.OPEN,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                self.db.add(purchase_order)
                self.db.flush()  # Get PO ID
                
                # Create line items
                for item in items:
                    line_item = PurchaseOrderLineItem(
                        po_line_item_id=f"{po_number}-{item['product'].product_code}",
                        po_number=po_number,
                        product_code=item['product'].product_code,
                        ordered_quantity=float(item['recommended_quantity']),
                        unit_price=item['estimated_unit_price'],
                        received_quantity=0,
                        status=POStatus.OPEN,
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    self.db.add(line_item)
                
                self.db.commit()
                generated_orders.append(purchase_order)
                
                logger.info(f"Generated purchase order {po_number} for supplier {supplier_id}")
                
            except Exception as e:
                logger.error(f"Failed to create purchase order for supplier {supplier_id}: {e}")
                self.db.rollback()
                continue
        
        return generated_orders
    
    async def get_replenishment_alerts(self) -> List[ReplenishmentAlert]:
        """Get urgent replenishment alerts."""
        alerts = []
        
        # Find products with critically low stock
        products = self.db.query(Product).filter(Product.is_active == True).all()
        
        for product in products:
            current_stock = self._get_current_stock_level(product.product_code)
            daily_consumption = self.demand_calc.calculate_average_daily_sales(product.product_code)
            
            if daily_consumption > 0:
                days_until_stockout = int(current_stock / daily_consumption)
                
                if days_until_stockout <= self.config.urgent_threshold_days:
                    alert = ReplenishmentAlert(
                        alert_id=f"ALERT-{uuid.uuid4().hex[:8].upper()}",
                        product_code=product.product_code,
                        alert_type="LOW_STOCK",
                        severity="URGENT" if days_until_stockout <= 1 else "HIGH",
                        message=f"Product {product.product_code} will stock out in {days_until_stockout} days",
                        current_stock=float(current_stock),
                        days_until_stockout=days_until_stockout,
                        recommended_action="Immediate replenishment required"
                    )
                    alerts.append(alert)
        
        return alerts

# Global replenishment engine instance
replenishment_engine: Optional[ReplenishmentEngine] = None

# Dependency functions
async def get_replenishment_engine(db: Session = Depends(get_db)) -> ReplenishmentEngine:
    """Dependency to get replenishment engine instance."""
    global replenishment_engine
    if replenishment_engine is None:
        replenishment_engine = ReplenishmentEngine(db)
    return replenishment_engine

# FastAPI Router
router = APIRouter(prefix="/replenishment", tags=["Replenishment"])

@router.post("/run-cycle", response_model=ReplenishmentResult)
async def run_replenishment_cycle(
    request: ReplenishmentRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    engine: ReplenishmentEngine = Depends(get_replenishment_engine)
):
    """Execute replenishment cycle with specified strategy."""
    return await engine.run_replenishment_cycle(
        strategy=request.strategy,
        product_codes=request.product_codes,
        supplier_ids=request.supplier_ids,
        dry_run=request.dry_run
    )

@router.get("/alerts", response_model=List[ReplenishmentAlert])
async def get_replenishment_alerts(
    engine: ReplenishmentEngine = Depends(get_replenishment_engine)
):
    """Get current replenishment alerts."""
    return await engine.get_replenishment_alerts()

@router.get("/products/{product_code}/info", response_model=ProductReplenishmentInfo)
async def get_product_replenishment_info(
    product_code: str,
    db: Session = Depends(get_db),
    engine: ReplenishmentEngine = Depends(get_replenishment_engine)
):
    """Get replenishment information for specific product."""
    product = db.query(Product).filter(Product.product_code == product_code).first()
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product {product_code} not found"
        )
    
    current_stock = engine._get_current_stock_level(product_code)
    supplier = engine._get_preferred_supplier(product)
    
    if supplier:
        reorder_point = engine.reorder_calc.calculate_reorder_point(product, supplier)
        target_stock = reorder_point * Decimal('2')
        recommended_qty = max(target_stock - current_stock, Decimal('0'))
        priority = engine._calculate_priority(current_stock, reorder_point)
        
        # Estimate stockout date
        daily_consumption = engine.demand_calc.calculate_average_daily_sales(product_code)
        days_until_stockout = int(current_stock / daily_consumption) if daily_consumption > 0 else None
        estimated_stockout_date = (
            datetime.now() + timedelta(days=days_until_stockout)
            if days_until_stockout is not None else None
        )
    else:
        reorder_point = target_stock = recommended_qty = Decimal('0')
        priority = ReplenishmentPriority.LOW
        estimated_stockout_date = None
    
    return ProductReplenishmentInfo(
        product_code=product_code,
        current_stock=float(current_stock),
        reorder_point=float(reorder_point),
        target_stock=float(target_stock),
        recommended_order_qty=float(recommended_qty),
        priority=priority,
        supplier_id=supplier.supplier_id if supplier else None,
        estimated_stockout_date=estimated_stockout_date
    )

@router.post("/schedule-automatic")
async def schedule_automatic_replenishment(
    background_tasks: BackgroundTasks,
    strategy: ReplenishmentStrategy = ReplenishmentStrategy.REORDER_POINT,
    interval_hours: int = Field(24, ge=1, le=168),
    engine: ReplenishmentEngine = Depends(get_replenishment_engine)
):
    """Schedule automatic replenishment cycles."""
    async def automatic_replenishment_task():
        """Background task for automatic replenishment."""
        while True:
            try:
                logger.info("Running scheduled replenishment cycle")
                result = await engine.run_replenishment_cycle(strategy)
                logger.info(f"Automatic replenishment completed: {result.orders_generated} orders generated")
            except Exception as e:
                logger.error(f"Automatic replenishment failed: {e}")
            
            # Wait for next cycle
            await asyncio.sleep(interval_hours * 3600)
    
    background_tasks.add_task(automatic_replenishment_task)
    
    return {
        "message": f"Automatic replenishment scheduled every {interval_hours} hours",
        "strategy": strategy
    }

# Export for main application
__all__ = [
    "router",
    "ReplenishmentEngine",
    "ReplenishmentConfig",
    "ReplenishmentStrategy",
    "ReplenishmentResult",
    "ReplenishmentAlert"
]
