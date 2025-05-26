"""
API endpoints for warehouse management system.
Provides comprehensive CRUD operations with validation, error handling, and relationships.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import IntegrityError
from typing import List, Optional
from datetime import datetime

# Import existing modules
from ..utils.db import get_db
from ..inventory.models import (
    Product as ORMProduct,
    InventoryBatch as ORMInventoryBatch,
    InventoryTransaction as ORMInventoryTransaction,
    StorageLocation as ORMStorageLocation,
    Supplier as ORMSupplier,
    PurchaseOrder as ORMPurchaseOrder,
    PurchaseOrderLineItem as ORMPurchaseOrderLineItem,
    QualityCheck as ORMQualityCheck,
    AnomalyDetection as ORMAnomalyDetection
)
from ..inventory.schemas import (
    # Product schemas
    Product, ProductCreate, ProductUpdate, ProductWithBatches,
    # Inventory schemas
    InventoryBatch, InventoryBatchCreate, InventoryBatchUpdate, InventoryBatchWithTransactions,
    # Transaction schemas
    InventoryTransaction, InventoryTransactionCreate,
    # Storage schemas
    StorageLocation, StorageLocationCreate, StorageLocationWithBatches,
    # Supplier schemas
    Supplier, SupplierCreate, SupplierWithOrders,
    # Purchase Order schemas
    PurchaseOrder, PurchaseOrderCreate, PurchaseOrderWithLineItems,
    PurchaseOrderLineItem, PurchaseOrderLineItemCreate,
    # Quality schemas
    QualityCheck, QualityCheckCreate, QualityCheckUpdate,
    # Anomaly schemas
    AnomalyDetection, AnomalyDetectionCreate, AnomalyDetectionUpdate,
    # Utility schemas
    PaginationMeta, PaginatedResponse, ErrorResponse, HealthCheck,
    # Filter schemas
    ProductFilter, InventoryBatchFilter, InventoryTransactionFilter,
    QualityCheckFilter, AnomalyDetectionFilter
)

# Create main router
router = APIRouter()

# ===== HEALTH CHECK =====

@router.get("/health", response_model=HealthCheck, tags=["health"])
def health_check():
    """Health check endpoint."""
    return HealthCheck()

# ===== PRODUCT ENDPOINTS =====

product_router = APIRouter(prefix="/products", tags=["products"])

@product_router.get("/", response_model=List[Product])
def list_products(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    product_code: Optional[str] = None,
    is_active: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """List products with optional filtering."""
    try:
        query = db.query(ORMProduct)
        
        # Apply filters
        if product_code:
            query = query.filter(ORMProduct.product_code.ilike(f"%{product_code}%"))
        if is_active is not None:
            query = query.filter(ORMProduct.is_active == is_active)
        
        products = query.offset(skip).limit(limit).all()
        return products
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

@product_router.get("/{product_code}", response_model=ProductWithBatches)
def get_product(product_code: str, db: Session = Depends(get_db)):
    """Get product with related batches."""
    product = db.query(ORMProduct).options(
        joinedload(ORMProduct.batches)
    ).filter(ORMProduct.product_code == product_code).first()
    
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product {product_code} not found"
        )
    return product

@product_router.post("/", response_model=Product, status_code=status.HTTP_201_CREATED)
def create_product(product: ProductCreate, db: Session = Depends(get_db)):
    """Create a new product."""
    # Check for existing product
    existing = db.query(ORMProduct).filter(
        ORMProduct.product_code == product.product_code
    ).first()
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Product {product.product_code} already exists"
        )
    
    try:
        db_product = ORMProduct(**product.model_dump())
        db.add(db_product)
        db.commit()
        db.refresh(db_product)
        return db_product
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Database constraint violation: {str(e)}"
        )

@product_router.put("/{product_code}", response_model=Product)
def update_product(
    product_code: str,
    update_data: ProductUpdate,
    db: Session = Depends(get_db)
):
    """Update an existing product."""
    product = db.query(ORMProduct).filter(
        ORMProduct.product_code == product_code
    ).first()
    
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product {product_code} not found"
        )
    
    try:
        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(product, field, value)
        
        product.updated_at = datetime.now()
        db.commit()
        db.refresh(product)
        return product
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Database constraint violation: {str(e)}"
        )

@product_router.delete("/{product_code}", status_code=status.HTTP_204_NO_CONTENT)
def delete_product(product_code: str, db: Session = Depends(get_db)):
    """Delete a product (soft delete by setting is_active=False)."""
    product = db.query(ORMProduct).filter(
        ORMProduct.product_code == product_code
    ).first()
    
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product {product_code} not found"
        )
    
    try:
        # Soft delete
        product.is_active = False
        product.updated_at = datetime.now()
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete product: {str(e)}"
        )

# ===== INVENTORY BATCH ENDPOINTS =====

batch_router = APIRouter(prefix="/inventory/batches", tags=["inventory-batches"])

@batch_router.get("/", response_model=List[InventoryBatch])
def list_inventory_batches(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    product_code: Optional[str] = None,
    storage_bin: Optional[str] = None,
    is_active: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """List inventory batches with filtering."""
    try:
        query = db.query(ORMInventoryBatch)
        
        if product_code:
            query = query.filter(ORMInventoryBatch.product_code == product_code)
        if storage_bin:
            query = query.filter(ORMInventoryBatch.storage_bin == storage_bin)
        if is_active is not None:
            query = query.filter(ORMInventoryBatch.is_active == is_active)
        
        batches = query.offset(skip).limit(limit).all()
        return batches
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

@batch_router.get("/{batch_number}", response_model=InventoryBatchWithTransactions)
def get_inventory_batch(batch_number: str, db: Session = Depends(get_db)):
    """Get inventory batch with transactions and quality checks."""
    batch = db.query(ORMInventoryBatch).options(
        joinedload(ORMInventoryBatch.transactions),
        joinedload(ORMInventoryBatch.quality_checks)
    ).filter(ORMInventoryBatch.batch_number == batch_number).first()
    
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch {batch_number} not found"
        )
    return batch

@batch_router.post("/", response_model=InventoryBatch, status_code=status.HTTP_201_CREATED)
def create_inventory_batch(batch: InventoryBatchCreate, db: Session = Depends(get_db)):
    """Create a new inventory batch."""
    # Verify product exists
    product = db.query(ORMProduct).filter(
        ORMProduct.product_code == batch.product_code
    ).first()
    if not product:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Product {batch.product_code} not found"
        )
    
    # Verify storage location exists
    storage = db.query(ORMStorageLocation).filter(
        ORMStorageLocation.storage_bin == batch.storage_bin
    ).first()
    if not storage:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Storage bin {batch.storage_bin} not found"
        )
    
    try:
        db_batch = ORMInventoryBatch(**batch.model_dump())
        db.add(db_batch)
        db.commit()
        db.refresh(db_batch)
        return db_batch
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Database constraint violation: {str(e)}"
        )

@batch_router.put("/{batch_number}", response_model=InventoryBatch)
def update_inventory_batch(
    batch_number: str,
    update_data: InventoryBatchUpdate,
    db: Session = Depends(get_db)
):
    """Update an inventory batch."""
    batch = db.query(ORMInventoryBatch).filter(
        ORMInventoryBatch.batch_number == batch_number
    ).first()
    
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch {batch_number} not found"
        )
    
    try:
        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(batch, field, value)
        
        batch.updated_at = datetime.now()
        db.commit()
        db.refresh(batch)
        return batch
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Database constraint violation: {str(e)}"
        )

# ===== INVENTORY TRANSACTION ENDPOINTS =====

transaction_router = APIRouter(prefix="/inventory/transactions", tags=["inventory-transactions"])

@transaction_router.get("/", response_model=List[InventoryTransaction])
def list_inventory_transactions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    batch_number: Optional[str] = None,
    transaction_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List inventory transactions with filtering."""
    try:
        query = db.query(ORMInventoryTransaction)
        
        if batch_number:
            query = query.filter(ORMInventoryTransaction.batch_number == batch_number)
        if transaction_type:
            query = query.filter(ORMInventoryTransaction.transaction_type == transaction_type)
        
        transactions = query.order_by(
            ORMInventoryTransaction.transaction_date.desc()
        ).offset(skip).limit(limit).all()
        return transactions
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

@transaction_router.get("/{transaction_id}", response_model=InventoryTransaction)
def get_inventory_transaction(transaction_id: str, db: Session = Depends(get_db)):
    """Get specific inventory transaction."""
    transaction = db.query(ORMInventoryTransaction).filter(
        ORMInventoryTransaction.transaction_id == transaction_id
    ).first()
    
    if not transaction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction {transaction_id} not found"
        )
    return transaction

@transaction_router.post("/", response_model=InventoryTransaction, status_code=status.HTTP_201_CREATED)
def create_inventory_transaction(transaction: InventoryTransactionCreate, db: Session = Depends(get_db)):
    """Create a new inventory transaction."""
    # Verify batch exists
    batch = db.query(ORMInventoryBatch).filter(
        ORMInventoryBatch.batch_number == transaction.batch_number
    ).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch {transaction.batch_number} not found"
        )
    
    try:
        db_transaction = ORMInventoryTransaction(**transaction.model_dump())
        db.add(db_transaction)
        db.commit()
        db.refresh(db_transaction)
        return db_transaction
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Database constraint violation: {str(e)}"
        )

# ===== STORAGE LOCATION ENDPOINTS =====

storage_router = APIRouter(prefix="/storage", tags=["storage-locations"])

@storage_router.get("/", response_model=List[StorageLocation])
def list_storage_locations(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    storage_type: Optional[str] = None,
    zone: Optional[str] = None,
    is_active: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """List storage locations with filtering."""
    try:
        query = db.query(ORMStorageLocation)
        
        if storage_type:
            query = query.filter(ORMStorageLocation.storage_type == storage_type)
        if zone:
            query = query.filter(ORMStorageLocation.zone == zone)
        if is_active is not None:
            query = query.filter(ORMStorageLocation.is_active == is_active)
        
        locations = query.offset(skip).limit(limit).all()
        return locations
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

@storage_router.get("/{storage_bin}", response_model=StorageLocationWithBatches)
def get_storage_location(storage_bin: str, db: Session = Depends(get_db)):
    """Get storage location with current batches."""
    location = db.query(ORMStorageLocation).options(
        joinedload(ORMStorageLocation.batches)
    ).filter(ORMStorageLocation.storage_bin == storage_bin).first()
    
    if not location:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Storage bin {storage_bin} not found"
        )
    return location

@storage_router.post("/", response_model=StorageLocation, status_code=status.HTTP_201_CREATED)
def create_storage_location(location: StorageLocationCreate, db: Session = Depends(get_db)):
    """Create a new storage location."""
    try:
        db_location = ORMStorageLocation(**location.model_dump())
        db.add(db_location)
        db.commit()
        db.refresh(db_location)
        return db_location
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Database constraint violation: {str(e)}"
        )

# ===== SUPPLIER ENDPOINTS =====

supplier_router = APIRouter(prefix="/suppliers", tags=["suppliers"])

@supplier_router.get("/", response_model=List[Supplier])
def list_suppliers(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    is_active: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """List suppliers."""
    try:
        query = db.query(ORMSupplier)
        
        if is_active is not None:
            query = query.filter(ORMSupplier.is_active == is_active)
        
        suppliers = query.offset(skip).limit(limit).all()
        return suppliers
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

@supplier_router.get("/{supplier_id}", response_model=SupplierWithOrders)
def get_supplier(supplier_id: str, db: Session = Depends(get_db)):
    """Get supplier with purchase orders."""
    supplier = db.query(ORMSupplier).options(
        joinedload(ORMSupplier.purchase_orders)
    ).filter(ORMSupplier.supplier_id == supplier_id).first()
    
    if not supplier:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Supplier {supplier_id} not found"
        )
    return supplier

@supplier_router.post("/", response_model=Supplier, status_code=status.HTTP_201_CREATED)
def create_supplier(supplier: SupplierCreate, db: Session = Depends(get_db)):
    """Create a new supplier."""
    try:
        db_supplier = ORMSupplier(**supplier.model_dump())
        db.add(db_supplier)
        db.commit()
        db.refresh(db_supplier)
        return db_supplier
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Database constraint violation: {str(e)}"
        )

# ===== PURCHASE ORDER ENDPOINTS =====

po_router = APIRouter(prefix="/purchase-orders", tags=["purchase-orders"])

@po_router.get("/", response_model=List[PurchaseOrder])
def list_purchase_orders(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    supplier_id: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List purchase orders."""
    try:
        query = db.query(ORMPurchaseOrder)
        
        if supplier_id:
            query = query.filter(ORMPurchaseOrder.supplier_id == supplier_id)
        if status:
            query = query.filter(ORMPurchaseOrder.status == status)
        
        orders = query.order_by(
            ORMPurchaseOrder.order_date.desc()
        ).offset(skip).limit(limit).all()
        return orders
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

@po_router.get("/{po_number}", response_model=PurchaseOrderWithLineItems)
def get_purchase_order(po_number: str, db: Session = Depends(get_db)):
    """Get purchase order with line items."""
    po = db.query(ORMPurchaseOrder).options(
        joinedload(ORMPurchaseOrder.line_items),
        joinedload(ORMPurchaseOrder.supplier)
    ).filter(ORMPurchaseOrder.po_number == po_number).first()
    
    if not po:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Purchase order {po_number} not found"
        )
    return po

@po_router.post("/", response_model=PurchaseOrder, status_code=status.HTTP_201_CREATED)
def create_purchase_order(po: PurchaseOrderCreate, db: Session = Depends(get_db)):
    """Create a new purchase order."""
    # Verify supplier exists
    supplier = db.query(ORMSupplier).filter(
        ORMSupplier.supplier_id == po.supplier_id
    ).first()
    if not supplier:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Supplier {po.supplier_id} not found"
        )
    
    try:
        db_po = ORMPurchaseOrder(**po.model_dump())
        db.add(db_po)
        db.commit()
        db.refresh(db_po)
        return db_po
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Database constraint violation: {str(e)}"
        )

# ===== QUALITY CHECK ENDPOINTS =====

quality_router = APIRouter(prefix="/quality-checks", tags=["quality-checks"])

@quality_router.get("/", response_model=List[QualityCheck])
def list_quality_checks(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    batch_number: Optional[str] = None,
    result: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List quality checks."""
    try:
        query = db.query(ORMQualityCheck)
        
        if batch_number:
            query = query.filter(ORMQualityCheck.batch_number == batch_number)
        if result:
            query = query.filter(ORMQualityCheck.result == result)
        
        checks = query.order_by(
            ORMQualityCheck.inspection_date.desc()
        ).offset(skip).limit(limit).all()
        return checks
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

@quality_router.post("/", response_model=QualityCheck, status_code=status.HTTP_201_CREATED)
def create_quality_check(check: QualityCheckCreate, db: Session = Depends(get_db)):
    """Create a new quality check."""
    try:
        db_check = ORMQualityCheck(**check.model_dump())
        db.add(db_check)
        db.commit()
        db.refresh(db_check)
        return db_check
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Database constraint violation: {str(e)}"
        )

@quality_router.put("/{inspection_id}", response_model=QualityCheck)
def update_quality_check(
    inspection_id: str,
    update_data: QualityCheckUpdate,
    db: Session = Depends(get_db)
):
    """Update quality check result."""
    check = db.query(ORMQualityCheck).filter(
        ORMQualityCheck.inspection_id == inspection_id
    ).first()
    
    if not check:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quality check {inspection_id} not found"
        )
    
    try:
        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(check, field, value)
        
        check.updated_at = datetime.now()
        db.commit()
        db.refresh(check)
        return check
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update quality check: {str(e)}"
        )

# ===== ANOMALY DETECTION ENDPOINTS =====

anomaly_router = APIRouter(prefix="/anomalies", tags=["anomaly-detection"])

@anomaly_router.get("/", response_model=List[AnomalyDetection])
def list_anomalies(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = None,
    severity: Optional[str] = None,
    anomaly_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List anomaly detections."""
    try:
        query = db.query(ORMAnomalyDetection)
        
        if status:
            query = query.filter(ORMAnomalyDetection.status == status)
        if severity:
            query = query.filter(ORMAnomalyDetection.severity == severity)
        if anomaly_type:
            query = query.filter(ORMAnomalyDetection.anomaly_type == anomaly_type)
        
        anomalies = query.order_by(
            ORMAnomalyDetection.detection_timestamp.desc()
        ).offset(skip).limit(limit).all()
        return anomalies
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

@anomaly_router.post("/", response_model=AnomalyDetection, status_code=status.HTTP_201_CREATED)
def create_anomaly(anomaly: AnomalyDetectionCreate, db: Session = Depends(get_db)):
    """Create a new anomaly detection."""
    try:
        db_anomaly = ORMAnomalyDetection(**anomaly.model_dump())
        db.add(db_anomaly)
        db.commit()
        db.refresh(db_anomaly)
        return db_anomaly
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Database constraint violation: {str(e)}"
        )

@anomaly_router.put("/{anomaly_id}", response_model=AnomalyDetection)
def update_anomaly(
    anomaly_id: str,
    update_data: AnomalyDetectionUpdate,
    db: Session = Depends(get_db)
):
    """Update anomaly status."""
    anomaly = db.query(ORMAnomalyDetection).filter(
        ORMAnomalyDetection.anomaly_id == anomaly_id
    ).first()
    
    if not anomaly:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Anomaly {anomaly_id} not found"
        )
    
    try:
        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(anomaly, field, value)
        
        anomaly.updated_at = datetime.now()
        db.commit()
        db.refresh(anomaly)
        return anomaly
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update anomaly: {str(e)}"
        )

# ===== REGISTER ALL ROUTERS =====

router.include_router(product_router)
router.include_router(batch_router)
router.include_router(transaction_router)
router.include_router(storage_router)
router.include_router(supplier_router)
router.include_router(po_router)
router.include_router(quality_router)
router.include_router(anomaly_router)
