"""
FIFO inventory consumption logic for warehouse management system.
Provides transaction-safe batch consumption with full audit trail.
"""
import logging
import uuid
from datetime import datetime
from typing import List, Tuple, Optional
from decimal import Decimal, ROUND_HALF_UP
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from ..inventory.models import InventoryBatch, InventoryTransaction
from ..enums import TransactionType

logger = logging.getLogger(__name__)

def consume_fifo(
    db: Session,
    product_code: str,
    quantity: float,
    transaction_date: Optional[datetime] = None,
    reference_document: Optional[str] = None,
    performed_by: Optional[str] = None,
    from_location: Optional[str] = None,
    notes: Optional[str] = None
) -> List[Tuple[str, float]]:
    """
    Consumes inventory using FIFO (First-In-First-Out) method for a specific product.
    
    Args:
        db: SQLAlchemy database session for transaction management
        product_code: Product identifier to consume inventory from
        quantity: Total quantity to consume (must be positive)
        transaction_date: Timestamp for consumption (defaults to current time)
        reference_document: Optional reference document ID (e.g., order number)
        performed_by: User/system identifier performing the consumption
        from_location: Specific storage location being consumed from
        notes: Additional context or notes for the transaction
        
    Returns:
        List of tuples containing (batch_number, consumed_quantity) for each batch affected
        
    Raises:
        ValueError: If quantity <= 0 or insufficient stock available
        SQLAlchemyError: For database-related errors during transaction
        
    Example:
        >>> consumption_log = consume_fifo(
        ...     db=db_session,
        ...     product_code="PROD-001",
        ...     quantity=150.5,
        ...     performed_by="user-1234",
        ...     reference_document="ORDER-12345"
        ... )
        >>> print(consumption_log)
        [('20250522-RAW-0001', 100.0), ('20250523-RAW-0002', 50.5)]
    """
    # Fix mutable default parameter
    if transaction_date is None:
        transaction_date = datetime.now()
    
    # Parameter validation
    if quantity <= 0:
        logger.error(f"Invalid consumption quantity: {quantity} for product {product_code}")
        raise ValueError("Consumption quantity must be positive")
    
    logger.info(f"Starting FIFO consumption: product={product_code}, quantity={quantity}")
    
    try:
        # Query available batches using FIFO ordering
        available_batches = db.query(InventoryBatch).filter(
            InventoryBatch.product_code == product_code,
            InventoryBatch.is_active == True,
            InventoryBatch.quantity > 0
        ).order_by(
            InventoryBatch.goods_receipt_date.asc(),
            InventoryBatch.batch_number.asc()
        ).all()
        
        if not available_batches:
            logger.warning(f"No available batches found for product {product_code}")
            raise ValueError(f"No available inventory for product {product_code}")
        
        # Check total available stock
        total_available = sum(batch.quantity for batch in available_batches)
        if total_available < quantity:
            logger.error(f"Insufficient stock for {product_code}: available={total_available}, requested={quantity}")
            raise ValueError(
                f"Insufficient stock for product {product_code}. "
                f"Available: {total_available}, Requested: {quantity}"
            )
        
        # Process consumption using FIFO logic
        remaining_quantity = Decimal(str(quantity))
        consumption_log = []
        transaction_counter = _get_next_transaction_counter(db, transaction_date)
        
        for batch in available_batches:
            if remaining_quantity <= 0:
                break
            
            # Calculate consumable amount from this batch
            batch_available = Decimal(str(batch.quantity))
            consume_from_batch = min(batch_available, remaining_quantity)
            consume_from_batch_float = float(consume_from_batch)
            
            # Update batch quantity with precision handling
            new_batch_quantity = batch_available - consume_from_batch
            batch.quantity = float(new_batch_quantity.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))
            
            # Deactivate batch if fully consumed
            if batch.quantity <= 0:
                batch.is_active = False
                logger.info(f"Batch {batch.batch_number} fully consumed and deactivated")
            
            batch.updated_at = transaction_date
            
            # Generate unique transaction ID following established pattern
            transaction_id = f"TRX-{transaction_date.strftime('%Y%m%d')}-{transaction_counter:03d}"
            transaction_counter += 1
            
            # Create consumption transaction record
            consumption_transaction = InventoryTransaction(
                transaction_id=transaction_id,
                batch_number=batch.batch_number,
                transaction_type=TransactionType.CONSUMPTION,
                quantity=consume_from_batch_float,
                transaction_date=transaction_date,
                reference_document=reference_document,
                from_location=from_location or batch.storage_bin,
                to_location=None,  # Consumption has no destination
                performed_by=performed_by,
                notes=notes or f"FIFO consumption from batch {batch.batch_number}",
                created_at=transaction_date,
                updated_at=transaction_date
            )
            
            db.add(consumption_transaction)
            consumption_log.append((batch.batch_number, consume_from_batch_float))
            
            # Update remaining quantity
            remaining_quantity -= consume_from_batch
            
            logger.debug(f"Consumed {consume_from_batch_float} from batch {batch.batch_number}")
        
        # Commit all changes
        db.commit()
        
        logger.info(f"FIFO consumption completed: {len(consumption_log)} batches affected, total consumed: {quantity}")
        return consumption_log
        
    except SQLAlchemyError as e:
        logger.error(f"Database error during FIFO consumption: {str(e)}")
        db.rollback()
        raise
    except Exception as e:
        logger.error(f"Unexpected error during FIFO consumption: {str(e)}")
        db.rollback()
        raise


def get_available_stock(db: Session, product_code: str) -> float:
    """
    Get total available stock quantity for a product.
    
    Args:
        db: Database session
        product_code: Product identifier
        
    Returns:
        Total available quantity across all active batches
    """
    try:
        total_stock = db.query(
            db.func.sum(InventoryBatch.quantity)
        ).filter(
            InventoryBatch.product_code == product_code,
            InventoryBatch.is_active == True,
            InventoryBatch.quantity > 0
        ).scalar()
        
        return float(total_stock or 0)
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving available stock for {product_code}: {str(e)}")
        raise


def validate_consumption_request(
    db: Session,
    product_code: str,
    quantity: float
) -> Tuple[bool, str]:
    """
    Validate if a consumption request can be fulfilled.
    
    Args:
        db: Database session
        product_code: Product identifier
        quantity: Requested consumption quantity
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        if quantity <= 0:
            return False, "Consumption quantity must be positive"
        
        available_stock = get_available_stock(db, product_code)
        
        if available_stock < quantity:
            return False, f"Insufficient stock. Available: {available_stock}, Requested: {quantity}"
        
        return True, "Consumption request is valid"
        
    except Exception as e:
        logger.error(f"Error validating consumption request: {str(e)}")
        return False, f"Validation error: {str(e)}"


def get_fifo_batch_order(db: Session, product_code: str) -> List[dict]:
    """
    Get the FIFO order of batches for a product (for preview/planning purposes).
    
    Args:
        db: Database session
        product_code: Product identifier
        
    Returns:
        List of batch information dictionaries in FIFO order
    """
    try:
        batches = db.query(InventoryBatch).filter(
            InventoryBatch.product_code == product_code,
            InventoryBatch.is_active == True,
            InventoryBatch.quantity > 0
        ).order_by(
            InventoryBatch.goods_receipt_date.asc(),
            InventoryBatch.batch_number.asc()
        ).all()
        
        return [
            {
                "batch_number": batch.batch_number,
                "quantity": batch.quantity,
                "goods_receipt_date": batch.goods_receipt_date,
                "expiry_date": batch.expiry_date,
                "storage_bin": batch.storage_bin
            }
            for batch in batches
        ]
        
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving FIFO batch order for {product_code}: {str(e)}")
        raise


def get_consumption_history(
    db: Session,
    product_code: Optional[str] = None,
    batch_number: Optional[str] = None,
    days_back: int = 30
) -> List[dict]:
    """
    Retrieve consumption transaction history.
    
    Args:
        db: Database session
        product_code: Optional product filter
        batch_number: Optional batch filter
        days_back: Number of days to look back
        
    Returns:
        List of consumption transaction dictionaries
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        query = db.query(InventoryTransaction).filter(
            InventoryTransaction.transaction_type == TransactionType.CONSUMPTION,
            InventoryTransaction.transaction_date >= cutoff_date
        )
        
        if product_code:
            query = query.join(InventoryBatch).filter(
                InventoryBatch.product_code == product_code
            )
        
        if batch_number:
            query = query.filter(InventoryTransaction.batch_number == batch_number)
        
        transactions = query.order_by(
            InventoryTransaction.transaction_date.desc()
        ).all()
        
        return [
            {
                "transaction_id": txn.transaction_id,
                "batch_number": txn.batch_number,
                "quantity": txn.quantity,
                "transaction_date": txn.transaction_date,
                "reference_document": txn.reference_document,
                "performed_by": txn.performed_by,
                "notes": txn.notes
            }
            for txn in transactions
        ]
        
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving consumption history: {str(e)}")
        raise


def _get_next_transaction_counter(db: Session, transaction_date: datetime) -> int:
    """
    Get the next transaction counter for the given date.
    
    Args:
        db: Database session
        transaction_date: Date for transaction ID generation
        
    Returns:
        Next available counter for the date
    """
    try:
        date_str = transaction_date.strftime('%Y%m%d')
        pattern = f"TRX-{date_str}-%"
        
        last_transaction = db.query(InventoryTransaction).filter(
            InventoryTransaction.transaction_id.like(pattern)
        ).order_by(
            InventoryTransaction.transaction_id.desc()
        ).first()
        
        if last_transaction:
            # Extract counter from transaction ID and increment
            last_id = last_transaction.transaction_id
            counter = int(last_id.split('-')[-1]) + 1
        else:
            counter = 1
            
        return counter
        
    except Exception as e:
        logger.warning(f"Error getting transaction counter, using UUID fallback: {str(e)}")
        # Fallback to timestamp-based counter if pattern matching fails
        return int(transaction_date.timestamp() % 1000)


# Import required for consumption history function
from datetime import timedelta
