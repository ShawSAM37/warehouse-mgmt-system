"""
NFC Gate Entry Processor for Warehouse Management System.
Handles automated gate entry processing with NFC tag scanning, validation,
and integration with existing inventory management workflows.
"""
import asyncio
import logging
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy import and_, or_

# Project imports
from ..utils.db import get_db
from ..inventory.models import (
    InventoryBatch, InventoryTransaction, Product, StorageLocation, 
    PurchaseOrder, PurchaseOrderLineItem
)
from ..inventory.schemas import (
    InventoryBatch as InventoryBatchSchema,
    InventoryBatchCreate, InventoryTransactionCreate,
    ErrorResponse
)
from ..enums import TransactionType, StorageType, UnitOfMeasurement, StockType, POStatus

# Configure logging
logger = logging.getLogger(__name__)

# Configuration and Data Models
class GateType(str, Enum):
    """Types of warehouse gates."""
    INBOUND = "INBOUND"
    OUTBOUND = "OUTBOUND"
    QUALITY_CHECK = "QUALITY_CHECK"
    CROSS_DOCK = "CROSS_DOCK"

class NFCReaderType(str, Enum):
    """Supported NFC reader types."""
    ACR122U = "ACR122U"
    PN532 = "PN532"
    SIMULATED = "SIMULATED"

@dataclass
class GateConfig:
    """Configuration for individual gate."""
    gate_id: str
    gate_type: GateType
    reader_type: NFCReaderType
    default_storage_zone: str
    auto_assign_storage: bool = True
    require_po_validation: bool = True
    max_concurrent_reads: int = 5
    read_timeout_seconds: int = 30
    retry_attempts: int = 3

class GateEntryRequest(BaseModel):
    """Request model for gate entry processing."""
    nfc_uid: str = Field(..., min_length=8, max_length=32, description="NFC tag UID")
    gate_id: str = Field(..., description="Gate identifier")
    manual_override: bool = Field(default=False, description="Manual entry override")
    expected_po_number: Optional[str] = Field(None, description="Expected purchase order")
    override_storage_bin: Optional[str] = Field(None, description="Override storage location")
    notes: Optional[str] = Field(None, description="Additional notes")

    @validator('nfc_uid')
    def validate_nfc_uid(cls, v):
        if not v.replace('-', '').replace(':', '').isalnum():
            raise ValueError("NFC UID must contain only alphanumeric characters and separators")
        return v.upper().replace('-', '').replace(':', '')

class GateEntryResponse(BaseModel):
    """Response model for gate entry processing."""
    success: bool
    entry_id: str
    batch_number: Optional[str] = None
    transaction_id: Optional[str] = None
    assigned_storage_bin: Optional[str] = None
    processing_time_ms: int
    warnings: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.now)

class GateStatus(BaseModel):
    """Gate status information."""
    gate_id: str
    is_active: bool
    reader_connected: bool
    last_read_time: Optional[datetime]
    total_entries_today: int
    error_count: int
    current_load: int

# Abstract NFC Reader Interface
class NFCReaderInterface(ABC):
    """Abstract interface for NFC readers."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the NFC reader."""
        pass
    
    @abstractmethod
    async def read_tag(self, timeout_seconds: int = 30) -> Optional[str]:
        """Read NFC tag and return UID."""
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if reader is connected."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect the reader."""
        pass

class SimulatedNFCReader(NFCReaderInterface):
    """Simulated NFC reader for testing."""
    
    def __init__(self, gate_id: str):
        self.gate_id = gate_id
        self.connected = False
        self.simulation_delay = 2.0
    
    async def initialize(self) -> bool:
        """Initialize simulated reader."""
        await asyncio.sleep(0.5)  # Simulate initialization time
        self.connected = True
        logger.info(f"Simulated NFC reader initialized for gate {self.gate_id}")
        return True
    
    async def read_tag(self, timeout_seconds: int = 30) -> Optional[str]:
        """Simulate NFC tag read."""
        if not self.connected:
            raise Exception("Reader not connected")
        
        await asyncio.sleep(self.simulation_delay)
        # Generate simulated UID
        simulated_uid = f"SIM{uuid.uuid4().hex[:12].upper()}"
        logger.debug(f"Simulated NFC read: {simulated_uid}")
        return simulated_uid
    
    async def is_connected(self) -> bool:
        """Check connection status."""
        return self.connected
    
    async def disconnect(self):
        """Disconnect simulated reader."""
        self.connected = False
        logger.info(f"Simulated NFC reader disconnected for gate {self.gate_id}")

class RealNFCReader(NFCReaderInterface):
    """Real NFC reader implementation."""
    
    def __init__(self, gate_id: str, reader_type: NFCReaderType):
        self.gate_id = gate_id
        self.reader_type = reader_type
        self.connected = False
        self.device = None
    
    async def initialize(self) -> bool:
        """Initialize real NFC reader."""
        try:
            # Import NFC library only when needed
            import nfc
            
            # Initialize based on reader type
            if self.reader_type == NFCReaderType.ACR122U:
                self.device = nfc.ContactlessFrontend('usb:072f:2200')
            elif self.reader_type == NFCReaderType.PN532:
                self.device = nfc.ContactlessFrontend('usb:04cc:0531')
            else:
                self.device = nfc.ContactlessFrontend('usb')
            
            self.connected = True
            logger.info(f"Real NFC reader {self.reader_type} initialized for gate {self.gate_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize NFC reader for gate {self.gate_id}: {e}")
            self.connected = False
            return False
    
    async def read_tag(self, timeout_seconds: int = 30) -> Optional[str]:
        """Read real NFC tag."""
        if not self.connected or not self.device:
            raise Exception("Reader not connected")
        
        try:
            # Run NFC read in executor to avoid blocking
            loop = asyncio.get_event_loop()
            tag = await loop.run_in_executor(
                None, 
                lambda: self.device.connect(rdwr={'on-connect': lambda tag: False}, timeout=timeout_seconds)
            )
            
            if tag and hasattr(tag, 'uid'):
                uid = tag.uid.hex().upper()
                logger.debug(f"Real NFC read: {uid}")
                return uid
            
            return None
            
        except Exception as e:
            logger.error(f"NFC read error for gate {self.gate_id}: {e}")
            raise
    
    async def is_connected(self) -> bool:
        """Check real reader connection."""
        return self.connected and self.device is not None
    
    async def disconnect(self):
        """Disconnect real reader."""
        if self.device:
            try:
                self.device.close()
            except:
                pass
            self.device = None
        self.connected = False
        logger.info(f"Real NFC reader disconnected for gate {self.gate_id}")

# Configuration Management
class GateConfigManager:
    """Manages gate configurations."""
    
    def __init__(self):
        self.configs: Dict[str, GateConfig] = {}
        self._load_configurations()
    
    def _load_configurations(self):
        """Load gate configurations from environment or defaults."""
        # Default configurations
        default_configs = [
            GateConfig(
                gate_id="GATE-001",
                gate_type=GateType.INBOUND,
                reader_type=NFCReaderType.SIMULATED,
                default_storage_zone="Inbound Raw",
                require_po_validation=True
            ),
            GateConfig(
                gate_id="GATE-002",
                gate_type=GateType.INBOUND,
                reader_type=NFCReaderType.SIMULATED,
                default_storage_zone="Inbound OEM",
                require_po_validation=True
            ),
            GateConfig(
                gate_id="GATE-QC",
                gate_type=GateType.QUALITY_CHECK,
                reader_type=NFCReaderType.SIMULATED,
                default_storage_zone="Quality Control",
                require_po_validation=False
            )
        ]
        
        for config in default_configs:
            self.configs[config.gate_id] = config
        
        # Override with environment variables if present
        self._load_from_environment()
    
    def _load_from_environment(self):
        """Load configurations from environment variables."""
        # Example: GATE_001_TYPE=INBOUND
        # Example: GATE_001_READER=ACR122U
        for gate_id in self.configs.keys():
            env_prefix = gate_id.replace('-', '_')
            
            if os.getenv(f"{env_prefix}_TYPE"):
                self.configs[gate_id].gate_type = GateType(os.getenv(f"{env_prefix}_TYPE"))
            
            if os.getenv(f"{env_prefix}_READER"):
                self.configs[gate_id].reader_type = NFCReaderType(os.getenv(f"{env_prefix}_READER"))
    
    def get_config(self, gate_id: str) -> Optional[GateConfig]:
        """Get configuration for specific gate."""
        return self.configs.get(gate_id)
    
    def get_all_configs(self) -> Dict[str, GateConfig]:
        """Get all gate configurations."""
        return self.configs.copy()

# Main Gate Entry Processor
class GateEntryProcessor:
    """Main processor for gate entry operations."""
    
    def __init__(self):
        self.config_manager = GateConfigManager()
        self.readers: Dict[str, NFCReaderInterface] = {}
        self.gate_stats: Dict[str, Dict[str, Any]] = {}
        self.active_reads: Dict[str, int] = {}
        self._initialize_gates()
    
    def _initialize_gates(self):
        """Initialize all configured gates."""
        for gate_id, config in self.config_manager.get_all_configs().items():
            self.gate_stats[gate_id] = {
                "total_entries_today": 0,
                "error_count": 0,
                "last_read_time": None
            }
            self.active_reads[gate_id] = 0
    
    async def initialize_reader(self, gate_id: str) -> bool:
        """Initialize NFC reader for specific gate."""
        config = self.config_manager.get_config(gate_id)
        if not config:
            logger.error(f"No configuration found for gate {gate_id}")
            return False
        
        try:
            # Create appropriate reader
            if config.reader_type == NFCReaderType.SIMULATED:
                reader = SimulatedNFCReader(gate_id)
            else:
                reader = RealNFCReader(gate_id, config.reader_type)
            
            # Initialize reader
            if await reader.initialize():
                self.readers[gate_id] = reader
                logger.info(f"Reader initialized for gate {gate_id}")
                return True
            else:
                logger.error(f"Failed to initialize reader for gate {gate_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing reader for gate {gate_id}: {e}")
            return False
    
    async def process_gate_entry(
        self, 
        request: GateEntryRequest, 
        db: Session
    ) -> GateEntryResponse:
        """Process gate entry with comprehensive validation and integration."""
        start_time = datetime.now()
        entry_id = f"ENTRY-{start_time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        warnings = []
        
        try:
            # Validate gate configuration
            config = self.config_manager.get_config(request.gate_id)
            if not config:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid gate ID: {request.gate_id}"
                )
            
            # Check concurrent read limit
            if self.active_reads.get(request.gate_id, 0) >= config.max_concurrent_reads:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Gate {request.gate_id} is at maximum concurrent read capacity"
                )
            
            self.active_reads[request.gate_id] = self.active_reads.get(request.gate_id, 0) + 1
            
            try:
                # Check for duplicate entry
                existing_batch = db.query(InventoryBatch).filter(
                    InventoryBatch.handling_unit_id == request.nfc_uid,
                    InventoryBatch.is_active == True
                ).first()
                
                if existing_batch and not request.manual_override:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f"NFC tag already associated with batch {existing_batch.batch_number}"
                    )
                
                # Validate against purchase orders if required
                po_line_item = None
                if config.require_po_validation and not request.manual_override:
                    po_line_item = await self._validate_against_purchase_orders(
                        request, db, warnings
                    )
                
                # Determine product and storage location
                product_code, storage_bin = await self._determine_product_and_storage(
                    request, config, po_line_item, db, warnings
                )
                
                # Create inventory batch
                batch = await self._create_inventory_batch(
                    request, product_code, storage_bin, entry_id, db
                )
                
                # Create gate entry transaction
                transaction = await self._create_gate_transaction(
                    batch, request, entry_id, db
                )
                
                # Update statistics
                self._update_gate_statistics(request.gate_id)
                
                # Calculate processing time
                processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
                
                return GateEntryResponse(
                    success=True,
                    entry_id=entry_id,
                    batch_number=batch.batch_number,
                    transaction_id=transaction.transaction_id,
                    assigned_storage_bin=storage_bin,
                    processing_time_ms=processing_time,
                    warnings=warnings
                )
                
            finally:
                self.active_reads[request.gate_id] -= 1
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Gate entry processing error: {e}")
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal error processing gate entry: {str(e)}"
            )
    
    async def _validate_against_purchase_orders(
        self, 
        request: GateEntryRequest, 
        db: Session, 
        warnings: List[str]
    ) -> Optional[PurchaseOrderLineItem]:
        """Validate gate entry against expected purchase orders."""
        
        # Look for open purchase orders expecting delivery
        expected_deliveries = db.query(PurchaseOrderLineItem).join(PurchaseOrder).filter(
            and_(
                PurchaseOrder.status.in_([POStatus.OPEN, POStatus.PARTIALLY_RECEIVED]),
                PurchaseOrder.expected_delivery_date >= datetime.now().date(),
                PurchaseOrderLineItem.received_quantity < PurchaseOrderLineItem.ordered_quantity
            )
        ).all()
        
        if not expected_deliveries:
            warnings.append("No expected deliveries found for today")
            return None
        
        # If specific PO provided, validate against it
        if request.expected_po_number:
            po_line_item = db.query(PurchaseOrderLineItem).filter(
                PurchaseOrderLineItem.po_number == request.expected_po_number
            ).first()
            
            if not po_line_item:
                warnings.append(f"Purchase order {request.expected_po_number} not found")
            else:
                return po_line_item
        
        # Return first available delivery for automatic assignment
        return expected_deliveries[0] if expected_deliveries else None
    
    async def _determine_product_and_storage(
        self,
        request: GateEntryRequest,
        config: GateConfig,
        po_line_item: Optional[PurchaseOrderLineItem],
        db: Session,
        warnings: List[str]
    ) -> tuple[str, str]:
        """Determine product code and storage location."""
        
        # Determine product code
        if po_line_item:
            product_code = po_line_item.product_code
        else:
            # Default product based on gate type
            if config.gate_type == GateType.INBOUND:
                product_code = "RAW-MATERIAL-DEFAULT"
            else:
                product_code = "UNKNOWN-PRODUCT"
            warnings.append(f"Using default product code: {product_code}")
        
        # Validate product exists
        product = db.query(Product).filter(Product.product_code == product_code).first()
        if not product:
            warnings.append(f"Product {product_code} not found in system")
            product_code = "UNKNOWN-PRODUCT"
        
        # Determine storage location
        if request.override_storage_bin:
            storage_bin = request.override_storage_bin
        elif config.auto_assign_storage:
            storage_bin = await self._auto_assign_storage_location(
                config.default_storage_zone, db, warnings
            )
        else:
            storage_bin = "TEMP-HOLDING"
            warnings.append("Using temporary holding location")
        
        return product_code, storage_bin
    
    async def _auto_assign_storage_location(
        self, 
        zone: str, 
        db: Session, 
        warnings: List[str]
    ) -> str:
        """Automatically assign optimal storage location."""
        
        # Find available storage locations in the zone
        available_locations = db.query(StorageLocation).filter(
            and_(
                StorageLocation.zone == zone,
                StorageLocation.is_active == True,
                StorageLocation.capacity_available > 0
            )
        ).order_by(StorageLocation.capacity_available.desc()).all()
        
        if available_locations:
            return available_locations[0].storage_bin
        else:
            warnings.append(f"No available storage in zone {zone}")
            return "OVERFLOW-AREA"
    
    async def _create_inventory_batch(
        self,
        request: GateEntryRequest,
        product_code: str,
        storage_bin: str,
        entry_id: str,
        db: Session
    ) -> InventoryBatch:
        """Create inventory batch from gate entry."""
        
        # Generate batch number
        batch_number = f"{datetime.now().strftime('%Y%m%d')}-GATE-{request.nfc_uid[:8]}"
        
        # Create batch
        batch_data = InventoryBatchCreate(
            batch_number=batch_number,
            product_code=product_code,
            storage_type=StorageType.AMBIENT,  # Default, can be enhanced
            storage_bin=storage_bin,
            quantity=1.0,  # Default quantity, can be enhanced with scale integration
            unit_of_measurement=UnitOfMeasurement.UNITS,
            stock_type=StockType.UNRESTRICTED,
            goods_receipt_date=datetime.now(),
            goods_receipt_time=datetime.now(),
            handling_unit_id=request.nfc_uid,
            country_of_origin="US",  # Default, can be enhanced
            document_number=entry_id
        )
        
        batch = InventoryBatch(**batch_data.model_dump())
        db.add(batch)
        db.flush()  # Get ID without committing
        
        logger.info(f"Created inventory batch {batch_number} from gate entry {entry_id}")
        return batch
    
    async def _create_gate_transaction(
        self,
        batch: InventoryBatch,
        request: GateEntryRequest,
        entry_id: str,
        db: Session
    ) -> InventoryTransaction:
        """Create transaction record for gate entry."""
        
        transaction_id = f"TRX-{datetime.now().strftime('%Y%m%d')}-{entry_id[-6:]}"
        
        transaction_data = InventoryTransactionCreate(
            transaction_id=transaction_id,
            batch_number=batch.batch_number,
            transaction_type=TransactionType.RECEIPT,
            quantity=batch.quantity,
            transaction_date=datetime.now(),
            reference_document=entry_id,
            to_location=batch.storage_bin,
            performed_by=f"GATE-{request.gate_id}",
            notes=f"Automated gate entry via NFC tag {request.nfc_uid}"
        )
        
        transaction = InventoryTransaction(**transaction_data.model_dump())
        db.add(transaction)
        db.commit()  # Commit both batch and transaction
        
        logger.info(f"Created gate entry transaction {transaction_id}")
        return transaction
    
    def _update_gate_statistics(self, gate_id: str):
        """Update gate statistics."""
        if gate_id in self.gate_stats:
            self.gate_stats[gate_id]["total_entries_today"] += 1
            self.gate_stats[gate_id]["last_read_time"] = datetime.now()
    
    async def get_gate_status(self, gate_id: str) -> GateStatus:
        """Get current status of specific gate."""
        config = self.config_manager.get_config(gate_id)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Gate {gate_id} not found"
            )
        
        reader_connected = False
        if gate_id in self.readers:
            reader_connected = await self.readers[gate_id].is_connected()
        
        stats = self.gate_stats.get(gate_id, {})
        
        return GateStatus(
            gate_id=gate_id,
            is_active=gate_id in self.readers,
            reader_connected=reader_connected,
            last_read_time=stats.get("last_read_time"),
            total_entries_today=stats.get("total_entries_today", 0),
            error_count=stats.get("error_count", 0),
            current_load=self.active_reads.get(gate_id, 0)
        )
    
    async def shutdown(self):
        """Shutdown all gate readers."""
        for gate_id, reader in self.readers.items():
            try:
                await reader.disconnect()
                logger.info(f"Disconnected reader for gate {gate_id}")
            except Exception as e:
                logger.error(f"Error disconnecting reader for gate {gate_id}: {e}")
        
        self.readers.clear()

# Global processor instance
gate_processor = GateEntryProcessor()

# Dependency functions
async def get_gate_processor() -> GateEntryProcessor:
    """Dependency to get gate processor instance."""
    return gate_processor

# FastAPI Router
router = APIRouter(prefix="/gate-entry", tags=["Gate Entry"])

@router.post("/process", response_model=GateEntryResponse)
async def process_gate_entry(
    request: GateEntryRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    processor: GateEntryProcessor = Depends(get_gate_processor)
):
    """Process gate entry from NFC scan."""
    return await processor.process_gate_entry(request, db)

@router.get("/gates/{gate_id}/status", response_model=GateStatus)
async def get_gate_status(
    gate_id: str,
    processor: GateEntryProcessor = Depends(get_gate_processor)
):
    """Get status of specific gate."""
    return await processor.get_gate_status(gate_id)

@router.get("/gates", response_model=List[GateStatus])
async def list_all_gates(
    processor: GateEntryProcessor = Depends(get_gate_processor)
):
    """List status of all configured gates."""
    statuses = []
    for gate_id in processor.config_manager.get_all_configs().keys():
        status = await processor.get_gate_status(gate_id)
        statuses.append(status)
    return statuses

@router.post("/gates/{gate_id}/initialize")
async def initialize_gate_reader(
    gate_id: str,
    processor: GateEntryProcessor = Depends(get_gate_processor)
):
    """Initialize NFC reader for specific gate."""
    success = await processor.initialize_reader(gate_id)
    if success:
        return {"message": f"Reader initialized for gate {gate_id}"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize reader for gate {gate_id}"
        )

@router.post("/simulate-nfc-read")
async def simulate_nfc_read(
    gate_id: str,
    simulated_uid: Optional[str] = None,
    processor: GateEntryProcessor = Depends(get_gate_processor),
    db: Session = Depends(get_db)
):
    """Simulate NFC read for testing purposes."""
    if simulated_uid is None:
        simulated_uid = f"SIM{uuid.uuid4().hex[:12].upper()}"
    
    request = GateEntryRequest(
        nfc_uid=simulated_uid,
        gate_id=gate_id,
        notes="Simulated NFC read for testing"
    )
    
    return await processor.process_gate_entry(request, db)

# Application lifecycle management
@asynccontextmanager
async def gate_entry_lifespan():
    """Manage gate entry processor lifecycle."""
    # Startup
    logger.info("Initializing gate entry processors...")
    
    # Initialize readers for all configured gates
    for gate_id in gate_processor.config_manager.get_all_configs().keys():
        await gate_processor.initialize_reader(gate_id)
    
    yield
    
    # Shutdown
    logger.info("Shutting down gate entry processors...")
    await gate_processor.shutdown()

# Export for main application
__all__ = [
    "router",
    "gate_processor", 
    "gate_entry_lifespan",
    "GateEntryRequest",
    "GateEntryResponse",
    "GateStatus"
]
