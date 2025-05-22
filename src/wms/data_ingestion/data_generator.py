"""
Generate synthetic data for development and testing of the warehouse management system.
This module creates realistic test data that complies with the data models defined in inventory/models.py.
"""
import pandas as pd
import numpy as np
import random
import uuid
import re
import os
import json
import logging
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
import pydantic

# Import models
from ..inventory.models import (
    StorageType, StockType, UnitOfMeasurement, DocumentCategory, 
    TransactionType, POStatus, QualityCheckResult, AnomalyStatus,
    SeverityLevel, AnomalyType, InspectionType, Product, InventoryBatch,
    InventoryTransaction, StorageLocation, Supplier, PurchaseOrder,
    PurchaseOrderLineItem, QualityCheck, AnomalyDetection, ForecastResult,
    InventorySnapshot
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
VALID_COUNTRY_CODES = ["IN", "US", "CN", "DE", "JP", "KR", "FR", "IT", "BR", "GB", "CA", "AU", "SG", "AE", "ZA"]


class DataGenerator:
    """Generate synthetic data for the warehouse management system."""
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        self.today = datetime.now()
        self.product_codes = []
        self.storage_bins = []
        self.batch_numbers = []
        self.supplier_ids = []
        self.po_numbers = []
        logger.info(f"DataGenerator initialized with seed {seed}")
    
    def generate_product_codes(self, count: int = 100) -> List[str]:
        """Generate synthetic product codes."""
        logger.info(f"Generating {count} product codes")
        self.product_codes = [f"PROD-{i:04d}" for i in range(1, count+1)]
        return self.product_codes
    
    def generate_storage_bins(self, count: int = 50) -> List[str]:
        """Generate storage bin locations."""
        logger.info(f"Generating {count} storage bins")
        self.storage_bins = []
        for row in "ABCDE":
            for col in range(1, 11):
                for level in range(1, 3):
                    # Ensure format matches regex: r'^[A-Z]\d{2}(-\d)?$'
                    self.storage_bins.append(f"{row}{col:02d}-{level}")
        return self.storage_bins[:count]
    
    def generate_batch_numbers(self, count: int = 200) -> List[str]:
        """Generate batch numbers in format YYYYMMDD-XXX-####."""
        logger.info(f"Generating {count} batch numbers")
        self.batch_numbers = []
        for i in range(1, count+1):
            # Generate a random date in the past 90 days
            days_ago = random.randint(0, 90)
            batch_date = self.today - timedelta(days=days_ago)
            batch_type = random.choice(["RAW", "OEM", "SUB"])
            
            # Ensure format matches regex: r'^\d{8}-[A-Z]{3}-\d{4}$'
            batch_number = f"{batch_date.strftime('%Y%m%d')}-{batch_type}-{i:04d}"
            self.batch_numbers.append(batch_number)
        
        return self.batch_numbers
    
    def generate_supplier_ids(self, count: int = 20) -> List[str]:
        """Generate supplier IDs."""
        logger.info(f"Generating {count} supplier IDs")
        self.supplier_ids = [f"SUP-{i:03d}" for i in range(1, count+1)]
        return self.supplier_ids
    
    def generate_po_numbers(self, count: int = 50) -> List[str]:
        """Generate purchase order numbers."""
        logger.info(f"Generating {count} PO numbers")
        # Ensure format matches regex: r'^PO-\d{5,6}$'
        self.po_numbers = [f"PO-{i:05d}" for i in range(1, count+1)]
        return self.po_numbers
    
    def generate_products(self, count: int = 20) -> List[Product]:
        """Generate synthetic product data."""
        logger.info(f"Generating {count} products")
        if not self.product_codes:
            self.generate_product_codes(count)
            
        products = []
        for product_code in self.product_codes:
            is_raw = random.random() > 0.5
            handling_unit = f"RM-{random.randint(10000, 99999)}" if is_raw else None
            
            # Follow Pareto distribution for shelf life - 20% products have long shelf life
            if random.random() < 0.2:
                shelf_life = random.randint(180, 365)  # Long shelf life
            else:
                shelf_life = random.randint(30, 179)   # Shorter shelf life
            
            product = Product(
                product_code=product_code,
                handling_unit=handling_unit,
                unit_of_measurement=random.choice(list(UnitOfMeasurement)),
                description=f"{'Raw' if is_raw else 'OEM'} Material {product_code}",
                hsn_sac_code=f"{random.randint(1000, 9999)}",
                default_shelf_life_days=shelf_life,
                weight_per_unit=max(0.01, round(random.uniform(0.1, 100.0), 2)),
                volume_per_unit=max(0.001, round(random.uniform(0.01, 1.0), 3)),
                is_active=True,
                created_at=self.today - timedelta(days=random.randint(1, 365)),
                updated_at=self.today - timedelta(days=random.randint(0, 30))
            )
            products.append(product)
        
        return products
    
    def generate_storage_locations(self, count: int = 50) -> List[StorageLocation]:
        """Generate storage location data."""
        logger.info(f"Generating {count} storage locations")
        if not self.storage_bins:
            self.generate_storage_bins(count)
            
        storage_locations = []
        for bin_id in self.storage_bins:
            # Parse the bin ID to extract zone, aisle, rack, level
            match = re.match(r"([A-Z])(\d{2})-(\d)", bin_id)
            if match:
                zone_letter, rack_num, level = match.groups()
                
                # Determine zone based on letter
                zone_map = {
                    "A": "Inbound Raw",
                    "B": "Inbound OEM",
                    "C": "Production",
                    "D": "Quality Control",
                    "E": "Outbound"
                }
                zone = zone_map.get(zone_letter, "General")
                
                # Assign storage type based on zone
                if zone.startswith("Inbound Raw"):
                    storage_type = StorageType.BULK
                elif zone.startswith("Inbound OEM"):
                    storage_type = StorageType.RACK
                elif zone == "Production":
                    storage_type = random.choice([StorageType.RACK, StorageType.AMBIENT])
                elif zone == "Quality Control":
                    storage_type = StorageType.AMBIENT
                else:
                    storage_type = random.choice(list(StorageType))
                
                # Generate capacity based on storage type
                if storage_type == StorageType.BULK:
                    capacity_total = max(1.0, round(random.uniform(50.0, 100.0), 1))
                elif storage_type == StorageType.RACK:
                    capacity_total = max(1.0, round(random.uniform(10.0, 30.0), 1))
                elif storage_type == StorageType.COLD_STORAGE:
                    capacity_total = max(1.0, round(random.uniform(5.0, 15.0), 1))
                else:
                    capacity_total = max(1.0, round(random.uniform(8.0, 20.0), 1))
                
                # Randomize available capacity - ensure it's not greater than total
                capacity_available = max(0.1, round(random.uniform(0.1, capacity_total), 1))
                
                storage_location = StorageLocation(
                    storage_type=storage_type,
                    storage_bin=bin_id,
                    capacity_total=capacity_total,
                    capacity_available=capacity_available,
                    is_active=True,
                    zone=zone,
                    aisle=zone_letter,
                    rack=rack_num,
                    level=level,
                    created_at=self.today - timedelta(days=random.randint(30, 365)),
                    updated_at=self.today - timedelta(days=random.randint(0, 30))
                )
                storage_locations.append(storage_location)
        
        return storage_locations
    
    def generate_suppliers(self, count: int = 20) -> List[Supplier]:
        """Generate supplier data."""
        logger.info(f"Generating {count} suppliers")
        if not self.supplier_ids:
            self.generate_supplier_ids(count)
            
        suppliers = []
        company_types = ["Ltd.", "Inc.", "GmbH", "Pvt. Ltd.", "LLC", "Co."]
        domains = ["example.com", "supplier.net", "materials.co", "industry.org", "manufacturing.com"]
        
        for supplier_id in self.supplier_ids:
            company_name = f"{random.choice(['Alpha', 'Beta', 'Gamma', 'Delta', 'Omega', 'Prime', 'Global', 'National'])} {random.choice(['Materials', 'Industries', 'Manufacturing', 'Products', 'Supply'])} {random.choice(company_types)}"
            
            first_name = random.choice(["John", "Jane", "David", "Sarah", "Michael", "Emma", "Robert", "Lisa"])
            last_name = random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson"])
            contact_person = f"{first_name} {last_name}"
            
            domain = random.choice(domains)
            email = f"{first_name.lower()}.{last_name.lower()}@{domain}"
            
            # Ensure phone number matches validator
            phone = f"+{random.randint(1, 99)}-{random.randint(1000000000, 9999999999)}"
            
            supplier = Supplier(
                supplier_id=supplier_id,
                supplier_name=company_name,
                partner_number=f"P{random.randint(10000, 99999)}",
                contact_person=contact_person,
                email=email,
                phone=phone,
                address=f"{random.randint(1, 999)} {random.choice(['Main', 'Oak', 'Maple', 'Industrial', 'Commerce'])} {random.choice(['St', 'Ave', 'Blvd', 'Rd'])}, {random.choice(['New York', 'Chicago', 'Los Angeles', 'Houston', 'Mumbai', 'Delhi', 'Bangalore', 'Shanghai', 'Tokyo'])}",
                is_active=random.random() > 0.1,  # 90% active
                created_at=self.today - timedelta(days=random.randint(30, 365)),
                updated_at=self.today - timedelta(days=random.randint(0, 30))
            )
            suppliers.append(supplier)
        
        return suppliers
    
    def generate_purchase_orders(self, count: int = 50) -> Tuple[List[PurchaseOrder], List[PurchaseOrderLineItem]]:
        """Generate purchase orders and line items."""
        logger.info(f"Generating {count} purchase orders")
        if not self.po_numbers:
            self.generate_po_numbers(count)
        
        if not self.supplier_ids:
            self.generate_supplier_ids()
            
        if not self.product_codes:
            self.generate_product_codes()
            
        purchase_orders = []
        line_items = []
        
        for po_number in self.po_numbers:
            # Select a random supplier
            supplier_id = random.choice(self.supplier_ids)
            
            # Generate order date in the past 60 days
            days_ago = random.randint(0, 60)
            order_date = self.today - timedelta(days=days_ago)
            
            # Expected delivery is 7-30 days after order
            expected_delivery = order_date + timedelta(days=random.randint(7, 30))
            
            # Determine PO status based on dates
            if expected_delivery > self.today:
                status = POStatus.OPEN
            else:
                # If delivery date has passed, either partially received or closed
                status = random.choice([POStatus.PARTIALLY_RECEIVED, POStatus.CLOSED])
            
            po = PurchaseOrder(
                po_number=po_number,
                supplier_id=supplier_id,
                order_date=order_date,
                expected_delivery_date=expected_delivery,
                status=status,
                created_at=order_date,
                updated_at=self.today - timedelta(days=random.randint(0, days_ago))
            )
            purchase_orders.append(po)
            
            # Generate 1-5 line items per PO
            num_items = random.randint(1, 5)
            for i in range(1, num_items + 1):
                # Select a random product
                product_code = random.choice(self.product_codes)
                
                # Generate quantities
                ordered_quantity = max(1, random.randint(100, 1000))
                
                # If PO is partially received or closed, some quantity has been received
                if status != POStatus.OPEN:
                    received_quantity = random.randint(0, ordered_quantity)
                    if received_quantity == ordered_quantity:
                        item_status = POStatus.CLOSED
                    elif received_quantity > 0:
                        item_status = POStatus.PARTIALLY_RECEIVED
                    else:
                        item_status = POStatus.OPEN
                else:
                    received_quantity = 0
                    item_status = POStatus.OPEN
                
                line_item = PurchaseOrderLineItem(
                    po_line_item_id=f"{po_number}-{i}",
                    po_number=po_number,
                    product_code=product_code,
                    ordered_quantity=ordered_quantity,
                    unit_price=max(0.01, round(random.uniform(10.0, 1000.0), 2)),
                    received_quantity=received_quantity,
                    status=item_status,
                    created_at=order_date,
                    updated_at=self.today - timedelta(days=random.randint(0, days_ago))
                )
                line_items.append(line_item)
        
        return purchase_orders, line_items
    
    def generate_inventory_batches(self, count: int = 200) -> List[InventoryBatch]:
        """Generate inventory batch data."""
        logger.info(f"Generating {count} inventory batches")
        if not self.batch_numbers:
            self.generate_batch_numbers(count)
            
        if not self.product_codes:
            self.generate_product_codes()
            
        if not self.storage_bins:
            self.generate_storage_bins()
            
        batches = []
        for batch_number in self.batch_numbers:
            # Parse batch number to extract date and type
            match = re.match(r"(\d{8})-([A-Z]{3})-(\d{4})", batch_number)
            if match:
                date_str, batch_type, _ = match.groups()
                batch_date = datetime.strptime(date_str, "%Y%m%d")
                
                # Select a random product
                product_code = random.choice(self.product_codes)
                
                # Select a random storage bin
                storage_bin = random.choice(self.storage_bins)
                
                # Determine storage type based on bin
                if storage_bin.startswith("A"):
                    storage_type = StorageType.BULK
                elif storage_bin.startswith("B"):
                    storage_type = StorageType.RACK
                elif storage_bin.startswith("C"):
                    storage_type = random.choice([StorageType.RACK, StorageType.AMBIENT])
                elif storage_bin.startswith("D"):
                    storage_type = StorageType.AMBIENT
                elif storage_bin.startswith("E"):
                    storage_type = StorageType.COLD_STORAGE
                else:
                    storage_type = random.choice(list(StorageType))
                
                # Generate quantity - ensure positive
                quantity = max(0.01, round(random.uniform(10.0, 1000.0), 2))
                
                # Generate document number
                document_number = f"IBD-{random.randint(10000, 99999)}"
                
                # Determine unit of measurement
                unit_of_measurement = random.choice(list(UnitOfMeasurement))
                
                # Generate stock type (most are unrestricted)
                stock_type = random.choices(
                    list(StockType),
                    weights=[0.8, 0.15, 0.05],  # 80% unrestricted, 15% QI, 5% blocked
                    k=1
                )[0]
                
                # Generate goods receipt date and time
                goods_receipt_date = batch_date
                goods_receipt_time = batch_date.replace(
                    hour=random.randint(8, 17),
                    minute=random.randint(0, 59),
                    second=random.randint(0, 59)
                )
                
                # Generate expiry date (30-365 days after receipt)
                expiry_date = goods_receipt_date + timedelta(days=random.randint(30, 365))
                
                # Generate country of origin - ensure valid ISO 3166-1 alpha-2 code
                country_of_origin = random.choice(VALID_COUNTRY_CODES)
                
                # Generate loading metrics - ensure positive
                loading_weight = max(0.01, round(quantity * random.uniform(0.5, 2.0), 2))
                loading_volume = max(0.001, round(quantity * random.uniform(0.01, 0.1), 3))
                
                # Generate capacity metrics - ensure positive
                capacity_consumed = max(0.001, round(loading_volume * random.uniform(0.8, 1.2), 3))
                capacity_left = max(0.001, round(random.uniform(0.1, 10.0), 3))
                
                batch = InventoryBatch(
                    batch_number=batch_number,
                    product_code=product_code,
                    storage_type=storage_type,
                    storage_bin=storage_bin,
                    quantity=quantity,
                    document_number=document_number,
                    unit_of_measurement=unit_of_measurement,
                    stock_type=stock_type,
                    goods_receipt_date=goods_receipt_date,
                    goods_receipt_time=goods_receipt_time,
                    restricted_use=stock_type != StockType.UNRESTRICTED,
                    country_of_origin=country_of_origin,
                    expiry_date=expiry_date,
                    is_active=True,
                    handling_unit_id=f"HU-{random.randint(10000, 99999)}" if random.random() > 0.5 else None,
                    loading_weight=loading_weight,
                    loading_volume=loading_volume,
                    capacity_consumed=capacity_consumed,
                    capacity_left=capacity_left,
                    created_at=goods_receipt_date,
                    updated_at=self.today - timedelta(days=random.randint(0, (self.today - goods_receipt_date).days))
                )
                batches.append(batch)
        
        return batches
    
    def generate_consumption_quantity(self) -> float:
        """
        Generate consumption quantity following a Pareto distribution.
        80% of consumption comes from 20% of transactions.
        """
        if random.random() < 0.2:
            return max(0.01, round(random.uniform(50.0, 500.0), 2))  # High volume
        else:
            return max(0.01, round(random.uniform(5.0, 50.0), 2))    # Low volume
    
    def generate_inventory_transactions(self, count: int = 300) -> List[InventoryTransaction]:
        """Generate inventory transaction data."""
        logger.info(f"Generating {count} inventory transactions")
        if not self.batch_numbers:
            self.generate_batch_numbers()
            
        if not self.storage_bins:
            self.generate_storage_bins()
            
        transactions = []
        for i in range(1, count + 1):
            # Select a random batch
            batch_number = random.choice(self.batch_numbers)
            
            # Determine transaction type
            transaction_type = random.choice(list(TransactionType))
            
            # Generate transaction date in the past 30 days
            days_ago = random.randint(0, 30)
            transaction_date = self.today - timedelta(days=days_ago)
            
            # Generate quantity based on transaction type and Pareto distribution
            if transaction_type == TransactionType.RECEIPT:
                quantity = max(0.01, round(random.uniform(50.0, 500.0), 2))
                reference_document = f"GR-{random.randint(10000, 99999)}"
                from_location = None
                to_location = random.choice(self.storage_bins)
            elif transaction_type == TransactionType.CONSUMPTION:
                quantity = self.generate_consumption_quantity()
                reference_document = f"CONS-{random.randint(10000, 99999)}"
                from_location = random.choice(self.storage_bins)
                to_location = None
            elif transaction_type == TransactionType.TRANSFER:
                quantity = max(0.01, round(random.uniform(20.0, 200.0), 2))
                reference_document = f"TR-{random.randint(10000, 99999)}"
                from_location = random.choice(self.storage_bins)
                to_location = random.choice([bin for bin in self.storage_bins if bin != from_location])
            elif transaction_type == TransactionType.ADJUSTMENT:
                quantity = max(0.01, round(random.uniform(1.0, 50.0), 2))
                reference_document = f"ADJ-{random.randint(10000, 99999)}"
                from_location = random.choice(self.storage_bins) if random.random() > 0.5 else None
                to_location = random.choice(self.storage_bins) if from_location is None else None
            else:  # RETURN
                quantity = max(0.01, round(random.uniform(5.0, 50.0), 2))
                reference_document = f"RET-{random.randint(10000, 99999)}"
                from_location = None
                to_location = random.choice(self.storage_bins)
            
            # Ensure transaction ID matches regex: r'^TRX-\d{8}-\d{3}$'
            transaction_id = f"TRX-{transaction_date.strftime('%Y%m%d')}-{i:03d}"
            
            transaction = InventoryTransaction(
                transaction_id=transaction_id,
                batch_number=batch_number,
                transaction_type=transaction_type,
                quantity=quantity,
                transaction_date=transaction_date,
                reference_document=reference_document,
                from_location=from_location,
                to_location=to_location,
                performed_by=f"User-{random.randint(1000, 9999)}",
                notes=f"{'Automated' if random.random() > 0.3 else 'Manual'} {transaction_type.value.lower()} transaction",
                created_at=transaction_date,
                updated_at=transaction_date
            )
            transactions.append(transaction)
        
        return transactions
    
    def generate_quality_checks(self, count: int = 100) -> List[QualityCheck]:
        """Generate quality check data."""
        logger.info(f"Generating {count} quality checks")
        if not self.batch_numbers:
            self.generate_batch_numbers()
            
        quality_checks = []
        for i in range(1, count + 1):
            # Select a random batch
            batch_number = random.choice(self.batch_numbers)
            
            # Determine inspection type
            inspection_type = random.choice(list(InspectionType))
            
            # Generate inspection date in the past 30 days
            days_ago = random.randint(0, 30)
            inspection_date = self.today - timedelta(days=days_ago)
            
            # Determine result (mostly pass)
            result = random.choices(
                list(QualityCheckResult),
                weights=[0.8, 0.15, 0.05],  # 80% pass, 15% fail, 5% pending
                k=1
            )[0]
            
            # Ensure inspection ID matches regex: r'^QC-\d{5,6}$'
            inspection_id = f"QC-{random.randint(10000, 99999)}"
            
            quality_check = QualityCheck(
                inspection_id=inspection_id,
                batch_number=batch_number,
                inspection_type=inspection_type,
                inspection_date=inspection_date,
                inspector=f"Inspector-{random.randint(1000, 9999)}",
                result=result,
                notes=f"{'All parameters within acceptable range' if result == QualityCheckResult.PASS else 'Some parameters out of range' if result == QualityCheckResult.FAIL else 'Awaiting additional testing'}",
                created_at=inspection_date,
                updated_at=inspection_date
            )
            quality_checks.append(quality_check)
        
        return quality_checks
    
    def generate_anomaly_detections(self, count: int = 50) -> List[AnomalyDetection]:
        """Generate anomaly detection data."""
        logger.info(f"Generating {count} anomaly detections")
        if not self.product_codes:
            self.generate_product_codes()
            
        if not self.batch_numbers:
            self.generate_batch_numbers()
            
        anomalies = []
        for i in range(1, count + 1):
            # Determine anomaly type
            anomaly_type = random.choice(list(AnomalyType))
            
            # Generate detection timestamp in the past 14 days
            days_ago = random.randint(0, 14)
            detection_timestamp = self.today - timedelta(days=days_ago)
            
            # Determine related entity based on anomaly type
            if anomaly_type in [AnomalyType.STOCKOUT_RISK, AnomalyType.CONSUMPTION_SPIKE, AnomalyType.CONSUMPTION_DROP]:
                related_entity_id = random.choice(self.product_codes)
            elif anomaly_type in [AnomalyType.EXPIRY_RISK, AnomalyType.QUALITY_ISSUE]:
                related_entity_id = random.choice(self.batch_numbers)
            else:
                related_entity_id = random.choice(self.product_codes + self.batch_numbers)
            
            # Determine severity
            severity = random.choice(list(SeverityLevel))
            
            # Generate description based on anomaly type
            if anomaly_type == AnomalyType.STOCKOUT_RISK:
                description = f"Product {related_entity_id} inventory below safety stock level"
            elif anomaly_type == AnomalyType.EXPIRY_RISK:
                description = f"Batch {related_entity_id} approaching expiry with >50% remaining quantity"
            elif anomaly_type == AnomalyType.CONSUMPTION_SPIKE:
                description = f"Unusual {random.randint(30, 80)}% increase in daily consumption rate for {related_entity_id}"
            elif anomaly_type == AnomalyType.CONSUMPTION_DROP:
                description = f"Unusual {random.randint(30, 80)}% decrease in daily consumption rate for {related_entity_id}"
            elif anomaly_type == AnomalyType.QUALITY_ISSUE:
                description = f"Multiple quality check failures for batch {related_entity_id}"
            elif anomaly_type == AnomalyType.SPACE_CONSTRAINT:
                description = f"Storage capacity for {related_entity_id} approaching limit"
            elif anomaly_type == AnomalyType.DELIVERY_DELAY:
                description = f"Expected delivery for {related_entity_id} delayed by {random.randint(3, 15)} days"
            else:  # UNEXPECTED_RECEIPT
                description = f"Unplanned receipt of {related_entity_id} detected"
            
            # Determine status based on detection time
            if days_ago < 3:
                status = AnomalyStatus.NEW
            elif days_ago < 7:
                status = AnomalyStatus.INVESTIGATING
            else:
                status = AnomalyStatus.RESOLVED
            
            # Ensure anomaly ID matches regex: r'^ANM-\d{5,6}$'
            anomaly_id = f"ANM-{random.randint(10000, 99999)}"
            
            anomaly = AnomalyDetection(
                anomaly_id=anomaly_id,
                detection_timestamp=detection_timestamp,
                anomaly_type=anomaly_type,
                related_entity_id=related_entity_id,
                severity=severity,
                description=description,
                status=status,
                created_at=detection_timestamp,
                updated_at=self.today - timedelta(days=random.randint(0, days_ago))
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def generate_forecast_results(self, count: int = 100) -> List[ForecastResult]:
        """Generate forecast result data."""
        logger.info(f"Generating {count} forecast results")
        if not self.product_codes:
            self.generate_product_codes()
            
        forecasts = []
        for i in range(1, count + 1):
            # Select a random product
            product_code = random.choice(self.product_codes)
            
            # Generate forecast date in the past 7 days
            days_ago = random.randint(0, 7)
            forecast_date = self.today - timedelta(days=days_ago)
            
            # Determine forecast period
            forecast_period = random.choice(["Daily", "Weekly", "Monthly"])
            
            # Generate predicted quantity - ensure positive
            predicted_quantity = max(0.01, round(random.uniform(100.0, 5000.0), 2))
            
            # Generate confidence interval (±10-30%)
            confidence_range = predicted_quantity * random.uniform(0.1, 0.3)
            confidence_interval_lower = max(0, round(predicted_quantity - confidence_range, 2))
            confidence_interval_upper = round(predicted_quantity + confidence_range, 2)
            
            # Generate model version and accuracy
            model_version = random.choice(["XGBoost", "Prophet", "LSTM", "ARIMA"]) + f"-v{random.uniform(1.0, 2.0):.1f}"
            model_accuracy = round(random.uniform(0.7, 0.95), 2)
            
            # Ensure forecast ID matches regex: r'^FC-\d{8}-\d{3}$'
            forecast_id = f"FC-{forecast_date.strftime('%Y%m%d')}-{i:03d}"
            
            forecast = ForecastResult(
                forecast_id=forecast_id,
                product_code=product_code,
                forecast_date=forecast_date,
                forecast_period=forecast_period,
                predicted_quantity=predicted_quantity,
                confidence_interval_lower=confidence_interval_lower,
                confidence_interval_upper=confidence_interval_upper,
                model_version=model_version,
                model_accuracy=model_accuracy,
                created_at=forecast_date
            )
            forecasts.append(forecast)
        
        return forecasts
    
    def generate_inventory_snapshots(self, count: int = 100) -> List[InventorySnapshot]:
        """Generate inventory snapshot data."""
        logger.info(f"Generating {count} inventory snapshots")
        if not self.product_codes:
            self.generate_product_codes()
            
        snapshots = []
        for i in range(1, count + 1):
            # Select a random product
            product_code = random.choice(self.product_codes)
            
            # Generate snapshot date in the past 30 days
            days_ago = random.randint(0, 30)
            snapshot_date = (self.today - timedelta(days=days_ago)).date()
            
            # Generate total quantity - ensure positive
            total_quantity = max(0.01, round(random.uniform(500.0, 10000.0), 2))
            
            # Generate bins count
            bins_count = random.randint(1, 10)
            
            # Generate storage type distribution
            storage_types = {}
            remaining_qty = total_quantity
            for storage_type in list(StorageType)[:-1]:  # All but the last
                if remaining_qty > 0:
                    type_qty = round(random.uniform(0, remaining_qty), 2)
                    storage_types[storage_type.value] = type_qty
                    remaining_qty -= type_qty
            
            # Assign any remaining to the last type
            if remaining_qty > 0:
                storage_types[list(StorageType)[-1].value] = round(remaining_qty, 2)
            
            # Generate batch dates
            oldest_batch_date = self.today - timedelta(days=random.randint(30, 180))
            nearest_expiry_date = self.today + timedelta(days=random.randint(7, 90))
            
            # Generate days of supply
            days_of_supply = round(random.uniform(5.0, 45.0), 1)
            
            # Ensure snapshot ID is unique
            snapshot_id = f"SNP-{snapshot_date.strftime('%Y%m%d')}-{product_code[-4:]}"
            
            snapshot = InventorySnapshot(
                snapshot_id=snapshot_id,
                snapshot_date=snapshot_date,
                product_code=product_code,
                total_quantity=total_quantity,
                bins_count=bins_count,
                storage_types=storage_types,
                oldest_batch_date=oldest_batch_date,
                nearest_expiry_date=nearest_expiry_date,
                days_of_supply=days_of_supply,
                created_at=self.today - timedelta(days=days_ago)
            )
            snapshots.append(snapshot)
        
        return snapshots
    
    def simulate_sap_response(self, po_number: str) -> dict:
        """Simulate SAP API response for a purchase order."""
        logger.info(f"Simulating SAP response for PO {po_number}")
        # Mock SAP response
        return {
            "po": po_number,
            "status": "Success",
            "timestamp": datetime.now().isoformat(),
            "items": [
                {
                    "line_item": f"{po_number}-{i}",
                    "material": f"PROD-{random.randint(1, 9999):04d}",
                    "quantity": random.randint(10, 1000),
                    "unit": random.choice(["KG", "L", "Units"])
                } for i in range(1, random.randint(2, 6))
            ]
        }
    
    def json_serializer(self, obj: Any) -> str:
        """Custom JSON serializer for datetime objects."""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    def _save_to_file(self, data: List[Any], filepath: str, chunk_size: int = None) -> None:
        """Save data to a file with proper error handling."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Convert Pydantic models to dictionaries based on version
            if pydantic.__version__.startswith('1.'):
                data_dict = [item.dict() for item in data]
            else:
                data_dict = [item.model_dump() for item in data]
            
            # Write to file
            with open(filepath, "w") as f:
                json.dump(data_dict, f, indent=2, default=self.json_serializer)
            
            logger.info(f"Successfully saved {len(data)} records to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {e}")
            # Create a backup file if possible
            try:
                backup_path = f"{filepath}.backup"
                with open(backup_path, "w") as f:
                    json.dump(data_dict[:10], f, indent=2, default=self.json_serializer)
                logger.info(f"Created backup file with partial data: {backup_path}")
            except:
                logger.error("Failed to create backup file")
    
    def _save_to_csv(self, data: List[Any], filepath: str) -> None:
        """Save data to CSV file with error handling."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Convert Pydantic models to dictionaries based on version
            if pydantic.__version__.startswith('1.'):
                data_dict = [item.dict() for item in data]
            else:
                data_dict = [item.model_dump() for item in data]
            
            # Convert to DataFrame and save
            df = pd.DataFrame(data_dict)
            df.to_csv(filepath, index=False)
            
            logger.info(f"Successfully saved {len(data)} records to CSV {filepath}")
        except Exception as e:
            logger.error(f"Error saving data to CSV {filepath}: {e}")
    
    def generate_all_data(self, output_dir: str = "data/synthetic", chunk_size: int = None) -> Dict[str, int]:
        """Generate all types of data and save to files."""
        logger.info(f"Starting generation of all data to {output_dir}")
        start_time = datetime.now()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all data with progress reporting
        logger.info("Generating products...")
        products = self.generate_products(20)
        logger.info(f"Generated {len(products)} products")
        
        logger.info("Generating storage locations...")
        storage_locations = self.generate_storage_locations(50)
        logger.info(f"Generated {len(storage_locations)} storage locations")
        
        logger.info("Generating suppliers...")
        suppliers = self.generate_suppliers(20)
        logger.info(f"Generated {len(suppliers)} suppliers")
        
        logger.info("Generating purchase orders...")
        purchase_orders, line_items = self.generate_purchase_orders(50)
        logger.info(f"Generated {len(purchase_orders)} purchase orders with {len(line_items)} line items")
        
        logger.info("Generating inventory batches...")
        batches = self.generate_inventory_batches(200)
        logger.info(f"Generated {len(batches)} inventory batches")
        
        logger.info("Generating inventory transactions...")
        transactions = self.generate_inventory_transactions(300)
        logger.info(f"Generated {len(transactions)} inventory transactions")
        
        logger.info("Generating quality checks...")
        quality_checks = self.generate_quality_checks(100)
        logger.info(f"Generated {len(quality_checks)} quality checks")
        
        logger.info("Generating anomaly detections...")
        anomalies = self.generate_anomaly_detections(50)
        logger.info(f"Generated {len(anomalies)} anomaly detections")
        
        logger.info("Generating forecast results...")
        forecasts = self.generate_forecast_results(100)
        logger.info(f"Generated {len(forecasts)} forecast results")
        
        logger.info("Generating inventory snapshots...")
        snapshots = self.generate_inventory_snapshots(100)
        logger.info(f"Generated {len(snapshots)} inventory snapshots")
        
        # Save to JSON files
        self._save_to_file(products, f"{output_dir}/products.json", chunk_size)
        self._save_to_file(storage_locations, f"{output_dir}/storage_locations.json", chunk_size)
        self._save_to_file(suppliers, f"{output_dir}/suppliers.json", chunk_size)
        self._save_to_file(purchase_orders, f"{output_dir}/purchase_orders.json", chunk_size)
        self._save_to_file(line_items, f"{output_dir}/purchase_order_line_items.json", chunk_size)
        self._save_to_file(batches, f"{output_dir}/inventory_batches.json", chunk_size)
        self._save_to_file(transactions, f"{output_dir}/inventory_transactions.json", chunk_size)
        self._save_to_file(quality_checks, f"{output_dir}/quality_checks.json", chunk_size)
        self._save_to_file(anomalies, f"{output_dir}/anomaly_detections.json", chunk_size)
        self._save_to_file(forecasts, f"{output_dir}/forecast_results.json", chunk_size)
        self._save_to_file(snapshots, f"{output_dir}/inventory_snapshots.json", chunk_size)
        
        # Also save as CSV for easier data analysis
        self._save_to_csv(products, f"{output_dir}/products.csv")
        self._save_to_csv(storage_locations, f"{output_dir}/storage_locations.csv")
        self._save_to_csv(suppliers, f"{output_dir}/suppliers.csv")
        self._save_to_csv(purchase_orders, f"{output_dir}/purchase_orders.csv")
        self._save_to_csv(line_items, f"{output_dir}/purchase_order_line_items.csv")
        self._save_to_csv(batches, f"{output_dir}/inventory_batches.csv")
        self._save_to_csv(transactions, f"{output_dir}/inventory_transactions.csv")
        self._save_to_csv(quality_checks, f"{output_dir}/quality_checks.csv")
        self._save_to_csv(anomalies, f"{output_dir}/anomaly_detections.csv")
        self._save_to_csv(forecasts, f"{output_dir}/forecast_results.csv")
        self._save_to_csv(snapshots, f"{output_dir}/inventory_snapshots.csv")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Data generation completed in {duration:.2f} seconds")
        
        # Return counts
        return {
            "products": len(products),
            "storage_locations": len(storage_locations),
            "suppliers": len(suppliers),
            "purchase_orders": len(purchase_orders),
            "purchase_order_line_items": len(line_items),
            "inventory_batches": len(batches),
            "inventory_transactions": len(transactions),
            "quality_checks": len(quality_checks),
            "anomaly_detections": len(anomalies),
            "forecast_results": len(forecasts),
            "inventory_snapshots": len(snapshots)
        }


if __name__ == "__main__":
    # Generate sample data
    generator = DataGenerator(seed=42)
    result = generator.generate_all_data()
    print("Generated synthetic data with the following counts:")
    for key, count in result.items():
        print(f"  {key}: {count}")
