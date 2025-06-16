"""
Warehouse Management System - Anomaly Detection Module
====================================================

Production-grade hybrid anomaly detection system for inventory and operational monitoring.
Combines statistical, ML-based, and forecast-residual detection methods.

Author: WMS Development Team
Version: 1.0.0
"""

import os
import logging
import json
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any, Tuple
from pathlib import Path
import joblib
from dataclasses import dataclass
from enum import Enum

# Core ML libraries
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Optional dependencies with graceful fallbacks
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

# Project imports
try:
    from sqlalchemy.orm import Session
    from src.wms.utils.db import get_db
    from src.wms.utils.cache import CacheManager
    from src.wms.inventory.models import InventoryTransaction, QualityCheck, Supplier, Product
    from src.wms.shared.enums import TransactionType
    from src.wms.ml.forecasting import ForecastingService
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    Session = None
    get_db = None
    CacheManager = None
    InventoryTransaction = None
    QualityCheck = None
    Supplier = None
    Product = None
    TransactionType = None
    ForecastingService = None
    SQLALCHEMY_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Model storage configuration
ANOMALY_MODEL_PATH = Path("models/anomaly")
ANOMALY_MODEL_PATH.mkdir(parents=True, exist_ok=True)

class AnomalySeverity(str, Enum):
    """Anomaly severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NORMAL = "normal"

class DetectionMethod(str, Enum):
    """Available detection methods."""
    STATISTICAL = "statistical"
    ISOLATION_FOREST = "isolation_forest"
    LOF = "local_outlier_factor"
    ONE_CLASS_SVM = "one_class_svm"
    FORECAST_RESIDUAL = "forecast_residual"

@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection parameters."""
    
    # Severity thresholds (standard deviations)
    SEVERITY_THRESHOLDS = {
        AnomalySeverity.CRITICAL: {"threshold": 3.0, "action": "immediate"},
        AnomalySeverity.HIGH: {"threshold": 2.5, "action": "investigate_hours"},
        AnomalySeverity.MEDIUM: {"threshold": 2.0, "action": "review_24h"},
        AnomalySeverity.LOW: {"threshold": 1.5, "action": "monitor_trend"}
    }
    
    # Detection method weights for ensemble
    METHOD_WEIGHTS = {
        DetectionMethod.STATISTICAL: 0.3,
        DetectionMethod.ISOLATION_FOREST: 0.25,
        DetectionMethod.LOF: 0.2,
        DetectionMethod.FORECAST_RESIDUAL: 0.25
    }
    
    # Model parameters
    ISOLATION_FOREST_CONTAMINATION = float(os.getenv("ANOMALY_IF_CONTAMINATION", "0.1"))
    LOF_N_NEIGHBORS = int(os.getenv("ANOMALY_LOF_NEIGHBORS", "20"))
    STATISTICAL_THRESHOLD = float(os.getenv("ANOMALY_STAT_THRESHOLD", "2.5"))
    
    # Performance settings
    MIN_HISTORY_DAYS = int(os.getenv("ANOMALY_MIN_HISTORY", "30"))
    MAX_HISTORY_DAYS = int(os.getenv("ANOMALY_MAX_HISTORY", "365"))
    CACHE_TTL = int(os.getenv("ANOMALY_CACHE_TTL", "1800"))
    
    # Real-time processing
    ALERT_COOLDOWN_MINUTES = int(os.getenv("ANOMALY_ALERT_COOLDOWN", "30"))
    BATCH_SIZE = int(os.getenv("ANOMALY_BATCH_SIZE", "1000"))

class AnomalyDetectionService:
    """
    Production-grade anomaly detection service for WMS operations.
    
    Combines statistical, ML-based, and forecast-residual detection methods
    with ensemble voting and real-time processing capabilities.
    """
    
    def __init__(self, product_code: str = None):
        """
        Initialize anomaly detection service.
        
        Args:
            product_code: Specific product to analyze (None for all products)
        """
        self.product_code = product_code
        self.config = AnomalyConfig()
        self.models = {}
        self.feature_scaler = StandardScaler()
        self.training_data = None
        self.last_alert_time = {}
        
        # Initialize cache if available
        if CacheManager:
            try:
                self.cache = CacheManager()
            except Exception as e:
                logger.warning(f"Could not initialize cache: {e}")
                self.cache = None
        else:
            self.cache = None
        
        # Initialize database session if available
        if SQLALCHEMY_AVAILABLE and get_db:
            try:
                self.db_session = next(get_db())
            except Exception as e:
                logger.warning(f"Could not initialize database session: {e}")
                self.db_session = None
        else:
            self.db_session = None
        
        # Initialize forecasting service if available
        if ForecastingService and self.product_code:
            try:
                self.forecasting_service = ForecastingService(self.product_code)
            except Exception as e:
                logger.warning(f"Could not initialize forecasting service: {e}")
                self.forecasting_service = None
        else:
            self.forecasting_service = None
    
    def load_anomaly_data(self, days_back: int = None) -> pd.DataFrame:
        """
        Load data for anomaly detection from multiple sources.
        
        Args:
            days_back: Number of days of history to load
            
        Returns:
            DataFrame with features for anomaly detection
        """
        cache_key = f"anomaly_data:{self.product_code or 'all'}:{days_back or self.config.MAX_HISTORY_DAYS}"
        
        # Try cache first
        if self.cache:
            try:
                cached_data = self.cache.get(cache_key)
                if cached_data:
                    logger.info(f"Retrieved cached anomaly data")
                    return pd.read_json(cached_data)
            except Exception as e:
                logger.warning(f"Cache retrieval failed: {e}")
        
        if not self.db_session:
            logger.warning("Database session not available, generating sample data")
            return self._generate_sample_anomaly_data()
        
        try:
            days_back = days_back or self.config.MAX_HISTORY_DAYS
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Load inventory transactions
            inventory_query = self.db_session.query(
                InventoryTransaction.transaction_date.label('timestamp'),
                InventoryTransaction.product_code,
                InventoryTransaction.quantity,
                InventoryTransaction.transaction_type,
                Product.unit_of_measurement
            ).join(
                Product, InventoryTransaction.product_code == Product.product_code
            ).filter(
                InventoryTransaction.transaction_date >= cutoff_date
            )
            
            if self.product_code:
                inventory_query = inventory_query.filter(
                    Product.product_code == self.product_code
                )
            
            inventory_df = pd.read_sql(inventory_query.statement, self.db_session.bind)
            
            # Load quality check data
            quality_query = self.db_session.query(
                QualityCheck.check_date.label('timestamp'),
                QualityCheck.product_code,
                QualityCheck.result,
                QualityCheck.notes
            ).filter(
                QualityCheck.check_date >= cutoff_date
            )
            
            if self.product_code:
                quality_query = quality_query.filter(
                    QualityCheck.product_code == self.product_code
                )
            
            quality_df = pd.read_sql(quality_query.statement, self.db_session.bind)
            
            # Combine and process data
            combined_df = self._combine_data_sources(inventory_df, quality_df)
            
            # Cache the result
            if self.cache:
                try:
                    self.cache.set(cache_key, combined_df.to_json(), ttl=self.config.CACHE_TTL)
                except Exception as e:
                    logger.warning(f"Cache storage failed: {e}")
            
            logger.info(f"Loaded {len(combined_df)} records for anomaly detection")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error loading anomaly data: {e}")
            return self._generate_sample_anomaly_data()
    
    def _combine_data_sources(self, inventory_df: pd.DataFrame, quality_df: pd.DataFrame) -> pd.DataFrame:
        """Combine inventory and quality data into unified dataset."""
        try:
            # Process inventory data
            if not inventory_df.empty:
                inventory_df['timestamp'] = pd.to_datetime(inventory_df['timestamp'])
                inventory_df['quantity'] = pd.to_numeric(inventory_df['quantity'], errors='coerce')
                
                # Aggregate by product and date
                inventory_agg = inventory_df.groupby([
                    inventory_df['timestamp'].dt.date,
                    'product_code'
                ]).agg({
                    'quantity': ['sum', 'count', 'std'],
                    'transaction_type': lambda x: x.mode().iloc[0] if len(x) > 0 else None
                }).reset_index()
                
                # Flatten column names
                inventory_agg.columns = [
                    'date', 'product_code', 'total_quantity', 'transaction_count',
                    'quantity_std', 'primary_transaction_type'
                ]
                inventory_agg['timestamp'] = pd.to_datetime(inventory_agg['date'])
            else:
                inventory_agg = pd.DataFrame()
            
            # Process quality data
            if not quality_df.empty:
                quality_df['timestamp'] = pd.to_datetime(quality_df['timestamp'])
                quality_agg = quality_df.groupby([
                    quality_df['timestamp'].dt.date,
                    'product_code'
                ]).agg({
                    'result': lambda x: (x == 'PASS').mean(),  # Pass rate
                }).reset_index()
                
                quality_agg.columns = ['date', 'product_code', 'quality_pass_rate']
                quality_agg['timestamp'] = pd.to_datetime(quality_agg['date'])
            else:
                quality_agg = pd.DataFrame()
            
            # Merge datasets
            if not inventory_agg.empty and not quality_agg.empty:
                combined = pd.merge(
                    inventory_agg, quality_agg,
                    on=['date', 'product_code', 'timestamp'],
                    how='outer'
                )
            elif not inventory_agg.empty:
                combined = inventory_agg
                combined['quality_pass_rate'] = 1.0  # Default to perfect quality
            elif not quality_agg.empty:
                combined = quality_agg
                combined['total_quantity'] = 0.0
                combined['transaction_count'] = 0
                combined['quantity_std'] = 0.0
            else:
                combined = pd.DataFrame()
            
            # Fill missing values
            if not combined.empty:
                combined = combined.fillna({
                    'total_quantity': 0.0,
                    'transaction_count': 0,
                    'quantity_std': 0.0,
                    'quality_pass_rate': 1.0
                })
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining data sources: {e}")
            return pd.DataFrame()
    
    def _generate_sample_anomaly_data(self) -> pd.DataFrame:
        """Generate sample data for testing when database is unavailable."""
        logger.info("Generating sample anomaly data for testing")
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=90),
            end=datetime.now(),
            freq='D'
        )
        
        np.random.seed(42)
        n_days = len(dates)
        
        # Generate realistic patterns with anomalies
        base_quantity = 100
        seasonal = 20 * np.sin(2 * np.pi * np.arange(n_days) / 7)
        trend = np.linspace(0, 10, n_days)
        noise = np.random.
