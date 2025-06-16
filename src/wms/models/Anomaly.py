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
        noise = np.random.normal(0, 5, n_days)
        
        # Add some anomalies
        anomaly_indices = np.random.choice(n_days, size=int(n_days * 0.05), replace=False)
        anomaly_multiplier = np.ones(n_days)
        anomaly_multiplier[anomaly_indices] = np.random.choice([0.1, 3.0], size=len(anomaly_indices))
        
        quantities = (base_quantity + seasonal + trend + noise) * anomaly_multiplier
        quantities = np.maximum(quantities, 0)
        
        return pd.DataFrame({
            'timestamp': dates,
            'product_code': self.product_code or 'SAMPLE-001',
            'total_quantity': quantities,
            'transaction_count': np.random.poisson(5, n_days),
            'quantity_std': np.random.exponential(2, n_days),
            'quality_pass_rate': np.random.beta(9, 1, n_days)  # Mostly high quality
        })
    
    def engineer_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features specifically for anomaly detection.
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        try:
            if df.empty:
                return df
            
            df = df.copy()
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Basic temporal features
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_month_end'] = df['timestamp'].dt.is_month_end.astype(int)
            
            # WMS-specific features
            df['consumption_rate'] = df['total_quantity'] / (df['transaction_count'] + 1)
            df['quantity_volatility'] = df['quantity_std'] / (df['total_quantity'] + 1)
            df['quality_score'] = df['quality_pass_rate']
            
            # Rolling statistics for trend detection
            for window in [7, 14, 30]:
                df[f'quantity_rolling_mean_{window}'] = df['total_quantity'].rolling(window=window).mean()
                df[f'quantity_rolling_std_{window}'] = df['total_quantity'].rolling(window=window).std()
                df[f'consumption_rate_rolling_mean_{window}'] = df['consumption_rate'].rolling(window=window).mean()
            
            # Lag features
            for lag in [1, 7]:
                df[f'quantity_lag_{lag}'] = df['total_quantity'].shift(lag)
                df[f'quality_lag_{lag}'] = df['quality_pass_rate'].shift(lag)
            
            # Change detection features
            df['quantity_change_1d'] = df['total_quantity'].diff(1)
            df['quantity_change_7d'] = df['total_quantity'].diff(7)
            df['quality_change_1d'] = df['quality_pass_rate'].diff(1)
            
            # Relative features
            df['quantity_vs_7d_avg'] = df['total_quantity'] / (df['quantity_rolling_mean_7'] + 1)
            df['quantity_vs_30d_avg'] = df['total_quantity'] / (df['quantity_rolling_mean_30'] + 1)
            
            # Remove rows with too many NaN values
            df = df.dropna(thresh=len(df.columns) * 0.7).reset_index(drop=True)
            
            logger.info(f"Engineered {len(df.columns)} features for anomaly detection")
            return df
            
        except Exception as e:
            logger.error(f"Error in anomaly feature engineering: {e}")
            return df
    
    def statistical_detection(self, data: pd.DataFrame) -> np.ndarray:
        """
        Perform statistical anomaly detection using Z-score and IQR methods.
        
        Args:
            data: Feature DataFrame
            
        Returns:
            Array of anomaly scores
        """
        try:
            if data.empty:
                return np.array([])
            
            # Select numeric features for statistical analysis
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['timestamp']]
            
            if not numeric_cols:
                return np.zeros(len(data))
            
            feature_data = data[numeric_cols].fillna(0)
            
            # Z-score based detection
            z_scores = np.abs(stats.zscore(feature_data, axis=0, nan_policy='omit'))
            z_score_anomalies = np.max(z_scores, axis=1)
            
            # IQR based detection
            iqr_anomalies = np.zeros(len(data))
            for col in numeric_cols:
                values = feature_data[col]
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (values < lower_bound) | (values > upper_bound)
                iqr_anomalies += outliers.astype(int)
            
            # Combine statistical methods
            combined_scores = (z_score_anomalies + iqr_anomalies / len(numeric_cols)) / 2
            
            logger.info(f"Statistical detection completed for {len(data)} samples")
            return combined_scores
            
        except Exception as e:
            logger.error(f"Error in statistical detection: {e}")
            return np.zeros(len(data))
    
    def isolation_forest_detection(self, data: pd.DataFrame) -> np.ndarray:
        """
        Perform anomaly detection using Isolation Forest.
        
        Args:
            data: Feature DataFrame
            
        Returns:
            Array of anomaly scores
        """
        try:
            if data.empty:
                return np.array([])
            
            # Select and prepare features
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['timestamp']]
            
            if not numeric_cols:
                return np.zeros(len(data))
            
            feature_data = data[numeric_cols].fillna(0)
            
            # Scale features
            scaled_data = self.feature_scaler.fit_transform(feature_data)
            
            # Train Isolation Forest
            iso_forest = IsolationForest(
                contamination=self.config.ISOLATION_FOREST_CONTAMINATION,
                random_state=42,
                n_jobs=-1
            )
            
            # Get anomaly scores (lower scores indicate anomalies)
            anomaly_scores = iso_forest.fit_predict(scaled_data)
            decision_scores = iso_forest.decision_function(scaled_data)
            
            # Convert to positive anomaly scores (higher = more anomalous)
            normalized_scores = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-8)
            anomaly_scores = 1 - normalized_scores  # Invert so higher = more anomalous
            
            logger.info(f"Isolation Forest detection completed for {len(data)} samples")
            return anomaly_scores
            
        except Exception as e:
            logger.error(f"Error in Isolation Forest detection: {e}")
            return np.zeros(len(data))
    
    def lof_detection(self, data: pd.DataFrame) -> np.ndarray:
        """
        Perform anomaly detection using Local Outlier Factor.
        
        Args:
            data: Feature DataFrame
            
        Returns:
            Array of anomaly scores
        """
        try:
            if data.empty:
                return np.array([])
            
            # Select and prepare features
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['timestamp']]
            
            if not numeric_cols or len(data) < self.config.LOF_N_NEIGHBORS:
                return np.zeros(len(data))
            
            feature_data = data[numeric_cols].fillna(0)
            
            # Scale features
            scaled_data = self.feature_scaler.fit_transform(feature_data)
            
            # Apply LOF
            n_neighbors = min(self.config.LOF_N_NEIGHBORS, len(data) - 1)
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
            
            # Get anomaly scores
            lof_scores = lof.fit_predict(scaled_data)
            negative_outlier_factor = lof.negative_outlier_factor_
            
            # Convert to positive anomaly scores
            normalized_scores = (negative_outlier_factor - negative_outlier_factor.min()) / (negative_outlier_factor.max() - negative_outlier_factor.min() + 1e-8)
            anomaly_scores = 1 - normalized_scores
            
            logger.info(f"LOF detection completed for {len(data)} samples")
            return anomaly_scores
            
        except Exception as e:
            logger.error(f"Error in LOF detection: {e}")
            return np.zeros(len(data))
    
    def forecast_residual_detection(self, data: pd.DataFrame) -> np.ndarray:
        """
        Perform anomaly detection based on forecasting residuals.
        
        Args:
            data: Feature DataFrame
            
        Returns:
            Array of anomaly scores
        """
        try:
            if data.empty or not self.forecasting_service:
                return np.zeros(len(data))
            
            # Get forecast predictions for comparison
            try:
                forecast_result = self.forecasting_service.forecast(retrain=False)
                forecast_data = pd.DataFrame(forecast_result['forecast'])
                
                if forecast_data.empty:
                    return np.zeros(len(data))
                
                # Align forecast with actual data
                forecast_data['date'] = pd.to_datetime(forecast_data['date'])
                data_with_date = data.copy()
                data_with_date['date'] = data_with_date['timestamp'].dt.date
                data_with_date['date'] = pd.to_datetime(data_with_date['date'])
                
                # Merge actual and forecast data
                merged = pd.merge(
                    data_with_date[['date', 'total_quantity']],
                    forecast_data[['date', 'forecast', 'lower_bound', 'upper_bound']],
                    on='date',
                    how='inner'
                )
                
                if merged.empty:
                    return np.zeros(len(data))
                
                # Calculate residuals and anomaly scores
                residuals = np.abs(merged['total_quantity'] - merged['forecast'])
                forecast_range = merged['upper_bound'] - merged['lower_bound']
                
                # Normalize residuals by forecast uncertainty
                normalized_residuals = residuals / (forecast_range + 1)
                
                # Convert to anomaly scores
                residual_scores = (normalized_residuals - normalized_residuals.min()) / (normalized_residuals.max() - normalized_residuals.min() + 1e-8)
                
                # Extend scores to match original data length
                if len(residual_scores) < len(data):
                    extended_scores = np.zeros(len(data))
                    extended_scores[:len(residual_scores)] = residual_scores
                    residual_scores = extended_scores
                elif len(residual_scores) > len(data):
                    residual_scores = residual_scores[:len(data)]
                
                logger.info(f"Forecast residual detection completed for {len(merged)} samples")
                return residual_scores
                
            except Exception as e:
                logger.warning(f"Could not get forecast predictions: {e}")
                return np.zeros(len(data))
            
        except Exception as e:
            logger.error(f"Error in forecast residual detection: {e}")
            return np.zeros(len(data))
    
    def ensemble_detection(self, results: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Combine multiple detection methods using weighted voting.
        
        Args:
            results: Dictionary of detection method results
            
        Returns:
            Dictionary with ensemble results and metadata
        """
        try:
            if not results:
                return {'ensemble_scores': np.array([]), 'method_contributions': {}}
            
            # Get the length of the longest result array
            max_length = max(len(scores) for scores in results.values() if len(scores) > 0)
            if max_length == 0:
                return {'ensemble_scores': np.array([]), 'method_contributions': {}}
            
            # Normalize all arrays to the same length
            normalized_results = {}
            for method, scores in results.items():
                if len(scores) == 0:
                    normalized_results[method] = np.zeros(max_length)
                elif len(scores) < max_length:
                    extended_scores = np.zeros(max_length)
                    extended_scores[:len(scores)] = scores
                    normalized_results[method] = extended_scores
                else:
                    normalized_results[method] = scores[:max_length]
            
            # Apply weighted voting
            ensemble_scores = np.zeros(max_length)
            total_weight = 0
            method_contributions = {}
            
            for method, scores in normalized_results.items():
                weight = self.config.METHOD_WEIGHTS.get(DetectionMethod(method), 0.2)
                ensemble_scores += weight * scores
                total_weight += weight
                method_contributions[method] = {
                    'weight': weight,
                    'mean_score': float(np.mean(scores)),
                    'max_score': float(np.max(scores)),
                    'anomaly_count': int(np.sum(scores > self.config.STATISTICAL_THRESHOLD))
                }
            
            # Normalize by total weight
            if total_weight > 0:
                ensemble_scores /= total_weight
            
            logger.info(f"Ensemble detection completed with {len(normalized_results)} methods")
            
            return {
                'ensemble_scores': ensemble_scores,
                'method_contributions': method_contributions,
                'total_weight': total_weight
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble detection: {e}")
            return {'ensemble_scores': np.array([]), 'method_contributions': {}}
    
    def classify_severity(self, anomaly_scores: np.ndarray) -> List[str]:
        """
        Classify anomaly severity based on scores.
        
        Args:
            anomaly_scores: Array of anomaly scores
            
        Returns:
            List of severity classifications
        """
        try:
            if len(anomaly_scores) == 0:
                return []
            
            severities = []
            for score in anomaly_scores:
                if score >= self.config.SEVERITY_THRESHOLDS[AnomalySeverity.CRITICAL]["threshold"]:
                    severities.append(AnomalySeverity.CRITICAL)
                elif score >= self.config.SEVERITY_THRESHOLDS[AnomalySeverity.HIGH]["threshold"]:
                    severities.append(AnomalySeverity.HIGH)
                elif score >= self.config.SEVERITY_THRESHOLDS[AnomalySeverity.MEDIUM]["threshold"]:
                    severities.append(AnomalySeverity.MEDIUM)
                elif score >= self.config.SEVERITY_THRESHOLDS[AnomalySeverity.LOW]["threshold"]:
                    severities.append(AnomalySeverity.LOW)
                else:
                    severities.append(AnomalySeverity.NORMAL)
            
            return severities
            
        except Exception as e:
            logger.error(f"Error classifying severity: {e}")
            return [AnomalySeverity.NORMAL] * len(anomaly_scores)
    
    def explain_anomaly(self, instance: pd.Series, feature_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Provide explanation for detected anomaly.
        
        Args:
            instance: Single data instance
            feature_data: Full feature dataset for context
            
        Returns:
            Dictionary with anomaly explanation
        """
        try:
            explanation = {
                'timestamp': instance.get('timestamp', datetime.now()).isoformat(),
                'product_code': instance.get('product_code', 'Unknown'),
                'anomaly_score': float(instance.get('anomaly_score', 0)),
                'severity': instance.get('severity', AnomalySeverity.NORMAL),
                'contributing_factors': [],
                'recommendations': []
            }
            
            # Analyze key metrics
            quantity = instance.get('total_quantity', 0)
            quality_rate = instance.get('quality_pass_rate', 1.0)
            consumption_rate = instance.get('consumption_rate', 0)
            
            # Compare with historical averages
            if not feature_data.empty:
                avg_quantity = feature_data['total_quantity'].mean()
                avg_quality = feature_data['quality_pass_rate'].mean()
                avg_consumption = feature_data['consumption_rate'].mean()
                
                # Identify contributing factors
                if abs(quantity - avg_quantity) > 2 * feature_data['total_quantity'].std():
                    factor = "Unusual quantity deviation"
                    if quantity > avg_quantity:
                        factor += f" (Current: {quantity:.1f}, Average: {avg_quantity:.1f}) - Demand spike detected"
                        explanation['recommendations'].append("Investigate demand drivers and ensure adequate supply")
                    else:
                        factor += f" (Current: {quantity:.1f}, Average: {avg_quantity:.1f}) - Low demand detected"
                        explanation['recommendations'].append("Review inventory levels and potential stockout causes")
                    explanation['contributing_factors'].append(factor)
                
                if quality_rate < avg_quality - 0.1:
                    factor = f"Quality degradation (Current: {quality_rate:.2f}, Average: {avg_quality:.2f})"
                    explanation['contributing_factors'].append(factor)
                    explanation['recommendations'].append("Investigate quality control processes and supplier performance")
                
                if abs(consumption_rate - avg_consumption) > 2 * feature_data['consumption_rate'].std():
                    factor = f"Abnormal consumption pattern (Current: {consumption_rate:.2f}, Average: {avg_consumption:.2f})"
                    explanation['contributing_factors'].append(factor)
                    explanation['recommendations'].append("Review transaction patterns and operational efficiency")
            
            # Add SHAP explanation if available
            if SHAP_AVAILABLE and hasattr(self, 'shap_explainer'):
                try:
                    # This would require a trained model and proper setup
                    pass
                except Exception as e:
                    logger.warning(f"SHAP explanation failed: {e}")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining anomaly: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'contributing_factors': [],
                'recommendations': []
            }
    
    def detect_anomalies(self, data_type: str = "all") -> Dict[str, Any]:
        """
        Main anomaly detection method combining all detection approaches.
        
        Args:
            data_type: Type of data to analyze ("all", "inventory", "quality")
            
        Returns:
            Dictionary with comprehensive anomaly detection results
        """
        try:
            logger.info(f"Starting anomaly detection for {data_type} data")
            
            # Load and prepare data
            raw_data = self.load_anomaly_data()
            if raw_data.empty:
                return {
                    'anomalies': [],
                    'summary': {'total_records': 0, 'anomalies_detected': 0},
                    'method_results': {},
                    'generated_at': datetime.now().isoformat()
                }
            
            # Engineer features
            feature_data = self.engineer_anomaly_features(raw_data)
            if feature_data.empty:
                return {
                    'anomalies': [],
                    'summary': {'total_records': len(raw_data), 'anomalies_detected': 0},
                    'method_results': {},
                    'generated_at': datetime.now().isoformat()
                }
            
            # Apply detection methods
            detection_results = {}
            
            # Statistical detection
            try:
                detection_results[DetectionMethod.STATISTICAL] = self.statistical_detection(feature_data)
            except Exception as e:
                logger.error(f"Statistical detection failed: {e}")
                detection_results[DetectionMethod.STATISTICAL] = np.zeros(len(feature_data))
            
            # Isolation Forest detection
            try:
                detection_results[DetectionMethod.ISOLATION_FOREST] = self.isolation_forest_detection(feature_data)
            except Exception as e:
                logger.error(f"Isolation Forest detection failed: {e}")
                detection_results[DetectionMethod.ISOLATION_FOREST] = np.zeros(len(feature_data))
            
            # LOF detection
            try:
                detection_results[DetectionMethod.LOF] = self.lof_detection(feature_data)
            except Exception as e:
                logger.error(f"LOF detection failed: {e}")
                detection_results[DetectionMethod.LOF] = np.zeros(len(feature_data))
            
            # Forecast residual detection
            try:
                detection_results[DetectionMethod.FORECAST_RESIDUAL] = self.forecast_residual_detection(feature_data)
            except Exception as e:
                logger.error(f"Forecast residual detection failed: {e}")
                detection_results[DetectionMethod.FORECAST_RESIDUAL] = np.zeros(len(feature_data))
            
            # Ensemble detection
            ensemble_result = self.ensemble_detection(detection_results)
            ensemble_scores = ensemble_result['ensemble_scores']
            
            # Classify severity
            severities = self.classify_severity(ensemble_scores)
            
            # Prepare anomaly records
            anomalies = []
            for i, (score, severity) in enumerate(zip(ensemble_scores, severities)):
                if severity != AnomalySeverity.NORMAL:
                    instance = feature_data.iloc[i]
                    instance_with_score = instance.copy()
                    instance_with_score['anomaly_score'] = score
                    instance_with_score['severity'] = severity
                    
                    explanation = self.explain_anomaly(instance_with_score, feature_data)
                    
                    anomaly_record = {
                        'id': f"anomaly_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        'timestamp': instance['timestamp'].isoformat() if pd.notna(instance['timestamp']) else datetime.now().isoformat(),
                        'product_code': instance.get('product_code', 'Unknown'),
                        'anomaly_score': float(score),
                        'severity': severity,
                        'data_point': {
                            'total_quantity': float(instance.get('total_quantity', 0)),
                            'transaction_count': int(instance.get('transaction_count', 0)),
                            'quality_pass_rate': float(instance.get('quality_pass_rate', 1.0)),
                            'consumption_rate': float(instance.get('consumption_rate', 0))
                        },
                        'explanation': explanation,
                        'detected_at': datetime.now().isoformat()
                    }
                    anomalies.append(anomaly_record)
            
            # Generate summary
            summary = {
                'total_records': len(feature_data),
                'anomalies_detected': len(anomalies),
                'severity_distribution': {
                    severity: severities.count(severity) for severity in AnomalySeverity
                },
                'detection_rate': len(anomalies) / len(feature_data) if len(feature_data) > 0 else 0
            }
            
            result = {
                'anomalies': anomalies,
                'summary': summary,
                'method_results': ensemble_result['method_contributions'],
                'ensemble_metadata': {
                    'total_weight': ensemble_result['total_weight'],
                    'methods_used': list(detection_results.keys())
                },
                'generated_at': datetime.now().isoformat(),
                'product_code': self.product_code
            }
            
            logger.info(f"Anomaly detection completed: {len(anomalies)} anomalies detected from {len(feature_data)} records")
            return result
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {
                'anomalies': [],
                'summary': {'total_records': 0, 'anomalies_detected': 0, 'error': str(e)},
                'method_results': {},
                'generated_at': datetime.now().isoformat(),
                'product_code': self.product_code
            }
    
    async def real_time_anomaly_check(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform real-time anomaly check on incoming transaction.
        
        Args:
            transaction_data: Single transaction data
            
        Returns:
            Real-time anomaly assessment
        """
        try:
            # Check alert cooldown
            product_code = transaction_data.get('product_code', 'unknown')
            current_time = datetime.now()
            last_alert = self.last_alert_time.get(product_code)
            
            if last_alert and (current_time - last_alert).seconds < self.config.ALERT_COOLDOWN_MINUTES * 60:
                return {'alert_suppressed': True, 'reason': 'cooldown_period'}
            
            # Quick anomaly assessment
            quantity = transaction_data.get('quantity', 0)
            
            # Load recent historical data for comparison
            recent_data = self.load_anomaly_data(days_back=30)
            if not recent_data.empty:
                avg_quantity = recent_data['total_quantity'].mean()
                std_quantity = recent_data['total_quantity'].std()
                
                # Simple z-score check
                z_score = abs(quantity - avg_quantity) / (std_quantity + 1e-8)
                
                if z_score > self.config.STATISTICAL_THRESHOLD:
                    severity = AnomalySeverity.HIGH if z_score > 3.0 else AnomalySeverity.MEDIUM
                    
                    # Update alert time
                    self.last_alert_time[product_code] = current_time
                    
                    return {
                        'anomaly_detected': True,
                        'severity': severity,
                        'z_score': float(z_score),
                        'transaction_data': transaction_data,
                        'alert_time': current_time.isoformat()
                    }
            
            return {'anomaly_detected': False, 'z_score': 0.0}
            
        except Exception as e:
            logger.error(f"Real-time anomaly check failed: {e}")
            return {'error': str(e), 'anomaly_detected': False}
    
    def save_anomaly_model(self, model_data: Dict[str, Any]) -> str:
        """
        Save anomaly detection model and metadata.
        
        Args:
            model_data: Model data and metadata
            
        Returns:
            Path to saved model
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"anomaly_model_{self.product_code or 'global'}_{timestamp}"
            model_path = ANOMALY_MODEL_PATH / f"{filename}.joblib"
            metadata_path = ANOMALY_MODEL_PATH / f"{filename}.json"
            
            # Save model
            joblib.dump(model_data, model_path)
            
            # Save metadata
            metadata = {
                'product_code': self.product_code,
                'model_type': 'hybrid_anomaly_detection',
                'training_timestamp': timestamp,
                'config': {
                    'severity_thresholds': self.config.SEVERITY_THRESHOLDS,
                    'method_weights': self.config.METHOD_WEIGHTS
                },
                'model_path': str(model_path)
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Anomaly model saved: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Error saving anomaly model: {e}")
            raise
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """Get list of available detection methods."""
        return [method.value for method in DetectionMethod]
    
    @staticmethod
    def get_severity_info() -> Dict[str, Any]:
        """Get information about severity levels and thresholds."""
        config = AnomalyConfig()
        return {
            'severity_levels': [severity.value for severity in AnomalySeverity],
            'thresholds': config.SEVERITY_THRESHOLDS,
            'method_weights': config.METHOD_WEIGHTS
        }

# Utility functions for API integration
def detect_anomalies_for_product(product_code: str) -> Dict[str, Any]:
    """
    Convenience function to detect anomalies for a single product.
    
    Args:
        product_code: Product code to analyze
        
    Returns:
        Anomaly detection results
    """
    service = AnomalyDetectionService(product_code)
    return service.detect_anomalies()

def batch_anomaly_detection(product_codes: List[str]) -> Dict[str, Any]:
    """
    Detect anomalies for multiple products.
    
    Args:
        product_codes: List of product codes
        
    Returns:
        Dictionary with results for each product
    """
    results = {}
    
    for product_code in product_codes:
        try:
            service = AnomalyDetectionService(product_code)
            results[product_code] = service.detect_anomalies()
        except Exception as e:
            logger.error(f"Failed to detect anomalies for {product_code}: {e}")
            results[product_code] = {'error': str(e)}
    
    return results

# Export main components
__all__ = [
    'AnomalyDetectionService',
    'AnomalyConfig',
    'AnomalySeverity',
    'DetectionMethod',
    'detect_anomalies_for_product',
    'batch_anomaly_detection'
]

logger.info("Anomaly Detection module initialized successfully")
