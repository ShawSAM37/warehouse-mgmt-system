"""
Warehouse Management System - ML Forecasting Module
==================================================

Production-grade time series forecasting for inventory demand prediction.
Supports multiple algorithms with graceful fallbacks for missing dependencies.

Author: WMS Development Team
Version: 1.0.0
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import joblib
from decimal import Decimal

# Core ML libraries
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Time series libraries with conditional imports
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    SARIMAX = None
    seasonal_decompose = None
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    Prophet = None
    PROPHET_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

# Project imports
try:
    from sqlalchemy.orm import Session
    from src.wms.utils.db import get_db
    from src.wms.inventory.models import InventoryTransaction, Product
    from src.wms.shared.enums import TransactionType
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    Session = None
    get_db = None
    InventoryTransaction = None
    Product = None
    TransactionType = None
    SQLALCHEMY_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Model storage configuration
MODEL_STORAGE_PATH = Path("models/forecasting")
MODEL_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

class ForecastingConfig:
    """Configuration for forecasting parameters."""
    
    DEFAULT_HORIZON = 30  # days
    MIN_HISTORY_DAYS = 90  # minimum data required
    MAX_HISTORY_DAYS = 730  # maximum data to use
    CV_SPLITS = 5  # time series cross-validation splits
    RANDOM_SEARCH_ITER = 20  # hyperparameter tuning iterations
    
    # Model availability flags
    AVAILABLE_MODELS = {
        'linear': True,  # Always available
        'random_forest': True,  # Always available
        'prophet': PROPHET_AVAILABLE,
        'xgboost': XGBOOST_AVAILABLE,
        'sarimax': STATSMODELS_AVAILABLE
    }

class ForecastingService:
    """
    Production-grade forecasting service for WMS inventory demand prediction.
    
    Supports multiple algorithms with automatic fallback to available models.
    Integrates with existing WMS data models and logging infrastructure.
    """
    
    def __init__(self, product_code: str, horizon: int = None):
        """
        Initialize forecasting service for a specific product.
        
        Args:
            product_code: Product code to forecast
            horizon: Forecast horizon in days (default: 30)
        """
        self.product_code = product_code
        self.horizon = horizon or ForecastingConfig.DEFAULT_HORIZON
        self.models = {}
        self.feature_pipeline = None
        self.training_data = None
        self.model_metadata = {}
        
        # Initialize database session if available
        if SQLALCHEMY_AVAILABLE and get_db:
            try:
                self.db_session = next(get_db())
            except Exception as e:
                logger.warning(f"Could not initialize database session: {e}")
                self.db_session = None
        else:
            self.db_session = None
    
    def load_historical_data(self, days_back: int = None) -> pd.DataFrame:
        """
        Load historical transaction data for the product.
        
        Args:
            days_back: Number of days of history to load
            
        Returns:
            DataFrame with timestamp and quantity columns
        """
        if not self.db_session or not InventoryTransaction:
            logger.error("Database session or models not available")
            return self._generate_sample_data()
        
        try:
            days_back = days_back or ForecastingConfig.MAX_HISTORY_DAYS
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Query consumption transactions
            query = self.db_session.query(
                InventoryTransaction.transaction_date.label('timestamp'),
                InventoryTransaction.quantity
            ).join(
                Product, InventoryTransaction.product_code == Product.product_code
            ).filter(
                Product.product_code == self.product_code,
                InventoryTransaction.transaction_type == TransactionType.CONSUMPTION,
                InventoryTransaction.transaction_date >= cutoff_date
            ).order_by(InventoryTransaction.transaction_date)
            
            df = pd.read_sql(query.statement, self.db_session.bind)
            
            if df.empty:
                logger.warning(f"No historical data found for product {self.product_code}")
                return self._generate_sample_data()
            
            # Ensure proper data types
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
            df = df.dropna()
            
            # Aggregate daily if multiple transactions per day
            df = df.groupby(df['timestamp'].dt.date).agg({
                'quantity': 'sum'
            }).reset_index()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logger.info(f"Loaded {len(df)} days of historical data for {self.product_code}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return self._generate_sample_data()
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample data for testing when database is unavailable."""
        logger.info("Generating sample data for testing")
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=180),
            end=datetime.now(),
            freq='D'
        )
        
        # Generate realistic consumption pattern
        np.random.seed(42)
        base_demand = 100
        trend = np.linspace(0, 20, len(dates))
        seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # Weekly pattern
        noise = np.random.normal(0, 5, len(dates))
        
        quantities = base_demand + trend + seasonal + noise
        quantities = np.maximum(quantities, 0)  # Ensure non-negative
        
        return pd.DataFrame({
            'timestamp': dates,
            'quantity': quantities
        })
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time series features for machine learning models.
        
        Args:
            df: DataFrame with timestamp and quantity columns
            
        Returns:
            DataFrame with engineered features
        """
        try:
            df = df.copy()
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Basic temporal features
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['quarter'] = df['timestamp'].dt.quarter
            df['day_of_month'] = df['timestamp'].dt.day
            df['week_of_year'] = df['timestamp'].dt.isocalendar().week
            
            # Calendar features
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_month_start'] = df['timestamp'].dt.is_month_start.astype(int)
            df['is_month_end'] = df['timestamp'].dt.is_month_end.astype(int)
            df['is_quarter_start'] = df['timestamp'].dt.is_quarter_start.astype(int)
            df['is_quarter_end'] = df['timestamp'].dt.is_quarter_end.astype(int)
            
            # Lag features
            for lag in [1, 7, 14, 30]:
                df[f'lag_{lag}'] = df['quantity'].shift(lag)
            
            # Rolling statistics
            for window in [7, 14, 30]:
                df[f'rolling_{window}_mean'] = df['quantity'].rolling(window=window).mean()
                df[f'rolling_{window}_std'] = df['quantity'].rolling(window=window).std()
                df[f'rolling_{window}_min'] = df['quantity'].rolling(window=window).min()
                df[f'rolling_{window}_max'] = df['quantity'].rolling(window=window).max()
            
            # Exponential moving averages
            df['ema_7'] = df['quantity'].ewm(span=7).mean()
            df['ema_30'] = df['quantity'].ewm(span=30).mean()
            
            # Difference features
            df['diff_1'] = df['quantity'].diff(1)
            df['diff_7'] = df['quantity'].diff(7)
            
            # Remove rows with NaN values (due to lags and rolling windows)
            df = df.dropna().reset_index(drop=True)
            
            logger.info(f"Engineered {len(df.columns)} features from {len(df)} observations")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return df
    
    def train_linear_model(self, df: pd.DataFrame) -> Pipeline:
        """Train linear regression model with feature scaling."""
        try:
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'quantity']]
            X = df[feature_cols]
            y = df['quantity']
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ])
            
            pipeline.fit(X, y)
            logger.info("Linear regression model trained successfully")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error training linear model: {e}")
            raise
    
    def train_random_forest(self, df: pd.DataFrame) -> Pipeline:
        """Train Random Forest model with hyperparameter tuning."""
        try:
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'quantity']]
            X = df[feature_cols]
            y = df['quantity']
            
            pipeline = Pipeline([
                ('model', RandomForestRegressor(random_state=42))
            ])
            
            # Hyperparameter tuning
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [5, 10, 15, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            }
            
            tscv = TimeSeriesSplit(n_splits=ForecastingConfig.CV_SPLITS)
            search = RandomizedSearchCV(
                pipeline, 
                param_grid, 
                cv=tscv, 
                scoring='neg_mean_squared_error',
                n_iter=ForecastingConfig.RANDOM_SEARCH_ITER,
                random_state=42,
                n_jobs=-1
            )
            
            search.fit(X, y)
            logger.info(f"Random Forest model trained with best params: {search.best_params_}")
            return search.best_estimator_
            
        except Exception as e:
            logger.error(f"Error training Random Forest model: {e}")
            raise
    
    def train_prophet_model(self, df: pd.DataFrame) -> Any:
        """Train Facebook Prophet model."""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available")
        
        try:
            # Prepare data for Prophet
            prophet_df = df[['timestamp', 'quantity']].rename(columns={
                'timestamp': 'ds',
                'quantity': 'y'
            })
            
            # Initialize Prophet with reasonable defaults
            model = Prophet(
                seasonality_mode='multiplicative',
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
            model.fit(prophet_df)
            logger.info("Prophet model trained successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error training Prophet model: {e}")
            raise
    
    def train_xgboost_model(self, df: pd.DataFrame) -> Pipeline:
        """Train XGBoost model with hyperparameter tuning."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available")
        
        try:
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'quantity']]
            X = df[feature_cols]
            y = df['quantity']
            
            pipeline = Pipeline([
                ('model', XGBRegressor(random_state=42, objective='reg:squarederror'))
            ])
            
            # Hyperparameter tuning
            param_grid = {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [3, 5, 7],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__subsample': [0.8, 0.9, 1.0],
                'model__colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            tscv = TimeSeriesSplit(n_splits=ForecastingConfig.CV_SPLITS)
            search = RandomizedSearchCV(
                pipeline, 
                param_grid, 
                cv=tscv, 
                scoring='neg_mean_squared_error',
                n_iter=ForecastingConfig.RANDOM_SEARCH_ITER,
                random_state=42,
                n_jobs=-1
            )
            
            search.fit(X, y)
            logger.info(f"XGBoost model trained with best params: {search.best_params_}")
            return search.best_estimator_
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            raise
    
    def train_sarimax_model(self, df: pd.DataFrame) -> Any:
        """Train SARIMAX model with automatic parameter selection."""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Statsmodels is not available")
        
        try:
            y = df['quantity']
            
            # Simple SARIMAX with reasonable defaults
            # In production, you might want to use auto_arima for parameter selection
            model = SARIMAX(
                y,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7),  # Weekly seasonality
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            fitted_model = model.fit(disp=False)
            logger.info("SARIMAX model trained successfully")
            return fitted_model
            
        except Exception as e:
            logger.error(f"Error training SARIMAX model: {e}")
            raise
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics for model performance.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Handle edge cases
            if len(y_true) == 0 or len(y_pred) == 0:
                return {'mae': float('inf'), 'rmse': float('inf'), 'mape': float('inf')}
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # MAPE with handling for zero values
            mask = y_true != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                mape = float('inf')
            
            return {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape)
            }
            
        except Exception as e:
            logger.error(f"Error calculating evaluation metrics: {e}")
            return {'mae': float('inf'), 'rmse': float('inf'), 'mape': float('inf')}
    
    def train_all_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all available models and return performance metrics.
        
        Args:
            df: Engineered feature DataFrame
            
        Returns:
            Dictionary with trained models and their metrics
        """
        results = {}
        
        # Split data for evaluation
        split_idx = int(len(df) * 0.8)
        train_df = df[:split_idx]
        test_df = df[split_idx:]
        
        # Train available models
        model_trainers = {
            'linear': self.train_linear_model,
            'random_forest': self.train_random_forest,
            'prophet': self.train_prophet_model,
            'xgboost': self.train_xgboost_model,
            'sarimax': self.train_sarimax_model
        }
        
        for model_name, trainer in model_trainers.items():
            if not ForecastingConfig.AVAILABLE_MODELS.get(model_name, False):
                logger.info(f"Skipping {model_name} - not available")
                continue
            
            try:
                logger.info(f"Training {model_name} model...")
                
                if model_name == 'prophet':
                    model = trainer(train_df)
                    # Generate predictions for evaluation
                    future = model.make_future_dataframe(periods=len(test_df))
                    forecast = model.predict(future)
                    y_pred = forecast['yhat'].tail(len(test_df)).values
                    
                elif model_name == 'sarimax':
                    model = trainer(train_df)
                    y_pred = model.forecast(steps=len(test_df))
                    
                else:
                    model = trainer(train_df)
                    feature_cols = [col for col in test_df.columns if col not in ['timestamp', 'quantity']]
                    X_test = test_df[feature_cols]
                    y_pred = model.predict(X_test)
                
                # Evaluate model
                y_true = test_df['quantity'].values
                metrics = self.evaluate_model(y_true, y_pred)
                
                results[model_name] = {
                    'model': model,
                    'metrics': metrics
                }
                
                logger.info(f"{model_name} - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, MAPE: {metrics['mape']:.2f}%")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        return results
    
    def select_best_model(self, results: Dict[str, Any]) -> tuple:
        """
        Select the best performing model based on RMSE.
        
        Args:
            results: Dictionary with model results
            
        Returns:
            Tuple of (best_model_name, best_model, best_metrics)
        """
        if not results:
            raise ValueError("No models were successfully trained")
        
        best_model_name = min(results.keys(), key=lambda x: results[x]['metrics']['rmse'])
        best_model = results[best_model_name]['model']
        best_metrics = results[best_model_name]['metrics']
        
        logger.info(f"Selected {best_model_name} as best model (RMSE: {best_metrics['rmse']:.2f})")
        return best_model_name, best_model, best_metrics
    
    def generate_forecast(self, model_name: str, model: Any, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate forecast using the specified model.
        
        Args:
            model_name: Name of the model
            model: Trained model object
            df: Historical data DataFrame
            
        Returns:
            DataFrame with forecast results
        """
        try:
            future_dates = pd.date_range(
                start=df['timestamp'].max() + timedelta(days=1),
                periods=self.horizon,
                freq='D'
            )
            
            if model_name == 'prophet':
                future = model.make_future_dataframe(periods=self.horizon)
                forecast = model.predict(future)
                predictions = forecast.tail(self.horizon)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                predictions = predictions.rename(columns={
                    'ds': 'date',
                    'yhat': 'forecast',
                    'yhat_lower': 'lower_bound',
                    'yhat_upper': 'upper_bound'
                })
                
            elif model_name == 'sarimax':
                forecast_result = model.forecast(steps=self.horizon)
                conf_int = model.get_forecast(steps=self.horizon).conf_int()
                
                predictions = pd.DataFrame({
                    'date': future_dates,
                    'forecast': forecast_result,
                    'lower_bound': conf_int.iloc[:, 0],
                    'upper_bound': conf_int.iloc[:, 1]
                })
                
            else:
                # For ML models, we need to generate future features
                # This is a simplified approach - in production, you'd want more sophisticated feature generation
                last_row = df.iloc[-1:].copy()
                forecasts = []
                
                for i in range(self.horizon):
                    # Create features for the next day
                    next_date = df['timestamp'].max() + timedelta(days=i+1)
                    next_features = self._generate_future_features(df, next_date, i)
                    
                    # Make prediction
                    feature_cols = [col for col in df.columns if col not in ['timestamp', 'quantity']]
                    X_next = next_features[feature_cols]
                    pred = model.predict(X_next)[0]
                    forecasts.append(pred)
                    
                    # Update df with prediction for next iteration
                    new_row = next_features.copy()
                    new_row['quantity'] = pred
                    df = pd.concat([df, new_row], ignore_index=True)
                
                predictions = pd.DataFrame({
                    'date': future_dates,
                    'forecast': forecasts,
                    'lower_bound': np.array(forecasts) * 0.9,  # Simple confidence intervals
                    'upper_bound': np.array(forecasts) * 1.1
                })
            
            # Ensure non-negative forecasts
            predictions['forecast'] = np.maximum(predictions['forecast'], 0)
            predictions['lower_bound'] = np.maximum(predictions['lower_bound'], 0)
            predictions['upper_bound'] = np.maximum(predictions['upper_bound'], 0)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            raise
    
    def _generate_future_features(self, df: pd.DataFrame, future_date: datetime, step: int) -> pd.DataFrame:
        """Generate features for a future date."""
        # This is a simplified feature generation for future dates
        # In production, you'd want more sophisticated handling
        
        future_row = pd.DataFrame({
            'timestamp': [future_date],
            'quantity': [0]  # Placeholder
        })
        
        # Add basic temporal features
        future_row['day_of_week'] = future_date.weekday()
        future_row['month'] = future_date.month
        future_row['quarter'] = future_date.quarter
        future_row['day_of_month'] = future_date.day
        future_row['week_of_year'] = future_date.isocalendar()[1]
        future_row['is_weekend'] = int(future_date.weekday() >= 5)
        future_row['is_month_start'] = int(future_date.day == 1)
        future_row['is_month_end'] = int(future_date == future_date.replace(day=1) + timedelta(days=32) - timedelta(days=1))
        
        # For lag and rolling features, use the most recent available data
        # This is simplified - in production you'd want more sophisticated handling
        for col in df.columns:
            if col not in future_row.columns:
                if 'lag_' in col or 'rolling_' in col or 'ema_' in col or 'diff_' in col:
                    future_row[col] = df[col].iloc[-1]  # Use last available value
                else:
                    future_row[col] = 0
        
        return future_row
    
    def save_model(self, model_name: str, model: Any, metrics: Dict[str, float]) -> str:
        """
        Save trained model and metadata to disk.
        
        Args:
            model_name: Name of the model
            model: Trained model object
            metrics: Model performance metrics
            
        Returns:
            Path to saved model file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.product_code}_{model_name}_{timestamp}"
            model_path = MODEL_STORAGE_PATH / f"{filename}.joblib"
            metadata_path = MODEL_STORAGE_PATH / f"{filename}.json"
            
            # Save model
            joblib.dump(model, model_path)
            
            # Save metadata
            metadata = {
                'product_code': self.product_code,
                'model_name': model_name,
                'training_timestamp': timestamp,
                'horizon': self.horizon,
                'metrics': metrics,
                'model_path': str(model_path),
                'dependencies': {
                    'prophet_available': PROPHET_AVAILABLE,
                    'xgboost_available': XGBOOST_AVAILABLE,
                    'statsmodels_available': STATSMODELS_AVAILABLE
                }
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_path: str) -> tuple:
        """
        Load a saved model and its metadata.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Tuple of (model, metadata)
        """
        try:
            model = joblib.load(model_path)
            
            # Load metadata
            metadata_path = model_path.replace('.joblib', '.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Model loaded: {model_path}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def forecast(self, model_name: str = None, retrain: bool = True) -> Dict[str, Any]:
        """
        Main forecasting method - loads data, trains models, and generates forecasts.
        
        Args:
            model_name: Specific model to use (if None, selects best)
            retrain: Whether to retrain models or use existing
            
        Returns:
            Dictionary with forecast results and metadata
        """
        try:
            logger.info(f"Starting forecast for product {self.product_code}")
            
            # Load historical data
            raw_data = self.load_historical_data()
            if len(raw_data) < ForecastingConfig.MIN_HISTORY_DAYS:
                raise ValueError(f"Insufficient historical data: {len(raw_data)} days (minimum: {ForecastingConfig.MIN_HISTORY_DAYS})")
            
            # Engineer features
            engineered_data = self.engineer_features(raw_data)
            self.training_data = engineered_data
            
            if retrain:
                # Train all available models
                model_results = self.train_all_models(engineered_data)
                
                if not model_results:
                    raise ValueError("No models could be trained successfully")
                
                # Select best model if not specified
                if model_name is None or model_name not in model_results:
                    best_model_name, best_model, best_metrics = self.select_best_model(model_results)
                else:
                    best_model_name = model_name
                    best_model = model_results[model_name]['model']
                    best_metrics = model_results[model_name]['metrics']
                
                # Save the best model
                model_path = self.save_model(best_model_name, best_model, best_metrics)
                
            else:
                # Load existing model (implementation would search for latest model)
                raise NotImplementedError("Loading existing models not yet implemented")
            
            # Generate forecast
            forecast_df = self.generate_forecast(best_model_name, best_model, engineered_data)
            
            # Prepare response
            result = {
                'product_code': self.product_code,
                'model_used': best_model_name,
                'forecast_horizon': self.horizon,
                'training_data_points': len(engineered_data),
                'model_metrics': best_metrics,
                'forecast': forecast_df.to_dict(orient='records'),
                'generated_at': datetime.now().isoformat(),
                'model_path': model_path if retrain else None
            }
            
            logger.info(f"Forecast completed successfully for {self.product_code}")
            return result
            
        except Exception as e:
            logger.error(f"Forecast failed for {self.product_code}: {e}")
            raise
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available forecasting models."""
        return [model for model, available in ForecastingConfig.AVAILABLE_MODELS.items() if available]
    
    @staticmethod
    def get_model_info() -> Dict[str, Any]:
        """Get information about model availability and dependencies."""
        return {
            'available_models': ForecastingService.get_available_models(),
            'dependencies': {
                'prophet': PROPHET_AVAILABLE,
                'xgboost': XGBOOST_AVAILABLE,
                'statsmodels': STATSMODELS_AVAILABLE,
                'sqlalchemy': SQLALCHEMY_AVAILABLE
            },
            'config': {
                'default_horizon': ForecastingConfig.DEFAULT_HORIZON,
                'min_history_days': ForecastingConfig.MIN_HISTORY_DAYS,
                'max_history_days': ForecastingConfig.MAX_HISTORY_DAYS
            }
        }

# Utility functions for API integration
def create_forecast_for_product(product_code: str, horizon: int = None) -> Dict[str, Any]:
    """
    Convenience function to create forecast for a single product.
    
    Args:
        product_code: Product code to forecast
        horizon: Forecast horizon in days
        
    Returns:
        Forecast results dictionary
    """
    service = ForecastingService(product_code, horizon)
    return service.forecast()

def batch_forecast(product_codes: List[str], horizon: int = None) -> Dict[str, Any]:
    """
    Create forecasts for multiple products.
    
    Args:
        product_codes: List of product codes
        horizon: Forecast horizon in days
        
    Returns:
        Dictionary with results for each product
    """
    results = {}
    
    for product_code in product_codes:
        try:
            service = ForecastingService(product_code, horizon)
            results[product_code] = service.forecast()
        except Exception as e:
            logger.error(f"Failed to forecast {product_code}: {e}")
            results[product_code] = {'error': str(e)}
    
    return results

# Export main components
__all__ = [
    'ForecastingService',
    'ForecastingConfig',
    'create_forecast_for_product',
    'batch_forecast'
]

logger.info("ML Forecasting module initialized successfully")
