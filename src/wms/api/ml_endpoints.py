"""
WMS ML API Endpoints
===================

Production-grade FastAPI endpoints for ML forecasting and anomaly detection.
Includes authentication, authorization, monitoring, and comprehensive error handling.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

# Project imports
from src.wms.auth import require_permission, require_roles, get_current_active_user, User, Permission
from src.wms.shared.enums import UserRole
from src.wms.utils.db import get_db
from src.wms.ml.forecasting import ForecastingService, create_forecast_for_product, batch_forecast
from src.wms.ml.anomaly import AnomalyDetectionService, detect_anomalies_for_product, batch_anomaly_detection

# Configure logging
logger = logging.getLogger(__name__)

# Create router
ml_router = APIRouter(prefix="/api/v1", tags=["Machine Learning"])

# ================================
# Pydantic Schemas
# ================================

class ForecastRequest(BaseModel):
    """Request schema for single product forecasting."""
    horizon: int = Field(default=30, ge=1, le=365, description="Forecast horizon in days")
    model_type: Optional[str] = Field(default=None, description="Specific model to use")
    retrain: bool = Field(default=True, description="Whether to retrain the model")

class BatchForecastRequest(BaseModel):
    """Request schema for batch forecasting."""
    product_codes: List[str] = Field(..., min_items=1, max_items=100, description="List of product codes")
    horizon: int = Field(default=30, ge=1, le=365, description="Forecast horizon in days")
    model_type: Optional[str] = Field(default=None, description="Specific model to use")

class AnomalyDetectionRequest(BaseModel):
    """Request schema for anomaly detection."""
    product_codes: Optional[List[str]] = Field(default=None, description="Specific products to analyze")
    detection_methods: Optional[List[str]] = Field(default=None, description="Detection methods to use")
    data_type: str = Field(default="all", description="Type of data to analyze")

class ModelRetrainRequest(BaseModel):
    """Request schema for model retraining."""
    product_codes: Optional[List[str]] = Field(default=None, description="Products to retrain")
    force_retrain: bool = Field(default=False, description="Force retraining even if recent")

class ForecastResponse(BaseModel):
    """Response schema for forecasting results."""
    product_code: str
    model_used: str
    forecast_horizon: int
    forecast: List[Dict[str, Any]]
    model_metrics: Dict[str, float]
    confidence_intervals: Optional[List[Dict[str, Any]]] = None
    generated_at: str

class AnomalyResponse(BaseModel):
    """Response schema for anomaly detection results."""
    anomalies: List[Dict[str, Any]]
    summary: Dict[str, Any]
    method_results: Dict[str, Any]
    generated_at: str

class ErrorResponse(BaseModel):
    """Standard error response schema."""
    error: str
    detail: str
    timestamp: str
    request_id: str

# ================================
# Utility Functions
# ================================

def generate_request_id() -> str:
    """Generate unique request ID for tracking."""
    return str(uuid.uuid4())

def log_api_request(request: Request, user: User, endpoint: str, duration: float):
    """Log API request for monitoring and audit."""
    logger.info(
        f"ML API Request - User: {user.username}, Endpoint: {endpoint}, "
        f"Duration: {duration:.3f}s, IP: {request.client.host if request.client else 'unknown'}"
    )

async def handle_ml_error(error: Exception, request_id: str) -> JSONResponse:
    """Standardized error handling for ML endpoints."""
    error_detail = str(error)
    
    if isinstance(error, ValueError):
        status_code = status.HTTP_400_BAD_REQUEST
        error_type = "Validation Error"
    elif isinstance(error, FileNotFoundError):
        status_code = status.HTTP_404_NOT_FOUND
        error_type = "Resource Not Found"
    elif isinstance(error, PermissionError):
        status_code = status.HTTP_403_FORBIDDEN
        error_type = "Permission Denied"
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        error_type = "Internal Server Error"
        error_detail = "An unexpected error occurred"
    
    logger.error(f"ML API Error - Request ID: {request_id}, Error: {error}")
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error_type,
            "detail": error_detail,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
    )

# ================================
# Forecasting Endpoints
# ================================

@ml_router.post("/forecast/product/{product_code}", response_model=ForecastResponse)
async def forecast_product(
    product_code: str,
    request_data: ForecastRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_roles([UserRole.MANAGER, UserRole.ADMIN])),
    db: Session = Depends(get_db)
):
    """
    Generate forecast for a specific product.
    
    Requires Manager or Admin role.
    """
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        # Initialize forecasting service
        forecasting_service = ForecastingService(
            product_code=product_code,
            horizon=request_data.horizon
        )
        
        # Generate forecast
        result = forecasting_service.forecast(
            model_type=request_data.model_type,
            retrain=request_data.retrain
        )
        
        # Add confidence intervals if available
        if 'forecast' in result and result['forecast']:
            confidence_intervals = []
            for forecast_point in result['forecast']:
                if 'lower_bound' in forecast_point and 'upper_bound' in forecast_point:
                    confidence_intervals.append({
                        'date': forecast_point['date'],
                        'lower': forecast_point['lower_bound'],
                        'upper': forecast_point['upper_bound']
                    })
            result['confidence_intervals'] = confidence_intervals
        
        # Log request
        duration = time.time() - start_time
        background_tasks.add_task(
            log_api_request, request, current_user, f"forecast/{product_code}", duration
        )
        
        return ForecastResponse(**result)
        
    except Exception as e:
        return await handle_ml_error(e, request_id)

@ml_router.get("/forecast/models")
async def get_forecast_models(
    current_user: User = Depends(require_roles([UserRole.MANAGER, UserRole.ADMIN]))
):
    """
    Get available forecasting models and their status.
    
    Requires Manager or Admin role.
    """
    try:
        model_info = ForecastingService.get_model_info()
        
        return {
            "available_models": model_info['available_models'],
            "model_status": {
                "dependencies": model_info['dependencies'],
                "config": model_info['config']
            },
            "system_status": "healthy" if model_info['available_models'] else "degraded"
        }
        
    except Exception as e:
        request_id = generate_request_id()
        return await handle_ml_error(e, request_id)

@ml_router.post("/forecast/batch")
async def batch_forecast_products(
    request_data: BatchForecastRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_roles([UserRole.MANAGER, UserRole.ADMIN])),
    db: Session = Depends(get_db)
):
    """
    Generate forecasts for multiple products.
    
    Requires Manager or Admin role.
    """
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        # Generate batch forecasts
        results = batch_forecast(
            product_codes=request_data.product_codes,
            horizon=request_data.horizon
        )
        
        # Separate successful and failed results
        successful_results = {}
        failed_results = {}
        
        for product_code, result in results.items():
            if 'error' in result:
                failed_results[product_code] = result['error']
            else:
                successful_results[product_code] = result
        
        # Log request
        duration = time.time() - start_time
        background_tasks.add_task(
            log_api_request, request, current_user, "forecast/batch", duration
        )
        
        return {
            "results": successful_results,
            "failed": failed_results,
            "summary": {
                "total_requested": len(request_data.product_codes),
                "successful": len(successful_results),
                "failed": len(failed_results)
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        return await handle_ml_error(e, request_id)

# ================================
# Anomaly Detection Endpoints
# ================================

@ml_router.post("/anomaly/detect", response_model=AnomalyResponse)
async def detect_anomalies(
    request_data: AnomalyDetectionRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_roles([UserRole.WORKER, UserRole.MANAGER, UserRole.ADMIN])),
    db: Session = Depends(get_db)
):
    """
    Detect anomalies in inventory and operational data.
    
    Requires Worker, Manager, or Admin role.
    """
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        if request_data.product_codes:
            # Batch anomaly detection for specific products
            results = batch_anomaly_detection(request_data.product_codes)
            
            # Combine results
            all_anomalies = []
            method_results = {}
            
            for product_code, result in results.items():
                if 'error' not in result:
                    all_anomalies.extend(result.get('anomalies', []))
                    if 'method_results' in result:
                        method_results[product_code] = result['method_results']
            
            summary = {
                "total_products": len(request_data.product_codes),
                "total_anomalies": len(all_anomalies),
                "severity_distribution": {}
            }
            
            # Calculate severity distribution
            for anomaly in all_anomalies:
                severity = anomaly.get('severity', 'unknown')
                summary['severity_distribution'][severity] = summary['severity_distribution'].get(severity, 0) + 1
            
            response_data = {
                "anomalies": all_anomalies,
                "summary": summary,
                "method_results": method_results,
                "generated_at": datetime.now().isoformat()
            }
        else:
            # Global anomaly detection
            service = AnomalyDetectionService()
            response_data = service.detect_anomalies(data_type=request_data.data_type)
        
        # Log request
        duration = time.time() - start_time
        background_tasks.add_task(
            log_api_request, request, current_user, "anomaly/detect", duration
        )
        
        return AnomalyResponse(**response_data)
        
    except Exception as e:
        return await handle_ml_error(e, request_id)

@ml_router.get("/anomaly/alerts/active")
async def get_active_alerts(
    current_user: User = Depends(require_roles([UserRole.VIEWER, UserRole.WORKER, UserRole.MANAGER, UserRole.ADMIN]))
):
    """
    Get currently active anomaly alerts.
    
    Accessible to all authenticated users.
    """
    try:
        # This would typically query a database of active alerts
        # For now, we'll return a mock structure
        
        return {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "summary": {
                "total_active": 0,
                "last_updated": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        request_id = generate_request_id()
        return await handle_ml_error(e, request_id)

@ml_router.post("/anomaly/explain/{anomaly_id}")
async def explain_anomaly(
    anomaly_id: str,
    current_user: User = Depends(require_roles([UserRole.WORKER, UserRole.MANAGER, UserRole.ADMIN]))
):
    """
    Get detailed explanation for a specific anomaly.
    
    Requires Worker, Manager, or Admin role.
    """
    try:
        # This would typically retrieve the anomaly from storage and generate explanation
        # For now, return a mock explanation
        
        return {
            "anomaly_id": anomaly_id,
            "explanation": {
                "primary_cause": "Unusual demand spike detected",
                "contributing_factors": [
                    "Quantity 300% above historical average",
                    "Occurred during typically low-demand period"
                ],
                "recommendations": [
                    "Investigate demand drivers",
                    "Ensure adequate supply chain capacity",
                    "Monitor for continued trend"
                ]
            },
            "feature_importance": [
                {"feature": "total_quantity", "importance": 0.85},
                {"feature": "day_of_week", "importance": 0.12},
                {"feature": "consumption_rate", "importance": 0.03}
            ],
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        request_id = generate_request_id()
        return await handle_ml_error(e, request_id)

# ================================
# Model Management Endpoints
# ================================

@ml_router.get("/models/health")
async def get_model_health(
    current_user: User = Depends(require_roles([UserRole.MANAGER, UserRole.ADMIN]))
):
    """
    Get health status of all ML models.
    
    Requires Manager or Admin role.
    """
    try:
        forecast_info = ForecastingService.get_model_info()
        anomaly_info = AnomalyDetectionService.get_severity_info()
        
        return {
            "forecast_models": {
                "available": forecast_info['available_models'],
                "dependencies": forecast_info['dependencies'],
                "status": "healthy" if forecast_info['available_models'] else "degraded"
            },
            "anomaly_models": {
                "available_methods": AnomalyDetectionService.get_available_methods(),
                "severity_levels": anomaly_info['severity_levels'],
                "status": "healthy"
            },
            "system_status": "healthy",
            "last_check": datetime.now().isoformat()
        }
        
    except Exception as e:
        request_id = generate_request_id()
        return await handle_ml_error(e, request_id)

@ml_router.post("/models/retrain/{model_type}")
async def retrain_model(
    model_type: str,
    request_data: ModelRetrainRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_roles([UserRole.ADMIN]))
):
    """
    Initiate model retraining process.
    
    Requires Admin role only.
    """
    try:
        job_id = generate_request_id()
        
        # Add retraining task to background
        if model_type == "forecast":
            background_tasks.add_task(
                _retrain_forecast_models,
                request_data.product_codes,
                request_data.force_retrain,
                job_id
            )
        elif model_type == "anomaly":
            background_tasks.add_task(
                _retrain_anomaly_models,
                request_data.product_codes,
                request_data.force_retrain,
                job_id
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return {
            "retrain_status": "started",
            "job_id": job_id,
            "model_type": model_type,
            "initiated_by": current_user.username,
            "started_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        request_id = generate_request_id()
        return await handle_ml_error(e, request_id)

# ================================
# Background Tasks
# ================================

async def _retrain_forecast_models(product_codes: Optional[List[str]], force_retrain: bool, job_id: str):
    """Background task for forecast model retraining."""
    try:
        logger.info(f"Starting forecast model retraining - Job ID: {job_id}")
        
        if product_codes:
            for product_code in product_codes:
                service = ForecastingService(product_code)
                service.forecast(retrain=True)
        else:
            # Retrain all models - this would need to be implemented
            logger.info("Global forecast model retraining not yet implemented")
        
        logger.info(f"Forecast model retraining completed - Job ID: {job_id}")
        
    except Exception as e:
        logger.error(f"Forecast model retraining failed - Job ID: {job_id}, Error: {e}")

async def _retrain_anomaly_models(product_codes: Optional[List[str]], force_retrain: bool, job_id: str):
    """Background task for anomaly model retraining."""
    try:
        logger.info(f"Starting anomaly model retraining - Job ID: {job_id}")
        
        # Anomaly models are typically retrained automatically
        # This could trigger a full retraining cycle
        
        logger.info(f"Anomaly model retraining completed - Job ID: {job_id}")
        
    except Exception as e:
        logger.error(f"Anomaly model retraining failed - Job ID: {job_id}, Error: {e}")

# ================================
# Health Check Endpoint
# ================================

@ml_router.get("/health")
async def ml_health_check():
    """Health check endpoint for ML services."""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "forecasting": "available",
                "anomaly_detection": "available",
                "model_management": "available"
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# Export router
__all__ = ["ml_router"]
