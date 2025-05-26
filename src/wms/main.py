"""
Main FastAPI application entry point for Warehouse Management System.
Implements production-ready configuration with proper error handling,
middleware setup, and database lifecycle management.
"""
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

# Internal imports
from .utils.db import init_db, check_database_health
from .api.endpoints import router as api_router
from .inventory.schemas import ErrorResponse

# Configure logging
def setup_logging() -> logging.Logger:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("warehouse_management.log")
        ]
    )
    
    # Configure uvicorn logger
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(logging.INFO)
    
    return logging.getLogger(__name__)

logger = setup_logging()

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown lifecycle.
    Handles database initialization and cleanup.
    """
    # Startup logic
    logger.info("Starting Warehouse Management System...")
    
    try:
        # Initialize database tables
        logger.info("Initializing database...")
        init_db()
        logger.info("Database initialized successfully")
        
        # Verify database connectivity
        if check_database_health():
            logger.info("Database health check passed")
        else:
            logger.error("Database health check failed")
            raise Exception("Database connection unavailable")
            
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    
    yield
    
    # Shutdown logic
    logger.info("Shutting down Warehouse Management System...")
    # Add any cleanup logic here if needed
    logger.info("Application shutdown completed")

def setup_middleware(app: FastAPI) -> None:
    """Configure application middleware."""
    
    # CORS middleware - configure origins for production
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware for security
    allowed_hosts = os.getenv("ALLOWED_HOSTS", "*").split(",")
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=allowed_hosts
    )

def setup_exception_handlers(app: FastAPI) -> None:
    """Configure global exception handlers."""
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle Pydantic validation errors."""
        logger.warning(f"Validation error on {request.url}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="Validation Error",
                message=f"Request validation failed: {str(exc)}",
                path=str(request.url.path)
            ).model_dump()
        )
    
    @app.exception_handler(SQLAlchemyError)
    async def database_exception_handler(request: Request, exc: SQLAlchemyError):
        """Handle database-related errors."""
        logger.error(f"Database error on {request.url}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Database Error",
                message="A database error occurred. Please try again later.",
                path=str(request.url.path)
            ).model_dump()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other unhandled exceptions."""
        logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Internal Server Error",
                message="An unexpected error occurred. Please contact support if the problem persists.",
                path=str(request.url.path)
            ).model_dump()
        )

def setup_routers(app: FastAPI) -> None:
    """Configure application routers and endpoints."""
    
    # Include main API router with versioning
    app.include_router(
        api_router,
        prefix="/api/v1",
        tags=["Warehouse Management API v1"]
    )
    
    # Root endpoint
    @app.get("/", tags=["System"], summary="Root endpoint")
    async def root() -> Dict[str, Any]:
        """Root endpoint providing basic service information."""
        return {
            "service": "Warehouse Management System",
            "description": "AI-powered warehouse management with inventory optimization",
            "version": "1.0.0",
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "docs_url": "/docs",
            "api_base": "/api/v1"
        }
    
    # Health check endpoint
    @app.get("/health", tags=["System"], summary="Health check")
    async def health_check() -> Dict[str, Any]:
        """Comprehensive health check endpoint."""
        try:
            db_healthy = check_database_health()
            overall_status = "healthy" if db_healthy else "unhealthy"
            
            return {
                "status": overall_status,
                "timestamp": datetime.now().isoformat(),
                "checks": {
                    "database": "connected" if db_healthy else "disconnected",
                    "api": "operational"
                },
                "version": "1.0.0"
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "checks": {
                    "database": "error",
                    "api": "operational"
                },
                "error": str(e),
                "version": "1.0.0"
            }
    
    # API info endpoint
    @app.get("/api/info", tags=["System"], summary="API information")
    async def api_info() -> Dict[str, Any]:
        """Provide detailed API information."""
        return {
            "api_version": "v1",
            "service": "Warehouse Management System",
            "features": [
                "Product Management",
                "Inventory Tracking",
                "Batch Management",
                "Quality Control",
                "Anomaly Detection",
                "Purchase Order Management",
                "Storage Location Management"
            ],
            "endpoints": {
                "products": "/api/v1/products",
                "inventory_batches": "/api/v1/inventory/batches",
                "transactions": "/api/v1/inventory/transactions",
                "storage": "/api/v1/storage",
                "suppliers": "/api/v1/suppliers",
                "purchase_orders": "/api/v1/purchase-orders",
                "quality_checks": "/api/v1/quality-checks",
                "anomalies": "/api/v1/anomalies"
            },
            "documentation": {
                "swagger_ui": "/docs",
                "redoc": "/redoc",
                "openapi_spec": "/openapi.json"
            }
        }

def create_application() -> FastAPI:
    """
    Application factory function.
    Creates and configures the FastAPI application.
    """
    # Get environment configuration
    environment = os.getenv("ENVIRONMENT", "development")
    debug_mode = environment == "development"
    
    # Create FastAPI application
    app = FastAPI(
        title="Warehouse Management System API",
        description="""
        **AI-powered warehouse management system** with comprehensive inventory optimization.
        
        ## Features
        
        * **Product Management** - Create, update, and track products
        * **Inventory Tracking** - Real-time batch and stock management
        * **Quality Control** - Inspection and quality assurance workflows
        * **Anomaly Detection** - AI-powered anomaly identification
        * **Purchase Orders** - Complete procurement management
        * **Storage Optimization** - Intelligent storage location management
        
        ## API Versioning
        
        This API uses URL versioning. Current version: **v1**
        
        Base URL: `/api/v1`
        """,
        version="1.0.0",
        contact={
            "name": "Warehouse Management System",
            "email": "support@warehouse-mgmt.com",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
        docs_url="/docs" if debug_mode else None,
        redoc_url="/redoc" if debug_mode else None,
        openapi_url="/openapi.json",
        lifespan=lifespan,
        debug=debug_mode
    )
    
    # Setup middleware
    setup_middleware(app)
    
    # Setup exception handlers
    setup_exception_handlers(app)
    
    # Setup routers
    setup_routers(app)
    
    logger.info(f"FastAPI application created in {environment} mode")
    return app

# Create application instance
app = create_application()

# Development server configuration
if __name__ == "__main__":
    import uvicorn
    
    # Environment-based configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    environment = os.getenv("ENVIRONMENT", "development")
    reload = environment == "development"
    
    logger.info(f"Starting development server on {host}:{port}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Reload enabled: {reload}")
    
    uvicorn.run(
        "src.wms.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True,
        reload_dirs=["src"] if reload else None
    )
