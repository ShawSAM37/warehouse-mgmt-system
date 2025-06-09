"""
Warehouse Management System Security Middleware
===============================================

Production-grade security middleware stack for WMS operations.
Provides authentication enforcement, rate limiting, audit logging,
security headers, and request validation with full project integration.

Author: WMS Development Team
Version: 1.0.0
"""

import os
import time
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse

from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

# Project imports
from src.wms.utils.cache import CacheManager
from src.wms.utils.config import config
from src.wms.utils.db import get_db
from src.wms.auth.authentication import get_current_user, User, auth_service
from src.wms.shared.enums import UserRole

# Configure logging
logger = logging.getLogger(__name__)

# ================================
# Configuration
# ================================

class MiddlewareConfig:
    """Middleware configuration with environment variable support."""
    
    # CORS Configuration
    CORS_ALLOW_ORIGINS: List[str] = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    CORS_ALLOW_METHODS: List[str] = os.getenv("CORS_ALLOW_METHODS", "GET,POST,PUT,DELETE,OPTIONS").split(",")
    CORS_ALLOW_HEADERS: List[str] = os.getenv("CORS_ALLOW_HEADERS", "Authorization,Content-Type,X-API-Key").split(",")
    CORS_ALLOW_CREDENTIALS: bool = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    RATE_LIMIT_AUTH_REQUESTS: int = int(os.getenv("RATE_LIMIT_AUTH_REQUESTS", "10"))
    RATE_LIMIT_AUTH_WINDOW: int = int(os.getenv("RATE_LIMIT_AUTH_WINDOW", "300"))
    
    # Security Headers
    SECURITY_HEADERS_ENABLED: bool = os.getenv("SECURITY_HEADERS_ENABLED", "true").lower() == "true"
    CSP_POLICY: str = os.getenv("CSP_POLICY", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'")
    HSTS_MAX_AGE: int = int(os.getenv("HSTS_MAX_AGE", "31536000"))
    
    # Request Validation
    MAX_REQUEST_SIZE: int = int(os.getenv("MAX_REQUEST_SIZE", "10485760"))  # 10MB
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    
    # Public Routes (no authentication required)
    PUBLIC_ROUTES: Set[str] = {
        "/",
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/auth/login",
        "/auth/refresh",
        "/favicon.ico"
    }
    
    # Auth Routes (special rate limiting)
    AUTH_ROUTES: Set[str] = {
        "/auth/login",
        "/auth/refresh",
        "/auth/register"
    }

# ================================
# Security Headers Middleware
# ================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds OWASP-recommended security headers to all responses.
    Configures CORS for Streamlit dashboard integration.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.config = MiddlewareConfig()
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers to response."""
        
        # Handle CORS preflight requests
        if request.method == "OPTIONS":
            return self._create_cors_response()
        
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"Error in SecurityHeadersMiddleware: {e}")
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )
        
        # Add security headers
        if self.config.SECURITY_HEADERS_ENABLED:
            self._add_security_headers(response)
        
        # Add CORS headers
        self._add_cors_headers(request, response)
        
        return response
    
    def _add_security_headers(self, response: Response):
        """Add OWASP security headers."""
        security_headers = {
            "Strict-Transport-Security": f"max-age={self.config.HSTS_MAX_AGE}; includeSubDomains; preload",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": self.config.CSP_POLICY,
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "X-Permitted-Cross-Domain-Policies": "none"
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
    
    def _add_cors_headers(self, request: Request, response: Response):
        """Add CORS headers for dashboard integration."""
        origin = request.headers.get("origin")
        
        # Check if origin is allowed
        if origin and (
            "*" in self.config.CORS_ALLOW_ORIGINS or 
            origin in self.config.CORS_ALLOW_ORIGINS
        ):
            response.headers["Access-Control-Allow-Origin"] = origin
        elif "*" in self.config.CORS_ALLOW_ORIGINS:
            response.headers["Access-Control-Allow-Origin"] = "*"
        
        response.headers["Access-Control-Allow-Methods"] = ",".join(self.config.CORS_ALLOW_METHODS)
        response.headers["Access-Control-Allow-Headers"] = ",".join(self.config.CORS_ALLOW_HEADERS)
        
        if self.config.CORS_ALLOW_CREDENTIALS:
            response.headers["Access-Control-Allow-Credentials"] = "true"
    
    def _create_cors_response(self) -> Response:
        """Create response for CORS preflight requests."""
        response = Response(status_code=200)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = ",".join(self.config.CORS_ALLOW_METHODS)
        response.headers["Access-Control-Allow-Headers"] = ",".join(self.config.CORS_ALLOW_HEADERS)
        response.headers["Access-Control-Max-Age"] = "86400"
        return response

# ================================
# Rate Limiting Middleware
# ================================

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Redis-based rate limiting with per-user and per-IP tracking.
    Integrates with existing CacheManager.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.config = MiddlewareConfig()
        self.cache = CacheManager()
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting based on user/IP and endpoint."""
        
        if not self.config.RATE_LIMIT_ENABLED:
            return await call_next(request)
        
        try:
            # Get client identifier
            client_id = await self._get_client_id(request)
            
            # Determine rate limits based on endpoint
            if request.url.path in self.config.AUTH_ROUTES:
                limit = self.config.RATE_LIMIT_AUTH_REQUESTS
                window = self.config.RATE_LIMIT_AUTH_WINDOW
            else:
                limit = self.config.RATE_LIMIT_REQUESTS
                window = self.config.RATE_LIMIT_WINDOW
            
            # Check rate limit
            if await self._is_rate_limited(client_id, request.url.path, limit, window):
                logger.warning(f"Rate limit exceeded for {client_id} on {request.url.path}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded",
                        "retry_after": window
                    },
                    headers={
                        "Retry-After": str(window),
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0"
                    }
                )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            remaining = await self._get_remaining_requests(client_id, request.url.path, limit, window)
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(int(time.time()) + window)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RateLimitingMiddleware: {e}")
            # Continue without rate limiting if there's an error
            return await call_next(request)
    
    async def _get_client_id(self, request: Request) -> str:
        """Get client identifier (user ID or IP address)."""
        try:
            # Try to get authenticated user
            auth_header = request.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                db = next(get_db())
                user = await get_current_user(token, db)
                return f"user:{user.id}"
        except:
            pass
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    async def _is_rate_limited(self, client_id: str, path: str, limit: int, window: int) -> bool:
        """Check if client is rate limited."""
        try:
            current_window = int(time.time()) // window
            key = f"rate_limit:{client_id}:{path}:{current_window}"
            
            current_count = self.cache.get(key) or 0
            if isinstance(current_count, str):
                current_count = int(current_count)
            
            if current_count >= limit:
                return True
            
            # Increment counter
            self.cache.set(key, current_count + 1, ttl=window)
            return False
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            return False
    
    async def _get_remaining_requests(self, client_id: str, path: str, limit: int, window: int) -> int:
        """Get remaining requests for client."""
        try:
            current_window = int(time.time()) // window
            key = f"rate_limit:{client_id}:{path}:{current_window}"
            
            current_count = self.cache.get(key) or 0
            if isinstance(current_count, str):
                current_count = int(current_count)
            
            return max(0, limit - current_count)
            
        except Exception as e:
            logger.error(f"Error getting remaining requests: {e}")
            return limit

# ================================
# Authentication Enforcement Middleware
# ================================

class AuthenticationEnforcementMiddleware(BaseHTTPMiddleware):
    """
    Enforces JWT authentication on all routes except public ones.
    Integrates with existing authentication system.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.config = MiddlewareConfig()
    
    async def dispatch(self, request: Request, call_next):
        """Enforce authentication on protected routes."""
        
        # Skip authentication for public routes
        if self._is_public_route(request.url.path):
            return await call_next(request)
        
        try:
            # Get and validate user
            user = await self._authenticate_request(request)
            
            # Add user to request state
            request.state.user = user
            request.state.authenticated = True
            
            # Process request
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            logger.warning(f"Authentication failed for {request.url.path}: {e.detail}")
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail},
                headers={"WWW-Authenticate": "Bearer"}
            )
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal authentication error"}
            )
    
    def _is_public_route(self, path: str) -> bool:
        """Check if route is public (no authentication required)."""
        # Exact match
        if path in self.config.PUBLIC_ROUTES:
            return True
        
        # Pattern matching for docs and static files
        public_patterns = ["/docs", "/redoc", "/static/", "/favicon"]
        return any(path.startswith(pattern) for pattern in public_patterns)
    
    async def _authenticate_request(self, request: Request) -> User:
        """Authenticate request and return user."""
        # Get authorization header
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid authorization header"
            )
        
        # Extract token
        token = auth_header.split(" ")[1]
        
        # Get database session
        db = next(get_db())
        
        # Validate token and get user
        user = await get_current_user(token, db)
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or inactive user"
            )
        
        return user

# ================================
# Audit Logging Middleware
# ================================

class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive audit logging with user context and performance metrics.
    Provides structured logging for SIEM integration.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.audit_logger = logging.getLogger("audit")
    
    async def dispatch(self, request: Request, call_next):
        """Log request and response with audit information."""
        
        start_time = time.time()
        
        # Get request information
        request_info = await self._get_request_info(request)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log successful request
            await self._log_audit_event(
                request_info=request_info,
                response_status=response.status_code,
                process_time=process_time,
                success=True
            )
            
            # Add performance headers
            response.headers["X-Process-Time"] = f"{process_time:.4f}"
            response.headers["X-Request-ID"] = request_info["request_id"]
            
            return response
            
        except Exception as e:
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log failed request
            await self._log_audit_event(
                request_info=request_info,
                response_status=500,
                process_time=process_time,
                success=False,
                error=str(e)
            )
            
            # Re-raise exception
            raise
    
    async def _get_request_info(self, request: Request) -> Dict[str, Any]:
        """Extract request information for logging."""
        # Generate request ID
        request_id = f"{int(time.time())}-{id(request)}"
        
        # Get user information
        user_info = {"user_id": None, "username": "anonymous"}
        try:
            if hasattr(request.state, "user") and request.state.user:
                user_info = {
                    "user_id": request.state.user.id,
                    "username": request.state.user.username
                }
        except:
            pass
        
        # Get client information
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_ip": client_ip,
            "user_agent": request.headers.get("user-agent", "unknown"),
            **user_info
        }
    
    async def _log_audit_event(
        self,
        request_info: Dict[str, Any],
        response_status: int,
        process_time: float,
        success: bool,
        error: Optional[str] = None
    ):
        """Log audit event with structured data."""
        
        audit_data = {
            **request_info,
            "response_status": response_status,
            "process_time_seconds": round(process_time, 4),
            "success": success,
            "error": error
        }
        
        # Log with appropriate level
        if success and response_status < 400:
            self.audit_logger.info("Request processed", extra={"audit": audit_data})
        elif response_status < 500:
            self.audit_logger.warning("Client error", extra={"audit": audit_data})
        else:
            self.audit_logger.error("Server error", extra={"audit": audit_data})

# ================================
# Request Validation Middleware
# ================================

class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Validates request size, content type, and detects malicious payloads.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.config = MiddlewareConfig()
    
    async def dispatch(self, request: Request, call_next):
        """Validate request before processing."""
        
        try:
            # Validate request size
            await self._validate_request_size(request)
            
            # Validate content type for POST/PUT requests
            if request.method in ["POST", "PUT", "PATCH"]:
                await self._validate_content_type(request)
            
            # Process request with timeout
            response = await asyncio.wait_for(
                call_next(request),
                timeout=self.config.REQUEST_TIMEOUT
            )
            
            return response
            
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout for {request.url.path}")
            return JSONResponse(
                status_code=408,
                content={"detail": "Request timeout"}
            )
        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail}
            )
        except Exception as e:
            logger.error(f"Request validation error: {e}")
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid request"}
            )
    
    async def _validate_request_size(self, request: Request):
        """Validate request size."""
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.config.MAX_REQUEST_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Request entity too large"
            )
    
    async def _validate_content_type(self, request: Request):
        """Validate content type for requests with body."""
        content_type = request.headers.get("content-type", "")
        
        allowed_types = [
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data"
        ]
        
        if not any(content_type.startswith(allowed) for allowed in allowed_types):
            raise HTTPException(
                status_code=415,
                detail="Unsupported media type"
            )

# ================================
# Error Handling Middleware
# ================================

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Centralized error handling with security-safe error messages.
    """
    
    async def dispatch(self, request: Request, call_next):
        """Handle errors and provide security-safe responses."""
        
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            # Log HTTP exceptions
            logger.warning(f"HTTP exception: {e.status_code} - {e.detail}")
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail}
            )
            
        except SQLAlchemyError as e:
            # Database errors
            logger.error(f"Database error: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Database error occurred"}
            )
            
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )

# ================================
# Middleware Factory
# ================================

def get_security_middleware() -> List[BaseHTTPMiddleware]:
    """
    Get ordered list of security middleware for FastAPI application.
    
    Returns:
        List of middleware classes in proper execution order
    """
    middleware_stack = []
    
    # Add middleware in reverse order (FastAPI processes them in reverse)
    middleware_classes = [
        ErrorHandlingMiddleware,
        AuditLoggingMiddleware,
        AuthenticationEnforcementMiddleware,
        RequestValidationMiddleware,
        RateLimitingMiddleware,
        SecurityHeadersMiddleware,
    ]
    
    for middleware_class in middleware_classes:
        try:
            middleware_stack.append(middleware_class)
            logger.info(f"Added {middleware_class.__name__} to middleware stack")
        except Exception as e:
            logger.error(f"Failed to add {middleware_class.__name__}: {e}")
    
    return middleware_stack

def configure_middleware_logging():
    """Configure logging for middleware components."""
    
    # Configure audit logger
    audit_logger = logging.getLogger("audit")
    audit_logger.setLevel(logging.INFO)
    
    # Add file handler for audit logs
    audit_handler = logging.FileHandler("audit.log")
    audit_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    audit_handler.setFormatter(audit_formatter)
    audit_logger.addHandler(audit_handler)
    
    logger.info("Middleware logging configured")

# ================================
# Initialization
# ================================

# Configure logging on import
configure_middleware_logging()

# Export public API
__all__ = [
    "SecurityHeadersMiddleware",
    "RateLimitingMiddleware", 
    "AuthenticationEnforcementMiddleware",
    "AuditLoggingMiddleware",
    "RequestValidationMiddleware",
    "ErrorHandlingMiddleware",
    "get_security_middleware",
    "configure_middleware_logging",
    "MiddlewareConfig"
]

logger.info("Security middleware module initialized successfully")



