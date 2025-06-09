"""
Warehouse Management System Security Middleware
===============================================

Production-grade middleware for security headers, rate limiting, audit logging,
and authentication enforcement. Fully compatible with WMS architecture.

Author: WMS Development Team
Version: 1.0.0
"""

import time
import logging
from fastapi import Request, HTTPException
from fastapi.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Scope, Receive, Send

# Project imports
from src.wms.auth.authentication import get_current_user
from src.wms.utils.config import config
from src.wms.utils.cache import CacheManager
from src.wms.utils.db import get_db
from src.wms.shared.enums import UserRole

logger = logging.getLogger(__name__)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds critical security headers to all responses
    Compatible with FastAPI/Starlette middleware system
    """
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers configuration
        headers = {
            "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=()",
            "Content-Security-Policy": "default-src 'self'"
        }
        
        # Merge headers without overwriting existing ones
        for header, value in headers.items():
            response.headers.setdefault(header, value)
            
        return response

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Redis-based rate limiting with API key and IP tracking
    Integrates with existing CacheManager
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.cache = CacheManager.get_redis_client()
        self.rate_limit = config.RATE_LIMIT
        self.window_sec = config.RATE_LIMIT_WINDOW

    async def dispatch(self, request: Request, call_next):
        # Get client identifier (API key or IP)
        client_id = request.headers.get("X-API-Key") or request.client.host
        
        # Create rate limit key
        key = f"rate_limit:{client_id}:{int(time.time()) // self.window_sec}"
        
        # Increment and check rate
        current = await self.cache.incr(key)
        await self.cache.expire(key, self.window_sec)
        
        if current > self.rate_limit:
            logger.warning(f"Rate limit exceeded for {client_id}")
            raise HTTPException(
                status_code=429,
                detail="Too many requests"
            )
            
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(self.rate_limit - current)
        return response

class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive audit logging with user context
    Integrates with existing authentication system
    """
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        
        # Try to get authenticated user
        user = None
        try:
            db = next(get_db())
            user = await get_current_user(request, db)
        except HTTPException:
            pass
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        
        # Log audit event
        audit_data = {
            "user": user.username if user else "anonymous",
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "process_time": process_time,
            "client": request.client.host if request.client else None
        }
        
        logger.info("Audit log entry", extra={"audit": audit_data})
        return response

class AuthenticationEnforcementMiddleware(BaseHTTPMiddleware):
    """
    Global authentication check with public route exceptions
    Integrates with existing authorization config
    """
    
    PUBLIC_ROUTES = {
        "/auth/login", 
        "/auth/refresh",
        "/health",
        "/docs",
        "/openapi.json"
    }
    
    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.PUBLIC_ROUTES:
            return await call_next(request)
            
        try:
            db = next(get_db())
            user = await get_current_user(request, db)
            request.state.user = user
        except HTTPException as e:
            logger.warning(f"Authentication failed: {e.detail}")
            raise
            
        return await call_next(request)

# Factory function for easy middleware setup
def get_security_middleware() -> list[Middleware]:
    return [
        Middleware(SecurityHeadersMiddleware),
        Middleware(RateLimitingMiddleware),
        Middleware(AuditLoggingMiddleware),
        Middleware(AuthenticationEnforcementMiddleware)
    ]
