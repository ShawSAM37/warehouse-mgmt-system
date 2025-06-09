"""
Integration Tests for WMS Security Middleware
=============================================

Validates all security middleware functionality in production-like scenarios.
"""

import json
import logging
from datetime import datetime, timedelta
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from jose import jwt
import redis

# Project imports
from src.wms.main import app
from src.wms.utils.config import config
from src.wms.utils.db import get_db
from src.wms.auth.authentication import auth_service
from src.wms.shared.enums import UserRole

# Configure test client
client = TestClient(app)

# Test configuration
TEST_USERNAME = "testuser"
TEST_PASSWORD = "testpass123!"
TEST_EMAIL = "test@wms.com"
MAX_REQUEST_SIZE = config.MAX_REQUEST_SIZE + 1  # Exceed config limit

@pytest.fixture(autouse=True)
def cleanup_redis():
    """Clear Redis cache before each test"""
    redis_client = redis.Redis.from_url(config.REDIS_URL)
    redis_client.flushall()
    yield
    redis_client.close()

@pytest.fixture
def admin_token():
    """Generate valid JWT token for admin user"""
    return jwt.encode(
        {"sub": TEST_USERNAME, "roles": [UserRole.ADMIN.value]},
        config.JWT_SECRET_KEY,
        algorithm="HS256"
    )

# --- Security Headers Middleware Tests ---
class TestSecurityHeadersMiddleware:
    """Validate OWASP security headers and CORS configuration"""
    
    def test_security_headers_present(self):
        response = client.get("/health")
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "Content-Security-Policy" in response.headers
        assert "X-Content-Type-Options" in response.headers

    def test_cors_allowed_origin(self):
        response = client.get(
            "/health",
            headers={"Origin": "https://dashboard.wms.com"}
        )
        assert response.headers["Access-Control-Allow-Origin"] == "https://dashboard.wms.com"

    def test_cors_disallowed_origin(self):
        response = client.get(
            "/health", 
            headers={"Origin": "https://malicious-site.com"}
        )
        assert "Access-Control-Allow-Origin" not in response.headers

# --- Rate Limiting Middleware Tests ---
class TestRateLimitingMiddleware:
    """Validate request rate limiting functionality"""
    
    def test_normal_rate_limit(self):
        for _ in range(config.RATE_LIMIT_REQUESTS):
            response = client.get("/api/v1/products")
            assert response.status_code == 200
        
        response = client.get("/api/v1/products")
        assert response.status_code == 429
        assert "Retry-After" in response.headers

    def test_auth_endpoint_rate_limit(self):
        auth_endpoint = "/auth/login"
        for _ in range(config.RATE_LIMIT_AUTH_REQUESTS):
            response = client.post(auth_endpoint, json={
                "username": TEST_USERNAME,
                "password": TEST_PASSWORD
            })
            assert response.status_code in [401, 429]
        
        response = client.post(auth_endpoint, json={
            "username": TEST_USERNAME,
            "password": TEST_PASSWORD
        })
        assert response.status_code == 429

    def test_public_endpoint_exemption(self):
        for _ in range(config.RATE_LIMIT_REQUESTS + 5):
            response = client.get("/health")
            assert response.status_code == 200

# --- Authentication Enforcement Middleware Tests ---
class TestAuthenticationEnforcementMiddleware:
    """Validate authentication requirements for protected routes"""
    
    def test_protected_route_without_token(self):
        response = client.get("/api/v1/users")
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers

    def test_protected_route_with_valid_token(self, admin_token):
        response = client.get(
            "/api/v1/users",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code in [200, 403]

    def test_public_route_access(self):
        response = client.get("/docs")
        assert response.status_code == 200

# --- Audit Logging Middleware Tests ---
class TestAuditLoggingMiddleware:
    """Validate request/response audit logging"""
    
    @patch("src.wms.auth.middleware.logger")
    def test_successful_request_logging(self, mock_logger):
        client.get("/health")
        mock_logger.info.assert_called()
        log_args = mock_logger.info.call_args[1]
        assert "audit" in log_args
        assert log_args["audit"]["path"] == "/health"
        assert log_args["audit"]["status"] == 200

    @patch("src.wms.auth.middleware.logger")
    def test_failed_request_logging(self, mock_logger):
        client.get("/api/v1/invalid-endpoint")
        mock_logger.warning.assert_called()
        log_args = mock_logger.warning.call_args[1]
        assert log_args["audit"]["status"] == 404

# --- Request Validation Middleware Tests ---
class TestRequestValidationMiddleware:
    """Validate request size and content type enforcement"""
    
    def test_request_size_validation(self):
        large_content = b"x" * MAX_REQUEST_SIZE
        response = client.post("/api/v1/upload", content=large_content)
        assert response.status_code == 413
        assert "Request entity too large" in response.text

    def test_content_type_validation(self):
        response = client.post(
            "/api/v1/products",
            content='{"name": "Test"}',
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 415
        assert "Unsupported media type" in response.text

# --- Error Handling Middleware Tests ---
class TestErrorHandlingMiddleware:
    """Validate proper error handling and responses"""
    
    def test_unhandled_exception_handling(self):
        response = client.get("/api/v1/trigger-error")
        assert response.status_code == 500
        assert "Internal server error" in response.text
        assert "debug" not in response.text  # No sensitive info

    def test_known_http_exception(self):
        response = client.get("/api/v1/protected-route")
        assert response.status_code == 401
        assert "detail" in response.json()

# --- Middleware Integration Tests ---
class TestMiddlewareIntegration:
    """Validate middleware interaction and ordering"""
    
    def test_headers_on_error_response(self):
        response = client.get("/api/v1/invalid-endpoint")
        assert response.status_code == 404
        assert "X-Process-Time" in response.headers
        assert "X-Frame-Options" in response.headers

    def test_rate_limit_headers(self):
        response = client.get("/api/v1/products")
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers

    @patch("src.wms.auth.middleware.logger")
    def test_full_request_cycle(self, mock_logger):
        # Authenticated request
        token = client.post("/auth/login", json={
            "username": TEST_USERNAME,
            "password": TEST_PASSWORD
        }).json()["access_token"]
        
        response = client.get(
            "/api/v1/users",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        # Verify all middleware processed
        assert response.status_code in [200, 403]
        assert "X-Process-Time" in response.headers
        mock_logger.info.assert_called()
