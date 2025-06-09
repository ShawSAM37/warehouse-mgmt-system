"""
Warehouse Management System Authentication & Authorization Package
================================================================

Production-ready authentication and authorization system for WMS operations.
Provides JWT-based authentication, role-based access control (RBAC), and
fine-grained permission management for FastAPI backend and Streamlit dashboard.

Features:
- JWT token-based authentication with refresh tokens
- Role-based access control (ADMIN, MANAGER, WORKER, VIEWER)
- Fine-grained permission system for all WMS operations
- Resource ownership validation
- Audit logging for all access attempts
- FastAPI dependency injection support
- Streamlit dashboard integration
- Production-grade security practices

Author: WMS Development Team
Version: 1.0.0
License: MIT
"""

import logging
from typing import List, Optional

# Package metadata
__version__ = "1.0.0"
__author__ = "WMS Development Team"
__license__ = "MIT"
__description__ = "Authentication and authorization system for Warehouse Management System"

# Configure logging
logger = logging.getLogger(__name__)

# ================================
# Authentication Module Imports
# ================================

try:
    from .authentication import (
        # SQLAlchemy Models
        User,
        
        # Pydantic Schemas
        UserBase,
        UserCreate,
        UserUpdate,
        UserResponse,
        Token,
        TokenData,
        LoginRequest,
        RefreshTokenRequest,
        
        # Service Classes
        AuthService,
        auth_service,
        
        # FastAPI Dependencies
        get_current_user,
        get_current_active_user,
        oauth2_scheme,
        
        # Streamlit Integration
        StreamlitAuth,
        
        # FastAPI Router
        auth_router,
        
        # Utility Functions
        check_user_permission,
        get_user_by_username,
        get_user_by_email,
    )
    
    logger.info("Authentication module imported successfully")
    
except ImportError as e:
    logger.error(f"Failed to import authentication module: {e}")
    raise ImportError(f"Authentication module import failed: {e}")

# ================================
# Authorization Module Imports
# ================================

try:
    from .authorization import (
        # Permission System
        Permission,
        ROLE_PERMISSIONS,
        
        # Service Classes
        AuthorizationService,
        authz_service,
        
        # FastAPI Dependencies
        require_permission,
        require_any_permission,
        require_roles,
        resource_owner_check,
        
        # Streamlit Integration
        StreamlitAuthz,
        
        # Utility Functions
        get_user_permissions,
        check_permission,
        check_role,
    )
    
    logger.info("Authorization module imported successfully")
    
except ImportError as e:
    logger.error(f"Failed to import authorization module: {e}")
    raise ImportError(f"Authorization module import failed: {e}")

# ================================
# Convenience Functions
# ================================

def get_auth_status() -> dict:
    """
    Get authentication system status and configuration.
    
    Returns:
        dict: Status information including available permissions and roles
    """
    try:
        from src.wms.shared.enums import UserRole
        
        return {
            "version": __version__,
            "authentication": {
                "jwt_enabled": True,
                "refresh_tokens": True,
                "account_lockout": True,
            },
            "authorization": {
                "rbac_enabled": True,
                "resource_ownership": True,
                "audit_logging": True,
            },
            "available_roles": [role.value for role in UserRole],
            "total_permissions": len(Permission),
            "streamlit_integration": True,
            "fastapi_integration": True,
        }
    except Exception as e:
        logger.error(f"Error getting auth status: {e}")
        return {"error": str(e)}

def initialize_auth_system() -> bool:
    """
    Initialize the authentication system.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        # Verify auth service is available
        if not auth_service:
            logger.error("Authentication service not available")
            return False
        
        # Verify authz service is available
        if not authz_service:
            logger.error("Authorization service not available")
            return False
        
        # Verify JWT secret is configured
        import os
        if not os.getenv("JWT_SECRET_KEY"):
            logger.warning("JWT_SECRET_KEY not configured - authentication may fail")
        
        logger.info("Authentication system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize auth system: {e}")
        return False

# ================================
# Public API Exports
# ================================

__all__ = [
    # Package metadata
    "__version__",
    "__author__",
    "__license__",
    "__description__",
    
    # Authentication - Models
    "User",
    
    # Authentication - Schemas
    "UserBase",
    "UserCreate", 
    "UserUpdate",
    "UserResponse",
    "Token",
    "TokenData",
    "LoginRequest",
    "RefreshTokenRequest",
    
    # Authentication - Services
    "AuthService",
    "auth_service",
    
    # Authentication - Dependencies
    "get_current_user",
    "get_current_active_user",
    "oauth2_scheme",
    
    # Authentication - Streamlit
    "StreamlitAuth",
    
    # Authentication - Router
    "auth_router",
    
    # Authentication - Utilities
    "check_user_permission",
    "get_user_by_username",
    "get_user_by_email",
    
    # Authorization - Permissions
    "Permission",
    "ROLE_PERMISSIONS",
    
    # Authorization - Services
    "AuthorizationService",
    "authz_service",
    
    # Authorization - Dependencies
    "require_permission",
    "require_any_permission",
    "require_roles",
    "resource_owner_check",
    
    # Authorization - Streamlit
    "StreamlitAuthz",
    
    # Authorization - Utilities
    "get_user_permissions",
    "check_permission",
    "check_role",
    
    # Package Functions
    "get_auth_status",
    "initialize_auth_system",
]

# ================================
# Module Initialization
# ================================

# Initialize the auth system on import
_initialization_success = initialize_auth_system()

if _initialization_success:
    logger.info(f"WMS Auth package v{__version__} loaded successfully")
else:
    logger.warning(f"WMS Auth package v{__version__} loaded with warnings")

# Log available components for debugging
logger.debug(f"Exported {len(__all__)} auth components")
logger.debug(f"Available permissions: {len(Permission) if 'Permission' in locals() else 0}")

# ================================
# Usage Examples (in docstring)
# ================================

"""
USAGE EXAMPLES
==============

## FastAPI Integration

