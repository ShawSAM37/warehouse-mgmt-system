"""
Warehouse Management System Authorization Module
===============================================

Production-grade Role-Based Access Control (RBAC) and resource authorization
for WMS operations. Provides fine-grained permission control, audit logging,
and resource ownership validation.

Author: WMS Development Team
Version: 1.0.0
"""

import logging
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import List, Optional, Dict, Any, Callable, Union
from fastapi import Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from sqlalchemy import inspect

# Project imports
from src.wms.shared.enums import UserRole
from src.wms.auth.authentication import get_current_active_user, User
from src.wms.utils.db import get_db

# Configure logging
logger = logging.getLogger(__name__)

# ================================
# Permission Definitions
# ================================

class Permission(str, Enum):
    """Enumeration of all system permissions."""
    
    # Inventory permissions
    INVENTORY_READ = "inventory:read"
    INVENTORY_WRITE = "inventory:write"
    INVENTORY_DELETE = "inventory:delete"
    
    # Product permissions
    PRODUCT_READ = "product:read"
    PRODUCT_WRITE = "product:write"
    PRODUCT_DELETE = "product:delete"
    
    # Transaction permissions
    TRANSACTION_READ = "transaction:read"
    TRANSACTION_CREATE = "transaction:create"
    TRANSACTION_UPDATE = "transaction:update"
    
    # Purchase Order permissions
    PO_READ = "po:read"
    PO_CREATE = "po:create"
    PO_UPDATE = "po:update"
    PO_APPROVE = "po:approve"
    
    # Quality Check permissions
    QC_READ = "qc:read"
    QC_CREATE = "qc:create"
    QC_UPDATE = "qc:update"
    
    # Supplier permissions
    SUPPLIER_READ = "supplier:read"
    SUPPLIER_WRITE = "supplier:write"
    SUPPLIER_DELETE = "supplier:delete"
    
    # Storage permissions
    STORAGE_READ = "storage:read"
    STORAGE_WRITE = "storage:write"
    
    # Reporting permissions
    REPORT_READ = "report:read"
    REPORT_EXPORT = "report:export"
    REPORT_FINANCIAL = "report:financial"
    
    # User management permissions
    USER_READ = "user:read"
    USER_CREATE = "user:create"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    
    # System administration
    SYSTEM_CONFIG = "system:config"
    SYSTEM_AUDIT = "system:audit"
    
    # Dashboard permissions
    DASHBOARD_VIEW = "dashboard:view"
    DASHBOARD_ADMIN = "dashboard:admin"
    
    # Gate entry permissions
    GATE_ENTRY = "gate:entry"
    GATE_CONFIG = "gate:config"
    
    # Replenishment permissions
    REPLENISH_READ = "replenish:read"
    REPLENISH_EXECUTE = "replenish:execute"
    REPLENISH_CONFIG = "replenish:config"

# ================================
# Role-Permission Mapping
# ================================

ROLE_PERMISSIONS: Dict[UserRole, List[Permission]] = {
    UserRole.ADMIN: [
        # Full system access
        Permission.INVENTORY_READ, Permission.INVENTORY_WRITE, Permission.INVENTORY_DELETE,
        Permission.PRODUCT_READ, Permission.PRODUCT_WRITE, Permission.PRODUCT_DELETE,
        Permission.TRANSACTION_READ, Permission.TRANSACTION_CREATE, Permission.TRANSACTION_UPDATE,
        Permission.PO_READ, Permission.PO_CREATE, Permission.PO_UPDATE, Permission.PO_APPROVE,
        Permission.QC_READ, Permission.QC_CREATE, Permission.QC_UPDATE,
        Permission.SUPPLIER_READ, Permission.SUPPLIER_WRITE, Permission.SUPPLIER_DELETE,
        Permission.STORAGE_READ, Permission.STORAGE_WRITE,
        Permission.REPORT_READ, Permission.REPORT_EXPORT, Permission.REPORT_FINANCIAL,
        Permission.USER_READ, Permission.USER_CREATE, Permission.USER_UPDATE, Permission.USER_DELETE,
        Permission.SYSTEM_CONFIG, Permission.SYSTEM_AUDIT,
        Permission.DASHBOARD_VIEW, Permission.DASHBOARD_ADMIN,
        Permission.GATE_ENTRY, Permission.GATE_CONFIG,
        Permission.REPLENISH_READ, Permission.REPLENISH_EXECUTE, Permission.REPLENISH_CONFIG,
    ],
    
    UserRole.MANAGER: [
        # Management level access
        Permission.INVENTORY_READ, Permission.INVENTORY_WRITE,
        Permission.PRODUCT_READ, Permission.PRODUCT_WRITE,
        Permission.TRANSACTION_READ, Permission.TRANSACTION_CREATE, Permission.TRANSACTION_UPDATE,
        Permission.PO_READ, Permission.PO_CREATE, Permission.PO_UPDATE, Permission.PO_APPROVE,
        Permission.QC_READ, Permission.QC_CREATE, Permission.QC_UPDATE,
        Permission.SUPPLIER_READ, Permission.SUPPLIER_WRITE,
        Permission.STORAGE_READ, Permission.STORAGE_WRITE,
        Permission.REPORT_READ, Permission.REPORT_EXPORT, Permission.REPORT_FINANCIAL,
        Permission.USER_READ,
        Permission.DASHBOARD_VIEW,
        Permission.GATE_ENTRY,
        Permission.REPLENISH_READ, Permission.REPLENISH_EXECUTE,
    ],
    
    UserRole.WORKER: [
        # Operational level access
        Permission.INVENTORY_READ, Permission.INVENTORY_WRITE,
        Permission.PRODUCT_READ,
        Permission.TRANSACTION_READ, Permission.TRANSACTION_CREATE,
        Permission.PO_READ,
        Permission.QC_READ, Permission.QC_CREATE,
        Permission.SUPPLIER_READ,
        Permission.STORAGE_READ,
        Permission.REPORT_READ,
        Permission.DASHBOARD_VIEW,
        Permission.GATE_ENTRY,
        Permission.REPLENISH_READ,
    ],
    
    UserRole.VIEWER: [
        # Read-only access
        Permission.INVENTORY_READ,
        Permission.PRODUCT_READ,
        Permission.TRANSACTION_READ,
        Permission.PO_READ,
        Permission.QC_READ,
        Permission.SUPPLIER_READ,
        Permission.STORAGE_READ,
        Permission.REPORT_READ,
        Permission.DASHBOARD_VIEW,
        Permission.REPLENISH_READ,
    ],
}

# ================================
# Authorization Service
# ================================

class AuthorizationService:
    """Service class for handling authorization logic."""
    
    def __init__(self):
        self.role_permissions = ROLE_PERMISSIONS
        self.audit_log = []
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """
        Check if user has the specified permission.
        
        Args:
            user: The authenticated user
            permission: The permission to check
            
        Returns:
            bool: True if user has permission, False otherwise
        """
        if not user or not user.is_active:
            return False
        
        user_roles = [UserRole(role) for role in user.roles if role in [r.value for r in UserRole]]
        
        for role in user_roles:
            if role in self.role_permissions:
                if permission in self.role_permissions[role]:
                    return True
        
        return False
    
    def has_any_permission(self, user: User, permissions: List[Permission]) -> bool:
        """
        Check if user has any of the specified permissions.
        
        Args:
            user: The authenticated user
            permissions: List of permissions to check
            
        Returns:
            bool: True if user has at least one permission, False otherwise
        """
        return any(self.has_permission(user, perm) for perm in permissions)
    
    def has_all_permissions(self, user: User, permissions: List[Permission]) -> bool:
        """
        Check if user has all of the specified permissions.
        
        Args:
            user: The authenticated user
            permissions: List of permissions to check
            
        Returns:
            bool: True if user has all permissions, False otherwise
        """
        return all(self.has_permission(user, perm) for perm in permissions)
    
    def has_role(self, user: User, role: UserRole) -> bool:
        """
        Check if user has the specified role.
        
        Args:
            user: The authenticated user
            role: The role to check
            
        Returns:
            bool: True if user has role, False otherwise
        """
        if not user or not user.is_active:
            return False
        
        return role.value in user.roles
    
    def has_any_role(self, user: User, roles: List[UserRole]) -> bool:
        """
        Check if user has any of the specified roles.
        
        Args:
            user: The authenticated user
            roles: List of roles to check
            
        Returns:
            bool: True if user has at least one role, False otherwise
        """
        return any(self.has_role(user, role) for role in roles)
    
    def log_access_attempt(
        self,
        user: User,
        permission: Optional[Permission] = None,
        roles: Optional[List[UserRole]] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        granted: bool = False,
        reason: Optional[str] = None
    ):
        """
        Log access attempt for audit purposes.
        
        Args:
            user: The user attempting access
            permission: Permission being checked
            roles: Roles being checked
            resource_type: Type of resource being accessed
            resource_id: ID of resource being accessed
            endpoint: API endpoint being accessed
            granted: Whether access was granted
            reason: Reason for access denial (if applicable)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user.id if user else None,
            "username": user.username if user else "anonymous",
            "permission": permission.value if permission else None,
            "roles": [role.value for role in roles] if roles else None,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "endpoint": endpoint,
            "granted": granted,
            "reason": reason,
        }
        
        self.audit_log.append(log_entry)
        
        # Log to application logger
        log_level = logging.INFO if granted else logging.WARNING
        log_message = (
            f"Access {'granted' if granted else 'denied'}: "
            f"user={user.username if user else 'anonymous'}, "
            f"permission={permission.value if permission else 'N/A'}, "
            f"endpoint={endpoint or 'N/A'}"
        )
        
        if not granted and reason:
            log_message += f", reason={reason}"
        
        logger.log(log_level, log_message)
    
    def get_user_permissions(self, user: User) -> List[Permission]:
        """
        Get all permissions for a user based on their roles.
        
        Args:
            user: The user to get permissions for
            
        Returns:
            List[Permission]: List of all permissions the user has
        """
        if not user or not user.is_active:
            return []
        
        permissions = set()
        user_roles = [UserRole(role) for role in user.roles if role in [r.value for r in UserRole]]
        
        for role in user_roles:
            if role in self.role_permissions:
                permissions.update(self.role_permissions[role])
        
        return list(permissions)

# ================================
# FastAPI Dependencies
# ================================

def require_permission(permission: Permission):
    """
    FastAPI dependency factory for permission-based access control.
    
    Args:
        permission: The permission required to access the endpoint
        
    Returns:
        Dependency function that checks user permission
        
    Example:
        @router.get("/inventory", dependencies=[Depends(require_permission(Permission.INVENTORY_READ))])
        async def get_inventory():
            pass
    """
    def permission_checker(
        request: Request,
        user: User = Depends(get_current_active_user)
    ) -> User:
        if not authz_service.has_permission(user, permission):
            authz_service.log_access_attempt(
                user=user,
                permission=permission,
                endpoint=str(request.url),
                granted=False,
                reason="Insufficient permissions"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission.value}"
            )
        
        authz_service.log_access_attempt(
            user=user,
            permission=permission,
            endpoint=str(request.url),
            granted=True
        )
        
        return user
    
    return permission_checker

def require_any_permission(permissions: List[Permission]):
    """
    FastAPI dependency factory for checking any of multiple permissions.
    
    Args:
        permissions: List of permissions, user needs at least one
        
    Returns:
        Dependency function that checks user has any permission
    """
    def permission_checker(
        request: Request,
        user: User = Depends(get_current_active_user)
    ) -> User:
        if not authz_service.has_any_permission(user, permissions):
            authz_service.log_access_attempt(
                user=user,
                endpoint=str(request.url),
                granted=False,
                reason=f"None of required permissions: {[p.value for p in permissions]}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of these permissions required: {[p.value for p in permissions]}"
            )
        
        authz_service.log_access_attempt(
            user=user,
            endpoint=str(request.url),
            granted=True
        )
        
        return user
    
    return permission_checker

def require_roles(required_roles: List[UserRole]):
    """
    FastAPI dependency factory for role-based access control.
    Compatible with existing authentication module.
    
    Args:
        required_roles: List of roles required to access the endpoint
        
    Returns:
        Dependency function that checks user roles
        
    Example:
        @router.post("/users", dependencies=[Depends(require_roles([UserRole.ADMIN]))])
        async def create_user():
            pass
    """
    def role_checker(
        request: Request,
        user: User = Depends(get_current_active_user)
    ) -> User:
        if not authz_service.has_any_role(user, required_roles):
            authz_service.log_access_attempt(
                user=user,
                roles=required_roles,
                endpoint=str(request.url),
                granted=False,
                reason="Insufficient role privileges"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of these roles required: {[role.value for role in required_roles]}"
            )
        
        authz_service.log_access_attempt(
            user=user,
            roles=required_roles,
            endpoint=str(request.url),
            granted=True
        )
        
        return user
    
    return role_checker

def resource_owner_check(
    resource_type: str,
    resource_id_param: str = "id",
    owner_field: str = "created_by"
):
    """
    FastAPI dependency factory for resource ownership validation.
    
    Args:
        resource_type: Type of resource (maps to model name)
        resource_id_param: Name of path parameter containing resource ID
        owner_field: Field name in the model that contains owner user ID
        
    Returns:
        Dependency function that checks resource ownership
        
    Example:
        @router.put("/inventory/{batch_id}", 
                   dependencies=[Depends(resource_owner_check("InventoryBatch", "batch_id"))])
        async def update_batch():
            pass
    """
    def ownership_checker(
        request: Request,
        db: Session = Depends(get_db),
        user: User = Depends(get_current_active_user)
    ) -> User:
        # Skip ownership check for admins
        if authz_service.has_role(user, UserRole.ADMIN):
            return user
        
        resource_id = request.path_params.get(resource_id_param)
        if not resource_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing resource ID parameter: {resource_id_param}"
            )
        
        # Import models dynamically to avoid circular imports
        try:
            from src.wms.inventory import models as inventory_models
            
            # Map resource types to models
            model_map = {
                "Product": inventory_models.Product,
                "InventoryBatch": inventory_models.InventoryBatch,
                "InventoryTransaction": inventory_models.InventoryTransaction,
                "PurchaseOrder": inventory_models.PurchaseOrder,
                "PurchaseOrderLineItem": inventory_models.PurchaseOrderLineItem,
                "QualityCheck": inventory_models.QualityCheck,
                "Supplier": inventory_models.Supplier,
                "StorageLocation": inventory_models.StorageLocation,
            }
            
            model_class = model_map.get(resource_type)
            if not model_class:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unknown resource type: {resource_type}"
                )
            
            # Query the resource
            resource = db.query(model_class).filter(
                getattr(model_class, "id") == resource_id
            ).first()
            
            if not resource:
                authz_service.log_access_attempt(
                    user=user,
                    resource_type=resource_type,
                    resource_id=str(resource_id),
                    endpoint=str(request.url),
                    granted=False,
                    reason="Resource not found"
                )
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Resource not found"
                )
            
            # Check ownership if owner field exists
            if hasattr(resource, owner_field):
                owner_id = getattr(resource, owner_field)
                if owner_id != user.id:
                    authz_service.log_access_attempt(
                        user=user,
                        resource_type=resource_type,
                        resource_id=str(resource_id),
                        endpoint=str(request.url),
                        granted=False,
                        reason="Not resource owner"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="You don't have permission to access this resource"
                    )
            
            authz_service.log_access_attempt(
                user=user,
                resource_type=resource_type,
                resource_id=str(resource_id),
                endpoint=str(request.url),
                granted=True
            )
            
            return user
            
        except ImportError as e:
            logger.error(f"Failed to import models for ownership check: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error during authorization"
            )
    
    return ownership_checker

# ================================
# Streamlit Integration Helpers
# ================================

class StreamlitAuthz:
    """Helper class for Streamlit authorization integration."""
    
    @staticmethod
    def check_permission(permission: Permission) -> bool:
        """
        Check if current Streamlit user has permission.
        
        Args:
            permission: Permission to check
            
        Returns:
            bool: True if user has permission
        """
        try:
            import streamlit as st
            from src.wms.auth.authentication import StreamlitAuth
            
            if not StreamlitAuth.is_authenticated():
                return False
            
            user_info = StreamlitAuth.get_user_info()
            if not user_info:
                return False
            
            # Create a minimal user object for permission checking
            class MinimalUser:
                def __init__(self, user_info):
                    self.id = user_info.get('id')
                    self.username = user_info.get('username')
                    self.roles = user_info.get('roles', [])
                    self.is_active = user_info.get('is_active', True)
            
            user = MinimalUser(user_info)
            return authz_service.has_permission(user, permission)
            
        except ImportError:
            logger.warning("Streamlit not available for permission check")
            return False
    
    @staticmethod
    def check_role(role: UserRole) -> bool:
        """
        Check if current Streamlit user has role.
        
        Args:
            role: Role to check
            
        Returns:
            bool: True if user has role
        """
        try:
            import streamlit as st
            from src.wms.auth.authentication import StreamlitAuth
            
            if not StreamlitAuth.is_authenticated():
                return False
            
            user_info = StreamlitAuth.get_user_info()
            if not user_info:
                return False
            
            return role.value in user_info.get('roles', [])
            
        except ImportError:
            logger.warning("Streamlit not available for role check")
            return False
    
    @staticmethod
    def require_permission_ui(permission: Permission, error_message: str = None):
        """
        Show error and stop execution if user lacks permission.
        
        Args:
            permission: Required permission
            error_message: Custom error message
        """
        try:
            import streamlit as st
            
            if not StreamlitAuthz.check_permission(permission):
                error_msg = error_message or f"Permission required: {permission.value}"
                st.error(error_msg)
                st.stop()
                
        except ImportError:
            logger.warning("Streamlit not available for UI permission check")

# ================================
# Global Service Instance
# ================================

# Create global authorization service instance
authz_service = AuthorizationService()

# ================================
# Utility Functions
# ================================

def get_user_permissions(user: User) -> List[Permission]:
    """
    Get all permissions for a user.
    
    Args:
        user: User to get permissions for
        
    Returns:
        List of permissions
    """
    return authz_service.get_user_permissions(user)

def check_permission(user: User, permission: Permission) -> bool:
    """
    Check if user has permission.
    
    Args:
        user: User to check
        permission: Permission to check
        
    Returns:
        bool: True if user has permission
    """
    return authz_service.has_permission(user, permission)

def check_role(user: User, role: UserRole) -> bool:
    """
    Check if user has role.
    
    Args:
        user: User to check
        role: Role to check
        
    Returns:
        bool: True if user has role
    """
    return authz_service.has_role(user, role)

# ================================
# Export Public API
# ================================

__all__ = [
    # Enums
    "Permission",
    
    # Service
    "AuthorizationService",
    "authz_service",
    
    # Dependencies
    "require_permission",
    "require_any_permission", 
    "require_roles",
    "resource_owner_check",
    
    # Streamlit helpers
    "StreamlitAuthz",
    
    # Utility functions
    "get_user_permissions",
    "check_permission",
    "check_role",
    
    # Constants
    "ROLE_PERMISSIONS",
]

logger.info("Authorization module initialized successfully")

# ================================
# Example Usage (in comments)
# ================================

"""
Example FastAPI endpoint usage:

from src.wms.auth.authorization import require_permission, require_roles, Permission

# Permission-based access
@router.get("/inventory", dependencies=[Depends(require_permission(Permission.INVENTORY_READ))])
async def get_inventory():
    pass

# Role-based access  
@router.post("/users", dependencies=[Depends(require_roles([UserRole.ADMIN]))])
async def create_user():
    pass

# Resource ownership
@router.put("/inventory/{batch_id}", 
           dependencies=[Depends(resource_owner_check("InventoryBatch", "batch_id"))])
async def update_batch():
    pass

# Multiple permissions
@router.get("/reports/financial", 
           dependencies=[Depends(require_any_permission([Permission.REPORT_FINANCIAL, Permission.SYSTEM_AUDIT]))])
async def get_financial_reports():
    pass

Example Streamlit usage:

from src.wms.auth.authorization import StreamlitAuthz, Permission

# Check permission in Streamlit
if StreamlitAuthz.check_permission(Permission.INVENTORY_WRITE):
    st.button("Edit Inventory")
else:
    st.info("You don't have permission to edit inventory")

# Require permission (stops execution if not authorized)
StreamlitAuthz.require_permission_ui(Permission.REPORT_FINANCIAL)
st.write("Financial report content here...")
"""
