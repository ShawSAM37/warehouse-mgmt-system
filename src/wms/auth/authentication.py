"""
Warehouse Management System Authentication Module
===============================================

Production-ready authentication and authorization system for WMS.
Provides JWT-based authentication, role-based access control, and
secure user management with full audit logging.

Author: WMS Development Team
Version: 1.0.0
"""

import os
import re
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Union, Dict, Any
from functools import wraps

from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, validator, EmailStr
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, Index, text
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

# Project imports
from src.wms.shared.enums import UserRole
from src.wms.utils.db import Base, get_db

# Configure logging
logger = logging.getLogger(__name__)

# Authentication configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY environment variable is required")

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ================================
# SQLAlchemy User Model
# ================================

class User(Base):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    roles = Column(JSON, nullable=False, default=lambda: ["viewer"])
    is_active = Column(Boolean, default=True, nullable=False)
    last_login = Column(DateTime, nullable=True)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_user_username_active', 'username', 'is_active'),
        Index('idx_user_email_active', 'email', 'is_active'),
    )

# ================================
# Pydantic Schemas
# ================================

class UserBase(BaseModel):
    """Base user schema with common fields."""
    username: str = Field(..., min_length=3, max_length=50, regex=r'^[a-zA-Z0-9_]+$')
    email: EmailStr
    roles: List[UserRole] = Field(default=[UserRole.VIEWER])
    
    @validator('username')
    def validate_username(cls, v):
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Username must contain only letters, numbers, and underscores')
        return v.lower()
    
    @validator('roles')
    def validate_roles(cls, v):
        if not v:
            return [UserRole.VIEWER]
        # Ensure all roles are valid
        valid_roles = [role.value for role in UserRole]
        for role in v:
            if role not in valid_roles:
                raise ValueError(f'Invalid role: {role}')
        return v

class UserCreate(UserBase):
    """Schema for creating a new user."""
    password: str = Field(..., min_length=8, max_length=128)
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password complexity."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one number')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        return v

class UserUpdate(BaseModel):
    """Schema for updating user information."""
    email: Optional[EmailStr] = None
    roles: Optional[List[UserRole]] = None
    is_active: Optional[bool] = None
    
    model_config = {"from_attributes": True}

class UserResponse(UserBase):
    """Schema for user response data."""
    id: int
    is_active: bool
    last_login: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    
    model_config = {"from_attributes": True}

class LoginRequest(BaseModel):
    """Schema for login request."""
    username: str
    password: str

class Token(BaseModel):
    """Schema for token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenData(BaseModel):
    """Schema for token data."""
    username: Optional[str] = None
    roles: List[str] = []
    exp: Optional[datetime] = None

class RefreshTokenRequest(BaseModel):
    """Schema for refresh token request."""
    refresh_token: str

# ================================
# Authentication Service
# ================================

class AuthService:
    """Service class for authentication and authorization operations."""
    
    def __init__(self):
        self.pwd_context = pwd_context
        self.secret_key = JWT_SECRET_KEY
        self.algorithm = JWT_ALGORITHM
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        try:
            return self.pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        return self.pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: dict) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> TokenData:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            roles: List[str] = payload.get("roles", [])
            exp: datetime = datetime.fromtimestamp(payload.get("exp", 0))
            token_type_payload: str = payload.get("type", "access")
            
            if username is None or token_type_payload != token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
            
            return TokenData(username=username, roles=roles, exp=exp)
        except JWTError as e:
            logger.warning(f"Token verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def authenticate_user(self, db: Session, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""
        try:
            user = db.query(User).filter(
                User.username == username.lower(),
                User.is_active == True
            ).first()
            
            if not user:
                logger.warning(f"Authentication failed: user not found - {username}")
                return None
            
            # Check if account is locked
            if user.locked_until and user.locked_until > datetime.now():
                logger.warning(f"Authentication failed: account locked - {username}")
                return None
            
            if not self.verify_password(password, user.hashed_password):
                # Increment failed login attempts
                user.failed_login_attempts += 1
                if user.failed_login_attempts >= 5:
                    user.locked_until = datetime.now() + timedelta(minutes=30)
                    logger.warning(f"Account locked due to failed attempts - {username}")
                db.commit()
                logger.warning(f"Authentication failed: invalid password - {username}")
                return None
            
            # Reset failed attempts on successful login
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.now()
            db.commit()
            
            logger.info(f"User authenticated successfully - {username}")
            return user
            
        except Exception as e:
            logger.error(f"Authentication error for user {username}: {e}")
            db.rollback()
            return None
    
    def create_user(self, db: Session, user_data: UserCreate) -> User:
        """Create a new user."""
        try:
            # Check if username or email already exists
            existing_user = db.query(User).filter(
                (User.username == user_data.username.lower()) |
                (User.email == user_data.email)
            ).first()
            
            if existing_user:
                if existing_user.username == user_data.username.lower():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Username already registered"
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Email already registered"
                    )
            
            # Create new user
            hashed_password = self.get_password_hash(user_data.password)
            db_user = User(
                username=user_data.username.lower(),
                email=user_data.email,
                hashed_password=hashed_password,
                roles=[role.value for role in user_data.roles]
            )
            
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            
            logger.info(f"User created successfully - {db_user.username}")
            return db_user
            
        except HTTPException:
            raise
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Database integrity error creating user: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User creation failed due to data conflict"
            )
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating user: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )

# Global auth service instance
auth_service = AuthService()

# ================================
# FastAPI Dependencies
# ================================

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token_data = auth_service.verify_token(token, "access")
        username = token_data.username
        if username is None:
            raise credentials_exception
    except HTTPException:
        raise credentials_exception
    
    user = db.query(User).filter(
        User.username == username,
        User.is_active == True
    ).first()
    
    if user is None:
        raise credentials_exception
    
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

def require_roles(required_roles: List[UserRole]):
    """Decorator factory for role-based access control."""
    def role_checker(current_user: User = Depends(get_current_active_user)):
        user_roles = current_user.roles or []
        if not any(role in user_roles for role in [r.value for r in required_roles]):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker

# ================================
# Streamlit Integration Helpers
# ================================

class StreamlitAuth:
    """Helper class for Streamlit authentication integration."""
    
    @staticmethod
    def store_token(access_token: str, refresh_token: str):
        """Store tokens in Streamlit session state."""
        import streamlit as st
        st.session_state['access_token'] = access_token
        st.session_state['refresh_token'] = refresh_token
        st.session_state['authenticated'] = True
    
    @staticmethod
    def get_token() -> Optional[str]:
        """Get access token from Streamlit session state."""
        import streamlit as st
        return st.session_state.get('access_token')
    
    @staticmethod
    def clear_tokens():
        """Clear tokens from Streamlit session state."""
        import streamlit as st
        for key in ['access_token', 'refresh_token', 'authenticated', 'user_info']:
            if key in st.session_state:
                del st.session_state[key]
    
    @staticmethod
    def is_authenticated() -> bool:
        """Check if user is authenticated in Streamlit."""
        import streamlit as st
        return st.session_state.get('authenticated', False)
    
    @staticmethod
    def get_user_info() -> Optional[Dict[str, Any]]:
        """Get user info from session state."""
        import streamlit as st
        return st.session_state.get('user_info')
    
    @staticmethod
    def store_user_info(user: User):
        """Store user info in session state."""
        import streamlit as st
        st.session_state['user_info'] = {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'roles': user.roles,
            'is_active': user.is_active
        }

# ================================
# FastAPI Router
# ================================

auth_router = APIRouter(prefix="/auth", tags=["authentication"])

@auth_router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Authenticate user and return tokens."""
    user = auth_service.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_service.create_access_token(
        data={"sub": user.username, "roles": user.roles},
        expires_delta=access_token_expires
    )
    refresh_token = auth_service.create_refresh_token(
        data={"sub": user.username, "roles": user.roles}
    )
    
    logger.info(f"User logged in successfully - {user.username}")
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@auth_router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_request: RefreshTokenRequest,
    db: Session = Depends(get_db)
):
    """Refresh access token using refresh token."""
    try:
        token_data = auth_service.verify_token(refresh_request.refresh_token, "refresh")
        username = token_data.username
        
        # Verify user still exists and is active
        user = db.query(User).filter(
            User.username == username,
            User.is_active == True
        ).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create new tokens
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = auth_service.create_access_token(
            data={"sub": user.username, "roles": user.roles},
            expires_delta=access_token_expires
        )
        refresh_token = auth_service.create_refresh_token(
            data={"sub": user.username, "roles": user.roles}
        )
        
        logger.info(f"Token refreshed for user - {user.username}")
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

@auth_router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user information."""
    return current_user

@auth_router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_active_user)
):
    """Logout user (client should discard tokens)."""
    logger.info(f"User logged out - {current_user.username}")
    return {"message": "Successfully logged out"}

@auth_router.post("/register", response_model=UserResponse)
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.ADMIN]))
):
    """Register a new user (admin only)."""
    user = auth_service.create_user(db, user_data)
    return user

@auth_router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.ADMIN]))
):
    """Update user information (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    try:
        if user_update.email is not None:
            user.email = user_update.email
        if user_update.roles is not None:
            user.roles = [role.value for role in user_update.roles]
        if user_update.is_active is not None:
            user.is_active = user_update.is_active
        
        user.updated_at = datetime.now()
        db.commit()
        db.refresh(user)
        
        logger.info(f"User updated by {current_user.username} - {user.username}")
        return user
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )

# ================================
# Utility Functions
# ================================

def check_user_permission(user: User, required_roles: List[UserRole]) -> bool:
    """Check if user has required roles."""
    user_roles = user.roles or []
    return any(role in user_roles for role in [r.value for r in required_roles])

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username."""
    return db.query(User).filter(
        User.username == username.lower(),
        User.is_active == True
    ).first()

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email."""
    return db.query(User).filter(
        User.email == email,
        User.is_active == True
    ).first()

# Export main components
__all__ = [
    "User",
    "UserCreate",
    "UserUpdate", 
    "UserResponse",
    "Token",
    "TokenData",
    "LoginRequest",
    "AuthService",
    "auth_service",
    "get_current_user",
    "get_current_active_user",
    "require_roles",
    "StreamlitAuth",
    "auth_router",
    "check_user_permission",
    "get_user_by_username",
    "get_user_by_email"
]

logger.info("Authentication module initialized successfully")
