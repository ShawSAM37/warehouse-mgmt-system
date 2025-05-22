"""
Database connection utilities for the warehouse management system.
This module provides SQLAlchemy session management and base classes.

Implementation follows a progressive enhancement approach:
1. Core functionality with robust error handling
2. Health check capabilities
3. Enhanced logging for troubleshooting
4. Optional connection pool tuning via environment variables
"""
import os
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get database URL from environment variable or use SQLite as default
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./warehouse.db')

# Log database connection info (without credentials)
def _get_sanitized_db_url():
    """Returns database URL with credentials removed for logging"""
    if DATABASE_URL.startswith('sqlite'):
        return DATABASE_URL
    
    import re
    # Replace username:password with username:***
    sanitized = re.sub(r'//([^:]+):([^@]+)@', r'//\1:***@', DATABASE_URL)
    return sanitized

logger.info(f"Connecting to database: {_get_sanitized_db_url()}")

# Create SQLAlchemy engine with appropriate connection arguments
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith('sqlite') else {},
    pool_pre_ping=True,  # Verify connections before using from pool
    echo=os.getenv('SQL_ECHO', 'false').lower() == 'true'  # Log SQL statements if enabled
)

# Add SQLite optimizations if using SQLite
if DATABASE_URL.startswith('sqlite'):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
            cursor.close()
            logger.debug("SQLite pragmas configured successfully")
        except Exception as e:
            logger.warning(f"SQLite pragma setup failed: {str(e)}")
            # Continue without WAL mode - will fall back to default journal mode

# Apply connection pool tuning if configured via environment variables
if not DATABASE_URL.startswith('sqlite'):  # SQLite doesn't use connection pooling
    pool_size = os.getenv('DB_POOL_SIZE')
    max_overflow = os.getenv('DB_MAX_OVERFLOW')
    pool_timeout = os.getenv('DB_POOL_TIMEOUT')
    pool_recycle = os.getenv('DB_POOL_RECYCLE')
    
    if any([pool_size, max_overflow, pool_timeout, pool_recycle]):
        logger.info("Applying custom database pool settings")
        
        if pool_size:
            engine.pool._pool.size = int(pool_size)
            logger.info(f"Pool size set to {pool_size}")
            
        if max_overflow:
            engine.pool._pool.max_overflow = int(max_overflow)
            logger.info(f"Max overflow set to {max_overflow}")
            
        if pool_timeout:
            engine.pool._pool.timeout = int(pool_timeout)
            logger.info(f"Pool timeout set to {pool_timeout} seconds")
            
        if pool_recycle:
            engine.pool._pool.recycle = int(pool_recycle)
            logger.info(f"Connection recycle time set to {pool_recycle} seconds")

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative class definitions
Base = declarative_base()

def get_db():
    """
    FastAPI dependency for database sessions.
    Yields a database session and ensures it is closed after use.
    
    Usage:
        @app.get("/items/")
        def read_items(db: Session = Depends(get_db)):
            return db.query(models.Item).all()
    """
    db = SessionLocal()
    logger.debug("Database session opened via get_db()")
    try:
        yield db
    finally:
        db.close()
        logger.debug("Database session closed via get_db()")

@contextmanager
def db_session():
    """
    Context manager for database sessions.
    Provides a transactional scope around a series of operations.
    
    Usage:
        with db_session() as session:
            session.add(some_object)
            session.commit()
    """
    session = SessionLocal()
    session_id = id(session)
    logger.debug(f"Database session {session_id} opened")
    try:
        yield session
        session.commit()
        logger.debug(f"Transaction committed for session {session_id}")
    except Exception as e:
        logger.error(f"Rolling back transaction for session {session_id} due to: {str(e)}")
        session.rollback()
        raise
    finally:
        session.close()
        logger.debug(f"Database session {session_id} closed")

def check_database_health():
    """
    Performs a lightweight check to verify database connectivity.
    Returns True if database is accessible, False otherwise.
    
    Usage:
        is_healthy = check_database_health()
        if not is_healthy:
            # Take appropriate action
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database health check: OK")
        return True
    except Exception as e:
        logger.critical(f"Database health check failed: {str(e)}")
        return False

def validate_database_url():
    """
    Validates the database URL format and scheme.
    Logs warnings for potentially unsupported configurations.
    """
    try:
        from urllib.parse import urlparse
        result = urlparse(DATABASE_URL)
        
        # Check scheme
        supported_schemes = [
            "sqlite", 
            "postgresql", "postgres", 
            "mysql", "mysql+pymysql",
            "oracle", 
            "mssql", "mssql+pyodbc"
        ]
        
        if result.scheme not in supported_schemes:
            logger.warning(f"Potentially unsupported database scheme: {result.scheme}")
            logger.warning(f"Supported schemes include: {', '.join(supported_schemes)}")
        
        # For SQLite, check if path exists or is creatable
        if result.scheme == "sqlite" and result.path != ":memory:":
            db_path = result.path.replace("///", "")
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                logger.warning(f"SQLite database directory does not exist: {db_dir}")
                
        logger.info(f"Database URL validation completed for {result.scheme}")
    except Exception as e:
        logger.warning(f"Database URL validation failed: {str(e)}")

def init_db():
    """
    Initialize the database by creating all tables.
    Should be called once at application startup.
    """
    try:
        # Validate database URL before attempting to create tables
        validate_database_url()
        
        # Verify database connectivity
        if not check_database_health():
            logger.error("Database health check failed during initialization")
            raise Exception("Cannot connect to database")
        
        # Import all modules that define models
        # IMPORTANT: As new model modules are added, import them here
        from ..inventory import orm_models
        # Uncomment these as they are implemented:
        # from ..gate_entry import orm_models
        # from ..quality import orm_models
        # from ..replenishment import orm_models
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Verify table creation
        with engine.connect() as conn:
            # Get list of tables
            if DATABASE_URL.startswith('sqlite'):
                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            else:
                # Generic query that works for most databases
                result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()"))
            
            tables = [row[0] for row in result]
            logger.info(f"Database initialized with tables: {', '.join(tables)}")
            
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise

# Run validation at module import time (non-blocking)
validate_database_url()
