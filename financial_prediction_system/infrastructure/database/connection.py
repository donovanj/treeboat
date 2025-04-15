"""
Database connection module implementing Singleton pattern for database connections
and providing a unified interface for data sources.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.ext.declarative import DeclarativeMeta
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Type
from abc import ABC, abstractmethod

# Load environment variables
load_dotenv()

# Base class for declarative models
Base = declarative_base()

class DataSource(ABC):
    """Abstract base class defining the interface for all data sources"""
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the data source"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the data source"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connection is established"""
        pass
    
    @abstractmethod
    def get_session(self):
        """Get a session or connection object to interact with the data source"""
        pass


class DatabaseConnection(DataSource):
    """
    Singleton implementation of a database connection.
    Ensures only one instance of the database connection exists.
    """
    _instances: Dict[Type, Any] = {}
    
    def __new__(cls, *args, **kwargs):
        """Implement the Singleton pattern"""
        if cls not in cls._instances:
            instance = super(DatabaseConnection, cls).__new__(cls)
            cls._instances[cls] = instance
        return cls._instances[cls]
    
    def __init__(self, connection_url: Optional[str] = None):
        """Initialize the database connection"""
        if not hasattr(self, '_initialized'):  # Prevent re-initialization
            self._connection_url = connection_url or os.getenv("DATABASE_URL")
            self._engine = None
            self._session_factory = None
            self._initialized = True
    
    def connect(self) -> None:
        """Establish connection to the database"""
        if not self._engine:
            self._engine = create_engine(self._connection_url)
            self._session_factory = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)
    
    def disconnect(self) -> None:
        """Close the database connection"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
    
    def is_connected(self) -> bool:
        """Check if database connection is established"""
        return self._engine is not None
    
    def get_session(self):
        """Get a new session for database operations"""
        if not self._session_factory:
            self.connect()
        return self._session_factory()
    
    def get_engine(self):
        """Get the SQLAlchemy engine instance"""
        if not self._engine:
            self.connect()
        return self._engine
    
    def get_base(self) -> DeclarativeMeta:
        """Get the declarative base for ORM models"""
        return Base


class DataSourceFactory:
    """Factory for creating different types of data sources"""
    
    @staticmethod
    def create_database_connection(connection_url: Optional[str] = None) -> DatabaseConnection:
        """Create and return a database connection instance"""
        db_connection = DatabaseConnection(connection_url)
        db_connection.connect()
        return db_connection


# Default database connection instance for backward compatibility
default_db_connection = DataSourceFactory.create_database_connection()
SessionLocal = default_db_connection.get_session
engine = default_db_connection.get_engine() 