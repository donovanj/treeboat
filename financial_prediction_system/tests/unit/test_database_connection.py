import pytest
import os
from unittest.mock import patch, MagicMock
from sqlalchemy.orm import Session
from sqlalchemy import text

from financial_prediction_system.infrastructure.database.connection import (
    DataSource, DatabaseConnection, DataSourceFactory, Base
)


class TestDatabaseConnection:
    """Test suite for the DatabaseConnection class"""
    
    def test_singleton_pattern(self):
        """Test that DatabaseConnection implements the singleton pattern correctly"""
        # Create two instances
        conn1 = DatabaseConnection()
        conn2 = DatabaseConnection()
        
        # Both variables should reference the same instance
        assert conn1 is conn2
        
        # Create instance with different connection URL
        conn3 = DatabaseConnection("sqlite:///test.db")
        
        # Should still be the same instance
        assert conn1 is conn3
        
        # But the connection URL should not have changed (initialization happens only once)
        assert conn1._connection_url != "sqlite:///test.db"
    
    @patch('financial_prediction_system.infrastructure.database.connection.create_engine')
    def test_connect(self, mock_create_engine):
        """Test that connect method creates engine and session factory"""
        # Setup
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        # Create a new instance to avoid interference from other tests
        # Clear the instances dictionary first
        DatabaseConnection._instances = {}
        conn = DatabaseConnection("sqlite:///test.db")
        
        # Test connect method
        conn.connect()
        
        # Verify create_engine was called with the correct URL
        mock_create_engine.assert_called_once_with("sqlite:///test.db")
        
        # Verify the engine was set
        assert conn._engine is mock_engine
        
        # Verify session_factory was created
        assert conn._session_factory is not None
    
    def test_is_connected(self):
        """Test is_connected method returns correct state"""
        # Clear the instances dictionary first
        DatabaseConnection._instances = {}
        conn = DatabaseConnection()
        
        # Initially should not be connected
        assert not conn.is_connected()
        
        # After connecting should return True
        with patch.object(conn, '_engine', MagicMock()):
            assert conn.is_connected()
    
    @patch('financial_prediction_system.infrastructure.database.connection.create_engine')
    def test_disconnect(self, mock_create_engine):
        """Test disconnect method properly cleans up resources"""
        # Setup
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        # Clear the instances dictionary first
        DatabaseConnection._instances = {}
        conn = DatabaseConnection()
        conn.connect()
        
        # Test disconnect
        conn.disconnect()
        
        # Verify engine dispose was called
        mock_engine.dispose.assert_called_once()
        
        # Verify engine and session_factory were cleared
        assert conn._engine is None
        assert conn._session_factory is None
    
    def test_get_session(self):
        """Test get_session method returns a session and auto-connects if needed"""
        # Clear the instances dictionary first
        DatabaseConnection._instances = {}
        conn = DatabaseConnection()
        
        # Mock the connect method
        with patch.object(conn, 'connect') as mock_connect:
            # Mock session factory
            mock_session = MagicMock()
            conn._session_factory = MagicMock(return_value=mock_session)
            
            # Call get_session
            session = conn.get_session()
            
            # Should not have called connect since we manually set session_factory
            mock_connect.assert_not_called()
            
            # Should have returned the session
            assert session is mock_session
        
        # Reset session_factory to None
        conn._session_factory = None
        
        # Now test auto-connect behavior
        with patch.object(conn, 'connect') as mock_connect:
            # Mock session factory that will be set by connect
            mock_session = MagicMock()
            mock_connect.side_effect = lambda: setattr(conn, '_session_factory', MagicMock(return_value=mock_session))
            
            # Call get_session
            session = conn.get_session()
            
            # Should have called connect
            mock_connect.assert_called_once()
            
            # Should have returned the session
            assert session is mock_session
    
    def test_get_base(self):
        """Test that get_base returns the correct Base class"""
        conn = DatabaseConnection()
        assert conn.get_base() is Base


class TestDataSourceFactory:
    """Test suite for the DataSourceFactory class"""
    
    @patch.object(DatabaseConnection, 'connect')
    def test_create_database_connection(self, mock_connect):
        """Test factory creates and initializes a DatabaseConnection correctly"""
        # Clear the instances dictionary first
        DatabaseConnection._instances = {}
        
        # Call factory method
        db_conn = DataSourceFactory.create_database_connection("sqlite:///test.db")
        
        # Verify it's a DatabaseConnection instance
        assert isinstance(db_conn, DatabaseConnection)
        
        # Verify connect was called
        mock_connect.assert_called_once()
        
        # Verify connection URL was set
        assert db_conn._connection_url == "sqlite:///test.db"

def test_database_session_creation(test_db):
    """Test that a database session can be created."""
    # The test_db fixture should provide a working session
    assert test_db is not None, "Database session should not be None"

def test_database_connection(test_db):
    """Test that the database connection works."""
    # Try to execute a simple query
    result = test_db.execute(text("SELECT 1"))
    row = result.fetchone()
    assert row[0] == 1, "Database query should return 1"

def test_database_tables_creation(test_db):
    """Test that database tables can be created."""
    # Check that the metadata contains tables
    from tests.conftest import Base
    
    # This is not an ideal test as we don't have models yet,
    # but it checks that the Base class is properly defined
    assert hasattr(Base, 'metadata'), "Base should have metadata attribute" 