import pytest
import sys
import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, project_root)

# Define Base locally for testing
Base = declarative_base()

@pytest.fixture
def test_db():
    """
    Create a test database for testing.
    
    This fixture creates a test database connection, sets up tables,
    provides a session to the test, and then tears down the tables after the test.
    
    Yields:
        Session: A SQLAlchemy session connected to the test database
    """
    # Use SQLite in-memory database for testing
    TEST_DATABASE_URL = "sqlite:///./test.db"
    
    # Create engine and session factory
    engine = create_engine(
        TEST_DATABASE_URL, connect_args={"check_same_thread": False}
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Provide session
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        
    # Drop tables
    Base.metadata.drop_all(bind=engine) 