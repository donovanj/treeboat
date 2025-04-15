from typing import Generator

from sqlalchemy.orm import Session

from financial_prediction_system.infrastructure.database.connection import SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get a database session.
    
    Yields:
        Session: A database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 