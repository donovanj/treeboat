# Utility to get all stock symbols from the database
from sqlalchemy.orm import Session
from .stock_loader import StockDataLoader

def get_all_symbols_from_db(session: Session):
    """
    Fetch all stock symbols from the database using StockDataLoader's private method.
    """
    loader = StockDataLoader(session)
    return loader._get_eligible_stocks()
