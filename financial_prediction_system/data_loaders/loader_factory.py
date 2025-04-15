from typing import Dict, Type, Optional
from sqlalchemy.orm import Session
from financial_prediction_system.logging_config import logger
from .base_loader import BaseDataLoader
from .stock_loader import StockDataLoader
from .treasury_loader import TreasuryDataLoader
from .index_loader import IndexDataLoader
from .cache_factory import CacheFactory
import pandas as pd
from datetime import date

class DataLoaderFactory:
    _loaders: Dict[str, Type[BaseDataLoader]] = {
        "stock": StockDataLoader,
        "treasury": TreasuryDataLoader,
        "index": IndexDataLoader
    }

    @classmethod
    def get_loader(cls, loader_type: str, db: Session, use_cache: bool = True) -> BaseDataLoader:
        """
        Get a data loader instance for the specified type.
        
        Args:
            loader_type: Type of loader to create
            db: Database session
            use_cache: Whether to enable caching for this loader
            
        Returns:
            Instance of the requested loader
            
        Raises:
            ValueError: If loader_type is not supported
        """
        logger.debug(f"Requesting loader for type: {loader_type}")
        if loader_type not in cls._loaders:
            logger.error(f"Invalid loader type requested: {loader_type}")
            raise ValueError(f"Invalid loader type: {loader_type}")
        
        loader_class = cls._loaders[loader_type]
        logger.info(f"Creating {loader_class.__name__} instance")
        
        # Initialize with or without cache
        if use_cache:
            cache = CacheFactory.get_redis_cache()
            return loader_class(db, cache)
        else:
            return loader_class(db)

    @classmethod
    def get_available_loaders(cls) -> list:
        """Get a list of available loader types"""
        logger.debug("Fetching available loader types")
        return list(cls._loaders.keys())

    @classmethod
    def get_index_data(cls, db: Session, start_date, end_date) -> Dict[str, pd.DataFrame]:
        """
        Get market index data for feature calculations directly from database.
        
        Args:
            db: Database session
            start_date: Start date for the data
            end_date: End date for the data
            
        Returns:
            Dictionary of DataFrames with index data
        """
        logger.debug(f"Loading index data from {start_date} to {end_date}")
        
        # Convert string dates to date objects if needed
        if isinstance(start_date, str):
            try:
                start_date = date.fromisoformat(start_date.split('T')[0])
            except ValueError:
                logger.error(f"Invalid start_date format: {start_date}")
                return {}
                
        if isinstance(end_date, str):
            try:
                end_date = date.fromisoformat(end_date.split('T')[0])
            except ValueError:
                logger.error(f"Invalid end_date format: {end_date}")
                return {}
        
        # Convert datetime to date if needed
        start_date = start_date.date() if hasattr(start_date, 'date') else start_date
        end_date = end_date.date() if hasattr(end_date, 'date') else end_date
        
        # Create index loader with caching disabled to prevent API calls
        index_loader = cls.get_loader("index", db, use_cache=False)
        
        # Load data for major indices
        index_data = {}
        for index_symbol in ['SPX', 'NDX', 'VIX', 'RUT', 'DJI', 'SOX']:
            try:
                # Get data from database only, don't try to load new data if missing
                model_class = index_loader.model_map.get(f"{index_symbol.lower()}_prices")
                if not model_class:
                    logger.warning(f"Unknown table for index: {index_symbol}")
                    continue
                    
                query = db.query(model_class).filter(
                    model_class.symbol == index_symbol,
                    model_class.date >= start_date,
                    model_class.date <= end_date
                ).order_by(model_class.date)
                
                data = [{
                    'date': record.date,
                    'open': float(record.open) if record.open is not None else None,
                    'high': float(record.high) if record.high is not None else None,
                    'low': float(record.low) if record.low is not None else None,
                    'close': float(record.close) if record.close is not None else None,
                    'volume': int(record.volume) if record.volume is not None else None
                } for record in query]
                
                if not data:
                    logger.warning(f"No data found for index {index_symbol} from {start_date} to {end_date}")
                    continue
                    
                df = pd.DataFrame(data)
                df.set_index('date', inplace=True)
                index_data[f"{index_symbol.lower()}_prices"] = df
                
            except Exception as e:
                logger.warning(f"Failed to load data for index {index_symbol}: {str(e)}")
        
        return index_data

    @classmethod
    def register_loader(cls, loader_type: str, loader_class: Type[BaseDataLoader]):
        """
        Register a new loader type.
        
        Args:
            loader_type: Type identifier for the loader
            loader_class: Class implementing the loader
        """
        logger.info(f"Registering new loader type: {loader_type}")
        if not issubclass(loader_class, BaseDataLoader):
            logger.error(f"Invalid loader class: {loader_class.__name__}")
            raise ValueError(f"{loader_class.__name__} must inherit from BaseDataLoader")
        
        cls._loaders[loader_type] = loader_class
        logger.debug(f"Successfully registered {loader_type} loader") 