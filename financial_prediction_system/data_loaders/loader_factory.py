from typing import Dict, Type, Optional
from sqlalchemy.orm import Session
from logging_config import logger
from .base_loader import BaseDataLoader
from .stock_loader import StockDataLoader
from .treasury_loader import TreasuryDataLoader
from .index_loader import IndexDataLoader
from .cache_factory import CacheFactory

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