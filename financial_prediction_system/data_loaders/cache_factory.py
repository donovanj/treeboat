from functools import lru_cache
from typing import Optional
from .cache import RedisCache, DataCache
import logging
from config import get_settings

logger = logging.getLogger(__name__)

class CacheFactory:
    """Factory for creating cache instances"""
    
    _redis_cache: Optional[RedisCache] = None
    _data_cache: Optional[DataCache] = None
    
    @classmethod
    @lru_cache()
    def get_redis_cache(cls) -> Optional[RedisCache]:
        """Get a Redis cache instance (Singleton pattern)"""
        if cls._redis_cache is None:
            try:
                cls._redis_cache = RedisCache()
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Redis cache: {str(e)}")
                cls._redis_cache = None
        return cls._redis_cache
        
    @classmethod
    @lru_cache()
    def get_data_cache(cls) -> Optional[DataCache]:
        """Get a data cache instance (Singleton pattern)"""
        redis_cache = cls.get_redis_cache()
        if redis_cache is None:
            return None
            
        if cls._data_cache is None:
            cls._data_cache = DataCache(redis_cache)
            logger.info("Data cache initialized successfully")
        return cls._data_cache
        
    @classmethod
    def clear_cache(cls):
        """Clear all cache data"""
        redis_cache = cls.get_redis_cache()
        if redis_cache:
            try:
                redis_cache.clear()
                logger.info("Cache cleared successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to clear cache: {str(e)}")
        return False 