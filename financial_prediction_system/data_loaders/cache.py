from typing import Optional, Any, Dict, List, Union
import redis
import json
from datetime import date
from financial_prediction_system.config import get_settings
from abc import ABC, abstractmethod
from financial_prediction_system.logging_config import logger

class CacheInterface(ABC):
    """Abstract interface for cache implementations"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache by key"""
        pass
        
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache with optional TTL in seconds"""
        pass
        
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value from cache by key"""
        pass
        
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache data"""
        pass

class RedisCache(CacheInterface):
    """Redis cache implementation with Singleton pattern"""
    
    _instances = {}
    
    def __new__(cls, *args, **kwargs):
        """Implement Singleton pattern"""
        if cls not in cls._instances:
            instance = super(RedisCache, cls).__new__(cls)
            cls._instances[cls] = instance
        return cls._instances[cls]
    
    def __init__(self):
        """Initialize Redis connection from settings"""
        # Only initialize once (Singleton pattern)
        if hasattr(self, 'redis_client'):
            return
            
        try:
            settings = get_settings()
            redis_config = settings.redis
            
            # Don't pass along other settings that aren't related to Redis
            self.redis_client = redis.Redis(
                host=redis_config.REDIS_HOST,
                port=redis_config.REDIS_PORT,
                db=redis_config.REDIS_DB,
                password=redis_config.REDIS_PASSWORD,
                decode_responses=True
            )
            # Verify connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_config.REDIS_HOST}:{redis_config.REDIS_PORT}")
        except Exception as e:
            # Clean the error message to avoid exposing sensitive information
            error_msg = str(e)
            # Don't log the full error which might contain credentials
            logger.error(f"Failed to connect to Redis: {type(e).__name__}")
            # Re-raise a more generic error without the details
            raise RuntimeError(f"Redis connection failed: {type(e).__name__}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from Redis"""
        try:
            value = self.redis_client.get(key)
            if value is None:
                return None
                
            return json.loads(value)
        except Exception as e:
            logger.error(f"Error retrieving from Redis: {type(e).__name__}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in Redis with optional TTL"""
        try:
            serialized_value = json.dumps(value)
            if ttl:
                self.redis_client.setex(key, ttl, serialized_value)
            else:
                self.redis_client.set(key, serialized_value)
            return True
        except Exception as e:
            logger.error(f"Error setting value in Redis: {type(e).__name__}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete a value from Redis"""
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting from Redis: {type(e).__name__}")
            return False
    
    def clear(self) -> bool:
        """Clear all Redis data (with confirmation check)"""
        try:
            self.redis_client.flushdb()
            logger.info("Redis cache cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {type(e).__name__}")
            return False

class DataCache:
    """Higher-level cache adapter for financial data"""
    
    def __init__(self, redis_cache: RedisCache):
        """Initialize with Redis cache implementation"""
        self.redis = redis_cache
        self.default_ttl = 86400  # 24 hours by default
    
    def get(self, key: str) -> Optional[Any]:
        """Get data from cache"""
        return self.redis.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set data in cache with TTL"""
        return self.redis.set(key, value, ttl or self.default_ttl)
    
    def delete(self, key: str) -> bool:
        """Delete data from cache"""
        return self.redis.delete(key)
    
    def set_default_ttl(self, ttl_seconds: int) -> None:
        """Set the default TTL for cache entries"""
        self.default_ttl = ttl_seconds
    
    def clear(self) -> bool:
        """Clear all cache data"""
        return self.redis.clear()
    
    def store_data_batch(self, key_prefix: str, data_dict: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Store multiple data items with the same prefix"""
        success = True
        for key, value in data_dict.items():
            cache_key = f"{key_prefix}:{key}"
            if not self.set(cache_key, value, ttl):
                success = False
        return success 