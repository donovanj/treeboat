from functools import wraps
from typing import Callable, Any, Optional, Dict, List
from datetime import date
import json
from .cache import RedisCache, DataCache
from financial_prediction_system.logging_config import logger

def cacheable(cache_key_prefix: str):
    """
    Decorator for caching data loader methods.
    
    Args:
        cache_key_prefix: Prefix for cache key
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if caching is enabled for this loader
            if not hasattr(self, 'cache') or self.cache is None:
                return func(self, *args, **kwargs)
                
            # Extract parameters for cache key
            start_date = kwargs.get('start_date')
            end_date = kwargs.get('end_date')
            symbol = kwargs.get('symbol', 'all')
            
            # Generate cache key
            date_range = ""
            if start_date and end_date:
                date_range = f"{start_date.isoformat()}_{end_date.isoformat()}"
            elif start_date:
                date_range = f"{start_date.isoformat()}_latest"
            elif end_date:
                date_range = f"earliest_{end_date.isoformat()}"
            else:
                date_range = "all"
                
            cache_key = f"{cache_key_prefix}:{symbol}:{date_range}"
            
            # Try to get from cache
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_data
                
            # Execute function if not in cache
            result = func(self, *args, **kwargs)
            
            # Cache the result
            self.cache.set(cache_key, result)
            logger.debug(f"Cached result for {cache_key}")
            
            return result
        return wrapper
    return decorator
    
def invalidate_cache(cache_key_prefix: str, symbol: Optional[str] = None):
    """Function to invalidate cache for a specific prefix and symbol"""
    def invalidate(cache: RedisCache):
        if symbol:
            pattern = f"{cache_key_prefix}:{symbol}:*"
        else:
            pattern = f"{cache_key_prefix}:*"
            
        try:
            keys = cache.redis_client.keys(pattern)
            if keys:
                cache.redis_client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache keys with pattern {pattern}")
            return True
        except Exception as e:
            logger.error(f"Error invalidating cache pattern {pattern}: {str(e)}")
            return False
    return invalidate 