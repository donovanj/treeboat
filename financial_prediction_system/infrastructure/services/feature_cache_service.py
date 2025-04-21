import redis
import pickle
import json
import hashlib
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

class FeatureCacheService:
    """Service for caching feature calculation results"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_expiry = 3600  # 1 hour
        self.composition_expiry = 86400  # 24 hours
    
    def _get_cache_key(self, composition_id: str, params: Dict[str, Any]) -> str:
        """Generate cache key from composition ID and parameters"""
        params_str = json.dumps(params, sort_keys=True)
        return f"feature:{composition_id}:{hashlib.md5(params_str.encode()).hexdigest()}"
    
    def _get_time_series_key(self, symbol: str, start_date: datetime, end_date: datetime) -> str:
        """Generate cache key for time series data"""
        date_str = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
        return f"timeseries:{symbol}:{date_str}"
    
    async def get_cached_features(
        self,
        composition_id: str,
        params: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Retrieve cached feature calculation results"""
        try:
            cache_key = self._get_cache_key(composition_id, params)
            cached_data = self.redis.get(cache_key)
            
            if cached_data:
                return pickle.loads(cached_data)
            
            return None
            
        except redis.RedisError as e:
            logger.error(f"Redis error in get_cached_features: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached features: {str(e)}")
            return None
    
    async def cache_features(
        self,
        composition_id: str,
        params: Dict[str, Any],
        features: np.ndarray,
        expiry: Optional[int] = None
    ) -> bool:
        """Cache feature calculation results"""
        try:
            cache_key = self._get_cache_key(composition_id, params)
            
            # Pickle numpy array for storage
            cached_data = pickle.dumps(features)
            
            expiry = expiry or self.default_expiry
            success = self.redis.setex(
                cache_key,
                expiry,
                cached_data
            )
            
            return bool(success)
            
        except redis.RedisError as e:
            logger.error(f"Redis error in cache_features: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error caching features: {str(e)}")
            return False
    
    async def cache_time_series(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        data: Dict[str, List[Any]]
    ) -> bool:
        """Cache time series data"""
        try:
            cache_key = self._get_time_series_key(symbol, start_date, end_date)
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in data.items()
            }
            
            success = self.redis.setex(
                cache_key,
                self.default_expiry,
                json.dumps(serializable_data)
            )
            
            return bool(success)
            
        except redis.RedisError as e:
            logger.error(f"Redis error in cache_time_series: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error caching time series: {str(e)}")
            return False
    
    async def get_cached_time_series(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[Dict[str, List[Any]]]:
        """Retrieve cached time series data"""
        try:
            cache_key = self._get_time_series_key(symbol, start_date, end_date)
            cached_data = self.redis.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            
            return None
            
        except redis.RedisError as e:
            logger.error(f"Redis error in get_cached_time_series: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached time series: {str(e)}")
            return None
    
    async def cache_composition(
        self,
        composition_id: str,
        composition_data: Dict[str, Any]
    ) -> bool:
        """Cache feature composition configuration"""
        try:
            cache_key = f"composition:{composition_id}"
            
            success = self.redis.setex(
                cache_key,
                self.composition_expiry,
                json.dumps(composition_data)
            )
            
            return bool(success)
            
        except redis.RedisError as e:
            logger.error(f"Redis error in cache_composition: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error caching composition: {str(e)}")
            return False
    
    async def get_cached_composition(
        self,
        composition_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached feature composition"""
        try:
            cache_key = f"composition:{composition_id}"
            cached_data = self.redis.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            
            return None
            
        except redis.RedisError as e:
            logger.error(f"Redis error in get_cached_composition: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached composition: {str(e)}")
            return None
    
    async def invalidate_feature_cache(
        self,
        composition_id: str
    ) -> bool:
        """Invalidate all cached data for a feature composition"""
        try:
            pattern = f"feature:{composition_id}:*"
            keys = self.redis.keys(pattern)
            
            if keys:
                self.redis.delete(*keys)
            
            # Also remove composition cache
            self.redis.delete(f"composition:{composition_id}")
            
            return True
            
        except redis.RedisError as e:
            logger.error(f"Redis error in invalidate_feature_cache: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error invalidating cache: {str(e)}")
            return False
    
    async def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries"""
        try:
            # Get all feature and composition keys
            feature_keys = self.redis.keys("feature:*")
            composition_keys = self.redis.keys("composition:*")
            time_series_keys = self.redis.keys("timeseries:*")
            
            all_keys = feature_keys + composition_keys + time_series_keys
            cleaned = 0
            
            for key in all_keys:
                if not self.redis.ttl(key):
                    self.redis.delete(key)
                    cleaned += 1
            
            return cleaned
            
        except redis.RedisError as e:
            logger.error(f"Redis error in cleanup_expired_cache: {str(e)}")
            return 0
        except Exception as e:
            logger.error(f"Error cleaning up cache: {str(e)}")
            return 0