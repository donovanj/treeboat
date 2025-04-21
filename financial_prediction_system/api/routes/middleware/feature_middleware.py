from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Callable, Dict, Any
import time
import redis
import json
from functools import wraps
import hashlib
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiting for feature engineering endpoints"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.window = 60  # 1 minute window
        self.max_requests = 100  # Maximum requests per window
    
    async def check_rate_limit(self, request: Request) -> None:
        client_ip = request.client.host
        current = int(time.time())
        key = f"ratelimit:{client_ip}:{current // self.window}"
        
        pipe = self.redis.pipeline()
        try:
            pipe.incr(key)
            pipe.expire(key, self.window)
            requests = pipe.execute()[0]
            
            if requests > self.max_requests:
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests. Please try again later."
                )
        except redis.RedisError as e:
            logger.error(f"Redis error in rate limiter: {str(e)}")
            # Allow request if Redis is down
            pass

def cache_response(
    expire: int = 3600,  # 1 hour default
    condition: Callable[[Dict[str, Any]], bool] = lambda _: True
):
    """Cache response data for feature calculations"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, request: Request, **kwargs):
            # Skip caching for non-GET requests
            if request.method != "GET":
                return await func(*args, request=request, **kwargs)
            
            # Generate cache key from request parameters
            params = dict(request.query_params)
            if not condition(params):
                return await func(*args, request=request, **kwargs)
                
            cache_key = f"cache:{request.url.path}:{hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()}"
            
            # Try to get from cache
            try:
                cached = request.app.state.redis.get(cache_key)
                if cached:
                    return JSONResponse(content=json.loads(cached))
            except redis.RedisError as e:
                logger.error(f"Redis error in cache retrieval: {str(e)}")
            
            # Execute function and cache result
            response = await func(*args, request=request, **kwargs)
            
            try:
                request.app.state.redis.setex(
                    cache_key,
                    expire,
                    json.dumps(response.body)
                )
            except redis.RedisError as e:
                logger.error(f"Redis error in cache storage: {str(e)}")
            
            return response
            
        return wrapper
    return decorator

class FeatureCalculationError(HTTPException):
    """Custom exception for feature calculation errors"""
    def __init__(self, detail: str):
        super().__init__(status_code=400, detail=detail)

async def handle_feature_errors(request: Request, call_next):
    """Global error handler for feature engineering endpoints"""
    try:
        return await call_next(request)
    except FeatureCalculationError as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail}
        )
    except Exception as e:
        logger.error(f"Unexpected error in feature calculation: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error during feature calculation"}
        )

def validate_data_requirements(data: Dict[str, Any]) -> None:
    """Validate data requirements for feature calculation"""
    required_fields = {'date', 'close'}
    missing_fields = required_fields - set(data.keys())
    
    if missing_fields:
        raise FeatureCalculationError(
            f"Missing required fields: {missing_fields}"
        )
    
    # Validate data types and lengths
    try:
        for field in required_fields:
            if not isinstance(data[field], list):
                raise FeatureCalculationError(
                    f"Field {field} must be an array"
                )
    except KeyError:
        pass  # Already handled by missing fields check

def setup_feature_middleware(app, redis_client: redis.Redis):
    """Setup all feature engineering middleware"""
    rate_limiter = RateLimiter(redis_client)
    
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        if request.url.path.startswith("/api/features"):
            await rate_limiter.check_rate_limit(request)
        return await call_next(request)
    
    @app.middleware("http")
    async def feature_error_middleware(request: Request, call_next):
        if request.url.path.startswith("/api/features"):
            return await handle_feature_errors(request, call_next)
        return await call_next(request)