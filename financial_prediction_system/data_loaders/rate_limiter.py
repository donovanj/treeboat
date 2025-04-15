import time
import threading
from functools import wraps
from typing import Callable, Optional, Dict, Any
from financial_prediction_system.logging_config import logger


class TokenBucket:
    """
    Token Bucket implementation for rate limiting.
    
    This algorithm allows for controlled bursts of operations while
    maintaining a steady long-term rate of operations.
    """
    
    def __init__(self, tokens_per_second: float, max_tokens: int):
        """
        Initialize the token bucket.
        
        Args:
            tokens_per_second: Rate at which tokens are added to the bucket
            max_tokens: Maximum number of tokens the bucket can hold
        """
        self.tokens_per_second = tokens_per_second
        self.max_tokens = max_tokens
        self.tokens = max_tokens  # Start with a full bucket
        self.last_refill_time = time.time()
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        
    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.time()
        elapsed = now - self.last_refill_time
        new_tokens = elapsed * self.tokens_per_second
        self.tokens = min(self.tokens + new_tokens, self.max_tokens)
        self.last_refill_time = now
        
        # Round to 6 decimal places to avoid floating point precision issues
        self.tokens = round(self.tokens, 6)
        
    def consume(self, tokens: int = 1, wait: bool = True) -> bool:
        """
        Attempt to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            wait: If True, wait for tokens to become available
                  If False, return immediately if tokens aren't available
                  
        Returns:
            True if tokens were consumed, False otherwise
        """
        if tokens > self.max_tokens:
            raise ValueError(f"Requested tokens ({tokens}) exceeds bucket capacity ({self.max_tokens})")
        
        with self.lock:
            if wait:
                wait_time = self._wait_time_for_tokens(tokens)
                if wait_time > 0:
                    time.sleep(wait_time)
                    
                # Refill after waiting
                self._refill()
                self.tokens -= tokens
                
                # Force to exact zero if we're very close to avoid floating point issues
                if self.tokens < 0.0001:
                    self.tokens = 0
                    
                return True
            else:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    
                    # Force to exact zero if we're very close to avoid floating point issues
                    if self.tokens < 0.0001:
                        self.tokens = 0
                        
                    return True
                return False
    
    def _wait_time_for_tokens(self, tokens: int) -> float:
        """Calculate time to wait for requested tokens to become available."""
        self._refill()
        if self.tokens >= tokens:
            return 0
        
        additional_tokens_needed = tokens - self.tokens
        return additional_tokens_needed / self.tokens_per_second


# Singleton registry of rate limiters
_rate_limiters: Dict[str, TokenBucket] = {}
_registry_lock = threading.RLock()


def get_rate_limiter(name: str, tokens_per_second: float = 1.0, max_tokens: int = 10) -> TokenBucket:
    """
    Get or create a rate limiter by name.
    
    Args:
        name: Name of the rate limiter
        tokens_per_second: Rate at which tokens are added (used only if creating new)
        max_tokens: Maximum number of tokens (used only if creating new)
        
    Returns:
        The token bucket rate limiter
    """
    with _registry_lock:
        if name not in _rate_limiters:
            _rate_limiters[name] = TokenBucket(tokens_per_second, max_tokens)
        return _rate_limiters[name]


def rate_limited(name: str, tokens: int = 1, tokens_per_second: float = 1.0, max_tokens: int = 10):
    """
    Decorator for rate limiting a function.
    
    Args:
        name: Name of the rate limiter
        tokens: Number of tokens this function consumes
        tokens_per_second: Rate at which tokens are added (used only if creating new)
        max_tokens: Maximum number of tokens (used only if creating new)
        
    Returns:
        Decorated function that respects rate limits
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            rate_limiter = get_rate_limiter(name, tokens_per_second, max_tokens)
            
            # Try to get tokens, waiting if necessary
            logger.debug(f"Rate limiter '{name}': Waiting for {tokens} token(s)")
            rate_limiter.consume(tokens, wait=True)
            logger.debug(f"Rate limiter '{name}': Got {tokens} token(s), executing function")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def with_retry(max_retries: int = 3, base_delay: float = 1.0, 
               backoff_factor: float = 2.0, 
               exceptions: tuple = (Exception,)) -> Callable:
    """
    Decorator to retry a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        backoff_factor: Factor by which the delay increases with each retry
        exceptions: Exception types that trigger a retry
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for retry in range(max_retries + 1):  # +1 for the initial attempt
                try:
                    if retry > 0:
                        delay = base_delay * (backoff_factor ** (retry - 1))
                        logger.warning(f"Retrying {func.__name__} after error. "
                                      f"Retry {retry}/{max_retries}, waiting {delay:.2f}s")
                        time.sleep(delay)
                    
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(f"Error in {func.__name__}: {str(e)}")
            
            # If we've exhausted all retries, re-raise the last exception
            logger.error(f"Failed after {max_retries} retries: {str(last_exception)}")
            raise last_exception
        
        return wrapper
    return decorator 