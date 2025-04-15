import unittest
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from financial_prediction_system.data_loaders.rate_limiter import (
    TokenBucket, rate_limited, with_retry, get_rate_limiter
)


class TestTokenBucket(unittest.TestCase):
    def test_token_consumption_single_thread(self):
        """Test that tokens are consumed correctly in a single thread"""
        bucket = TokenBucket(tokens_per_second=10, max_tokens=10)
        self.assertEqual(bucket.tokens, 10)
        
        # Consume 5 tokens
        self.assertTrue(bucket.consume(5, wait=False))
        self.assertEqual(bucket.tokens, 5)
        
        # Consume 5 more tokens
        self.assertTrue(bucket.consume(5, wait=False))
        self._assert_approx_equal(bucket.tokens, 0)
        
        # Try to consume 1 token, should fail with wait=False
        self.assertFalse(bucket.consume(1, wait=False))
        self._assert_approx_equal(bucket.tokens, 0)
        
    def test_token_refill(self):
        """Test that tokens are refilled over time"""
        bucket = TokenBucket(tokens_per_second=10, max_tokens=10)
        self.assertEqual(bucket.tokens, 10)
        
        # Consume all tokens
        self.assertTrue(bucket.consume(10, wait=False))
        self.assertEqual(bucket.tokens, 0)
        
        # Wait for tokens to refill
        time.sleep(0.5)  # Wait for 0.5 seconds, should get ~5 tokens
        self._assert_approx_equal(bucket.tokens, 0)  # Still 0 until refill is called
        
        # Try to consume 4 tokens, should trigger refill and succeed
        self.assertTrue(bucket.consume(4, wait=False))
        self._assert_approx_equal(bucket.tokens, 1, delta=1)  # Should have ~1 token left
        
    def test_wait_for_tokens(self):
        """Test that consume waits for tokens when wait=True"""
        bucket = TokenBucket(tokens_per_second=10, max_tokens=10)
        self.assertEqual(bucket.tokens, 10)
        
        # Consume all tokens
        self.assertTrue(bucket.consume(10, wait=False))
        self.assertEqual(bucket.tokens, 0)
        
        # Start timing
        start_time = time.time()
        
        # Try to consume 5 tokens with wait=True
        self.assertTrue(bucket.consume(5, wait=True))
        
        # Check that we waited approximately 0.5 seconds
        elapsed = time.time() - start_time
        self._assert_approx_equal(elapsed, 0.5, delta=0.1)
        
    def test_rate_limited_decorator(self):
        """Test that the rate_limited decorator works"""
        call_times = []
        
        @rate_limited(name="test", tokens=1, tokens_per_second=10, max_tokens=10)
        def test_func():
            call_times.append(time.time())
            return "test"
        
        # Call function 10 times rapidly
        for _ in range(10):
            test_func()
            
        # Check that the first call was immediate
        self.assertEqual(len(call_times), 10)
        
        # Call 5 more times (should be rate limited)
        start_time = time.time()
        for _ in range(5):
            test_func()
        elapsed = time.time() - start_time
        
        # Should have waited approximately 0.5 seconds
        self._assert_approx_equal(elapsed, 0.5, delta=0.1)
        
    def test_concurrent_access(self):
        """Test that the token bucket is thread-safe"""
        bucket = TokenBucket(tokens_per_second=10, max_tokens=10)
        
        # Function to consume tokens
        def consume_token():
            return bucket.consume(1, wait=True)
        
        # Run 20 threads concurrently
        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(lambda _: consume_token(), range(20)))
            
        # All should have succeeded
        self.assertEqual(sum(results), 20)
        
        # Bucket should be empty or nearly empty
        self._assert_approx_equal(bucket.tokens, 0, delta=1)
        
    def test_retry_decorator(self):
        """Test that the with_retry decorator works"""
        attempts = [0]
        
        @with_retry(max_retries=3, base_delay=0.1)
        def fail_then_succeed():
            attempts[0] += 1
            if attempts[0] <= 2:
                raise ValueError("Simulated failure")
            return "success"
            
        # Should retry twice then succeed
        result = fail_then_succeed()
        self.assertEqual(result, "success")
        self.assertEqual(attempts[0], 3)
        
    def test_get_rate_limiter(self):
        """Test the get_rate_limiter function returns singleton instances"""
        limiter1 = get_rate_limiter("test_singleton")
        limiter2 = get_rate_limiter("test_singleton")
        
        # Should be the same instance
        self.assertIs(limiter1, limiter2)
        
        # Consuming from one should affect the other
        limiter1.consume(5, wait=False)
        self.assertEqual(limiter2.tokens, 5)
        
    def _assert_approx_equal(self, a, b, delta=0.01):
        """Assert that two values are approximately equal"""
        self.assertTrue(abs(a - b) <= delta, f"{a} != {b} within {delta}")
        
        
if __name__ == "__main__":
    unittest.main() 