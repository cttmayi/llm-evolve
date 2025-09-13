"""
Process utilities using pebble for better process management
Compatible with concurrent.futures.ProcessPoolExecutor API
"""
import os
import logging
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Callable, Optional, TypeVar
from pebble import ProcessPool, ProcessFuture
from pebble.common import ProcessExpired

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ProcessFuture:
    """Wrapper around pebble.ProcessFuture to match concurrent.futures.Future API"""
    
    def __init__(self, pebble_future: ProcessFuture, timeout: Optional[float] = None):
        self._pebble_future = pebble_future
        self._timeout = timeout
    
    def result(self, timeout: Optional[float] = None) -> Any:
        """Get the result of the future, with timeout support"""
        try:
            # Use provided timeout or the one from submit
            effective_timeout = timeout if timeout is not None else self._timeout
            return self._pebble_future.result(timeout=effective_timeout)
        except ProcessExpired as e:
            logger.warning(f"Process expired: {e}")
            raise RuntimeError(f"Process expired: {e}") from e
        except TimeoutError as e:
            # Convert pebble TimeoutError to concurrent.futures.TimeoutError
            raise FutureTimeoutError from e
        except Exception as e:
            logger.error(f"Error getting future result: {e}")
            raise
    
    def exception(self, timeout: Optional[float] = None) -> Optional[BaseException]:
        """Get the exception of the future"""
        try:
            effective_timeout = timeout if timeout is not None else self._timeout
            return self._pebble_future.exception(timeout=effective_timeout)
        except ProcessExpired as e:
            return RuntimeError(f"Process expired: {e}")
        except TimeoutError:
            return FutureTimeoutError()
        except Exception as e:
            return e
    
    def cancel(self) -> bool:
        """Attempt to cancel the future"""
        try:
            return self._pebble_future.cancel()
        except Exception as e:
            logger.warning(f"Failed to cancel future: {e}")
            return False
    
    def cancelled(self) -> bool:
        """Return True if the future was cancelled"""
        try:
            return self._pebble_future.cancelled()
        except Exception:
            return False
    
    def running(self) -> bool:
        """Return True if the future is currently executing"""
        try:
            return self._pebble_future.running()
        except Exception:
            return False
    
    def done(self) -> bool:
        """Return True if the future has finished executing"""
        try:
            return self._pebble_future.done()
        except Exception:
            return True  # Assume done if we can't determine
    
    def add_done_callback(self, fn: Callable[['ProcessFuture'], None]) -> None:
        """Add a callback function to be called when the future finishes"""
        def callback_wrapper(pebble_future):
            try:
                fn(self)
            except Exception as e:
                logger.error(f"Error in done callback: {e}")
        
        try:
            self._pebble_future.add_done_callback(callback_wrapper)
        except Exception as e:
            logger.error(f"Failed to add done callback: {e}")


class ProcessPoolExecutor:
    """Process pool executor using pebble for better process management
    
    Compatible with concurrent.futures.ProcessPoolExecutor API but with additional
    features like proper process termination after function execution.
    """
    
    def __init__(self, max_workers: Optional[int] = None,
                 initializer: Optional[Callable] = None,
                 initargs: tuple = ()):
        """Initialize the process pool executor
        
        Args:
            max_workers: Maximum number of worker processes
            initializer: Function to call in each worker process on startup
            initargs: Arguments to pass to the initializer function
        """
        self._max_workers = max_workers or (os.cpu_count() or 1)
        self._initializer = initializer
        self._initargs = initargs
        self._pool: Optional[ProcessPool] = None
        self._shutdown = False
        
        logger.debug(f"Initialized ProcessPoolExecutor with max_workers={self._max_workers}")
    
    def _ensure_pool(self) -> ProcessPool:
        """Ensure the process pool is created"""
        if self._shutdown:
            raise RuntimeError("Cannot submit to shutdown executor")

        if self._pool is None:
            pool_kwargs = {}
            if self._max_workers is not None:
                pool_kwargs['max_workers'] = self._max_workers
            if self._initializer is not None:
                pool_kwargs['initializer'] = self._initializer
            if self._initargs:
                pool_kwargs['initargs'] = self._initargs

            self._pool = ProcessPool(**pool_kwargs)
            logger.debug(f"Created process pool with {self._max_workers} workers")
        return self._pool
    
    def submit(self, fn: Callable[..., T], *args, timeout: Optional[float] = None, **kwargs) -> ProcessFuture:
        """Submit a function to be executed in a worker process
        
        Args:
            fn: Function to execute
            *args: Positional arguments for the function
            timeout: Optional timeout in seconds for the function execution
            **kwargs: Keyword arguments for the function
            
        Returns:
            ProcessFuture that represents the execution of the function
        """
        if self._shutdown:
            raise RuntimeError("Cannot submit to shutdown executor")
        
        pool = self._ensure_pool()
        
        try:
            # Schedule the function with pebble
            schedule_kwargs = {
                'fn': fn,
                'args': list(args),
            }
            if timeout is not None:
                schedule_kwargs['timeout'] = timeout
            
            pebble_future = pool.schedule(**schedule_kwargs, **kwargs)

            # Wrap in our ProcessFuture
            return ProcessFuture(pebble_future, timeout)
            
        except Exception as e:
            logger.error(f"Failed to submit function {fn.__name__}: {e}")
            raise
    
    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Shutdown the process pool
        
        Args:
            wait: Whether to wait for all futures to complete
            cancel_futures: Whether to cancel all pending futures
        """
        self._shutdown = True
        
        if self._pool is not None:
            try:
                if cancel_futures:
                    self._pool.stop()  # Cancel all pending futures
                else:
                    self._pool.close()
                
                if wait:
                    self._pool.join()
                    
                logger.debug("Process pool shutdown completed")
                
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
            finally:
                self._pool = None
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown(wait=True)
        return False


def run_function_in_process(func: Callable[..., T], *args, timeout: Optional[float] = None, **kwargs) -> T:
    """Simple function to run a function in a separate process with timeout

    Args:
        func: Function to execute in the process
        *args: Positional arguments for the function
        timeout: Optional timeout in seconds
        **kwargs: Keyword arguments for the function

    Returns:
        The result of the function execution

    Raises:
        RuntimeError: If process expires or other execution errors
        TimeoutError: If execution exceeds timeout
    """
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, timeout=timeout, **kwargs)
        return future.result(timeout=timeout)


# Drop-in replacement aliases for easy migration
ProcessPoolExecutor = ProcessPoolExecutor
Future = ProcessFuture


if __name__ == "__main__":
    # Simple test
    import time
    
    def test_function(duration: float, name: str) -> str:
        """Test function that sleeps for duration"""
        logger.info(f"Process {os.getpid()}: Starting {name} for {duration}s")
        time.sleep(duration)
        logger.info(f"Process {os.getpid()}: Finished {name}")
        return f"{name} completed after {duration}s"
    
    # Test with timeout
    def test_timeout_function() -> str:
        """Function that will timeout"""
        logger.info(f"Process {os.getpid()}: Starting long-running task")
        time.sleep(10)  # This will timeout
        return "Should not reach here"
    
    # Run tests
    logging.basicConfig(level=logging.INFO)
    
    with ProcessPoolExecutor(max_workers=2) as executor:
        # Test normal execution
        future1 = executor.submit(test_function, 1.0, "Task1")
        future2 = executor.submit(test_function, 2.0, "Task2")
        
        print(f"Task1 result: {future1.result()}")
        print(f"Task2 result: {future2.result()}")
        
        # Test timeout
        future3 = executor.submit(test_timeout_function, timeout=3.0)
        try:
            result = future3.result(timeout=4.0)
            print(f"Timeout task result: {result}")
        except FutureTimeoutError:
            print("Task timed out as expected")
        except Exception as e:
            print(f"Task failed: {e}")
    
    print("All tests completed")

    # Test the new simple function
    print("\n=== Testing run_function_in_process ===")

    # Test normal execution
    result = run_function_in_process(test_function, 0.5, "SimpleTest")
    print(f"Simple function result: {result}")

    # Test with timeout
    try:
        result = run_function_in_process(test_timeout_function, timeout=2.0)
        print(f"Should not reach here: {result}")
    except TimeoutError:
        print("Simple function timeout test passed")
    except Exception as e:
        print(f"Simple function timeout test failed: {e}")