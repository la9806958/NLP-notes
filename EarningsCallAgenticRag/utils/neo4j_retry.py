"""neo4j_retry.py - Neo4j transaction retry logic with exponential backoff
================================================================
Provides decorators and utilities to handle Neo4j deadlocks and transient errors
with automatic retry logic, exponential backoff, and jitter.
"""

import time
import random
import functools
from typing import Any, Callable, Dict, List, Optional, Type, Union
from neo4j.exceptions import TransientError, ServiceUnavailable
import logging

# Configure logging
logger = logging.getLogger(__name__)

class Neo4jRetryConfig:
    """Configuration for Neo4j retry logic."""

    def __init__(
        self,
        max_retries: int = 5,
        initial_delay: float = 0.1,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [
            TransientError,
            ServiceUnavailable,
        ]

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff and optional jitter."""
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )

        if self.jitter:
            # Add jitter: random value between 50% and 100% of calculated delay
            delay = delay * (0.5 + random.random() * 0.5)

        return delay

def neo4j_retry(
    config: Optional[Neo4jRetryConfig] = None,
    max_retries: int = 5,
    initial_delay: float = 0.1,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """
    Decorator for automatic retry of Neo4j operations with exponential backoff.

    Args:
        config: Neo4jRetryConfig instance. If provided, other parameters are ignored.
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delay

    Usage:
        @neo4j_retry(max_retries=3, initial_delay=0.5)
        def my_neo4j_operation(session):
            return session.run("MATCH (n) RETURN n")
    """

    if config is None:
        config = Neo4jRetryConfig(
            max_retries=max_retries,
            initial_delay=initial_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter
        )

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_retries + 1):  # +1 for initial attempt
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    # Check if this is a retryable exception
                    if not any(isinstance(e, exc_type) for exc_type in config.retryable_exceptions):
                        # Not a retryable exception, re-raise immediately
                        raise

                    # Check for specific deadlock detection
                    if isinstance(e, TransientError) and "DeadlockDetected" in str(e):
                        logger.warning(f"Neo4j deadlock detected on attempt {attempt + 1}: {e}")
                    else:
                        logger.warning(f"Neo4j transient error on attempt {attempt + 1}: {e}")

                    last_exception = e

                    # If this was the last attempt, re-raise the exception
                    if attempt >= config.max_retries:
                        logger.error(f"Max retries ({config.max_retries}) exceeded for Neo4j operation")
                        raise last_exception

                    # Calculate and apply delay
                    delay = config.calculate_delay(attempt)
                    logger.info(f"Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{config.max_retries})")
                    time.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper
    return decorator

def execute_with_retry(
    operation: Callable,
    config: Optional[Neo4jRetryConfig] = None,
    *args,
    **kwargs
) -> Any:
    """
    Execute a Neo4j operation with retry logic.

    Args:
        operation: Function to execute
        config: Retry configuration
        *args, **kwargs: Arguments to pass to the operation

    Returns:
        Result of the operation
    """
    if config is None:
        config = Neo4jRetryConfig()

    @neo4j_retry(config=config)
    def _execute():
        return operation(*args, **kwargs)

    return _execute()

class Neo4jRetrySession:
    """
    Wrapper around Neo4j session that automatically retries failed operations.
    """

    def __init__(self, session, config: Optional[Neo4jRetryConfig] = None):
        self.session = session
        self.config = config or Neo4jRetryConfig()

    @neo4j_retry()
    def run(self, query: str, parameters: Optional[Dict] = None, **kwargs):
        """Execute a query with automatic retry on transient errors."""
        return self.session.run(query, parameters, **kwargs)

    @neo4j_retry()
    def write_transaction(self, tx_function: Callable, *args, **kwargs):
        """Execute a write transaction with automatic retry on transient errors."""
        return self.session.write_transaction(tx_function, *args, **kwargs)

    @neo4j_retry()
    def read_transaction(self, tx_function: Callable, *args, **kwargs):
        """Execute a read transaction with automatic retry on transient errors."""
        return self.session.read_transaction(tx_function, *args, **kwargs)

    def close(self):
        """Close the underlying session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def create_retry_session(driver, config: Optional[Neo4jRetryConfig] = None, **session_kwargs):
    """
    Create a Neo4j session with automatic retry capabilities.

    Args:
        driver: Neo4j driver instance
        config: Retry configuration
        **session_kwargs: Additional arguments for session creation

    Returns:
        Neo4jRetrySession instance
    """
    session = driver.session(**session_kwargs)
    return Neo4jRetrySession(session, config)

# Transaction-level retry utilities
def retry_transaction(
    session,
    tx_function: Callable,
    tx_args: tuple = (),
    tx_kwargs: Optional[Dict] = None,
    config: Optional[Neo4jRetryConfig] = None,
    is_write: bool = True
) -> Any:
    """
    Execute a transaction function with retry logic.

    Args:
        session: Neo4j session
        tx_function: Transaction function to execute
        tx_args: Arguments for transaction function
        tx_kwargs: Keyword arguments for transaction function
        config: Retry configuration
        is_write: Whether this is a write transaction

    Returns:
        Result of the transaction
    """
    if config is None:
        config = Neo4jRetryConfig()

    if tx_kwargs is None:
        tx_kwargs = {}

    @neo4j_retry(config=config)
    def _execute_transaction():
        if is_write:
            return session.write_transaction(tx_function, *tx_args, **tx_kwargs)
        else:
            return session.read_transaction(tx_function, *tx_args, **tx_kwargs)

    return _execute_transaction()

# Batch operation utilities
def execute_batch_with_retry(
    session,
    batch_operations: List[Dict[str, Any]],
    config: Optional[Neo4jRetryConfig] = None,
    batch_size: int = 100
) -> List[Any]:
    """
    Execute a batch of operations with retry logic and chunking.

    Args:
        session: Neo4j session
        batch_operations: List of operations, each with 'query' and optional 'parameters'
        config: Retry configuration
        batch_size: Size of each batch chunk

    Returns:
        List of results from all operations
    """
    if config is None:
        config = Neo4jRetryConfig()

    results = []

    # Process in chunks to avoid large transactions
    for i in range(0, len(batch_operations), batch_size):
        chunk = batch_operations[i:i + batch_size]

        @neo4j_retry(config=config)
        def _execute_chunk():
            chunk_results = []
            with session.begin_transaction() as tx:
                for op in chunk:
                    query = op['query']
                    parameters = op.get('parameters', {})
                    result = tx.run(query, parameters)
                    chunk_results.append(result.data())
                tx.commit()
            return chunk_results

        chunk_results = _execute_chunk()
        results.extend(chunk_results)

    return results