"""
Circuit Breaker Module for External Services (Milvus, LLM APIs).
Prevents cascading failures by short-circuiting calls during downstream service outages.
"""

import time
import logging
from enum import Enum
from typing import Callable, Any, Dict, Optional

logger = logging.getLogger("circuit_breaker")

class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class CircuitBreakerOpenException(Exception):
    pass

class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 3, recovery_timeout: float = 30.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_state_change = time.time()

    def __call__(self, func: Callable, *args, **kwargs) -> Any:
        return self.call(func, *args, **kwargs)

    def call(self, func: Callable, *args, **kwargs) -> Any:
        now = time.time()

        if self.state == CircuitState.OPEN:
            if now - self.last_state_change > self.recovery_timeout:
                logger.info(f"CircuitBreaker [{self.name}] transitioning OPEN -> HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = now
            else:
                logger.warning(f"CircuitBreaker [{self.name}] is OPEN. Short-circuiting call.")
                raise CircuitBreakerOpenException(f"Service [{self.name}] is currently unavailable (Circuit Breaker OPEN).")

        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure(e)
            raise e

    async def call_async(self, async_func: Callable, *args, **kwargs) -> Any:
        now = time.time()

        if self.state == CircuitState.OPEN:
            if now - self.last_state_change > self.recovery_timeout:
                logger.info(f"CircuitBreaker [{self.name}] transitioning OPEN -> HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = now
            else:
                logger.warning(f"CircuitBreaker [{self.name}] is OPEN. Short-circuiting call.")
                raise CircuitBreakerOpenException(f"Service [{self.name}] is currently unavailable (Circuit Breaker OPEN).")

        try:
            result = await async_func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure(e)
            raise e

    def on_success(self):
        if self.state == CircuitState.HALF_OPEN or self.failure_count > 0:
            logger.info(f"CircuitBreaker [{self.name}] reset to CLOSED after successful call.")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_state_change = time.time()

    def on_failure(self, exception: Exception):
        self.failure_count += 1
        logger.error(f"CircuitBreaker [{self.name}] observed failure #{self.failure_count}: {exception}")
        if self.failure_count >= self.failure_threshold or self.state == CircuitState.HALF_OPEN:
            logger.error(f"CircuitBreaker [{self.name}] tripped to OPEN state.")
            self.state = CircuitState.OPEN
            self.last_state_change = time.time()

llm_circuit_breaker = CircuitBreaker("LLM_Service", failure_threshold=3, recovery_timeout=30.0)
milvus_circuit_breaker = CircuitBreaker("Milvus_Vector_DB", failure_threshold=3, recovery_timeout=30.0)
