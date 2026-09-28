"""
Centralized Metrics, Telemetry & LangSmith Tracing Manager.
Tracks system metrics (latency, tokens, cost, throughput, errors) and exposes Prometheus/JSON metrics + LangSmith telemetry.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("monitoring")

class MetricsRegistry:
    def __init__(self):
        self.start_time = time.time()
        self.request_total = 0
        self.error_total = 0
        self.latency_sum_ms = 0.0
        self.latency_count = 0

        self.token_input_total = 0
        self.token_output_total = 0
        self.token_cost_usd_total = 0.0

        self.cache_hit_total = 0
        self.cache_miss_total = 0

        self.security_blocked_total = 0
        self.pii_detected_total = 0

    def record_request(self, latency_ms: float, is_error: bool = False):
        self.request_total += 1
        if is_error:
            self.error_total += 1
        self.latency_sum_ms += latency_ms
        self.latency_count += 1

    def record_tokens(self, input_tokens: int, output_tokens: int, cost_usd: float):
        self.token_input_total += input_tokens
        self.token_output_total += output_tokens
        self.token_cost_usd_total += cost_usd

    def record_cache(self, hit: bool):
        if hit:
            self.cache_hit_total += 1
        else:
            self.cache_miss_total += 1

    def record_security(self, blocked: bool = False, pii_count: int = 0):
        if blocked:
            self.security_blocked_total += 1
        if pii_count > 0:
            self.pii_detected_total += pii_count

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        uptime_seconds = time.time() - self.start_time
        avg_latency = round(self.latency_sum_ms / self.latency_count, 2) if self.latency_count > 0 else 0.0
        throughput = round(self.request_total / max(1.0, uptime_seconds), 4)
        hit_rate = round(self.cache_hit_total / max(1, self.cache_hit_total + self.cache_miss_total), 4)

        return {
            "uptime_seconds": round(uptime_seconds, 2),
            "request_total": self.request_total,
            "error_total": self.error_total,
            "throughput_req_per_sec": throughput,
            "latency_ms": {
                "sum": round(self.latency_sum_ms, 2),
                "count": self.latency_count,
                "average_ms": avg_latency
            },
            "token_usage": {
                "input_total": self.token_input_total,
                "output_total": self.token_output_total,
                "grand_total": self.token_input_total + self.token_output_total,
                "cost_usd_total": round(self.token_cost_usd_total, 6)
            },
            "cache": {
                "hits": self.cache_hit_total,
                "misses": self.cache_miss_total,
                "hit_rate": hit_rate
            },
            "security": {
                "blocked_requests": self.security_blocked_total,
                "pii_detected_count": self.pii_detected_total
            }
        }

metrics_registry = MetricsRegistry()

class LangSmithTracer:
    def __init__(self):
        self.tracing_enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true" or os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        self._setup_env()

    def _setup_env(self):
        if os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"):
            os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
            os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
            os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY", "")
            os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag-chatbot-production")
            logger.info("LangSmith tracing environment configured successfully.")

    def log_trace_metadata(self, run_id: str, name: str, metadata: Dict[str, Any], tags: Optional[List[str]] = None):
        """
        Logs trace metadata and tags for LangSmith if active.
        """
        if not self.tracing_enabled:
            return
        
        try:
            from langsmith import Client
            client = Client()
            client.create_run(
                name=name,
                run_type="chain",
                inputs=metadata.get("inputs", {}),
                outputs=metadata.get("outputs", {}),
                extra={"metadata": metadata},
                tags=tags or ["rag-pipeline"]
            )
        except Exception as exc:
            logger.debug(f"LangSmith trace log skipped or failed: {exc}")

langsmith_tracer = LangSmithTracer()
