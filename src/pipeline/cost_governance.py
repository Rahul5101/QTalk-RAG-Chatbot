"""
Cost Governance & Budget Manager.
Tracks input/output tokens, estimates costs, routes queries, and enforces daily budget limits.
"""

import os
import time
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger("cost_governance")

# Standard Google Gemini Pricing per 1,000,000 tokens (in USD)
PRICING_CATALOG = {
    "gemini-1.5-flash": {
        "input_per_m": 0.075,
        "output_per_m": 0.30
    },
    "gemini-1.5-pro": {
        "input_per_m": 1.25,
        "output_per_m": 5.00
    },
    "gemini-2.0-flash": {
        "input_per_m": 0.10,
        "output_per_m": 0.40
    },
    "default": {
        "input_per_m": 0.10,
        "output_per_m": 0.40
    }
}

class CostManager:
    def __init__(self):
        self.max_daily_budget = float(os.getenv("MAX_DAILY_BUDGET_USD", "10.0"))
        self.daily_cost_acc = 0.0
        self.last_reset_time = time.time()
        self.pricing = PRICING_CATALOG

    def _check_day_reset(self):
        now = time.time()
        # Reset budget after 24 hours (86400s)
        if now - self.last_reset_time >= 86400:
            self.daily_cost_acc = 0.0
            self.last_reset_time = now

    def estimate_tokens(self, text: str) -> int:
        """
        Approximate token count (1 token ~= 4 characters for English/multilingual text).
        """
        if not text:
            return 0
        return max(1, len(text) // 4)

    def calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate total cost in USD based on input and output tokens.
        """
        rates = self.pricing.get(model_name.lower(), self.pricing["default"])
        input_cost = (input_tokens / 1_000_000.0) * rates["input_per_m"]
        output_cost = (output_tokens / 1_000_000.0) * rates["output_per_m"]
        return round(input_cost + output_cost, 6)

    def check_budget_and_record(self, cost_usd: float) -> Tuple[bool, float]:
        """
        Checks if the daily budget limit is exceeded. Returns (within_budget, current_total_cost).
        """
        self._check_day_reset()
        if self.daily_cost_acc + cost_usd > self.max_daily_budget:
            logger.error(f"Daily budget exceeded: ${self.daily_cost_acc + cost_usd:.4f} > ${self.max_daily_budget:.2f}")
            return False, self.daily_cost_acc

        self.daily_cost_acc += cost_usd
        return True, self.daily_cost_acc

    def route_model(self, query: str) -> str:
        """
        Dynamically route query to appropriate model tier based on task complexity.
        """
        query_len = len(query)
        primary_model = os.getenv("PRIMARY_MODEL", "gemini-1.5-flash")
        
        # Complex multi-step queries or legal analytical requests may use pro model if explicitly flagged
        if "deep analysis" in query.lower() or "draft contract" in query.lower() or query_len > 1500:
            return os.getenv("COMPLEX_MODEL", "gemini-1.5-pro")
        
        return primary_model

cost_manager = CostManager()
