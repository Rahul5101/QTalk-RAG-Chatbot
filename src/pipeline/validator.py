"""
Output Validator Module.
Validates LLM outputs, executes retries with exponential backoff, and routes to fallback models.
"""

import json
import re
import time
import logging
from typing import Dict, Any, Tuple, Optional, Callable

logger = logging.getLogger("output_validator")

REQUIRED_KEYS = ["response", "bold_words", "meta_data", "follow_up", "table_data", "confidence_score"]

class OutputValidator:
    def __init__(self, max_retries: int = 2, backoff_factor: float = 1.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def validate_structure(self, data: Any) -> Tuple[bool, Dict[str, Any], str]:
        """
        Validates structure of the LLM response.
        Returns: (is_valid, normalized_dict, error_message)
        """
        if not isinstance(data, dict):
            return False, {}, "Response is not a valid JSON dictionary."

        # Map alternate field names if model returns raw schema format
        normalized = {
            "response": data.get("response") or data.get("Explanation") or "",
            "bold_words": data.get("bold_words") or [],
            "meta_data": data.get("meta_data") or [],
            "follow_up": data.get("follow_up") or data.get("Follow_up") or "",
            "table_data": data.get("table_data") or [],
            "confidence_score": data.get("confidence_score") if data.get("confidence_score") is not None else data.get("Confidence_Score", 0.0),
            "ucid": data.get("ucid", "99_18")
        }

        if not normalized["response"]:
            return False, normalized, "Missing main response/explanation text."

        return True, normalized, ""

    def parse_and_validate(self, raw_text: str) -> Tuple[bool, Dict[str, Any], str]:
        """
        Parses JSON from string and validates schema.
        """
        cleaned = (raw_text or "").strip()
        if not cleaned:
            return False, {}, "Empty raw text response."

        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        parsed = None
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(cleaned[start:end + 1])
                except json.JSONDecodeError:
                    pass

        if not parsed:
            return False, {}, "Failed to parse JSON string."

        return self.validate_structure(parsed)

    def execute_with_retry_and_fallback(
        self,
        primary_fn: Callable[[], Any],
        fallback_fn: Optional[Callable[[], Any]] = None,
        default_fallback_payload: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], int, str]:
        """
        Executes primary_fn with retries. On failure, triggers fallback_fn or default_fallback_payload.
        Returns: (final_response_dict, attempt_count, executed_mode)
        """
        attempt = 0
        last_error = ""

        while attempt <= self.max_retries:
            attempt += 1
            try:
                raw_res = primary_fn()
                if isinstance(raw_res, dict):
                    valid, norm_data, err = self.validate_structure(raw_res)
                else:
                    valid, norm_data, err = self.parse_and_validate(str(raw_res))

                if valid:
                    return norm_data, attempt, "primary"
                
                last_error = err
                logger.warning(f"Validation failed on attempt {attempt}/{self.max_retries + 1}: {err}")
            except Exception as exc:
                last_error = str(exc)
                logger.error(f"Primary execution error on attempt {attempt}/{self.max_retries + 1}: {exc}")

            if attempt <= self.max_retries:
                sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                time.sleep(sleep_time)

        # Retries exhausted -> Execute Fallback
        logger.warning(f"Primary model retries exhausted. Invoking fallback strategy. Last error: {last_error}")
        if fallback_fn:
            try:
                fallback_res = fallback_fn()
                if isinstance(fallback_res, dict):
                    v, norm_f, _ = self.validate_structure(fallback_res)
                    if v:
                        return norm_f, attempt, "fallback_model"
            except Exception as fb_exc:
                logger.error(f"Fallback model execution failed: {fb_exc}")

        # Ultimate Fallback Payload
        fallback_payload = default_fallback_payload or {
            "response": "I encountered an issue generating a formatted answer. Please rephrase your question or try again.",
            "bold_words": [],
            "meta_data": [],
            "follow_up": "Would you like to try asking a simpler question?",
            "table_data": [],
            "confidence_score": 0.0,
            "ucid": "99_18"
        }
        return fallback_payload, attempt, "fallback_payload"

output_validator = OutputValidator()
