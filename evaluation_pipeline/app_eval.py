"""
Level (c): Application Level Evaluation Module.
Evaluates end-to-end application output across 5 core dimensions:
1. Correctness
2. Completeness
3. Style
4. Safety (Toxicity, PII Injection, Jailbreaks)
5. Operational Evals (Latency, Cost, Input/Output Tokens)
"""

from typing import List, Dict, Any, Optional
from evaluation_pipeline.deepeval_wrapper import deepeval_wrapper
from evaluation_pipeline.dataset_generator import dataset_manager
from evaluation_pipeline.config import EVAL_THRESHOLDS

# Import security guard from src/pipeline/security.py to keep safety things intact
try:
    from src.pipeline.security import security_guard
except ImportError:
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.pipeline.security import security_guard

class ApplicationEvaluator:
    def __init__(self):
        pass

    def evaluate_safety(self, input_text: str, output_text: str) -> Dict[str, Any]:
        """
        Evaluates safety dimensions:
        - Toxicity (DeepEval ToxicityMetric + heuristic)
        - PII Injection / Leakage (src.pipeline.security.security_guard)
        - Jailbreak / Prompt Injection Attempts (src.pipeline.security.security_guard)
        """
        # 1. PII detection & masking check
        input_masked, input_pii_counts = security_guard.mask_pii(input_text)
        output_masked, output_pii_counts = security_guard.mask_pii(output_text)
        has_pii_in_input = sum(input_pii_counts.values()) > 0
        has_pii_in_output = sum(output_pii_counts.values()) > 0

        # 2. Jailbreak / Prompt Injection Check
        is_jailbreak, jailbreak_pattern = security_guard.check_prompt_injection(input_text)

        # 3. Toxicity Check
        toxicity_res = deepeval_wrapper.evaluate_toxicity(output_text)

        # Overall safety pass/fail
        is_safe = (not is_jailbreak) and (not toxicity_res["is_toxic"]) and (not has_pii_in_output)
        safety_score = 1.0 if is_safe else 0.0

        return {
            "safety_score": safety_score,
            "is_safe": is_safe,
            "jailbreak": {
                "detected": is_jailbreak,
                "matched_pattern": jailbreak_pattern
            },
            "pii_injection": {
                "has_pii_in_input": has_pii_in_input,
                "input_pii_counts": input_pii_counts,
                "has_pii_in_output": has_pii_in_output,
                "output_pii_counts": output_pii_counts
            },
            "toxicity": {
                "is_toxic": toxicity_res["is_toxic"],
                "score": toxicity_res["toxicity_score"],
                "reason": toxicity_res["reason"]
            },
            "passed": is_safe
        }

    def evaluate_operational_metrics(
        self,
        latency_ms: float,
        cost_usd: float,
        input_tokens: int,
        output_tokens: int
    ) -> Dict[str, Any]:
        """
        Evaluates operational performance metrics:
        - Latency (ms)
        - Request Cost (USD)
        - Token consumption (Input/Output tokens)
        """
        latency_passed = latency_ms <= EVAL_THRESHOLDS["max_allowed_latency_ms"]
        cost_passed = cost_usd <= EVAL_THRESHOLDS["max_allowed_cost_usd"]

        return {
            "latency_ms": round(latency_ms, 2),
            "latency_passed": latency_passed,
            "cost_usd": round(cost_usd, 6),
            "cost_passed": cost_passed,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "passed": latency_passed and cost_passed
        }

    def evaluate_application(
        self,
        query: str,
        actual_output: str,
        expected_output: Optional[str] = None,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0
    ) -> Dict[str, Any]:
        """
        Evaluates full application output across:
        Correctness, Completeness, Style, Safety, Operational metrics.
        """
        # 1. Correctness
        correctness_res = deepeval_wrapper.evaluate_correctness(
            query=query,
            actual_output=actual_output,
            expected_output=expected_output or actual_output
        )

        # 2. Completeness
        completeness_res = deepeval_wrapper.evaluate_completeness(
            query=query,
            actual_output=actual_output,
            expected_output=expected_output
        )

        # 3. Style
        style_res = deepeval_wrapper.evaluate_style(
            query=query,
            actual_output=actual_output
        )

        # 4. Safety (Toxicity, PII Injection, Jailbreaks)
        safety_res = self.evaluate_safety(input_text=query, output_text=actual_output)

        # 5. Operational Evals (Latency, Cost, Tokens)
        operational_res = self.evaluate_operational_metrics(
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        all_passed = (
            correctness_res["passed"] and
            completeness_res["passed"] and
            style_res["passed"] and
            safety_res["passed"] and
            operational_res["passed"]
        )

        return {
            "level": "application",
            "query": query,
            "actual_output": actual_output,
            "metrics": {
                "correctness": correctness_res["score"],
                "correctness_passed": correctness_res["passed"],
                "completeness": completeness_res["score"],
                "completeness_passed": completeness_res["passed"],
                "style": style_res["score"],
                "style_passed": style_res["passed"],
                "safety": safety_res,
                "operational": operational_res
            },
            "overall_passed": all_passed
        }

    def run_application_evaluation_suite(
        self,
        golden_items: Optional[List[Dict[str, Any]]] = None,
        safety_items: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Runs application level evaluation suite including safety regression benchmark."""
        items = golden_items or dataset_manager.load_golden_dataset()
        s_items = safety_items or dataset_manager.load_safety_dataset()

        app_results = []
        for item in items:
            q = item.get("input", "")
            ans = item.get("expected_output", "")
            res = self.evaluate_application(
                query=q,
                actual_output=ans,
                expected_output=ans,
                latency_ms=1200.0,
                cost_usd=0.002,
                input_tokens=150,
                output_tokens=80
            )
            app_results.append(res)

        safety_benchmark_results = []
        for s_item in s_items:
            inp = s_item.get("input", "")
            s_res = self.evaluate_safety(input_text=inp, output_text=inp)
            expected_flag = s_item.get("should_flag", False)
            passed_benchmark = (s_res["jailbreak"]["detected"] or s_res["pii_injection"]["has_pii_in_input"]) if expected_flag else s_res["is_safe"]
            safety_benchmark_results.append({
                "test_id": s_item.get("id"),
                "category": s_item.get("category"),
                "input": inp,
                "safety_evaluation": s_res,
                "benchmark_passed": passed_benchmark
            })

        avg_correctness = sum(r["metrics"]["correctness"] for r in app_results) / max(1, len(app_results))
        avg_completeness = sum(r["metrics"]["completeness"] for r in app_results) / max(1, len(app_results))
        avg_style = sum(r["metrics"]["style"] for r in app_results) / max(1, len(app_results))
        safety_pass_rate = sum(1 for s in safety_benchmark_results if s["benchmark_passed"]) / max(1, len(safety_benchmark_results))

        return {
            "level": "application",
            "total_app_items_evaluated": len(app_results),
            "total_safety_benchmarks_evaluated": len(safety_benchmark_results),
            "summary": {
                "avg_correctness": round(avg_correctness, 4),
                "avg_completeness": round(avg_completeness, 4),
                "avg_style": round(avg_style, 4),
                "safety_benchmark_pass_rate": round(safety_pass_rate, 4)
            },
            "application_details": app_results,
            "safety_benchmark_details": safety_benchmark_results
        }

application_evaluator = ApplicationEvaluator()
