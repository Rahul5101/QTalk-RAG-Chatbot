"""
Level (b): Pipeline Level Evaluation Module.
Evaluates the entire end-to-end RAG pipeline response using mandatory 3 metrics:
1. Answer Relevance
2. Context Relevance
3. Faithfulness
"""

from typing import List, Dict, Any, Optional
from evaluation_pipeline.deepeval_wrapper import deepeval_wrapper
from evaluation_pipeline.dataset_generator import dataset_manager

class PipelineEvaluator:
    def __init__(self):
        pass

    def evaluate_pipeline_response(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluates the RAG pipeline end-to-end.
        Required metrics: Answer Relevance, Context Relevance, Faithfulness.
        """
        # 1. Answer Relevance
        ans_relevance_res = deepeval_wrapper.evaluate_answer_relevance(query, answer)

        # 2. Context Relevance
        ctx_relevance_res = deepeval_wrapper.evaluate_context_relevance(query, retrieved_chunks)

        # 3. Faithfulness
        faithfulness_res = deepeval_wrapper.evaluate_faithfulness(query, answer, retrieved_chunks)

        # Overall pipeline score calculation
        overall_score = round(
            (ans_relevance_res["score"] + ctx_relevance_res["score"] + faithfulness_res["score"]) / 3.0, 4
        )

        all_passed = (
            ans_relevance_res["passed"] and
            ctx_relevance_res["passed"] and
            faithfulness_res["passed"]
        )

        return {
            "level": "pipeline",
            "query": query,
            "answer": answer,
            "retrieved_chunks_count": len(retrieved_chunks),
            "metrics": {
                "answer_relevance": ans_relevance_res["score"],
                "answer_relevance_reason": ans_relevance_res["reason"],
                "answer_relevance_passed": ans_relevance_res["passed"],

                "context_relevance": ctx_relevance_res["score"],
                "context_relevance_reason": ctx_relevance_res["reason"],
                "context_relevance_passed": ctx_relevance_res["passed"],

                "faithfulness": faithfulness_res["score"],
                "faithfulness_reason": faithfulness_res["reason"],
                "faithfulness_passed": faithfulness_res["passed"]
            },
            "overall_pipeline_score": overall_score,
            "overall_passed": all_passed
        }

    def run_pipeline_evaluation_suite(self, golden_items: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Runs pipeline level evaluation over golden dataset."""
        items = golden_items or dataset_manager.load_golden_dataset()
        suite_results = []

        for item in items:
            q = item.get("input", "")
            ans = item.get("expected_output", "")
            retrieved = item.get("retrieved_contexts", [])

            res = self.evaluate_pipeline_response(query=q, answer=ans, retrieved_chunks=retrieved)
            suite_results.append(res)

        avg_ans_rel = sum(r["metrics"]["answer_relevance"] for r in suite_results) / max(1, len(suite_results))
        avg_ctx_rel = sum(r["metrics"]["context_relevance"] for r in suite_results) / max(1, len(suite_results))
        avg_faith = sum(r["metrics"]["faithfulness"] for r in suite_results) / max(1, len(suite_results))
        avg_overall = sum(r["overall_pipeline_score"] for r in suite_results) / max(1, len(suite_results))

        return {
            "level": "pipeline",
            "total_items_evaluated": len(suite_results),
            "summary": {
                "avg_answer_relevance": round(avg_ans_rel, 4),
                "avg_context_relevance": round(avg_ctx_rel, 4),
                "avg_faithfulness": round(avg_faith, 4),
                "avg_overall_pipeline_score": round(avg_overall, 4)
            },
            "details": suite_results
        }

pipeline_evaluator = PipelineEvaluator()
