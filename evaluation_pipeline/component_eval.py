"""
Level (a): Component Level Evaluation Module.
Evaluates RAG pipeline components (Retriever & Generator) independently in isolation.
"""

from typing import List, Dict, Any, Optional
from evaluation_pipeline.deepeval_wrapper import deepeval_wrapper
from evaluation_pipeline.dataset_generator import dataset_manager

class ComponentEvaluator:
    def __init__(self):
        pass

    def evaluate_retriever(
        self,
        query: str,
        retrieved_chunks: List[str],
        ground_truth_chunks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates Retriever Component in isolation using:
        - recall@k (Contextual Recall)
        - precision@k (Contextual Precision)
        """
        precision_res = deepeval_wrapper.evaluate_retriever_precision(query, retrieved_chunks, ground_truth_chunks)
        recall_res = deepeval_wrapper.evaluate_retriever_recall(query, retrieved_chunks, ground_truth_chunks)

        overall_passed = precision_res["passed"] and recall_res["passed"]

        return {
            "component": "retriever",
            "query": query,
            "metrics": {
                "precision_at_k": precision_res["score"],
                "precision_reason": precision_res["reason"],
                "precision_passed": precision_res["passed"],
                "recall_at_k": recall_res["score"],
                "recall_reason": recall_res["reason"],
                "recall_passed": recall_res["passed"]
            },
            "overall_passed": overall_passed
        }

    def evaluate_generator(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[str],
        expected_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates Generator Component in isolation with custom QnA / Golden dataset using:
        - Faithfulness / Hallucination
        - Answer Relevance
        - Citation / Reference Alignment
        - Reference-Free LLM Judge
        """
        faithfulness_res = deepeval_wrapper.evaluate_faithfulness(query, answer, retrieved_chunks)
        relevance_res = deepeval_wrapper.evaluate_answer_relevance(query, answer)
        citation_res = deepeval_wrapper.evaluate_citation_fidelity(answer, retrieved_chunks)
        llm_judge_res = deepeval_wrapper.evaluate_reference_free_llm_judge(query, answer, retrieved_chunks)

        overall_passed = (
            faithfulness_res["passed"] and
            relevance_res["passed"] and
            citation_res["passed"] and
            llm_judge_res["passed"]
        )

        return {
            "component": "generator",
            "query": query,
            "answer": answer,
            "metrics": {
                "faithfulness": faithfulness_res["score"],
                "faithfulness_reason": faithfulness_res["reason"],
                "faithfulness_passed": faithfulness_res["passed"],
                "answer_relevance": relevance_res["score"],
                "relevance_reason": relevance_res["reason"],
                "relevance_passed": relevance_res["passed"],
                "citation_fidelity": citation_res["score"],
                "has_citations": citation_res["has_explicit_citations"],
                "citation_passed": citation_res["passed"],
                "reference_free_llm_judge": llm_judge_res["score"],
                "llm_judge_reason": llm_judge_res["reason"],
                "llm_judge_passed": llm_judge_res["passed"]
            },
            "overall_passed": overall_passed
        }

    def run_component_evaluation_suite(self, golden_items: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Runs evaluation over a dataset of test cases in isolation."""
        items = golden_items or dataset_manager.load_golden_dataset()
        retriever_results = []
        generator_results = []

        for item in items:
            q = item.get("input", "")
            ans = item.get("expected_output", "") # Use expected answer as generated input for generator isolation test
            retrieved = item.get("retrieved_contexts", [])
            gt = item.get("ground_truth_contexts", [])

            retriever_res = self.evaluate_retriever(q, retrieved, gt)
            generator_res = self.evaluate_generator(q, ans, retrieved, expected_answer=ans)

            retriever_results.append(retriever_res)
            generator_results.append(generator_res)

        avg_precision = sum(r["metrics"]["precision_at_k"] for r in retriever_results) / max(1, len(retriever_results))
        avg_recall = sum(r["metrics"]["recall_at_k"] for r in retriever_results) / max(1, len(retriever_results))

        avg_faithfulness = sum(g["metrics"]["faithfulness"] for g in generator_results) / max(1, len(generator_results))
        avg_relevance = sum(g["metrics"]["answer_relevance"] for g in generator_results) / max(1, len(generator_results))
        avg_citation = sum(g["metrics"]["citation_fidelity"] for g in generator_results) / max(1, len(generator_results))
        avg_judge = sum(g["metrics"]["reference_free_llm_judge"] for g in generator_results) / max(1, len(generator_results))

        return {
            "level": "component",
            "total_items_evaluated": len(items),
            "retriever_summary": {
                "avg_precision_at_k": round(avg_precision, 4),
                "avg_recall_at_k": round(avg_recall, 4)
            },
            "generator_summary": {
                "avg_faithfulness": round(avg_faithfulness, 4),
                "avg_answer_relevance": round(avg_relevance, 4),
                "avg_citation_fidelity": round(avg_citation, 4),
                "avg_reference_free_llm_judge": round(avg_judge, 4)
            },
            "retriever_details": retriever_results,
            "generator_details": generator_results
        }

component_evaluator = ComponentEvaluator()
