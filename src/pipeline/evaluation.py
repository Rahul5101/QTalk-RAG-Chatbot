"""
RAG Evaluation Bridge Pipeline.
Integrates with the main DeepEval evaluation pipeline in `/evaluation_pipeline`.
"""

from typing import List, Dict, Any, Tuple
from evaluation_pipeline import (
    component_evaluator,
    pipeline_evaluator,
    application_evaluator
)

class RAGEvaluator:
    """
    RAG Evaluator Bridge connecting src/pipeline to the core DeepEval framework in /evaluation_pipeline.
    """
    def __init__(self):
        self.component_evaluator = component_evaluator
        self.pipeline_evaluator = pipeline_evaluator
        self.application_evaluator = application_evaluator

    def evaluate_context_precision(self, query: str, retrieved_chunks: List[str]) -> float:
        res = self.component_evaluator.evaluate_retriever(query, retrieved_chunks)
        return res["metrics"]["precision_at_k"]

    def evaluate_context_recall(self, query: str, retrieved_chunks: List[str]) -> float:
        res = self.component_evaluator.evaluate_retriever(query, retrieved_chunks)
        return res["metrics"]["recall_at_k"]

    def evaluate_answer_relevance(self, query: str, answer: str) -> float:
        res = self.pipeline_evaluator.evaluate_pipeline_response(query, answer, [])
        return res["metrics"]["answer_relevance"]

    def evaluate_faithfulness(self, answer: str, retrieved_chunks: List[str]) -> Tuple[float, bool]:
        res = self.pipeline_evaluator.evaluate_pipeline_response("eval", answer, retrieved_chunks)
        faith_score = res["metrics"]["faithfulness"]
        is_hallucination = faith_score < 0.5
        return faith_score, is_hallucination

    def run_full_evaluation(self, query: str, answer: str, retrieved_chunks: List[str]) -> Dict[str, Any]:
        """
        Runs comprehensive DeepEval evaluation across Component, Pipeline, and Application levels.
        """
        # Pipeline level evaluation (Answer relevance, Context relevance, Faithfulness)
        pipe_res = self.pipeline_evaluator.evaluate_pipeline_response(query, answer, retrieved_chunks)

        # Component level evaluation (Precision@k, Recall@k, Citation, LLM Judge)
        comp_retriever = self.component_evaluator.evaluate_retriever(query, retrieved_chunks)
        comp_generator = self.component_evaluator.evaluate_generator(query, answer, retrieved_chunks)

        # Application level safety & quality evaluation (Correctness, Completeness, Style, Safety)
        app_res = self.application_evaluator.evaluate_application(query, answer)

        return {
            "context_precision": comp_retriever["metrics"]["precision_at_k"],
            "context_recall": comp_retriever["metrics"]["recall_at_k"],
            "answer_relevance": pipe_res["metrics"]["answer_relevance"],
            "context_relevance": pipe_res["metrics"]["context_relevance"],
            "faithfulness": pipe_res["metrics"]["faithfulness"],
            "citation_fidelity": comp_generator["metrics"]["citation_fidelity"],
            "reference_free_llm_judge": comp_generator["metrics"]["reference_free_llm_judge"],
            "correctness": app_res["metrics"]["correctness"],
            "completeness": app_res["metrics"]["completeness"],
            "style": app_res["metrics"]["style"],
            "safety": app_res["metrics"]["safety"],
            "is_hallucination": pipe_res["metrics"]["faithfulness"] < 0.5,
            "overall_score": pipe_res["overall_pipeline_score"]
        }

rag_evaluator = RAGEvaluator()

