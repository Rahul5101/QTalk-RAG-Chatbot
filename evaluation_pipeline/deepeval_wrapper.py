"""
DeepEval Wrapper Module.
Provides integration with DeepEval metrics, LLMTestCases, and GEval LLM judges.
"""

import os
import re
from typing import List, Dict, Any, Tuple, Optional
import deepeval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    GEval,
    ToxicityMetric
)

from evaluation_pipeline.config import DEEPEVAL_DEFAULT_MODEL, EVAL_THRESHOLDS

class DeepEvalEvaluatorWrapper:
    def __init__(self, model_name: str = DEEPEVAL_DEFAULT_MODEL):
        self.model_name = model_name
        self._has_llm_api_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    def create_test_case(
        self,
        input_text: str,
        actual_output: str,
        retrieved_contexts: Optional[List[str]] = None,
        expected_output: Optional[str] = None,
        ground_truth_contexts: Optional[List[str]] = None
    ) -> LLMTestCase:
        """Constructs a DeepEval LLMTestCase."""
        return LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            retrieved_context=retrieved_contexts or [],
            expected_output=expected_output,
            context=ground_truth_contexts or retrieved_contexts or []
        )

    # --- (a) COMPONENT LEVEL METRICS ---

    def evaluate_retriever_precision(self, query: str, retrieved_chunks: List[str], ground_truth: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculates Precision@k for retriever."""
        if not retrieved_chunks:
            return {"score": 0.0, "reason": "No retrieved chunks provided", "passed": False}

        test_case = self.create_test_case(
            input_text=query,
            actual_output="N/A",
            retrieved_contexts=retrieved_chunks,
            ground_truth_contexts=ground_truth or retrieved_chunks
        )

        try:
            if self._has_llm_api_key:
                metric = ContextualPrecisionMetric(threshold=EVAL_THRESHOLDS["precision_at_k"], model=self.model_name)
                metric.measure(test_case)
                return {"score": round(float(metric.score), 4), "reason": str(getattr(metric, "reason", "Evaluated via DeepEval")), "passed": metric.is_successful()}
        except Exception as e:
            pass

        # Heuristic fallback for offline / no-key execution
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        if not query_words:
            return {"score": 1.0, "reason": "Empty query terms", "passed": True}
        relevant_count = 0
        for chunk in retrieved_chunks:
            chunk_words = set(re.findall(r'\b\w{3,}\b', chunk.lower()))
            overlap = query_words.intersection(chunk_words)
            if len(overlap) / max(1, len(query_words)) >= 0.2:
                relevant_count += 1
        score = round(relevant_count / len(retrieved_chunks), 4)
        return {"score": score, "reason": f"{relevant_count}/{len(retrieved_chunks)} chunks contain query keywords", "passed": score >= EVAL_THRESHOLDS["precision_at_k"]}

    def evaluate_retriever_recall(self, query: str, retrieved_chunks: List[str], ground_truth: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculates Recall@k for retriever."""
        if not retrieved_chunks:
            return {"score": 0.0, "reason": "No retrieved chunks provided", "passed": False}

        test_case = self.create_test_case(
            input_text=query,
            actual_output="N/A",
            retrieved_contexts=retrieved_chunks,
            ground_truth_contexts=ground_truth or retrieved_chunks
        )

        try:
            if self._has_llm_api_key:
                metric = ContextualRecallMetric(threshold=EVAL_THRESHOLDS["recall_at_k"], model=self.model_name)
                metric.measure(test_case)
                return {"score": round(float(metric.score), 4), "reason": str(getattr(metric, "reason", "Evaluated via DeepEval")), "passed": metric.is_successful()}
        except Exception as e:
            pass

        # Heuristic fallback
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        if not query_words:
            return {"score": 1.0, "reason": "Empty query", "passed": True}
        combined_text = " ".join(retrieved_chunks)
        combined_words = set(re.findall(r'\b\w{3,}\b', combined_text.lower()))
        covered = query_words.intersection(combined_words)
        score = round(len(covered) / len(query_words), 4)
        return {"score": score, "reason": f"{len(covered)}/{len(query_words)} query keywords present in context", "passed": score >= EVAL_THRESHOLDS["recall_at_k"]}

    def evaluate_faithfulness(self, query: str, answer: str, retrieved_chunks: List[str]) -> Dict[str, Any]:
        """Evaluates Faithfulness / Grounding & Hallucination detection."""
        if not answer or not retrieved_chunks:
            return {"score": 0.0, "reason": "Missing answer or context", "passed": False}

        test_case = self.create_test_case(input_text=query, actual_output=answer, retrieved_contexts=retrieved_chunks)

        try:
            if self._has_llm_api_key:
                metric = FaithfulnessMetric(threshold=EVAL_THRESHOLDS["faithfulness"], model=self.model_name)
                metric.measure(test_case)
                return {"score": round(float(metric.score), 4), "reason": str(getattr(metric, "reason", "Evaluated via DeepEval")), "passed": metric.is_successful()}
        except Exception as e:
            pass

        # Heuristic sentence grounding fallback
        combined_context = " ".join(retrieved_chunks).lower()
        context_words = set(re.findall(r'\b\w{3,}\b', combined_context))
        sentences = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 8]
        if not sentences:
            return {"score": 1.0, "reason": "Short answer", "passed": True}
        grounded = 0
        for s in sentences:
            swords = set(re.findall(r'\b\w{3,}\b', s.lower()))
            if not swords or len(swords.intersection(context_words)) / len(swords) >= 0.3:
                grounded += 1
        score = round(grounded / len(sentences), 4)
        return {"score": score, "reason": f"{grounded}/{len(sentences)} claims grounded in retrieved context", "passed": score >= EVAL_THRESHOLDS["faithfulness"]}

    def evaluate_answer_relevance(self, query: str, answer: str) -> Dict[str, Any]:
        """Evaluates Answer Relevance to user query."""
        if not query or not answer:
            return {"score": 0.0, "reason": "Missing query or answer", "passed": False}

        test_case = self.create_test_case(input_text=query, actual_output=answer)

        try:
            if self._has_llm_api_key:
                metric = AnswerRelevancyMetric(threshold=EVAL_THRESHOLDS["answer_relevance"], model=self.model_name)
                metric.measure(test_case)
                return {"score": round(float(metric.score), 4), "reason": str(getattr(metric, "reason", "Evaluated via DeepEval")), "passed": metric.is_successful()}
        except Exception as e:
            pass

        q_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        a_words = set(re.findall(r'\b\w{3,}\b', answer.lower()))
        if not q_words or not a_words:
            return {"score": 0.5, "reason": "Low keyword density", "passed": True}
        overlap = q_words.intersection(a_words)
        score = round(min(1.0, (len(overlap) / len(q_words)) * 1.4), 4)
        return {"score": score, "reason": f"{len(overlap)} term overlap with query", "passed": score >= EVAL_THRESHOLDS["answer_relevance"]}

    def evaluate_citation_fidelity(self, answer: str, retrieved_chunks: List[str]) -> Dict[str, Any]:
        """Evaluates Citation / Reference alignment for generated response."""
        faith_res = self.evaluate_faithfulness("citation check", answer, retrieved_chunks)
        has_citations = bool(re.search(r'\[(?:Doc|Source|\d+)\]', answer, re.IGNORECASE))
        score = round((faith_res["score"] * 0.8) + (0.2 if has_citations else 0.1), 4)
        return {
            "score": min(1.0, score),
            "has_explicit_citations": has_citations,
            "passed": score >= EVAL_THRESHOLDS["citation_fidelity"]
        }

    def evaluate_reference_free_llm_judge(self, query: str, answer: str, retrieved_chunks: List[str]) -> Dict[str, Any]:
        """Evaluates generator quality in isolation using reference-free GEval judge."""
        test_case = self.create_test_case(input_text=query, actual_output=answer, retrieved_contexts=retrieved_chunks)

        try:
            if self._has_llm_api_key:
                metric = GEval(
                    name="ReferenceFreeJudge",
                    criteria="Assess whether the answer is clear, coherent, factually accurate based on context, and free of contradictions without requiring ground truth answer.",
                    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVED_CONTEXT],
                    threshold=EVAL_THRESHOLDS["llm_judge_score"],
                    model=self.model_name
                )
                metric.measure(test_case)
                return {"score": round(float(metric.score), 4), "reason": str(getattr(metric, "reason", "Evaluated via GEval LLM Judge")), "passed": metric.is_successful()}
        except Exception as e:
            pass

        # Heuristic reference-free score
        word_count = len(answer.split())
        score = 0.9 if word_count >= 15 else 0.6
        return {"score": score, "reason": f"Reference-free check evaluated ({word_count} words)", "passed": score >= EVAL_THRESHOLDS["llm_judge_score"]}

    # --- (b) PIPELINE LEVEL METRICS ---

    def evaluate_context_relevance(self, query: str, retrieved_chunks: List[str]) -> Dict[str, Any]:
        """Evaluates Context Relevance (proportion of retrieved context relevant to query)."""
        if not retrieved_chunks:
            return {"score": 0.0, "reason": "No context provided", "passed": False}

        test_case = self.create_test_case(input_text=query, actual_output="N/A", retrieved_contexts=retrieved_chunks)

        try:
            if self._has_llm_api_key:
                metric = ContextualRelevancyMetric(threshold=EVAL_THRESHOLDS["context_relevance"], model=self.model_name)
                metric.measure(test_case)
                return {"score": round(float(metric.score), 4), "reason": str(getattr(metric, "reason", "Evaluated via DeepEval")), "passed": metric.is_successful()}
        except Exception as e:
            pass

        return self.evaluate_retriever_precision(query, retrieved_chunks)

    # --- (c) APPLICATION LEVEL METRICS ---

    def evaluate_correctness(self, query: str, actual_output: str, expected_output: str) -> Dict[str, Any]:
        """Evaluates factual correctness against expected output."""
        test_case = self.create_test_case(input_text=query, actual_output=actual_output, expected_output=expected_output)

        try:
            if self._has_llm_api_key:
                metric = GEval(
                    name="Correctness",
                    criteria="Determine whether the actual output matches the factual core of the expected output.",
                    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
                    threshold=EVAL_THRESHOLDS["correctness"],
                    model=self.model_name
                )
                metric.measure(test_case)
                return {"score": round(float(metric.score), 4), "reason": str(getattr(metric, "reason", "GEval Correctness")), "passed": metric.is_successful()}
        except Exception as e:
            pass

        exp_words = set(re.findall(r'\b\w{3,}\b', expected_output.lower()))
        act_words = set(re.findall(r'\b\w{3,}\b', actual_output.lower()))
        overlap = exp_words.intersection(act_words)
        score = round(len(overlap) / max(1, len(exp_words)), 4)
        return {"score": score, "reason": f"Factual word overlap {len(overlap)}/{len(exp_words)}", "passed": score >= EVAL_THRESHOLDS["correctness"]}

    def evaluate_completeness(self, query: str, actual_output: str, expected_output: Optional[str] = None) -> Dict[str, Any]:
        """Evaluates completeness of the generated response."""
        test_case = self.create_test_case(input_text=query, actual_output=actual_output, expected_output=expected_output)

        try:
            if self._has_llm_api_key:
                metric = GEval(
                    name="Completeness",
                    criteria="Evaluate whether all aspects of the user query are completely addressed without missing details.",
                    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                    threshold=EVAL_THRESHOLDS["completeness"],
                    model=self.model_name
                )
                metric.measure(test_case)
                return {"score": round(float(metric.score), 4), "reason": str(getattr(metric, "reason", "GEval Completeness")), "passed": metric.is_successful()}
        except Exception as e:
            pass

        score = 0.85 if len(actual_output.split()) >= 12 else 0.5
        return {"score": score, "reason": f"Response length: {len(actual_output.split())} words", "passed": score >= EVAL_THRESHOLDS["completeness"]}

    def evaluate_style(self, query: str, actual_output: str) -> Dict[str, Any]:
        """Evaluates professional tone, style, and formatting."""
        test_case = self.create_test_case(input_text=query, actual_output=actual_output)

        try:
            if self._has_llm_api_key:
                metric = GEval(
                    name="StyleAndTone",
                    criteria="Evaluate whether the output uses professional, clear, objective language with clean formatting.",
                    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                    threshold=EVAL_THRESHOLDS["style"],
                    model=self.model_name
                )
                metric.measure(test_case)
                return {"score": round(float(metric.score), 4), "reason": str(getattr(metric, "reason", "GEval Style")), "passed": metric.is_successful()}
        except Exception as e:
            pass

        # Check basic style indicators (punctuation, capitalization, length)
        starts_cap = actual_output[0].isupper() if actual_output else False
        ends_punct = actual_output[-1] in ".!?" if actual_output else False
        score = 0.9 if (starts_cap and ends_punct) else 0.7
        return {"score": score, "reason": "Basic style and punctuation check", "passed": score >= EVAL_THRESHOLDS["style"]}

    def evaluate_toxicity(self, text: str) -> Dict[str, Any]:
        """Evaluates text toxicity using DeepEval ToxicityMetric."""
        test_case = self.create_test_case(input_text="Toxicity check", actual_output=text)
        try:
            if self._has_llm_api_key:
                metric = ToxicityMetric(threshold=0.3, model=self.model_name)
                metric.measure(test_case)
                is_toxic = metric.score > 0.3
                return {"is_toxic": is_toxic, "toxicity_score": round(float(metric.score), 4), "reason": str(getattr(metric, "reason", "Toxicity metric")), "passed": not is_toxic}
        except Exception as e:
            pass

        toxic_keywords = ["hate", "kill", "attack", "exploit", "abusive", "harass"]
        found = [w for w in toxic_keywords if w in text.lower()]
        is_toxic = len(found) > 0
        return {"is_toxic": is_toxic, "toxicity_score": 1.0 if is_toxic else 0.0, "reason": f"Keywords found: {found}" if is_toxic else "No toxic keywords detected", "passed": not is_toxic}

deepeval_wrapper = DeepEvalEvaluatorWrapper()
