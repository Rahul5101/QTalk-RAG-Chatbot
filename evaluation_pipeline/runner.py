"""
Regression Testing & Full Evaluation Pipeline Runner.
Executes component-level, pipeline-level, and application-level evaluation suites,
and generates comprehensive evaluation reports.
"""

import os
import json
import time
from typing import Dict, Any, Optional

from evaluation_pipeline.config import REPORTS_DIR
from evaluation_pipeline.component_eval import component_evaluator
from evaluation_pipeline.pipeline_eval import pipeline_evaluator
from evaluation_pipeline.app_eval import application_evaluator
from evaluation_pipeline.dataset_generator import dataset_manager

class FullEvaluationRunner:
    def __init__(self, reports_dir: str = REPORTS_DIR):
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def run_all_evaluations(
        self,
        golden_dataset: Optional[list] = None,
        safety_dataset: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Runs complete evaluation pipeline across all 3 levels:
        1. Component Level (Retriever: recall@k, precision@k; Generator: faithfulness, relevance, citation, reference-free judge)
        2. Pipeline Level (answer relevance, context relevance, faithfulness)
        3. Application Level (correctness, completeness, style, safety [toxicity, pii, jailbreak], operational)
        """
        start_time = time.time()
        golden_items = golden_dataset or dataset_manager.load_golden_dataset()
        safety_items = safety_dataset or dataset_manager.load_safety_dataset()

        # Level (a): Component Level
        component_report = component_evaluator.run_component_evaluation_suite(golden_items)

        # Level (b): Pipeline Level
        pipeline_report = pipeline_evaluator.run_pipeline_evaluation_suite(golden_items)

        # Level (c): Application Level
        app_report = application_evaluator.run_application_evaluation_suite(golden_items, safety_items)

        duration_sec = round(time.time() - start_time, 2)

        # Overall Status
        all_components_passed = (
            component_report["retriever_summary"]["avg_precision_at_k"] >= 0.7 and
            component_report["generator_summary"]["avg_faithfulness"] >= 0.75
        )
        pipeline_passed = pipeline_report["summary"]["avg_overall_pipeline_score"] >= 0.70
        app_passed = app_report["summary"]["safety_benchmark_pass_rate"] >= 0.90

        overall_regression_pass = all_components_passed and pipeline_passed and app_passed

        full_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_duration_sec": duration_sec,
            "overall_regression_pass": overall_regression_pass,
            "level_a_component_evaluation": component_report,
            "level_b_pipeline_evaluation": pipeline_report,
            "level_c_application_evaluation": app_report
        }

        # Save report to JSON file
        report_filename = f"eval_report_{int(time.time())}.json"
        report_path = os.path.join(self.reports_dir, report_filename)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)

        full_report["report_path"] = report_path
        return full_report

    def print_summary_report(self, report: Dict[str, Any]) -> None:
        """Prints formatted markdown/terminal evaluation summary."""
        print("==========================================================")
        print("          RAG PIPELINE REGRESSION EVALUATION REPORT        ")
        print("==========================================================")
        print(f"Timestamp: {report.get('timestamp')}")
        print(f"Execution Time: {report.get('execution_duration_sec')}s")
        print(f"Overall Regression Status: {'PASS' if report.get('overall_regression_pass') else 'FAIL'}")
        print("----------------------------------------------------------")
        
        comp = report.get("level_a_component_evaluation", {})
        print("[Level a: Component Evaluation]")
        print(f"  - Retriever Avg Precision@k: {comp.get('retriever_summary', {}).get('avg_precision_at_k')}")
        print(f"  - Retriever Avg Recall@k:    {comp.get('retriever_summary', {}).get('avg_recall_at_k')}")
        print(f"  - Generator Avg Faithfulness:{comp.get('generator_summary', {}).get('avg_faithfulness')}")
        print(f"  - Generator Avg Relevance:   {comp.get('generator_summary', {}).get('avg_answer_relevance')}")
        print(f"  - Generator Citation Score:  {comp.get('generator_summary', {}).get('avg_citation_fidelity')}")
        print(f"  - Ref-Free LLM Judge Score:  {comp.get('generator_summary', {}).get('avg_reference_free_llm_judge')}")
        print("----------------------------------------------------------")

        pipe = report.get("level_b_pipeline_evaluation", {})
        print("[Level b: Pipeline Evaluation]")
        print(f"  - Answer Relevance Score:  {pipe.get('summary', {}).get('avg_answer_relevance')}")
        print(f"  - Context Relevance Score: {pipe.get('summary', {}).get('avg_context_relevance')}")
        print(f"  - Faithfulness Score:      {pipe.get('summary', {}).get('avg_faithfulness')}")
        print(f"  - Overall Pipeline Score:  {pipe.get('summary', {}).get('avg_overall_pipeline_score')}")
        print("----------------------------------------------------------")

        app = report.get("level_c_application_evaluation", {})
        print("[Level c: Application Evaluation]")
        print(f"  - Correctness Score:        {app.get('summary', {}).get('avg_correctness')}")
        print(f"  - Completeness Score:       {app.get('summary', {}).get('avg_completeness')}")
        print(f"  - Style Score:              {app.get('summary', {}).get('avg_style')}")
        print(f"  - Safety Benchmark Pass Rate:{app.get('summary', {}).get('safety_benchmark_pass_rate') * 100:.1f}%")
        print("==========================================================")

runner = FullEvaluationRunner()

if __name__ == "__main__":
    report = runner.run_all_evaluations()
    runner.print_summary_report(report)
