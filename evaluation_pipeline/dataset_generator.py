"""
Dataset Generator & Manager for Evaluation Pipeline.
Handles loading, saving, creating, and validating golden datasets and safety test cases.
"""

import os
import json
from typing import List, Dict, Any, Optional

from evaluation_pipeline.config import GOLDEN_DATASET_PATH, SAFETY_DATASET_PATH, DATASETS_DIR

class GoldenDatasetManager:
    def __init__(self, golden_path: str = GOLDEN_DATASET_PATH, safety_path: str = SAFETY_DATASET_PATH):
        self.golden_path = golden_path
        self.safety_path = safety_path
        os.makedirs(DATASETS_DIR, exist_ok=True)

    def load_golden_dataset(self) -> List[Dict[str, Any]]:
        """Loads golden dataset items from file."""
        if not os.path.exists(self.golden_path):
            return []
        with open(self.golden_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_golden_dataset(self, data: List[Dict[str, Any]]) -> None:
        """Saves golden dataset items to file."""
        with open(self.golden_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_golden_item(
        self,
        item_id: str,
        query: str,
        expected_output: str,
        retrieved_contexts: List[str],
        ground_truth_contexts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Adds or updates a golden dataset item."""
        dataset = self.load_golden_dataset()
        item = {
            "id": item_id,
            "input": query,
            "expected_output": expected_output,
            "retrieved_contexts": retrieved_contexts,
            "ground_truth_contexts": ground_truth_contexts or retrieved_contexts
        }
        # Update existing or append
        dataset = [d for d in dataset if d.get("id") != item_id]
        dataset.append(item)
        self.save_golden_dataset(dataset)
        return item

    def load_safety_dataset(self) -> List[Dict[str, Any]]:
        """Loads safety test cases from file."""
        if not os.path.exists(self.safety_path):
            return []
        with open(self.safety_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_synthetic_qna_pairs(self, text_chunks: List[str]) -> List[Dict[str, Any]]:
        """
        Generates synthetic QnA pairs from text chunks for isolated generator evaluation.
        Can be enhanced with LLM generation when API credentials are live.
        """
        synthetic_items = []
        for idx, chunk in enumerate(text_chunks):
            # Formulate simple synthetic question/answer pair from chunk
            first_sentence = chunk.split('.')[0] if '.' in chunk else chunk
            synthetic_items.append({
                "id": f"synth_{idx+1:03d}",
                "input": f"What does the following document state about: {first_sentence[:40]}...?",
                "expected_output": chunk[:200],
                "retrieved_contexts": [chunk],
                "ground_truth_contexts": [chunk]
            })
        return synthetic_items

dataset_manager = GoldenDatasetManager()
