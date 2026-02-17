"""RAG evaluation logic."""

from typing import Dict, List, Any

from eval_runner.config import EvalConfig


class RetrieverEvaluator:
    """Evaluate RAG retrieval performance."""

    def __init__(self, config: EvalConfig = None):
        self.config = config or EvalConfig()

    def evaluate(
        self,
        retrieved_sources: List[Dict[str, Any]],
        expected_chunk_id: str,
        expected_page: int
    ) -> Dict[str, Any]:
        """Check if expected source is in retrieved results.

        Args:
            retrieved_sources: List of retrieved sources
            expected_chunk_id: Expected chunk ID
            expected_page: Expected page number

        Returns:
            Evaluation result dictionary
        """
        result = {
            "found_in_top_k": False,
            "found_at_rank": None,
            "correct_page": False,
            "retrieved_pages": [],
            "retrieved_chunk_ids": []
        }

        top_k = self.config.top_k

        for idx, source in enumerate(retrieved_sources[:top_k], 1):
            result["retrieved_pages"].append(source.get("page_number"))
            result["retrieved_chunk_ids"].append(source.get("chunk_id"))

            if source.get("chunk_id") == expected_chunk_id:
                result["found_in_top_k"] = True
                result["found_at_rank"] = idx

            if source.get("page_number") == expected_page:
                result["correct_page"] = True

        return result

    def calculate_metrics(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate aggregate metrics from evaluation results.

        Args:
            results: List of individual evaluation results

        Returns:
            Aggregated metrics dictionary
        """
        total = len(results)
        found = sum(1 for r in results if r["found_in_top_k"])
        correct_page = sum(1 for r in results if r["correct_page"])

        # Category breakdown
        categories = {}
        for r in results:
            cat = r.get("category", "unknown")
            if cat not in categories:
                categories[cat] = {"total": 0, "found": 0}
            categories[cat]["total"] += 1
            if r["found_in_top_k"]:
                categories[cat]["found"] += 1

        return {
            "total_questions": total,
            "found_in_top_k": found,
            "found_percentage": round(found / total * 100, 1) if total > 0 else 0,
            "correct_page": correct_page,
            "correct_page_percentage": round(correct_page / total * 100, 1) if total > 0 else 0,
            "category_breakdown": categories
        }
