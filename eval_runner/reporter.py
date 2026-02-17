"""Evaluation reporting utilities."""

import json
from typing import Dict, List, Any


class ConsoleReporter:
    """Print evaluation results to console."""

    @staticmethod
    def print_header(title: str, width: int = 80) -> None:
        """Print a formatted header."""
        print("=" * width)
        print(title)
        print("=" * width)

    @staticmethod
    def print_question_result(
        question_id: int,
        question: str,
        expected_page: int,
        expected_chunk: str,
        evaluation: Dict[str, Any]
    ) -> None:
        """Print result for a single question."""
        print(f"Q{question_id:02d}: {question}")
        print(f"      Expected: Page {expected_page}, Chunk {expected_chunk[:20]}...")

        if evaluation['found_in_top_k']:
            status = f"[OK] FOUND at rank {evaluation['found_at_rank']}"
        else:
            status = "[FAIL] NOT FOUND in top-6"

        if evaluation['correct_page']:
            page_status = "[OK]"
        else:
            page_status = "[FAIL]"

        print(f"      Result: {status}")
        print(f"      Correct Page: {page_status} (retrieved pages: {evaluation['retrieved_pages'][:3]}...)")
        print()

    @staticmethod
    def print_summary(metrics: Dict[str, Any]) -> None:
        """Print evaluation summary."""
        ConsoleReporter.print_header("EVALUATION SUMMARY")

        print(f"Total Questions: {metrics['total_questions']}")
        print(f"Found in Top-6: {metrics['found_in_top_k']} ({metrics['found_percentage']}%)")
        print(f"Correct Page Retrieved: {metrics['correct_page']} ({metrics['correct_page_percentage']}%)")
        print()

        print("By Category:")
        for cat, stats in sorted(metrics['category_breakdown'].items()):
            pct = stats['found'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"  {cat:20s}: {stats['found']}/{stats['total']} ({pct:.1f}%)")

        print()
        ConsoleReporter.print_header("")


class JSONReporter:
    """Save evaluation results to JSON file."""

    @staticmethod
    def save(
        metrics: Dict[str, Any],
        detailed_results: List[Dict[str, Any]],
        output_path: str
    ) -> None:
        """Save results to JSON file.

        Args:
            metrics: Aggregated metrics
            detailed_results: Detailed per-question results
            output_path: Output file path
        """
        output = {
            "summary": metrics,
            "detailed_results": detailed_results
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Detailed results saved to: {output_path}")
