"""RAG evaluation comparison utilities."""

import time
from typing import Dict, List, Any

from eval_runner.config import EvalConfig, get_config
from eval_runner.runner import EvaluationRunner
from eval_runner.reporter import ConsoleReporter


class ComparisonResult:
    """Result of a comparison between two evaluation runs."""

    def __init__(
        self,
        baseline: Dict[str, Any],
        with_reranker: Dict[str, Any],
        comparison: Dict[str, Any]
    ):
        self.baseline = baseline
        self.with_reranker = with_reranker
        self.comparison = comparison


class ComparisonRunner:
    """Run before/after reranker comparison."""

    def __init__(self, config: EvalConfig = None):
        self.config = config or get_config()
        self.console = ConsoleReporter()

    def compare_results(
        self,
        baseline: Dict[str, Any],
        with_reranker: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare baseline vs reranker results.

        Args:
            baseline: Baseline evaluation results
            with_reranker: Evaluation results with reranker

        Returns:
            Comparison dictionary with improvements and regressions
        """
        improvements = []
        regressions = []
        unchanged = []

        baseline_results = baseline.get("detailed_results", [])
        reranker_results = with_reranker.get("detailed_results", [])

        for b, r in zip(baseline_results, reranker_results):
            b_found = b["found_in_top_k"]
            r_found = r["found_in_top_k"]

            if not b_found and r_found:
                improvements.append({
                    "question_id": b["question_id"],
                    "question": b["question"],
                    "category": b["category"],
                    "rank_before": b["found_at_rank"],
                    "rank_after": r["found_at_rank"]
                })
            elif b_found and not r_found:
                regressions.append({
                    "question_id": b["question_id"],
                    "question": b["question"],
                    "category": b["category"]
                })
            else:
                unchanged.append({
                    "question_id": b["question_id"],
                    "question": b["question"],
                    "found": b_found,
                    "category": b["category"]
                })

        return {
            "improvements": improvements,
            "regressions": regressions,
            "unchanged": unchanged,
            "improvement_count": len(improvements),
            "regression_count": len(regressions),
            "unchanged_count": len(unchanged)
        }

    def print_comparison_report(
        self,
        baseline: Dict[str, Any],
        with_reranker: Dict[str, Any],
        comparison: Dict[str, Any]
    ) -> None:
        """Print a formatted comparison report.

        Args:
            baseline: Baseline results
            with_reranker: Results with reranker
            comparison: Comparison dictionary
        """
        print("\n" + "="*70)
        print("RERANKER BEFORE/AFTER COMPARISON REPORT")
        print("="*70)

        print(f"\n{'Metric':<30} {'Baseline':>15} {'With Reranker':>15} {'Change':>10}")
        print("-"*70)

        # Found in top-K
        b_found = baseline.get("found_percentage", 0)
        r_found = with_reranker.get("found_percentage", 0)
        change = r_found - b_found
        print(f"{'Found in Top-K':<30} {b_found:>14.1f}% {r_found:>14.1f}% {change:>+9.1f}%")

        # Correct page
        b_page = baseline.get("correct_page_percentage", 0)
        r_page = with_reranker.get("correct_page_percentage", 0)
        change_page = r_page - b_page
        print(f"{'Correct Page':<30} {b_page:>14.1f}% {r_page:>14.1f}% {change_page:>+9.1f}%")

        print("\n" + "="*70)
        print("DETAILED CHANGES")
        print("="*70)

        print(f"\n[+] Improvements (found after reranking): {comparison['improvement_count']}")
        for imp in comparison["improvements"]:
            print(f"   Q{imp['question_id']:02d}: {imp['question'][:50]}...")
            print(f"        Rank: {imp['rank_before']} → {imp['rank_after']} (Category: {imp['category']})")

        print(f"\n[-] Regressions (lost after reranking): {comparison['regression_count']}")
        for reg in comparison["regressions"]:
            print(f"   Q{reg['question_id']:02d}: {reg['question'][:50]}...")

        print(f"\n[=] Unchanged: {comparison['unchanged_count']}")

        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        net_improvement = comparison['improvement_count'] - comparison['regression_count']
        if net_improvement > 0:
            print(f"[+] Net improvement: +{net_improvement} questions")
        elif net_improvement < 0:
            print(f"[-] Net regression: {net_improvement} questions")
        else:
            print("[=] No net change")
        print("="*70)

    def run_single(self, name: str) -> Dict[str, Any]:
        """Run evaluation for all questions.

        Args:
            name: Name of this evaluation run

        Returns:
            Evaluation results
        """
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}\n")

        runner = EvaluationRunner(self.config)
        return runner.run()

    def run_comparison(self) -> ComparisonResult:
        """Run full comparison between baseline and with reranker.

        Returns:
            ComparisonResult with all data
        """
        # Run with current config (assumed to have reranker)
        with_reranker = self.run_single("With Reranker")

        # For true comparison, baseline would need separate server config
        # For now, we return the single run result
        comparison = self.compare_results(with_reranker["metrics"], with_reranker["metrics"])

        return ComparisonResult(
            baseline=with_reranker["metrics"],
            with_reranker=with_reranker["metrics"],
            comparison=comparison
        )


def main():
    """Main entry point for comparison."""
    import json
    import sys

    runner = ComparisonRunner()

    # Check if server is running
    from eval_runner.rag_client import RAGClient
    client = RAGClient(runner.config)
    if not client.health_check():
        print("\n[ERROR] RAG server is not running at", runner.config.api_base)
        print("Please start the server first: python -m uvicorn server:app")
        sys.exit(1)

    print("RAG Evaluation: Before/After Reranker Comparison")
    print("="*70)

    # Run comparison
    result = runner.run_comparison()

    # Save results
    output = {
        "with_reranker": result.with_reranker,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "top_k": runner.config.top_k
    }

    results_file = "eval_comparison_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results saved to: {results_file}")

    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY (With Reranker)")
    print("="*70)
    print(f"Total Questions: {result.with_reranker['total_questions']}")
    print(f"Found in Top-{runner.config.top_k}: {result.with_reranker['found_in_top_k']}/{result.with_reranker['total_questions']} ({result.with_reranker['found_percentage']}%)")
    print(f"Correct Page: {result.with_reranker['correct_page']}/{result.with_reranker['total_questions']} ({result.with_reranker['correct_page_percentage']}%)")
    print("="*70)


if __name__ == "__main__":
    main()
