"""Main evaluation runner."""

from typing import Dict, List, Any

from eval_runner.config import EvalConfig, get_config
from eval_runner.rag_client import RAGClient
from eval_runner.loader import QuestionLoader
from eval_runner.evaluator import RetrieverEvaluator
from eval_runner.reporter import ConsoleReporter, JSONReporter


class EvaluationRunner:
    """Run RAG evaluation."""

    def __init__(self, config: EvalConfig = None):
        self.config = config or get_config()
        self.client = RAGClient(self.config)
        self.evaluator = RetrieverEvaluator(self.config)
        self.console = ConsoleReporter()

    def run(self) -> Dict[str, Any]:
        """Run the full evaluation.

        Returns:
            Evaluation results with metrics
        """
        # Load questions
        questions = QuestionLoader.load(self.config.questions_file)

        # Print header
        self.console.print_header("RAG MINI EVALUATION SET")
        print(f"Total Questions: {len(questions)}")
        print(f"API Endpoint: {self.config.api_base}/ask")
        print(f"Top-K: {self.config.top_k}")
        self.console.print_header("")
        print()

        # Run evaluation
        results = []
        for q in questions:
            # Query RAG
            response = self.client.query(q['question'])
            sources = response.get("sources", [])

            # Evaluate
            evaluation = self.evaluator.evaluate(
                sources,
                q['expected_chunk_id'],
                q['expected_page']
            )

            # Print result
            self.console.print_question_result(
                q['id'],
                q['question'],
                q['expected_page'],
                q['expected_chunk_id'],
                evaluation
            )

            # Store result
            results.append({
                "question_id": q['id'],
                "question": q['question'],
                "category": q['category'],
                **evaluation
            })

        # Calculate metrics
        metrics = self.evaluator.calculate_metrics(results)

        # Print summary
        self.console.print_summary(metrics)

        # Save results
        JSONReporter.save(metrics, results, self.config.results_file)

        return {
            "metrics": metrics,
            "detailed_results": results
        }


def main():
    """Main entry point."""
    runner = EvaluationRunner()
    runner.run()


if __name__ == "__main__":
    main()
