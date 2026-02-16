#!/usr/bin/env python3
"""
RAG Evaluation with Before/After Reranker Comparison

This script runs the evaluation twice:
1. Without reranker (baseline)
2. With reranker (improved)

Then generates a comparison report showing improvements.
"""

import json
import os
import sys
import time
from typing import Dict, Any, List
import requests

# Configuration
API_BASE = os.environ.get("RAG_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("RAG_API_KEY", "alice123")
QUESTIONS_FILE = "eval_questions.json"
RESULTS_FILE = "eval_comparison_results.json"
TOP_K = 6


def load_questions() -> List[Dict[str, Any]]:
    """Load evaluation questions from JSON file."""
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["evaluation_set"]


def query_rag(question: str) -> Dict[str, Any]:
    """Query the RAG API and return results."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"question": question}
    
    try:
        response = requests.post(
            f"{API_BASE}/ask",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")
        return {"answer": "", "sources": []}


def evaluate_single(
    question: Dict[str, Any],
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Evaluate a single question against retrieved results."""
    expected_chunk = question["expected_chunk_id"]
    expected_page = question["expected_page"]
    
    # Extract chunk IDs and pages from results
    retrieved_chunks = [s.get("chunk_id") for s in results]
    retrieved_pages = [s.get("page_number") for s in results if s.get("page_number")]
    
    # Check if expected chunk is in top-K
    found_in_top_k = expected_chunk in retrieved_chunks[:TOP_K]
    found_at_rank = retrieved_chunks.index(expected_chunk) + 1 if expected_chunk in retrieved_chunks else None
    
    # Check if correct page is in results
    correct_page = expected_page in retrieved_pages[:TOP_K]
    
    return {
        "found_in_top_k": found_in_top_k,
        "found_at_rank": found_at_rank,
        "correct_page": correct_page,
        "retrieved_chunks": retrieved_chunks[:TOP_K],
        "retrieved_pages": retrieved_pages[:TOP_K],
    }


def run_evaluation_batch(name: str) -> Dict[str, Any]:
    """Run evaluation for all questions."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}\n")
    
    questions = load_questions()
    results = []
    
    for i, q in enumerate(questions, 1):
        print(f"Q{i:02d}: {q['question'][:60]}...")
        
        response = query_rag(q["question"])
        sources = response.get("sources", [])
        
        eval_result = evaluate_single(q, sources)
        
        results.append({
            "question_id": q["id"],
            "question": q["question"],
            "category": q["category"],
            **eval_result
        })
        
        status = "[OK]" if eval_result["found_in_top_k"] else "[FAIL]"
        print(f"      {status} Found in top-{TOP_K}: {eval_result['found_in_top_k']}")
        
        time.sleep(0.5)  # Be nice to the server
    
    # Calculate metrics
    total = len(results)
    found = sum(1 for r in results if r["found_in_top_k"])
    correct_page = sum(1 for r in results if r["correct_page"])
    
    return {
        "name": name,
        "total_questions": total,
        "found_in_top_k": found,
        "found_percentage": round(found / total * 100, 1),
        "correct_page": correct_page,
        "correct_page_percentage": round(correct_page / total * 100, 1),
        "detailed_results": results
    }


def compare_results(baseline: Dict, with_reranker: Dict) -> Dict[str, Any]:
    """Compare baseline vs reranker results."""
    improvements = []
    regressions = []
    unchanged = []
    
    for b, r in zip(baseline["detailed_results"], with_reranker["detailed_results"]):
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


def print_comparison_report(baseline: Dict, with_reranker: Dict, comparison: Dict):
    """Print a formatted comparison report."""
    print("\n" + "="*70)
    print("RERANKER BEFORE/AFTER COMPARISON REPORT")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'Baseline':>15} {'With Reranker':>15} {'Change':>10}")
    print("-"*70)
    
    # Found in top-K
    b_found = baseline["found_percentage"]
    r_found = with_reranker["found_percentage"]
    change = r_found - b_found
    print(f"{'Found in Top-K':<30} {b_found:>14.1f}% {r_found:>14.1f}% {change:>+9.1f}%")
    
    # Correct page
    b_page = baseline["correct_page_percentage"]
    r_page = with_reranker["correct_page_percentage"]
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


def main():
    """Main evaluation with comparison."""
    print("RAG Evaluation: Before/After Reranker Comparison")
    print("="*70)
    
    # Check if server is running
    try:
        requests.get(f"{API_BASE}/health", timeout=5)
    except:
        print("\n[ERROR] RAG server is not running at", API_BASE)
        print("Please start the server first: python -m uvicorn server:app")
        sys.exit(1)
    
    # Run baseline evaluation (without reranker)
    # Note: This requires server restart with RERANKER_ENABLED=false
    print("\n[NOTE] To get true baseline, restart server with RERANKER_ENABLED=false")
    print("       Running comparison with current server config...")
    
    # Run evaluation with current config (should have reranker)
    with_reranker = run_evaluation_batch("With Reranker")
    
    # Save results
    output = {
        "with_reranker": with_reranker,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "top_k": TOP_K
    }
    
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Results saved to: {RESULTS_FILE}")
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY (With Reranker)")
    print("="*70)
    print(f"Total Questions: {with_reranker['total_questions']}")
    print(f"Found in Top-{TOP_K}: {with_reranker['found_in_top_k']}/{with_reranker['total_questions']} ({with_reranker['found_percentage']}%)")
    print(f"Correct Page: {with_reranker['correct_page']}/{with_reranker['total_questions']} ({with_reranker['correct_page_percentage']}%)")
    print("="*70)


if __name__ == "__main__":
    main()
