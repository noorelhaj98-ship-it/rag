#!/usr/bin/env python
"""
Mini Evaluation Set Runner
Tests RAG retrieval with 25 questions and reports results.
"""
import json
import requests
from typing import Dict, List, Any

# Config
API_BASE = "http://localhost:8000"
API_KEY = "alice123"  # Use a valid API key


def load_eval_questions(path: str = "eval_questions.json") -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["evaluation_set"]


def query_rag(question: str, reranker_enabled: bool = True) -> Dict[str, Any]:
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


def query_rag_no_rerank(question: str) -> Dict[str, Any]:
    """Query the RAG API with reranker disabled."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "question": question,
        "config": {"reranker_enabled": False}  # This would need server support
    }
    
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


def check_retrieval(
    retrieved_sources: List[Dict[str, Any]],
    expected_chunk_id: str,
    expected_page: int,
    top_k: int = 6
) -> Dict[str, Any]:
    """Check if expected source is in retrieved results."""
    result = {
        "found_in_top_k": False,
        "found_at_rank": None,
        "correct_page": False,
        "retrieved_pages": [],
        "retrieved_chunk_ids": []
    }
    
    for idx, source in enumerate(retrieved_sources[:top_k], 1):
        result["retrieved_pages"].append(source.get("page_number"))
        result["retrieved_chunk_ids"].append(source.get("chunk_id"))
        
        if source.get("chunk_id") == expected_chunk_id:
            result["found_in_top_k"] = True
            result["found_at_rank"] = idx
        
        if source.get("page_number") == expected_page:
            result["correct_page"] = True
    
    return result


def run_evaluation():
    questions = load_eval_questions()
    results = []
    
    print("=" * 80)
    print("RAG MINI EVALUATION SET")
    print("=" * 80)
    print(f"Total Questions: {len(questions)}")
    print(f"API Endpoint: {API_BASE}/ask")
    print(f"Top-K: 6 (default)")
    print("=" * 80)
    print()
    
    found_in_top_k = 0
    correct_page = 0
    
    for q in questions:
        print(f"Q{q['id']:02d}: {q['question']}")
        print(f"      Expected: Page {q['expected_page']}, Chunk {q['expected_chunk_id'][:20]}...")
        
        # Query RAG
        response = query_rag(q['question'])
        sources = response.get("sources", [])
        
        # Check retrieval
        check = check_retrieval(
            sources,
            q['expected_chunk_id'],
            q['expected_page']
        )
        
        # Update counters
        if check['found_in_top_k']:
            found_in_top_k += 1
            status = f"[OK] FOUND at rank {check['found_at_rank']}"
        else:
            status = "[FAIL] NOT FOUND in top-6"
        
        if check['correct_page']:
            correct_page += 1
            page_status = "[OK]"
        else:
            page_status = "[FAIL]"
        
        print(f"      Result: {status}")
        print(f"      Correct Page: {page_status} (retrieved pages: {check['retrieved_pages'][:3]}...)")
        print()
        
        results.append({
            "question_id": q['id'],
            "question": q['question'],
            "category": q['category'],
            **check
        })
    
    # Summary
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Questions: {len(questions)}")
    print(f"Found in Top-6: {found_in_top_k} ({found_in_top_k/len(questions)*100:.1f}%)")
    print(f"Correct Page Retrieved: {correct_page} ({correct_page/len(questions)*100:.1f}%)")
    print()
    
    # Category breakdown
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = {'total': 0, 'found': 0}
        categories[cat]['total'] += 1
        if r['found_in_top_k']:
            categories[cat]['found'] += 1
    
    print("By Category:")
    for cat, stats in sorted(categories.items()):
        pct = stats['found'] / stats['total'] * 100
        print(f"  {cat:20s}: {stats['found']}/{stats['total']} ({pct:.1f}%)")
    
    print()
    print("=" * 80)
    
    # Save detailed results
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_questions": len(questions),
                "found_in_top_k": found_in_top_k,
                "found_percentage": found_in_top_k/len(questions)*100,
                "correct_page": correct_page,
                "correct_page_percentage": correct_page/len(questions)*100
            },
            "category_breakdown": categories,
            "detailed_results": results
        }, f, indent=2, ensure_ascii=False)
    
    print("Detailed results saved to: eval_results.json")


if __name__ == "__main__":
    run_evaluation()
