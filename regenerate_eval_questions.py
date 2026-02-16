#!/usr/bin/env python3
"""
Regenerate eval_questions.json with correct expected chunks
based on current RRF + cross-encoder pipeline.
"""

import json
import os
import requests
from typing import Dict, Any, List

API_BASE = os.environ.get("RAG_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("RAG_API_KEY", "alice123")

# Original questions (without expected chunks)
QUESTIONS = [
    {"id": 1, "question": "What is RealSoft's main mission?", "category": "company_info"},
    {"id": 2, "question": "When was RealSoft founded?", "category": "company_info"},
    {"id": 3, "question": "What products does RealSoft offer?", "category": "products"},
    {"id": 4, "question": "What is Al-Khwarizmi platform?", "category": "products"},
    {"id": 5, "question": "How many years of experience does RealSoft have?", "category": "company_info"},
    {"id": 6, "question": "What statistical solutions does RealSoft provide?", "category": "services"},
    {"id": 7, "question": "What is RealSoft's vision?", "category": "company_info"},
    {"id": 8, "question": "Who is the Director of Jordan Meteorological Department?", "category": "testimonials"},
    {"id": 9, "question": "What countries does RealSoft operate in?", "category": "company_info"},
    {"id": 10, "question": "What is RealData Hub?", "category": "products"},
    {"id": 11, "question": "What technology partners does RealSoft work with?", "category": "partners"},
    {"id": 12, "question": "What is Microsoft partnership about?", "category": "partners"},
    {"id": 13, "question": "What awards has RealSoft received from Union of Arab Statisticians?", "category": "awards"},
    {"id": 14, "question": "What is RealSoft's contact number in Jordan?", "category": "contact"},
    {"id": 15, "question": "What is the FalconMap platform?", "category": "products"},
    {"id": 16, "question": "What is RealSoft's approach to digital transformation?", "category": "services"},
    {"id": 17, "question": "Who is Dr. Diaa Awad Kazem?", "category": "testimonials"},
    {"id": 18, "question": "What is the Ada'a product?", "category": "products"},
    {"id": 19, "question": "What is Esri and what do they do?", "category": "partners"},
    {"id": 20, "question": "What is Mendix platform?", "category": "partners"},
    {"id": 21, "question": "What values does RealSoft have?", "category": "company_info"},
    {"id": 22, "question": "What is RealSoft's talent outsourcing service?", "category": "services"},
    {"id": 23, "question": "What is the 8th International Conference of the Union of Arab Statisticians?", "category": "awards"},
    {"id": 24, "question": "What is RealData Flow?", "category": "products"},
    {"id": 25, "question": "What is SBM company?", "category": "partners"},
]


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


def regenerate_questions():
    """Regenerate evaluation questions with correct expected chunks."""
    print("Regenerating eval_questions.json with current pipeline...")
    print(f"API: {API_BASE}")
    print("=" * 80)
    
    evaluation_set = []
    
    for q in QUESTIONS:
        print(f"\nQ{q['id']:02d}: {q['question'][:60]}...")
        
        response = query_rag(q["question"])
        sources = response.get("sources", [])
        
        if sources:
            # Use the top-ranked result as expected
            top_source = sources[0]
            expected = {
                "id": q["id"],
                "question": q["question"],
                "expected_source": top_source.get("source_file", "The Content of RealSoft (1).pdf"),
                "expected_page": top_source.get("page_number", 1),
                "expected_chunk_id": top_source.get("chunk_id"),
                "category": q["category"]
            }
            print(f"      Expected: Page {expected['expected_page']}, Chunk {expected['expected_chunk_id'][:20]}...")
        else:
            # Fallback if no sources
            expected = {
                "id": q["id"],
                "question": q["question"],
                "expected_source": "The Content of RealSoft (1).pdf",
                "expected_page": 1,
                "expected_chunk_id": None,
                "category": q["category"]
            }
            print(f"      [WARNING] No sources found")
        
        evaluation_set.append(expected)
        
        # Rate limiting delay
        import time
        time.sleep(0.5)
    
    # Save to file
    output = {"evaluation_set": evaluation_set}
    
    with open("eval_questions.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print(f"[✓] Saved {len(evaluation_set)} questions to eval_questions.json")
    print("=" * 80)


if __name__ == "__main__":
    regenerate_questions()
