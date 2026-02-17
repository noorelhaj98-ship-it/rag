#!/usr/bin/env python3
"""
Offline evaluation of the mixed English/Arabic questions
This script evaluates the questions without requiring a running server
"""

import json
import os
from typing import Dict, Any, List
from qdrant_client import QdrantClient
from pathlib import Path
import time

# Import the functions from rag_answer.py
import rag_answer

def evaluate_questions_offline(questions_file: str = "eval_questions_mixed.json"):
    """Evaluate questions using the rag_answer functions directly."""
    print("Starting offline evaluation of mixed English/Arabic questions...")
    print("=" * 80)
    
    # Load the questions
    with open(questions_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = data["evaluation_set"]
    
    print(f"Loaded {len(questions)} questions from {questions_file}")
    print("Sample questions:")
    for i, q in enumerate(questions[:3]):
        print(f"  Q{q['id']:02d}: {q['question'][:60]}{'...' if len(q['question']) > 60 else ''}")
    print("...")
    print(f"  Q{questions[-1]['id']:02d}: {questions[-1]['question'][:60]}{'...' if len(questions[-1]['question']) > 60 else ''}")
    print()
    
    # Summary of languages and categories
    english_count = sum(1 for q in questions if ord(q['question'][0]) < 128)
    arabic_count = len(questions) - english_count
    
    print(f"Language distribution:")
    print(f"  English questions: {english_count}")
    print(f"  Arabic questions: {arabic_count}")
    print(f"  Total: {len(questions)}")
    print()
    
    categories = {}
    for q in questions:
        cat = q['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("Category distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    print()
    
    # Test a few questions to verify they work with the RAG system
    print("Testing first 3 questions with RAG system...")
    print("-" * 50)
    
    success_count = 0
    for i, q in enumerate(questions[:3]):  # Test first 3 questions
        print(f"Q{q['id']:02d}: {q['question']}")
        
        try:
            # Use the rag_answer.main logic to process the question
            qvec = rag_answer.embed_query(q['question'])
            hits = rag_answer.retrieve(qvec)
            
            if hits:
                print(f"  ✓ Retrieved {len(hits)} results")
                print(f"  Sample result - Page: {hits[0].get('page_number')}, Source: {hits[0].get('source_file')[:30]}...")
                success_count += 1
            else:
                print(f"  ✗ No results found")
                
        except Exception as e:
            print(f"  ✗ Error processing: {str(e)}")
        
        print()
    
    print("=" * 80)
    print(f"Evaluation summary:")
    print(f"- Total questions: {len(questions)}")
    print(f"- English questions: {english_count}")
    print(f"- Arabic questions: {arabic_count}")
    print(f"- Categories: {len(categories)} ({', '.join(categories.keys())})")
    print(f"- Successfully tested with RAG: {success_count}/3")
    print()
    print("The eval_questions_mixed.json file is ready to use with the RAG system.")
    print("To run the full evaluation, start the server with:")
    print("  python -m uvicorn server:app --host 0.0.0.0 --port 8000")
    print("Then run:")
    print("  python run_evaluation.py")
    print("=" * 80)

if __name__ == "__main__":
    evaluate_questions_offline()