#!/usr/bin/env python3
"""
Generate 100 mixed English/Arabic eval questions based on RealSoft content
"""

import json
from typing import Dict, Any, List

# Questions mixing English and Arabic
QUESTIONS = [
    {"id": 1, "question": "What is the main mission of RealSoft?", "category": "company_info"},
    {"id": 2, "question": "ما هي رؤية شركة ريلسوفت؟", "category": "company_info"},
    {"id": 3, "question": "When was RealSoft founded?", "category": "company_info"},
    {"id": 4, "question": "What products does RealSoft offer?", "category": "products"},
    {"id": 5, "question": "ما هو منصة الخوارزمي؟", "category": "products"},
    {"id": 6, "question": "How many years of experience does RealSoft have?", "category": "company_info"},
    {"id": 7, "question": "What statistical solutions does RealSoft provide?", "category": "services"},
    {"id": 8, "question": "What is the vision of RealSoft?", "category": "company_info"},
    {"id": 9, "question": "Who is the Director of Jordan Meteorological Department?", "category": "testimonials"},
    {"id": 10, "question": "في أي دول تعمل ريلسوفت؟", "category": "company_info"},
    {"id": 11, "question": "What is RealData Hub?", "category": "products"},
    {"id": 12, "question": "What technology partners does RealSoft work with?", "category": "partners"},
    {"id": 13, "question": "What is Microsoft partnership about?", "category": "partners"},
    {"id": 14, "question": "What awards has RealSoft received from Union of Arab Statisticians?", "category": "awards"},
    {"id": 15, "question": "ما هو رقم هاتف ريلسوفت في الأردن؟", "category": "contact"},
    {"id": 16, "question": "What is the FalconMap platform?", "category": "products"},
    {"id": 17, "question": "What is RealSoft's approach to digital transformation?", "category": "services"},
    {"id": 18, "question": "Who is Dr. Diaa Awad Kazem?", "category": "testimonials"},
    {"id": 19, "question": "What is the Ada'a product?", "category": "products"},
    {"id": 20, "question": "ما هي شركة إسري وما الذي تفعله؟", "category": "partners"},
    {"id": 21, "question": "What is Mendix platform?", "category": "partners"},
    {"id": 22, "question": "What values does RealSoft have?", "category": "company_info"},
    {"id": 23, "question": "What is RealSoft's talent outsourcing service?", "category": "services"},
    {"id": 24, "question": "What is the 8th International Conference of the Union of Arab Statisticians?", "category": "awards"},
    {"id": 25, "question": "What is RealData Flow?", "category": "products"},
    {"id": 26, "question": "What is SBM company?", "category": "partners"},
    {"id": 27, "question": "ما هو منتج ريل كستومز؟", "category": "products"},
    {"id": 28, "question": "What is the Workflow Engine product?", "category": "products"},
    {"id": 29, "question": "What is the motto of RealSoft?", "category": "company_info"},
    {"id": 30, "question": "من هو رائد رافد الخطاب؟", "category": "testimonials"},
    {"id": 31, "question": "How many employees does RealSoft have?", "category": "company_info"},
    {"id": 32, "question": "How many clients does RealSoft serve?", "category": "company_info"},
    {"id": 33, "question": "What is the main focus of RealSoft's AI solutions?", "category": "services"},
    {"id": 34, "question": "ما هي قيم ريلسوفت الأساسية؟", "category": "company_info"},
    {"id": 35, "question": "What is the first core value of RealSoft?", "category": "company_info"},
    {"id": 36, "question": "What is the second core value of RealSoft?", "category": "company_info"},
    {"id": 37, "question": "What is the third core value of RealSoft?", "category": "company_info"},
    {"id": 38, "question": "What is the fourth core value of RealSoft?", "category": "company_info"},
    {"id": 39, "question": "ما هي مزايا منصة أدا؟", "category": "products"},
    {"id": 40, "question": "What branches does RealSoft have?", "category": "company_info"},
    {"id": 41, "question": "What types of solutions does RealSoft specialize in?", "category": "services"},
    {"id": 42, "question": "What additional services does RealSoft provide?", "category": "services"},
    {"id": 43, "question": "What technologies does RealSoft utilize?", "category": "services"},
    {"id": 44, "question": "ما هو برنامج فوكس إن إس أو؟", "category": "products"},
    {"id": 45, "question": "What does RealSoft's solution serve?", "category": "services"},
    {"id": 46, "question": "What statistical domains does RealSoft's solution cover?", "category": "services"},
    {"id": 47, "question": "What workflow does RealSoft's solution power?", "category": "services"},
    {"id": 48, "question": "What technology does RealSoft integrate with?", "category": "services"},
    {"id": 49, "question": "What capabilities does RealSoft's solution have?", "category": "services"},
    {"id": 50, "question": "ما هي ميزات منصة الخوارزمي؟", "category": "products"},
    {"id": 51, "question": "What makes Al-Khwarizmi flexible?", "category": "products"},
    {"id": 52, "question": "What did a client say about using Al-Khwarizmi?", "category": "testimonials"},
    {"id": 53, "question": "How long has a client been using Al-Khwarizmi?", "category": "testimonials"},
    {"id": 54, "question": "What benefits did a client experience with Al-Khwarizmi?", "category": "testimonials"},
    {"id": 55, "question": "ما هي وظيفة نظام إدارة سير العمل؟", "category": "products"},
    {"id": 56, "question": "What can be configured in the Workflow Engine?", "category": "products"},
    {"id": 57, "question": "What types of workflows can be created with the Workflow Engine?", "category": "products"},
    {"id": 58, "question": "What is the purpose of FalconMap?", "category": "products"},
    {"id": 59, "question": "What is FalconMap powered by?", "category": "products"},
    {"id": 60, "question": "ما هي ميزات فوكس ماب؟", "category": "products"},
    {"id": 61, "question": "What map editing capabilities does FalconMap have?", "category": "products"},
    {"id": 62, "question": "What devices does FalconMap support?", "category": "products"},
    {"id": 63, "question": "What is the Multi-Base Maps feature?", "category": "products"},
    {"id": 64, "question": "What does the Topology Rules feature do?", "category": "products"},
    {"id": 65, "question": "What is the Field task management component?", "category": "products"},
    {"id": 66, "question": "What are the Online-offline Modes in FalconMap?", "category": "products"},
    {"id": 67, "question": "What does RealData Flow do?", "category": "products"},
    {"id": 68, "question": "ما هي مميزات ريل داتا فلو؟", "category": "products"},
    {"id": 69, "question": "What are the Key Features & Capabilities of RealData Flow?", "category": "products"},
    {"id": 70, "question": "What is Multi-Source Support in RealData Flow?", "category": "products"},
    {"id": 71, "question": "What does Automated Workflows do in RealData Flow?", "category": "products"},
    {"id": 72, "question": "What clients has RealSoft worked with?", "category": "clients"},
    {"id": 73, "question": "What is the main purpose of RealData Hub?", "category": "products"},
    {"id": 74, "question": "ما هي مكونات منصة ريل داتا هب؟", "category": "products"},
    {"id": 75, "question": "What does RealData Hub provide?", "category": "products"},
    {"id": 76, "question": "What is the main benefit of RealCustoms?", "category": "products"},
    {"id": 77, "question": "What problem did RealSoft help solve for clients?", "category": "services"},
    {"id": 78, "question": "What was the first challenge in the client's problem?", "category": "services"},
    {"id": 79, "question": "What was the second challenge in the client's problem?", "category": "services"},
    {"id": 80, "question": "ما كان التحدي الثالث في مشكلة العميل؟", "category": "services"},
    {"id": 81, "question": "What was the fourth challenge in the client's problem?", "category": "services"},
    {"id": 82, "question": "What makes RealSoft a trusted provider?", "category": "company_info"},
    {"id": 83, "question": "How does RealSoft help partners?", "category": "services"},
    {"id": 84, "question": "From which countries does RealSoft provide talents?", "category": "services"},
    {"id": 85, "question": "What does RealSoft's vision aim for?", "category": "company_info"},
    {"id": 86, "question": "How does RealSoft foster trust?", "category": "company_info"},
    {"id": 87, "question": "What does RealSoft do to commit to deliver?", "category": "company_info"},
    {"id": 88, "question": "ما معنى أن تكون مستعدًا لغدًا في ريلسوفت؟", "category": "company_info"},
    {"id": 89, "question": "What geographic regions does RealSoft operate in?", "category": "company_info"},
    {"id": 90, "question": "What types of solutions does RealSoft specialize in?", "category": "services"},
    {"id": 91, "question": "What additional services does RealSoft provide?", "category": "services"},
    {"id": 92, "question": "What technologies does RealSoft work with?", "category": "services"},
    {"id": 93, "question": "What is the motto of RealSoft regarding data?", "category": "company_info"},
    {"id": 94, "question": "ما هي رسالة ريلسوفت حول البيانات؟", "category": "company_info"},
    {"id": 95, "question": "How many countries does RealSoft operate in?", "category": "company_info"},
    {"id": 96, "question": "What is the primary focus of RealSoft's work?", "category": "company_info"},
    {"id": 97, "question": "What does RealSoft aim to provide with its solutions?", "category": "services"},
    {"id": 98, "question": "What makes RealSoft's applications special?", "category": "services"},
    {"id": 99, "question": "What is the core philosophy of RealSoft?", "category": "company_info"},
    {"id": 100, "question": "ما هو شعار ريلسوفت الذي يعبر عن توجهها نحو الأعمال؟", "category": "company_info"},
]

def generate_questions():
    """Generate evaluation questions with placeholder expected values."""
    print("Generating 100 mixed English/Arabic evaluation questions...")
    print("=" * 80)
    
    evaluation_set = []
    
    for q in QUESTIONS:
        expected = {
            "id": q["id"],
            "question": q["question"],
            "expected_source": "The Content of RealSoft (1).pdf",
            "expected_page": 1,  # Placeholder - will be updated by regeneration script
            "expected_chunk_id": "",  # Placeholder - will be updated by regeneration script
            "category": q["category"]
        }
        evaluation_set.append(expected)
        
        # Print progress
        print(f"Added Q{q['id']:02d}: {q['question'][:50]}{'...' if len(q['question']) > 50 else ''}")
    
    # Save to file
    output = {"evaluation_set": evaluation_set}
    
    with open("eval_questions_mixed.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print(f"[✓] Generated {len(evaluation_set)} mixed English/Arabic questions")
    print("Saved to: eval_questions_mixed.json")
    print("Run 'python regenerate_eval_questions.py' to update expected chunks")
    print("=" * 80)

if __name__ == "__main__":
    generate_questions()