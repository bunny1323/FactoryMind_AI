"""
Debug script to test full RAG pipeline including answer generation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import settings
from backend.dependencies import container
from backend.services.rag_service import rag_service

# Test query
query = "According to the safety section, what safety hazard is represented by icon 13031GE07?"
print(f"Testing full RAG pipeline for query: '{query}'")
print("="*70)

# Test get_grounded_answer
try:
    answer, hits = rag_service.get_grounded_answer(query, top_k_per_coll=15, user_id="default_user")
    
    print(f"Answer: {answer}")
    print("="*70)
    print(f"Number of retrieved hits: {len(hits)}")
    print("="*70)
    
    for i, hit in enumerate(hits[:5]):
        print(f"\nHit {i+1}:")
        print(f"  Score: {hit.get('score', 0):.4f}")
        print(f"  Title: {hit.get('title', 'N/A')}")
        print(f"  Text: {hit.get('text', '')[:300]}...")
        print(f"  Payload: {hit.get('payload', {})}")
        
except Exception as e:
    print(f"RAG Pipeline Error: {e}")
    import traceback
    traceback.print_exc()

print("="*70)
