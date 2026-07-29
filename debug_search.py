"""
Debug script to test search pipeline for icon query.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import settings
from backend.dependencies import container
from backend.services.rag_service import rag_service

# Test query
query = "According to the safety section, what safety hazard is represented by icon 13031GE07?"
print(f"Testing query: '{query}'")
print("="*70)

# Test intent classification
from backend.services.rag_service import classify_intent
intent = classify_intent(query)
print(f"Classified Intent: {intent}")
print("="*70)

# Test vector store connection
print(f"Vector Store Type: {type(container.vector_store).__name__}")
print(f"Embedder Type: {type(container.embedder).__name__}")
print(f"Reranker Type: {type(container.reranker).__name__}")
print("="*70)

# Test search
print("Performing search...")
try:
    results = rag_service.search_all_collections(query, top_k=10, user_id="default_user")
    print(f"Search Results:")
    for collection, hits in results.items():
        print(f"\nCollection: {collection}")
        print(f"  Hits: {len(hits)}")
        for i, hit in enumerate(hits[:3]):
            print(f"  Hit {i+1}:")
            print(f"    Score: {hit.get('score', 0):.4f}")
            print(f"    Title: {hit.get('title', 'N/A')}")
            print(f"    Text Preview: {hit.get('text', '')[:100]}...")
            print(f"    Payload: {hit.get('payload', {})}")
except Exception as e:
    print(f"Search Error: {e}")
    import traceback
    traceback.print_exc()

print("="*70)

# Test direct search for icon references
print("Searching for '13031GE07' directly...")
try:
    icon_results = rag_service.search_all_collections("13031GE07 icon safety", top_k=10, user_id="default_user")
    for collection, hits in icon_results.items():
        print(f"\nCollection: {collection}")
        print(f"  Hits: {len(hits)}")
        for i, hit in enumerate(hits[:3]):
            print(f"  Hit {i+1}:")
            print(f"    Score: {hit.get('score', 0):.4f}")
            print(f"    Text: {hit.get('text', '')[:200]}...")
except Exception as e:
    print(f"Icon Search Error: {e}")
    import traceback
    traceback.print_exc()

print("="*70)

# Check if vector store has data
print("Checking vector store stats...")
try:
    stats = container.vector_store.get_stats()
    print(f"Stats: {stats}")
except Exception as e:
    print(f"Stats Error: {e}")
    import traceback
    traceback.print_exc()
