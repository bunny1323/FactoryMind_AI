"""Script to test specific questions and print retrieval details."""
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from backend.services.query_planner import query_planner
from backend.services.rag_service import rag_service

print("="*80)
print("TESTING RETRIEVAL FOR SPECIFIC QUESTIONS")
print("="*80)

# Test questions covering different intents
test_questions = [
    "What is the engine oil capacity?",
    "How do I replace the hydraulic filter?",
    "What are the torque specifications for track bolts?",
    "Show me the electrical wiring diagram",
    "What safety precautions should I follow?",
    "How do I troubleshoot low hydraulic pressure?",
]

for i, question in enumerate(test_questions, 1):
    print(f"\n{'='*80}")
    print(f"TEST {i}: {question}")
    print(f"{'='*80}")

    # Step 1: Query Planning
    plan = query_planner.plan(question)
    print(f"\n--- QUERY PLANNING ---")
    print(f"Intent: {plan.intent}")
    print(f"Rewritten Query: {plan.rewritten_query}")
    print(f"Target Collections: {plan.target_collections}")
    print(f"Is Conversational: {plan.is_conversational}")
    print(f"Requires Visual: {plan.requires_visual}")
    print(f"Metadata Filters: {plan.metadata_filters}")

    if plan.is_conversational:
        print("⚠️  Conversational intent - skipping retrieval")
        continue

    # Step 2: Retrieval
    print(f"\n--- RETRIEVAL ---")
    try:
        retrieval_results = rag_service.search_all_collections(question, top_k=10, user_id="default_user")
        
        total_hits = sum(len(hits) for hits in retrieval_results.values())
        print(f"Total Hits: {total_hits}")
        
        for coll, hits in retrieval_results.items():
            print(f"\nCollection '{coll}': {len(hits)} hits")
            for j, hit in enumerate(hits[:3], 1):  # Show top 3 hits per collection
                payload = hit.get("payload", {})
                print(f"  [{j}] Score: {hit.get('score', 0):.4f}")
                print(f"      Document: {payload.get('document_name', 'Unknown')}")
                print(f"      Page: {payload.get('page', 'N/A')}")
                print(f"      Section: {payload.get('heading', 'General')}")
                print(f"      Text Preview: {hit.get('text', '')[:100]}...")
        
        if total_hits == 0:
            print("❌ NO RESULTS RETRIEVED")
        else:
            print(f"✅ SUCCESSFULLY RETRIEVED {total_hits} chunks")
            
    except Exception as e:
        print(f"❌ ERROR during retrieval: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
