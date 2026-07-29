"""
Diagnostic script for FactoryMind AI RAG system.
Identifies root cause of "No relevant information" errors.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import settings
from backend.dependencies import container
from rag.embeddings import build_embedder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_1_embedder_type():
    """TEST 1: Check what embedder is loaded."""
    print_section("TEST 1: Embedder Type Check")
    
    embedder = build_embedder(
        settings.EMBEDDING_BACKEND, 
        settings.EMBEDDING_MODEL, 
        settings.EMBEDDING_DIMENSION
    )
    
    embedder_name = embedder.__class__.__name__
    print(f"✅ Embedder loaded: {embedder_name}")
    print(f"✅ Embedding dimension: {embedder.dimension}")
    print(f"✅ Embedding model: {settings.EMBEDDING_MODEL}")
    
    if embedder_name == "HashEmbedder":
        print("\n❌ CRITICAL: HashEmbedder loaded instead of FastEmbed!")
        print("   This means semantic search will NOT work.")
        print("   FIX: pip install fastembed")
        return False
    else:
        print("\n✅ GOOD: FastEmbed loaded for semantic search")
        return True

def test_2_qdrant_connection():
    """TEST 2: Check Qdrant connection."""
    print_section("TEST 2: Qdrant Connection Check")
    
    print(f"✅ Vector backend: {settings.VECTOR_BACKEND}")
    print(f"✅ Qdrant URL: {settings.QDRANT_URL}")
    print(f"✅ Qdrant API key set: {bool(settings.QDRANT_API_KEY)}")
    
    if settings.VECTOR_BACKEND == "memory":
        print("\n⚠️  WARNING: Using in-memory vector store (not Qdrant)")
        print("   Data will be lost on restart")
        return True
    
    if settings.VECTOR_BACKEND == "qdrant":
        try:
            vector_store = container.vector_store
            print(f"✅ Vector store type: {vector_store.__class__.__name__}")
            
            # Try to list collections
            if hasattr(vector_store, 'client'):
                collections = vector_store.client.get_collections()
                print(f"✅ Connected to Qdrant Cloud")
                print(f"✅ Collections found: {[c.name for c in collections.collections]}")
                return True
            else:
                print("\n❌ ERROR: Qdrant client not initialized")
                return False
        except Exception as e:
            print(f"\n❌ ERROR: Cannot connect to Qdrant: {e}")
            print("   FIX: Check QDRANT_URL and QDRANT_API_KEY in .env")
            return False
    
    return True

def test_3_collection_stats():
    """TEST 3: Check collection sizes."""
    print_section("TEST 3: Collection Statistics")
    
    vector_store = container.vector_store
    collections_to_check = ["manuals", "error_codes", "spare_parts", "maintenance_logs", "sop"]
    
    has_data = False
    for coll_name in collections_to_check:
        try:
            if hasattr(vector_store, 'client'):
                try:
                    info = vector_store.client.get_collection(coll_name)
                    count = info.points_count
                    print(f"✅ {coll_name}: {count} documents")
                    if count > 0:
                        has_data = True
                except Exception:
                    print(f"⚠️  {coll_name}: Collection does not exist")
            else:
                # In-memory store
                print(f"⚠️  {coll_name}: In-memory store (cannot check)")
                has_data = True  # Assume data exists
        except Exception as e:
            print(f"❌ {coll_name}: Error checking - {e}")
    
    if not has_data:
        print("\n❌ CRITICAL: All collections are empty!")
        print("   FIX: Run ingestion pipeline to populate collections")
        return False
    else:
        print("\n✅ GOOD: At least one collection has data")
        return True

def test_4_embedding_dimension():
    """TEST 4: Check embedding dimension consistency."""
    print_section("TEST 4: Embedding Dimension Check")
    
    embedder = build_embedder(
        settings.EMBEDDING_BACKEND, 
        settings.EMBEDDING_MODEL, 
        settings.EMBEDDING_DIMENSION
    )
    
    query_dim = embedder.dimension
    config_dim = settings.EMBEDDING_DIMENSION
    
    print(f"✅ Config dimension: {config_dim}")
    print(f"✅ Actual embedder dimension: {query_dim}")
    
    if query_dim != config_dim:
        print(f"\n❌ CRITICAL: Dimension mismatch!")
        print(f"   Config says {config_dim} but embedder produces {query_dim}")
        return False
    else:
        print(f"\n✅ GOOD: Dimensions match ({query_dim})")
        return True

def test_5_search_test():
    """TEST 5: Test actual search with sample query."""
    print_section("TEST 5: Search Test")
    
    try:
        from backend.services.rag_service import rag_service
        
        test_query = "safety hazard"
        print(f"✅ Test query: '{test_query}'")
        
        # Try search
        results = rag_service.search_all_collections(test_query, top_k=5, user_id="default_user")
        
        total_results = sum(len(hits) for hits in results.values())
        print(f"✅ Total results found: {total_results}")
        
        if total_results == 0:
            print("\n❌ CRITICAL: Search returned NO results")
            print("   Possible causes:")
            print("   - Collections are empty")
            print("   - Embedding dimension mismatch")
            print("   - HashEmbedder instead of FastEmbed")
            return False
        
        print("\n✅ Results by collection:")
        for coll, hits in results.items():
            if hits:
                print(f"   {coll}: {len(hits)} results")
                for i, hit in enumerate(hits[:2]):
                    score = hit.get('score', 0)
                    text_preview = hit.get('text', '')[:50]
                    print(f"     [{i+1}] score={score:.3f} text='{text_preview}...'")
        
        return True
    except Exception as e:
        print(f"\n❌ ERROR during search test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_6_rag_pipeline():
    """TEST 6: Test full RAG pipeline via agent orchestrator."""
    print_section("TEST 6: Full RAG Pipeline Test")
    
    try:
        from agents.graph import agent_orchestrator
        
        test_query = "What are the safety procedures?"
        print(f"✅ Test query: '{test_query}'")
        
        # Try full RAG via agent orchestrator
        state = agent_orchestrator.run(
            query=test_query,
            machine_id="R215L",
            user_id="default_user"
        )
        
        answer = state.get("final_answer", "")
        docs = state.get("retrieved_documents", [])
        
        if not answer:
            print("\n❌ CRITICAL: RAG returned no answer")
            return False
        
        print(f"✅ Answer generated: {answer[:300]}...")
        print(f"✅ Citations: {len(docs)}")
        
        if len(docs) == 0:
            print("\n⚠️  WARNING: No citations in answer")
        else:
            print("\n✅ Top citation:")
            hit = docs[0]
            print(f"   Score: {hit.get('score', 0):.3f}")
            print(f"   Source: {hit.get('payload', {}).get('document_name', 'Unknown')}")
        
        return True
    except Exception as e:
        print(f"\n❌ ERROR during RAG test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests."""
    print("\n" + "="*60)
    print("  FactoryMind AI - RAG Diagnostic Tool")
    print("  Identifying root cause of 'No Relevant Information'")
    print("="*60)
    
    results = {
        "Embedder Type": test_1_embedder_type(),
        "Qdrant Connection": test_2_qdrant_connection(),
        "Collection Stats": test_3_collection_stats(),
        "Embedding Dimension": test_4_embedding_dimension(),
        "Search Test": test_5_search_test(),
        "RAG Pipeline": test_6_rag_pipeline(),
    }
    
    print_section("DIAGNOSTIC SUMMARY")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED - System should be working")
    else:
        print("❌ SOME TESTS FAILED - See details above")
        print("\nMost likely issues (in order):")
        print("1. HashEmbedder instead of FastEmbed → pip install fastembed")
        print("2. Collections empty → Re-run ingestion pipeline")
        print("3. Dimension mismatch → Check EMBEDDING_DIMENSION config")
        print("4. Qdrant connection → Check QDRANT_URL and API_KEY")
        print("5. Relevance threshold → Lower RAG_MIN_RELEVANCE_SCORE")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
