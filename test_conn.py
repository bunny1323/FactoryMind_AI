import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import traceback
import sys

print("Starting diagnostics...")
try:
    print("1. Loading config...")
    from backend.config import settings
    print(f"   QDRANT_URL={settings.QDRANT_URL}")
    
    print("2. Importing fastembed and trying to load Dense model...")
    from fastembed import TextEmbedding
    print("   fastembed imported. Instantiating TextEmbedding...")
    dense_model = TextEmbedding(model_name=settings.EMBEDDING_MODEL)
    print("   Dense model loaded successfully!")

    print("3. Importing fastembed and trying to load Sparse model...")
    from fastembed import SparseTextEmbedding
    print("   Instantiating SparseTextEmbedding...")
    sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_EMBEDDING_MODEL)
    print("   Sparse model loaded successfully!")

    print("4. Importing reranker...")
    from rag.reranker import build_reranker
    reranker = build_reranker(settings.RERANKER_BACKEND, settings.RERANKER_MODEL)
    print("   Reranker loaded successfully!")

    print("5. Initializing dependencies container...")
    from backend.dependencies import container
    print("   Container initialized successfully!")
    
except Exception as e:
    print("\n--- DIAGNOSTICS FAILURE TRACEBACK ---")
    traceback.print_exc()
    sys.exit(1)
