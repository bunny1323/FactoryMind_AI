from __future__ import annotations

import logging
from backend.config import settings
from rag.embeddings import build_embedder
from rag.sparse_embeddings import build_sparse_embedder
from rag.vector_store import InMemoryHybridVectorStore, QdrantHybridVectorStore, VectorStore
from rag.reranker import build_reranker, Reranker

logger = logging.getLogger("factorymind")

class Container:
    def __init__(self):
        logger.info("Initializing dependency container...")
        self.embedder = build_embedder(settings.EMBEDDING_BACKEND, settings.EMBEDDING_MODEL, settings.EMBEDDING_DIMENSION)
        self.sparse_embedder = build_sparse_embedder(settings.SPARSE_EMBEDDING_BACKEND, settings.SPARSE_EMBEDDING_MODEL)
        
        # Log dense embedder status
        if self.embedder.__class__.__name__ == "FastEmbedDenseEmbedder":
            logger.info(f"DENSE EMBEDDER: Successfully loaded real FastEmbed {settings.EMBEDDING_MODEL} model.")
        else:
            logger.warning("DENSE EMBEDDER: FastEmbed model not available - fell back to offline HashEmbedder.")
            
        # Log sparse embedder status
        if self.sparse_embedder.__class__.__name__ == "FastEmbedBm25SparseEmbedder":
            logger.info("SPARSE EMBEDDER: Successfully loaded real FastEmbed Qdrant/bm25 sparse model.")
        else:
            logger.warning("SPARSE EMBEDDER: FastEmbed sparse model not available - fell back to offline HashLexicalSparseEmbedder.")

        logger.info(f"Resolved VECTOR_BACKEND={settings.VECTOR_BACKEND}")
        if settings.VECTOR_BACKEND == "qdrant":
            masked_url = settings.QDRANT_URL
            if settings.QDRANT_URL and "://" in settings.QDRANT_URL:
                parts = settings.QDRANT_URL.split("://")
                if len(parts) == 2:
                    protocol, host = parts
                    if len(host) > 10:
                        masked_url = f"{protocol}://{host[:4]}...{host[-8:]}"
            logger.info(f"Connecting to Qdrant Cloud/Server at {masked_url}...")
            self.vector_store: VectorStore = QdrantHybridVectorStore(
                self.embedder, 
                self.sparse_embedder, 
                settings.QDRANT_URL, 
                settings.QDRANT_API_KEY, 
                self.embedder.dimension
            )
            try:
                # Test connectivity
                self.vector_store.client.get_collections()
                logger.info("Successfully pinged Qdrant Cloud cluster.")
            except Exception as e:
                logger.error(f"Failed to connect or ping Qdrant Cloud: {e}")
                raise e
        else:
            logger.info("Initializing in-memory hybrid vector store...")
            self.vector_store: VectorStore = InMemoryHybridVectorStore(self.embedder)
            
        self.reranker: Reranker = build_reranker(settings.RERANKER_BACKEND, settings.RERANKER_MODEL)
        
        # Log reranker status
        if self.reranker.__class__.__name__ == "CrossEncoderReranker":
            logger.info(f"RERANKER: Successfully loaded real CrossEncoder model ({settings.RERANKER_MODEL}).")
        else:
            logger.warning(f"RERANKER: CrossEncoder not available - fell back to offline FallbackReranker.")

        # Log LLM status
        provider = settings.LLM_PROVIDER.lower()
        if provider == "groq" and settings.GROQ_API_KEY:
            logger.info(f"LLM PROVIDER: Successfully initialized Groq (model: {settings.GROQ_MODEL}).")
        elif provider == "openai" and settings.OPENAI_API_KEY:
            logger.info(f"LLM PROVIDER: Successfully initialized OpenAI (model: {settings.OPENAI_MODEL}).")
        elif provider == "ollama":
            logger.info(f"LLM PROVIDER: Successfully initialized Ollama (url: {settings.OLLAMA_URL}, model: {settings.OLLAMA_MODEL}).")
        elif provider == "anthropic" and settings.ANTHROPIC_API_KEY:
            logger.info(f"LLM PROVIDER: Successfully initialized Anthropic (model: {settings.ANTHROPIC_MODEL}).")
        elif provider == "mock":
            logger.info("LLM PROVIDER: Initialized Mock LLM for demo answers.")
        else:
            logger.warning(f"LLM PROVIDER: '{settings.LLM_PROVIDER}' requested but API keys are missing. Will fall back to Mock at runtime.")

container = Container()
