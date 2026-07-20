from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple
from backend.dependencies import container
from backend.services.llm_service import llm_service
from backend.config import settings

logger = logging.getLogger("factorymind")

SYSTEM_RULES = """You are a heavy machinery maintenance intelligence assistant. Use only the supplied technical context.
Do not speculate, make up error codes, or claim absolute certainty if not detailed in the context.
Be direct, highly technical, and engineering-focused. Use markdown formatting. Cite the document names or error codes used."""

class RagService:
    def __init__(self):
        self.vector_store = container.vector_store
        self.reranker = container.reranker
        self._answer_cache: dict[str, tuple[str, list[dict[str, Any]]]] = {}

    def clear_cache(self):
        self._answer_cache.clear()
        logger.info("RAG Service query answer cache cleared.")

    def search_all_collections(self, query: str, top_k: int = 5, user_id: str = "default_user") -> dict[str, list[dict[str, Any]]]:
        results = {}
        collections = ["manuals", "sop", "error_codes", "spare_parts"]
        
        for coll in collections:
            try:
                # User-level document isolation filter
                filters = {"user_id": user_id}
                hits = self.vector_store.search(coll, query, top_k=top_k, filters=filters)
                results[coll] = hits
            except Exception as e:
                logger.error(f"Error searching collection {coll}: {e}")
                results[coll] = []
        return results

    def get_grounded_answer(self, query: str, top_k_per_coll: int = 3, user_id: str = "default_user") -> tuple[str, list[dict[str, Any]]]:
        cache_key = f"{user_id}:{query.strip().lower()}"
        if cache_key in self._answer_cache:
            logger.info(f"CACHE HIT: Returning cached answer for query: '{query}' under user {user_id}")
            return self._answer_cache[cache_key]

        # 1. Retrieve raw hits from collections with user isolation
        all_hits = self.search_all_collections(query, top_k=top_k_per_coll, user_id=user_id)
        
        # 2. Combine and rerank
        flat_hits = []
        for coll, hits in all_hits.items():
            for hit in hits:
                flat_hits.append(hit)

        # Apply reranking
        reranked_hits = self.reranker.rerank(query, flat_hits, top_k=5)
        
        # Filter by minimum score
        filtered_hits = [
            hit for hit in reranked_hits 
            if hit.get("score", 0.0) >= settings.RAG_MIN_RELEVANCE_SCORE
        ]

        if not filtered_hits:
            # Try relaxing score filter if no hits returned
            filtered_hits = reranked_hits[:3]

        if not filtered_hits:
            return (
                "No relevant information was found.",
                []
            )

        # 3. Construct context
        context_blocks = []
        for hit in filtered_hits:
            source = hit.get("title", hit.get("id"))
            text = hit.get("text", "")
            context_blocks.append(f"SOURCE: {source}\nCONTENT: {text}")
        
        context = "\n\n".join(context_blocks)

        # 4. Synthesize final answer using LLM
        synthesized_text = llm_service.synthesize(query, context, SYSTEM_RULES)
        
        # Save to cache
        self._answer_cache[cache_key] = (synthesized_text, filtered_hits)
        
        return synthesized_text, filtered_hits

    def post_process_layout_aware(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Restores document layout relationships by fetching adjacent/referenced chunks."""
        processed_hits = []
        retrieved_ids = {hit["id"] for hit in hits}
        
        for hit in hits:
            processed_hits.append(hit)
            payload = hit.get("payload", {})
            filename = payload.get("document_name")
            chunk_index = payload.get("chunk_index")
            collection = payload.get("collection", "manuals")
            
            if filename and chunk_index is not None:
                # 1. Fetch Previous Chunk to restore context
                prev_id = f"{filename}_chunk_{chunk_index - 1}"
                if prev_id not in retrieved_ids:
                    try:
                        prev_hit = self.vector_store.get_point(collection, prev_id)
                        if prev_hit:
                            prev_hit["score"] = round(hit.get("score", 0.90) - 0.05, 4)
                            processed_hits.append(prev_hit)
                            retrieved_ids.add(prev_id)
                    except Exception:
                        pass
                
                # 2. Fetch Next Chunk to restore context
                next_id = f"{filename}_chunk_{chunk_index + 1}"
                if next_id not in retrieved_ids:
                    try:
                        next_hit = self.vector_store.get_point(collection, next_id)
                        if next_hit:
                            next_hit["score"] = round(hit.get("score", 0.90) - 0.05, 4)
                            processed_hits.append(next_hit)
                            retrieved_ids.add(next_id)
                    except Exception:
                        pass
                        
        return processed_hits

rag_service = RagService()
