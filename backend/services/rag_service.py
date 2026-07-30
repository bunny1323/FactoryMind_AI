from __future__ import annotations

import logging
import asyncio
from typing import Any, Dict, List, Tuple
from backend.dependencies import container
from backend.services.llm_service import llm_service
from backend.services.vision_service import vision_service
from backend.services.language_service import language_service
from backend.services.conversation_memory import conversation_memory
from backend.services.query_planner import query_planner
from backend.config import settings

logger = logging.getLogger("factorymind")

SYSTEM_RULES = """You are FactoryMind AI, an expert industrial maintenance assistant for the Hyundai R215L Smart Plus excavator.

CRITICAL RULES:
1. Answer ONLY using the provided context from indexed maintenance manuals.
2. If the context doesn't contain the answer, say exactly: "I could not find this information inside the indexed manuals."
3. NEVER fabricate part numbers, torque values, repair procedures, safety precautions, or manual page numbers.
4. Structure your response into the following 10 sections:
   1. Final Answer
   2. Supporting Evidence
   3. Manual
   4. Section
   5. Page Number
   6. Retrieved Images
   7. Diagrams
   8. Confidence Score
   9. Related Components
   10. Suggested Follow-up Questions
5. Always cite the exact manual filename, section title, and page numbers.
6. Keep technical answers precise, complete, and engineering-focused."""


def classify_intent(query: str) -> str:
    """Delegates intent classification to query_planner for backward compatibility."""
    return query_planner.classify_intent(query)


class RagService:
    def __init__(self):
        self.vector_store = container.vector_store
        self.reranker = container.reranker
        self._answer_cache: dict[str, tuple[str, list[dict[str, Any]]]] = {}

    def clear_cache(self):
        self._answer_cache.clear()
        logger.info("RAG Service query answer cache cleared.")

    def search_by_intent(self, query: str, top_k: int = 10, user_id: str = "default_user") -> dict[str, list[dict[str, Any]]]:
        """
        Executes intent-aware hybrid retrieval (Dense + BM25 Sparse + RRF).
        Retrieves up to 50 combined candidate chunks across target collections.
        """
        plan = query_planner.plan(query)
        logger.info(f"RAG RETRIEVAL INTENT: '{plan.intent}' | Rewritten: '{plan.rewritten_query}'")

        if plan.is_conversational:
            return {"manuals": []}

        collections = plan.target_collections
        search_query = plan.rewritten_query

        results = {}
        # Build filters: metadata filters from query planner only
        # NOTE: user_id filter removed — ingestion stores "default_user" but
        # authenticated users resolve to "user-<name>", causing 0 Qdrant results.
        # All authenticated users should access all indexed manuals.
        filters = None
        if plan.metadata_filters:
            filters = dict(plan.metadata_filters)
            logger.info(f"Applied metadata filters: {plan.metadata_filters}")
        
        # Calculate per-collection fetch count to yield ~50 candidate chunks
        top_k_per_coll = max(10, 50 // len(collections)) if collections else 10

        for coll in collections:
            try:
                hits = self.vector_store.search(coll, search_query, top_k=top_k_per_coll, filters=filters)
                results[coll] = hits
            except Exception as e:
                logger.error(f"Error searching collection {coll}: {e}")
                results[coll] = []

        return results

    def search_all_collections(self, query: str, top_k: int = 10, user_id: str = "default_user") -> dict[str, list[dict[str, Any]]]:
        """Targeted hybrid search wrapper."""
        return self.search_by_intent(query, top_k=top_k, user_id=user_id)

    def get_grounded_answer(self, query: str, top_k_per_coll: int = 10, user_id: str = "default_user") -> tuple[str, list[dict[str, Any]]]:
        cache_key = f"{user_id}:{query.strip().lower()}"
        if cache_key in self._answer_cache:
            logger.info(f"CACHE HIT: Returning cached answer for query: '{query}' under user {user_id}")
            return self._answer_cache[cache_key]

        # Add user query to conversation memory
        conversation_memory.add_message(user_id, "user", query)

        # Detect query language for multilingual response
        detected_lang = language_service.detect_language(query)
        language_instruction = language_service.get_system_prompt_language(detected_lang)
        logger.info(f"Query language: {language_service.LANGUAGE_NAMES.get(detected_lang, detected_lang)}")

        # 1. Retrieve up to 50 candidate chunks
        all_hits = self.search_all_collections(query, top_k=top_k_per_coll, user_id=user_id)
        
        flat_hits = []
        for coll, hits in all_hits.items():
            for hit in hits:
                flat_hits.append(hit)

        if not flat_hits:
            no_result_msg = "I could not find this information inside the indexed manuals."
            if detected_lang != "en":
                no_result_msg = f"{language_instruction} {no_result_msg}"
            conversation_memory.add_message(user_id, "assistant", no_result_msg)
            return (no_result_msg, [])

        # 2. Rerank 50 chunks down to Top 10 using CrossEncoder
        try:
            reranked_hits = self.reranker.rerank(query, flat_hits, top_k=10)
        except Exception as e:
            logger.warning(f"Reranker failed ({e}). Using initial score ranking.")
            reranked_hits = sorted(flat_hits, key=lambda x: x.get("score", 0), reverse=True)[:10]

        # Filter by minimum relevance threshold
        filtered_hits = [
            hit for hit in reranked_hits 
            if hit.get("score", 0.0) >= settings.RAG_MIN_RELEVANCE_SCORE
        ]

        if not filtered_hits:
            filtered_hits = reranked_hits[:5]

        # Select Top 5 for LLM context window
        llm_context_hits = filtered_hits[:5]

        # 3. Restore layout-aware context (siblings/parents)
        expanded_hits = self.post_process_layout_aware(llm_context_hits)

        # 4. Construct context block
        context_blocks = []
        for hit in expanded_hits:
            payload = hit.get("payload", {})
            source_manual = payload.get("document_name", "Unknown Manual")
            page = payload.get("page", "?")
            heading = payload.get("heading", "General")
            chunk_type = payload.get("chunk_type", "text")
            text = hit.get("text", "")
            
            ref = f"MANUAL: {source_manual} | SECTION: {heading} | PAGE: {page}"
            
            if chunk_type == "table":
                context_blocks.append(f"[{ref}] (Table Specifications)\n{text}")
            elif chunk_type == "image":
                image_path = payload.get("image_path")
                caption = payload.get("caption") or "Schematic diagram"
                context_blocks.append(f"[{ref}] (Diagram Reference: {image_path})\nCaption: {caption}\nDescription: {text}")
            else:
                context_blocks.append(f"[{ref}]\n{text}")

        context = "\n\n".join(context_blocks)

        # 5. Synthesize final answer using LLM
        enhanced_system_rules = f"{SYSTEM_RULES}\n\n{language_instruction}"
        synthesized_text = llm_service.synthesize(query, context, enhanced_system_rules)
        
        conversation_memory.add_message(user_id, "assistant", synthesized_text)
        self._answer_cache[cache_key] = (synthesized_text, expanded_hits)
        
        return synthesized_text, expanded_hits

    def post_process_layout_aware(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Restores document layout relationships by fetching adjacent/referenced chunks."""
        processed_hits = []
        retrieved_ids = {hit["id"] for hit in hits if "id" in hit}
        
        for hit in hits:
            processed_hits.append(hit)
            payload = hit.get("payload", {})
            filename = payload.get("document_name")
            chunk_index = payload.get("chunk_index")
            collection = payload.get("collection", "manuals")
            
            if filename and chunk_index is not None:
                # 1. Fetch Previous Sibling Chunk
                prev_id = f"{filename}_chunk_{chunk_index - 1}"
                if prev_id not in retrieved_ids:
                    try:
                        prev_hit = self.vector_store.get_point(collection, prev_id)
                        if prev_hit:
                            prev_hit["score"] = round(hit.get("score", 0.90) - 0.05, 4)
                            processed_hits.append(prev_hit)
                            retrieved_ids.add(prev_id)
                    except Exception as e:
                        logger.warning(f"Failed to fetch previous chunk {prev_id}: {e}")
                
                # 2. Fetch Next Sibling Chunk
                next_id = f"{filename}_chunk_{chunk_index + 1}"
                if next_id not in retrieved_ids:
                    try:
                        next_hit = self.vector_store.get_point(collection, next_id)
                        if next_hit:
                            next_hit["score"] = round(hit.get("score", 0.90) - 0.05, 4)
                            processed_hits.append(next_hit)
                            retrieved_ids.add(next_id)
                    except Exception as e:
                        logger.warning(f"Failed to fetch next chunk {next_id}: {e}")

        return processed_hits


rag_service = RagService()
