from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple
from backend.dependencies import container
from backend.services.llm_service import llm_service
from backend.config import settings

logger = logging.getLogger("factorymind")

SYSTEM_RULES = """You are a heavy machinery maintenance intelligence assistant. Use only the supplied technical context.
Do not speculate, make up error codes, or claim absolute certainty if not detailed in the context.
Be direct, highly technical, and engineering-focused. Use markdown formatting.

Cite the document names, section headings, and page numbers used for your answer.
If diagrams or schematics are referenced in the context, explicitly call them out and explain what they show.
If tables exist, render them as clean markdown tables in your response."""

def classify_intent(query: str) -> str:
    """Classifies user query into retrieval intents for smart Qdrant collection routing."""
    q = query.lower().strip()
    
    visual_kw = ["diagram", "schematic", "figure", "drawing", "illustration", "layout", "view", "picture", "photo", "image", "show me", "what does", "look like"]
    if any(kw in q for kw in visual_kw):
        return "VISUAL_DIAGRAM"
        
    parts_kw = ["part", "spare", "stock", "quantity", "catalog", "part number", "part no", "sp-", "component stock", "replace part"]
    if any(kw in q for kw in parts_kw):
        return "SPARE_PARTS"
        
    logs_kw = ["maintenance log", "service history", "past maintenance", "work order", "log entry", "serviced on", "repaired on", "m101 log", "m102 log", "m103 log"]
    if any(kw in q for kw in logs_kw):
        return "MAINTENANCE_LOGS"
        
    return "TROUBLESHOOTING"


class RagService:
    def __init__(self):
        self.vector_store = container.vector_store
        self.reranker = container.reranker
        self._answer_cache: dict[str, tuple[str, list[dict[str, Any]]]] = {}

    def clear_cache(self):
        self._answer_cache.clear()
        logger.info("RAG Service query answer cache cleared.")

    def search_by_intent(self, query: str, top_k: int = 5, user_id: str = "default_user") -> dict[str, list[dict[str, Any]]]:
        """Performs intent-driven collection routing & query expansion across vector collections."""
        intent = classify_intent(query)
        logger.info(f"RAG RETRIEVAL INTENT: '{intent}' for query: '{query}'")
        
        search_query = query
        if intent == "VISUAL_DIAGRAM":
            collections = ["manuals"]
            search_query = f"{query} diagram schematic figure visual illustration caption layout"
        elif intent == "SPARE_PARTS":
            collections = ["spare_parts", "manuals"]
        elif intent == "MAINTENANCE_LOGS":
            collections = ["maintenance_logs", "manuals"]
        else: # TROUBLESHOOTING
            collections = ["manuals", "error_codes", "sop"]
            
        results = {}
        for coll in collections:
            try:
                filters = {"user_id": user_id} if coll in ["manuals", "sop"] else None
                hits = self.vector_store.search(coll, search_query, top_k=top_k, filters=filters)
                results[coll] = hits
            except Exception as e:
                logger.error(f"Error searching collection {coll}: {e}")
                results[coll] = []
        return results

    def search_all_collections(self, query: str, top_k: int = 5, user_id: str = "default_user") -> dict[str, list[dict[str, Any]]]:
        """Performs targeted hybrid search using intent classification."""
        return self.search_by_intent(query, top_k=top_k, user_id=user_id)

    def get_grounded_answer(self, query: str, top_k_per_coll: int = 15, user_id: str = "default_user") -> tuple[str, list[dict[str, Any]]]:
        cache_key = f"{user_id}:{query.strip().lower()}"
        if cache_key in self._answer_cache:
            logger.info(f"CACHE HIT: Returning cached answer for query: '{query}' under user {user_id}")
            return self._answer_cache[cache_key]

        # 1. Retrieve raw hits from collections with user isolation (Retrieve Top 50 combined)
        all_hits = self.search_all_collections(query, top_k=top_k_per_coll, user_id=user_id)
        
        flat_hits = []
        for coll, hits in all_hits.items():
            for hit in hits:
                flat_hits.append(hit)

        # 2. Rerank using CrossEncoder Reranker (Rerank top 50, keep top 8)
        reranked_hits = self.reranker.rerank(query, flat_hits, top_k=8)
        
        # Filter by minimum score
        filtered_hits = [
            hit for hit in reranked_hits 
            if hit.get("score", 0.0) >= settings.RAG_MIN_RELEVANCE_SCORE
        ]

        if not filtered_hits:
            filtered_hits = reranked_hits[:3]

        if not filtered_hits:
            return (
                "No relevant information was found.",
                []
            )

        # 3. Perform Context Expansion (Fetch siblings, parents, tables, and pictures)
        expanded_hits = self.post_process_layout_aware(filtered_hits)

        # 4. Construct context
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
        synthesized_text = llm_service.synthesize(query, context, SYSTEM_RULES)
        
        # Save to cache
        self._answer_cache[cache_key] = (synthesized_text, expanded_hits)
        
        return synthesized_text, expanded_hits

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
                # 1. Fetch Previous Sibling Chunk to restore context
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
                
                # 2. Fetch Next Sibling Chunk to restore context
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
                        
                # 3. Fetch Parent Heading Chunk if present
                parent_chunk_id = payload.get("parent_chunk")
                if parent_chunk_id and parent_chunk_id not in retrieved_ids:
                    try:
                        parent_hit = self.vector_store.get_point(collection, parent_chunk_id)
                        if parent_hit:
                            parent_hit["score"] = round(hit.get("score", 0.90) - 0.05, 4)
                            processed_hits.append(parent_hit)
                            retrieved_ids.add(parent_chunk_id)
                    except Exception as e:
                        logger.warning(f"Failed to fetch parent chunk {parent_chunk_id}: {e}")

                # 4. Fetch linked Figures or Tables in the same page range
                page = payload.get("page")
                if page:
                    for offset in [-2, -1, 1, 2]:
                        sibling_id = f"{filename}_chunk_{chunk_index + offset}"
                        if sibling_id not in retrieved_ids:
                            try:
                                sib_hit = self.vector_store.get_point(collection, sibling_id)
                                if sib_hit:
                                    sib_payload = sib_hit.get("payload", {})
                                    if sib_payload.get("chunk_type") in ["table", "image"]:
                                        sib_hit["score"] = round(hit.get("score", 0.90) - 0.05, 4)
                                        processed_hits.append(sib_hit)
                                        retrieved_ids.add(sibling_id)
                            except Exception as e:
                                logger.warning(f"Failed to fetch sibling chunk {sibling_id}: {e}")
                                
        return processed_hits

rag_service = RagService()
