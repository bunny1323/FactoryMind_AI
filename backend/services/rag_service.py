from __future__ import annotations

import logging
import asyncio
from typing import Any, Dict, List, Tuple
from backend.dependencies import container
from backend.services.llm_service import llm_service
from backend.services.vision_service import vision_service
from backend.services.language_service import language_service
from backend.services.conversation_memory import conversation_memory
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
    
    # Safety intent (HIGH PRIORITY - must come before error code)
    safety_kw = ["safety", "hazard", "danger", "warning", "caution", "risk", "protect", "guard", "personal protective equipment", "ppe", "icon", "symbol", "sign", "alert", "precaution"]
    if any(kw in q for kw in safety_kw):
        return "SAFETY"
    
    # Greeting intent
    greeting_kw = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings"]
    if any(kw in q for kw in greeting_kw):
        return "GREETING"
    
    # Visual/Diagram intent
    visual_kw = ["diagram", "schematic", "figure", "drawing", "illustration", "layout", "view", "picture", "photo", "image", "show me", "what does", "look like", "visual", "display", "chart", "graph"]
    if any(kw in q for kw in visual_kw):
        return "VISUAL_DIAGRAM"
    
    # Specification intent
    spec_kw = ["specification", "spec", "specs", "technical spec", "dimensions", "capacity", "weight", "performance", "parameters", "rating", "tolerance"]
    if any(kw in q for kw in spec_kw):
        return "SPECIFICATION"
    
    # Manual lookup intent
    manual_kw = ["manual", "handbook", "guide", "documentation", "instruction", "procedure", "how to", "steps", "operation", "operating"]
    if any(kw in q for kw in manual_kw):
        return "MANUAL_LOOKUP"
    
    # Maintenance intent
    maintenance_kw = ["maintenance", "service", "repair", "fix", "replace", "adjust", "calibrate", "lubricate", "inspect", "check", "overhaul"]
    if any(kw in q for kw in maintenance_kw):
        return "MAINTENANCE"
    
    # Prediction intent
    prediction_kw = ["predict", "forecast", "future", "upcoming", "expected", "likely", "trend", "projection", "anticipate"]
    if any(kw in q for kw in prediction_kw):
        return "PREDICTION"
    
    # Error code intent (specific to error codes, not general errors)
    error_kw = ["error code", "error code:", "e-", "fault code", "diagnostic code", "trouble code"]
    if any(kw in q for kw in error_kw):
        return "ERROR_CODE"
    
    # Spare parts intent
    parts_kw = ["part", "spare", "stock", "quantity", "catalog", "part number", "part no", "sp-", "component stock", "replace part", "inventory"]
    if any(kw in q for kw in parts_kw):
        return "SPARE_PARTS"
    
    # Torque intent
    torque_kw = ["torque", "tightening", "bolt torque", "nut torque", "nm", "ft-lb", "tighten"]
    if any(kw in q for kw in torque_kw):
        return "TORQUE"
    
    # Hydraulic intent
    hydraulic_kw = ["hydraulic", "hydraulics", "fluid", "pump", "cylinder", "valve", "pressure", "flow", "hose"]
    if any(kw in q for kw in hydraulic_kw):
        return "HYDRAULIC"
    
    # Electrical intent
    electrical_kw = ["electrical", "electric", "wiring", "circuit", "voltage", "current", "battery", "alternator", "starter", "relay"]
    if any(kw in q for kw in electrical_kw):
        return "ELECTRICAL"
    
    # Default to troubleshooting
    return "TROUBLESHOOTING"


class RagService:
    def __init__(self):
        self.vector_store = container.vector_store
        self.reranker = container.reranker
        self._answer_cache: dict[str, tuple[str, list[dict[str, Any]]]] = {}

    def clear_cache(self):
        self._answer_cache.clear()
        logger.info("RAG Service query answer cache cleared.")

    def search_by_intent(self, query: str, top_k: int = 10, user_id: str = "default_user") -> dict[str, list[dict[str, Any]]]:
        """Performs intent-driven collection routing with optimized retrieval (Top 30 per collection)."""
        intent = classify_intent(query)
        logger.info(f"RAG RETRIEVAL INTENT: '{intent}' for query: '{query}'")
        
        search_query = query
        
        # Route collections based on intent
        if intent == "GREETING":
            return {"manuals": []}  # No search needed for greetings
        
        elif intent == "SAFETY":
            # Safety queries should ONLY search manuals, not error codes
            collections = ["manuals"]
            search_query = f"{query} safety hazard warning caution danger risk protection ppe icon symbol"
        
        elif intent == "VISUAL_DIAGRAM":
            collections = ["manuals"]
            search_query = f"{query} diagram schematic figure visual illustration caption layout"
        
        elif intent == "SPECIFICATION":
            collections = ["manuals", "sop"]
            search_query = f"{query} specification spec technical parameter dimension capacity weight performance"
        
        elif intent == "MANUAL_LOOKUP":
            collections = ["manuals", "sop"]
            search_query = f"{query} manual handbook guide documentation instruction procedure operation"
        
        elif intent == "MAINTENANCE":
            collections = ["manuals", "sop", "maintenance_logs"]
            search_query = f"{query} maintenance service repair fix replace adjust calibrate lubricate inspect"
        
        elif intent == "PREDICTION":
            collections = ["manuals", "maintenance_logs"]
            search_query = f"{query} predict forecast future upcoming expected trend projection anticipate"
        
        elif intent == "ERROR_CODE":
            # Only search error_codes for specific error code queries
            collections = ["error_codes", "manuals"]
            search_query = f"{query} error code fault failure malfunction troubleshoot diagnose"
        
        elif intent == "SPARE_PARTS":
            collections = ["spare_parts", "manuals"]
            search_query = f"{query} part spare stock quantity catalog part number inventory component"
        
        elif intent == "TORQUE":
            collections = ["manuals", "sop"]
            search_query = f"{query} torque tightening bolt nut nm ft-lb specification"
        
        elif intent == "HYDRAULIC":
            collections = ["manuals", "sop"]
            search_query = f"{query} hydraulic hydraulics fluid pump cylinder valve pressure flow hose"
        
        elif intent == "ELECTRICAL":
            collections = ["manuals", "sop"]
            search_query = f"{query} electrical electric wiring circuit voltage current battery alternator starter relay"
        
        else:  # TROUBLESHOOTING (default)
            collections = ["manuals", "error_codes", "sop"]
            search_query = query
            
        # Synchronous retrieval across collections
        results = {}
        for coll in collections:
            try:
                filters = {"user_id": user_id} if coll in ["manuals", "sop"] else None
                # Retrieve Top 30 per collection for better RRF input
                hits = self.vector_store.search(coll, search_query, top_k=30, filters=filters)
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

        # Add user query to conversation memory
        conversation_memory.add_message(user_id, "user", query)

        # Detect query language for multilingual response
        detected_lang = language_service.detect_language(query)
        language_instruction = language_service.get_system_prompt_language(detected_lang)
        logger.info(f"Query language: {language_service.LANGUAGE_NAMES.get(detected_lang, detected_lang)}")

        # Resolve follow-up references
        resolved_query = conversation_memory.resolve_references(query, user_id)
        if resolved_query != query:
            logger.info(f"Resolved follow-up query: '{query}' -> '{resolved_query}'")
            query = resolved_query

        # Get conversation context for LLM
        conversation_context = conversation_memory.format_conversation_context(user_id)

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
            no_result_msg = "No relevant information was found."
            if detected_lang != "en":
                no_result_msg = f"{language_instruction} {no_result_msg}"
            # Add assistant response to conversation memory
            conversation_memory.add_message(user_id, "assistant", no_result_msg)
            return (no_result_msg, [])

        # 3. Perform Context Expansion (Fetch siblings, parents, tables, and pictures)
        expanded_hits = self.post_process_layout_aware(filtered_hits)

        # 4. Construct context
        context_blocks = []
        if conversation_context:
            context_blocks.append(conversation_context)
        
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

        # 5. Synthesize final answer using LLM with language instruction
        enhanced_system_rules = f"{SYSTEM_RULES}\n\n{language_instruction}"
        synthesized_text = llm_service.synthesize(query, context, enhanced_system_rules)
        
        # Add assistant response to conversation memory
        conversation_memory.add_message(user_id, "assistant", synthesized_text)
        
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
