"""Query and RAG routes."""
from __future__ import annotations

import uuid
import datetime
import logging
from fastapi import APIRouter, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Any, Dict
from backend.models.schemas import QueryRequest
from backend.auth.jwt_auth import get_current_user
from backend.telemetry import get_telemetry
from agents.graph import agent_orchestrator, has_visual_intent
from backend.services.rag_service import rag_service

logger = logging.getLogger("factorymind")

router = APIRouter()

# In-memory store for generated reports
LAST_ANSWERS: Dict[str, Dict[str, Any]] = {}


@router.get("/debug/query")
async def debug_query(
    query: str,
    machine_id: str = "M101",
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Debug endpoint to trace full pipeline without LLM synthesis."""
    logger.info(f"=== DEBUG PIPELINE TRACE ===")
    logger.info(f"QUERY: '{query}'")
    logger.info(f"MACHINE ID: {machine_id}")

    from agents.graph import agent_orchestrator
    from backend.services.query_planner import query_planner
    from backend.services.rag_service import rag_service

    # Step 1: Query Planning
    plan = query_planner.plan(query)
    planning_info = {
        "intent": plan.intent,
        "rewritten_query": plan.rewritten_query,
        "target_collections": plan.target_collections,
        "is_conversational": plan.is_conversational,
        "requires_visual": plan.requires_visual,
        "metadata_filters": plan.metadata_filters
    }

    # Step 2: Retrieval
    retrieval_results = rag_service.search_all_collections(query, top_k=10, user_id=current_user.get("uid"))
    retrieval_info = {
        "collections_searched": list(retrieval_results.keys()),
        "total_hits": sum(len(hits) for hits in retrieval_results.values()),
        "hits_by_collection": {coll: len(hits) for coll, hits in retrieval_results.items()},
        "sample_hits": []
    }

    # Add sample hits from each collection
    for coll, hits in retrieval_results.items():
        for hit in hits[:2]:  # Top 2 hits per collection
            retrieval_info["sample_hits"].append({
                "collection": coll,
                "score": hit.get("score"),
                "title": hit.get("title", "")[:50],
                "document": hit.get("payload", {}).get("document_name", "Unknown"),
                "page": hit.get("payload", {}).get("page", "N/A")
            })

    return {
        "debug_trace": {
            "query_planning": planning_info,
            "retrieval": retrieval_info
        }
    }


@router.post("/query")
async def run_query(
    req: QueryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Execute query using multi-agent orchestrator."""
    query = req.query
    machine_id = req.machine_id

    logger.info(f"=== PIPELINE TRACE START ===")
    logger.info(f"INCOMING QUERY: '{query}'")
    logger.info(f"MACHINE ID: {machine_id}")
    logger.info(f"USER ID: {current_user.get('uid')}")
    
    try:
        # Fetch active telemetry values for this machine
        telemetry = get_telemetry(machine_id)
        
        # Run the multi-agent supervisor graph with user isolation
        state = agent_orchestrator.run(query, machine_id, telemetry, user_id=current_user.get("uid"))
        
        # Construct the evidence bundle from active agent state
        citations = state.get("retrieved_documents", [])
        confidence_breakdown = state.get("confidence_breakdown", {"overall": 75, "retrieval": 75, "graph": 75, "evidence": 75, "answer": 75})
        
        evidence_bundle = {
            "citations": [
                {
                    "id": c.get("id"),
                    "title": c.get("title"),
                    "text": c.get("text"),
                    "score": c.get("score"),
                    "source_type": c.get("source_type", "unknown"),
                    "payload": c.get("payload", {})
                } for c in citations
            ],
            "sensor_values": state.get("sensor_values") or telemetry,
            "kg_path": state.get("graph_path") or [],
            "confidence_score": 1.0 if confidence_breakdown.get("overall") == "High" else 0.5 if confidence_breakdown.get("overall") == "Medium" else 0.25,
            "confidence_breakdown": confidence_breakdown,
            "llm_prompt": state.get("llm_prompt", "Prompt details not logged by agent.")
        }
        
        # Find top image_url if visual intent is detected
        image_url = None
        image_description = None
        if has_visual_intent(query):
            for doc in citations:
                payload = doc.get("payload", {}) if doc.get("payload") else {}
                if payload.get("image_path"):
                    image_url = payload.get("image_path")
                    # Generate image description using vision service
                    try:
                        from backend.services.vision_service import vision_service
                        import os
                        # Convert relative path to absolute path
                        image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "frontend", "public", image_url.lstrip("/"))
                        image_description = vision_service.describe_image(image_path, prompt="Describe this technical diagram or figure in detail, including labels, components, and relationships.")
                    except Exception as e:
                        logger.warning(f"Failed to generate image description: {e}")
                    break

        # Generate query ID and cache for PDF download
        query_id = str(uuid.uuid4())
        LAST_ANSWERS[query_id] = {
            "query": query,
            "machine_id": machine_id,
            "answer": state["final_answer"],
            "evidence": evidence_bundle,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return {
            "query_id": query_id,
            "answer": state["final_answer"],
            "evidence": evidence_bundle,
            "image_url": image_url,
            "image_description": image_description
        }
    except Exception as e:
        logger.exception(f"Unhandled exception in /query route: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "pipeline_failure",
                "detail": f"Agent query pipeline execution failed: {str(e)}"
            }
        )


def get_last_answer(query_id: str) -> Dict[str, Any] | None:
    """Retrieve cached answer by query ID."""
    return LAST_ANSWERS.get(query_id)
