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


@router.post("/query")
async def run_query(
    req: QueryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Execute query using multi-agent orchestrator."""
    query = req.query
    machine_id = req.machine_id
    
    logger.info(f"Received query: '{query}' for machine: {machine_id} by user: {current_user.get('uid')}")
    
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
