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


@router.get("/debug/image")
async def debug_image(
    image_path: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """STEP 8: Diagnostic endpoint to check image accessibility."""
    import os
    logger.info(f"=== STEP 8: IMAGE DIAGNOSTIC ===")
    logger.info(f"Input image_path: {image_path}")

    # Try multiple path resolutions
    possible_paths = [
        image_path,  # As-is
        os.path.join("frontend", "public", image_path.lstrip("/")),  # Relative to project root
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "frontend", "public", image_path.lstrip("/")),  # Absolute from backend
    ]

    result = {
        "input_path": image_path,
        "possible_paths": possible_paths,
        "exists": False,
        "absolute_path": None,
        "file_size": 0,
        "mime": None,
        "public_url": None,
        "http_url": None,
        "browser_accessible": False
    }

    for path in possible_paths:
        if os.path.exists(path):
            result["exists"] = True
            result["absolute_path"] = os.path.abspath(path)
            result["file_size"] = os.path.getsize(path)
            
            # Determine MIME type
            import mimetypes
            mime, _ = mimetypes.guess_type(path)
            result["mime"] = mime or "unknown"
            
            # Generate public URL (browser-accessible)
            # Extract filename and create /extracted_images/<filename> format
            filename = os.path.basename(path)
            result["public_url"] = f"/extracted_images/{filename}"
            result["http_url"] = f"http://localhost:3000/extracted_images/{filename}"
            result["browser_accessible"] = True
            
            logger.info(f"Image FOUND: {path}")
            logger.info(f"  Size: {result['file_size']} bytes")
            logger.info(f"  MIME: {result['mime']}")
            logger.info(f"  Public URL: {result['public_url']}")
            break
    
    if not result["exists"]:
        logger.error(f"Image NOT FOUND: Tried {len(possible_paths)} paths")

    return result


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
        
        # Collect all retrieved images from citations for multi-image response
        logger.info(f"=== STEP 1: INSPECT API RESPONSE - IMAGE COLLECTION ===")
        logger.info(f"Total citations to inspect: {len(citations)}")
        
        retrieved_images = []
        for doc in citations:
            payload = doc.get("payload", {}) if isinstance(doc.get("payload"), dict) else {}
            img_path = payload.get("image_path") or payload.get("image_url")
            
            # Also extract from text if chunk_type was merged during layout awareness
            if not img_path and "[Image Path]:" in str(doc.get("text", "")):
                import re
                match = re.search(r"\[Image Path\]:\s*(\S+)", str(doc.get("text", "")))
                if match:
                    img_path = match.group(1)

            if img_path and img_path not in [img["image_path"] for img in retrieved_images]:
                image_obj = {
                    "image_path": img_path,
                    "caption": payload.get("caption") or "Technical Diagram",
                    "page": payload.get("page", "N/A"),
                    "document": payload.get("document_name", "Manual"),
                    "figure_number": payload.get("figure_number", ""),
                    "confidence": doc.get("score", 0.90)
                }
                retrieved_images.append(image_obj)
                logger.info(f"Found image object: {image_obj}")
            else:
                logger.info(f"Document has no image_path - payload keys: {list(payload.keys())}")

        logger.info(f"Total retrieved_images: {len(retrieved_images)}")
        for i, img in enumerate(retrieved_images):
            logger.info(f"Image {i+1}: path='{img['image_path']}', caption='{img['caption']}', page={img['page']}")

        # STEP 2: Verify Image Exists
        logger.info(f"=== STEP 2: VERIFY IMAGE EXISTS ON DISK ===")
        import os
        for i, img in enumerate(retrieved_images):
            img_path = img["image_path"]
            # Try multiple path resolutions
            possible_paths = [
                img_path,  # As-is
                os.path.join("frontend", "public", img_path.lstrip("/")),  # Relative to project root
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "frontend", "public", img_path.lstrip("/")),  # Absolute from backend
            ]
            
            found = False
            for path in possible_paths:
                if os.path.exists(path):
                    file_size = os.path.getsize(path)
                    logger.info(f"Image {i+1} EXISTS: {path} (size: {file_size} bytes)")
                    found = True
                    break
            
            if not found:
                logger.error(f"Image {i+1} NOT FOUND: Tried paths: {possible_paths}")

        # STEP 3: Fix Image Path Generation - Convert to browser-accessible URLs
        logger.info(f"=== STEP 3: CONVERT IMAGE PATHS TO BROWSER-ACCESSIBLE URLs ===")
        for i, img in enumerate(retrieved_images):
            original_path = img["image_path"]
            
            # Normalize Windows backslashes to forward slashes
            normalized_path = original_path.replace("\\", "/")
            
            # Remove common prefixes
            prefixes_to_remove = [
                "frontend/public/",
                "/frontend/public/",
                "extracted_images/",
                "/extracted_images/"
            ]
            
            for prefix in prefixes_to_remove:
                if normalized_path.startswith(prefix):
                    normalized_path = normalized_path[len(prefix):]
                    break
            
            # Ensure it starts with extracted_images/
            if not normalized_path.startswith("extracted_images/"):
                normalized_path = f"extracted_images/{normalized_path}"
            
            # Remove leading slash if present (will be added by browser)
            if normalized_path.startswith("/"):
                normalized_path = normalized_path[1:]
            
            # Final URL format: /extracted_images/<filename>
            final_url = f"/{normalized_path}"
            
            img["image_path"] = final_url  # Replace with browser-accessible URL
            logger.info(f"Image {i+1}: '{original_path}' → '{final_url}'")

        image_url = retrieved_images[0]["image_path"] if retrieved_images else None
        image_description = None
        if image_url and has_visual_intent(query):
            try:
                from backend.services.vision_service import vision_service
                import os
                abs_img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "frontend", "public", image_url.lstrip("/"))
                image_description = vision_service.describe_image(abs_img_path, prompt="Describe this technical diagram or figure in detail, including labels, components, and relationships.")
            except Exception as e:
                logger.warning(f"Failed to generate image description: {e}")

        # Attach retrieved_images to evidence bundle
        evidence_bundle["retrieved_images"] = retrieved_images

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
            "retrieved_images": retrieved_images,
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
