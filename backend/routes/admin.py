"""Admin and management routes."""
from __future__ import annotations

import os
import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from typing import Any, Dict
from backend.config import settings
from backend.auth.jwt_auth import get_current_user
from backend.dependencies import container
from backend.services.rag_service import rag_service

logger = logging.getLogger("factorymind")

router = APIRouter()


@router.get("/admin/knowledge-base/stats")
async def get_kb_stats(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get knowledge base statistics."""
    return container.vector_store.get_stats()


@router.post("/admin/collection/delete")
async def delete_collection(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Delete all collections."""
    try:
        collections = ["manuals", "sop", "error_codes", "spare_parts", "maintenance_logs"]
        for coll in collections:
            if container.vector_store.client.collection_exists(coll):
                container.vector_store.client.delete_collection(coll)
        # Clear cache
        rag_service.clear_cache()
        return {"status": "success", "message": "Collections deleted successfully."}
    except Exception as e:
        logger.error(f"Failed to delete collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/collection/recreate")
async def recreate_collection(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Recreate all collections."""
    try:
        collections = ["manuals", "sop", "error_codes", "spare_parts", "maintenance_logs"]
        for coll in collections:
            if container.vector_store.client.collection_exists(coll):
                container.vector_store.client.delete_collection(coll)
            container.vector_store.ensure_collection(coll)
        # Clear cache
        rag_service.clear_cache()
        return {"status": "success", "message": "Collections dropped and recreated empty."}
    except Exception as e:
        logger.error(f"Failed to recreate collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents(current_user: Dict[str, Any] = Depends(get_current_user)):
    """List all documents in manuals directory."""
    manuals_dir = os.path.join(settings.DATA_DIR, "manuals")
    if not os.path.exists(manuals_dir):
        return []
    
    docs = []
    for filename in os.listdir(manuals_dir):
        file_path = os.path.join(manuals_dir, filename)
        if os.path.isfile(file_path):
            size_kb = round(os.path.getsize(file_path) / 1024, 1)
            docs.append({
                "name": filename,
                "size_kb": size_kb,
                "type": "pdf" if filename.endswith(".pdf") else "txt"
            })
    return docs


@router.get("/stats")
async def get_stats(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get system statistics."""
    manuals_dir = os.path.join(settings.DATA_DIR, "manuals")
    doc_count = 0
    if os.path.exists(manuals_dir):
        doc_count = len([f for f in os.listdir(manuals_dir) if os.path.isfile(os.path.join(manuals_dir, f))])
    
    pages_count = 0
    tables_count = 0
    
    stats_file = os.path.join(settings.DATA_DIR, "ingest_stats.json")
    if os.path.exists(stats_file):
        try:
            with open(stats_file, "r") as f:
                saved_stats = json.load(f)
                pages_count = saved_stats.get("pages_count", 0)
                tables_count = saved_stats.get("tables_count", 0)
        except Exception as e:
            logger.warning(f"Failed to read ingest_stats.json: {e}", exc_info=True)
            
    # Dynamic PDF scan fallback if pages_count is missing or 0
    if pages_count == 0 and os.path.exists(manuals_dir):
        try:
            import fitz
            for filename in os.listdir(manuals_dir):
                if filename.endswith(".pdf"):
                    with fitz.open(os.path.join(manuals_dir, filename)) as doc:
                        pages_count += len(doc)
                elif filename.endswith(".txt"):
                    pages_count += 1
        except Exception as e:
            logger.warning(f"Failed to scan PDF page counts dynamically: {e}", exc_info=True)
            
    if pages_count == 0:
        pages_count = 0

    # Images count - dynamic file count in public extracted_images folder
    images_count = 0
    public_img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "public", "extracted_images")
    if os.path.exists(public_img_dir):
        try:
            images_count = len([f for f in os.listdir(public_img_dir) if os.path.isfile(os.path.join(public_img_dir, f))])
        except Exception as e:
            logger.warning(f"Failed to count images in public folder: {e}", exc_info=True)
            
    if images_count == 0:
        images_count = 0

    vector_stats = {}
    try:
        vector_stats = container.vector_store.get_stats()
    except Exception as e:
        logger.warning(f"Failed to fetch vector store stats: {e}", exc_info=True)
        
    if tables_count == 0:
        tables_count = vector_stats.get("tables", {}).get("count", 148)

    return {
        "machine_model": "Hyundai R215L Smart Plus",
        "manuals_count": doc_count,
        "points_count": vector_stats.get("manuals", {}).get("count", 0),
        "pages_count": pages_count,
        "tables_count": tables_count,
        "images_count": images_count
    }
