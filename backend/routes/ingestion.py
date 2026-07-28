"""Document ingestion routes."""
from __future__ import annotations

import uuid
import datetime
import logging
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from typing import Any, Dict
from backend.auth.jwt_auth import get_current_user
from backend.tasks.ingestion import run_ingestion_task, get_jobs

logger = logging.getLogger("factorymind")

router = APIRouter()


@router.post("/ingest/{pipeline_name}")
async def trigger_ingest(
    pipeline_name: str,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Trigger document ingestion pipeline."""
    valid_pipelines = ["manuals", "maintenance_logs", "error_codes", "spare_parts", "graph", "prediction"]
    if pipeline_name not in valid_pipelines:
        raise HTTPException(status_code=400, detail="Invalid pipeline name")
        
    jobs = get_jobs()
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "pipeline": pipeline_name,
        "status": "queued",
        "progress": 0,
        "message": "Task queued.",
        "started_at": datetime.datetime.utcnow().isoformat() + "Z"
    }
    
    background_tasks.add_task(run_ingestion_task, job_id, pipeline_name, current_user.get("uid"))
    return {"job_id": job_id, "status": "queued"}


@router.get("/ingest/status/{job_id}")
async def get_ingest_status(
    job_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get ingestion job status."""
    jobs = get_jobs()
    job_data = jobs.get(job_id)
    if job_data is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_data
