"""Background task for document ingestion."""
from __future__ import annotations

import os
import logging
import datetime
from typing import Any, Dict
from backend.config import settings
from backend.dependencies import container
from backend.services.rag_service import rag_service

logger = logging.getLogger("factorymind")

# In-memory Job Store
_jobs: Dict[str, Dict[str, Any]] = {}


def get_jobs() -> Dict[str, Dict[str, Any]]:
    """Get the jobs dictionary."""
    return _jobs


def run_ingestion_task(job_id: str, pipeline_name: str, user_id: str = "default_user"):
    """Execute ingestion pipeline as background task."""
    _jobs[job_id] = {
        "status": "processing",
        "progress": 10,
        "message": "Initializing pipeline...",
        "started_at": datetime.datetime.utcnow().isoformat() + "Z"
    }
    
    try:
        vector_store = container.vector_store
        data_dir = settings.DATA_DIR
        
        if pipeline_name == "manuals":
            _jobs[job_id]["progress"] = 30
            _jobs[job_id]["message"] = "Processing manuals files..."
            from ingestion.ingest_manuals import run_manuals_ingestion
            count = run_manuals_ingestion(vector_store, os.path.join(data_dir, "manuals"), "manuals", user_id=user_id)
            
            jobs[job_id]["progress"] = 70
            jobs[job_id]["message"] = "Processing SOP files..."
            sop_count = run_manuals_ingestion(vector_store, os.path.join(data_dir, "sop"), "sop", user_id=user_id)
            
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["progress"] = 100
            _jobs[job_id]["message"] = f"Ingested {count} manuals and {sop_count} SOP segments successfully."
            
        elif pipeline_name == "maintenance_logs":
            _jobs[job_id]["progress"] = 50
            _jobs[job_id]["message"] = "Parsing CSV logs..."
            from ingestion.ingest_logs import run_logs_ingestion
            count = run_logs_ingestion(vector_store, os.path.join(data_dir, "maintenance_logs", "maintenance_logs.csv"))
            
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["progress"] = 100
            _jobs[job_id]["message"] = f"Ingested {count} maintenance logs."
            
        elif pipeline_name == "error_codes":
            _jobs[job_id]["progress"] = 50
            _jobs[job_id]["message"] = "Parsing error codes JSON..."
            from ingestion.ingest_errors import run_errors_ingestion
            count = run_errors_ingestion(vector_store, os.path.join(data_dir, "error_codes", "error_codes.json"))
            
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["progress"] = 100
            _jobs[job_id]["message"] = f"Ingested {count} error codes."
            
        elif pipeline_name == "spare_parts":
            _jobs[job_id]["progress"] = 50
            _jobs[job_id]["message"] = "Parsing parts CSV..."
            from ingestion.ingest_parts import run_parts_ingestion
            count = run_parts_ingestion(vector_store, os.path.join(data_dir, "spare_parts", "spare_parts.csv"))
            
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["progress"] = 100
            _jobs[job_id]["message"] = f"Ingested {count} parts listings."
            
        elif pipeline_name == "graph":
            _jobs[job_id]["progress"] = 30
            _jobs[job_id]["message"] = "Building Graph connections..."
            from ingestion.ingest_graph import run_graph_ingestion
            
            # Execute graph ingestion
            count = run_graph_ingestion()
            
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["progress"] = 100
            _jobs[job_id]["message"] = f"Extracted and loaded {count} component-failure-repair relationships into graph store."
            
        elif pipeline_name == "prediction":
            _jobs[job_id]["progress"] = 20
            _jobs[job_id]["message"] = "Starting model training pipeline..."
            from prediction.train import main as train_main
            
            # Execute model training
            train_main()
            
            # Reload prediction engine in memory
            from prediction.infer import prediction_engine
            prediction_engine.load_model()
            
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["progress"] = 100
            _jobs[job_id]["message"] = "Trained XGBoost models (binary + multiclass) and reloaded prediction engine in memory."
            
        else:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        # Clear RAG cache for fresh results
        try:
            rag_service.clear_cache()
        except Exception as ce:
            logger.warning(f"Could not clear RAG cache: {ce}")
            
    except Exception as e:
        logger.exception(f"Error executing ingestion {pipeline_name}")
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["message"] = f"Error: {str(e)}"
        _jobs[job_id]["progress"] = 100
