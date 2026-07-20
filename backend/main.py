from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import logging
import uuid
import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from fastapi.responses import FileResponse, StreamingResponse
import io

from backend.config import settings
from backend.dependencies import container
from backend.services.rag_service import rag_service
from prediction.router import router as prediction_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("factorymind")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="FactoryMind AI: Predictive and RAG Industrial Maintenance Intelligence Platform for the Hyundai R215L Excavator."
)

app.include_router(prediction_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import jwt
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Query

# In-memory Job Store
jobs: dict[str, dict[str, Any]] = {}

# JWT Helpers & Security Dependency
security = HTTPBearer(auto_error=False)

def create_access_token(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)

from backend.auth.jwt_auth import get_current_user

# Mock machine telemetry store
TELEMETRY_DATA = {
    "M101": {
        "air_temperature": 298.2,     # Kelvin
        "process_temperature": 308.6, # Kelvin
        "rotational_speed": 1850,     # RPM
        "torque": 45.2,               # Nm
        "tool_wear": 120,             # Minutes
        "vibration": 0.22             # mm peak-to-peak
    },
    "M102": {
        "air_temperature": 296.5,
        "process_temperature": 307.2,
        "rotational_speed": 1420,
        "torque": 38.1,
        "tool_wear": 45,
        "vibration": 0.04
    },
    "M103": {
        "air_temperature": 299.1,
        "process_temperature": 309.8,
        "rotational_speed": 1600,
        "torque": 41.5,
        "tool_wear": 80,
        "vibration": 0.08
    }
}

# --- Pydantic Schemas ---
class QueryRequest(BaseModel):
    query: str
    machine_id: str = "M101"

class RetrieveDebugRequest(BaseModel):
    query: str
    top_k: Optional[int] = 8

class PredictRequest(BaseModel):
    air_temp: float
    process_temp: float
    rotational_speed: float
    torque: float
    tool_wear: float

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: str
    role: str = "user"

# Static mock users for local session authentication (role switcher)
MOCK_USERS = {
    "onepiece": {"username": "onepiece", "password": "luffy", "display_name": "Luffy", "role": "admin"},
    "zoro": {"username": "zoro", "password": "swordsman", "display_name": "Zoro", "role": "user"}
}

# --- Background Task Workers ---
def run_ingestion_task(job_id: str, pipeline_name: str, user_id: str = "default_user"):
    jobs[job_id] = {
        "status": "processing",
        "progress": 10,
        "message": "Initializing pipeline...",
        "started_at": datetime.datetime.utcnow().isoformat() + "Z"
    }
    
    try:
        vector_store = container.vector_store
        data_dir = settings.DATA_DIR
        
        if pipeline_name == "manuals":
            jobs[job_id]["progress"] = 30
            jobs[job_id]["message"] = "Processing manuals files..."
            from ingestion.ingest_manuals import run_manuals_ingestion
            count = run_manuals_ingestion(vector_store, os.path.join(data_dir, "manuals"), "manuals", user_id=user_id)
            
            jobs[job_id]["progress"] = 70
            jobs[job_id]["message"] = "Processing SOP files..."
            sop_count = run_manuals_ingestion(vector_store, os.path.join(data_dir, "sop"), "sop", user_id=user_id)
            
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["message"] = f"Ingested {count} manuals and {sop_count} SOP segments successfully."
            
        elif pipeline_name == "maintenance_logs":
            jobs[job_id]["progress"] = 50
            jobs[job_id]["message"] = "Parsing CSV logs..."
            from ingestion.ingest_logs import run_logs_ingestion
            count = run_logs_ingestion(vector_store, os.path.join(data_dir, "maintenance_logs", "maintenance_logs.csv"))
            
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["message"] = f"Ingested {count} maintenance logs."
            
        elif pipeline_name == "error_codes":
            jobs[job_id]["progress"] = 50
            jobs[job_id]["message"] = "Parsing error codes JSON..."
            from ingestion.ingest_errors import run_errors_ingestion
            count = run_errors_ingestion(vector_store, os.path.join(data_dir, "error_codes", "error_codes.json"))
            
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["message"] = f"Ingested {count} error codes."
            
        elif pipeline_name == "spare_parts":
            jobs[job_id]["progress"] = 50
            jobs[job_id]["message"] = "Parsing parts CSV..."
            from ingestion.ingest_parts import run_parts_ingestion
            count = run_parts_ingestion(vector_store, os.path.join(data_dir, "spare_parts", "spare_parts.csv"))
            
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["message"] = f"Ingested {count} parts listings."
            
        elif pipeline_name == "graph":
            jobs[job_id]["progress"] = 30
            jobs[job_id]["message"] = "Building Graph connections..."
            from ingestion.ingest_graph import run_graph_ingestion
            
            # Execute graph ingestion
            count = run_graph_ingestion()
            
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["message"] = f"Extracted and loaded {count} component-failure-repair relationships into graph store."
            
        elif pipeline_name == "prediction":
            jobs[job_id]["progress"] = 20
            jobs[job_id]["message"] = "Starting model training pipeline..."
            from prediction.train import main as train_main
            
            # Execute model training
            train_main()
            
            # Reload prediction engine in memory
            prediction_engine.load_model()
            
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["message"] = "Trained XGBoost models (binary + multiclass) and reloaded prediction engine in memory."
            
        else:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        # Clear RAG cache for fresh results
        try:
            rag_service.clear_cache()
        except Exception as ce:
            logger.warning(f"Could not clear RAG cache: {ce}")
            
    except Exception as e:
        logger.exception(f"Error executing ingestion {pipeline_name}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = f"Error: {str(e)}"
        jobs[job_id]["progress"] = 100

# Import os for directories
import os

# Create data subdirectories if not present
os.makedirs(os.path.join(settings.DATA_DIR, "manuals"), exist_ok=True)
os.makedirs(os.path.join(settings.DATA_DIR, "sop"), exist_ok=True)
os.makedirs(os.path.join(settings.DATA_DIR, "maintenance_logs"), exist_ok=True)
os.makedirs(os.path.join(settings.DATA_DIR, "spare_parts"), exist_ok=True)
os.makedirs(os.path.join(settings.DATA_DIR, "error_codes"), exist_ok=True)

# --- Routes ---

@app.post("/auth/login")
async def login(req: LoginRequest):
    user = MOCK_USERS.get(req.username.lower())
    if not user or user["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
        
    token = create_access_token(user["username"], user["role"])
    return {
        "token": token,
        "username": user["display_name"],
        "role": user["role"]
    }

@app.post("/auth/register")
async def register(req: RegisterRequest):
    token = create_access_token(req.username, req.role)
    return {
        "token": token,
        "username": req.display_name,
        "role": req.role
    }

@app.post("/ingest/{pipeline_name}")
async def trigger_ingest(
    pipeline_name: str, 
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    valid_pipelines = ["manuals", "maintenance_logs", "error_codes", "spare_parts", "graph", "prediction"]
    if pipeline_name not in valid_pipelines:
        raise HTTPException(status_code=400, detail="Invalid pipeline name")
        
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

@app.get("/ingest/status/{job_id}")
async def get_ingest_status(job_id: str, current_user: dict = Depends(get_current_user)):
    job_data = jobs.get(job_id)
    if job_data is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_data

@app.get("/admin/knowledge-base/stats")
async def get_kb_stats(current_user: dict = Depends(get_current_user)):
    return container.vector_store.get_stats()

@app.get("/machines")
async def list_machines():
    return ["M101", "M102", "M103"]

@app.get("/machines/{id}/graph")
async def get_machine_graph(id: str, current_user: dict = Depends(get_current_user)):
    return graph_client.get_path_for_query("", id)

@app.get("/machines/{id}/history")
async def get_machine_history(id: str):
    from backend.db import get_machine_history_logs
    logs = get_machine_history_logs(id)
    return {
        "machine_id": id,
        "logs": logs
    }

from prediction.infer import prediction_engine

@app.post("/predict")
async def predict_failure(req: PredictRequest):
    try:
        res = prediction_engine.predict(
            air_temp=req.air_temp,
            process_temp=req.process_temp,
            rotational_speed=req.rotational_speed,
            torque=req.torque,
            tool_wear=req.tool_wear
        )
        return res
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from agents.graph import agent_orchestrator

# In-memory store for generated reports
LAST_ANSWERS: dict[str, dict[str, Any]] = {}

@app.post("/query")
async def run_query(req: QueryRequest, current_user: dict = Depends(get_current_user)):
    query = req.query
    machine_id = req.machine_id
    
    logger.info(f"Received query: '{query}' for machine: {machine_id} by user: {current_user.get('uid')}")
    
    # Fetch active telemetry values for this machine
    telemetry = TELEMETRY_DATA.get(machine_id, TELEMETRY_DATA["M101"])
    
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
        "evidence": evidence_bundle
    }

@app.get("/reports/{query_id}/pdf")
async def download_report(query_id: str):
    # Fetch report data
    report_data = LAST_ANSWERS.get(query_id)
    if not report_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Maintenance report not found or expired."
        )

    # Generate styled PDF report using pymupdf (fitz)
    import fitz
    
    doc = fitz.open()
    page = doc.new_page()
    
    # Page Border
    page.draw_rect((20, 20, 575, 820), color=(0.4, 0.2, 0.1), width=1)
    
    # Header Banner
    page.draw_rect((30, 30, 565, 75), color=(0.95, 0.92, 0.88), fill=(0.95, 0.92, 0.88))
    page.insert_text((45, 58), "FACTORYMIND AI - EXPLAINABLE INDUSTRIAL COPILOT", fontsize=14, fontname="helv-bold", color=(0.4, 0.2, 0.1))
    
    # Title
    page.insert_text((45, 110), "MAINTENANCE DISPATCH & PROCEDURAL REPORT", fontsize=11, fontname="helv-bold", color=(0.2, 0.2, 0.2))
    page.draw_line((45, 115), (550, 115), color=(0.8, 0.8, 0.8), width=0.5)
    
    # Metadata block
    y = 135
    page.insert_text((45, y), f"Report UUID: {query_id}", fontsize=8, fontname="helv", color=(0.5, 0.5, 0.5))
    y += 12
    page.insert_text((45, y), f"Timestamp: {report_data.get('timestamp')}", fontsize=8, fontname="helv", color=(0.5, 0.5, 0.5))
    y += 12
    page.insert_text((45, y), f"Target Machine: Hyundai R215L Excavator ({report_data.get('machine_id')})", fontsize=8, fontname="helv", color=(0.5, 0.5, 0.5))
    y += 12
    page.insert_text((45, y), f"Engineer Query: \"{report_data.get('query')}\"", fontsize=8, fontname="helv-oblique", color=(0.3, 0.3, 0.3))
    y += 20
    page.draw_line((45, y), (550, y), color=(0.4, 0.2, 0.1), width=1)
    y += 20
    
    # Render answer text sections
    answer = report_data.get("answer", "")
    lines = answer.split("\n")
    
    for line in lines:
        if y > 780:
            page = doc.new_page()
            # New page border
            page.draw_rect((20, 20, 575, 820), color=(0.4, 0.2, 0.1), width=1)
            y = 50
        
        # Format headers and text lines
        if line.strip().startswith("###"):
            cleaned = line.replace("###", "").strip()
            # Draw section header with color
            page.insert_text((45, y), cleaned, fontsize=10, fontname="helv-bold", color=(0.4, 0.2, 0.1))
            y += 15
        elif line.strip().startswith("##"):
            cleaned = line.replace("##", "").strip()
            page.insert_text((45, y), cleaned, fontsize=11, fontname="helv-bold", color=(0.4, 0.2, 0.1))
            y += 18
        elif line.strip():
            # Paragraph formatting with simple word wrapping
            text = line.strip()
            words = text.split()
            chunk = []
            for word in words:
                chunk.append(word)
                if len(" ".join(chunk)) > 85:
                    page.insert_text((45, y), " ".join(chunk[:-1]), fontsize=8.5, fontname="helv", color=(0.2, 0.2, 0.2))
                    y += 13
                    if y > 780:
                        page = doc.new_page()
                        page.draw_rect((20, 20, 575, 820), color=(0.4, 0.2, 0.1), width=1)
                        y = 50
                    chunk = [word]
            if chunk:
                page.insert_text((45, y), " ".join(chunk), fontsize=8.5, fontname="helv", color=(0.2, 0.2, 0.2))
                y += 13
        else:
            y += 8
            
    # Sign-off footer on the last page
    if y > 720:
        page = doc.new_page()
        page.draw_rect((20, 20, 575, 820), color=(0.4, 0.2, 0.1), width=1)
        y = 50
    y += 20
    page.draw_line((45, y), (550, y), color=(0.8, 0.8, 0.8), width=0.5)
    y += 15
    page.insert_text((45, y), "Report Generated Dynamically by FactoryMind AI Industrial Agent Core.", fontsize=7.5, fontname="helv-oblique", color=(0.6, 0.6, 0.6))
    
    pdf_bytes = doc.write()
    doc.close()
    
    buf = io.BytesIO(pdf_bytes)
    buf.seek(0)
    
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=FactoryMind_Report_{query_id}.pdf"}
    )

@app.get("/documents")
async def list_documents(current_user: dict = Depends(get_current_user)):
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

@app.get("/stats")
async def get_stats(current_user: dict = Depends(get_current_user)):
    manuals_dir = os.path.join(settings.DATA_DIR, "manuals")
    doc_count = 0
    if os.path.exists(manuals_dir):
        doc_count = len([f for f in os.listdir(manuals_dir) if os.path.isfile(os.path.join(manuals_dir, f))])
    
    stats = {}
    try:
        stats = container.vector_store.get_stats()
    except Exception:
        pass
        
    manuals_stats = stats.get("manuals", {"count": 0})
    
    return {
        "machine_model": "Hyundai R215L Smart Plus",
        "manuals_count": doc_count,
        "points_count": manuals_stats.get("count", 0),
        "pages_count": 845,
        "tables_count": 148,
        "images_count": 79
    }

@app.post("/debug/retrieve")
async def debug_retrieve(req: RetrieveDebugRequest, current_user: dict = Depends(get_current_user)):
    query = req.query
    top_k = req.top_k or 8
    
    # Search all collections with user isolation
    all_hits = rag_service.search_all_collections(query, top_k=top_k, user_id=current_user.get("uid"))
    flat_hits = []
    for coll, hits in all_hits.items():
        for hit in hits:
            flat_hits.append(hit)
            
    # Rerank (Top 50 -> Top 8)
    reranked = rag_service.reranker.rerank(query, flat_hits, top_k=top_k)
    
    chunks = []
    for hit in reranked:
        payload = hit.get("payload", {})
        chunks.append({
            "id": hit.get("id"),
            "score": round(hit.get("score", 0.0), 4),
            "document_name": payload.get("document_name", "Unknown"),
            "page": payload.get("page", 0),
            "heading": payload.get("heading", "General"),
            "text": hit.get("text", "")[:300] + "..." if hit.get("text") else ""
        })
        
    return {
        "query": query,
        "top_k": top_k,
        "chunks": chunks
    }

@app.post("/admin/collection/delete")
async def delete_collection(current_user: dict = Depends(get_current_user)):
    try:
        collections = ["manuals", "sop", "error_codes", "spare_parts"]
        for coll in collections:
            if container.vector_store.client.collection_exists(coll):
                container.vector_store.client.delete_collection(coll)
        # Clear cache
        rag_service.clear_cache()
        return {"status": "success", "message": "Collections deleted successfully."}
    except Exception as e:
        logger.error(f"Failed to delete collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/collection/recreate")
async def recreate_collection(current_user: dict = Depends(get_current_user)):
    try:
        collections = ["manuals", "sop", "error_codes", "spare_parts"]
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

@app.get("/admin/collection/status")
async def get_collection_status():
    try:
        stats = {}
        collections = ["manuals", "sop", "error_codes", "spare_parts"]
        total_chunks = 0
        for coll in collections:
            if container.vector_store.client.collection_exists(coll):
                count = container.vector_store.client.count(collection_name=coll).count
                stats[coll] = count
                total_chunks += count
            else:
                stats[coll] = 0
                
        return {
            "status": "success",
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "dimension": container.vector_store.dimension,
            "total_chunks": total_chunks,
            "breakdown": stats,
            "last_indexed": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Failed to get collection status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
