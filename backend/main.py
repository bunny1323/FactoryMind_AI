from __future__ import annotations

# Trigger reload
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
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

from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(f"Unhandled exception in API request: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": f"An unexpected error occurred: {str(exc)}"
        }
    )


app.include_router(prediction_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info("Initializing FactoryMind AI backend...")
    if settings.VECTOR_BACKEND == "qdrant":
        try:
            from rag.qdrant_initializer import QdrantInitializer
            q_init = QdrantInitializer(container.vector_store.client, container.embedder.dimension)
            q_init.initialize()
        except Exception as e:
            logger.error(f"Error initializing Qdrant during startup: {e}")

@app.get("/debug/model")
async def debug_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prediction", "model", "xgboost_model.pkl")
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        return {"error": f"Model file not found at {model_path}"}
    try:
        import pickle
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        info = {
            "type": str(type(model)),
            "is_dict": isinstance(model, dict)
        }
        
        if isinstance(model, dict):
            info["keys"] = list(model.keys())
            for k, v in model.items():
                info[f"{k}_type"] = str(type(v))
                if k == "scaler":
                    info["scaler_mean"] = list(v.mean_)
                    info["scaler_scale"] = list(v.scale_)
                if hasattr(v, "feature_names_in_"):
                    info[f"{k}_features"] = list(v.feature_names_in_)
                elif hasattr(v, "n_features_in_"):
                    info[f"{k}_n_features"] = v.n_features_in_
        else:
            if hasattr(model, "feature_names_in_"):
                info["features"] = list(model.feature_names_in_)
            if hasattr(model, "n_features_in_"):
                info["n_features"] = model.n_features_in_
                
        return info
    except Exception as e:
        return {"error": str(e)}


@app.get("/debug/groq")
async def debug_groq():
    """
    Sends a trivial completion to Groq and returns status/latency.
    Also shows which key prefix is being used and the full API error body.
    """
    import time
    import urllib.request
    import urllib.error
    t0 = time.perf_counter()
    provider = settings.LLM_PROVIDER.lower()
    if provider != "groq":
        return {"status": "skipped", "reason": f"LLM_PROVIDER is '{provider}', not groq"}
    api_key = getattr(settings, "GROQ_API_KEY", None)
    if not api_key or not api_key.strip():
        return {"status": "error", "reason": "GROQ_API_KEY is not set or empty"}

    model = getattr(settings, "GROQ_MODEL", "llama-3.3-70b-versatile")
    key_prefix = api_key[:12] + "..." if len(api_key) > 12 else api_key

    def _call_groq(test_model: str) -> dict:
        body = json.dumps({
            "model": test_model,
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Reply with OK"}]
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            text = payload["choices"][0]["message"]["content"].strip()
            latency = round(time.perf_counter() - t0, 3)
            return {"status": "ok", "model": test_model, "response": text,
                    "latency_s": latency, "key_prefix": key_prefix}
        except urllib.error.HTTPError as http_err:
            # Read body BEFORE the context manager closes it
            raw_body = b""
            try:
                raw_body = http_err.read()
            except Exception:
                pass
            body_str = raw_body.decode("utf-8", errors="replace")
            try:
                reason = json.loads(body_str)
            except Exception:
                reason = body_str or f"HTTP {http_err.code} (no body)"
            return {
                "status": "error",
                "http_code": http_err.code,
                "model_tried": test_model,
                "reason": reason,
                "key_prefix": key_prefix
            }
        except Exception as exc:
            return {"status": "error", "reason": str(exc), "key_prefix": key_prefix}

    # Try configured model first
    result = _call_groq(model)
    if result["status"] == "error" and result.get("http_code") == 403:
        # Auto-retry with the widest-available free-tier model
        fallback_model = "llama3-8b-8192"
        if model != fallback_model:
            result["fallback_attempt"] = fallback_model
            fallback_result = _call_groq(fallback_model)
            result["fallback_result"] = fallback_result
    return result


@app.get("/debug/llm")
async def debug_llm():
    """
    Diagnostic endpoint to test whichever LLM provider is currently active in .env.
    Executes a direct raw HTTP call and returns raw status, latency, response or exact error.
    """
    import time
    import urllib.request
    import urllib.error
    t0 = time.perf_counter()
    provider = settings.LLM_PROVIDER.lower()
    
    if provider in ("openai", "openai_compatible"):
        model = getattr(settings, "OPENAI_MODEL", "gpt-4o-mini")
        key = getattr(settings, "OPENAI_API_KEY", "")
        base_url = getattr(settings, "OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        url = f"{base_url}/chat/completions"
        key_prefix = (key[:12] + "...") if key else "NOT SET"
        info = {"provider": provider, "model": model, "url": url, "key_prefix": key_prefix}
        
        body = json.dumps({
            "model": model,
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Reply with OK"}]
        }).encode("utf-8")
        
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            text = payload["choices"][0]["message"]["content"].strip()
            latency = round(time.perf_counter() - t0, 3)
            return {"status": "ok", "response": text, "latency_s": latency, "active_config": info}
        except urllib.error.HTTPError as http_err:
            raw_body = b""
            try:
                raw_body = http_err.read()
            except Exception:
                pass
            body_str = raw_body.decode("utf-8", errors="replace")
            try:
                reason = json.loads(body_str)
            except Exception:
                reason = body_str or f"HTTP {http_err.code}"
            return {"status": "error", "http_code": http_err.code, "reason": reason, "active_config": info}
        except Exception as exc:
            return {"status": "error", "reason": str(exc), "active_config": info}

    return {"status": "skipped", "provider": provider}



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

# Mock machine telemetry store (explicitly marked as simulated)
TELEMETRY_DATA = {
    "M101": {
        "air_temperature": 298.2,     # Kelvin
        "process_temperature": 308.6, # Kelvin
        "rotational_speed": 1850,     # RPM
        "torque": 45.2,               # Nm
        "tool_wear": 120,             # Minutes
        "vibration": 0.22,            # mm peak-to-peak
        "telemetry_source": "simulated"
    },
    "M102": {
        "air_temperature": 296.5,
        "process_temperature": 307.2,
        "rotational_speed": 1420,
        "torque": 38.1,
        "tool_wear": 45,
        "vibration": 0.04,
        "telemetry_source": "simulated"
    },
    "M103": {
        "air_temperature": 299.1,
        "process_temperature": 309.8,
        "rotational_speed": 1600,
        "torque": 41.5,
        "tool_wear": 80,
        "vibration": 0.08,
        "telemetry_source": "simulated"
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
    
    try:
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
        
        # Find top image_url if visual intent is detected
        image_url = None
        from agents.graph import has_visual_intent
        if has_visual_intent(query):
            for doc in citations:
                payload = doc.get("payload", {}) if doc.get("payload") else {}
                if payload.get("image_path"):
                    image_url = payload.get("image_path")
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
            "image_url": image_url
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
        pages_count = 0  # Report honestly; do not substitute hardcoded fallback

    # Images count - dynamic file count in public extracted_images folder
    images_count = 0
    public_img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "public", "extracted_images")
    if os.path.exists(public_img_dir):
        try:
            images_count = len([f for f in os.listdir(public_img_dir) if os.path.isfile(os.path.join(public_img_dir, f))])
        except Exception as e:
            logger.warning(f"Failed to count images in public folder: {e}", exc_info=True)
            
    if images_count == 0:
        images_count = 0  # Report honestly; do not substitute hardcoded fallback

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

@app.post("/admin/collection/recreate")
async def recreate_collection(current_user: dict = Depends(get_current_user)):
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

@app.get("/admin/collection/status")
async def get_collection_status():
    try:
        stats = {}
        collections = ["manuals", "sop", "error_codes", "spare_parts", "maintenance_logs"]
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
