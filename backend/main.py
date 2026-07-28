"""FactoryMind AI - Main FastAPI Application."""
from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.config import settings
from backend.logging_config import setup_logging, get_logger
from backend.exceptions import handle_exception

# Configure structured logging
setup_logging()
logger = get_logger("factorymind")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="FactoryMind AI: Predictive and RAG Industrial Maintenance Intelligence Platform for the Hyundai R215L Excavator."
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(f"Unhandled exception in API request: {exc}")
    http_exc = handle_exception(exc)
    return JSONResponse(
        status_code=http_exc.status_code,
        content=http_exc.detail
    )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include route modules
from backend.routes import auth, query, ingestion, admin, machines, debug, reports, prediction
from prediction.router import router as prediction_router

app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(query.router, tags=["Query"])
app.include_router(ingestion.router, tags=["Ingestion"])
app.include_router(admin.router, tags=["Admin"])
app.include_router(machines.router, tags=["Machines"])
app.include_router(debug.router, prefix="/debug", tags=["Debug"])
app.include_router(reports.router, prefix="/reports", tags=["Reports"])
app.include_router(prediction.router, tags=["Prediction"])
app.include_router(prediction_router)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Initializing FactoryMind AI backend...")
    
    # Create data directories
    os.makedirs(os.path.join(settings.DATA_DIR, "manuals"), exist_ok=True)
    os.makedirs(os.path.join(settings.DATA_DIR, "sop"), exist_ok=True)
    os.makedirs(os.path.join(settings.DATA_DIR, "maintenance_logs"), exist_ok=True)
    os.makedirs(os.path.join(settings.DATA_DIR, "spare_parts"), exist_ok=True)
    os.makedirs(os.path.join(settings.DATA_DIR, "error_codes"), exist_ok=True)
    
    # Initialize Qdrant if configured
    if settings.VECTOR_BACKEND == "qdrant":
        try:
            from rag.qdrant_initializer import QdrantInitializer
            from backend.dependencies import container
            q_init = QdrantInitializer(container.vector_store.client, container.embedder.dimension)
            q_init.initialize()
        except Exception as e:
            logger.error(f"Error initializing Qdrant during startup: {e}")
