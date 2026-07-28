"""Machine-related routes."""
from __future__ import annotations

import logging
from fastapi import APIRouter, Depends
from typing import Any, Dict
from backend.auth.jwt_auth import get_current_user
from backend.telemetry import get_available_machines
from backend.db import get_machine_history_logs
from graph.neo4j_client import graph_client

logger = logging.getLogger("factorymind")

router = APIRouter()


@router.get("/machines")
async def list_machines():
    """List all available machines."""
    return get_available_machines()


@router.get("/machines/{id}/graph")
async def get_machine_graph(id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get knowledge graph path for a machine."""
    return graph_client.get_path_for_query("", id)


@router.get("/machines/{id}/history")
async def get_machine_history(id: str):
    """Get maintenance history for a machine."""
    logs = get_machine_history_logs(id)
    return {
        "machine_id": id,
        "logs": logs
    }
