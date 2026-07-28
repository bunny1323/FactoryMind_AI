"""Pydantic models for API requests and responses."""
from __future__ import annotations

from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str
    machine_id: str = "M101"


class RetrieveDebugRequest(BaseModel):
    """Request model for debug retrieve endpoint."""
    query: str
    top_k: Optional[int] = 8


class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""
    air_temp: float
    process_temp: float
    rotational_speed: float
    torque: float
    tool_wear: float


class LoginRequest(BaseModel):
    """Request model for login endpoint."""
    username: str
    password: str


class RegisterRequest(BaseModel):
    """Request model for register endpoint."""
    username: str
    password: str
    display_name: str
    role: str = "user"
