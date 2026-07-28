"""Authentication routes."""
from __future__ import annotations

import jwt
import datetime
from fastapi import APIRouter, HTTPException, Depends
from backend.config import settings
from backend.models.schemas import LoginRequest, RegisterRequest
from backend.auth.jwt_auth import get_current_user
from backend.constants import MOCK_USERS

router = APIRouter()


def create_access_token(username: str, role: str) -> str:
    """Create JWT access token."""
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


@router.post("/login")
async def login(req: LoginRequest):
    """Login endpoint with mock authentication."""
    user = MOCK_USERS.get(req.username.lower())
    if not user or user["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
        
    token = create_access_token(user["username"], user["role"])
    return {
        "token": token,
        "username": user["display_name"],
        "role": user["role"]
    }


@router.post("/register")
async def register(req: RegisterRequest):
    """Register endpoint (simplified for demo)."""
    token = create_access_token(req.username, req.role)
    return {
        "token": token,
        "username": req.display_name,
        "role": req.role
    }
