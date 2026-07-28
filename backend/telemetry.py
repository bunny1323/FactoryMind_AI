"""Telemetry data management for machine sensors."""
from __future__ import annotations

from typing import Dict, Any
from backend.constants import DEFAULT_TELEMETRY, SUPPORTED_MACHINES


def get_telemetry(machine_id: str) -> Dict[str, Any]:
    """Get telemetry data for a specific machine."""
    return DEFAULT_TELEMETRY.get(machine_id, DEFAULT_TELEMETRY["M101"])


def get_available_machines() -> list[str]:
    """Get list of available machine IDs."""
    return SUPPORTED_MACHINES
