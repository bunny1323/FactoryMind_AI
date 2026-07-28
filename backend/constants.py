"""Application constants and configuration values."""
from __future__ import annotations

from typing import Dict, Any


# Mock users for local development authentication
MOCK_USERS = {
    "onepiece": {"username": "onepiece", "password": "luffy", "display_name": "Luffy", "role": "admin"},
    "zoro": {"username": "zoro", "password": "swordsman", "display_name": "Zoro", "role": "user"}
}


# Machine configuration
MACHINE_MODEL = "Hyundai R215L Smart Plus"
SUPPORTED_MACHINES = ["M101", "M102", "M103"]


# Default telemetry data (marked as simulated)
DEFAULT_TELEMETRY: Dict[str, Dict[str, Any]] = {
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


# Intent detection responses
GREETING_RESPONSES = {
    "hi": "Hello! I am FactoryMind AI, your Explainable Multimodal Industrial Copilot for the Hyundai R215L excavator. How can I assist you with maintenance, diagnostics, or troubleshooting today?",
    "hello": "Hello! I am FactoryMind AI, your Explainable Multimodal Industrial Copilot for the Hyundai R215L excavator. How can I assist you with maintenance, diagnostics, or troubleshooting today?",
    "hey": "Hello! I am FactoryMind AI. How can I assist you with the Hyundai R215L today?",
    "good morning": "Good morning! I am FactoryMind AI. Ready to assist with excavator maintenance, diagnostic checks, or spare parts lookup.",
    "good afternoon": "Good afternoon! Ready to assist with excavator maintenance, fault diagnosis, or spare parts lookup.",
    "good evening": "Good evening! Ready to assist with excavator maintenance, fault diagnosis, or spare parts lookup.",
    "how are you": "I am operating at peak efficiency, monitoring all telemetry streams. How can I help you troubleshoot or maintain the excavator today?",
    "thank you": "You're welcome! Let me know if you need any more manual citations, repair SOPs, or diagnostic assessments.",
    "thanks": "You're welcome! Let me know if you need any more manual citations, repair SOPs, or diagnostic assessments.",
    "who are you": "I am FactoryMind AI, an Explainable Multimodal Industrial Copilot powered by Layout-Aware Agentic RAG. I assist maintenance engineers with Hyundai R215L excavators by combining RAG manuals, telemetry prediction models, and knowledge graphs.",
    "what are you": "I am FactoryMind AI — an AI-powered industrial maintenance copilot for the Hyundai R215L Smart Plus excavator. I combine vector search over indexed service manuals, XGBoost predictive failure detection, and a multi-agent RAG pipeline.",
    "help": "I can help you troubleshoot faults, search service manuals, lookup spare parts, retrieve step-by-step SOPs, and analyze telemetry. Try asking: 'Machine M101 is showing increased vibration. What should I do?'",
    "about": "FactoryMind AI is a premium Industry 4.0 copilot. I analyze structural/hydraulic sensor telemetry, query Neo4j knowledge graphs, and retrieve layout-aware manuals to deliver explainable, evidence-backed repair dispatch plans.",
    "capabilities": "My capabilities include:\n1. **Layout-Aware Multimodal RAG** (manuals, tables, diagrams)\n2. **Knowledge Graph** query mapping\n3. **Telemetry risk analysis** (XGBoost predictive failure model)\n4. **Automated PDF Maintenance Report** generation\n\nAsk me about hydraulic pressures, error codes, spare parts, or SOPs!",
    "introduce yourself": "Hello! I am FactoryMind AI, your Explainable Multimodal Industrial Copilot. I combine manuals, telemetry assessments, and component knowledge graphs under a multi-agent supervisor to assist you like an experienced maintenance engineer.",
}


def get_greeting_response(key: str) -> str | None:
    """Get greeting response by key."""
    return GREETING_RESPONSES.get(key.lower())


def get_machine_info_response() -> str:
    """Get machine information response."""
    return (
        f"I support the **{MACHINE_MODEL}** crawler excavator.\n\n"
        "My knowledge base includes service manuals, maintenance SOPs, error code tables, "
        "spare parts catalogue, and maintenance logs specifically for this model.\n\n"
        "**Tracked machines in this deployment:**\n"
        + "\n".join([f"- **{machine}** — {MACHINE_MODEL} unit" for machine in SUPPORTED_MACHINES]) +
        "\n\n"
        "Ask me anything about maintenance, fault diagnosis, hydraulic specs, or spare parts!"
    )
