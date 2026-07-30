"""
Intelligent Query Planner & Query Rewriter for FactoryMind AI.
Classifies user intent across 23 domain-specific categories and rewrites queries
to maximize hybrid retrieval precision.
"""

from __future__ import annotations

import re
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("factorymind")


@dataclass
class QueryPlan:
    original_query: str
    rewritten_query: str
    intent: str
    target_collections: List[str]
    is_conversational: bool
    requires_visual: bool
    metadata_filters: Dict[str, Any]


class QueryPlanner:
    """
    Intelligent Query Planner for FactoryMind AI RAG system.
    Determines query intent, target collections, rewrite strategy, and context expansion.
    """

    VALID_INTENTS = [
        "GREETING", "MACHINE_OVERVIEW", "SUMMARY", "TROUBLESHOOTING", "SPECIFICATION",
        "SAFETY", "MAINTENANCE", "HYDRAULIC", "ELECTRICAL", "ENGINE", "TORQUE",
        "OIL", "FILTERS", "COMPONENT", "PART_NUMBER", "DIAGRAM_REQUEST", "IMAGE_REQUEST",
        "INSPECTION", "PREVENTIVE_MAINTENANCE", "PREDICTIVE_MAINTENANCE",
        "DOCUMENT_QUESTION", "METADATA_QUESTION", "GENERAL_CHAT"
    ]

    def __init__(self):
        pass

    def classify_intent(self, query: str) -> str:
        """Classify user query into one of 23 fine-grained intents."""
        q = query.lower().strip()

        # 1. Greetings & General Chat
        if q in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "thanks", "thank you"]:
            return "GREETING"
        
        if any(kw in q for kw in ["who are you", "what can you do", "introduce yourself", "help me"]):
            return "GENERAL_CHAT"

        # 2. Metadata Questions & Document Summaries
        if any(kw in q for kw in ["how many manuals", "how many documents", "list manuals", "list documents", "number of manuals"]):
            return "METADATA_QUESTION"

        if re.search(r"\b(page\s+\d+|section\s+\d+|chapter\s+\d+)\b", q):
            return "DOCUMENT_QUESTION"

        if any(kw in q for kw in ["summarize", "summary", "overview of document", "brief me"]):
            return "SUMMARY"

        if any(kw in q for kw in ["which machine", "what machine", "supported excavator", "excavator model"]):
            return "MACHINE_OVERVIEW"

        # 3. Diagram / Image / Figure Requests
        if any(kw in q for kw in [
            "show figure", "figure", "fig.", "fig ", "show diagram", "hydraulic schematic", 
            "wiring layout", "exploded view", "circuit diagram", "schematic", "layout", 
            "diagram", "illustration", "drawing", "flowchart", "flow chart"
        ]):
            return "DIAGRAM_REQUEST"

        if any(kw in q for kw in ["show image", "show picture", "look like", "view figure", "photo"]):
            return "IMAGE_REQUEST"

        # 4. Specific Component Intents
        if any(kw in q for kw in ["oil", "lubricant", "grease", "viscosity", "fluid capacity"]):
            return "OIL"

        if any(kw in q for kw in ["filter", "air filter", "fuel filter", "oil filter", "return filter"]):
            return "FILTERS"

        if any(kw in q for kw in ["torque", "tightening", "bolt torque", "nut torque", "nm", "ft-lb"]):
            return "TORQUE"

        if any(kw in q for kw in ["part number", "part no", "sp-", "spare part", "replacement part"]):
            return "PART_NUMBER"

        if any(kw in q for kw in ["hydraulic", "pump", "valve", "cylinder", "spool", "relief valve", "manifold"]):
            return "HYDRAULIC"

        if any(kw in q for kw in ["electrical", "wiring", "harness", "sensor", "relay", "fuse", "battery", "alternator", "starter"]):
            return "ELECTRICAL"

        if any(kw in q for kw in ["engine", "cummins", "diesel", "rpm", "piston", "injector", "turbocharger"]):
            return "ENGINE"

        if any(kw in q for kw in ["warning", "safety", "hazard", "caution", "danger", "ppe", "icon", "symbol"]):
            return "SAFETY"

        if any(kw in q for kw in ["inspection", "daily check", "clearance check", "gauge"]):
            return "INSPECTION"

        if any(kw in q for kw in ["predict", "forecast", "future failure", "sensor trend", "telemetry"]):
            return "PREDICTIVE_MAINTENANCE"

        if any(kw in q for kw in ["preventive", "scheduled service", "250 hour", "500 hour", "1000 hour"]):
            return "PREVENTIVE_MAINTENANCE"

        if any(kw in q for kw in ["maintenance", "service procedure", "overhaul", "replace", "disassembly", "assembly"]):
            return "MAINTENANCE"

        if any(kw in q for kw in ["spec", "specification", "dimension", "weight", "operating capacity", "flow rate"]):
            return "SPECIFICATION"

        if any(kw in q for kw in ["component", "bearing", "seal", "motor", "turntable", "track"]):
            return "COMPONENT"

        # Default fallback
        return "TROUBLESHOOTING"

    def rewrite_query(self, query: str, intent: str) -> str:
        """
        Rewrites short/ambiguous user queries to expand domain concepts for vector search.
        """
        q = query.strip()
        q_lower = q.lower()

        # If query is already lengthy (> 8 words), keep as is with minor enrichment
        if len(q.split()) > 8:
            return q

        # Domain expansions based on intent
        if intent == "OIL":
            if "oil" in q_lower and len(q.split()) <= 4:
                return f"What engine oil, hydraulic fluid, and lubricant specifications and capacities are recommended for Hyundai R215L Smart Plus excavator? ({q})"

        elif intent == "TORQUE":
            if "torque" in q_lower and len(q.split()) <= 4:
                return f"What are the bolt tightening torque values and specifications for Hyundai R215L Smart Plus excavator components? ({q})"

        elif intent == "FILTERS":
            return f"What are the filter replacement procedures, intervals, and part numbers for Hyundai R215L Smart Plus? ({q})"

        elif intent == "SAFETY":
            return f"What are the safety warnings, precautions, hazards, and safety icons for Hyundai R215L Smart Plus? ({q})"

        elif intent == "HYDRAULIC":
            if len(q.split()) <= 3:
                return f"Hyundai R215L Smart Plus hydraulic system, main pump, control valves, pressure settings: {q}"

        elif intent == "ELECTRICAL":
            if len(q.split()) <= 3:
                return f"Hyundai R215L Smart Plus electrical wiring circuit diagram, relays, fuses, starter motor: {q}"

        elif intent == "TROUBLESHOOTING":
            if len(q.split()) <= 4:
                return f"Troubleshooting guide, failure diagnosis, cause and repair procedure for Hyundai R215L Smart Plus: {q}"

        elif intent == "DIAGRAM_REQUEST" or intent == "IMAGE_REQUEST":
            # Extract specific figure identifier if present (e.g. "Figure 3-12", "Fig 4")
            fig_match = re.search(r"(?i)\b(fig(?:ure|\.)?\s*\d+(?:[-.]\d+)?)\b", q)
            if fig_match:
                fig_ref = fig_match.group(1)
                return f"{fig_ref} {fig_ref} image {fig_ref} illustration {fig_ref} diagram schematic figure"
            return f"{q} schematic diagram figure illustration layout component view"

        return q

    def plan(self, query: str) -> QueryPlan:
        """Generates a complete execution plan for a user query."""
        intent = self.classify_intent(query)
        rewritten = self.rewrite_query(query, intent)

        is_conversational = intent in ["GREETING", "GENERAL_CHAT", "METADATA_QUESTION"]
        requires_visual = intent in ["DIAGRAM_REQUEST", "IMAGE_REQUEST"]

        # Route target collections dynamically
        if intent in ["SAFETY", "DIAGRAM_REQUEST", "IMAGE_REQUEST", "DOCUMENT_QUESTION"]:
            collections = ["manuals"]
        elif intent in ["OIL", "TORQUE", "SPECIFICATION", "FILTERS"]:
            collections = ["manuals", "sop"]
        elif intent in ["PART_NUMBER", "COMPONENT"]:
            collections = ["spare_parts", "manuals"]
        elif intent in ["PREDICTIVE_MAINTENANCE", "MAINTENANCE"]:
            collections = ["manuals", "maintenance_logs", "sop"]
        elif intent == "TROUBLESHOOTING":
            collections = ["manuals", "error_codes", "maintenance_logs", "sop"]
        else:
            collections = ["manuals", "sop", "maintenance_logs", "error_codes", "spare_parts"]

        return QueryPlan(
            original_query=query,
            rewritten_query=rewritten,
            intent=intent,
            target_collections=collections,
            is_conversational=is_conversational,
            requires_visual=requires_visual,
            metadata_filters={}
        )


query_planner = QueryPlanner()
