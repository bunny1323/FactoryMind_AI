from __future__ import annotations

import logging
import datetime
from typing import Any, Dict, List, TypedDict
from backend.services.rag_service import rag_service
from backend.services.llm_service import llm_service
from prediction.infer import prediction_engine
from backend.config import settings

logger = logging.getLogger("factorymind")

class AgentState(TypedDict):
    query: str
    machine_id: str
    user_id: str
    sensor_values: dict[str, Any]
    retrieved_documents: list[dict[str, Any]]
    prediction_result: dict[str, Any]
    graph_path: list[dict[str, Any]]
    maintenance_plan: dict[str, Any]
    sub_agent_history: list[str]
    final_answer: str
    confidence_breakdown: dict[str, int]

# --- Intent Detection Agent ---
def intent_detection_agent(q: str) -> str | None:
    """Classifies if query is a greeting or a maintenance question."""
    q = q.lower().strip()
    
    # Dynamic document counting and listing check
    if any(phrase in q for phrase in ["how many documents", "how many manuals", "list documents", "list manuals", "what documents", "what manuals", "number of manuals", "number of documents"]):
        import os
        from backend.config import settings
        manuals_dir = os.path.join(settings.DATA_DIR, "manuals")
        if os.path.exists(manuals_dir):
            files = [f for f in os.listdir(manuals_dir) if os.path.isfile(os.path.join(manuals_dir, f))]
            count = len(files)
            if count == 0:
                return "I currently have 0 documents indexed in the knowledge base."
            
            file_list = "\n".join([f"- {f}" for f in files])
            return f"I currently have {count} manuals indexed in the knowledge base:\n\n{file_list}"
        return "The manuals directory is not configured."

    greetings = {
        "hi": "Hello! I am FactoryMind AI, your Explainable Multimodal Industrial Copilot for the Hyundai R215L excavator. How can I assist you with maintenance, diagnostics, or troubleshooting today?",
        "hello": "Hello! I am FactoryMind AI, your Explainable Multimodal Industrial Copilot for the Hyundai R215L excavator. How can I assist you with maintenance, diagnostics, or troubleshooting today?",
        "good morning": "Good morning! I am FactoryMind AI. Ready to assist with excavator maintenance, diagnostic checks, or spare parts lookup.",
        "how are you": "I am operating at peak efficiency, monitoring all telemetry streams. How can I help you troubleshoot or maintain the excavator today?",
        "thank you": "You're welcome! Let me know if you need any more manual citations, repair SOPs, or diagnostic assessments.",
        "thanks": "You're welcome! Let me know if you need any more manual citations, repair SOPs, or diagnostic assessments.",
        "who are you": "I am FactoryMind AI, an Explainable Multimodal Industrial Copilot powered by Layout-Aware Agentic RAG. I assist maintenance engineers with Hyundai R215L excavators by combining RAG manuals, telemetry prediction models, and knowledge graphs.",
        "help": "I can help you troubleshoot faults, search service manuals, lookup spare parts, retrieve step-by-step SOPs, and analyze telemetry. Try asking: 'Machine M101 is showing increased vibration. What should I do?'",
        "about": "FactoryMind AI is a premium Industry 4.0 copilot. I analyze structural/hydraulic sensor telemetry, query Neo4j knowledge graphs, and retrieve layout-aware manuals to deliver explainable, evidence-backed repair dispatch plans.",
        "capabilities": "My capabilities include: 1. Layout-Aware Multimodal RAG (manuals, tables, diagrams). 2. Knowledge Graph query mapping. 3. Telemetry risk analysis (IoT pre-wired). 4. Automated PDF Maintenance Report generation.",
        "introduce yourself": "Hello! I am FactoryMind AI, your Explainable Multimodal Industrial Copilot. I combine manuals, telemetry assessments, and component knowledge graphs under a multi-agent supervisor to assist you like an experienced maintenance engineer."
    }
    
    # Exact or starts/ends match
    for key, response in greetings.items():
        if q == key or q.startswith(key + " ") or q.endswith(" " + key):
            return response
            
    # Substring matching for greetings
    if q in ["hi", "hello", "hey", "help", "about", "capabilities"]:
        return greetings[q]
        
    return None

def intent_detection_agent_node(state: AgentState) -> dict[str, Any]:
    """Node wrapper for the Intent Detection Agent."""
    logger.info("Executing Intent Detection Agent...")
    reply = intent_detection_agent(state["query"])
    return {
        "final_answer": reply or "",
        "sub_agent_history": state["sub_agent_history"] + ["intent_detection_agent"]
    }

# --- Supervisor Agent ---
def supervisor_agent_node(state: AgentState) -> dict[str, Any]:
    """Orchestrates query routing across active sub-agents."""
    logger.info("Executing Supervisor Agent...")
    return {
        "sub_agent_history": state["sub_agent_history"] + ["supervisor_agent"]
    }

# --- Document Retrieval Agent ---
def document_retrieval_agent_node(state: AgentState) -> dict[str, Any]:
    """Searches vector database (dense + sparse hybrid) and retrieves layout-aware chunks."""
    logger.info("Executing Document Retrieval Agent...")
    query = state["query"]
    
    # Bypass RAG if greeting detected
    if state.get("final_answer"):
        return {"retrieved_documents": [], "sub_agent_history": state["sub_agent_history"] + ["document_retrieval_agent"]}
        
    # Search all collections with user isolation
    all_hits = rag_service.search_all_collections(query, top_k=3, user_id=state.get("user_id", "default_user"))
    flat_hits = []
    for coll, hits in all_hits.items():
        for hit in hits:
            flat_hits.append(hit)
            
    # Rerank
    reranked = rag_service.reranker.rerank(query, flat_hits, top_k=5)
    
    # Bypass adjacent chunk lookup to ensure focus on highly relevant manual passages
    layout_restored = [hit for hit in reranked if hit.get("score", 0.0) >= 0.45]
    if not layout_restored:
        layout_restored = reranked[:3]
        
    return {
        "retrieved_documents": layout_restored,
        "sub_agent_history": state["sub_agent_history"] + ["document_retrieval_agent"]
    }

# --- Knowledge Graph Agent ---
from graph.neo4j_client import graph_client

def knowledge_graph_agent_node(state: AgentState) -> dict[str, Any]:
    """Traces machine -> component -> failure -> repair relationships."""
    logger.info("Executing Knowledge Graph Agent...")
    if state.get("final_answer"):
        return {"graph_path": [], "sub_agent_history": state["sub_agent_history"] + ["knowledge_graph_agent"]}
        
    query = state["query"]
    machine_id = state["machine_id"]
    path = graph_client.get_path_for_query(query, machine_id)
    return {
        "graph_path": path,
        "sub_agent_history": state["sub_agent_history"] + ["knowledge_graph_agent"]
    }

# --- Future Prediction Agent (Placeholder) ---
def future_prediction_agent_node(state: AgentState) -> dict[str, Any]:
    """Reports decoupled predictive IoT streaming status."""
    logger.info("Executing Future Prediction Agent...")
    if state.get("final_answer"):
        return {"prediction_result": {}, "sub_agent_history": state["sub_agent_history"] + ["future_prediction_agent"]}
        
    # Standard values
    sensors = state.get("sensor_values") or {
        "air_temperature": 298.2,
        "process_temperature": 308.6,
        "rotational_speed": 1850,
        "torque": 45.2,
        "tool_wear": 120,
    }
    
    pred = prediction_engine.predict(
        air_temp=sensors["air_temperature"],
        process_temp=sensors["process_temperature"],
        rotational_speed=sensors["rotational_speed"],
        torque=sensors["torque"],
        tool_wear=sensors["tool_wear"]
    )
    
    return {
        "prediction_result": pred,
        "sensor_values": pred["telemetry"],
        "sub_agent_history": state["sub_agent_history"] + ["future_prediction_agent"]
    }

# --- Evidence Aggregation Agent ---
def evidence_aggregation_agent_node(state: AgentState) -> dict[str, Any]:
    """Collates evidence elements and calculates multi-dimensional retrieval confidence ratings."""
    logger.info("Executing Evidence Aggregation Agent...")
    if state.get("final_answer"):
        return {
            "confidence_breakdown": {"overall": "High", "retrieval": "High", "graph": "High", "evidence": "High", "agreement": "High"},
            "sub_agent_history": state["sub_agent_history"] + ["evidence_aggregation_agent"]
        }
        
    docs = state.get("retrieved_documents", [])
    path = state.get("graph_path", [])
    
    # 1. Retrieval confidence
    avg_score = sum(doc.get("score", 0.80) for doc in docs) / len(docs) if docs else 0.0
    retrieval_rating = "High" if avg_score > 0.82 else "Medium" if avg_score > 0.65 else "Low"
    
    # 2. Graph consistency (Neo4j link verification)
    graph_rating = "High" if len(path) > 0 else "Low"
    
    # 3. Evidence coverage
    evidence_rating = "High" if len(docs) >= 3 else "Medium" if len(docs) > 0 else "Low"
    
    # 4. Document agreement
    agreement_rating = "High" if avg_score > 0.75 else "Medium"
    
    # 5. Overall Confidence
    overall_rating = "High"
    if retrieval_rating == "Low" or evidence_rating == "Low":
        overall_rating = "Low"
    elif retrieval_rating == "Medium" or evidence_rating == "Medium":
        overall_rating = "Medium"
        
    breakdown = {
        "overall": overall_rating,
        "retrieval": retrieval_rating,
        "graph": graph_rating,
        "evidence": evidence_rating,
        "agreement": agreement_rating
    }
    
    return {
        "confidence_breakdown": breakdown,
        "sub_agent_history": state["sub_agent_history"] + ["evidence_aggregation_agent"]
    }

# --- Maintenance Planner Agent ---
def maintenance_planner_agent_node(state: AgentState) -> dict[str, Any]:
    """Formulates dispatch recommendations based strictly on retrieved documents."""
    logger.info("Executing Maintenance Planner Agent...")
    if state.get("final_answer"):
        return {"maintenance_plan": {}, "sub_agent_history": state["sub_agent_history"] + ["maintenance_planner_agent"]}
        
    # Initial plan structure - components will be populated dynamically by the final synthesizer from text
    plan = {
        "tools_required": [],
        "spare_parts_dispatched": [],
        "estimated_downtime": "Under review"
    }
    
    return {
        "maintenance_plan": plan,
        "sub_agent_history": state["sub_agent_history"] + ["maintenance_planner_agent"]
    }

# --- Report Generator Agent ---
def report_generator_agent_node(state: AgentState) -> dict[str, Any]:
    """Compiles the dispatch plan structure ready for PDF generation."""
    logger.info("Executing Report Generator Agent...")
    return {
        "sub_agent_history": state["sub_agent_history"] + ["report_generator_agent"]
    }

# --- Synthesizer Node ---
def synthesizer_node(state: AgentState) -> dict[str, Any]:
    """Orchestrates final response synthesis formatting like an experienced field engineer based ONLY on uploaded manuals."""
    logger.info("Executing Synthesizer Node...")
    if state.get("final_answer"):
        return {} # Already handled by Intent Detection
        
    query = state["query"]
    docs = state.get("retrieved_documents", [])
    pred = state.get("prediction_result", {})
    plan = state.get("maintenance_plan", {})
    
    doc_context = "\n\n".join([f"SOURCE: {doc['title']} (Page {doc.get('payload', {}).get('page', 'N/A')})\n{doc['text']}" for doc in docs])
    
    combined_context = f"""
=== RETRIEVED MANUAL CONTEXT ===
{doc_context}
"""
    
    system_rules = """You are an industrial maintenance assistant for the Hyundai R215L Smart Plus excavator.
Answer the user's question ONLY using the supplied retrieved context.
Do not use prior knowledge. Never fabricate or guess specifications, error codes, tools, or spare parts.
If the answer cannot be found in the retrieved context, or if the context is empty, explicitly say:
'No relevant information was found.'

Do NOT output standard boilerplate headers (like Root Cause, Tools, or Spare Parts) unless they are directly relevant to the user query and supported by specifications in the retrieved manual chunks. If the user asks for a simple summary, return only a summary. If they ask for a pressure value or oil type, return only that information. Keep your tone direct, technical, and engineering-focused."""

    final_answer = llm_service.synthesize(query, combined_context, system_rules)
    return {
        "final_answer": final_answer,
        "llm_prompt": f"SYSTEM RULES:\n{system_rules}\n\nCONTEXT:\n{combined_context}\n\nUSER QUERY:\n{query}",
        "sub_agent_history": state["sub_agent_history"] + ["synthesizer"]
    }

# --- Multi-Agent Orchestrator ---
class LangGraphOrchestrator:
    def __init__(self):
        self._setup_graph()

    def _setup_graph(self):
        try:
            from langgraph.graph import StateGraph, END
            
            builder = StateGraph(AgentState)
            
            builder.add_node("intent_detection", intent_detection_agent_node)
            builder.add_node("supervisor", supervisor_agent_node)
            builder.add_node("document_retrieval", document_retrieval_agent_node)
            builder.add_node("knowledge_graph", knowledge_graph_agent_node)
            builder.add_node("future_prediction", future_prediction_agent_node)
            builder.add_node("evidence_aggregation", evidence_aggregation_agent_node)
            builder.add_node("maintenance_planner", maintenance_planner_agent_node)
            builder.add_node("report_generator", report_generator_agent_node)
            builder.add_node("synthesizer", synthesizer_node)
            
            # Simple sequential execution DAG
            builder.add_edge("intent_detection", "supervisor")
            builder.add_edge("supervisor", "document_retrieval")
            builder.add_edge("document_retrieval", "knowledge_graph")
            builder.add_edge("knowledge_graph", "future_prediction")
            builder.add_edge("future_prediction", "evidence_aggregation")
            builder.add_edge("evidence_aggregation", "maintenance_planner")
            builder.add_edge("maintenance_planner", "report_generator")
            builder.add_edge("report_generator", "synthesizer")
            builder.add_edge("synthesizer", END)
            
            builder.set_entry_point("intent_detection")
            
            self.graph = builder.compile()
            self.use_langgraph = True
            logger.info("LangGraph workflow successfully compiled with new agents.")
        except ImportError:
            logger.warning("langgraph library not installed. Falling back to native Python sequential orchestrator.")
            self.use_langgraph = False

    def run(self, query: str, machine_id: str, sensor_values: dict[str, Any] | None = None, user_id: str = "default_user") -> AgentState:
        state: AgentState = {
            "query": query,
            "machine_id": machine_id,
            "user_id": user_id,
            "sensor_values": sensor_values or {},
            "retrieved_documents": [],
            "prediction_result": {},
            "graph_path": [],
            "maintenance_plan": {},
            "sub_agent_history": [],
            "final_answer": "",
            "confidence_breakdown": {"overall": 0, "retrieval": 0, "graph": 0, "evidence": 0, "answer": 0}
        }
        
        # 1. Intent Detection Check
        state.update(intent_detection_agent_node(state))
        
        # If it is a greeting/conversational, bypass all other agents
        if state["final_answer"]:
            state["confidence_breakdown"] = {"overall": 100, "retrieval": 100, "graph": 100, "evidence": 100, "answer": 100}
            return state
            
        # 2. Execute full agent pipeline
        state.update(supervisor_agent_node(state))
        state.update(document_retrieval_agent_node(state))
        state.update(knowledge_graph_agent_node(state))
        state.update(future_prediction_agent_node(state))
        state.update(evidence_aggregation_agent_node(state))
        state.update(maintenance_planner_agent_node(state))
        state.update(report_generator_agent_node(state))
        state.update(synthesizer_node(state))
        
        return state

agent_orchestrator = LangGraphOrchestrator()
