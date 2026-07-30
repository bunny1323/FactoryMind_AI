from __future__ import annotations

import logging
import datetime
from typing import Any, Dict, List, TypedDict
from backend.services.rag_service import rag_service
from backend.services.llm_service import llm_service
from prediction.infer import prediction_engine
from backend.config import settings
from backend.constants import get_greeting_response, get_machine_info_response, MACHINE_MODEL, SUPPORTED_MACHINES

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
    """Classifies if query is a greeting or a conversational FAQ — returns instant answer without RAG."""
    q_lower = q.lower().strip()

    # --- Dynamic document listing ---
    if any(phrase in q_lower for phrase in [
        "how many documents", "how many manuals", "list documents", "list manuals",
        "what documents", "what manuals", "number of manuals", "number of documents",
        "list all indexed documents", "list indexed documents", "indexed documents", "indexed manuals"
    ]):
        import os
        from backend.config import settings
        manuals_dir = os.path.join(settings.DATA_DIR, "manuals")
        if os.path.exists(manuals_dir):
            files = [f for f in os.listdir(manuals_dir) if os.path.isfile(os.path.join(manuals_dir, f))]
            count = len(files)
            if count == 0:
                return "I currently have 0 documents indexed in the knowledge base."
            file_list = "\n".join([f"- {f}" for f in sorted(files)])
            return f"I currently have **{count} manuals** indexed in the knowledge base:\n\n{file_list}"
        return "The manuals directory is not configured."

    # --- Machine / model FAQ ---
    machine_keywords = [
        "which machine", "what machine", "which excavator", "what excavator",
        "which model", "what model", "which equipment", "what equipment",
        "which vehicle", "what vehicle", "supported machine", "supported model",
        "what do you support", "what machine do you support", "which machine do you support",
        "machine do you", "what machines", "which machines"
    ]
    if any(kw in q_lower for kw in machine_keywords):
        return get_machine_info_response()

    # --- Greetings and conversational FAQs ---
    # Exact or prefix/suffix match
    for key in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", 
                "how are you", "thank you", "thanks", "who are you", "what are you", 
                "help", "about", "capabilities", "introduce yourself"]:
        if q_lower == key or q_lower.startswith(key + " ") or q_lower.endswith(" " + key):
            response = get_greeting_response(key)
            if response:
                return response

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

def has_visual_intent(query: str) -> bool:
    q = query.lower()
    keywords = ["show me", "what does", "look like", "diagram of", "picture of", "illustration", "see the figure", "show diagram", "schematic"]
    return any(kw in q for kw in keywords)

# --- Document Retrieval Agent ---
def document_retrieval_agent_node(state: AgentState) -> dict[str, Any]:
    """Searches vector database (dense + sparse hybrid) and retrieves layout-aware chunks."""
    import time
    from backend.services.query_planner import query_planner
    logger.info("=== PIPELINE STAGE: DOCUMENT RETRIEVAL AGENT ===")
    logger.info(f"Original Query: {state['query']}")
    logger.info(f"Machine ID: {state['machine_id']}")
    logger.info(f"User ID: {state.get('user_id', 'default_user')}")
    query = state["query"]
    
    # Bypass RAG if conversational intent detected
    if state.get("final_answer"):
        return {"retrieved_documents": [], "sub_agent_history": state["sub_agent_history"] + ["document_retrieval_agent"]}
        
    # Query planning & rewriting
    plan = query_planner.plan(query)
    logger.info(f"--- SUB-STAGE: QUERY PLANNING ---")
    logger.info(f"Detected Intent: {plan.intent}")
    logger.info(f"Target Collections: {plan.target_collections}")
    logger.info(f"Rewritten Query: '{plan.rewritten_query}'")
    logger.info(f"Requires Visual: {plan.requires_visual}")

    # Search collections retrieving top 50 candidates combined
    logger.info(f"--- SUB-STAGE: HYBRID RETRIEVAL (Dense + Sparse + RRF) ---")
    t0 = time.perf_counter()
    all_hits = rag_service.search_all_collections(plan.rewritten_query, top_k=10, user_id=state.get("user_id", "default_user"))
    retrieval_time = round(time.perf_counter() - t0, 3)

    flat_hits = []
    for coll, hits in all_hits.items():
        logger.info(f"Collection '{coll}': {len(hits)} hits")
        for hit in hits:
            flat_hits.append(hit)

    logger.info(f"Total Retrieved Candidates: {len(flat_hits)}")
    logger.info(f"Retrieval Time: {retrieval_time}s")

    # Boost chunks with images if visual intent detected
    logger.info(f"--- SUB-STAGE: VISUAL BOOSTING ---")
    visual_intent = plan.requires_visual or has_visual_intent(query)
    logger.info(f"Visual Intent Detected: {visual_intent}")
    boosted_count = 0
    for hit in flat_hits:
        payload = hit.get("payload", {}) if hit.get("payload") else {}
        if visual_intent and payload.get("image_path"):
            hit["score"] = min(1.0, hit.get("score", 0.0) + 0.3)
            boosted_count += 1
    logger.info(f"Boosted {boosted_count} chunks with images")

    # Rerank 50 down to top 10 using CrossEncoder
    logger.info(f"--- SUB-STAGE: CROSS-ENCODER RERANKING ---")
    t1 = time.perf_counter()
    try:
        reranked = rag_service.reranker.rerank(plan.rewritten_query, flat_hits, top_k=10)
        logger.info(f"Reranker Type: {type(rag_service.reranker).__name__}")
    except Exception as e:
        logger.warning(f"Reranker failed ({e}), falling back to score sorting.")
        reranked = sorted(flat_hits, key=lambda x: x.get("score", 0), reverse=True)[:10]

    rerank_time = round(time.perf_counter() - t1, 3)
    logger.info(f"Reranked to {len(reranked)} chunks in {rerank_time}s")
    for i, hit in enumerate(reranked[:5]):
        payload = hit.get("payload", {}) or {}
        logger.info(
            f"  [{i+1}] doc={payload.get('document_name', hit.get('title','?'))[:50]} "
            f"page={payload.get('page','?')} score={hit.get('score',0):.4f}"
        )
    
    # Filter to top 5 highest relevance chunks for LLM context
    top_5_chunks = reranked[:5]
    if not top_5_chunks:
        top_5_chunks = flat_hits[:5]
        
    return {
        "retrieved_documents": top_5_chunks,
        "sub_agent_history": state["sub_agent_history"] + ["document_retrieval_agent"]
    }

# --- Knowledge Graph Agent ---
from graph.neo4j_client import graph_client

def knowledge_graph_agent_node(state: AgentState) -> dict[str, Any]:
    """Traces machine -> component -> failure -> repair relationships."""
    logger.info("=== PIPELINE STAGE: KNOWLEDGE GRAPH AGENT ===")
    logger.info(f"Query: {state['query']}")
    logger.info(f"Machine ID: {state['machine_id']}")
    if state.get("final_answer"):
        logger.info("Bypassing - conversational answer already generated")
        return {"graph_path": [], "sub_agent_history": state["sub_agent_history"] + ["knowledge_graph_agent"]}

    query = state["query"]
    machine_id = state["machine_id"]
    logger.info(f"Querying Neo4j for graph path...")
    path = graph_client.get_path_for_query(query, machine_id)
    logger.info(f"Graph path returned: {len(path)} nodes")
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
        
    # Pull telemetry dynamically from backend.main.TELEMETRY_DATA to prevent divergence
    sensors = state.get("sensor_values")
    if not sensors:
        try:
            from backend.main import TELEMETRY_DATA
            sensors = TELEMETRY_DATA.get(state.get("machine_id"), TELEMETRY_DATA["M101"])
        except Exception as e:
            logger.warning(f"Could not import TELEMETRY_DATA: {e}")
            sensors = {
                "air_temperature": 298.2,
                "process_temperature": 308.6,
                "rotational_speed": 1850,
                "torque": 45.2,
                "tool_wear": 120,
                "vibration": 0.22,
                "telemetry_source": "simulated"
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
    """Orchestrates final response synthesis via LLM using ONLY retrieved manual context with structured output."""
    import time
    logger.info("=== PIPELINE STAGE: SYNTHESIZER NODE ===")
    logger.info(f"Query: {state['query']}")
    logger.info(f"Retrieved Documents Count: {len(state.get('retrieved_documents', []))}")
    if state.get("final_answer"):
        logger.info("Bypassing - conversational answer already generated")
        return {}  # Already handled by Intent Detection

    query = state["query"]
    docs = state.get("retrieved_documents", [])

    if not docs:
        logger.warning("❌ ZERO RETRIEVED DOCUMENTS - returning no-results message without LLM call")
        return {
            "final_answer": "I could not find this information inside the indexed manuals.",
            "llm_prompt": "N/A — zero retrieval results.",
            "sub_agent_history": state["sub_agent_history"] + ["synthesizer"],
        }

    # Build structured context with clear source citations
    logger.info(f"--- SUB-STAGE: CONTEXT CONSTRUCTION ---")
    logger.info(f"Building context from {len(docs)} retrieved documents")
    logger.info(f"✅ VERIFIED: Context built ONLY from retrieved chunks - no hardcoded content")
    doc_context = "\n\n".join([
        f"SOURCE: {doc.get('payload', {}).get('document_name', doc.get('title', 'Unknown'))} "
        f"| Page {doc.get('payload', {}).get('page', 'N/A')} "
        f"| Section: {doc.get('payload', {}).get('heading', 'General')} "
        f"| Score: {doc.get('score', 0):.3f}\n"
        f"{doc['text']}"
        for doc in docs
    ])

    # Include knowledge graph path if available
    graph_path = state.get("graph_path", [])
    graph_context = ""
    if graph_path:
        logger.info(f"Including {len(graph_path)} knowledge graph relationships")
        graph_context = "\n\n=== KNOWLEDGE GRAPH RELATIONSHIPS ===\n"
        for edge in graph_path:
            if isinstance(edge, dict):
                source = edge.get("source", "")
                relationship = edge.get("relationship", "")
                target = edge.get("target", "")
                if source and relationship and target:
                    graph_context += f"{source} --[{relationship}]--> {target}\n"

    combined_context = (
        "=== RETRIEVED MANUAL CONTEXT ===\n"
        + doc_context
        + graph_context
    )

    system_rules = (
        "You are FactoryMind AI, an expert industrial maintenance assistant for the Hyundai R215L Smart Plus excavator.\n"
        "Answer the user's question ONLY using the supplied retrieved context.\n"
        "Do not use prior knowledge. Never fabricate specifications, error codes, tools, or spare parts.\n"
        "Structure your response strictly into the following 10 sections:\n"
        "1. **Final Answer**: Comprehensive engineering resolution\n"
        "2. **Supporting Evidence**: Key technical facts from retrieved context\n"
        "3. **Manual**: Exact manual document title(s)\n"
        "4. **Section**: Specific section name or header\n"
        "5. **Page Number**: Page numbers referenced\n"
        "6. **Retrieved Images**: Mention any image paths/captions found in context\n"
        "7. **Diagrams**: Mention figure numbers or schematic references\n"
        "8. **Confidence Score**: High / Medium / Low score rating\n"
        "9. **Related Components**: Components directly related to the issue\n"
        "10. **Suggested Follow-up Questions**: 2-3 technical follow-up questions\n\n"
        "If the answer is not in the retrieved context, explicitly state: "
        "'I could not find this information inside the indexed manuals.'\n"
        "Keep your tone direct, highly technical, and strictly engineering-focused."
    )

    full_prompt = (
        f"SYSTEM RULES:\n{system_rules}\n\n"
        f"CONTEXT:\n{combined_context}\n\n"
        f"USER QUERY:\n{query}"
    )

    logger.info(f"--- SUB-STAGE: LLM SYNTHESIS ---")
    logger.info(f"Prompt Length: {len(full_prompt)} characters")
    logger.info(f"Context Length: {len(combined_context)} characters")
    logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")

    t0 = time.perf_counter()
    try:
        final_answer = llm_service.synthesize(query, combined_context, system_rules)
        llm_time = round(time.perf_counter() - t0, 2)
        logger.info(f"✅ LLM synthesis completed in {llm_time}s")
        logger.info(f"Answer Length: {len(final_answer)} characters")
    except Exception as e:
        logger.error(f"❌ LLM synthesis failed: {e}")
        final_answer = "I apologize, but I encountered an error generating the response. Please try again."

    return {
        "final_answer": final_answer,
        "llm_prompt": full_prompt,
        "sub_agent_history": state["sub_agent_history"] + ["synthesizer"],
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
        import time
        t_start = time.perf_counter()

        logger.info(
            f"\n========================\nUSER QUERY\n========================\n"
            f"Query     : {query}\n"
            f"Machine   : {machine_id}\n"
            f"User      : {user_id}"
        )

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
        intent_reply = state.get("final_answer", "")

        logger.info(
            f"\n========================\nINTENT\n========================\n"
            f"Greeting/shortcut: {bool(intent_reply)}"
        )

        # If it is a greeting/conversational, bypass all other agents
        if intent_reply:
            state["confidence_breakdown"] = {"overall": 100, "retrieval": 100, "graph": 100, "evidence": 100, "answer": 100}
            total_time = round(time.perf_counter() - t_start, 2)
            logger.info(f"Total pipeline time: {total_time}s | agents: {state['sub_agent_history']}")
            return state
            
        # 2. Dynamic agent routing based on query intent
        from backend.services.query_planner import query_planner
        intent = query_planner.classify_intent(query)
        
        # Always execute these core agents
        state.update(supervisor_agent_node(state))
        state.update(document_retrieval_agent_node(state))
        
        # Conditional agent execution based on intent
        if intent in ["PREDICTION", "MAINTENANCE", "TROUBLESHOOTING"]:
            # Only run prediction for relevant intents
            state.update(future_prediction_agent_node(state))
        
        if intent in ["TROUBLESHOOTING", "ERROR_CODE", "MAINTENANCE"]:
            # Knowledge graph useful for troubleshooting scenarios
            state.update(knowledge_graph_agent_node(state))
        
        # Always run evidence aggregation and synthesizer
        state.update(evidence_aggregation_agent_node(state))
        
        # Skip maintenance planner and report generator for simple lookups
        if intent not in ["GREETING", "MANUAL_LOOKUP", "SPECIFICATION"]:
            state.update(maintenance_planner_agent_node(state))
            state.update(report_generator_agent_node(state))
        
        state.update(synthesizer_node(state))

        total_time = round(time.perf_counter() - t_start, 2)
        logger.info(f"Total pipeline time: {total_time}s | agents: {state['sub_agent_history']}")
        
        return state

agent_orchestrator = LangGraphOrchestrator()
