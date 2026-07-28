"""Debug and diagnostic routes."""
from __future__ import annotations

import os
import json
import time
import logging
import urllib.request
import urllib.error
from fastapi import APIRouter
from typing import Any, Dict
from backend.config import settings
from backend.services.rag_service import rag_service
from backend.services.conversation_memory import conversation_memory
from backend.services.language_service import language_service

logger = logging.getLogger("factorymind")

router = APIRouter()


@router.get("/model")
async def debug_model():
    """Debug endpoint to check model file and configuration."""
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


@router.get("/groq")
async def debug_groq():
    """Debug endpoint to test Groq API connectivity."""
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

    result = _call_groq(model)
    if result["status"] == "error" and result.get("http_code") == 403:
        fallback_model = "llama3-8b-8192"
        if model != fallback_model:
            result["fallback_attempt"] = fallback_model
            fallback_result = _call_groq(fallback_model)
            result["fallback_result"] = fallback_result
    return result


@router.get("/llm")
async def debug_llm():
    """Debug endpoint to test LLM provider connectivity with detailed error classification."""
    t0 = time.perf_counter()
    provider = settings.LLM_PROVIDER.lower()
    
    diagnostics = {
        "provider": provider,
        "configured": True,
        "api_key_set": False,
        "model": None,
        "base_url": None,
        "test_result": None,
        "error_details": None
    }
    
    try:
        if provider == "groq":
            key = getattr(settings, "GROQ_API_KEY", None)
            diagnostics["api_key_set"] = bool(key and key.strip())
            diagnostics["model"] = settings.GROQ_MODEL
            
        elif provider in ("openai", "openai_compatible"):
            key = getattr(settings, "OPENAI_API_KEY", None)
            diagnostics["api_key_set"] = bool(key and key.strip())
            diagnostics["model"] = settings.OPENAI_MODEL
            diagnostics["base_url"] = settings.OPENAI_BASE_URL
            
        elif provider == "anthropic":
            key = getattr(settings, "ANTHROPIC_API_KEY", None)
            diagnostics["api_key_set"] = bool(key and key.strip())
            diagnostics["model"] = settings.ANTHROPIC_MODEL
            
        elif provider == "ollama":
            diagnostics["api_key_set"] = True  # Ollama doesn't need API key
            diagnostics["model"] = settings.OLLAMA_MODEL
            diagnostics["base_url"] = settings.OLLAMA_URL
            
        elif provider == "mock":
            diagnostics["api_key_set"] = True
            diagnostics["model"] = "mock"
            diagnostics["test_result"] = "skipped"
            return diagnostics
            
        else:
            diagnostics["configured"] = False
            diagnostics["error_details"] = f"Unknown provider: {provider}"
            return diagnostics
            
        # Test actual API call
        if provider in ("openai", "openai_compatible") and diagnostics["api_key_set"]:
            test_result = await _test_openai_compatible(
                diagnostics["base_url"],
                key,
                diagnostics["model"]
            )
            diagnostics["test_result"] = test_result
            
        elif provider == "groq" and diagnostics["api_key_set"]:
            test_result = await _test_groq(key, diagnostics["model"])
            diagnostics["test_result"] = test_result
            
        elif provider == "ollama":
            test_result = await _test_ollama(diagnostics["base_url"], diagnostics["model"])
            diagnostics["test_result"] = test_result
            
        return diagnostics
        
    except Exception as e:
        diagnostics["error_details"] = str(e)
        logger.error(f"LLM debug failed: {e}", exc_info=True)
        return diagnostics


async def _test_openai_compatible(base_url: str, api_key: str, model: str) -> dict:
    """Test OpenAI-compatible provider connectivity with detailed error classification."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    t0 = time.perf_counter()
    
    try:
        body = json.dumps({
            "model": model,
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Reply with OK"}]
        }).encode("utf-8")
        
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
            
        latency = round(time.perf_counter() - t0, 3)
        return {
            "status": "success",
            "response": payload.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "http_code": response.status,
            "latency_s": latency
        }
        
    except urllib.error.HTTPError as http_err:
        try:
            err_body = http_err.read().decode("utf-8")
            err_json = json.loads(err_body)
            error_obj = err_json.get("error", {})
            
            # Classify error type
            error_type = "Unknown"
            if http_err.code == 401:
                error_type = "Authentication Failed"
            elif http_err.code == 429:
                error_type = "Rate Limit Exceeded"
            elif http_err.code == 403:
                error_type = "Permission Denied"
            elif http_err.code == 404:
                error_type = "Model Not Found"
            elif http_err.code == 500:
                error_type = "Server Error"
            elif http_err.code == 503:
                error_type = "Service Unavailable"
                
            error_message = error_obj.get("message") if isinstance(error_obj, dict) else str(error_obj)
            
            return {
                "status": "error",
                "http_code": http_err.code,
                "error_type": error_type,
                "error_message": error_message,
                "full_response": err_body[:500]
            }
        except Exception:
            return {
                "status": "error",
                "http_code": http_err.code,
                "error_type": "HTTP Error",
                "error_message": f"HTTP {http_err.code}"
            }
            
    except urllib.error.URLError as url_err:
        return {
            "status": "error",
            "error_type": "Network Error",
            "error_message": str(url_err)
        }
    except Exception as exc:
        return {
            "status": "error",
            "error_type": "Unexpected Error",
            "error_message": str(exc)
        }


async def _test_groq(api_key: str, model: str) -> dict:
    """Test Groq provider connectivity."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    return await _test_openai_compatible(url, api_key, model)


async def _test_ollama(base_url: str, model: str) -> dict:
    """Test Ollama provider connectivity."""
    url = f"{base_url.rstrip('/')}/api/chat"
    t0 = time.perf_counter()
    
    try:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": "Reply with OK"}],
            "stream": False
        }).encode("utf-8")
        
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
            
        latency = round(time.perf_counter() - t0, 3)
        return {
            "status": "success",
            "response": payload.get("message", {}).get("content", ""),
            "http_code": response.status,
            "latency_s": latency
        }
        
    except urllib.error.HTTPError as http_err:
        return {
            "status": "error",
            "http_code": http_err.code,
            "error_type": "HTTP Error",
            "error_message": f"HTTP {http_err.code}"
        }
    except urllib.error.URLError as url_err:
        return {
            "status": "error",
            "error_type": "Network Error",
            "error_message": str(url_err)
        }
    except Exception as exc:
        return {
            "status": "error",
            "error_type": "Unexpected Error",
            "error_message": str(exc)
        }


@router.post("/retrieve")
async def debug_retrieve(req: Dict[str, Any], current_user: Dict[str, Any] = None):
    """Debug endpoint to test retrieval without LLM synthesis."""
    from backend.models.schemas import RetrieveDebugRequest
    from backend.services.rag_service import rag_service
    
    # Convert dict to Pydantic model
    request = RetrieveDebugRequest(**req)
    query = request.query
    top_k = request.top_k or 8
    
    # Get current user if not provided
    if current_user is None:
        current_user = {"uid": "default_user"}
    
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


@router.get("/dashboard")
async def debug_dashboard():
    """Comprehensive debug dashboard showing system status and metrics."""
    t0 = time.perf_counter()
    
    # Vector store stats
    vector_stats = {}
    try:
        vector_stats = rag_service.vector_store.get_stats()
    except Exception as e:
        logger.error(f"Failed to get vector store stats: {e}")
        vector_stats = {"error": str(e)}
    
    # RAG cache status
    cache_status = {
        "cached_queries": len(rag_service._answer_cache),
        "cache_keys": list(rag_service._answer_cache.keys())[:5]  # Show first 5 keys
    }
    
    # Conversation memory status
    conv_memory_status = {
        "active_users": len(conversation_memory.conversations),
        "total_conversations": sum(len(msgs) for msgs in conversation_memory.conversations.values())
    }
    
    # Language service status
    lang_service_status = {
        "supported_languages": list(language_service.LANGUAGE_NAMES.keys()),
        "default_language": language_service.default_language
    }
    
    # LLM configuration
    llm_config = {
        "provider": settings.LLM_PROVIDER,
        "fallback_provider": settings.LLM_FALLBACK_PROVIDER,
        "max_retries": settings.LLM_MAX_RETRIES,
        "retry_delay": settings.LLM_RETRY_DELAY
    }
    
    # Retrieval configuration
    retrieval_config = {
        "top_k_per_collection": 30,
        "rerank_top_k": 8,
        "min_relevance_score": settings.RAG_MIN_RELEVANCE_SCORE,
        "embedding_model": settings.EMBEDDING_MODEL,
        "sparse_model": settings.SPARSE_EMBEDDING_MODEL,
        "reranker_model": settings.RERANKER_MODEL
    }
    
    elapsed = round(time.perf_counter() - t0, 3)
    
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "response_time_ms": round(elapsed * 1000, 2),
        "vector_store": vector_stats,
        "rag_cache": cache_status,
        "conversation_memory": conv_memory_status,
        "language_service": lang_service_status,
        "llm_config": llm_config,
        "retrieval_config": retrieval_config,
        "system_status": "healthy"
    }


@router.get("/conversation/{user_id}")
async def debug_conversation(user_id: str):
    """Debug endpoint to view conversation history for a specific user."""
    history = conversation_memory.get_conversation_history(user_id, max_messages=10)
    summary = conversation_memory.get_conversation_summary(user_id)
    
    formatted_history = []
    for msg in history:
        formatted_history.append({
            "role": msg["role"],
            "content": msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"],
            "timestamp": msg["timestamp"].isoformat()
        })
    
    return {
        "user_id": user_id,
        "summary": summary,
        "history": formatted_history
    }


@router.delete("/conversation/{user_id}")
async def clear_conversation(user_id: str):
    """Debug endpoint to clear conversation history for a specific user."""
    conversation_memory.clear_conversation(user_id)
    return {
        "user_id": user_id,
        "status": "cleared"
    }


@router.get("/language/detect")
async def debug_language_detection(query: str):
    """Debug endpoint to test language detection."""
    detected_lang = language_service.detect_language(query)
    language_name = language_service.LANGUAGE_NAMES.get(detected_lang, detected_lang)
    instruction = language_service.get_system_prompt_language(detected_lang)
    
    return {
        "query": query,
        "detected_language": detected_lang,
        "language_name": language_name,
        "system_instruction": instruction
    }
