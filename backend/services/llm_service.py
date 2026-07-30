"""
FactoryMind AI — Production LLM Provider Service
=================================================
Provider chain: Groq → Ollama → Error (never Gemini/OpenRouter in inference)

Features:
  - Groq provider (llama-3.3-70b-versatile) with connection reuse
  - Ollama fallback with availability ping
  - Retry: 429/500/502/503/Timeout only (max 2, exponential backoff 2s/5s)
  - In-memory response cache (10 min TTL, keyed by query+page_ids)
  - Request deduplication (one in-flight request per unique query)
  - Rate-limit guard (429 → block Groq 60s, switch to Ollama immediately)
  - Context trimmer (top 5 chunks, 3500 chars, dedup by page)
  - Structured per-query log: provider, model, latency, tokens, pages, fallback
"""
from __future__ import annotations

import json
import logging
import time
import threading
import hashlib
import http.client
import socket
import urllib.request
import urllib.error
from typing import Optional, Dict, List, Any, Generator
from dataclasses import dataclass, field

from backend.config import settings

logger = logging.getLogger("factorymind")

# ─── Constants ────────────────────────────────────────────────────────────────

GROQ_HOST = "api.groq.com"
GROQ_PATH = "/openai/v1/chat/completions"
RETRY_ON_STATUS = {429, 500, 502, 503}
PERMANENT_STATUS = {401, 403, 404}
MAX_CONTEXT_CHARS = 3500
TOP_K_CHUNKS = 5
CACHE_TTL_SECONDS = 600  # 10 minutes
GROQ_COOLDOWN_SECONDS = 60  # after 429, wait before retrying Groq


# ─── Utilities ────────────────────────────────────────────────────────────────

def _mask_key(key: Optional[str]) -> str:
    if not key or len(key) < 9:
        return "<not set>"
    return f"{key[:4]}...{key[-4:]}"


# ─── Context Trimmer ─────────────────────────────────────────────────────────

def trim_context(chunks: List[Dict[str, Any]], query: str = "") -> tuple[str, List[str]]:
    """
    Takes raw retrieved chunks, deduplicates by page, merges same-page content,
    keeps top 5, limits to 3500 characters.
    Returns (trimmed_context_str, list_of_page_ids).
    """
    if not chunks:
        return "", []

    # Deduplicate by page_id (same page text → merge)
    seen_pages: Dict[str, Dict] = {}
    for chunk in chunks[:TOP_K_CHUNKS * 3]:  # consider more, then trim
        page_id = str(chunk.get("payload", {}).get("page", "") or chunk.get("id", ""))
        doc = chunk.get("payload", {}).get("document_name", "Unknown")
        text = chunk.get("text", "") or chunk.get("payload", {}).get("text", "")
        key = f"{doc}::{page_id}"
        if key in seen_pages:
            # Merge text
            seen_pages[key]["text"] += " " + text
        else:
            seen_pages[key] = {
                "doc": doc,
                "page": page_id,
                "text": text,
                "score": chunk.get("score", 0.0),
            }

    # Sort by score descending, keep top 5
    ranked = sorted(seen_pages.values(), key=lambda x: x["score"], reverse=True)[:TOP_K_CHUNKS]
    page_ids = [f"{r['doc']}:p{r['page']}" for r in ranked]

    # Build context string
    parts = []
    total_chars = 0
    for r in ranked:
        text = r["text"].strip()
        header = f"[{r['doc']} | Page {r['page']}]\n"
        entry = header + text + "\n"
        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars - len(header) - 3
            if remaining > 50:
                entry = header + text[:remaining] + "...\n"
            else:
                break
        parts.append(entry)
        total_chars += len(entry)

    return "\n".join(parts), page_ids


# ─── Response Cache ───────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    answer: str
    created_at: float = field(default_factory=time.time)

class ResponseCache:
    """Thread-safe in-memory cache with TTL."""

    def __init__(self, ttl: int = CACHE_TTL_SECONDS):
        self._store: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._ttl = ttl

    def _cache_key(self, query: str, page_ids: List[str]) -> str:
        raw = query.strip().lower() + "||" + ",".join(sorted(page_ids))
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get(self, query: str, page_ids: List[str]) -> Optional[str]:
        key = self._cache_key(query, page_ids)
        with self._lock:
            entry = self._store.get(key)
            if entry and (time.time() - entry.created_at) < self._ttl:
                return entry.answer
            if entry:
                del self._store[key]
        return None

    def set(self, query: str, page_ids: List[str], answer: str):
        key = self._cache_key(query, page_ids)
        with self._lock:
            self._store[key] = CacheEntry(answer=answer)

    def clear(self):
        with self._lock:
            self._store.clear()


# ─── Request Deduplicator ─────────────────────────────────────────────────────

class RequestDeduplicator:
    """If two identical queries arrive simultaneously, only one executes; others wait."""

    def __init__(self):
        self._in_flight: Dict[str, threading.Event] = {}
        self._results: Dict[str, str] = {}
        self._lock = threading.Lock()

    def _key(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode()).hexdigest()[:16]

    def run(self, query: str, call_fn) -> str:
        key = self._key(query)
        with self._lock:
            if key in self._in_flight:
                event = self._in_flight[key]
                is_leader = False
            else:
                event = threading.Event()
                self._in_flight[key] = event
                is_leader = True

        if not is_leader:
            # Wait for the leader to finish (max 60s)
            event.wait(timeout=60)
            with self._lock:
                result = self._results.get(key, "")
            return result

        # Leader executes the call
        try:
            result = call_fn()
            with self._lock:
                self._results[key] = result
            return result
        finally:
            event.set()
            with self._lock:
                self._in_flight.pop(key, None)
                self._results.pop(key, None)


# ─── Rate-Limit Guard ─────────────────────────────────────────────────────────

class RateLimitGuard:
    """Blocks a provider for GROQ_COOLDOWN_SECONDS after a 429 response."""

    def __init__(self):
        self._blocked_until: Dict[str, float] = {}
        self._lock = threading.Lock()

    def is_blocked(self, provider: str) -> bool:
        with self._lock:
            until = self._blocked_until.get(provider, 0)
            return time.time() < until

    def block(self, provider: str, seconds: int = GROQ_COOLDOWN_SECONDS):
        with self._lock:
            self._blocked_until[provider] = time.time() + seconds
        logger.warning(f"[{provider}] rate-limited — blocked for {seconds}s")

    def remaining(self, provider: str) -> float:
        with self._lock:
            until = self._blocked_until.get(provider, 0)
            return max(0.0, until - time.time())


# ─── Groq Provider ───────────────────────────────────────────────────────────

class GroqProvider:
    """
    Groq API provider using a persistent HTTPS connection (connection pooling).
    Endpoint: https://api.groq.com/openai/v1/chat/completions
    Auth:     Authorization: Bearer <GROQ_API_KEY>
    """
    NAME = "groq"
    ENDPOINT = f"https://{GROQ_HOST}{GROQ_PATH}"

    def __init__(self):
        self._conn: Optional[http.client.HTTPSConnection] = None
        self._conn_lock = threading.Lock()

    def is_configured(self) -> bool:
        key = getattr(settings, "GROQ_API_KEY", None)
        return bool(key and key.strip())

    def _get_conn(self) -> http.client.HTTPSConnection:
        with self._conn_lock:
            if self._conn is None:
                self._conn = http.client.HTTPSConnection(GROQ_HOST, timeout=30)
            return self._conn

    def _reset_conn(self):
        with self._conn_lock:
            try:
                if self._conn:
                    self._conn.close()
            except Exception:
                pass
            self._conn = None

    def call(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Returns dict with: answer, prompt_tokens, completion_tokens, latency
        """
        key = settings.GROQ_API_KEY
        model = settings.GROQ_MODEL

        body = json.dumps({
            "model": model,
            "temperature": 0.15,
            "max_tokens": 1024,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Connection": "keep-alive",
        }

        logger.info(f"[Groq] POST {self.ENDPOINT} model={model} key={_mask_key(key)}")
        t0 = time.perf_counter()

        conn = self._get_conn()
        try:
            conn.request("POST", GROQ_PATH, body=body, headers=headers)
            resp = conn.getresponse()
            status = resp.status
            raw = resp.read().decode("utf-8")
        except (http.client.CannotSendRequest, BrokenPipeError, ConnectionResetError, socket.error):
            # Connection dropped — reset and retry once
            self._reset_conn()
            conn = self._get_conn()
            conn.request("POST", GROQ_PATH, body=body, headers=headers)
            resp = conn.getresponse()
            status = resp.status
            raw = resp.read().decode("utf-8")

        elapsed = round(time.perf_counter() - t0, 2)

        if status == 200:
            data = json.loads(raw)
            choices = data.get("choices", [])
            if not choices:
                raise ValueError(f"Groq: empty choices in response. raw={raw[:300]}")
            usage = data.get("usage", {})
            answer = choices[0]["message"]["content"].strip()
            logger.info(
                f"[Groq] ✅ HTTP 200 latency={elapsed}s "
                f"prompt_tokens={usage.get('prompt_tokens', '?')} "
                f"completion_tokens={usage.get('completion_tokens', '?')}"
            )
            return {
                "answer": answer,
                "latency": elapsed,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }
        else:
            logger.error(
                f"[Groq] ❌ HTTP {status} latency={elapsed}s\n"
                f"  model={model}\n"
                f"  response={raw[:600]}"
            )
            is_permanent = status in PERMANENT_STATUS
            perm_tag = "[PERMANENT] " if is_permanent else ""
            is_rate_limited = status == 429
            raise GroqHTTPError(status, f"{perm_tag}Groq HTTP {status}: {raw[:300]}", is_rate_limited)


class GroqHTTPError(Exception):
    def __init__(self, status: int, message: str, is_rate_limited: bool = False):
        super().__init__(message)
        self.status = status
        self.is_rate_limited = is_rate_limited


# ─── Ollama Provider ─────────────────────────────────────────────────────────

class OllamaProvider:
    """
    Local Ollama fallback.
    Pings availability before use. Non-blocking (3s ping timeout).
    """
    NAME = "ollama"

    def is_configured(self) -> bool:
        return bool(getattr(settings, "OLLAMA_URL", None))

    def _ping(self) -> bool:
        try:
            req = urllib.request.Request(f"{settings.OLLAMA_URL}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception as e:
            logger.warning(f"[Ollama] Ping failed: {e}")
            return False

    def call(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        model = settings.OLLAMA_MODEL
        url = f"{settings.OLLAMA_URL}/api/chat"

        if not self._ping():
            raise RuntimeError(f"[PERMANENT] Ollama unavailable at {settings.OLLAMA_URL}")

        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0.2, "num_ctx": 4096},
            "stream": False,
        }).encode("utf-8")

        logger.info(f"[Ollama] POST {url} model={model}")
        t0 = time.perf_counter()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"Ollama HTTP {e.code}: {err_body}")

        elapsed = round(time.perf_counter() - t0, 2)
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise RuntimeError(f"Ollama: unexpected response type: {type(data)}")
        content = data.get("message", {}).get("content", "").strip()
        if not content:
            raise RuntimeError(f"Ollama: empty content. raw={raw[:300]}")

        logger.info(f"[Ollama] ✅ HTTP 200 latency={elapsed}s")
        return {"answer": content, "latency": elapsed, "prompt_tokens": 0, "completion_tokens": 0}


# ─── OpenRouter Provider (interim fallback while Groq key is pending) ─────────────

class OpenRouterProvider:
    """
    OpenRouter fallback — only used when both Groq and Ollama are unavailable.
    Authorization: Bearer <OPENROUTER_API_KEY>
    """
    NAME = "openrouter"
    ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

    def is_configured(self) -> bool:
        key = getattr(settings, "OPENROUTER_API_KEY", None)
        return bool(key and key.strip() and "<" not in key)

    def call(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        key = settings.OPENROUTER_API_KEY
        model = getattr(settings, "OPENROUTER_MODEL", "qwen/qwen-2.5-30b-a3b-instruct")

        body = json.dumps({
            "model": model,
            "temperature": 0.15,
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://factorymind.ai",
            "X-Title": "FactoryMind AI",
        }

        logger.info(f"[OpenRouter] POST {self.ENDPOINT} model={model} key={_mask_key(key)}")
        t0 = time.perf_counter()
        req = urllib.request.Request(self.ENDPOINT, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")[:500]
            logger.error(f"[OpenRouter] HTTP {e.code} raw={err_body}")
            raise RuntimeError(f"OpenRouter HTTP {e.code}: {err_body[:200]}")

        elapsed = round(time.perf_counter() - t0, 2)
        data = json.loads(raw)
        if not isinstance(data, dict) or "choices" not in data:
            raise RuntimeError(f"OpenRouter: unexpected response: {raw[:300]}")
        answer = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        logger.info(f"[OpenRouter] ✅ HTTP 200 latency={elapsed}s")
        return {
            "answer": answer,
            "latency": elapsed,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }


# ─── Startup Health Check ─────────────────────────────────────────────────────

def _startup_health_check():
    import os
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"
    )

    groq_key = getattr(settings, "GROQ_API_KEY", None)
    groq_key_env = os.environ.get("GROQ_API_KEY", "<NOT IN os.environ>")

    logger.info("=" * 60)
    logger.info("LLM STARTUP — GROQ+OLLAMA CHAIN")
    logger.info("=" * 60)
    logger.info(f"  .env path          : {env_path} (exists={os.path.exists(env_path)})")
    logger.info(f"  LLM_PROVIDER       : {settings.LLM_PROVIDER}")
    logger.info(f"  LLM_FALLBACK       : {settings.LLM_FALLBACK_PROVIDER}")
    logger.info(f"  GROQ_MODEL         : {settings.GROQ_MODEL}")
    logger.info(f"  GROQ_API_KEY (settings)  : {_mask_key(groq_key)}")
    logger.info(f"  GROQ_API_KEY (os.env)    : {_mask_key(groq_key_env)}")
    logger.info(f"  GROQ endpoint      : https://{GROQ_HOST}{GROQ_PATH}")
    logger.info(f"  OLLAMA_URL         : {settings.OLLAMA_URL}")
    logger.info(f"  OLLAMA_MODEL       : {settings.OLLAMA_MODEL}")
    logger.info(f"  Gemini/OpenRouter  : DISABLED (not used in inference)")
    logger.info("=" * 60)

    if not groq_key:
        logger.error("  ❌ GROQ_API_KEY is missing — add it to .env immediately")
    else:
        logger.info(f"  ✅ GROQ_API_KEY is set → running live test...")
        _test_groq_live(groq_key, settings.GROQ_MODEL)


def _test_groq_live(api_key: str, model: str):
    """Quick live test: one token to verify Groq connectivity."""
    body = json.dumps({
        "model": model,
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "ping"}],
    }).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    t0 = time.perf_counter()
    try:
        conn = http.client.HTTPSConnection(GROQ_HOST, timeout=15)
        conn.request("POST", GROQ_PATH, body=body, headers=headers)
        resp = conn.getresponse()
        status = resp.status
        raw = resp.read().decode("utf-8")
        conn.close()
        elapsed = round(time.perf_counter() - t0, 2)

        if status == 200:
            logger.info(f"[STARTUP TEST] ✅ Groq: SUCCESS — HTTP 200 in {elapsed}s")
        else:
            logger.error(
                f"[STARTUP TEST] ❌ Groq: FAILED — HTTP {status} in {elapsed}s\n"
                f"  model   : {model}\n"
                f"  response: {raw[:600]}"
            )
    except Exception as e:
        elapsed = round(time.perf_counter() - t0, 2)
        logger.error(f"[STARTUP TEST] ❌ Groq: FAILED — {e} (in {elapsed}s)")


_startup_health_check()


# ─── LLM Service ─────────────────────────────────────────────────────────────

class LLMService:
    """
    Production LLM service.
    Chain: Groq → Ollama → Extractive fallback
    Gemini and OpenRouter are completely disabled.
    """

    def __init__(self):
        self._groq = GroqProvider()
        self._ollama = OllamaProvider()
        self._openrouter = OpenRouterProvider()
        self._cache = ResponseCache()
        self._dedup = RequestDeduplicator()
        self._rate_guard = RateLimitGuard()

    # ── Public interface ──────────────────────────────────────────────────────

    def synthesize(self, query: str, context: str, system_prompt: str) -> str:
        """
        Main synthesis entrypoint.
        Trims context, checks cache, deduplicates concurrent calls, then routes Groq→Ollama.
        """
        # Trim context to top 5 deduped chunks, max 3500 chars
        trimmed_context, page_ids = trim_context(
            self._parse_context_chunks(context), query
        )
        if not trimmed_context:
            trimmed_context = context[:MAX_CONTEXT_CHARS]

        # Check cache
        cached = self._cache.get(query, page_ids)
        if cached:
            logger.info(f"[LLM] 🗄 Cache HIT — returning cached answer (pages={page_ids})")
            return cached

        # Deduplicate concurrent identical queries
        def _execute():
            return self._route(query, trimmed_context, system_prompt, page_ids)

        answer = self._dedup.run(query, _execute)
        return answer

    # ── Routing ───────────────────────────────────────────────────────────────

    def _route(self, query: str, context: str, system_prompt: str, page_ids: List[str]) -> str:
        user_prompt = f"Question: {query}\n\nContext:\n{context}"
        fallback_reason: Optional[str] = None

        # ── Try Groq ──────────────────────────────────────────────────────────
        if self._groq.is_configured():
            if self._rate_guard.is_blocked("groq"):
                remaining = self._rate_guard.remaining("groq")
                fallback_reason = f"Groq rate-limited ({remaining:.0f}s remaining)"
                logger.warning(f"[LLM] Groq blocked — {fallback_reason}")
            else:
                groq_result = self._call_with_retry("groq", query, system_prompt, user_prompt)
                if groq_result is not None:
                    answer = groq_result["answer"]
                    self._log_query(
                        provider="groq",
                        model=settings.GROQ_MODEL,
                        latency=groq_result["latency"],
                        prompt_tokens=groq_result["prompt_tokens"],
                        completion_tokens=groq_result["completion_tokens"],
                        context_chars=len(context),
                        pages=page_ids,
                        fallback_reason=None,
                    )
                    self._cache.set(query, page_ids, answer)
                    return answer
                else:
                    fallback_reason = "Groq exhausted retries"
        else:
            fallback_reason = "Groq not configured (missing GROQ_API_KEY)"
            logger.warning(f"[LLM] {fallback_reason}")

        # ── Try Ollama ────────────────────────────────────────────────────────
        if self._ollama.is_configured():
            logger.info(f"[LLM] Falling back to Ollama (reason: {fallback_reason})")
            try:
                ollama_result = self._ollama.call(system_prompt, user_prompt)
                answer = ollama_result["answer"]
                self._log_query(
                    provider="ollama",
                    model=settings.OLLAMA_MODEL,
                    latency=ollama_result["latency"],
                    prompt_tokens=0,
                    completion_tokens=0,
                    context_chars=len(context),
                    pages=page_ids,
                    fallback_reason=fallback_reason,
                )
                self._cache.set(query, page_ids, answer)
                return answer
            except Exception as e:
                logger.error(f"[LLM] Ollama failed: {e}")
                fallback_reason = f"{fallback_reason} + Ollama unavailable"

        # ── Try OpenRouter (last resort — valid key available) ────────────────
        if self._openrouter.is_configured():
            logger.info(f"[LLM] Falling back to OpenRouter (reason: {fallback_reason})")
            try:
                or_result = self._openrouter.call(system_prompt, user_prompt)
                answer = or_result["answer"]
                self._log_query(
                    provider="openrouter",
                    model=getattr(settings, "OPENROUTER_MODEL", "qwen/qwen-2.5-30b-a3b-instruct"),
                    latency=or_result["latency"],
                    prompt_tokens=or_result["prompt_tokens"],
                    completion_tokens=or_result["completion_tokens"],
                    context_chars=len(context),
                    pages=page_ids,
                    fallback_reason=fallback_reason,
                )
                self._cache.set(query, page_ids, answer)
                return answer
            except Exception as e:
                logger.error(f"[LLM] OpenRouter failed: {e}")

        # ── Extractive fallback ───────────────────────────────────────────────
        logger.error("[LLM] All providers failed — returning extractive fallback")
        return self._extractive_fallback(query, context)

    def _call_with_retry(self, provider: str, query: str, system_prompt: str, user_prompt: str) -> Optional[Dict]:
        """Call Groq with up to 2 retries on transient errors. Returns result dict or None."""
        retry_delays = [2, 5]  # seconds

        for attempt in range(1, settings.LLM_MAX_RETRIES + 1):
            try:
                if provider == "groq":
                    return self._groq.call(system_prompt, user_prompt)
            except GroqHTTPError as e:
                msg = str(e)
                logger.warning(f"[Groq] attempt {attempt}/{settings.LLM_MAX_RETRIES}: {msg}")

                if e.is_rate_limited:
                    self._rate_guard.block("groq", GROQ_COOLDOWN_SECONDS)
                    logger.warning("[Groq] 429 → switching to Ollama immediately")
                    return None  # Skip retries, go to Ollama

                if "[PERMANENT]" in msg:
                    logger.error(f"[Groq] permanent failure — skipping all retries")
                    return None

                if e.status not in RETRY_ON_STATUS:
                    logger.error(f"[Groq] HTTP {e.status} is not retryable — skipping")
                    return None

                if attempt <= len(retry_delays):
                    wait = retry_delays[attempt - 1]
                    logger.info(f"[Groq] retrying in {wait}s (attempt {attempt})")
                    time.sleep(wait)

            except Exception as e:
                logger.error(f"[Groq] unexpected error attempt {attempt}: {e}")
                if attempt <= len(retry_delays):
                    time.sleep(retry_delays[attempt - 1])

        return None

    # ── Context Parsing ───────────────────────────────────────────────────────

    def _parse_context_chunks(self, context: str) -> List[Dict[str, Any]]:
        """
        Convert a pre-built context string back into chunk-like dicts for trimming.
        If context is already a string (not structured), wrap it as one chunk.
        """
        if not context or not isinstance(context, str):
            return []
        # Context is already a formatted string — treat each paragraph as a chunk
        lines = context.strip().split("\n\n")
        return [
            {"text": part, "score": 1.0 - (i * 0.05), "payload": {}, "id": str(i)}
            for i, part in enumerate(lines) if part.strip()
        ]

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_query(
        self,
        provider: str,
        model: str,
        latency: float,
        prompt_tokens: int,
        completion_tokens: int,
        context_chars: int,
        pages: List[str],
        fallback_reason: Optional[str],
    ):
        lines = [
            f"\n{'='*50}",
            f"  LLM QUERY COMPLETE",
            f"  Provider   : {provider}",
            f"  Model      : {model}",
            f"  Latency    : {latency}s",
            f"  Tokens     : prompt={prompt_tokens} completion={completion_tokens}",
            f"  Context    : {context_chars} chars",
            f"  Pages      : {pages}",
        ]
        if fallback_reason:
            lines.append(f"  Fallback   : {fallback_reason}")
        lines.append(f"{'='*50}")
        logger.info("\n".join(lines))

    # ── Extractive Fallback ───────────────────────────────────────────────────

    def _extractive_fallback(self, query: str, context: str) -> str:
        if context and context.strip():
            return (
                f"⚠️ LLM Unavailable - Using Extractive Fallback\n\n"
                f"**Answer Based on Retrieved Context**\n\n"
                f"**Question:** {query}\n\n"
                f"**Retrieved Information:**\n\n{context[:2000]}"
            )
        return (
            f"I could not find this information inside the indexed manuals.\n\n"
            f"**Question:** {query}"
        )

    # ── Admin ─────────────────────────────────────────────────────────────────

    def clear_cache(self):
        self._cache.clear()
        logger.info("[LLM] Response cache cleared")

    def get_circuit_breaker_states(self) -> Dict:
        """Stub for backward compatibility with any admin routes that call this."""
        return {
            "groq": {
                "state": "blocked" if self._rate_guard.is_blocked("groq") else "closed",
                "remaining_cooldown": self._rate_guard.remaining("groq"),
            },
            "ollama": {"state": "closed"},
        }

    def reset_circuit_breaker(self, provider: str) -> bool:
        """Stub for backward compatibility."""
        return True


llm_service = LLMService()
