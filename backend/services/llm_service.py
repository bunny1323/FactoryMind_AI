from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.error

from backend.config import settings

logger = logging.getLogger("factorymind")


class LLMProviderError(Exception):
    """Raised when a provider API call fails after exhausting retries."""
    pass


def _validate_provider_keys():
    """
    Called at startup. Logs the configured LLM provider.
    Raises RuntimeError for missing keys so the server fails fast.
    """
    provider = settings.LLM_PROVIDER.lower()

    if provider == "groq":
        key = getattr(settings, "GROQ_API_KEY", None)
        if not key or not key.strip():
            raise RuntimeError(
                "GROQ_API_KEY is missing or empty. Set it in your .env file and restart."
            )
        logger.info(f"LLM Provider: groq  model={settings.GROQ_MODEL}")

    elif provider in ("openai", "openai_compatible"):
        key = getattr(settings, "OPENAI_API_KEY", None)
        base_url = getattr(settings, "OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = getattr(settings, "OPENAI_MODEL", "gpt-4o-mini")
        if not key or not key.strip():
            raise RuntimeError(
                f"OPENAI_API_KEY is missing or empty (needed for LLM_PROVIDER={provider}). "
                "Set it in your .env file and restart."
            )
        logger.info(
            f"LLM Provider: {provider}  model={model}  base_url={base_url}  "
            f"key_prefix={key[:12]}..."
        )

    elif provider == "anthropic":
        key = getattr(settings, "ANTHROPIC_API_KEY", None)
        if not key or not key.strip():
            raise RuntimeError(
                "ANTHROPIC_API_KEY is missing or empty. Set it in your .env file and restart."
            )
        logger.info(f"LLM Provider: anthropic  model={settings.ANTHROPIC_MODEL}")

    elif provider == "ollama":
        logger.info(
            f"LLM Provider: ollama  model={settings.OLLAMA_MODEL}  "
            f"url={settings.OLLAMA_URL}  (no key needed)"
        )

    elif provider == "mock":
        logger.info("LLM Provider: mock (no real LLM — responses will be extractive fallback only)")

    else:
        logger.warning(f"LLM Provider: unknown value '{provider}' in .env — will use extractive fallback")


# Run validation at import time so the server fails fast.
_validate_provider_keys()



class LLMService:
    def synthesize(self, query: str, context: str, system_prompt: str) -> str:
        """
        Route the request to the configured LLM provider.
        On failure, log the actual provider error and return an honest extractive fallback.
        Never return hardcoded canned answers.
        """
        provider = settings.LLM_PROVIDER.lower()

        if provider == "mock":
            return (
                f"[MOCK] Based on retrieved context for '{query}':\n\n"
                f"{context[:500]}..."
            )

        t0 = time.perf_counter()

        if provider == "groq":
            active_model = getattr(settings, "GROQ_MODEL", "n/a")
        elif provider in ("openai", "openai_compatible"):
            active_model = getattr(settings, "OPENAI_MODEL", "n/a")
        elif provider == "ollama":
            active_model = getattr(settings, "OLLAMA_MODEL", "n/a")
        elif provider == "anthropic":
            active_model = getattr(settings, "ANTHROPIC_MODEL", "n/a")
        else:
            active_model = "n/a"

        try:
            logger.info(
                "\n========================\nCALLING LLM\n========================\n"
                f"Provider : {provider}\n"
                f"Model    : {active_model}\n"
                f"Query    : {query[:120]}"
            )

            if provider == "groq" and getattr(settings, "GROQ_API_KEY", None):
                answer = self._call_openai_compatible(
                    url="https://api.groq.com/openai/v1/chat/completions",
                    api_key=settings.GROQ_API_KEY,
                    model=settings.GROQ_MODEL,
                    system_prompt=system_prompt,
                    user_prompt=f"Question: {query}\n\nContext:\n{context}",
                )
            elif provider == "openai" and getattr(settings, "OPENAI_API_KEY", None):
                base_url = getattr(settings, "OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
                answer = self._call_openai_compatible(
                    url=f"{base_url}/chat/completions",
                    api_key=settings.OPENAI_API_KEY,
                    model=settings.OPENAI_MODEL,
                    system_prompt=system_prompt,
                    user_prompt=f"Question: {query}\n\nContext:\n{context}",
                )
            elif provider == "openai_compatible" and getattr(settings, "OPENAI_API_KEY", None):
                # Any OpenAI-compatible provider via OPENAI_BASE_URL
                base_url = getattr(settings, "OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
                answer = self._call_openai_compatible(
                    url=f"{base_url}/chat/completions",
                    api_key=settings.OPENAI_API_KEY,
                    model=settings.OPENAI_MODEL,
                    system_prompt=system_prompt,
                    user_prompt=f"Question: {query}\n\nContext:\n{context}",
                )
            elif provider == "ollama":
                answer = self._call_ollama(
                    system_prompt=system_prompt,
                    user_prompt=f"Question: {query}\n\nContext:\n{context}",
                )
            elif provider == "anthropic" and getattr(settings, "ANTHROPIC_API_KEY", None):
                answer = self._call_anthropic(
                    system_prompt=system_prompt,
                    user_prompt=f"Question: {query}\n\nContext:\n{context}",
                )
            else:
                raise LLMProviderError(
                    f"Provider '{provider}' is not configured correctly. "
                    "Check your .env file for the required API key."
                )

            elapsed = round(time.perf_counter() - t0, 2)
            logger.info(
                "\n========================\nGROQ RESPONSE\n========================\n"
                f"Latency  : {elapsed}s\n"
                f"Preview  : {answer[:500]}"
            )
            return answer

        except LLMProviderError as exc:
            logger.error(
                f"LLM provider '{provider}' call failed: {exc}\n"
                "Returning honest extractive fallback — NOT a canned answer."
            )
            if context and context.strip():
                return (
                    "⚠️ **LLM Unavailable** — Showing top retrieved context directly:\n\n"
                    + context[:1200]
                    + "\n\n*(LLM synthesis was skipped due to a provider error. "
                    "The above is raw retrieved text from the indexed manuals.)*"
                )
            return (
                "⚠️ **LLM Unavailable** and no relevant context was retrieved. "
                "Please try again or check server logs."
            )

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _call_openai_compatible(
        self, url: str, api_key: str, model: str, system_prompt: str, user_prompt: str
    ) -> str:
        body = json.dumps({
            "model": model,
            "temperature": 0.15,
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload["choices"][0]["message"]["content"].strip()

        except urllib.error.HTTPError as http_err:
            try:
                err_body = http_err.read().decode("utf-8")
                err_json = json.loads(err_body)
                # Groq surfaces the real reason inside error.message
                reason = (
                    err_json.get("error", {}).get("message")
                    or err_json.get("error", {}).get("code")
                    or err_body[:300]
                )
            except Exception:
                reason = f"HTTP {http_err.code}"
            logger.error(
                f"HTTP {http_err.code} from {url}. Provider reason: {reason}"
            )
            raise LLMProviderError(f"HTTP {http_err.code}: {reason}")

        except Exception as exc:
            logger.error(f"OpenAI-compatible call to {url} failed: {exc}")
            raise LLMProviderError(str(exc))

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{settings.OLLAMA_URL}/api/chat"
        body = json.dumps({
            "model": settings.OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0.15},
            "stream": False,
        }).encode("utf-8")

        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload["message"]["content"].strip()

        except urllib.error.HTTPError as http_err:
            try:
                reason = http_err.read().decode("utf-8")[:300]
            except Exception:
                reason = f"HTTP {http_err.code}"
            logger.error(f"Ollama HTTP {http_err.code}: {reason}")
            raise LLMProviderError(f"Ollama HTTP {http_err.code}: {reason}")

        except Exception as exc:
            logger.error(f"Ollama call failed: {exc}")
            raise LLMProviderError(str(exc))

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        url = "https://api.anthropic.com/v1/messages"
        body = json.dumps({
            "model": settings.ANTHROPIC_MODEL,
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "x-api-key": settings.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload["content"][0]["text"].strip()

        except urllib.error.HTTPError as http_err:
            try:
                reason = http_err.read().decode("utf-8")[:300]
            except Exception:
                reason = f"HTTP {http_err.code}"
            logger.error(f"Anthropic HTTP {http_err.code}: {reason}")
            raise LLMProviderError(f"Anthropic HTTP {http_err.code}: {reason}")

        except Exception as exc:
            logger.error(f"Anthropic call failed: {exc}")
            raise LLMProviderError(str(exc))


llm_service = LLMService()
