from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.error
from typing import Optional

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
        Route the request to the configured LLM provider with multi-provider fallback chain.
        Fallback chain: OpenAI → Groq → Anthropic → Ollama → Retrieval-only answer.
        Never returns "LLM unavailable" - always provides retrieved context with citations.
        """
        # Define fallback chain in priority order
        provider_chain = ["openai", "groq", "anthropic", "ollama"]
        
        # Start with configured primary provider, then try others in chain
        primary_provider = settings.LLM_PROVIDER.lower()
        max_retries = settings.LLM_MAX_RETRIES
        retry_delay = settings.LLM_RETRY_DELAY

        if primary_provider == "mock":
            return self._extractive_fallback(query, context, "mock", "Mock mode - no LLM synthesis")

        # Build ordered provider list (primary first, then remaining in chain order)
        ordered_providers = [primary_provider]
        for provider in provider_chain:
            if provider != primary_provider and provider not in ordered_providers:
                ordered_providers.append(provider)

        # Try each provider in order
        for provider in ordered_providers:
            # Check if provider has required API key/configuration
            if not self._is_provider_configured(provider):
                logger.info(f"Provider '{provider}' not configured, skipping...")
                continue

            # Try provider with retries
            for attempt in range(max_retries):
                try:
                    answer = self._call_provider(provider, query, context, system_prompt, attempt + 1)
                    logger.info(f"Successfully used provider: {provider}")
                    return answer
                except LLMProviderError as exc:
                    logger.warning(
                        f"Provider '{provider}' attempt {attempt + 1}/{max_retries} failed: {exc}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"Provider '{provider}' failed after {max_retries} attempts")
                        break  # Move to next provider

        # All providers failed - return extractive fallback with context
        logger.warning("All LLM providers failed, returning extractive fallback with retrieved context")
        return self._extractive_fallback(query, context, "all_providers", "All LLM providers unavailable")

    def _is_provider_configured(self, provider: str) -> bool:
        """Check if provider has required API key or configuration."""
        if provider == "openai" or provider == "openai_compatible":
            return bool(getattr(settings, "OPENAI_API_KEY", None))
        elif provider == "groq":
            return bool(getattr(settings, "GROQ_API_KEY", None))
        elif provider == "anthropic":
            return bool(getattr(settings, "ANTHROPIC_API_KEY", None))
        elif provider == "ollama":
            return True  # Ollama doesn't require API key
        return False

    def _call_provider(self, provider: str, query: str, context: str, system_prompt: str, attempt: int) -> str:
        """Call a specific provider with error handling."""
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
                f"\n========================\nCALLING LLM (Attempt {attempt})\n========================\n"
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
                f"\n========================\nLLM RESPONSE SUCCESS\n========================\n"
                f"Provider : {provider}\n"
                f"Latency  : {elapsed}s\n"
                f"Preview  : {answer[:500]}"
            )
            return answer

        except LLMProviderError:
            raise  # Re-raise LLMProviderError to be handled by retry logic
        except Exception as exc:
            logger.error(f"Unexpected error calling provider '{provider}': {exc}", exc_info=True)
            raise LLMProviderError(f"Unexpected error: {str(exc)}")

    def _extractive_fallback(self, query: str, context: str, provider: str, error: str) -> str:
        """Return extractive fallback with retrieved context and citations when LLM is unavailable."""
        if context and context.strip():
            # Parse context to extract source information
            lines = context.split('\n')
            sources = []
            content_lines = []
            
            for line in lines:
                if line.startswith('MANUAL:') or line.startswith('SOURCE:'):
                    sources.append(line)
                else:
                    content_lines.append(line)
            
            content = '\n'.join(content_lines).strip()
            source_info = '\n'.join(sources) if sources else "Source information not available"
            
            return (
                f"**Answer Based on Retrieved Context**\n\n"
                f"**Question:** {query}\n\n"
                f"**Retrieved Information:**\n\n{content[:2000]}\n\n"
                f"**Sources:**\n{source_info}\n\n"
                f"*(Note: LLM synthesis was unavailable due to provider error: {provider} - {error}. "
                f"This answer is based directly on retrieved manual context.)*"
            )
        return (
            f"**No Relevant Information Found**\n\n"
            f"**Question:** {query}\n\n"
            f"No relevant information was found in the indexed manuals for this query.\n\n"
            f"*(Note: LLM synthesis was unavailable due to provider error: {provider} - {error}. "
            f"No relevant context was retrieved from the manuals.)*"
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
                
                # Extract detailed error information
                error_obj = err_json.get("error", {})
                if isinstance(error_obj, dict):
                    reason = error_obj.get("message") or error_obj.get("code") or str(error_obj)
                else:
                    reason = str(error_obj) if error_obj else err_body[:300]
                
                # Add specific error type classification
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
                
                reason = f"{error_type}: {reason}"
                    
            except Exception as parse_err:
                reason = f"HTTP {http_err.code} (failed to parse error body: {parse_err})"
            
            logger.error(
                f"HTTP {http_err.code} from {url}. Provider reason: {reason}"
            )
            raise LLMProviderError(f"HTTP {http_err.code}: {reason}")

        except urllib.error.URLError as url_err:
            logger.error(f"Network error connecting to {url}: {url_err}")
            raise LLMProviderError(f"Network Error: {str(url_err)}")
        except Exception as exc:
            logger.error(f"OpenAI-compatible call to {url} failed: {exc}", exc_info=True)
            raise LLMProviderError(f"Unexpected Error: {str(exc)}")

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
