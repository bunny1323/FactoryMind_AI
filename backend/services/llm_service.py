from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from backend.config import settings

logger = logging.getLogger("factorymind")

MOCK_DEMO_ANSWER = """### Diagnostic Assessment: Main Pump Rotational Vibration
Based on the service manual and telemetry readings, **Machine M101** is experiencing severe main pump rotational vibration. 

#### Recommended Actions:
1. **Shaft Alignment & Coupling Inspection**:
   - Immediately measure the shaft alignment between the Cummins 6BTAA5.9 engine flywheel and the Kawasaki K3V112DT main pump using a shaft alignment laser. Runout must be less than **0.05 mm**.
   - Inspect the **Engine-Pump Coupling Insert (Part ID: SP-CPL-332)**. Worn coupling inserts cause severe torsional vibration and must be replaced immediately if cracked.
2. **Slewing Turntable Bearing Check**:
   - Verify the radial and axial play of the turntable swing circle. Radial play must not exceed **1.5 mm**.
   - If bearings show fatigue or metal shavings are found in the grease, schedule the **Slewing & Main Pump Bearing Kit (Part ID: SP-BRG-215)** replacement procedure.
3. **Emergency Stop Warning**:
   - Telemetry indicates a vibration level near the critical warning threshold. Casing temperatures exceeding **355 K (82°C)** indicate high friction. Shut down the system if vibration climbs above **0.30 mm** to prevent cataclysmic pump lock-up.

Please refer to standard operating procedure **SOP-MNT-R215-087** for the complete disassembly and shaft-re-alignment steps.
"""

class LLMService:
    def synthesize(self, query: str, context: str, system_prompt: str) -> str:
        provider = settings.LLM_PROVIDER.lower()
        
        # Check if the query matches the demo scenario
        is_demo_query = "vibration" in query.lower() or "m101" in query.lower()
        if provider == "mock":
            if is_demo_query:
                return MOCK_DEMO_ANSWER
            return f"Mock LLM Response: Based on the context provided, here is the answer to your query: '{query}'.\n\nContext used:\n{context[:200]}..."

        if provider == "groq" and settings.GROQ_API_KEY:
            return self._call_openai_compatible(
                url="https://api.groq.com/openai/v1/chat/completions",
                api_key=settings.GROQ_API_KEY,
                model=settings.GROQ_MODEL,
                system_prompt=system_prompt,
                user_prompt=f"Question: {query}\n\nContext:\n{context}"
            )
        
        if provider == "openai" and settings.OPENAI_API_KEY:
            return self._call_openai_compatible(
                url="https://api.openai.com/v1/chat/completions",
                api_key=settings.OPENAI_API_KEY,
                model=settings.OPENAI_MODEL,
                system_prompt=system_prompt,
                user_prompt=f"Question: {query}\n\nContext:\n{context}"
            )

        if provider == "ollama":
            return self._call_ollama(
                system_prompt=system_prompt,
                user_prompt=f"Question: {query}\n\nContext:\n{context}"
            )

        if provider == "anthropic" and settings.ANTHROPIC_API_KEY:
            return self._call_anthropic(
                system_prompt=system_prompt,
                user_prompt=f"Question: {query}\n\nContext:\n{context}"
            )

        # Fallback to Mock if provider not configured properly
        logger.warning(f"LLM Provider '{provider}' not configured properly. Falling back to Mock.")
        if is_demo_query:
            return MOCK_DEMO_ANSWER
        return f"Grounded Extractive Summary (Fallback): Here is the top retrieved context: {context[:500]}..."

    def _call_openai_compatible(self, url: str, api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
        body = json.dumps({
            "model": model,
            "temperature": 0.15,
            "max_tokens": 800,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }).encode("utf-8")
        
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"OpenAI compatible API call failed: {e}")
            raise e

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{settings.OLLAMA_URL}/api/chat"
        body = json.dumps({
            "model": settings.OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "options": {"temperature": 0.15},
            "stream": False
        }).encode("utf-8")
        
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            raise e

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        url = "https://api.anthropic.com/v1/messages"
        body = json.dumps({
            "model": settings.ANTHROPIC_MODEL,
            "max_tokens": 800,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ]
        }).encode("utf-8")
        
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "x-api-key": settings.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload["content"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise e

llm_service = LLMService()
