"""
Test script to verify .env configuration is loaded correctly.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import settings

print("="*70)
print("Configuration Test")
print("="*70)
print(f"LLM_PROVIDER: {settings.LLM_PROVIDER}")
print(f"OLLAMA_URL: {settings.OLLAMA_URL}")
print(f"OLLAMA_MODEL: {settings.OLLAMA_MODEL}")
print(f"LLM_MAX_RETRIES: {settings.LLM_MAX_RETRIES}")
print(f"LLM_RETRY_DELAY: {settings.LLM_RETRY_DELAY}")
print("="*70)

# Test if Ollama provider is configured
from backend.services.llm_service import llm_service
is_configured = llm_service._is_provider_configured("ollama")
print(f"Ollama provider configured: {is_configured}")
print("="*70)
