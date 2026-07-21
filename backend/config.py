from __future__ import annotations

import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "FactoryMind AI Backend"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    JWT_SECRET: str = "factorymind-jwt-secret-key-32-chars-long!!!"
    JWT_ALGORITHM: str = "HS256"

    # Directory Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    
    # Vector DB settings (Defaults to memory if no cloud/local url is provided)
    VECTOR_BACKEND: str = "memory"  # 'memory' or 'qdrant'
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    
    # Embedding settings
    EMBEDDING_BACKEND: str = "fastembed"  # 'hash' or 'fastembed'
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    
    SPARSE_EMBEDDING_BACKEND: str = "fastembed"  # 'hash_lexical' or 'fastembed'
    SPARSE_EMBEDDING_MODEL: str = "Qdrant/bm25"
    
    RERANKER_BACKEND: str = "cross_encoder"  # 'fallback' or 'cross_encoder'
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RAG_MIN_RELEVANCE_SCORE: float = 0.35

    # LLM Settings
    # Providers: 'mock', 'groq', 'openai', 'openai_compatible', 'ollama', 'anthropic'
    LLM_PROVIDER: str = "mock"
    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    # Set OPENAI_BASE_URL to use any OpenAI-compatible provider:
    #   Google AI Studio: https://generativelanguage.googleapis.com/v1beta/openai/
    #   OpenRouter:       https://openrouter.ai/api/v1
    #   Together AI:      https://api.together.xyz/v1
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"
    ANTHROPIC_API_KEY: str | None = None
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-latest"
    
    # Neo4j Settings
    NEO4J_URI: str | None = None
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    model_config = SettingsConfigDict(
        # Use absolute path so .env is always found regardless of working directory
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Debug: confirm what was loaded from .env
import logging as _logging
_cfg_logger = _logging.getLogger("factorymind")
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
_env_exists = os.path.exists(_env_path)
_key_preview = (settings.OPENAI_API_KEY[:12] + "...") if settings.OPENAI_API_KEY else "NOT SET"
_cfg_logger.info(
    f"CONFIG LOADED: .env path={_env_path!r} (exists={_env_exists}), "
    f"LLM_PROVIDER={settings.LLM_PROVIDER!r}, "
    f"OPENAI_API_KEY={_key_preview}"
)

