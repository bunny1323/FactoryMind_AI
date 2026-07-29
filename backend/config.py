from __future__ import annotations

import os
import logging
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, model_validator
from typing import Literal

logger = logging.getLogger("factorymind")

class Settings(BaseSettings):
    """Application settings with validation."""
    
    APP_NAME: str = "FactoryMind AI Backend"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    JWT_SECRET: str = "factorymind-jwt-secret-key-32-chars-long!!!"
    JWT_ALGORITHM: str = "HS256"

    # Directory Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    
    # Vector DB settings (Defaults to memory if no cloud/local url is provided)
    VECTOR_BACKEND: Literal["memory", "qdrant"] = "memory"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    
    # Embedding settings
    EMBEDDING_BACKEND: Literal["hash", "fastembed"] = "fastembed"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    
    SPARSE_EMBEDDING_BACKEND: Literal["hash_lexical", "fastembed"] = "fastembed"
    SPARSE_EMBEDDING_MODEL: str = "Qdrant/bm25"
    
    RERANKER_BACKEND: Literal["fallback", "cross_encoder"] = "cross_encoder"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RAG_MIN_RELEVANCE_SCORE: float = 0.20  # Reduced from 0.35 to allow more results (CrossEncoder scale: filters documents with relevance < 20%)

    # LLM Settings
    # Providers: 'mock', 'groq', 'openai', 'openai_compatible', 'ollama', 'anthropic'
    # IMPORTANT: Set to actual provider (groq/openai/anthropic/ollama) - NOT mock for production!
    LLM_PROVIDER: Literal["mock", "groq", "openai", "openai_compatible", "ollama", "anthropic"] = "groq"
    LLM_FALLBACK_PROVIDER: Literal["mock", "groq", "openai", "openai_compatible", "ollama", "anthropic"] = "mock"
    LLM_MAX_RETRIES: int = 3
    LLM_RETRY_DELAY: float = 1.0  # seconds
    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"
    ANTHROPIC_API_KEY: str | None = None
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-latest"
    
    # Neo4j Settings
    NEO4J_URI: str | None = None
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    @field_validator("JWT_SECRET")
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        """Validate JWT secret is sufficiently long."""
        if len(v) < 32:
            logger.warning("JWT_SECRET is less than 32 characters. Consider using a stronger secret.")
        return v

    @field_validator("RAG_MIN_RELEVANCE_SCORE")
    @classmethod
    def validate_relevance_score(cls, v: float) -> float:
        """Validate relevance score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("RAG_MIN_RELEVANCE_SCORE must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def validate_llm_configuration(self) -> "Settings":
        """Validate LLM provider has required API keys."""
        provider = self.LLM_PROVIDER
        
        if provider == "groq" and not self.GROQ_API_KEY:
            logger.warning("LLM_PROVIDER is 'groq' but GROQ_API_KEY is not set. Queries will use extractive fallback.")
        
        if provider in ("openai", "openai_compatible") and not self.OPENAI_API_KEY:
            logger.warning(f"LLM_PROVIDER is '{provider}' but OPENAI_API_KEY is not set. Queries will use extractive fallback.")
        
        if provider == "anthropic" and not self.ANTHROPIC_API_KEY:
            logger.warning("LLM_PROVIDER is 'anthropic' but ANTHROPIC_API_KEY is not set. Queries will use extractive fallback.")
        
        if provider == "ollama":
            logger.info(f"LLM_PROVIDER is 'ollama' - ensure Ollama is running at {self.OLLAMA_URL}")
        
        return self

    @model_validator(mode="after")
    def validate_vector_backend(self) -> "Settings":
        """Validate vector backend configuration."""
        if self.VECTOR_BACKEND == "qdrant" and not self.QDRANT_URL:
            logger.warning("VECTOR_BACKEND is 'qdrant' but QDRANT_URL is not set. Falling back to memory backend.")
            self.VECTOR_BACKEND = "memory"
        
        return self

    @model_validator(mode="after")
    def validate_neo4j_configuration(self) -> "Settings":
        """Validate Neo4j configuration."""
        if self.NEO4J_URI and not (self.NEO4J_USER and self.NEO4J_PASSWORD):
            logger.warning("NEO4J_URI is set but NEO4J_USER or NEO4J_PASSWORD is missing. Graph features may not work.")
        
        return self

    model_config = SettingsConfigDict(
        # Use absolute path so .env is always found regardless of working directory
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


def validate_settings() -> Settings:
    """Validate and return settings with logging."""
    settings = Settings()
    
    # Log configuration summary
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    env_exists = os.path.exists(env_path)
    
    logger.info(
        f"CONFIG LOADED: .env path={env_path!r} (exists={env_exists}), "
        f"ENVIRONMENT={settings.ENVIRONMENT!r}, "
        f"VECTOR_BACKEND={settings.VECTOR_BACKEND!r}, "
        f"LLM_PROVIDER={settings.LLM_PROVIDER!r}"
    )
    
    return settings


settings = validate_settings()

