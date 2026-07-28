"""Custom exceptions and error handling for FactoryMind AI."""
from __future__ import annotations

from typing import Any, Optional
from fastapi import HTTPException, status


class FactoryMindError(Exception):
    """Base exception for FactoryMind AI application."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(FactoryMindError):
    """Raised when configuration is invalid or missing."""
    pass


class VectorStoreError(FactoryMindError):
    """Raised when vector store operations fail."""
    pass


class IngestionError(FactoryMindError):
    """Raised when document ingestion fails."""
    pass


class RetrievalError(FactoryMindError):
    """Raised when document retrieval fails."""
    pass


class LLMError(FactoryMindError):
    """Raised when LLM operations fail."""
    pass


class PredictionError(FactoryMindError):
    """Raised when prediction operations fail."""
    pass


class AuthenticationError(FactoryMindError):
    """Raised when authentication fails."""
    pass


class GraphError(FactoryMindError):
    """Raised when knowledge graph operations fail."""
    pass


class OCRError(FactoryMindError):
    """Raised when OCR operations fail."""
    pass


class EmbeddingError(FactoryMindError):
    """Raised when embedding generation fails."""
    pass


class VisionError(FactoryMindError):
    """Raised when vision/image analysis operations fail."""
    pass


def handle_exception(exc: Exception) -> HTTPException:
    """Convert application exceptions to HTTP exceptions."""
    
    if isinstance(exc, HTTPException):
        return exc
    
    if isinstance(exc, ConfigurationError):
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "configuration_error",
                "message": exc.message,
                "details": exc.details
            }
        )
    
    if isinstance(exc, VectorStoreError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "vector_store_error",
                "message": exc.message,
                "details": exc.details
            }
        )
    
    if isinstance(exc, IngestionError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "ingestion_error",
                "message": exc.message,
                "details": exc.details
            }
        )
    
    if isinstance(exc, RetrievalError):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "retrieval_error",
                "message": exc.message,
                "details": exc.details
            }
        )
    
    if isinstance(exc, LLMError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "llm_error",
                "message": exc.message,
                "details": exc.details
            }
        )
    
    if isinstance(exc, PredictionError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "prediction_error",
                "message": exc.message,
                "details": exc.details
            }
        )
    
    if isinstance(exc, AuthenticationError):
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "authentication_error",
                "message": exc.message,
                "details": exc.details
            }
        )
    
    if isinstance(exc, GraphError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "graph_error",
                "message": exc.message,
                "details": exc.details
            }
        )
    
    if isinstance(exc, OCRError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "ocr_error",
                "message": exc.message,
                "details": exc.details
            }
        )
    
    if isinstance(exc, EmbeddingError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "embedding_error",
                "message": exc.message,
                "details": exc.details
            }
        )
    
    if isinstance(exc, VisionError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "vision_error",
                "message": exc.message,
                "details": exc.details
            }
        )
    
    # Default fallback
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={
            "error": "internal_server_error",
            "message": str(exc) if exc else "An unexpected error occurred"
        }
    )
