from __future__ import annotations

import logging
import hashlib
import math
import re
from abc import ABC, abstractmethod

logger = logging.getLogger("factorymind")

TOKEN_RE = re.compile(r"[a-z0-9]+")

class Embedder(ABC):
    dimension: int

    @abstractmethod
    def encode(self, text: str) -> list[float]:
        raise NotImplementedError

class HashEmbedder(Embedder):
    """Offline deterministic embedder for local testing and testing without external downloads."""

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in TOKEN_RE.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

class FastEmbedDenseEmbedder(Embedder):
    """Primary dense embedder using FastEmbed with BAAI/bge-small-en-v1.5."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            from fastembed import TextEmbedding
            logger.info(f"Initializing FastEmbed TextEmbedding with model: {model_name}")
            self.model = TextEmbedding(model_name=model_name)
            # Fetch dimension from first embedded element or default to 384 for BAAI/bge-small-en-v1.5
            self.dimension = 384
        except Exception as e:
            logger.error(f"Failed to initialize FastEmbed dense embedder: {e}. Falling back to HashEmbedder.")
            raise e

    def encode(self, text: str) -> list[float]:
        # FastEmbed returns a generator of embeddings
        embeddings = list(self.model.embed([text]))
        if not embeddings:
            return [0.0] * self.dimension
        return embeddings[0].tolist()

def build_embedder(backend: str = "fastembed", model_name: str = "BAAI/bge-small-en-v1.5", dimension: int = 384) -> Embedder:
    if backend == "fastembed":
        try:
            return FastEmbedDenseEmbedder(model_name)
        except Exception as e:
            logger.warning(f"FastEmbed not available, using local HashEmbedder: {e}", exc_info=True)
    return HashEmbedder(dimension)
