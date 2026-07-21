from __future__ import annotations

import logging
import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger("factorymind")

TOKEN_RE = re.compile(r"[a-z0-9]+")

@dataclass
class SparseEmbedding:
    indices: list[int]
    values: list[float]

class SparseEmbedder(ABC):
    @abstractmethod
    def encode(self, text: str) -> SparseEmbedding:
        raise NotImplementedError

class HashLexicalSparseEmbedder(SparseEmbedder):
    """Deterministic lexical sparse encoder for local development and fallback."""

    def __init__(self, buckets: int = 1_000_003):
        self.buckets = buckets

    def encode(self, text: str) -> SparseEmbedding:
        counts = Counter(TOKEN_RE.findall(text.lower()))
        pairs: dict[int, float] = {}
        for token, count in counts.items():
            idx = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big") % self.buckets
            pairs[idx] = pairs.get(idx, 0.0) + (1.0 + math.log(count))
        ordered = sorted(pairs.items())
        return SparseEmbedding(indices=[idx for idx, _ in ordered], values=[value for _, value in ordered])

class FastEmbedBm25SparseEmbedder(SparseEmbedder):
    """Primary sparse embedder using FastEmbed's Qdrant/bm25."""

    def __init__(self, model_name: str = "Qdrant/bm25"):
        try:
            from fastembed import SparseTextEmbedding
            logger.info(f"Initializing FastEmbed SparseTextEmbedding with model: {model_name}")
            self.model = SparseTextEmbedding(model_name=model_name)
        except Exception as e:
            logger.error(f"Failed to initialize FastEmbed sparse embedder: {e}. Falling back to HashLexicalSparseEmbedder.")
            raise e

    def encode(self, text: str) -> SparseEmbedding:
        vector = next(iter(self.model.embed([text])))
        return SparseEmbedding(indices=vector.indices.tolist(), values=vector.values.tolist())

def build_sparse_embedder(backend: str = "fastembed", model_name: str = "Qdrant/bm25") -> SparseEmbedder:
    if backend == "fastembed":
        try:
            return FastEmbedBm25SparseEmbedder(model_name)
        except Exception as e:
            logger.warning(f"FastEmbed sparse model not available, using local HashLexicalSparseEmbedder: {e}", exc_info=True)
    return HashLexicalSparseEmbedder()
