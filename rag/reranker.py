import math
import logging
from typing import Any

logger = logging.getLogger("factorymind")

class Reranker:
    def rerank(self, query: str, hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        raise NotImplementedError

class FallbackReranker(Reranker):
    def rerank(self, query: str, hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # Already sorted by score, just returns top k
        return hits[:top_k]

class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Initializing SentenceTransformers CrossEncoder: {model_name}")
            self.model = CrossEncoder(model_name)
        except Exception as e:
            logger.error(f"Failed to initialize CrossEncoder: {e}. Falling back to FallbackReranker.")
            raise e

    def rerank(self, query: str, hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not hits:
            return []
        
        pairs = [[query, hit["text"]] for hit in hits]
        try:
            scores = self.model.predict(pairs)
            for idx, score in enumerate(scores):
                # Always normalize raw logit scores to 0-1 range using sigmoid.
                normalized_score = float(1.0 / (1.0 + math.exp(-float(score))))
                hits[idx]["score"] = round(normalized_score, 4)
                hits[idx]["rerank_score"] = round(normalized_score, 4)
            
            # Sort by rerank score
            sorted_hits = sorted(hits, key=lambda x: x.get("score", 0.0), reverse=True)
            return sorted_hits[:top_k]
        except Exception as e:
            logger.error(f"Error during CrossEncoder prediction: {e}")
            return hits[:top_k]

def build_reranker(backend: str = "cross_encoder", model_name: str = "BAAI/bge-reranker-large") -> Reranker:
    if backend == "cross_encoder":
        try:
            return CrossEncoderReranker(model_name)
        except Exception as e:
            logger.warning(f"CrossEncoder not available, using FallbackReranker: {e}", exc_info=True)
    return FallbackReranker()
