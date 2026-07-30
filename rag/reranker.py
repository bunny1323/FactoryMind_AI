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
                # Blend cross-encoder score with existing score (which includes visual boost)
                existing_score = hits[idx].get("score", 0.0)
                blended = max(normalized_score, existing_score) if (
                    hits[idx].get("payload", {}).get("chunk_type") == "image" or 
                    bool(hits[idx].get("payload", {}).get("image_path"))
                ) else normalized_score
                hits[idx]["score"] = round(blended, 4)
                hits[idx]["rerank_score"] = round(normalized_score, 4)
            
            # Sort by score
            sorted_hits = sorted(hits, key=lambda x: x.get("score", 0.0), reverse=True)
            
            # Ensure visual chunks survive top_k cutoff if any were retrieved
            visual_hits = [
                h for h in hits 
                if h.get("payload", {}).get("chunk_type") == "image" or bool(h.get("payload", {}).get("image_path"))
            ]
            if visual_hits and not any(
                h.get("payload", {}).get("chunk_type") == "image" or bool(h.get("payload", {}).get("image_path"))
                for h in sorted_hits[:top_k]
            ):
                # Guarantee at least 2 top visual chunks survive in top_k
                non_visual = [
                    h for h in sorted_hits 
                    if not (h.get("payload", {}).get("chunk_type") == "image" or bool(h.get("payload", {}).get("image_path")))
                ]
                return (visual_hits[:2] + non_visual)[:top_k]

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
