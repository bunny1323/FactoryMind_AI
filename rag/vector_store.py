from __future__ import annotations

import logging
import uuid
import math
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from rag.embeddings import Embedder
from rag.sparse_embeddings import SparseEmbedder

logger = logging.getLogger("factorymind")
TOKEN_RE = re.compile(r"[a-z0-9]+")

@dataclass
class VectorRecord:
    id: str
    title: str
    text: str
    source_type: str
    payload: dict[str, Any]
    vector: list[float]

class VectorStore(ABC):
    @abstractmethod
    def upsert(self, collection: str, records: list[dict[str, Any]]) -> int:
        raise NotImplementedError

    @abstractmethod
    def search(self, collection: str, query: str, top_k: int, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def ping(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> dict[str, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_point(self, collection: str, point_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

class InMemoryHybridVectorStore(VectorStore):
    """Fallback local in-memory database supporting hybrid retrieval simulation."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        # Map of collection_name -> list of records
        self.collections: dict[str, list[VectorRecord]] = {}
        self.last_updated: dict[str, str] = {}

    def upsert(self, collection: str, records: list[dict[str, Any]]) -> int:
        bucket = self.collections.setdefault(collection, [])
        existing = {item.id: item for item in bucket}
        for record in records:
            rec_id = str(record["id"])
            existing[rec_id] = VectorRecord(
                id=rec_id,
                title=record["title"],
                text=record["text"],
                source_type=record["source_type"],
                payload=record.get("payload", {}),
                vector=self.embedder.encode(record["text"]),
            )
        self.collections[collection] = list(existing.values())
        import datetime
        self.last_updated[collection] = datetime.datetime.utcnow().isoformat() + "Z"
        return len(records)

    def search(self, collection: str, query: str, top_k: int, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        q_vector = self.embedder.encode(query)
        q_tokens = set(TOKEN_RE.findall(query.lower()))
        scored = []
        for item in self.collections.get(collection, []):
            if filters and not _matches(item.payload, filters):
                continue
            
            # Compute cosine similarity
            dense = max(0.0, _cosine(q_vector, item.vector))
            
            # Compute simple lexical overlap
            tokens = set(TOKEN_RE.findall((item.title + " " + item.text).lower()))
            lexical = len(q_tokens & tokens) / max(1, len(q_tokens))
            
            exact_bonus = 0.20 if query.lower() in (item.title + " " + item.text).lower() else 0.0
            
            # Blend score
            score = min(1.0, 0.55 * dense + 0.35 * lexical + exact_bonus)
            
            scored.append({
                "id": item.id,
                "title": item.title,
                "text": item.text,
                "source_type": item.source_type,
                "payload": item.payload,
                "score": round(score, 4),
            })
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]

    def ping(self) -> bool:
        return True

    def get_stats(self) -> dict[str, dict[str, Any]]:
        import datetime
        stats = {}
        for name in ["manuals", "sop", "maintenance_logs", "error_codes", "spare_parts"]:
            count = len(self.collections.get(name, []))
            stats[name] = {
                "count": count,
                "last_updated": self.last_updated.get(name, "Never")
            }
        return stats

    def get_point(self, collection: str, point_id: str) -> dict[str, Any] | None:
        records = self.collections.get(collection, [])
        for record in records:
            if record.id == point_id:
                return {
                    "id": record.id,
                    "score": 0.90,
                    "title": record.title,
                    "text": record.text,
                    "source_type": record.source_type,
                    "payload": record.payload,
                }
        return None

class QdrantHybridVectorStore(VectorStore):
    """Production Qdrant-backed dense + sparse retrieval with RRF."""

    def __init__(self, embedder: Embedder, sparse_embedder: SparseEmbedder, url: str, api_key: str | None, dimension: int):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        self.client = QdrantClient(url=url, api_key=api_key, timeout=15, check_compatibility=False)
        self.models = models
        self.embedder = embedder
        self.sparse_embedder = sparse_embedder
        self.dimension = dimension

    def ensure_collection(self, collection: str) -> None:
        models = self.models
        exists = self.client.collection_exists(collection)
        if exists:
            try:
                info = self.client.get_collection(collection)
                current_size = None
                # Check for dict or object structure in qdrant configuration response
                if info.config.params.vectors:
                    if isinstance(info.config.params.vectors, dict):
                        dense_cfg = info.config.params.vectors.get("dense")
                        if dense_cfg:
                            current_size = getattr(dense_cfg, "size", None)
                    else:
                        current_size = getattr(info.config.params.vectors, "size", None)
                if current_size and current_size != self.dimension:
                    logger.info(f"Dimension mismatch in collection '{collection}': current={current_size}, target={self.dimension}. Recreating collection...")
                    self.client.delete_collection(collection)
                    exists = False
            except Exception as get_err:
                logger.warning(f"Could not inspect collection config: {get_err}")

        if not exists:
            self.client.create_collection(
                collection_name=collection,
                vectors_config={"dense": models.VectorParams(size=self.dimension, distance=models.Distance.COSINE)},
                sparse_vectors_config={"sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
            )
            logger.info(f"Created collection '{collection}'.")

        # Idempotently ensure the user_id payload index exists.
        # Required for filtered queries on manuals and sop collections.
        # Qdrant allows calling create_payload_index even if the index already exists.
        try:
            self.client.create_payload_index(
                collection_name=collection,
                field_name="user_id",
                field_schema="keyword",
            )
            logger.info(f"Payload index on 'user_id' ensured for collection '{collection}'.")
        except Exception as idx_err:
            # Ignore "already exists" errors; log anything unexpected.
            err_str = str(idx_err).lower()
            if "already exists" in err_str or "conflict" in err_str:
                logger.debug(f"user_id index already exists for '{collection}' — OK.")
            else:
                logger.warning(f"Could not create user_id payload index for '{collection}': {idx_err}")


    def upsert(self, collection: str, records: list[dict[str, Any]]) -> int:
        self.ensure_collection(collection)
        points = []
        for record in records:
            canonical_id = str(record["id"])
            # Generate UUID from collection + ID to avoid conflicts
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"factorymind:{collection}:{canonical_id}"))
            sparse = self.sparse_embedder.encode(record["text"])
            payload = {
                **record.get("payload", {}),
                "canonical_id": canonical_id,
                "title": record["title"],
                "text": record["text"],
                "source_type": record["source_type"]
            }
            points.append(self.models.PointStruct(
                id=point_id,
                vector={
                    "dense": self.embedder.encode(record["text"]),
                    "sparse": self.models.SparseVector(indices=sparse.indices, values=sparse.values),
                },
                payload=payload,
            ))
            
        # Batch upsert: 100 points per call
        batch_size = 100
        import time
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            attempts = 3
            backoff = 1.0
            for attempt in range(attempts):
                try:
                    self.client.upsert(collection_name=collection, points=batch, wait=True)
                    break
                except Exception as e:
                    if attempt == attempts - 1:
                        logger.error(f"Failed to upsert batch in collection '{collection}' after {attempts} attempts: {e}")
                        raise e
                    logger.warning(f"Upsert batch to collection '{collection}' attempt {attempt + 1} failed: {e}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2.0
        return len(points)

    def search(self, collection: str, query: str, top_k: int, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        # Startup initializer guarantees collections & indexes exist. Skip per-query checks for maximum speed.
        q_filter = None
        if filters:
            must = [self.models.FieldCondition(key=key, match=self.models.MatchValue(value=value)) for key, value in filters.items() if value is not None]
            q_filter = self.models.Filter(must=must) if must else None
        
        sparse = self.sparse_embedder.encode(query)
        prefetch_limit = max(top_k * 4, 20)
        
        try:
            response = self.client.query_points(
                collection_name=collection,
                prefetch=[
                    self.models.Prefetch(query=self.embedder.encode(query), using="dense", limit=prefetch_limit, filter=q_filter),
                    self.models.Prefetch(query=self.models.SparseVector(indices=sparse.indices, values=sparse.values), using="sparse", limit=prefetch_limit, filter=q_filter),
                ],
                query=self.models.FusionQuery(fusion=self.models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
        except Exception as err:
            logger.warning(f"Qdrant query_points failed for collection '{collection}': {err}")
            return []
        
        results = []
        for point in response.points:
            payload = dict(point.payload or {})
            canonical_id = str(payload.pop("canonical_id", point.id))
            results.append({
                "id": canonical_id,
                "score": round(min(1.0, max(0.0, float(point.score))), 4),
                "title": payload.pop("title", canonical_id),
                "text": payload.pop("text", ""),
                "source_type": payload.pop("source_type", "unknown"),
                "payload": payload,
            })
        return results

    def ping(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.warning(f"Qdrant ping failed: {e}")
            return False

    def get_stats(self) -> dict[str, dict[str, Any]]:
        stats = {}
        for name in ["manuals", "sop", "maintenance_logs", "error_codes", "spare_parts"]:
            try:
                if self.client.collection_exists(name):
                    info = self.client.get_collection(name)
                    stats[name] = {
                        "count": info.points_count,
                        "last_updated": "Recently"
                    }
                else:
                    stats[name] = {"count": 0, "last_updated": "Never"}
            except Exception as e:
                logger.warning(f"Failed to fetch collection stats for '{name}': {e}", exc_info=True)
                stats[name] = {"count": 0, "last_updated": "Error"}
        return stats

    def get_point(self, collection: str, point_id: str) -> dict[str, Any] | None:
        try:
            import uuid
            qdrant_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"factorymind:{collection}:{point_id}"))
            records = self.client.retrieve(collection_name=collection, ids=[qdrant_uuid])
            if records:
                point = records[0]
                payload = dict(point.payload or {})
                canonical_id = str(payload.pop("canonical_id", point.id))
                return {
                    "id": canonical_id,
                    "score": 0.90,
                    "title": payload.pop("title", canonical_id),
                    "text": payload.pop("text", ""),
                    "source_type": payload.pop("source_type", "unknown"),
                    "payload": payload,
                }
        except Exception as e:
            logger.warning(f"Could not retrieve point {point_id} from {collection}: {e}")
        return None

def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / ((math.sqrt(sum(x * x for x in a)) or 1.0) * (math.sqrt(sum(y * y for y in b)) or 1.0))

def _matches(payload: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if expected is None:
            continue
        actual = payload.get(key)
        if isinstance(expected, str):
            if str(actual).lower() != expected.lower():
                return False
        elif actual != expected:
            return False
    return True
