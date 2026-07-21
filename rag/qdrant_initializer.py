from __future__ import annotations

import logging
from typing import Any, List

logger = logging.getLogger("factorymind")

COLLECTIONS = ["manuals", "sop", "maintenance_logs", "error_codes", "spare_parts"]
INDEX_FIELDS = ["user_id", "machine_model", "error_code", "part_number"]


class QdrantInitializer:
    """
    Handles startup-only initialization of Qdrant collections and payload indexes.
    Prevents runtime index creation overhead during retrieval queries.
    """

    def __init__(self, client: Any, dimension: int):
        self.client = client
        self.dimension = dimension
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            logger.info("Qdrant collections and payload indexes already initialized.")
            return

        try:
            from qdrant_client.http import models

            for collection in COLLECTIONS:
                self._ensure_collection_and_indexes(collection, models)

            self._initialized = True
            logger.info("✅ Qdrant initialization complete. All collections and payload indexes are active.")
        except Exception as e:
            logger.error(f"❌ Qdrant initialization failed during startup: {e}", exc_info=True)

    def _ensure_collection_and_indexes(self, collection: str, models: Any) -> None:
        exists = False
        try:
            exists = self.client.collection_exists(collection)
        except Exception as e:
            logger.warning(f"Could not check existence of collection '{collection}': {e}")
            return

        if exists:
            try:
                info = self.client.get_collection(collection)
                current_size = None
                if info.config.params.vectors:
                    if isinstance(info.config.params.vectors, dict):
                        dense_cfg = info.config.params.vectors.get("dense")
                        if dense_cfg:
                            current_size = getattr(dense_cfg, "size", None)
                    else:
                        current_size = getattr(info.config.params.vectors, "size", None)
                if current_size and current_size != self.dimension:
                    logger.info(
                        f"Dimension mismatch in collection '{collection}': current={current_size}, target={self.dimension}. Recreating collection..."
                    )
                    self.client.delete_collection(collection)
                    exists = False
            except Exception as get_err:
                logger.warning(f"Could not inspect collection config for '{collection}': {get_err}")

        if not exists:
            try:
                self.client.create_collection(
                    collection_name=collection,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=self.dimension, distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(
                            index=models.SparseIndexParams(on_disk=False)
                        )
                    },
                )
                logger.info(f"Created Qdrant collection '{collection}' (dense dim={self.dimension}).")
            except Exception as create_err:
                logger.error(f"Failed to create collection '{collection}': {create_err}")
                return

        # Ensure payload indexes idempotently
        for field_name in INDEX_FIELDS:
            try:
                self.client.create_payload_index(
                    collection_name=collection,
                    field_name=field_name,
                    field_schema="keyword",
                )
                logger.debug(f"Payload index '{field_name}' ensured for '{collection}'.")
            except Exception as idx_err:
                err_str = str(idx_err).lower()
                if "already exists" in err_str or "conflict" in err_str:
                    pass
                else:
                    logger.debug(f"Payload index '{field_name}' info for '{collection}': {idx_err}")
