from __future__ import annotations

import os
import json
import logging
from typing import Any
from rag.vector_store import VectorStore

logger = logging.getLogger("factorymind")

def run_errors_ingestion(vector_store: VectorStore, errors_path: str, collection_name: str = "error_codes") -> int:
    """Reads error codes JSON, embeds code + cause + solution, and upserts to Qdrant."""
    if not os.path.exists(errors_path):
        logger.warning(f"Error codes path {errors_path} does not exist.")
        return 0

    with open(errors_path, "r", encoding="utf-8") as f:
        try:
            error_list = json.load(f)
        except Exception as e:
            logger.error(f"Failed to parse error codes JSON: {e}")
            return 0

    records = []
    for idx, item in enumerate(error_list):
        code = item.get("code", "Unknown")
        cause = item.get("cause", "").strip()
        solution = item.get("solution", "").strip()

        combined_text = f"Error Code {code}. Cause: {cause}. Solution: {solution}."
        record_id = f"error_{code}_{idx}"

        records.append({
            "id": record_id,
            "title": f"Error Code {code}",
            "text": combined_text,
            "source_type": "error_code",
            "payload": {
                "code": code,
                "cause": cause,
                "solution": solution,
                "collection": collection_name
            }
        })

    if records:
        counter = vector_store.upsert(collection_name, records)
        logger.info(f"Successfully ingested {counter} error codes into Qdrant collection: {collection_name}")
        return counter
    return 0
