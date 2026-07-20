from __future__ import annotations

import os
import csv
import logging
from typing import Any
from rag.vector_store import VectorStore

logger = logging.getLogger("factorymind")

def run_parts_ingestion(vector_store: VectorStore, parts_path: str, collection_name: str = "spare_parts") -> int:
    """Reads spare parts CSV, embeds part details, and upserts to Qdrant."""
    if not os.path.exists(parts_path):
        logger.warning(f"Spare parts path {parts_path} does not exist.")
        return 0

    records = []

    with open(parts_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            part_id = row.get("PartID", "Unknown")
            part_name = row.get("PartName", "").strip()
            compatible = row.get("CompatibleMachine", "").strip()
            stock = row.get("Stock", "0")

            combined_text = f"Spare Part {part_id}: {part_name}. Compatible with {compatible}. Current Stock: {stock} units."
            record_id = f"part_{part_id}_{idx}"

            records.append({
                "id": record_id,
                "title": f"Spare Part: {part_name} ({part_id})",
                "text": combined_text,
                "source_type": "spare_part",
                "payload": {
                    "part_id": part_id,
                    "part_name": part_name,
                    "compatible_machine": compatible,
                    "stock": int(stock) if stock else 0,
                    "collection": collection_name
                }
            })

    if records:
        counter = vector_store.upsert(collection_name, records)
        logger.info(f"Successfully ingested {counter} spare parts into Qdrant collection: {collection_name}")
        return counter
    return 0
