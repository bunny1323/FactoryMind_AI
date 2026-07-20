from __future__ import annotations

import os
import csv
import logging
from typing import Any
from rag.vector_store import VectorStore

logger = logging.getLogger("factorymind")

def run_logs_ingestion(vector_store: VectorStore, logs_path: str, collection_name: str = "maintenance_logs") -> int:
    """Reads maintenance logs CSV, embeds combined Issue + Action text, and upserts to Qdrant."""
    if not os.path.exists(logs_path):
        logger.warning(f"Maintenance logs path {logs_path} does not exist.")
        return 0

    records = []
    
    with open(logs_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            machine_id = row.get("MachineID", "Unknown")
            date = row.get("Date", "Unknown")
            issue = row.get("Issue", "").strip()
            action = row.get("Action", "").strip()
            downtime = row.get("DowntimeHours", "0")

            combined_text = f"Machine {machine_id} on {date}. Issue: {issue}. Action: {action}."
            record_id = f"log_{machine_id}_{idx}"
            
            records.append({
                "id": record_id,
                "title": f"Maintenance Log: {machine_id} ({date})",
                "text": combined_text,
                "source_type": "maintenance_log",
                "payload": {
                    "machine_id": machine_id,
                    "date": date,
                    "issue": issue,
                    "action": action,
                    "downtime_hours": float(downtime) if downtime else 0.0,
                    "collection": collection_name
                }
            })

    if records:
        counter = vector_store.upsert(collection_name, records)
        logger.info(f"Successfully ingested {counter} maintenance logs into Qdrant collection: {collection_name}")
        return counter
    return 0
