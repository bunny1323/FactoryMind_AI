"""
Forwarding module to avoid duplicate code logic.
All manual ingestion logic is consolidated inside ingestion/ingest_manuals.py.
"""
from ingestion.ingest_manuals import ingest_manuals, run_manuals_ingestion
