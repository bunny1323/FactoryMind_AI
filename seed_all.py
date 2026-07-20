import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import logging

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.dependencies import container
from backend.config import settings
from ingestion.ingest_manuals import run_manuals_ingestion
from ingestion.ingest_logs import run_logs_ingestion
from ingestion.ingest_errors import run_errors_ingestion
from ingestion.ingest_parts import run_parts_ingestion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("factorymind_seed")

def main():
    logger.info("Starting local offline data seeding...")
    vector_store = container.vector_store
    data_dir = settings.DATA_DIR

    # 1. Manuals
    logger.info("Ingesting Manuals...")
    manual_count = run_manuals_ingestion(vector_store, os.path.join(data_dir, "manuals"), "manuals")
    logger.info(f"Manuals ingested: {manual_count}")

    # 2. SOPs
    logger.info("Ingesting SOPs...")
    sop_count = run_manuals_ingestion(vector_store, os.path.join(data_dir, "sop"), "sop")
    logger.info(f"SOPs ingested: {sop_count}")

    # 3. Logs
    logger.info("Ingesting Maintenance Logs...")
    logs_path = os.path.join(data_dir, "maintenance_logs", "maintenance_logs.csv")
    logs_count = run_logs_ingestion(vector_store, logs_path)
    logger.info(f"Logs ingested: {logs_count}")

    # 4. Error Codes
    logger.info("Ingesting Error Codes...")
    errors_path = os.path.join(data_dir, "error_codes", "error_codes.json")
    errors_count = run_errors_ingestion(vector_store, errors_path)
    logger.info(f"Error Codes ingested: {errors_count}")

    # 5. Spare Parts
    logger.info("Ingesting Spare Parts...")
    parts_path = os.path.join(data_dir, "spare_parts", "spare_parts.csv")
    parts_count = run_parts_ingestion(vector_store, parts_path)
    logger.info(f"Spare Parts ingested: {parts_count}")

    logger.info("Data seeding complete!")
    
    # Test a search to verify
    logger.info("Testing search on 'vibration'...")
    results = vector_store.search("manuals", "vibration", top_k=2)
    logger.info("Search Results:")
    for r in results:
        logger.info(f"- [{r['score']}] {r['title']}: {r['text'][:120]}...")

if __name__ == "__main__":
    main()
