import os
import sys
import logging

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.dependencies import container
from backend.services.rag_service import rag_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("factorymind_test_rag")

def main():
    logger.info("Initializing RAG validation test...")
    
    # Idempotency check: run ingestion twice and check stats count doesn't double
    from ingestion.ingest_manuals import run_manuals_ingestion
    logger.info("Running idempotency verification...")
    run_manuals_ingestion(container.vector_store, "data/manuals", "manuals")
    count_1 = container.vector_store.get_stats()["manuals"]["count"]
    run_manuals_ingestion(container.vector_store, "data/manuals", "manuals")
    count_2 = container.vector_store.get_stats()["manuals"]["count"]
    logger.info(f"Idempotency verification: run 1 count = {count_1}, run 2 count = {count_2}")
    assert count_1 == count_2, f"Idempotency check failed: count doubled from {count_1} to {count_2}"
    logger.info("Idempotency verification SUCCESS!")
    
    # 1. Run Seeding first if the DB is empty
    stats = container.vector_store.get_stats()
    if stats["manuals"]["count"] == 0:
        logger.info("Vector store is empty. Seeding data first...")
        import seed_all
        seed_all.main()

    # 2. Query test
    query = "Machine M101 is showing increased vibration. What should I do?"
    logger.info(f"Querying: '{query}'")
    
    answer, citations = rag_service.get_grounded_answer(query)
    
    logger.info("\n=== ANSWER ===")
    logger.info(answer)
    logger.info("==============\n")
    
    logger.info("=== EVIDENCE CITATIONS ===")
    for idx, c in enumerate(citations):
        logger.info(f"{idx+1}. [{c.get('score')}] {c.get('title')} ({c.get('source_type')})")
        logger.info(f"   Text: {c.get('text')[:150]}...")
    logger.info("==========================\n")
    
    if len(citations) > 0:
        logger.info("Verification SUCCESS: The query returned citations from the seed manuals!")
    else:
        logger.info("Verification FAILED: No citations returned.")

if __name__ == "__main__":
    main()
