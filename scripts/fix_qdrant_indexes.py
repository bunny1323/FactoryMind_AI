"""
scripts/fix_qdrant_indexes.py
─────────────────────────────────────────────────────────────────────────────
One-off migration: create the missing `user_id` keyword payload index on all
five Qdrant collections.

Run from the project root:
    .venv\\Scripts\\python scripts/fix_qdrant_indexes.py

This is fully idempotent — safe to run multiple times.
"""

import os
import sys
import logging

# Make sure the project root is on the path so we can import settings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("fix_qdrant_indexes")

try:
    from backend.config import settings
except ImportError as e:
    logger.error(f"Could not import settings: {e}")
    sys.exit(1)

try:
    from qdrant_client import QdrantClient
except ImportError:
    logger.error("qdrant-client not installed. Run: pip install qdrant-client")
    sys.exit(1)


COLLECTIONS = ["manuals", "sop", "error_codes", "spare_parts", "maintenance_logs"]


def main():
    url = settings.QDRANT_URL
    api_key = getattr(settings, "QDRANT_API_KEY", None)

    logger.info(f"Connecting to Qdrant at: {url}")
    client = QdrantClient(url=url, api_key=api_key, timeout=15, check_compatibility=False)

    # Verify connection
    try:
        existing = client.get_collections()
        existing_names = {c.name for c in existing.collections}
        logger.info(f"Connected. Existing collections: {existing_names}")
    except Exception as e:
        logger.error(f"Could not connect to Qdrant: {e}")
        sys.exit(1)

    success_count = 0
    for coll in COLLECTIONS:
        if coll not in existing_names:
            logger.warning(f"Collection '{coll}' does not exist — skipping.")
            continue

        try:
            client.create_payload_index(
                collection_name=coll,
                field_name="user_id",
                field_schema="keyword",
            )
            logger.info(f"  ✓ Created user_id index on '{coll}'")
            success_count += 1
        except Exception as err:
            err_str = str(err).lower()
            if "already exists" in err_str or "conflict" in err_str:
                logger.info(f"  ✓ user_id index already exists on '{coll}' — OK")
                success_count += 1
            else:
                logger.error(f"  ✗ Failed to create index on '{coll}': {err}")

    logger.info(f"\nDone. {success_count}/{len(COLLECTIONS)} collections indexed.")


if __name__ == "__main__":
    main()
