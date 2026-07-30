"""
Script to inspect actual points and payloads in Qdrant collections.
Prints key fields to check if image metadata and chunks exist.
"""
import sys, os
sys.path.insert(0, ".")

from backend.config import settings
from qdrant_client import QdrantClient

client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

for coll in ["manuals", "sop", "maintenance_logs", "error_codes", "spare_parts"]:
    print(f"\n{'='*60}\nCOLLECTION: {coll}\n{'='*60}")
    try:
        info = client.get_collection(coll)
        print(f"Total Points: {info.points_count}")
        
        if info.points_count > 0:
            res = client.scroll(collection_name=coll, limit=5, with_payload=True)
            points = res[0]
            print(f"Sampled {len(points)} points:")
            for i, p in enumerate(points):
                payload = p.payload or {}
                print(f" Point {i+1}: canonical_id={payload.get('canonical_id', p.id)}")
                print(f"   chunk_type: {payload.get('chunk_type')}")
                print(f"   document_name: {payload.get('document_name')}")
                print(f"   page: {payload.get('page')}")
                print(f"   image_path: {payload.get('image_path')}")
                print(f"   caption: {payload.get('caption')}")
                print(f"   payload keys: {list(payload.keys())}")
    except Exception as e:
        print(f"Error reading collection {coll}: {e}")
