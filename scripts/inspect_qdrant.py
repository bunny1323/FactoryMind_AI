"""Script to inspect Qdrant collections and verify data integrity."""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL:
    print("ERROR: QDRANT_URL not found in environment")
    sys.exit(1)

print(f"Connecting to Qdrant at: {QDRANT_URL}")

from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

print("\n" + "="*80)
print("QDRANT COLLECTIONS INSPECTION")
print("="*80)

collections = ["manuals", "sop", "maintenance_logs", "error_codes", "spare_parts"]

for collection_name in collections:
    print(f"\n--- Collection: {collection_name} ---")
    
    try:
        # Check if collection exists
        exists = client.collection_exists(collection_name)
        if not exists:
            print(f"  ❌ Collection does not exist")
            continue
        
        # Get collection info
        info = client.get_collection(collection_name)
        print(f"  ✅ Collection exists")
        print(f"  Points count: {info.points_count}")
        print(f"  Vectors config: {info.config.params.vectors}")
        # Check for sparse vectors - handle different API versions
        if hasattr(info.config.params, 'sparse_vectors_config'):
            print(f"  Sparse vectors: {info.config.params.sparse_vectors_config}")
        elif hasattr(info.config.params, 'sparse_vectors'):
            print(f"  Sparse vectors: {info.config.params.sparse_vectors}")
        elif hasattr(info.config, 'params') and hasattr(info.config.params, 'vectors') and isinstance(info.config.params.vectors, dict):
            if 'sparse' in info.config.params.vectors:
                print(f"  Sparse vectors: {info.config.params.vectors['sparse']}")
            else:
                print(f"  Sparse vectors: NOT CONFIGURED (CRITICAL ISSUE - hybrid retrieval disabled)")
        else:
            print(f"  Sparse vectors: NOT CONFIGURED (CRITICAL ISSUE - hybrid retrieval disabled)")
            print(f"  Config params attributes: {[attr for attr in dir(info.config.params) if not attr.startswith('_')]}")
        
        # Get sample points to inspect payload structure
        if info.points_count > 0:
            print(f"\n  Sample payload inspection:")
            try:
                # Scroll to get first few points
                records, _ = client.scroll(
                    collection_name=collection_name,
                    limit=3,
                    with_payload=True,
                    with_vectors=False
                )
                
                for idx, record in enumerate(records):
                    print(f"\n  Point {idx + 1}:")
                    print(f"    ID: {record.id}")
                    payload = record.payload
                    print(f"    Payload keys: {list(payload.keys())}")
                    
                    # Check for expected fields
                    expected_fields = ["page", "document_name", "heading", "section", "chunk_type"]
                    for field in expected_fields:
                        if field in payload:
                            print(f"    ✓ {field}: {payload[field]}")
                        else:
                            print(f"    ✗ {field}: MISSING")
                    
                    # Check for image metadata
                    if "image_path" in payload:
                        print(f"    ✓ image_path: {payload['image_path']}")
                    if "caption" in payload:
                        print(f"    ✓ caption: {payload['caption']}")
                    
            except Exception as e:
                print(f"  Error scrolling points: {e}")
        else:
            print(f"  ⚠️  Collection is empty (0 points)")
            
    except Exception as e:
        print(f"  ❌ Error inspecting collection: {e}")

print("\n" + "="*80)
print("PAYLOAD INDEXES INSPECTION")
print("="*80)

for collection_name in collections:
    print(f"\n--- Collection: {collection_name} ---")
    try:
        if not client.collection_exists(collection_name):
            print(f"  ⚠️  Collection does not exist, skipping index check")
            continue
            
        info = client.get_collection(collection_name)
        if hasattr(info, 'payload_schema') and info.payload_schema:
            print(f"  Payload schema: {info.payload_schema}")
        else:
            print(f"  Payload schema: Not available via API")
            
    except Exception as e:
        print(f"  Error checking payload schema: {e}")

print("\n" + "="*80)
print("INSPECTION COMPLETE")
print("="*80)
