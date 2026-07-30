"""Script to fix sparse vectors configuration in Qdrant collections."""
import os
import sys
from dotenv import load_dotenv

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
print("FIXING SPARSE VECTORS CONFIGURATION")
print("="*80)

collections = ["manuals", "sop", "maintenance_logs", "error_codes", "spare_parts"]
dimension = 384  # BGE-small dimension

for collection_name in collections:
    print(f"\n--- Collection: {collection_name} ---")
    
    try:
        # Check if collection exists
        exists = client.collection_exists(collection_name)
        if not exists:
            print(f"  ⚠️  Collection does not exist, skipping")
            continue
        
        # Get current info
        info = client.get_collection(collection_name)
        points_count = info.points_count
        
        print(f"  Current points: {points_count}")

        # Check if sparse vectors are configured
        has_sparse = hasattr(info.config.params, 'sparse_vectors_config') and info.config.params.sparse_vectors_config

        if has_sparse:
            print(f"  ✓ Sparse vectors already configured, skipping")
            continue

        print(f"  ❌ Sparse vectors NOT configured - need to recreate collection")

        # For large collections (>1000 points), skip backup and require re-ingestion
        all_points = None  # Initialize for both paths
        if points_count > 1000:
            print(f"  ⚠️  Large collection ({points_count} points) - will delete without backup")
            print(f"  ⚠️  You will need to re-run ingestion after this fix")
            user_input = input(f"  Continue with deletion of {collection_name}? (yes/no): ")
            if user_input.lower() != 'yes':
                print(f"  Skipping {collection_name}")
                continue
        else:
            # Backup: Get all points before deletion for small collections
            if points_count > 0:
                print(f"  Backing up {points_count} points...")
                all_points = []
                offset = None
                batch_size = 100

                while True:
                    records, offset = client.scroll(
                        collection_name=collection_name,
                        limit=batch_size,
                        with_payload=True,
                        with_vectors=True,
                        offset=offset
                    )
                    all_points.extend(records)
                    if offset is None:
                        break

                print(f"  Backed up {len(all_points)} points")
        
        # Delete collection
        print(f"  Deleting collection...")
        client.delete_collection(collection_name)
        
        # Recreate with sparse vectors
        print(f"  Recreating collection with sparse vectors...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dimension, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )
        print(f"  ✓ Collection recreated with sparse vectors")
        
        # Restore points if any
        if points_count > 0 and all_points:
            print(f"  Restoring {len(all_points)} points...")
            
            # Need to re-encode sparse vectors for each point
            from rag.sparse_embeddings import build_sparse_embedder
            sparse_embedder = build_sparse_embedder("fastembed", "Qdrant/bm25")
            
            points_to_upsert = []
            for record in all_points:
                # Get text for sparse encoding
                text = record.payload.get("text", "")
                if not text:
                    text = record.payload.get("title", "")
                
                # Encode sparse vector
                sparse = sparse_embedder.encode(text)
                
                point = models.PointStruct(
                    id=record.id,
                    vector={
                        "dense": record.vector if hasattr(record, 'vector') else None,
                        "sparse": models.SparseVector(indices=sparse.indices, values=sparse.values),
                    },
                    payload=record.payload
                )
                points_to_upsert.append(point)
            
            # Batch upsert
            batch_size = 100
            for i in range(0, len(points_to_upsert), batch_size):
                batch = points_to_upsert[i:i + batch_size]
                client.upsert(collection_name=collection_name, points=batch)
            
            print(f"  ✓ Restored {len(points_to_upsert)} points with sparse vectors")
        
        # Recreate payload indexes
        print(f"  Recreating payload indexes...")
        index_fields = ["user_id", "machine_model", "error_code", "part_number"]
        for field_name in index_fields:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema="keyword",
                )
                print(f"    ✓ Index created for {field_name}")
            except Exception as idx_err:
                err_str = str(idx_err).lower()
                if "already exists" in err_str or "conflict" in err_str:
                    print(f"    - Index {field_name} already exists")
                else:
                    print(f"    ⚠️  Index {field_name}: {idx_err}")
        
        print(f"  ✅ Collection {collection_name} fixed successfully")
            
    except Exception as e:
        print(f"  ❌ Error fixing collection {collection_name}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("SPARSE VECTORS FIX COMPLETE")
print("="*80)
