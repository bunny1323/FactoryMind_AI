"""Script to re-ingest manuals after sparse vectors fix."""
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from backend.dependencies import container
from backend.config import settings
from ingestion.ingest_manuals import run_manuals_ingestion

print("="*80)
print("RE-INGESTING MANUALS COLLECTION")
print("="*80)

data_dir = settings.DATA_DIR
manuals_dir = os.path.join(data_dir, "manuals")
sop_dir = os.path.join(data_dir, "sop")

print(f"Manuals directory: {manuals_dir}")
print(f"SOP directory: {sop_dir}")
print(f"Vector store type: {type(container.vector_store).__name__}")

# Check if directories exist
if not os.path.exists(manuals_dir):
    print(f"ERROR: Manuals directory does not exist: {manuals_dir}")
    sys.exit(1)

# List PDF files
pdf_files = [f for f in os.listdir(manuals_dir) if f.endswith('.pdf')]
print(f"Found {len(pdf_files)} PDF files in manuals directory:")
for pdf in pdf_files:
    print(f"  - {pdf}")

try:
    print("\nStarting manuals ingestion...")
    count = run_manuals_ingestion(container.vector_store, manuals_dir, "manuals", user_id="default_user")
    print(f"✅ Ingested {count} manual chunks")

    if os.path.exists(sop_dir):
        print("\nStarting SOP ingestion...")
        sop_count = run_manuals_ingestion(container.vector_store, sop_dir, "sop", user_id="default_user")
        print(f"✅ Ingested {sop_count} SOP chunks")
    else:
        print("⚠️  SOP directory does not exist, skipping")

    print("\n" + "="*80)
    print("RE-INGESTION COMPLETE")
    print("="*80)

except Exception as e:
    print(f"❌ Error during ingestion: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
