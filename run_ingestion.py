"""
Run all ingestion pipelines to populate Qdrant collections.
Usage: python run_ingestion.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Run all ingestion pipelines sequentially."""
    print("="*70)
    print("FactoryMind AI - Ingestion Pipeline Runner")
    print("="*70)
    
    # Import manuals ingestion
    from ingestion.ingest_manuals import ingest_manuals
    
    print("\nRunning Manuals Ingestion (Refactored)...")
    print("-"*70)
    
    try:
        total_chunks = ingest_manuals(
            manuals_dir="data/manuals",
            public_img_dir="public/extracted_images",
            collection_name="manuals",
            user_id="default_user",
            state_file="ingest_state.json",
            clear_state=True   # Always re-index to pick up pipeline improvements
        )
        
        print(f"\n✅ Manuals ingestion completed: {total_chunks} chunks indexed")
        
        # Run other pipelines (using existing task system)
        from backend.tasks.ingestion import run_ingestion_task, get_jobs
        
        other_pipelines = [
            ("error_codes", "Error codes database"),
            ("spare_parts", "Spare parts catalog"),
            ("maintenance_logs", "Maintenance logs"),
            ("graph", "Knowledge graph relationships"),
        ]
        
        for pipeline_id, description in other_pipelines:
            print(f"\n{'='*70}")
            print(f"Running: {description}")
            print(f"{'='*70}")
            
            job_id = f"ingest_{pipeline_id}"
            
            try:
                run_ingestion_task(job_id, pipeline_id, user_id="default_user")
                
                jobs = get_jobs()
                job = jobs.get(job_id, {})
                
                if job.get("status") == "completed":
                    print(f"✅ {description}: {job.get('message', 'Completed')}")
                else:
                    print(f"❌ {description}: {job.get('message', 'Failed')}")
                    
            except Exception as e:
                print(f"❌ Error running {pipeline_id}: {e}")
        
        print("\n" + "="*70)
        print("✅ All ingestion pipelines completed successfully")
        print("="*70)
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
