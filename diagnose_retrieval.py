"""
FactoryMind AI — Retrieval Diagnostic & Repair Script
======================================================
Runs the full diagnostic pipeline outlined in the master prompt:
  1. Check Qdrant collection point counts
  2. Sample 5 chunk payloads from each non-empty collection
  3. Verify that key terms (engine oil, hydraulic, torque, safety, maintenance, filters) appear
  4. Test retrieval for 7 canonical queries
  5. Show embedding model type (real FastEmbed vs Hash fallback)
  6. Show reranker type
  7. Clear ingest_state.json for manuals still marked as completed=false
     so that re-ingesting them becomes possible via the UI.

No data is modified unless --reset-state flag is given.
"""
from __future__ import annotations
import sys
import os
import json
import logging

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("factorymind").setLevel(logging.WARNING)
logging.getLogger("fastembed").setLevel(logging.ERROR)

RESET_STATE = "--reset-state" in sys.argv

SEPARATOR = "=" * 68

def sep(title: str = ""):
    if title:
        pad = max(0, 68 - len(title) - 4)
        print(f"\n{'='*2} {title} {'='*pad}")
    else:
        print(SEPARATOR)

# ── Step 0: Bootstrap dependencies ───────────────────────────────────────────
sep("Step 0 — Bootstrap (loading container)")
try:
    from backend.config import settings
    from backend.dependencies import container
    vs = container.vector_store
    embedder = container.embedder
    reranker = container.reranker
    print(f"  VECTOR_BACKEND : {settings.VECTOR_BACKEND}")
    print(f"  Embedder       : {embedder.__class__.__name__} (dim={getattr(embedder, 'dimension', '?')})")
    print(f"  Reranker       : {reranker.__class__.__name__}")
    print(f"  Qdrant URL     : {settings.QDRANT_URL[:50]}...")
except Exception as e:
    print(f"  FATAL: could not load container: {e}")
    sys.exit(1)

# ── Step 1: Collection point counts ──────────────────────────────────────────
sep("Step 1 — Collection point counts")
COLLECTIONS = ["manuals", "sop", "maintenance_logs", "error_codes", "spare_parts"]
try:
    stats = vs.get_stats()
    for name in COLLECTIONS:
        st = stats.get(name, {})
        count = st.get("count", "N/A")
        flag = " ⚠️  EMPTY — needs re-ingestion!" if isinstance(count, int) and count < 20 else ""
        print(f"  {name:<22} {count:>6} points{flag}")
except Exception as e:
    print(f"  ERROR reading stats: {e}")

# ── Step 2: Sample chunk payloads ─────────────────────────────────────────────
sep("Step 2 — Sample chunk payloads from manuals collection")
KEY_TERMS = ["engine oil", "hydraulic", "torque", "safety", "maintenance", "filter"]
terms_found: dict[str, bool] = {t: False for t in KEY_TERMS}

try:
    if settings.VECTOR_BACKEND == "qdrant":
        # Use Qdrant scroll to pull first 10 points
        scroll_result = vs.client.scroll(
            collection_name="manuals",
            limit=10,
            with_payload=True,
            with_vectors=False
        )
        points = scroll_result[0]
        if not points:
            print("  ⚠️  NO POINTS FOUND in manuals collection — collection is empty!")
        else:
            for i, pt in enumerate(points[:5]):
                payload = pt.payload or {}
                text = payload.get("text", "")[:300]
                doc = payload.get("document_name", "?")
                page = payload.get("page", "?")
                chunk_type = payload.get("chunk_type", "?")
                print(f"\n  [Chunk {i+1}] doc={doc}  page={page}  type={chunk_type}")
                print(f"  text preview: {text[:200]!r}")
                # Check key terms
                full_text = (payload.get("text", "") + " " + payload.get("heading", "")).lower()
                for term in KEY_TERMS:
                    if term in full_text:
                        terms_found[term] = True
    else:
        # In-memory store
        records = list(vs.collections.get("manuals", []))
        if not records:
            print("  ⚠️  In-memory manuals collection is EMPTY — ingestion has not run yet.")
        for i, rec in enumerate(records[:5]):
            text = (rec.text or "")[:300]
            doc = rec.payload.get("document_name", "?")
            page = rec.payload.get("page", "?")
            print(f"\n  [Chunk {i+1}] doc={doc}  page={page}")
            print(f"  text preview: {text[:200]!r}")
            full_text = (rec.text + " " + rec.title).lower()
            for term in KEY_TERMS:
                if term in full_text:
                    terms_found[term] = True

    print("\n  Key-term coverage:")
    for term, found in terms_found.items():
        status = "✓ FOUND" if found else "✗ MISSING — retrieval for this term will fail"
        print(f"    {term:<20} {status}")
except Exception as e:
    print(f"  ERROR sampling chunks: {e}")

# ── Step 3: Live retrieval tests ──────────────────────────────────────────────
sep("Step 3 — Live retrieval tests (hybrid search, no reranking)")
TEST_QUERIES = [
    "What engine oil grade is recommended?",
    "What is the torque specification for the swing motor bolts?",
    "Show hydraulic circuit diagram",
    "What safety warnings are listed?",
    "Filter replacement interval",
    "Maintenance schedule 500 hours",
    "List all indexed documents",
]

for query in TEST_QUERIES:
    try:
        hits = vs.search("manuals", query, top_k=3)
        if not hits:
            print(f"\n  QUERY: {query!r}")
            print(f"    → 0 results — ⚠️  retrieval returning empty")
        else:
            top = hits[0]
            print(f"\n  QUERY: {query!r}")
            print(f"    → {len(hits)} hits | top score={top['score']:.4f}")
            print(f"       title: {top.get('title','?')[:80]}")
            print(f"       text:  {top.get('text','')[:120]!r}")
    except Exception as e:
        print(f"\n  QUERY: {query!r}")
        print(f"    → ERROR: {e}")

# ── Step 4: Full query planner + rag_service end-to-end ──────────────────────
sep("Step 4 — QueryPlanner intent & collection routing")
try:
    from backend.services.query_planner import QueryPlanner
    planner = QueryPlanner()
    for query in TEST_QUERIES[:5]:
        plan = planner.plan(query)
        print(f"  Q: {query!r}")
        print(f"     intent={plan.intent}  collections={plan.target_collections}")
        print(f"     rewritten={plan.rewritten_query[:80]!r}")
        print()
except Exception as e:
    print(f"  ERROR: {e}")

# ── Step 5: Check ingest_state.json ──────────────────────────────────────────
sep("Step 5 — ingest_state.json completeness check")
STATE_FILE = os.path.join(ROOT, "ingest_state.json")
try:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        incomplete = []
        for filename, info in state.items():
            pages_indexed = len(info.get("indexed_pages", []))
            completed = info.get("completed", False)
            print(f"  {filename}")
            print(f"    indexed_pages={pages_indexed}  completed={completed}")
            if not completed:
                incomplete.append(filename)
        if incomplete:
            print(f"\n  ⚠️  {len(incomplete)} manual(s) were NOT fully ingested (completed=false):")
            for f in incomplete:
                print(f"       - {f}")
            if RESET_STATE:
                for fn in incomplete:
                    state[fn]["completed"] = False
                    state[fn]["indexed_pages"] = []
                with open(STATE_FILE, "w") as fp:
                    json.dump(state, fp, indent=2)
                print("\n  ✓ ingest_state.json RESET for incomplete manuals.")
                print("    Re-trigger ingestion from the Admin panel to re-index these files.")
        else:
            print("\n  ✓ All manuals marked as completed.")
    else:
        print("  ingest_state.json not found — no manuals have been ingested yet.")
except Exception as e:
    print(f"  ERROR reading ingest_state.json: {e}")

# ── Step 6: Data directory audit ──────────────────────────────────────────────
sep("Step 6 — Data directory audit")
data_dir = settings.DATA_DIR
for sub in ["manuals", "sop", "maintenance_logs", "error_codes", "spare_parts"]:
    path = os.path.join(data_dir, sub)
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        total_bytes = sum(os.path.getsize(os.path.join(path, f)) for f in files)
        print(f"  {sub:<22} {len(files):>4} files   {total_bytes/1024/1024:.2f} MB")
    else:
        print(f"  {sub:<22}  DIR DOES NOT EXIST")

# ── Summary ───────────────────────────────────────────────────────────────────
sep("SUMMARY & RECOMMENDED ACTIONS")
print("""
  1. If any collection shows 0 points → run ingestion from Admin panel.
  2. If embedder is HashEmbedder → FastEmbed failed to load; check pip install fastembed.
  3. If key terms are missing from sampled chunks → text extraction or chunking is broken.
  4. If retrieval returns 0 results for all queries → the collection is empty (ingestion never ran
     or ran on a different VECTOR_BACKEND than the one currently active in .env).
  5. If ingest_state.json shows completed=false → run --reset-state and re-ingest.

  RUN RESET:  python diagnose_retrieval.py --reset-state
  THEN re-trigger ingestion from the Admin panel (Ingest Manuals pipeline).
""")
sep()
