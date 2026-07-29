# FactoryMind_AI - Complete Hackathon Fix Prompt

## Overview
This prompt contains ALL necessary changes to fix the FactoryMind_AI RAG system and make it production-ready for a hackathon submission.

---

## PROMPT TO USE WITH CLAUDE/ANY AI ASSISTANT

```
You are a senior full-stack AI/ML engineer fixing a Retrieval Augmented Generation (RAG) system for a hackathon.

The system is experiencing two critical errors:
1. "LLM unavailable for the questions asked"
2. "Could not find relevant information in indexed manuals"

ROOT CAUSE: Multiple interconnected bugs in the ingestion pipeline, LLM configuration, and RAG retrieval logic.

CRITICAL BUGS TO FIX:

[BUG #1] INGESTION PIPELINE CRASH
File: backend/tasks/ingestion.py
Lines: 38-39
Current code:
```
    jobs[job_id]["progress"] = 70
    jobs[job_id]["message"] = "Processing SOP files..."
```
Issue: NameError - 'jobs' is not defined (should be '_jobs')
This prevents ANY data from being indexed into Qdrant Cloud
Fix: Replace 'jobs' with '_jobs' on lines 38-39

[BUG #2] WRONG DEFAULT LLM PROVIDER
File: backend/config.py
Line: 45
Current code:
```python
LLM_PROVIDER: Literal["mock", "groq", "openai", ...] = "mock"
```
Issue: Default is "mock" mode - no real LLM synthesis happens
Impact: All queries return extractive fallback only
Fix: Change default to "groq" and add warning comment that user must set API key

[BUG #3] NO API KEY VALIDATION AT STARTUP
File: backend/dependencies.py
After line 96:
Issue: Server starts successfully even with missing API keys
Error only appears at query time with cryptic messages
Fix: Add explicit fail-fast check:
```python
if provider not in ("mock", "ollama") and not self._has_api_key_for_provider(provider):
    raise RuntimeError(
        f"LLM_PROVIDER='{provider}' configured but required API key not found in .env. "
        f"Set GROQ_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY and restart."
    )
```

[BUG #4] RAG RELEVANCE SCORE FILTER TOO STRICT
File: backend/config.py
Line: 41
Current code:
```python
RAG_MIN_RELEVANCE_SCORE: float = 0.35
```
Issue: CrossEncoder reranker scores range 0.1-0.9, not 0-1. Threshold 0.35 filters valid results.
Impact: Good documents get excluded, returns "No relevant information"
Fix: Reduce to 0.20 and update comment:
```python
RAG_MIN_RELEVANCE_SCORE: float = 0.20  # CrossEncoder scale: filters documents with relevance < 20%
```

[BUG #5] NO ERROR HANDLING IN RERANKER
File: backend/services/rag_service.py
Line: 211
Current code:
```python
reranked_hits = self.reranker.rerank(query, flat_hits, top_k=8)
```
Issue: If reranker fails silently, pipeline returns empty results
Impact: Unhandled exceptions crash the query pipeline
Fix: Wrap in try-except:
```python
try:
    reranked_hits = self.reranker.rerank(query, flat_hits, top_k=8)
except Exception as e:
    logger.warning(f"Reranker failed ({e}). Using initial ranking.")
    reranked_hits = sorted(flat_hits, key=lambda x: x.get("score", 0), reverse=True)[:8]
```

[BUG #6] MISSING USER_ID ISOLATION IN SEARCH
File: backend/services/rag_service.py
Lines: 164-165
Current code:
```python
for coll in collections:
    try:
        results[coll] = self.vector_store.search(coll, search_query, top_k=top_k_per_coll)
```
Issue: No user_id filter passed to search - all users see same results
Impact: Multi-tenant isolation broken
Fix: Pass filters parameter:
```python
filters = {"user_id": user_id} if user_id != "default_user" else None
results[coll] = self.vector_store.search(coll, search_query, top_k=top_k_per_coll, filters=filters)
```

[BUG #7] CONTEXT WINDOW OVERFLOW
File: backend/services/rag_service.py
Line: 257
Current code:
```python
context = "\n\n".join(context_blocks)
```
Issue: No token limit - can exceed LLM max_tokens (1024), truncating context
Impact: LLM gets incomplete information
Fix: Add token limiting:
```python
# Estimate tokens (rough: 1 token ≈ 4 chars)
max_context_chars = 3000  # ~750 tokens, leaving room for query
context = "\n\n".join(context_blocks)
if len(context) > max_context_chars:
    context = context[:max_context_chars] + "\n[... content truncated ...]"
```

ENVIRONMENT CONFIGURATION FIXES:

Current .env likely has:
- LLM_PROVIDER=mock (or missing)
- Missing GROQ_API_KEY / OPENAI_API_KEY
- QDRANT_URL=http://localhost:6333 (not cloud URL)

Fix: Update .env with actual credentials:
```env
# Vector DB - MUST USE QDRANT CLOUD
VECTOR_BACKEND=qdrant
QDRANT_URL=https://<YOUR-CLUSTER>.qdrant.tech:6333
QDRANT_API_KEY=<YOUR-ACTUAL-API-KEY>

# Embeddings
EMBEDDING_BACKEND=fastembed
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSION=384
SPARSE_EMBEDDING_BACKEND=fastembed
SPARSE_EMBEDDING_MODEL=Qdrant/bm25

# Reranker
RERANKER_BACKEND=cross_encoder
RERANKER_MODEL=BAAI/bge-reranker-base
RAG_MIN_RELEVANCE_SCORE=0.20

# LLM Provider - CHOOSE ONE
# Option 1: Groq (fastest, free tier available)
LLM_PROVIDER=groq
GROQ_API_KEY=<your-groq-api-key>
GROQ_MODEL=llama-3.3-70b-versatile

# Option 2: Google AI Studio (free, no rate limits)
# LLM_PROVIDER=openai_compatible
# OPENAI_API_KEY=<your-google-aistudio-key>
# OPENAI_MODEL=gemini-2.0-flash
# OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/

# Option 3: OpenRouter (free tier, many models)
# LLM_PROVIDER=openai_compatible
# OPENAI_API_KEY=<your-openrouter-key>
# OPENAI_MODEL=meta-llama/llama-3.1-8b-instruct:free
# OPENAI_BASE_URL=https://openrouter.ai/api/v1

# Option 4: Local Ollama (full offline capability)
# LLM_PROVIDER=ollama
# OLLAMA_URL=http://localhost:11434
# OLLAMA_MODEL=qwen2.5:7b
```

VERIFICATION & TESTING AFTER FIXES:

1. Verify Qdrant connectivity:
   ```bash
   curl -H "api-key: $QDRANT_API_KEY" \
     https://your-cluster.qdrant.tech:6333/collections
   ```
   Should return: `{"result": {"collections": []}}`

2. Run ingestion for all pipelines:
   ```python
   # After fixing ingestion.py bug
   for pipeline in ["manuals", "error_codes", "spare_parts", "maintenance_logs"]:
       ingestion_task.run_ingestion_task(f"job_{pipeline}", pipeline)
   ```

3. Verify data indexed:
   ```bash
   curl -H "api-key: $QDRANT_API_KEY" \
     https://your-cluster.qdrant.tech:6333/collections/manuals
   ```
   Should show: `"points_count": > 100`

4. Test query end-to-end:
   ```bash
   curl -X POST http://localhost:8000/api/query \
     -H "Authorization: Bearer $(jwt_token)" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What are the maintenance procedures for the hydraulic system?",
       "machine_id": "R215L"
     }'
   ```
   
   Expected response:
   ```json
   {
     "query_id": "...",
     "answer": "Based on Hyundai R215L Manual Section 5.2, maintenance procedures include: 1. Check hydraulic fluid level... [SYNTHESIZED BY LLM]",
     "evidence": {
       "citations": [
         {
           "id": "...",
           "title": "Hydraulic System Maintenance",
           "text": "...",
           "score": 0.75,
           "source_type": "manual"
         }
       ],
       "confidence_score": 0.95
     }
   }
   ```

5. Should NOT see:
   - ❌ "LLM unavailable"
   - ❌ "No relevant information found"
   - ❌ Any NameError or uncaught exceptions

ADDITIONAL QUALITY IMPROVEMENTS (For Hackathon Winning Score):

1. Add response streaming:
   File: backend/routes/query.py
   Add: `@router.post("/query/stream")` endpoint that streams answer as it's generated
   
2. Add multilingual support:
   Already implemented but verify: language_service detects query language
   
3. Add confidence scoring:
   File: backend/services/rag_service.py
   Calculate: avg(citation_scores) * reranker_confidence * llm_certainty
   
4. Add visual diagram detection:
   Already implemented - ensure image_url and image_description return properly
   
5. Add conversation context memory:
   Already implemented - verify conversation_memory stores multi-turn context

TROUBLESHOOTING IF ERRORS PERSIST:

Error: "Collection not found in Qdrant"
→ Run ingestion pipeline for that collection
→ Verify QDRANT_URL and API_KEY are correct

Error: "Embedding dimension mismatch"
→ Check EMBEDDING_DIMENSION in config matches Qdrant collection
→ Rebuild collection if dimension was changed

Error: "LLM timeout"
→ Increase timeout in llm_service.py (currently 30s)
→ Check LLM provider rate limits
→ Switch to faster model (groq is fastest free option)

Error: "Qdrant API key unauthorized"
→ Verify API_KEY is correct (no extra spaces/quotes)
→ Check URL uses HTTPS not HTTP
→ Verify API key has read/write permissions

FINAL DEPLOYMENT CHECKLIST:

☐ Fixed ingestion.py (jobs → _jobs)
☐ Updated backend/config.py (LLM_PROVIDER, RAG_MIN_RELEVANCE_SCORE)
☐ Updated backend/services/rag_service.py (error handling, user_id filter)
☐ Updated backend/dependencies.py (startup validation)
☐ Created .env with actual credentials
☐ VECTOR_BACKEND=qdrant with cloud URL
☐ LLM_PROVIDER set to actual provider with API key
☐ Ran ingestion for all collections
☐ Verified data exists in Qdrant
☐ Tested query returns answer + citations
☐ No "LLM unavailable" or "No relevant information" errors
☐ Response includes confidence_score and source citations
☐ MultiLanguage support working
☐ Visual diagrams detected and returned
☐ Conversation context maintained across turns

YOU ARE NOW READY FOR HACKATHON SUBMISSION!
```

---

## Quick Copy-Paste Fixes

### Fix #1: ingestion.py
Find:
```python
jobs[job_id]["progress"] = 70
jobs[job_id]["message"] = "Processing SOP files..."
```
Replace with:
```python
_jobs[job_id]["progress"] = 70
_jobs[job_id]["message"] = "Processing SOP files..."
```

### Fix #2: config.py (Line 41)
Find:
```python
RAG_MIN_RELEVANCE_SCORE: float = 0.35
```
Replace with:
```python
RAG_MIN_RELEVANCE_SCORE: float = 0.20  # Reduced from 0.35 to allow more results
```

### Fix #3: config.py (Line 45)
Find:
```python
LLM_PROVIDER: Literal["mock", "groq", "openai", "openai_compatible", "ollama", "anthropic"] = "mock"
```
Replace with:
```python
# IMPORTANT: Set to actual provider (groq/openai/anthropic/ollama) - NOT mock!
LLM_PROVIDER: Literal["mock", "groq", "openai", "openai_compatible", "ollama", "anthropic"] = "groq"
```

### Fix #4: rag_service.py (Add around line 211)
Find:
```python
reranked_hits = self.reranker.rerank(query, flat_hits, top_k=8)
```
Replace with:
```python
try:
    reranked_hits = self.reranker.rerank(query, flat_hits, top_k=8)
except Exception as e:
    logger.warning(f"Reranker failed: {e}. Using raw scores for ranking.")
    reranked_hits = sorted(flat_hits, key=lambda x: x.get("score", 0), reverse=True)[:8]
```

### Fix #5: rag_service.py (Line 164-165)
Find:
```python
for coll in collections:
    try:
        results[coll] = self.vector_store.search(coll, search_query, top_k=top_k_per_coll)
```
Replace with:
```python
filters = {"user_id": user_id} if user_id != "default_user" else None
for coll in collections:
    try:
        results[coll] = self.vector_store.search(coll, search_query, top_k=top_k_per_coll, filters=filters)
```

---

## Expected Results After Fixes

### ✅ BEFORE FIX (Broken)
```json
{
  "error": "pipeline_failure",
  "detail": "No relevant information was found in the indexed manuals for your query."
}
```

### ✅ AFTER FIX (Working)
```json
{
  "query_id": "550e8400-e29b-41d4-a716-446655440000",
  "answer": "Based on the Hyundai R215L maintenance manual, here are the key steps for hydraulic system maintenance:\n\n1. **Fluid Check**: Ensure hydraulic fluid level is between min and max marks on the sight glass...",
  "evidence": {
    "citations": [
      {
        "id": "hyundai_r215l_manual_chunk_42",
        "title": "Hydraulic System Maintenance Procedures",
        "text": "Monthly: Check fluid level, inspect hoses for leaks. Quarterly: Change filter...",
        "score": 0.87,
        "source_type": "manual",
        "payload": {
          "document_name": "Hyundai_R215L_Manual.pdf",
          "page": 87,
          "heading": "5.2 Hydraulic System",
          "chunk_type": "text"
        }
      }
    ],
    "confidence_score": 0.92,
    "confidence_breakdown": {
      "overall": "High",
      "retrieval": 0.87,
      "relevance": 0.92,
      "answer_quality": 0.95
    }
  }
}
```

---

## Support & Debugging

**Test Qdrant directly**:
```bash
# List collections
curl -H "api-key: YOUR_KEY" https://cluster.qdrant.tech:6333/collections

# Get collection info
curl -H "api-key: YOUR_KEY" https://cluster.qdrant.tech:6333/collections/manuals

# Search manually
curl -X POST -H "api-key: YOUR_KEY" \
  https://cluster.qdrant.tech:6333/collections/manuals/points/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, ...], "limit": 5}'
```

**Enable debug logging**:
In .env:
```env
LOG_LEVEL=DEBUG
```

**Monitor ingestion**:
```python
# In backend/tasks/ingestion.py
logger.info(f"Processing {len(records)} records...")
logger.info(f"Successfully ingested {result} records")
logger.error(f"Failed: {e}", exc_info=True)
```

---

## Success Criteria for Hackathon

✅ System retrieves relevant documents from manuals
✅ LLM synthesizes coherent answers with proper citations
✅ Response includes confidence scores and source references
✅ Handles multi-turn conversations with context memory
✅ Supports visual diagram detection and description
✅ No crashes or "unavailable" error messages
✅ Fast response times (<5 seconds per query)
✅ Clean, well-documented codebase
✅ Proper error handling and logging

**Expected Outcome**: Judges can ask questions about R215L excavator maintenance, system retrieves relevant sections from manuals, LLM synthesizes coherent technical answers, and provides citations showing where information came from.
