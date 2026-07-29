# FactoryMind_AI - CRITICAL BUGS ANALYSIS & FIXES

## Executive Summary
Found **5 CRITICAL ISSUES** causing "LLM unavailable" and "No relevant information found" errors. Issues span ingestion pipeline, LLM configuration, RAG retrieval, and reranking logic.

---

## 🔴 CRITICAL BUG #1: Ingestion Pipeline Crash
**File**: `backend/tasks/ingestion.py` Line 38  
**Severity**: CRITICAL  
**Impact**: Data never gets indexed into Qdrant Cloud

### Problem
```python
# LINE 38 - WRONG
jobs[job_id]["progress"] = 70  # NameError: 'jobs' is not defined
jobs[job_id]["message"] = "Processing SOP files..."
```

Should be:
```python
# LINE 38 - CORRECT
_jobs[job_id]["progress"] = 70  # Use underscore prefix
_jobs[job_id]["message"] = "Processing SOP files..."
```

### Why This Breaks Everything
1. Ingestion task crashes immediately
2. Collections never get created in Qdrant
3. No data → Search returns empty → "No relevant information" error
4. Users think embeddings work, but collections are empty

---

## 🔴 CRITICAL BUG #2: Default LLM Provider is "mock"
**File**: `backend/config.py` Line 45  
**Severity**: CRITICAL  
**Impact**: All queries get extractive fallback instead of synthesized answers

### Problem
```python
LLM_PROVIDER: Literal[...] = "mock"  # Line 45 - DEFAULT IS MOCK!
```

When `LLM_PROVIDER=mock`:
- No actual LLM synthesis occurs
- Falls back to extractive answers only
- Users see "No relevant information" even when data exists

### Why This Causes "LLM unavailable"
```python
# In llm_service.py line 90-91
if primary_provider == "mock":
    return self._extractive_fallback(query, context, "mock", "Mock mode - no LLM synthesis")
```

**Result**: Always returns fallback message even with perfect RAG data

---

## 🔴 CRITICAL BUG #3: Missing API Key Validation at Startup
**File**: `backend/dependencies.py` Line 92-96  
**Severity**: HIGH  
**Impact**: Server starts but all queries fail with cryptic "provider unavailable" errors

### Problem
No early validation that required API keys exist. Error only appears at query time.

```python
# Current code silently continues even with no API key
# Should FAIL FAST on startup
logger.error(f"LLM_PROVIDER: '{settings.LLM_PROVIDER}' configured but the required API key is missing...")
```

### Fix: Add explicit startup check
```python
if provider != "mock" and not has_configured_provider:
    raise RuntimeError(f"LLM_PROVIDER={provider} but no API key found. Cannot start server.")
```

---

## 🔴 CRITICAL BUG #4: RAG Relevance Score Filter Too Strict
**File**: `backend/services/rag_service.py` Line 216  
**Severity**: MEDIUM  
**Impact**: Valid search results get filtered out, returns "No relevant information"

### Problem
```python
filtered_hits = [
    hit for hit in reranked_hits 
    if hit.get("score", 0.0) >= settings.RAG_MIN_RELEVANCE_SCORE  # Default = 0.35
]

if not filtered_hits:
    filtered_hits = reranked_hits[:3]  # Fallback to top 3 if nothing passes filter
```

**Issue**: 
- CrossEncoder reranker scores can be 0.1-0.9 scale (not 0-1)
- Default 0.35 threshold is too aggressive
- Even good results get filtered

### Example Scenario
- User asks: "How do I replace hydraulic pump?"
- RAG retrieves document with score 0.32 (actually relevant!)
- 0.32 < 0.35 threshold → filtered out
- No fallback docs if all scores < threshold
- Returns: "No relevant information" ❌

---

## 🔴 CRITICAL BUG #5: Embedding Dimension Mismatch Risk
**File**: `rag/embeddings.py` & `backend/config.py`  
**Severity**: MEDIUM  
**Impact**: Query embeddings don't match indexed vectors → Zero results

### Problem
```python
# In embeddings.py line 45
self.dimension = 384  # Hardcoded for BAAI/bge-small-en-v1.5

# But config.py allows different dimensions
EMBEDDING_DIMENSION: int = 384  # Configurable!
```

### Scenario Where This Breaks
1. User ingests data with EMBEDDING_DIMENSION=256 (custom config)
2. Config later changes to EMBEDDING_DIMENSION=384
3. New queries use 384-dim embeddings
4. Qdrant has 256-dim vectors
5. Search fails silently or returns garbage results

---

## 🟡 MAJOR ISSUES (Not Blocking but Impact Quality)

### Issue #6: No Context Window Management
**File**: `backend/services/rag_service.py` Line 257  
**Problem**: Context block concatenation doesn't limit tokens
- Can exceed LLM max_tokens (1024 tokens in config)
- LLM truncates important context mid-sentence
- **Fix**: Implement token counting and smart truncation

### Issue #7: Missing Error Handling in Reranker
**File**: `backend/services/rag_service.py` Line 211  
**Problem**: If reranker fails silently, returns empty list
```python
reranked_hits = self.reranker.rerank(query, flat_hits, top_k=8)
# No try-catch! If reranker.py crashes, pipeline dies
```

### Issue #8: Collection Filter User Isolation Not Enforced
**File**: `backend/services/rag_service.py` Line 164-165  
**Problem**: `search_by_intent()` retrieves by intent but doesn't filter by `user_id`
```python
results = {}
for coll in collections:
    try:
        results[coll] = self.vector_store.search(coll, search_query, top_k=top_k_per_coll)
        # MISSING: user_id filter! All users see same results
```

---

## 📊 Root Cause Analysis: Why Users See These Errors

### Error: "LLM unavailable for the questions asked"
**Root Cause Chain**:
1. Default `LLM_PROVIDER=mock` (Bug #2)
2. No API key configured (Bug #3)
3. Query gets extractive fallback
4. If RAG finds anything → Good
5. If RAG finds nothing → Message shows "LLM unavailable"

### Error: "Could not find relevant information in indexed manuals"
**Root Cause Chain**:
1. Ingestion crashes (Bug #1) → No data in Qdrant
2. OR relevance score filters too strict (Bug #4) → Valid results filtered
3. OR embedding dimension mismatch (Bug #5) → Search returns garbage
4. Empty results → Fallback message

---

## ✅ STEP-BY-STEP FIX GUIDE

### FIX #1: Repair ingestion.py (CRITICAL)
```bash
Line 38: jobs → _jobs
Line 39: jobs → _jobs
```

### FIX #2: Update .env Configuration (CRITICAL)
```env
# MUST BE SET - Choose one LLM provider
LLM_PROVIDER=groq
GROQ_API_KEY=your_actual_key

# OR
LLM_PROVIDER=openai_compatible
OPENAI_API_KEY=your_key
OPENAI_MODEL=gemini-2.0-flash
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/

# Qdrant Cloud must be configured
VECTOR_BACKEND=qdrant
QDRANT_URL=https://your-cluster.qdrant.tech:6333
QDRANT_API_KEY=your_api_key

# Recommended RAG settings
RAG_MIN_RELEVANCE_SCORE=0.20  # Lower threshold for more results
EMBEDDING_DIMENSION=384
```

### FIX #3: Add Startup Validation (HIGH)
Insert in `backend/dependencies.py` after line 96:
```python
if provider not in ("mock", "ollama"):
    if provider == "groq" and not groq_key:
        raise RuntimeError("LLM_PROVIDER=groq but GROQ_API_KEY not set!")
    if provider == "openai" and not openai_key:
        raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY not set!")
```

### FIX #4: Lower Relevance Threshold (MEDIUM)
In `backend/config.py` line 41:
```python
RAG_MIN_RELEVANCE_SCORE: float = 0.20  # Was 0.35 (too strict)
```

### FIX #5: Add Error Handling (MEDIUM)
In `backend/services/rag_service.py` line 211:
```python
try:
    reranked_hits = self.reranker.rerank(query, flat_hits, top_k=8)
except Exception as e:
    logger.warning(f"Reranker failed: {e}. Using top-8 by initial scores.")
    reranked_hits = sorted(flat_hits, key=lambda x: x.get("score", 0), reverse=True)[:8]
```

---

## 🚀 DEPLOYMENT CHECKLIST FOR HACKATHON

- [ ] **CRITICAL**: Fix `ingestion.py` line 38-39 (jobs → _jobs)
- [ ] **CRITICAL**: Set `LLM_PROVIDER` to actual provider (groq/openai_compatible/ollama)
- [ ] **CRITICAL**: Configure API key for chosen provider
- [ ] **CRITICAL**: Set `VECTOR_BACKEND=qdrant` with valid URL and API key
- [ ] **HIGH**: Lower `RAG_MIN_RELEVANCE_SCORE` to 0.20
- [ ] **HIGH**: Run full ingestion pipeline (manuals → error_codes → spare_parts)
- [ ] **HIGH**: Test query end-to-end: "How do I replace hydraulic pump?"
- [ ] **MEDIUM**: Add error handling in reranker
- [ ] **MEDIUM**: Verify embedding dimensions match across all components
- [ ] **MEDIUM**: Add user_id filtering to search_by_intent()
- [ ] Test with actual Qdrant Cloud (not localhost)
- [ ] Verify collections exist: `GET /collections` on Qdrant
- [ ] Check data was indexed: Get point counts per collection

---

## 🧪 TESTING COMMANDS

```bash
# Test Qdrant connection
curl -H "api-key: $QDRANT_API_KEY" https://your-cluster.qdrant.tech:6333/collections

# Test ingestion (after fixing bugs)
python -m backend.tasks.ingestion run_ingestion_task("test_job", "manuals")

# Test query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I maintain the hydraulic system?",
    "machine_id": "R215L"
  }'

# Expected output should have:
# - "answer": <synthesized response from LLM>
# - "evidence": { "citations": [<retrieved docs>] }
# - NOT "No relevant information" message
```

---

## 📋 FILES REQUIRING CHANGES

| File | Issue | Lines | Severity |
|------|-------|-------|----------|
| `backend/tasks/ingestion.py` | NameError: jobs → _jobs | 38-39 | CRITICAL |
| `backend/config.py` | Wrong LLM_PROVIDER default | 45 | CRITICAL |
| `.env` | Missing API keys | ALL | CRITICAL |
| `backend/config.py` | RAG threshold too high | 41 | MEDIUM |
| `backend/services/rag_service.py` | No reranker error handling | 211 | MEDIUM |
| `backend/services/rag_service.py` | Missing user_id filter | 164-165 | MEDIUM |

---

## 🎯 Expected Outcomes After Fixes

✅ Ingestion completes successfully  
✅ Collections appear in Qdrant with data  
✅ Search returns 3-8 relevant documents  
✅ Reranker scores these documents  
✅ LLM synthesizes coherent answer with citations  
✅ User gets: "Based on manual XYZ, section ABC: <answer>"  
✅ NOT: "LLM unavailable" or "No relevant information"  
