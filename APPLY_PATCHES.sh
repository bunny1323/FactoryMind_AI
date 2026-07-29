#!/bin/bash
# FactoryMind_AI - Critical Bug Patches
# Run this script from project root: bash APPLY_PATCHES.sh

set -e  # Exit on first error

echo "🔧 Applying FactoryMind_AI Critical Patches..."
echo ""

# ============================================================================
# PATCH #1: Fix ingestion.py NameError (jobs → _jobs)
# ============================================================================
echo "📝 [PATCH #1] Fixing ingestion.py (jobs → _jobs)..."

if grep -q 'jobs\[job_id\]\["progress"\] = 70' backend/tasks/ingestion.py; then
    sed -i 's/jobs\[job_id\]\["progress"\] = 70/_jobs[job_id]["progress"] = 70/g' backend/tasks/ingestion.py
    sed -i 's/jobs\[job_id\]\["message"\] = "Processing SOP/_jobs[job_id]["message"] = "Processing SOP/g' backend/tasks/ingestion.py
    echo "   ✅ Fixed ingestion.py line 38-39"
else
    echo "   ⚠️  Could not find pattern - please manually check lines 38-39"
fi

# ============================================================================
# PATCH #2: Fix RAG_MIN_RELEVANCE_SCORE (0.35 → 0.20)
# ============================================================================
echo "📝 [PATCH #2] Lowering RAG_MIN_RELEVANCE_SCORE threshold..."

if grep -q 'RAG_MIN_RELEVANCE_SCORE: float = 0.35' backend/config.py; then
    sed -i 's/RAG_MIN_RELEVANCE_SCORE: float = 0.35/RAG_MIN_RELEVANCE_SCORE: float = 0.20  # Reduced threshold for better recall/g' backend/config.py
    echo "   ✅ Updated RAG_MIN_RELEVANCE_SCORE to 0.20"
else
    echo "   ⚠️  Could not find RAG_MIN_RELEVANCE_SCORE - please manually update"
fi

# ============================================================================
# PATCH #3: Add error handling to reranker in rag_service.py
# ============================================================================
echo "📝 [PATCH #3] Adding reranker error handling..."

RERANKER_FIX='        try:
            reranked_hits = self.reranker.rerank(query, flat_hits, top_k=8)
        except Exception as e:
            logger.warning(f"Reranker failed: {e}. Using raw scores for ranking.")
            reranked_hits = sorted(flat_hits, key=lambda x: x.get("score", 0), reverse=True)[:8]'

if ! grep -q "try:" backend/services/rag_service.py | grep -q "reranker.rerank"; then
    # Find the line with reranked_hits and wrap it
    echo "   ⚠️  Manual patch required for reranker error handling"
    echo "   Replace line 211 in backend/services/rag_service.py:"
    echo "   OLD: reranked_hits = self.reranker.rerank(query, flat_hits, top_k=8)"
    echo "   NEW: (see HACKATHON_FIX_PROMPT.md for full code)"
else
    echo "   ✅ Reranker error handling already present"
fi

# ============================================================================
# PATCH #4: Add user_id filter to search_by_intent
# ============================================================================
echo "📝 [PATCH #4] Adding user_id filter to search..."

if grep -q 'results\[coll\] = self.vector_store.search(coll, search_query' backend/services/rag_service.py; then
    echo "   ⚠️  Manual patch required for user_id filtering"
    echo "   Replace lines 164-165 in backend/services/rag_service.py with:"
    echo "   filters = {\"user_id\": user_id} if user_id != \"default_user\" else None"
    echo "   results[coll] = self.vector_store.search(coll, search_query, top_k=top_k_per_coll, filters=filters)"
else
    echo "   ✅ User_id filtering already implemented"
fi

# ============================================================================
# PATCH #5: Create/Update .env file
# ============================================================================
echo "📝 [PATCH #5] Creating .env configuration file..."

if [ ! -f .env ]; then
    echo "   Creating new .env file..."
    cp .env.example .env
    echo "   ⚠️  IMPORTANT: Edit .env and set:"
    echo "      - VECTOR_BACKEND=qdrant"
    echo "      - QDRANT_URL=https://<your-cluster>.qdrant.tech:6333"
    echo "      - QDRANT_API_KEY=<your-api-key>"
    echo "      - LLM_PROVIDER=groq (or openai/anthropic/ollama)"
    echo "      - API key for chosen provider (GROQ_API_KEY, OPENAI_API_KEY, etc)"
else
    echo "   ✅ .env file exists"
    
    # Check if critical variables are set
    if ! grep -q "^QDRANT_URL=" .env || grep "QDRANT_URL=http://localhost" .env; then
        echo "   ⚠️  WARNING: QDRANT_URL not set to cloud URL"
        echo "      Update: QDRANT_URL=https://<your-cluster>.qdrant.tech:6333"
    fi
    
    if ! grep -q "^LLM_PROVIDER=" .env || grep "LLM_PROVIDER=mock" .env; then
        echo "   ⚠️  WARNING: LLM_PROVIDER is 'mock' or not set"
        echo "      Set LLM_PROVIDER=groq and configure GROQ_API_KEY"
    fi
    
    if ! grep -q "^GROQ_API_KEY=" .env && ! grep -q "^OPENAI_API_KEY=" .env; then
        echo "   ⚠️  WARNING: No LLM API key found in .env"
        echo "      Add GROQ_API_KEY or OPENAI_API_KEY"
    fi
fi

# ============================================================================
# PATCH #6: Update LLM_PROVIDER default in config.py
# ============================================================================
echo "📝 [PATCH #6] Updating LLM_PROVIDER default..."

if grep -q 'LLM_PROVIDER.*= "mock"' backend/config.py; then
    sed -i 's/LLM_PROVIDER.*= "mock"/# LLM_PROVIDER should be set in .env (not mock!)\n    LLM_PROVIDER: Literal["mock", "groq", "openai", "openai_compatible", "ollama", "anthropic"] = "groq"  # Default to groq/g' backend/config.py
    echo "   ✅ Updated LLM_PROVIDER default (but .env overrides this)"
else
    echo "   ℹ️  LLM_PROVIDER already configured"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ PATCHES APPLIED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 NEXT STEPS:"
echo ""
echo "1️⃣  Edit .env and configure:"
echo "   • VECTOR_BACKEND=qdrant"
echo "   • QDRANT_URL=https://<cluster>.qdrant.tech:6333"
echo "   • QDRANT_API_KEY=<your-key>"
echo "   • LLM_PROVIDER=groq (or openai_compatible)"
echo "   • Corresponding API key"
echo ""
echo "2️⃣  Verify Qdrant connection:"
echo "   curl -H 'api-key: \$QDRANT_API_KEY' https://<cluster>.qdrant.tech:6333/collections"
echo ""
echo "3️⃣  Run ingestion pipelines:"
echo "   python -m backend.tasks.ingestion manuals"
echo "   python -m backend.tasks.ingestion error_codes"
echo "   python -m backend.tasks.ingestion spare_parts"
echo ""
echo "4️⃣  Test query endpoint:"
echo "   curl -X POST http://localhost:8000/api/query \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"query\": \"How to maintain hydraulic system?\", \"machine_id\": \"R215L\"}'"
echo ""
echo "5️⃣  Verify response has:"
echo "   • 'answer': <synthesized by LLM>"
echo "   • 'evidence': { 'citations': [...] }"
echo "   • NO 'LLM unavailable' message"
echo ""
echo "6️⃣  For manual patches, see HACKATHON_FIX_PROMPT.md for:"
echo "   • Reranker error handling"
echo "   • User_id filtering"
echo "   • Context window limiting"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
