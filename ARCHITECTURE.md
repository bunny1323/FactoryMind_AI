# FactoryMind AI v2.0 - Architecture Documentation

## Overview

FactoryMind AI is an industrial maintenance intelligence platform for the Hyundai R215L Smart Plus excavator. It combines layout-aware multimodal RAG, knowledge graph querying, and predictive maintenance models to provide explainable, evidence-backed maintenance assistance.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Web Chat Interface                          │
│                    (Next.js + React 19)                        │
└─────────────────────┬───────────────────────────────────────────┘
                      │
              Conversation Memory
                      │
                      ▼
              Language Detection
                      │
                      ▼
          Agentic Query Planner (LangGraph)
                      │
      ┌───────────────┼───────────────┬───────────────┐
      ▼               ▼               ▼               ▼
  Manuals        Error Codes      SOPs      Spare Parts
      │               │               │               │
      └───────────────┴───────────────┴───────────────┘
                      │
              Hybrid Retrieval
      Dense + BM25 + RRF + Reranker
                      │
                      ▼
        Knowledge Graph Expansion (Neo4j)
                      │
                      ▼
      Multimodal Context Builder
   Text + Tables + Images + Diagrams
                      │
                      ▼
       OpenAI-Compatible LLM (Gemini/Groq)
                      │
                      ▼
     Explainable Response Generator
                      │
                      ▼
  Answer + Sources + Pages + Confidence
```

## Backend Architecture

### Directory Structure

```
backend/
├── main.py                      # FastAPI application entry point
├── config.py                    # Centralized configuration with validation
├── logging_config.py            # Structured logging configuration
├── exceptions.py                # Custom exceptions and error handling
├── constants.py                 # Application constants
├── telemetry.py                 # Machine telemetry management
├── db.py                        # SQLite database operations
├── dependencies.py              # Dependency injection container
├── auth/                        # Authentication module
│   ├── __init__.py
│   └── jwt_auth.py              # JWT authentication
├── models/                      # Pydantic models
│   ├── __init__.py
│   └── schemas.py               # Request/response schemas
├── routes/                      # API routes
│   ├── __init__.py
│   ├── auth.py                  # Authentication endpoints
│   ├── query.py                 # Query and RAG endpoints
│   ├── ingestion.py             # Document ingestion endpoints
│   ├── admin.py                 # Admin management endpoints
│   ├── machines.py              # Machine-related endpoints
│   ├── debug.py                 # Debug and diagnostic endpoints
│   ├── reports.py               # Report generation endpoints
│   └── prediction.py            # Prediction endpoints
├── services/                    # Business logic services
│   ├── rag_service.py           # RAG orchestration
│   └── llm_service.py           # LLM provider abstraction
└── tasks/                       # Background tasks
    ├── __init__.py
    └── ingestion.py             # Document ingestion tasks
```

### Core Components

#### 1. Configuration Management (`config.py`)
- **Purpose**: Centralized configuration with validation
- **Features**:
  - Pydantic-based settings with type hints
  - Environment variable loading from `.env`
  - Validation for API keys, URLs, and configuration values
  - Automatic fallback for missing configurations
  - Literal type constraints for enum-like values

#### 2. Structured Logging (`logging_config.py`)
- **Purpose**: Centralized logging with context support
- **Features**:
  - Structured logger with context injection
  - Multiple log levels (DEBUG, INFO, WARNING, ERROR)
  - Console and file handlers
  - Request context tracking

#### 3. Error Handling (`exceptions.py`)
- **Purpose**: Custom exception hierarchy and HTTP mapping
- **Features**:
  - Base `FactoryMindError` exception
  - Specific exceptions for different components
  - Automatic HTTP status code mapping
  - Detailed error responses with context

#### 4. Dependency Injection (`dependencies.py`)
- **Purpose**: Service container for singleton instances
- **Features**:
  - Lazy initialization of services
  - Singleton pattern for expensive resources
  - Connection pooling for vector store
  - Embedder and reranker initialization

#### 5. Route Modules
- **`auth.py`**: Authentication endpoints (`/login`, `/register`)
- **`query.py`**: Query processing with multi-agent orchestration
- **`ingestion.py`**: Document ingestion pipeline management
- **`admin.py`**: Admin operations (collection management, stats)
- **`machines.py`**: Machine telemetry and history
- **`debug.py`**: Diagnostic endpoints for troubleshooting
- **`reports.py`**: PDF report generation
- **`prediction.py`**: Predictive maintenance endpoints

### Data Flow

#### Query Processing Flow

1. **Request Reception**: FastAPI receives query request
2. **Authentication**: JWT token validation via `get_current_user`
3. **Intent Detection**: Classifies query type (greeting, visual, parts, etc.)
4. **Agent Orchestration**: LangGraph executes multi-agent pipeline:
   - Intent Detection Agent
   - Document Retrieval Agent
   - Knowledge Graph Agent
   - Future Prediction Agent
   - Evidence Aggregation Agent
   - Maintenance Planner Agent
   - Synthesizer Agent
5. **Response Generation**: LLM synthesizes final answer with citations
6. **Response Delivery**: JSON response with evidence bundle

#### Document Ingestion Flow

1. **File Upload**: Documents uploaded to data directories
2. **Pipeline Selection**: User selects ingestion pipeline
3. **Background Task**: Async task execution via FastAPI BackgroundTasks
4. **Document Processing**:
   - PDF text extraction (PyMuPDF)
   - OCR fallback (PaddleOCR → pytesseract → PyMuPDF)
   - Table extraction (pdfplumber → Camelot)
   - Image extraction (raster + vector graphics)
   - Layout parsing (headings, sections, tables)
5. **Chunking**: Structure-aware chunking with metadata
6. **Embedding**: Dense + sparse embedding generation
7. **Vector Storage**: Upsert to Qdrant/memory store
8. **Status Updates**: Real-time progress tracking

### Technology Stack

#### Backend
- **Framework**: FastAPI 0.115.12
- **Python**: 3.9+
- **Vector Database**: Qdrant (cloud/local) or in-memory
- **Embeddings**: FastEmbed (BAAI/bge-small-en-v1.5)
- **Sparse Embeddings**: Qdrant/bm25
- **Reranker**: Cross-encoder (BAAI/bge-reranker-base)
- **LLM**: OpenAI-compatible (Gemini, Groq, Anthropic, Ollama)
- **Knowledge Graph**: Neo4j
- **Document Processing**: PyMuPDF, pdfplumber, PaddleOCR
- **Machine Learning**: XGBoost, scikit-learn
- **Agent Framework**: LangGraph

#### Frontend
- **Framework**: Next.js 15.1.3
- **UI Library**: React 19.0.0
- **Styling**: Tailwind CSS 3.4.16
- **Icons**: Lucide React
- **Animations**: Framer Motion
- **Charts**: Recharts
- **Graph Visualization**: ReactFlow

## Key Design Patterns

### 1. Repository Pattern
- **Implementation**: `backend/db.py` for SQLite operations
- **Purpose**: Abstract data access logic
- **Benefits**: Testability, separation of concerns

### 2. Service Layer Pattern
- **Implementation**: `backend/services/` directory
- **Purpose**: Business logic encapsulation
- **Benefits**: Reusability, maintainability

### 3. Dependency Injection
- **Implementation**: `backend/dependencies.py`
- **Purpose**: Manage service lifecycles
- **Benefits**: Loose coupling, testability

### 4. Strategy Pattern
- **Implementation**: Multiple LLM providers, vector backends
- **Purpose**: Runtime configuration switching
- **Benefits**: Flexibility, extensibility

### 5. Agent Pattern
- **Implementation**: LangGraph multi-agent system
- **Purpose**: Complex query orchestration
- **Benefits**: Modular reasoning, explainability

## Security Considerations

### Authentication
- JWT-based authentication with configurable secret
- Role-based access control (admin/user)
- Mock authentication for development

### Data Isolation
- User-specific filtering in vector queries
- Collection-level access control
- Telemetry data isolation per machine

### API Security
- CORS configuration for allowed origins
- Input validation via Pydantic models
- SQL injection prevention via parameterized queries
- Secret management via environment variables

## Performance Optimizations

### Vector Store
- Singleton Qdrant client
- Connection pooling
- Batch upsert operations (100 points per batch)
- Payload indexing for filtered queries
- Hybrid retrieval with RRF (Reciprocal Rank Fusion)

### Caching
- RAG service answer cache
- Embedding model caching
- LLM response caching

### Async Processing
- Background task execution for ingestion
- Async route handlers where applicable
- Parallel page processing during ingestion

## Deployment Architecture

### Development
- Local development with in-memory vector store
- Mock LLM provider for testing
- SQLite database for persistence

### Production
- Qdrant Cloud for vector storage
- OpenAI-compatible LLM (Gemini via Google AI Studio)
- Neo4j Cloud for knowledge graph
- PostgreSQL for production database (recommended)

### Monitoring
- Structured logging with request context
- Performance metrics (latency, retrieval scores)
- Error tracking with exception handling
- Health check endpoints

## Scalability Considerations

### Horizontal Scaling
- Stateless API design
- Shared vector store (Qdrant Cloud)
- External Neo4j instance
- Load balancer support

### Vertical Scaling
- Connection pooling
- Batch processing
- Memory-efficient embeddings
- Lazy loading of models

## Future Extensions

### Phase 2-16 Roadmap
1. **Advanced Multimodal Document Ingestion**: Support for DOCX, PPTX, scanned manuals
2. **Structure-Aware Chunking**: Heading-based, table-aware chunking
3. **Multilingual RAG**: Support for 10+ languages
4. **Hybrid Retrieval 2.0**: Enhanced intent routing
5. **Agentic Query Planner**: Intelligent collection routing
6. **Contextual Retrieval**: Anthropic-style context expansion
7. **Knowledge Graph Layer**: Entity relationships
8. **Multimodal Retrieval**: Diagram and image retrieval
9. **Explainable AI**: Confidence scores and citations
10. **Predictive Maintenance**: IoT sensor integration
11. **Admin Dashboard**: Web-based management interface
12. **Retrieval Analytics**: Debug dashboard
13. **Performance Optimization**: Further optimizations
14. **Competition UI**: Professional industrial design
15. **Evaluation & Benchmarking**: RAG evaluation metrics

## Configuration Reference

### Environment Variables

```env
# Application
APP_NAME=FactoryMind AI Backend
APP_VERSION=2.0.0
ENVIRONMENT=development
LOG_LEVEL=INFO

# Vector Database
VECTOR_BACKEND=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key

# Embeddings
EMBEDDING_BACKEND=fastembed
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSION=384

# LLM Provider
LLM_PROVIDER=openai_compatible
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/

# Knowledge Graph
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Security
JWT_SECRET=your_jwt_secret_minimum_32_characters
JWT_ALGORITHM=HS256
```

## API Documentation

### Authentication Endpoints
- `POST /auth/login` - User login
- `POST /auth/register` - User registration

### Query Endpoints
- `POST /query` - Execute query with RAG
- `POST /debug/retrieve` - Debug retrieval without LLM

### Ingestion Endpoints
- `POST /ingest/{pipeline_name}` - Trigger ingestion pipeline
- `GET /ingest/status/{job_id}` - Get ingestion status

### Admin Endpoints
- `GET /admin/knowledge-base/stats` - Get vector store statistics
- `POST /admin/collection/delete` - Delete all collections
- `POST /admin/collection/recreate` - Recreate collections
- `GET /documents` - List indexed documents
- `GET /stats` - Get system statistics

### Machine Endpoints
- `GET /machines` - List available machines
- `GET /machines/{id}/graph` - Get knowledge graph path
- `GET /machines/{id}/history` - Get maintenance history

### Debug Endpoints
- `GET /debug/model` - Debug model configuration
- `GET /debug/groq` - Test Groq API
- `GET /debug/llm` - Test LLM provider

### Report Endpoints
- `GET /reports/{query_id}/pdf` - Download PDF report

### Prediction Endpoints
- `POST /predict` - Predict machine failure

## Development Guidelines

### Code Style
- Follow PEP 8 guidelines
- Use type hints for all functions
- Maximum line length: 100 characters
- Use `ruff` for linting
- Use `mypy` for type checking

### Testing
- Unit tests for business logic
- Integration tests for API endpoints
- Mock external dependencies (LLM, vector store)
- Test coverage target: 80%

### Documentation
- Docstrings for all public functions
- Type hints for function signatures
- Inline comments for complex logic
- Architecture diagrams for major components

## Troubleshooting

### Common Issues

1. **Vector Store Connection Failed**
   - Check Qdrant URL and API key
   - Verify network connectivity
   - Check Qdrant service status

2. **LLM Provider Errors**
   - Verify API key is set
   - Check API rate limits
   - Test API connectivity via debug endpoints

3. **Ingestion Failures**
   - Check file permissions
   - Verify document format support
   - Review ingestion logs for specific errors

4. **Memory Issues**
   - Reduce batch size for ingestion
- Use in-memory vector store for development
- Monitor system resources

## Changelog

### Version 2.0.0 (Current)
- **Refactoring**: Extracted monolithic main.py into modular structure
- **Configuration**: Added comprehensive validation and type hints
- **Logging**: Implemented structured logging with context
- **Error Handling**: Custom exception hierarchy with HTTP mapping
- **Code Quality**: Added linting configuration (ruff, mypy, black)
- **Constants**: Centralized hardcoded values
- **Documentation**: Comprehensive architecture documentation

### Version 1.0.0 (Previous)
- Initial release with basic RAG functionality
- Multi-agent orchestration with LangGraph
- Document ingestion pipeline
- Predictive maintenance with XGBoost
- Knowledge graph integration with Neo4j
