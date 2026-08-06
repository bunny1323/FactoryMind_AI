FactoryMind AI 🚜🤖
An Explainable Multimodal Hybrid RAG System for Intelligent Industrial Maintenance
<p align="center"> <b>AI-Powered Industrial Copilot for Heavy Equipment Maintenance</b><br> Retrieval-Augmented Generation (RAG) • Hybrid Search • Multimodal AI • Explainable Responses </p>
📌 Overview

FactoryMind AI is an enterprise-grade Multimodal Hybrid Retrieval-Augmented Generation (RAG) system designed to transform static industrial manuals into an intelligent maintenance assistant.

Instead of forcing technicians to manually search through thousands of pages of documentation, FactoryMind AI retrieves the most relevant technical information, understands maintenance queries, and generates grounded, explainable answers with citations and visual schematics.

The system is specifically built for Hyundai R215L Excavator manuals, but its modular architecture allows it to support any industrial domain by simply ingesting new documentation.

🎯 Problem Statement

Industrial technicians often spend significant time searching through maintenance manuals to identify:

Hydraulic troubleshooting procedures
Electrical wiring diagrams
Torque specifications
Error codes
Spare parts information
Maintenance procedures

This results in:

Increased machine downtime
Longer repair cycles
Human errors
Knowledge dependency on experienced technicians
Poor accessibility of critical information
💡 Our Solution

FactoryMind AI converts unstructured industrial documentation into a searchable knowledge base using Hybrid RAG.

The platform enables users to:

💬 Ask natural language questions
🔍 Retrieve the most relevant manual sections
🖼 View corresponding diagrams and schematics
📄 Receive grounded answers with citations
📚 Explore multiple manuals simultaneously
🧠 Understand relationships between components through a lightweight knowledge graph
✨ Key Features
📖 Multimodal Document Intelligence
🔍 Hybrid Dense + Sparse Retrieval
🏆 Cross-Encoder Re-ranking
🖼 Automatic Diagram & Figure Retrieval
📑 Citation-Based Responses
🧠 Knowledge Graph Integration
⚡ High-Speed Groq LLM Inference
🔐 Firebase Authentication
📊 Admin Dashboard
📂 Manual Management
📈 Retrieval Analytics
🚀 Production Ready Architecture
🏗 System Architecture
Industrial Manuals
        │
        ▼
PDF Processing & Image Extraction
        │
        ▼
Chunking + Metadata Generation
        │
        ▼
Embedding Generation
(Dense + Sparse)
        │
        ▼
Qdrant Vector Database
        │
        ▼
─────────────────────────────────
User Query
        │
Intent Detection
        │
Query Planning
        │
Hybrid Retrieval
(Dense + Sparse)
        │
Cross Encoder Reranking
        │
Context Construction
        │
Groq LLM
        │
Final Explainable Answer
        │
Retrieved Images
        │
Page Citations
🛠 Tech Stack
Frontend
Next.js 15
React
TypeScript
Tailwind CSS
ShadCN UI
Lucide Icons
Backend
FastAPI
Python
Async Architecture
REST APIs
Pydantic
Authentication
Firebase Authentication
AI & Machine Learning
Hybrid Retrieval-Augmented Generation (RAG)
FastEmbed
Cross Encoder Re-ranking
Query Planning
Semantic Search
Metadata Filtering
Large Language Model

Primary

Groq
Llama-3.3-70B-Versatile

Fallback

Ollama
Qwen2.5
Vector Database
Qdrant Cloud

Supports:

Dense Embeddings
Sparse Embeddings
Hybrid Search
Metadata Filtering
Reciprocal Rank Fusion (RRF)
Embedding Models
Dense Embedding
BAAI/bge-small-en-v1.5 (FastEmbed)
Sparse Embedding
BM25 Sparse Embedding (FastEmbed)
Reranker
BAAI/bge-reranker-base

Used to improve retrieval precision before answer generation.

PDF Processing
PyMuPDF (fitz)

Used for:

PDF Parsing
Text Extraction
Table Extraction
Image Extraction
Vector Graphic Extraction
OCR
PaddleOCR

Used when manuals contain scanned pages.

Image Processing
PyMuPDF Raster Extraction
Vector Diagram Rendering
Knowledge Graph

Lightweight JSON-based graph storing relationships between:

Components
Systems
Error Codes
Manuals
Procedures
Storage
MongoDB
Qdrant Cloud
Local Image Repository
📂 Supported Documents
Maintenance Manuals
Hydraulic Manuals
Electrical Manuals
Mechatronics Manuals
Structure & Function Manuals
Spare Parts Manuals
SOPs
Maintenance Logs
Error Code Manuals
🔍 Retrieval Pipeline
User Question
      │
      ▼
Intent Detection
      │
      ▼
Query Rewriting
      │
      ▼
Dense Search
      │
Sparse Search
      │
      ▼
Reciprocal Rank Fusion
      │
      ▼
Cross Encoder Reranker
      │
      ▼
Context Builder
      │
      ▼
Groq LLM
      │
      ▼
Answer + Citations + Images
🖼 Multimodal Capabilities

FactoryMind AI not only understands text but also retrieves:

Hydraulic Schematics
Electrical Wiring Diagrams
Mechanical Illustrations
Figures
Tables
Component Images
Assembly Diagrams

Each visual is linked to:

Document Name
Page Number
Figure Caption
Confidence Score
📊 Example Queries
Maintenance
What is the torque specification for the swing motor bolts?
Troubleshooting
Why is the hydraulic pressure low?
Visual Retrieval
Show the hydraulic circuit diagram.
Figure Retrieval
Show Figure 3-12.
Electrical
Display the electrical wiring diagram.
Maintenance Procedure
How do I replace the boom cylinder seal?
Error Code
What causes error code E-102?
Component Search
Explain the function of the pilot valve.
🚀 Future Roadmap
Phase 1 ✅

Hybrid Multimodal RAG

Completed
Phase 2

IoT Integration

Live sensor streaming
MQTT
OPC-UA
Predictive analytics
Digital Twin
Phase 3

Knowledge Graph Expansion

Neo4j Integration
Root Cause Analysis
Multi-hop Reasoning
Fault Propagation
Phase 4

Predictive Maintenance

Remaining Useful Life (RUL)
Failure Prediction
Maintenance Scheduling
AI Recommendations
🌍 SDG Alignment

FactoryMind AI directly contributes to:

SDG 9

Industry, Innovation and Infrastructure

By enabling:

Smart Manufacturing
AI-driven Maintenance
Reduced Downtime
Knowledge Digitization
Industrial Automation
📈 Impact
⏱ Faster information retrieval
📉 Reduced maintenance downtime
📚 Easy access to complex manuals
🔍 Explainable AI with citations
🛠 Improved technician productivity
🌍 Scalable across multiple industries
📷 Project Demo

Users can:

Upload industrial manuals
Search using natural language
Retrieve diagrams automatically
View cited manual pages
Explore visual schematics
Receive explainable AI responses

🤝 Contributors

Kenche Srikar
Maddimadugu Raju
Mohammad Anas

AI & Full Stack Developer

RAG Pipeline
Hybrid Retrieval
Multimodal Search
Backend Development
Frontend Development
Vector Search
AI Integration
⭐ Why FactoryMind AI?

Unlike conventional chatbots that generate generic responses, FactoryMind AI delivers grounded, explainable, and evidence-based answers by combining Hybrid Retrieval, Cross-Encoder Re-ranking, Visual Document Intelligence, and Large Language Models. Every response is backed by citations and relevant diagrams, making it a trustworthy AI assistant for industrial maintenance and technical support.

⭐ Support

If you found this project useful, consider giving it a ⭐ Star on GitHub to support the project and future development.
