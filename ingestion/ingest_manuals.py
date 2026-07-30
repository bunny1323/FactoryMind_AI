"""
Production-grade Manual Ingestion Pipeline for FactoryMind AI.
Uses modular OCR Manager, Image Extractor, Table Extractor, Smart Chunker.
With automatic dependency detection, graceful fallbacks, and multi-tenant isolation.
"""
import os
import sys
import json
import logging
import hashlib
from typing import Any, List, Dict
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vector_store import VectorStore
from backend.config import settings

# Import modular components
from ingestion.dependency_manager import get_dependency_manager
from ingestion.ocr_manager import get_ocr_manager, OCRResult
from ingestion.image_extractor import ImageExtractor, ImageMetadata
from ingestion.table_extractor import TableExtractor, TableMetadata
from ingestion.smart_chunker import SmartChunker, Chunk
from ingestion.parallel_processor import ParallelProcessor
from ingestion.ingestion_report import ReportGenerator

logger = logging.getLogger("factorymind")


@dataclass
class DocElement:
    """Document element with metadata."""
    id: str
    type: str  # "text", "heading", "table", "image", "warning", "steps"
    text: str
    page: int
    heading: str
    section: str
    image_path: str | None = None
    table_id: str | None = None
    image_id: str | None = None
    caption: str | None = None
    width: float | None = None
    height: float | None = None
    bbox: list[float] | None = None
    markdown: str | None = None
    csv: str | None = None
    html: str | None = None


def clean_text(text: str) -> str:
    """Clean text by normalizing whitespace."""
    import re
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_file_hash(file_path: str) -> str:
    """Generate MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def is_heading_block(text: str) -> bool:
    """Check if text block is a heading."""
    import re
    text_clean = text.strip()
    if len(text_clean) > 120:
        return False
    if re.match(r"^(\d+\.)+\d*\s+", text_clean):
        return True
    if text_clean.upper() == text_clean and len(text_clean) > 3:
        return True
    if any(text_clean.lower().startswith(p) for p in ["section ", "chapter ", "part ", "table ", "figure "]):
        return True
    return False


def load_indexing_state(state_file: str = "ingest_state.json") -> Dict[str, Any]:
    """Load ingestion state from file."""
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load state file: {e}")
    return {}


def save_indexing_state(state: Dict[str, Any], state_file: str = "ingest_state.json"):
    """Save ingestion state to file."""
    try:
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save state file: {e}")


def update_stats_json(manuals_dir: str, public_img_dir: str, tables_count: int):
    """Update global statistics JSON."""
    stats_file = os.path.join(manuals_dir, "ingest_stats.json")
    try:
        stats = {}
        if os.path.exists(stats_file):
            with open(stats_file, "r") as f:
                stats = json.load(f)
        
        stats["total_manuals"] = len([f for f in os.listdir(manuals_dir) if f.endswith(".pdf")])
        stats["total_images"] = len([f for f in os.listdir(public_img_dir) if f.endswith(".png")]) if os.path.exists(public_img_dir) else 0
        stats["total_tables"] = tables_count
        
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to update stats JSON: {e}")


def ingest_manuals(
    manuals_dir: str = "data/manuals",
    public_img_dir: str = "public/extracted_images",
    collection_name: str = "manuals",
    user_id: str = "default_user",
    state_file: str = "ingest_state.json",
    clear_state: bool = False
) -> int:
    """
    Main ingestion function using refactored modular components.
    """
    logger.info("="*70)
    logger.info("INITIALIZING INGESTION PIPELINE")
    logger.info("="*70)
    
    if clear_state and os.path.exists(state_file):
        try:
            os.remove(state_file)
            logger.info(f"Cleared existing state file: {state_file}")
        except Exception as e:
            logger.warning(f"Could not remove state file: {e}")

    dep_manager = get_dependency_manager()
    dep_manager.print_report()
    
    ocr_manager = get_ocr_manager()
    ocr_manager.print_status()
    
    image_extractor = ImageExtractor(output_dir=public_img_dir)
    table_extractor = TableExtractor()
    chunker = SmartChunker(max_chunk_size=1200, overlap=150)
    parallel_processor = ParallelProcessor(max_workers=4)
    report_generator = ReportGenerator()
    
    report_generator.set_dependency_status({
        name: info.installed for name, info in dep_manager._cache.items()
    })
    
    from rag.embeddings import build_embedder
    from rag.sparse_embeddings import build_sparse_embedder
    from rag.vector_store import InMemoryHybridVectorStore, QdrantHybridVectorStore
    
    embedder = build_embedder(settings.EMBEDDING_BACKEND, settings.EMBEDDING_MODEL, settings.EMBEDDING_DIMENSION)
    sparse_embedder = build_sparse_embedder(settings.SPARSE_EMBEDDING_BACKEND, settings.SPARSE_EMBEDDING_MODEL)
    
    if settings.VECTOR_BACKEND == "qdrant":
        vector_store = QdrantHybridVectorStore(
            embedder,
            sparse_embedder,
            settings.QDRANT_URL,
            settings.QDRANT_API_KEY,
            embedder.dimension
        )
    else:
        vector_store = InMemoryHybridVectorStore(embedder)
    
    state = load_indexing_state(state_file)
    os.makedirs(manuals_dir, exist_ok=True)
    os.makedirs(public_img_dir, exist_ok=True)
    
    total_upserted = 0
    global_tables_count = 0
    
    for filename in os.listdir(manuals_dir):
        file_path = os.path.join(manuals_dir, filename)
        if not os.path.isfile(file_path) or not filename.endswith(".pdf"):
            continue
        
        doc_hash = get_file_hash(file_path)
        logger.info(f"\nProcessing manual: {filename}")
        
        manual_state = state.setdefault(filename, {
            "doc_hash": doc_hash,
            "indexed_pages": [],
            "completed": False
        })
        
        if manual_state.get("completed") and manual_state.get("doc_hash") == doc_hash:
            logger.info(f"Skipping already ingested manual: {filename}")
            continue
        
        manual_report_data = report_generator.start_manual(filename, 0)
        
        import fitz
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            logger.error(f"Failed to open manual {filename}: {e}")
            continue
        
        total_pages = len(doc)
        manual_report_data["total_pages"] = total_pages
        
        file_elements: List[DocElement] = []
        current_heading = "General Overview"
        current_section = "Main"
        
        for page_idx in range(total_pages):
            page_num = page_idx + 1
            
            if page_num in manual_state.get("indexed_pages", []):
                logger.info(f"Skipping page {page_num} (already indexed)")
                continue
            
            page = doc[page_idx]
            ocr_result = ocr_manager.extract_from_page(page, page_num, min_text_length=50)
            
            if ocr_result.text:
                method = "native" if ocr_result.engine == "PyMuPDF" else "ocr"
                report_generator.record_page(manual_report_data, page_num, method)
                
                blocks = page.get_text("blocks")
                for block in blocks:
                    block_text = block[4].strip()
                    if not block_text:
                        continue
                    
                    if is_heading_block(block_text):
                        current_heading = clean_text(block_text)
                        if "section" in current_heading.lower():
                            current_section = current_heading
                        file_elements.append(DocElement(
                            id=f"{filename}_p{page_num}_heading_{len(file_elements)}",
                            type="heading",
                            text=current_heading,
                            page=page_num,
                            heading=current_heading,
                            section=current_section
                        ))
                    else:
                        cleaned_para = clean_text(block_text)
                        if len(cleaned_para) > 20:
                            file_elements.append(DocElement(
                                id=f"{filename}_p{page_num}_text_{len(file_elements)}",
                                type="text",
                                text=cleaned_para,
                                page=page_num,
                                heading=current_heading,
                                section=current_section
                            ))
            
            images = image_extractor.extract_from_page(page, page_num, filename)
            for img in images:
                file_elements.append(DocElement(
                    id=img.id,
                    type="image",
                    text=img.caption or f"Image on page {page_num}",
                    page=page_num,
                    heading=current_heading,
                    section=current_section,
                    image_path=img.image_path,
                    image_id=img.id,
                    caption=img.caption,
                    width=img.width,
                    height=img.height,
                    bbox=list(img.bbox) if img.bbox else None
                ))
            
            report_generator.record_images(manual_report_data, len(images))
            
            tables = table_extractor.extract_from_page(page, page_num, filename)
            for tbl in tables:
                file_elements.append(DocElement(
                    id=tbl.id,
                    type="table",
                    text=tbl.markdown or "",
                    page=page_num,
                    heading=current_heading,
                    section=current_section,
                    table_id=tbl.id,
                    caption=tbl.caption,
                    markdown=tbl.markdown,
                    csv=tbl.csv,
                    html=tbl.html
                ))
            
            report_generator.record_tables(manual_report_data, len(tables))
            manual_state["indexed_pages"].append(page_num)
            save_indexing_state(state, state_file)
        
        final_chunks: List[dict] = []
        for idx, elem in enumerate(file_elements):
            if elem.type == "heading":
                continue
            
            chunk_context = []
            if idx > 0 and file_elements[idx - 1].type == "text":
                chunk_context.append(f"[Previous Context]: {file_elements[idx - 1].text[:300]}")
            
            if elem.type == "image":
                chunk_context.append(f"[Image]: {elem.caption or 'Diagram'}")
                if elem.image_path:
                    chunk_context.append(f"[Image Path]: {elem.image_path}")
            elif elem.type == "table":
                chunk_context.append(f"[Table]: {elem.caption or 'Technical Specifications'}")
            else:
                chunk_context.append(f"[Text]: {elem.text}")
            
            if idx < len(file_elements) - 1 and file_elements[idx + 1].type == "text":
                chunk_context.append(f"[Next Context]: {file_elements[idx + 1].text[:300]}")
            
            combined_text = "\n\n".join(chunk_context)
            record_id = f"{filename}_chunk_{idx}"
            
            payload = {
                "document_name": filename,
                "chunk_index": idx,
                "collection": collection_name,
                "page": elem.page,
                "doc_hash": doc_hash,
                "heading": elem.heading,
                "section": elem.section,
                "chunk_type": elem.type,
                "caption": elem.caption,
                "image_path": elem.image_path,
                "table_id": elem.table_id,
                "image_id": elem.image_id,
                "machine_model": "Hyundai R215L Smart Plus",
                "user_id": user_id
            }
            
            final_chunks.append({
                "id": record_id,
                "title": f"{filename} - {elem.heading} (Page {elem.page})",
                "text": combined_text,
                "source_type": "manual" if collection_name == "manuals" else "sop",
                "payload": payload
            })
            
        avg_chunk_size = int(sum(len(c["text"]) for c in final_chunks) / max(1, len(final_chunks)))
        report_generator.record_chunks(manual_report_data, len(final_chunks), avg_chunk_size)
        
        upserted_count = 0
        if final_chunks:
            upserted_count = vector_store.upsert(collection_name, final_chunks)
            total_upserted += upserted_count
        
        manual_state["completed"] = True
        save_indexing_state(state, state_file)
        report_generator.finish_manual(manual_report_data)
        
        print(f"\n{'='*70}")
        print(f"INGESTION SUMMARY: {filename}")
        print(f"{'='*70}")
        print(f"Pages Parsed:    {total_pages}")
        print(f"Images Extracted: {manual_report_data['images_extracted']}")
        print(f"Tables Extracted: {manual_report_data['tables_extracted']}")
        print(f"Chunks Created:  {manual_report_data['chunks_created']}")
        print(f"Embedding Count: {upserted_count}")
        print(f"Success Rate:    {manual_report_data.get('success_rate', 0):.1f}%")
        print(f"{'='*70}\n")
        
        doc.close()
    
    update_stats_json(manuals_dir, public_img_dir, global_tables_count)
    
    image_extractor.print_statistics()
    table_extractor.print_statistics()
    chunker.print_statistics()
    parallel_processor.print_statistics()
    
    available_ocr = ocr_manager.get_available_backends()
    ocr_engine = available_ocr[0] if available_ocr else "None"
    
    available_tables = table_extractor.dep_manager.has_capability("table_extraction")
    table_extractor_name = available_tables[0] if available_tables else "None"
    
    final_report = report_generator.generate_report(ocr_engine, table_extractor_name)
    final_report.print_summary()
    report_generator.save_report(final_report)
    
    logger.info(f"Total chunks upserted: {total_upserted}")
    return total_upserted


def run_manuals_ingestion(vector_store: VectorStore, manuals_dir: str, collection_name: str = "manuals", user_id: str = "default_user") -> int:
    """
    Compatibility wrapper matching legacy codebase references.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    public_img_dir = os.path.join(project_root, "frontend", "public", "extracted_images")
    return ingest_manuals(
        manuals_dir=manuals_dir,
        public_img_dir=public_img_dir,
        collection_name=collection_name,
        user_id=user_id
    )
