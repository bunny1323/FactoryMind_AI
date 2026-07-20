from __future__ import annotations

import os
import re
import logging
import hashlib
from typing import Any
from rag.vector_store import VectorStore

logger = logging.getLogger("factorymind")

def clean_text(text: str) -> str:
    # Normalize spacing
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def get_file_hash(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def is_heading_block(text: str) -> bool:
    text_clean = text.strip()
    if len(text_clean) > 120:
        return False
    # Check if starts with digits indicating section e.g. 1.2, Section 3, Chapter, or is all caps
    if re.match(r"^(\d+\.)+\d*\s+", text_clean):
        return True
    if text_clean.upper() == text_clean and len(text_clean) > 3:
        return True
    if any(text_clean.lower().startswith(p) for p in ["section ", "chapter ", "part ", "table ", "figure "]):
        return True
    return False

def run_manuals_ingestion(vector_store: VectorStore, manuals_dir: str, collection_name: str = "manuals", user_id: str = "default_user") -> int:
    """Reads manuals/SOP files, applies layout-aware PyMuPDF block chunking, MD5 hash deduplication, and indexes to Qdrant."""
    if not os.path.exists(manuals_dir):
        logger.warning(f"Manuals directory {manuals_dir} does not exist.")
        return 0

    total_upserted = 0
    processed_chunk_hashes = set()

    for filename in os.listdir(manuals_dir):
        file_path = os.path.join(manuals_dir, filename)
        if not os.path.isfile(file_path):
            continue

        logger.info(f"Ingesting manuals file: {filename}")
        doc_hash = get_file_hash(file_path)
        file_records = []

        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            if not text.strip():
                logger.warning(f"No text extracted from {filename}")
                continue
            
            cleaned = clean_text(text)
            # TXT Fallback paragraph chunker
            paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [cleaned]

            for idx, para in enumerate(paragraphs):
                chunk_hash = hashlib.md5(para.encode("utf-8")).hexdigest()
                if chunk_hash in processed_chunk_hashes:
                    continue
                processed_chunk_hashes.add(chunk_hash)

                record_id = f"{filename}_chunk_{idx}"
                file_records.append({
                    "id": record_id,
                    "title": f"{filename} - Part {idx+1}",
                    "text": para,
                    "source_type": "manual" if collection_name == "manuals" else "sop",
                    "payload": {
                        "document_name": filename,
                        "chunk_index": idx,
                        "collection": collection_name,
                        "doc_hash": doc_hash,
                        "chunk_hash": chunk_hash,
                        "heading": "General Info",
                        "section": "General",
                        "machine_model": "Hyundai R215L Smart Plus",
                        "user_id": user_id
                    }
                })

        elif filename.endswith(".pdf"):
            try:
                import fitz
                doc = fitz.open(file_path)
                
                # Base public directory for saving extracted images
                public_img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "public", "extracted_images")
                os.makedirs(public_img_dir, exist_ok=True)
                
                pdf_chunks = []
                current_heading = "General Overview"
                current_section = "Main"
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    page_hash = hashlib.md5(page_text.encode("utf-8")).hexdigest()
                    
                    # 1. Image extraction (extract figures/diagrams)
                    image_path = None
                    try:
                        images = page.get_images(full=True)
                        if images:
                            xref = images[0][0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]
                            
                            img_filename = f"{filename}_page_{page_num + 1}_img_0.{image_ext}"
                            img_save_path = os.path.join(public_img_dir, img_filename)
                            
                            with open(img_save_path, "wb") as f_img:
                                f_img.write(image_bytes)
                                
                            image_path = f"/extracted_images/{img_filename}"
                    except Exception as img_err:
                        logger.debug(f"Could not extract image on {filename} page {page_num + 1}: {img_err}")
                    
                    # 2. Extract Tables (PyMuPDF TableFinder)
                    table_markdowns = []
                    try:
                        tables = page.find_tables()
                        for table in tables:
                            data = table.extract()
                            if data:
                                md_rows = []
                                for row in data:
                                    clean_row = [str(cell or "").strip().replace("\n", " ") for cell in row]
                                    md_rows.append("| " + " | ".join(clean_row) + " |")
                                if md_rows:
                                    cols_count = len(data[0])
                                    separator = "| " + " | ".join(["---"] * cols_count) + " |"
                                    md_rows.insert(1, separator)
                                    table_markdowns.append("\n".join(md_rows))
                    except Exception as tbl_err:
                        logger.debug(f"Table parsing failed on page {page_num+1}: {tbl_err}")

                    # 3. Block-Level Layout-Aware Parsing
                    blocks = page.get_text("blocks")
                    for block in blocks:
                        block_text = block[4].strip()
                        if not block_text:
                            continue
                        
                        if is_heading_block(block_text):
                            current_heading = clean_text(block_text)
                            if "section" in current_heading.lower():
                                current_section = current_heading
                            continue
                        
                        # Add normal paragraph block
                        cleaned_para = clean_text(block_text)
                        if len(cleaned_para) < 20: # skip footer page numbers/short artifacts
                            continue

                        pdf_chunks.append({
                            "text": cleaned_para,
                            "page": page_num + 1,
                            "heading": current_heading,
                            "section": current_section,
                            "image_path": image_path,
                            "page_hash": page_hash
                        })
                    
                    # Add parsed Tables as separate chunks
                    for table_md in table_markdowns:
                        pdf_chunks.append({
                            "text": f"Table specifications:\n{table_md}",
                            "page": page_num + 1,
                            "heading": f"Table: {current_heading}",
                            "section": current_section,
                            "image_path": image_path,
                            "page_hash": page_hash
                        })

                # Convert parsed chunks to DB records with MD5 deduplication
                for idx, item in enumerate(pdf_chunks):
                    chunk_text_data = item["text"]
                    chunk_hash = hashlib.md5(chunk_text_data.encode("utf-8")).hexdigest()
                    
                    if chunk_hash in processed_chunk_hashes:
                        continue
                    processed_chunk_hashes.add(chunk_hash)

                    record_id = f"{filename}_chunk_{idx}"
                    payload = {
                        "document_name": filename,
                        "chunk_index": idx,
                        "collection": collection_name,
                        "page": item["page"],
                        "doc_hash": doc_hash,
                        "page_hash": item["page_hash"],
                        "chunk_hash": chunk_hash,
                        "heading": item["heading"],
                        "section": item["section"],
                        "machine_model": "Hyundai R215L Smart Plus",
                        "user_id": user_id
                    }
                    if item["image_path"]:
                        payload["image_path"] = item["image_path"]

                    file_records.append({
                        "id": record_id,
                        "title": f"{filename} - {item['heading']} (Page {item['page']})",
                        "text": chunk_text_data,
                        "source_type": "manual" if collection_name == "manuals" else "sop",
                        "payload": payload
                    })
                    
            except Exception as pdf_err:
                logger.error(f"PyMuPDF failed to parse {filename}: {pdf_err}")
                continue
        else:
            continue

        if file_records:
            import math
            batches_count = math.ceil(len(file_records) / 100)
            logger.info(f"Ingesting {filename}: {len(file_records)} chunks in {batches_count} batches")
            upserted = vector_store.upsert(collection_name, file_records)
            total_upserted += upserted

    return total_upserted
