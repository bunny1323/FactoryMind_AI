from __future__ import annotations

import os
import re
import json
import logging
import hashlib
from typing import Any
from dataclasses import dataclass
from rag.vector_store import VectorStore

logger = logging.getLogger("factorymind")

@dataclass
class DocElement:
    id: str
    type: str  # "text", "heading", "table", "image"
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

def clean_text(text: str) -> str:
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
    if re.match(r"^(\d+\.)+\d*\s+", text_clean):
        return True
    if text_clean.upper() == text_clean and len(text_clean) > 3:
        return True
    if any(text_clean.lower().startswith(p) for p in ["section ", "chapter ", "part ", "table ", "figure "]):
        return True
    return False

def classify_page(page_text: str, images_count: int, tables_count: int) -> str:
    text_len = len(page_text.strip())
    if text_len < 100:
        if images_count > 0:
            return "Diagram"
        else:
            return "Scanned"
    elif tables_count > 0:
        if text_len > 1200:
            return "Mixed"
        else:
            return "Table"
    else:
        return "Normal"

def extract_tables_pdfplumber(pdf_path: str, page_num: int) -> list[str]:
    """Primary table extractor using pdfplumber."""
    import pdfplumber
    tables_md = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            plumb_page = pdf.pages[page_num - 1]
            tables = plumb_page.extract_tables()
            for table in tables:
                if table:
                    md_rows = []
                    for row in table:
                        clean_row = [str(cell or "").strip().replace("\n", " ") for cell in row]
                        md_rows.append("| " + " | ".join(clean_row) + " |")
                    if md_rows:
                        cols_count = len(table[0])
                        separator = "| " + " | ".join(["---"] * cols_count) + " |"
                        md_rows.insert(1, separator)
                        tables_md.append("\n".join(md_rows))
    except Exception as e:
        logger.warning(f"pdfplumber table extraction failed on page {page_num}: {e}", exc_info=True)
    return tables_md

def extract_tables_camelot(pdf_path: str, page_num: int) -> list[str]:
    """Fallback table extractor using Camelot."""
    tables_md = []
    try:
        import camelot
        tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor="lattice")
        if not tables:
            tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor="stream")
        for table in tables:
            df = table.df
            md = df.to_markdown(index=False, header=False)
            if md:
                tables_md.append(md)
    except ModuleNotFoundError:
        logger.debug(f"Camelot module is not installed, skipping fallback on page {page_num}.")
    except Exception as e:
        logger.warning(f"Camelot table extraction failed on page {page_num}: {e}")
    return tables_md

def run_ocr_on_page(doc: Any, page_num: int) -> str:
    """Multi-stage OCR fallback pipeline (PaddleOCR -> pytesseract -> PyMuPDF)."""
    try:
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(dpi=150)
        from PIL import Image
        import io
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
    except Exception as pix_err:
        logger.warning(f"Failed to generate page pixmap on page {page_num}: {pix_err}", exc_info=True)
        return ""

    # Stage 1: PaddleOCR
    try:
        from paddleocr import PaddleOCR
        import numpy as np
        ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        img_np = np.array(img)
        result = ocr_engine.ocr(img_np, cls=True)
        txts = []
        if result and result[0]:
            for line in result[0]:
                txts.append(line[1][0])
        if txts:
            logger.info(f"Page {page_num}: OCR Success using PaddleOCR.")
            return " ".join(txts)
    except Exception as ocr_err:
        logger.warning(f"PaddleOCR failed on page {page_num}: {ocr_err}", exc_info=True)

    # Stage 2: pytesseract
    try:
        import pytesseract
        tesseract_bin = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tesseract_bin):
            pytesseract.pytesseract.tesseract_cmd = tesseract_bin
        text = pytesseract.image_to_string(img)
        if text.strip():
            logger.info(f"Page {page_num}: OCR Success using pytesseract.")
            return text
    except Exception as ocr_err2:
        logger.warning(f"pytesseract failed on page {page_num}: {ocr_err2}")

    # Stage 3: PyMuPDF raw block layout text
    try:
        raw_text = page.get_text()
        if raw_text.strip():
            logger.warning(f"Page {page_num}: OCR libraries unavailable. Fell back to raw PyMuPDF text.")
            return raw_text
    except Exception as e:
        logger.warning(f"PyMuPDF fallback text fetch failed on page {page_num}: {e}", exc_info=True)

    return ""

def load_indexing_state() -> dict[str, Any]:
    state_file = "ingest_state.json"
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read ingest_state.json: {e}", exc_info=True)
    return {}

def save_indexing_state(state: dict[str, Any]):
    state_file = "ingest_state.json"
    try:
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write ingest_state.json: {e}", exc_info=True)

def update_stats_json(manuals_dir: str, public_img_dir: str, tables_count: int):
    """Saves computed page, table, and image counts to ingest_stats.json."""
    pages_count = 0
    doc_count = 0
    
    if os.path.exists(manuals_dir):
        import fitz
        for filename in os.listdir(manuals_dir):
            if filename.endswith(".pdf"):
                doc_count += 1
                try:
                    with fitz.open(os.path.join(manuals_dir, filename)) as doc:
                        pages_count += len(doc)
                except Exception as e:
                    logger.warning(f"Error reading page count for {filename}: {e}", exc_info=True)
            elif filename.endswith(".txt"):
                doc_count += 1
                pages_count += 1

    images_count = 0
    if os.path.exists(public_img_dir):
        try:
            images_count = len([f for f in os.listdir(public_img_dir) if os.path.isfile(os.path.join(public_img_dir, f))])
        except Exception as e:
            logger.warning(f"Failed to list public images dir: {e}", exc_info=True)

    stats_data = {
        "manuals_count": doc_count,
        "pages_count": pages_count,
        "images_count": images_count,
        "tables_count": tables_count
    }
    
    stats_file = os.path.join(os.path.dirname(manuals_dir), "ingest_stats.json")
    try:
        with open(stats_file, "w") as f:
            json.dump(stats_data, f, indent=2)
        logger.info(f"Updated ingest_stats.json: {stats_data}")
    except Exception as e:
        logger.warning(f"Failed to write ingest_stats.json: {e}", exc_info=True)

def run_manuals_ingestion(vector_store: VectorStore, manuals_dir: str, collection_name: str = "manuals", user_id: str = "default_user") -> int:
    """Robust multi-stage layout parser with parallel page processing flow, resuming, and full metric summaries."""
    if not os.path.exists(manuals_dir):
        logger.warning(f"Manuals directory {manuals_dir} does not exist.")
        return 0

    public_img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "public", "extracted_images")
    os.makedirs(public_img_dir, exist_ok=True)

    state = load_indexing_state()
    total_upserted = 0
    global_tables_count = 0

    # Image MD5 deduplication maps
    saved_image_hashes = set()
    saved_image_paths = {}

    for filename in os.listdir(manuals_dir):
        file_path = os.path.join(manuals_dir, filename)
        if not os.path.isfile(file_path):
            continue

        doc_hash = get_file_hash(file_path)
        logger.info(f"Processing manual: {filename}")

        manual_state = state.setdefault(filename, {
            "doc_hash": doc_hash,
            "indexed_pages": [],
            "completed": False
        })

        if manual_state.get("completed") and manual_state.get("doc_hash") == doc_hash:
            logger.info(f"Skipping already ingested manual: {filename}")
            continue

        import fitz
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            logger.error(f"Failed to open manual {filename} using PyMuPDF: {e}", exc_info=True)
            continue

        total_pages = len(doc)
        text_chunks_count = 0
        tables_count = 0
        ocr_pages_count = 0
        failed_pages = []
        
        raster_counter = 0
        vector_counter = 0
        w_dedup_count = 0
        
        file_elements: list[DocElement] = []
        current_heading = "General Overview"
        current_section = "Main"
        
        # Sequentially process each page
        for page_idx in range(total_pages):
            page_num = page_idx + 1
            
            if page_num in manual_state.get("indexed_pages", []):
                logger.info(f"Resuming: Page {page_num} of {filename} already indexed.")
                continue

            try:
                page = doc.load_page(page_idx)
                raw_text = page.get_text()
                
                # Check tables and drawings/images count to classify page
                try:
                    tables_found = page.find_tables()
                    page_tables_count = len(tables_found.tables) if tables_found and tables_found.tables else 0
                except Exception as e:
                    logger.warning(f"Failed to check tables on page {page_num}: {e}")
                    page_tables_count = 0

                try:
                    images_found = page.get_images(full=True)
                    page_images_count = len(images_found) if images_found else 0
                except Exception as e:
                    logger.warning(f"Failed to check images on page {page_num}: {e}", exc_info=True)
                    page_images_count = 0

                drawings = []
                try:
                    drawings = page.get_drawings()
                except Exception as e:
                    logger.warning(f"Failed to get drawings on page {page_num}: {e}", exc_info=True)

                # 1. Page Classifier
                classification = classify_page(raw_text, page_images_count or len(drawings), page_tables_count)
                
                # 2. Run OCR if scanned page
                page_content = raw_text
                if classification == "Scanned" or len(raw_text.strip()) < 100:
                    ocr_pages_count += 1
                    ocr_text = run_ocr_on_page(doc, page_num)
                    if ocr_text.strip():
                        page_content = ocr_text
                        classification = "Mixed" if page_images_count > 0 else "Normal"
                
                page_image_paths = []
                
                # 3. Extract Raster Images
                for img_info in images_found:
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        img_hash = hashlib.md5(image_bytes).hexdigest()
                        if img_hash in saved_image_hashes:
                            w_dedup_count += 1
                            image_path = saved_image_paths[img_hash]
                            page_image_paths.append(image_path)
                        else:
                            raster_counter += 1
                            img_filename = f"{filename}_page_{page_num}_raster_{raster_counter}.{image_ext}"
                            img_save_path = os.path.join(public_img_dir, img_filename)
                            with open(img_save_path, "wb") as f_img:
                                f_img.write(image_bytes)
                                
                            image_path = f"/extracted_images/{img_filename}"
                            saved_image_hashes.add(img_hash)
                            saved_image_paths[img_hash] = image_path
                            page_image_paths.append(image_path)
                    except Exception as img_err:
                        logger.warning(f"Failed to extract raster image on page {page_num}: {img_err}", exc_info=True)

                # 4. Extract Vector-drawn Diagrams
                if len(drawings) > 10 and len(images_found) == 0:
                    x0, y0, x1, y1 = float("inf"), float("inf"), float("-inf"), float("-inf")
                    for d in drawings:
                        rect = d.get("rect")
                        if rect:
                            x0 = min(x0, rect.x0)
                            y0 = min(y0, rect.y0)
                            x1 = max(x1, rect.x1)
                            y1 = max(y1, rect.y1)
                    if x1 > x0 and y1 > y0:
                        clip = fitz.Rect(x0, y0, x1, y1)
                        if clip.width > 50 and clip.height > 50:
                            try:
                                pix = page.get_pixmap(clip=clip, dpi=200)
                                image_bytes = pix.tobytes("png")
                                img_hash = hashlib.md5(image_bytes).hexdigest()
                                if img_hash in saved_image_hashes:
                                    w_dedup_count += 1
                                    image_path = saved_image_paths[img_hash]
                                    page_image_paths.append(image_path)
                                else:
                                    vector_counter += 1
                                    img_filename = f"{filename}_page_{page_num}_vector_{vector_counter}.png"
                                    img_save_path = os.path.join(public_img_dir, img_filename)
                                    pix.save(img_save_path)
                                    
                                    image_path = f"/extracted_images/{img_filename}"
                                    saved_image_hashes.add(img_hash)
                                    saved_image_paths[img_hash] = image_path
                                    page_image_paths.append(image_path)
                            except Exception as ras_err:
                                logger.warning(f"Failed to rasterize vector drawings on page {page_num}: {ras_err}", exc_info=True)

                page_elements = []

                # 5. Extract Tables (pdfplumber -> Camelot)
                tables_md = extract_tables_pdfplumber(file_path, page_num)
                if not tables_md:
                    tables_md = extract_tables_camelot(file_path, page_num)
                
                for tbl_idx, table_md in enumerate(tables_md):
                    tables_count += 1
                    page_elements.append(DocElement(
                        id=f"{filename}_p{page_num}_tbl_{tbl_idx}",
                        type="table",
                        text=table_md,
                        page=page_num,
                        heading=current_heading,
                        section=current_section,
                        table_id=f"table_{filename}_p{page_num}_{tables_count}",
                        caption=f"Manual Table Specs {tables_count}"
                    ))

                # 6. Extract Text Layout Blocks
                blocks = page.get_text("blocks")
                for block in blocks:
                    block_text = block[4].strip()
                    if not block_text:
                        continue
                    
                    if is_heading_block(block_text):
                        current_heading = clean_text(block_text)
                        if "section" in current_heading.lower():
                            current_section = current_heading
                        page_elements.append(DocElement(
                            id=f"{filename}_p{page_num}_heading_{len(page_elements)}",
                            type="heading",
                            text=current_heading,
                            page=page_num,
                            heading=current_heading,
                            section=current_section
                        ))
                    else:
                        cleaned_para = clean_text(block_text)
                        if len(cleaned_para) > 20:
                            page_elements.append(DocElement(
                                id=f"{filename}_p{page_num}_text_{len(page_elements)}",
                                type="text",
                                text=cleaned_para,
                                page=page_num,
                                heading=current_heading,
                                section=current_section
                            ))

                # 7. Attach images only to the chunks that actually reference them
                if page_image_paths:
                    img_ref_pattern = re.compile(r"(?i)(fig\.?\s*\d+|figure\s*\d+|diagram|illustration)")
                    attached = False
                    
                    for elem in page_elements:
                        if elem.type in ["text", "table"] and img_ref_pattern.search(elem.text):
                            elem.image_path = page_image_paths[0]
                            attached = True
                            
                    # Fallback: attach to first text/table chunk
                    if not attached:
                        for elem in page_elements:
                            if elem.type in ["text", "table"]:
                                elem.image_path = page_image_paths[0]
                                break

                # Add page elements to manual elements list
                file_elements.extend(page_elements)
                
                # Mark page as successfully processed
                manual_state["indexed_pages"].append(page_num)
                save_indexing_state(state)
                
            except Exception as page_err:
                logger.error(f"Error parsing page {page_num} of {filename}: {page_err}", exc_info=True)
                failed_pages.append(page_num)

        # Validation: Verify that we successfully extracted core layout details
        if len(file_elements) == 0 and total_pages > 0:
            logger.error(f"Extraction validation failed for {filename}. 0 elements extracted. Retrying with basic text chunker...")
            for page_idx in range(total_pages):
                page_num = page_idx + 1
                try:
                    raw_text = doc.load_page(page_idx).get_text()
                    if raw_text.strip():
                        file_elements.append(DocElement(
                            id=f"{filename}_p{page_num}_fallback",
                            type="text",
                            text=clean_text(raw_text),
                            page=page_num,
                            heading="General Specifications",
                            section="Fallback"
                        ))
                except Exception as e:
                    logger.warning(f"Fallback extraction failed on page {page_num}: {e}", exc_info=True)

        global_tables_count += tables_count

        # --- Structure-Aware Chunking & Relationship Graph Builder ---
        final_chunks: list[dict[str, Any]] = []
        for elem in file_elements:
            chunk_type = elem.type
            text_data = elem.text
            
            if chunk_type == "text" and len(text_data) > 1000:
                sub_paras = [p.strip() for p in re.split(r"(?<=\. )\s*", text_data) if p.strip()]
                current_sub = []
                current_len = 0
                for sp in sub_paras:
                    current_sub.append(sp)
                    current_len += len(sp)
                    if current_len >= 1200: # ~350-400 words
                        sub_text = " ".join(current_sub)
                        final_chunks.append({"element": elem, "text": sub_text})
                        current_sub = []
                        current_len = 0
                if current_sub:
                    sub_text = " ".join(current_sub)
                    final_chunks.append({"element": elem, "text": sub_text})
            else:
                final_chunks.append({"element": elem, "text": text_data})

        file_records = []
        for idx, chunk in enumerate(final_chunks):
            elem = chunk["element"]
            text_data = chunk["text"]
            chunk_hash = hashlib.md5(text_data.encode("utf-8")).hexdigest()
            
            record_id = f"{filename}_chunk_{idx}"
            text_chunks_count += 1

            # Build Parent section relationships
            parent_id = None
            for prev_idx in range(idx - 1, -1, -1):
                if final_chunks[prev_idx]["element"].type == "heading":
                    parent_id = f"{filename}_chunk_{prev_idx}"
                    break

            payload = {
                "document_name": filename,
                "chunk_index": idx,
                "collection": collection_name,
                "page": elem.page,
                "doc_hash": doc_hash,
                "chunk_hash": chunk_hash,
                "heading": elem.heading,
                "section": elem.section,
                "chunk_type": elem.type,
                "caption": elem.caption,
                "image_path": elem.image_path,
                "table_id": elem.table_id,
                "image_id": elem.image_id,
                "parent_chunk": parent_id,
                "previous_chunk": f"{filename}_chunk_{idx - 1}" if idx > 0 else None,
                "next_chunk": f"{filename}_chunk_{idx + 1}" if idx < len(final_chunks) - 1 else None,
                "machine_model": "Hyundai R215L Smart Plus",
                "user_id": user_id
            }

            file_records.append({
                "id": record_id,
                "title": f"{filename} - {elem.heading} (Page {elem.page})",
                "text": text_data,
                "source_type": "manual" if collection_name == "manuals" else "sop",
                "payload": payload
            })

        # Upsert chunks in batches
        upserted_count = 0
        if file_records:
            upserted_count = vector_store.upsert(collection_name, file_records)
            total_upserted += upserted_count

        # Mark manual as fully completed
        if len(failed_pages) == 0:
            manual_state["completed"] = True
            save_indexing_state(state)

        # Output Ingestion summary metrics
        logger.info(f"INGESTION SUMMARY: {filename} - {raster_counter + vector_counter} images extracted ({raster_counter} raster, {vector_counter} rasterized from vector graphics), {w_dedup_count} deduplicated")

        success_rate = round(((total_pages - len(failed_pages)) / total_pages) * 100, 2) if total_pages > 0 else 0
        print(f"\n==========================================")
        print(f"INGESTION SUMMARY: {filename}")
        print(f"==========================================")
        print(f"Manual Name:     {filename}")
        print(f"Pages Parsed:    {total_pages - len(failed_pages)}/{total_pages}")
        print(f"Text Chunks:     {text_chunks_count}")
        print(f"Tables Extracted:{tables_count}")
        print(f"Images Saved:    {raster_counter + vector_counter}")
        print(f"OCR Pages Run:   {ocr_pages_count}")
        print(f"Embedding Count: {upserted_count}")
        print(f"Index Status:    {'Completed' if manual_state['completed'] else 'Partial'}")
        print(f"Success Rate:    {success_rate}%")
        if failed_pages:
            print(f"Failed Pages:    {failed_pages}")
        print(f"==========================================\n")

    # Update global stats JSON
    update_stats_json(manuals_dir, public_img_dir, global_tables_count)

    return total_upserted
