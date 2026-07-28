"""Report generation routes."""
from __future__ import annotations

import io
import logging
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from typing import Any, Dict
from backend.routes.query import get_last_answer

logger = logging.getLogger("factorymind")

router = APIRouter()


@router.get("/reports/{query_id}/pdf")
async def download_report(query_id: str):
    """Generate and download PDF report for a query."""
    # Fetch report data
    report_data = get_last_answer(query_id)
    if not report_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Maintenance report not found or expired."
        )

    # Generate styled PDF report using pymupdf (fitz)
    import fitz
    
    doc = fitz.open()
    page = doc.new_page()
    
    # Page Border
    page.draw_rect((20, 20, 575, 820), color=(0.4, 0.2, 0.1), width=1)
    
    # Header Banner
    page.draw_rect((30, 30, 565, 75), color=(0.95, 0.92, 0.88), fill=(0.95, 0.92, 0.88))
    page.insert_text((45, 58), "FACTORYMIND AI - EXPLAINABLE INDUSTRIAL COPILOT", fontsize=14, fontname="helv-bold", color=(0.4, 0.2, 0.1))
    
    # Title
    page.insert_text((45, 110), "MAINTENANCE DISPATCH & PROCEDURAL REPORT", fontsize=11, fontname="helv-bold", color=(0.2, 0.2, 0.2))
    page.draw_line((45, 115), (550, 115), color=(0.8, 0.8, 0.8), width=0.5)
    
    # Metadata block
    y = 135
    page.insert_text((45, y), f"Report UUID: {query_id}", fontsize=8, fontname="helv", color=(0.5, 0.5, 0.5))
    y += 12
    page.insert_text((45, y), f"Timestamp: {report_data.get('timestamp')}", fontsize=8, fontname="helv", color=(0.5, 0.5, 0.5))
    y += 12
    page.insert_text((45, y), f"Target Machine: Hyundai R215L Excavator ({report_data.get('machine_id')})", fontsize=8, fontname="helv", color=(0.5, 0.5, 0.5))
    y += 12
    page.insert_text((45, y), f"Engineer Query: \"{report_data.get('query')}\"", fontsize=8, fontname="helv-oblique", color=(0.3, 0.3, 0.3))
    y += 20
    page.draw_line((45, y), (550, y), color=(0.4, 0.2, 0.1), width=1)
    y += 20
    
    # Render answer text sections
    answer = report_data.get("answer", "")
    lines = answer.split("\n")
    
    for line in lines:
        if y > 780:
            page = doc.new_page()
            # New page border
            page.draw_rect((20, 20, 575, 820), color=(0.4, 0.2, 0.1), width=1)
            y = 50
        
        # Format headers and text lines
        if line.strip().startswith("###"):
            cleaned = line.replace("###", "").strip()
            # Draw section header with color
            page.insert_text((45, y), cleaned, fontsize=10, fontname="helv-bold", color=(0.4, 0.2, 0.1))
            y += 15
        elif line.strip().startswith("##"):
            cleaned = line.replace("##", "").strip()
            page.insert_text((45, y), cleaned, fontsize=11, fontname="helv-bold", color=(0.4, 0.2, 0.1))
            y += 18
        elif line.strip():
            # Paragraph formatting with simple word wrapping
            text = line.strip()
            words = text.split()
            chunk = []
            for word in words:
                chunk.append(word)
                if len(" ".join(chunk)) > 85:
                    page.insert_text((45, y), " ".join(chunk[:-1]), fontsize=8.5, fontname="helv", color=(0.2, 0.2, 0.2))
                    y += 13
                    if y > 780:
                        page = doc.new_page()
                        page.draw_rect((20, 20, 575, 820), color=(0.4, 0.2, 0.1), width=1)
                        y = 50
                    chunk = [word]
            if chunk:
                page.insert_text((45, y), " ".join(chunk), fontsize=8.5, fontname="helv", color=(0.2, 0.2, 0.2))
                y += 13
        else:
            y += 8
            
    # Sign-off footer on the last page
    if y > 720:
        page = doc.new_page()
        page.draw_rect((20, 20, 575, 820), color=(0.4, 0.2, 0.1), width=1)
        y = 50
    y += 20
    page.draw_line((45, y), (550, y), color=(0.8, 0.8, 0.8), width=0.5)
    y += 15
    page.insert_text((45, y), "Report Generated Dynamically by FactoryMind AI Industrial Agent Core.", fontsize=7.5, fontname="helv-oblique", color=(0.6, 0.6, 0.6))
    
    pdf_bytes = doc.write()
    doc.close()
    
    buf = io.BytesIO(pdf_bytes)
    buf.seek(0)
    
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=FactoryMind_Report_{query_id}.pdf"}
    )
