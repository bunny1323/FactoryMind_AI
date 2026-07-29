"""
Ingestion Report Generator for FactoryMind AI.
Generates comprehensive reports after indexing.
"""
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger("factorymind")


@dataclass
class ManualReport:
    """Report for a single manual."""
    filename: str
    pages_total: int
    pages_native: int
    pages_ocr: int
    images_extracted: int
    tables_extracted: int
    chunks_created: int
    avg_chunk_size: int
    skipped_pages: List[int]
    failed_pages: List[int]
    elapsed_time: float
    success_rate: float


@dataclass
class IngestionReport:
    """Comprehensive ingestion report."""
    timestamp: str
    total_manuals: int
    total_pages: int
    total_chunks: int
    total_images: int
    total_tables: int
    ocr_engine_used: str
    table_extractor_used: str
    manuals: List[ManualReport]
    dependency_status: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def print_summary(self):
        """Print a human-readable summary."""
        print("\n" + "="*70)
        print("  INGESTION REPORT")
        print("="*70)
        print(f"  Timestamp:          {self.timestamp}")
        print(f"  Total Manuals:      {self.total_manuals}")
        print(f"  Total Pages:        {self.total_pages}")
        print(f"  Total Chunks:       {self.total_chunks}")
        print(f"  Total Images:       {self.total_images}")
        print(f"  Total Tables:       {self.total_tables}")
        print(f"  OCR Engine:         {self.ocr_engine_used}")
        print(f"  Table Extractor:    {self.table_extractor_used}")
        print("="*70)
        
        if self.manuals:
            print("\n  MANUAL DETAILS:")
            print("-" * 70)
            for manual in self.manuals:
                print(f"\n  {manual.filename}")
                print(f"    Pages:           {manual.pages_total} (native: {manual.pages_native}, OCR: {manual.pages_ocr})")
                print(f"    Images:          {manual.images_extracted}")
                print(f"    Tables:          {manual.tables_extracted}")
                print(f"    Chunks:          {manual.chunks_created}")
                print(f"    Avg Chunk Size:  {manual.avg_chunk_size} chars")
                print(f"    Success Rate:    {manual.success_rate:.1f}%")
                print(f"    Time:            {manual.elapsed_time:.2f}s")
                
                if manual.skipped_pages:
                    print(f"    Skipped Pages:    {manual.skipped_pages}")
                if manual.failed_pages:
                    print(f"    Failed Pages:     {manual.failed_pages}")
        
        print("\n" + "="*70 + "\n")


class ReportGenerator:
    """Generates ingestion reports."""
    
    def __init__(self):
        self.manual_reports: List[ManualReport] = []
        self.start_time = datetime.now()
        self.dependency_status = {}
    
    def start_manual(self, filename: str, total_pages: int):
        """Start tracking a manual."""
        return {
            "filename": filename,
            "total_pages": total_pages,
            "start_time": datetime.now(),
            "pages_native": 0,
            "pages_ocr": 0,
            "images_extracted": 0,
            "tables_extracted": 0,
            "chunks_created": 0,
            "skipped_pages": [],
            "failed_pages": []
        }
    
    def record_page(self, manual_data: Dict[str, Any], page_num: int, method: str):
        """Record a page processing result."""
        if method == "native":
            manual_data["pages_native"] += 1
        elif method == "ocr":
            manual_data["pages_ocr"] += 1
    
    def record_images(self, manual_data: Dict[str, Any], count: int):
        """Record image extraction."""
        manual_data["images_extracted"] += count
    
    def record_tables(self, manual_data: Dict[str, Any], count: int):
        """Record table extraction."""
        manual_data["tables_extracted"] += count
    
    def record_chunks(self, manual_data: Dict[str, Any], count: int, avg_size: int):
        """Record chunking results."""
        manual_data["chunks_created"] = count
        manual_data["avg_chunk_size"] = avg_size
    
    def record_skipped_page(self, manual_data: Dict[str, Any], page_num: int):
        """Record a skipped page."""
        manual_data["skipped_pages"].append(page_num)
    
    def record_failed_page(self, manual_data: Dict[str, Any], page_num: int):
        """Record a failed page."""
        manual_data["failed_pages"].append(page_num)
    
    def finish_manual(self, manual_data: Dict[str, Any]) -> ManualReport:
        """Finish tracking a manual and generate report."""
        elapsed = (datetime.now() - manual_data["start_time"]).total_seconds()
        total_processed = manual_data["pages_native"] + manual_data["pages_ocr"]
        success_rate = (total_processed / manual_data["total_pages"] * 100) if manual_data["total_pages"] > 0 else 0
        
        report = ManualReport(
            filename=manual_data["filename"],
            pages_total=manual_data["total_pages"],
            pages_native=manual_data["pages_native"],
            pages_ocr=manual_data["pages_ocr"],
            images_extracted=manual_data["images_extracted"],
            tables_extracted=manual_data["tables_extracted"],
            chunks_created=manual_data["chunks_created"],
            avg_chunk_size=manual_data["avg_chunk_size"],
            skipped_pages=manual_data["skipped_pages"],
            failed_pages=manual_data["failed_pages"],
            elapsed_time=elapsed,
            success_rate=success_rate
        )
        
        self.manual_reports.append(report)
        return report
    
    def set_dependency_status(self, status: Dict[str, Any]):
        """Set dependency status."""
        self.dependency_status = status
    
    def generate_report(self, ocr_engine: str, table_extractor: str) -> IngestionReport:
        """Generate final ingestion report."""
        total_pages = sum(m.pages_total for m in self.manual_reports)
        total_chunks = sum(m.chunks_created for m in self.manual_reports)
        total_images = sum(m.images_extracted for m in self.manual_reports)
        total_tables = sum(m.tables_extracted for m in self.manual_reports)
        
        return IngestionReport(
            timestamp=datetime.now().isoformat(),
            total_manuals=len(self.manual_reports),
            total_pages=total_pages,
            total_chunks=total_chunks,
            total_images=total_images,
            total_tables=total_tables,
            ocr_engine_used=ocr_engine,
            table_extractor_used=table_extractor,
            manuals=self.manual_reports,
            dependency_status=self.dependency_status
        )
    
    def save_report(self, report: IngestionReport, filepath: str = "ingestion_report.json"):
        """Save report to JSON file."""
        with open(filepath, "w") as f:
            f.write(report.to_json())
        logger.info(f"Ingestion report saved to {filepath}")
