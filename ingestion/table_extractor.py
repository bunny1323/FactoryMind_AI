"""
Table Extractor for FactoryMind AI Ingestion Pipeline.
Extracts tables as Markdown, CSV, and HTML formats.
Preserves metadata: table_id, caption, paragraph, section, page.
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("factorymind")


@dataclass
class TableMetadata:
    """Metadata for extracted tables."""
    id: str
    page: int
    table_id: str
    caption: Optional[str] = None
    section: Optional[str] = None
    heading: Optional[str] = None
    markdown: Optional[str] = None
    csv: Optional[str] = None
    html: Optional[str] = None
    rows: int = 0
    cols: int = 0


class TableExtractor:
    """
    Extracts tables from PDF pages with multiple format outputs.
    Supports pdfplumber and Camelot with graceful fallbacks.
    """
    
    def __init__(self):
        self.dep_manager = None
        self._check_dependencies()
        self.stats = {
            "total_tables": 0,
            "pdfplumber": 0,
            "camelot": 0,
            "failed": 0
        }
    
    def _check_dependencies(self):
        """Check available table extraction dependencies."""
        from .dependency_manager import get_dependency_manager
        self.dep_manager = get_dependency_manager()
    
    def extract_from_page(self, page, page_num: int, filename: str) -> List[TableMetadata]:
        """
        Extract tables from a PDF page.
        Returns list of TableMetadata objects.
        """
        tables = []
        
        # Try pdfplumber first
        if self.dep_manager.is_available("pdfplumber"):
            tables.extend(self._extract_with_pdfplumber(page, page_num, filename))
        
        # Try Camelot as fallback if no tables found
        if not tables and self.dep_manager.is_available("camelot"):
            tables.extend(self._extract_with_camelot(page, page_num, filename))
        
        return tables
    
    def _extract_with_pdfplumber(self, page, page_num: int, filename: str) -> List[TableMetadata]:
        """Extract tables using pdfplumber."""
        tables = []
        
        try:
            import pdfplumber
            import fitz
            
            # Convert fitz page to pdfplumber
            pdf_bytes = page.parent.tobytes()
            import io
            pdf_file = io.BytesIO(pdf_bytes)
            
            with pdfplumber.open(pdf_file) as pdf:
                plumb_page = pdf.pages[page_num - 1]
                extracted_tables = plumb_page.extract_tables()
                
                for idx, table in enumerate(extracted_tables):
                    if not table or not any(row for row in table):
                        continue
                    
                    # Generate metadata
                    table_id = f"table_{filename}_p{page_num}_{idx}"
                    markdown = self._table_to_markdown(table)
                    csv = self._table_to_csv(table)
                    html = self._table_to_html(table)
                    
                    metadata = TableMetadata(
                        id=f"{filename}_p{page_num}_tbl_{idx}",
                        page=page_num,
                        table_id=table_id,
                        markdown=markdown,
                        csv=csv,
                        html=html,
                        rows=len(table),
                        cols=len(table[0]) if table else 0
                    )
                    
                    tables.append(metadata)
                    self.stats["pdfplumber"] += 1
                    self.stats["total_tables"] += 1
                    
                    logger.debug(f"Extracted table {idx} from page {page_num} using pdfplumber")
        
        except Exception as e:
            logger.warning(f"pdfplumber table extraction failed on page {page_num}: {e}")
            self.stats["failed"] += 1
        
        return tables
    
    def _extract_with_camelot(self, page, page_num: int, filename: str) -> List[TableMetadata]:
        """Extract tables using Camelot."""
        tables = []
        
        try:
            import camelot
            import fitz
            
            # Camelot works with file paths
            pdf_bytes = page.parent.tobytes()
            import tempfile
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            
            try:
                # Try lattice mode first
                extracted_tables = camelot.read_pdf(tmp_path, pages=str(page_num), flavor="lattice")
                
                if extracted_tables.n == 0:
                    # Try stream mode as fallback
                    extracted_tables = camelot.read_pdf(tmp_path, pages=str(page_num), flavor="stream")
                
                for idx, table in enumerate(extracted_tables):
                    if table.df.empty:
                        continue
                    
                    # Generate metadata
                    table_id = f"table_{filename}_p{page_num}_{idx}"
                    markdown = self._dataframe_to_markdown(table.df)
                    csv = table.df.to_csv(index=False)
                    html = table.df.to_html(index=False)
                    
                    metadata = TableMetadata(
                        id=f"{filename}_p{page_num}_tbl_{idx}",
                        page=page_num,
                        table_id=table_id,
                        markdown=markdown,
                        csv=csv,
                        html=html,
                        rows=len(table.df),
                        cols=len(table.df.columns)
                    )
                    
                    tables.append(metadata)
                    self.stats["camelot"] += 1
                    self.stats["total_tables"] += 1
                    
                    logger.debug(f"Extracted table {idx} from page {page_num} using Camelot")
            
            finally:
                import os
                os.unlink(tmp_path)
        
        except Exception as e:
            logger.warning(f"Camelot table extraction failed on page {page_num}: {e}")
            self.stats["failed"] += 1
        
        return tables
    
    def _table_to_markdown(self, table: List[List[Any]]) -> str:
        """Convert table to Markdown format."""
        if not table:
            return ""
        
        # Clean cells
        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell or "").strip().replace("\n", " ") for cell in row]
            cleaned_table.append(cleaned_row)
        
        if not cleaned_table:
            return ""
        
        # Generate Markdown
        md_rows = []
        for row in cleaned_table:
            md_rows.append("| " + " | ".join(row) + " |")
        
        # Add separator after header
        if len(cleaned_table) > 0:
            cols_count = len(cleaned_table[0])
            separator = "| " + " | ".join(["---"] * cols_count) + " |"
            md_rows.insert(1, separator)
        
        return "\n".join(md_rows)
    
    def _table_to_csv(self, table: List[List[Any]]) -> str:
        """Convert table to CSV format."""
        if not table:
            return ""
        
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        for row in table:
            cleaned_row = [str(cell or "").strip().replace("\n", " ") for cell in row]
            writer.writerow(cleaned_row)
        
        return output.getvalue()
    
    def _table_to_html(self, table: List[List[Any]]) -> str:
        """Convert table to HTML format."""
        if not table:
            return ""
        
        html = ["<table>"]
        
        for i, row in enumerate(table):
            tag = "th" if i == 0 else "td"
            html.append("  <tr>")
            for cell in row:
                html.append(f"    <{tag}>{str(cell or '')}</{tag}>")
            html.append("  </tr>")
        
        html.append("</table>")
        return "\n".join(html)
    
    def _dataframe_to_markdown(self, df) -> str:
        """Convert pandas DataFrame to Markdown."""
        table = [df.columns.tolist()] + df.values.tolist()
        return self._table_to_markdown(table)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return self.stats.copy()
    
    def print_statistics(self):
        """Print extraction statistics."""
        print("\n" + "="*70)
        print("  TABLE EXTRACTION STATISTICS")
        print("="*70)
        print(f"  Total tables:       {self.stats['total_tables']}")
        print(f"  pdfplumber:         {self.stats['pdfplumber']}")
        print(f"  Camelot:            {self.stats['camelot']}")
        print(f"  Failed:             {self.stats['failed']}")
        print("="*70 + "\n")
