"""
Smart Chunker for FactoryMind AI Ingestion Pipeline.
Section-aware, figure-aware, table-aware, heading-aware chunking.
Never splits tables, captions, diagrams, warning boxes, or steps.
"""
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("factorymind")


@dataclass
class Chunk:
    """A text chunk with metadata."""
    id: str
    text: str
    chunk_type: str  # text, heading, table, image, warning, steps
    page: int
    heading: str
    section: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SmartChunker:
    """
    Intelligent chunking that respects document structure.
    Preserves context and relationships between chunks.
    """
    
    # Patterns for special content that should not be split
    WARNING_PATTERNS = [
        r"(?i)(warning|caution|danger|attention)",
        r"(?i)(⚠️|⚡|🔥|❗)",
    ]
    
    STEP_PATTERNS = [
        r"(?i)(step\s+\d+|procedure|instruction)",
        r"^\d+\.",
        r"^[a-z]\)",
    ]
    
    def __init__(self, max_chunk_size: int = 1200, overlap: int = 150):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.stats = {
            "total_chunks": 0,
            "by_type": {},
            "avg_chunk_size": 0,
            "split_chunks": 0
        }
    
    def chunk_elements(self, elements: List[Any], filename: str) -> List[Chunk]:
        """
        Chunk document elements intelligently.
        Elements can be text, headings, tables, images, etc.
        """
        chunks = []
        current_chunk = []
        current_size = 0
        current_type = "text"
        current_heading = "General Overview"
        current_section = "Main"
        
        for idx, elem in enumerate(elements):
            elem_type = getattr(elem, "type", "text")
            elem_text = getattr(elem, "text", "")
            elem_heading = getattr(elem, "heading", current_heading)
            elem_section = getattr(elem, "section", current_section)
            
            # Update heading/section context
            if elem_type == "heading":
                current_heading = elem_heading
                current_section = elem_section
                # Headings always start new chunks
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, filename, len(chunks),
                        current_type, current_heading, current_section
                    ))
                    current_chunk = []
                    current_size = 0
                
                # Add heading as its own chunk
                chunks.append(Chunk(
                    id=f"{filename}_chunk_{len(chunks)}",
                    text=elem_text,
                    chunk_type="heading",
                    page=getattr(elem, "page", 0),
                    heading=current_heading,
                    section=current_section
                ))
                self.stats["by_type"]["heading"] = self.stats["by_type"].get("heading", 0) + 1
                continue
            
            # Check if element should not be split
            if elem_type in ["table", "image", "warning", "steps"]:
                # These are atomic - always their own chunk
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, filename, len(chunks),
                        current_type, current_heading, current_section
                    ))
                    current_chunk = []
                    current_size = 0
                
                chunks.append(Chunk(
                    id=f"{filename}_chunk_{len(chunks)}",
                    text=elem_text,
                    chunk_type=elem_type,
                    page=getattr(elem, "page", 0),
                    heading=current_heading,
                    section=current_section,
                    metadata={
                        "image_path": getattr(elem, "image_path", None),
                        "caption": getattr(elem, "caption", None),
                        "table_id": getattr(elem, "table_id", None)
                    }
                ))
                self.stats["by_type"][elem_type] = self.stats["by_type"].get(elem_type, 0) + 1
                continue
            
            # Regular text - check if we should split
            elem_size = len(elem_text)
            
            # If adding this element would exceed max size, finalize current chunk
            if current_size + elem_size > self.max_chunk_size and current_chunk:
                chunks.append(self._create_chunk(
                    current_chunk, filename, len(chunks),
                    current_type, current_heading, current_section
                ))
                current_chunk = []
                current_size = 0
                self.stats["split_chunks"] += 1
            
            # Add element to current chunk
            current_chunk.append(elem)
            current_size += elem_size
            current_type = elem_type
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk, filename, len(chunks),
                current_type, current_heading, current_section
            ))
        
        # Add relationship metadata
        self._add_relationships(chunks)
        
        # Update statistics
        self.stats["total_chunks"] = len(chunks)
        if chunks:
            total_size = sum(len(chunk.text) for chunk in chunks)
            self.stats["avg_chunk_size"] = total_size // len(chunks)
        
        return chunks
    
    def _create_chunk(self, elements: List[Any], filename: str, chunk_idx: int,
                     chunk_type: str, heading: str, section: str) -> Chunk:
        """Create a chunk from a list of elements."""
        # Combine text from elements
        text_parts = []
        for elem in elements:
            elem_text = getattr(elem, "text", "")
            if elem_text:
                text_parts.append(elem_text)
        
        combined_text = "\n\n".join(text_parts)
        
        # Determine page (use first element's page)
        page = getattr(elements[0], "page", 0) if elements else 0
        
        return Chunk(
            id=f"{filename}_chunk_{chunk_idx}",
            text=combined_text,
            chunk_type=chunk_type,
            page=page,
            heading=heading,
            section=section
        )
    
    def _add_relationships(self, chunks: List[Chunk]):
        """Add parent/child/sibling relationships to chunks."""
        for idx, chunk in enumerate(chunks):
            # Find parent (previous heading)
            for prev_idx in range(idx - 1, -1, -1):
                if chunks[prev_idx].chunk_type == "heading":
                    chunk.metadata = chunk.metadata or {}
                    chunk.metadata["parent_chunk"] = chunks[prev_idx].id
                    break
            
            # Add siblings
            if idx > 0:
                chunk.metadata = chunk.metadata or {}
                chunk.metadata["previous_chunk"] = chunks[idx - 1].id
            
            if idx < len(chunks) - 1:
                chunk.metadata = chunk.metadata or {}
                chunk.metadata["next_chunk"] = chunks[idx + 1].id
    
    def detect_special_content(self, text: str) -> Optional[str]:
        """Detect if text is special content that should not be split."""
        for pattern in self.WARNING_PATTERNS:
            if re.search(pattern, text):
                return "warning"
        
        for pattern in self.STEP_PATTERNS:
            if re.search(pattern, text):
                return "steps"
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get chunking statistics."""
        return self.stats.copy()
    
    def print_statistics(self):
        """Print chunking statistics."""
        print("\n" + "="*70)
        print("  CHUNKING STATISTICS")
        print("="*70)
        print(f"  Total chunks:       {self.stats['total_chunks']}")
        print(f"  Average size:      {self.stats['avg_chunk_size']} chars")
        print(f"  Split chunks:      {self.stats['split_chunks']}")
        
        if self.stats['by_type']:
            print("\n  By type:")
            for chunk_type, count in self.stats['by_type'].items():
                print(f"    {chunk_type:15} {count}")
        
        print("="*70 + "\n")
