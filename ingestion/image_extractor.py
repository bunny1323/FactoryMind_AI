"""
Image Extractor for FactoryMind AI Ingestion Pipeline.
Extracts figures, tables, diagrams, flowcharts, schematics, and icons.
Preserves metadata: page, bbox, caption, image_path, hash.
"""
import os
import hashlib
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import io

logger = logging.getLogger("factorymind")


@dataclass
class ImageMetadata:
    """Metadata for extracted images."""
    id: str
    page: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    image_path: str
    hash: str
    caption: Optional[str] = None
    image_type: str = "figure"  # figure, table, diagram, flowchart, schematic, icon
    width: Optional[float] = None
    height: Optional[float] = None
    confidence: float = 0.0


class ImageExtractor:
    """
    Extracts images from PDF pages with metadata preservation.
    Supports raster images and vector-drawn diagrams.
    """
    
    # Patterns for detecting different image types
    DIAGRAM_PATTERNS = [
        r"(?i)(fig\.?\s*\d+|figure\s*\d+)",
        r"(?i)(diagram|illustration|schematic|drawing)",
        r"(?i)(flowchart|flow\s*chart)",
        r"(?i)(circuit|wiring)",
    ]
    
    ICON_PATTERNS = [
        r"(?i)(icon|symbol)",
        r"(?i)(warning|caution|danger)",
        r"(?i)(safety|hazard)",
    ]
    
    def __init__(self, output_dir: str = "extracted_images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Deduplication tracking
        self.saved_hashes: Dict[str, str] = {}  # hash -> image_path
        
        # Statistics
        self.stats = {
            "total_extracted": 0,
            "raster_images": 0,
            "vector_diagrams": 0,
            "deduplicated": 0,
            "by_type": {}
        }
    
    def extract_from_page(self, page, page_num: int, filename: str) -> List[ImageMetadata]:
        """
        Extract all images from a PDF page.
        Returns list of ImageMetadata objects.
        """
        images = []
        
        # Extract raster images
        raster_images = self._extract_raster_images(page, page_num, filename)
        images.extend(raster_images)
        
        # Extract vector-drawn diagrams
        vector_images = self._extract_vector_diagrams(page, page_num, filename)
        images.extend(vector_images)
        
        # Detect captions for images
        self._detect_captions(page, images)
        
        # Classify image types
        self._classify_image_types(page, images)
        
        return images
    
    def _extract_raster_images(self, page, page_num: int, filename: str) -> List[ImageMetadata]:
        """Extract raster images embedded in the page."""
        images = []
        
        try:
            image_list = page.get_images()
            
            for img_index, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    base_image = page.parent.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Generate hash for deduplication
                    img_hash = hashlib.md5(image_bytes).hexdigest()
                    
                    if img_hash in self.saved_hashes:
                        self.stats["deduplicated"] += 1
                        logger.debug(f"Duplicate raster image skipped (hash: {img_hash[:8]})")
                        continue
                    
                    # Save image
                    img_filename = f"{filename}_page_{page_num}_raster_{img_index}.png"
                    img_path = os.path.join(self.output_dir, img_filename)
                    
                    with open(img_path, "wb") as f:
                        f.write(image_bytes)
                    
                    # Get image dimensions
                    img = Image.open(io.BytesIO(image_bytes))
                    width, height = img.size
                    
                    # Store reference
                    self.saved_hashes[img_hash] = f"/extracted_images/{img_filename}"
                    
                    metadata = ImageMetadata(
                        id=f"{filename}_p{page_num}_raster_{img_index}",
                        page=page_num,
                        bbox=(0, 0, width, height),  # Full image bbox
                        image_path=f"/extracted_images/{img_filename}",
                        hash=img_hash,
                        width=width,
                        height=height,
                        image_type="figure"
                    )
                    
                    images.append(metadata)
                    self.stats["raster_images"] += 1
                    self.stats["total_extracted"] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to extract raster image {img_index} from page {page_num}: {e}")
        
        except Exception as e:
            logger.warning(f"Failed to get image list from page {page_num}: {e}")
        
        return images
    
    def _extract_vector_diagrams(self, page, page_num: int, filename: str) -> List[ImageMetadata]:
        """Extract vector-drawn diagrams by rasterizing drawing areas."""
        images = []
        
        try:
            drawings = page.get_drawings()
            
            if len(drawings) < 10:  # Skip pages with few drawings (likely not diagrams)
                return images
            
            # Calculate bounding box of all drawings
            x0, y0, x1, y1 = float("inf"), float("inf"), float("-inf"), float("-inf")
            for d in drawings:
                rect = d.get("rect")
                if rect:
                    x0 = min(x0, rect.x0)
                    y0 = min(y0, rect.y0)
                    x1 = max(x1, rect.x1)
                    y1 = max(y1, rect.y1)
            
            if x1 > x0 and y1 > y0:
                clip = page.rect.intersect((x0, y0, x1, y1))
                
                # Only extract if area is large enough
                if clip.width > 50 and clip.height > 50:
                    try:
                        pix = page.get_pixmap(clip=clip, dpi=200)
                        image_bytes = pix.tobytes("png")
                        
                        # Generate hash
                        img_hash = hashlib.md5(image_bytes).hexdigest()
                        
                        if img_hash in self.saved_hashes:
                            self.stats["deduplicated"] += 1
                            logger.debug(f"Duplicate vector diagram skipped (hash: {img_hash[:8]})")
                        else:
                            # Save image
                            img_filename = f"{filename}_page_{page_num}_vector_{len(images)}.png"
                            img_path = os.path.join(self.output_dir, img_filename)
                            pix.save(img_path)
                            
                            # Store reference
                            self.saved_hashes[img_hash] = f"/extracted_images/{img_filename}"
                            
                            metadata = ImageMetadata(
                                id=f"{filename}_p{page_num}_vector_{len(images)}",
                                page=page_num,
                                bbox=(x0, y0, x1, y1),
                                image_path=f"/extracted_images/{img_filename}",
                                hash=img_hash,
                                width=clip.width,
                                height=clip.height,
                                image_type="diagram"
                            )
                            
                            images.append(metadata)
                            self.stats["vector_diagrams"] += 1
                            self.stats["total_extracted"] += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to rasterize vector diagram on page {page_num}: {e}")
        
        except Exception as e:
            logger.warning(f"Failed to extract vector diagrams from page {page_num}: {e}")
        
        return images
    
    def _detect_captions(self, page, images: List[ImageMetadata]):
        """Detect captions for images based on nearby text."""
        try:
            blocks = page.get_text("blocks")
            
            for img in images:
                img_center_x = (img.bbox[0] + img.bbox[2]) / 2
                img_bottom_y = img.bbox[3]
                
                # Look for text below the image
                for block in blocks:
                    # Handle different block formats from PyMuPDF
                    if not isinstance(block, (list, tuple)) or len(block) < 5:
                        continue
                    
                    block_bbox = block[0]
                    block_text = block[4].strip() if isinstance(block[4], str) else ""
                    
                    if not block_text or len(block_text) > 200:
                        continue
                    
                    # Check if block_bbox is a valid tuple/list
                    if not isinstance(block_bbox, (list, tuple)) or len(block_bbox) < 4:
                        continue
                    
                    # Check if block is below image and horizontally aligned
                    block_center_x = (block_bbox[0] + block_bbox[2]) / 2
                    block_top_y = block_bbox[1]
                    
                    if (abs(block_center_x - img_center_x) < 100 and
                        block_top_y > img_bottom_y and
                        block_top_y - img_bottom_y < 100):
                        
                        # Check if it looks like a caption (short, contains figure ref)
                        if any(pattern.search(block_text) for pattern in self.DIAGRAM_PATTERNS):
                            img.caption = block_text
                            break
        
        except Exception as e:
            logger.warning(f"Failed to detect captions: {e}")
    
    def _classify_image_types(self, page, images: List[ImageMetadata]):
        """Classify images based on context and patterns."""
        try:
            text = page.get_text().lower()
            
            for img in images:
                # Check for icon patterns
                if any(re.search(pattern, text) for pattern in self.ICON_PATTERNS):
                    img.image_type = "icon"
                    self.stats["by_type"]["icon"] = self.stats["by_type"].get("icon", 0) + 1
                
                # Check for diagram patterns
                elif any(re.search(pattern, text) for pattern in self.DIAGRAM_PATTERNS):
                    img.image_type = "diagram"
                    self.stats["by_type"]["diagram"] = self.stats["by_type"].get("diagram", 0) + 1
                
                # Default to figure
                else:
                    self.stats["by_type"]["figure"] = self.stats["by_type"].get("figure", 0) + 1
        
        except Exception as e:
            logger.warning(f"Failed to classify image types: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return {
            **self.stats,
            "unique_images": self.stats["total_extracted"] - self.stats["deduplicated"]
        }
    
    def print_statistics(self):
        """Print extraction statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("  IMAGE EXTRACTION STATISTICS")
        print("="*70)
        print(f"  Total extracted:     {stats['total_extracted']}")
        print(f"  Raster images:      {stats['raster_images']}")
        print(f"  Vector diagrams:    {stats['vector_diagrams']}")
        print(f"  Deduplicated:       {stats['deduplicated']}")
        print(f"  Unique images:      {stats['unique_images']}")
        
        if stats['by_type']:
            print("\n  By type:")
            for img_type, count in stats['by_type'].items():
                print(f"    {img_type:15} {count}")
        
        print("="*70 + "\n")
