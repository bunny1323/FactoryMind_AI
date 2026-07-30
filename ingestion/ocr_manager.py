"""
OCR Manager for FactoryMind AI Ingestion Pipeline.
Clean, production-grade OCR pipeline with PyMuPDF native text extraction
and PaddleOCR for scanned/fallback page processing.
"""
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from PIL import Image
import io

from .dependency_manager import get_dependency_manager

logger = logging.getLogger("factorymind")


@dataclass
class OCRResult:
    """Result from OCR operation."""
    text: str
    engine: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class OCRBackend:
    """Base class for OCR backends."""
    
    def __init__(self, name: str):
        self.name = name
        self.available = False
        self.version = None
        self._engine = None
    
    def initialize(self) -> bool:
        """Initialize the OCR engine. Returns True if successful."""
        try:
            self._initialize_engine()
            self.available = True
            logger.info(f"✅ {self.name} initialized successfully")
            return True
        except Exception as e:
            logger.warning(f"⚠️  {self.name} initialization failed: {e}")
            self.available = False
            return False
    
    def _initialize_engine(self):
        """Override in subclass to initialize specific engine."""
        raise NotImplementedError
    
    def extract_text(self, image: Image.Image, **kwargs) -> Optional[OCRResult]:
        """Extract text from image. Override in subclass."""
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """Check if backend is available."""
        return self.available


class PyMuPDFBackend(OCRBackend):
    """Native PDF text extraction using PyMuPDF."""
    
    def __init__(self):
        super().__init__("PyMuPDF")
        self.dep_manager = get_dependency_manager()
    
    def _initialize_engine(self):
        if not self.dep_manager.is_available("pymupdf"):
            raise ImportError("PyMuPDF not available")
        import fitz
        self.version = self.dep_manager.get_version("pymupdf")
        self._engine = fitz
    
    def extract_text_from_page(self, page) -> Optional[OCRResult]:
        """Extract text directly from PyMuPDF page."""
        if not self.available:
            return None
        
        try:
            text = page.get_text()
            if text.strip():
                return OCRResult(
                    text=text.strip(),
                    engine="PyMuPDF",
                    confidence=1.0,
                    metadata={"method": "native_extraction"}
                )
        except Exception as e:
            logger.warning(f"PyMuPDF text extraction failed: {e}")
        
        return None


class PaddleOCRBackend(OCRBackend):
    """PaddleOCR for fallback text extraction on scanned pages."""
    
    def __init__(self):
        super().__init__("PaddleOCR")
        self.dep_manager = get_dependency_manager()
    
    def _initialize_engine(self):
        if not self.dep_manager.is_available("paddleocr"):
            raise ImportError("PaddleOCR not available")
        
        import os
        os.environ['FLAGS_use_mkldnn'] = '0'
        os.environ['MKL_THREADING_LAYER'] = 'GNU'
        
        from paddleocr import PaddleOCR
        
        try:
            self._engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            self.api_version = "v3_modern"
            logger.info("PaddleOCR initialized with v3 modern API")
        except Exception as e:
            logger.warning(f"PaddleOCR modern API failed: {e}, trying simple API...")
            try:
                self._engine = PaddleOCR(lang='en', show_log=False)
                self.api_version = "v3_simple"
            except Exception as e2:
                logger.error(f"All PaddleOCR initialization attempts failed: {e2}")
                raise ImportError(f"PaddleOCR API not compatible: {e2}")
        
        self.version = self.dep_manager.get_version("paddleocr")
        logger.info(f"PaddleOCR version: {self.version}")
    
    def extract_text(self, image: Image.Image, **kwargs) -> Optional[OCRResult]:
        """Extract text using PaddleOCR."""
        if not self.available:
            return None
        
        try:
            import numpy as np
            img_np = np.array(image)
            
            if self.api_version == "v3_modern":
                result = self._engine.ocr(img_np, cls=True)
            else:
                result = self._engine.ocr(img_np)
            
            txts = []
            if result and result[0]:
                for line in result[0]:
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        if isinstance(line[1], (list, tuple)) and len(line[1]) >= 1:
                            txts.append(line[1][0])
                        elif isinstance(line[1], str):
                            txts.append(line[1])
            
            if txts:
                return OCRResult(
                    text=" ".join(txts),
                    engine="PaddleOCR",
                    confidence=0.85,
                    metadata={"api_version": self.api_version, "lines_detected": len(txts)}
                )
        except Exception as e:
            logger.warning(f"PaddleOCR extraction failed: {e}")
        
        return None


class OCRManager:
    """
    OCR Manager supporting only PyMuPDF and PaddleOCR.
    Priority: PyMuPDF -> PaddleOCR
    """

    def __init__(self):
        self.dep_manager = get_dependency_manager()
        self.backends: Dict[str, OCRBackend] = {}
        self._initialize_backends()

    def _initialize_backends(self):
        """Initialize available backends in priority order."""
        backend_classes = [
            ("PyMuPDF", PyMuPDFBackend),
            ("PaddleOCR", PaddleOCRBackend),
        ]
        
        for name, backend_class in backend_classes:
            backend = backend_class()
            backend.initialize()
            self.backends[name] = backend
    
    def extract_from_page(self, page, page_num: int, min_text_length: int = 50) -> OCRResult:
        """
        Extract text from a PDF page. Skips OCR if native text is sufficient.
        """
        logger.info(f"📄 Processing page {page_num}")
        
        # Stage 1: Try native PyMuPDF text first
        pymupdf_backend = self.backends.get("PyMuPDF")
        if pymupdf_backend and pymupdf_backend.is_available():
            logger.info(f"  Page {page_num}: Trying PyMuPDF native text...")
            result = pymupdf_backend.extract_text_from_page(page)
            if result and len(result.text) >= min_text_length:
                logger.info(f"  Page {page_num}: PyMuPDF ✓ (native text sufficient)")
                return result
            elif result:
                logger.info(f"  Page {page_num}: PyMuPDF text too short ({len(result.text)} chars), trying PaddleOCR...")
        
        # Stage 2: Render page to image and try PaddleOCR
        try:
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))
            
            paddle_backend = self.backends.get("PaddleOCR")
            if paddle_backend and paddle_backend.is_available():
                logger.info(f"  Page {page_num}: Trying PaddleOCR...")
                result = paddle_backend.extract_text(image)
                if result:
                    logger.info(f"  Page {page_num}: PaddleOCR ✓")
                    return result
                else:
                    logger.info(f"  Page {page_num}: PaddleOCR ✗")
            else:
                logger.info(f"  Page {page_num}: PaddleOCR skipped (not installed)")
            
        except Exception as e:
            logger.warning(f"  Page {page_num}: Image rendering failed: {e}")
        
        # Final fallback: Return whatever native text we have
        if pymupdf_backend and pymupdf_backend.is_available():
            result = pymupdf_backend.extract_text_from_page(page)
            if result:
                logger.info(f"  Page {page_num}: Using PyMuPDF fallback (short text)")
                return result
        
        logger.warning(f"  Page {page_num}: All OCR methods failed")
        return OCRResult(text="", engine="None", confidence=0.0)
    
    def get_available_backends(self) -> list[str]:
        """Get list of available OCR backends."""
        return [name for name, backend in self.backends.items() if backend.is_available()]
    
    def print_status(self):
        """Print status of all OCR backends."""
        print("\n" + "="*70)
        print("  OCR BACKEND STATUS")
        print("="*70 + "\n")
        
        for name, backend in self.backends.items():
            status = "✅ Available" if backend.is_available() else "❌ Not Available"
            version = f"v{backend.version}" if backend.version else "version unknown"
            print(f"  {name:15} {status:20} {version}")
        
        print(f"\n  Available engines: {', '.join(self.get_available_backends())}")
        print("="*70 + "\n")


# Global instance
_ocr_manager = None


def get_ocr_manager() -> OCRManager:
    """Get the global OCR manager instance."""
    global _ocr_manager
    if _ocr_manager is None:
        _ocr_manager = OCRManager()
    return _ocr_manager
