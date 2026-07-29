"""
OCR Manager for FactoryMind AI Ingestion Pipeline.
Robust OCR fallback chain with automatic dependency detection.
Never crashes - always falls back gracefully.
"""
import logging
import inspect
from typing import Optional, Dict, Any, Tuple
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
        """PyMuPDF is already loaded via fitz."""
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
                    confidence=1.0,  # Native text is highest confidence
                    metadata={"method": "native_extraction"}
                )
        except Exception as e:
            logger.warning(f"PyMuPDF text extraction failed: {e}")
        
        return None


class DoclingBackend(OCRBackend):
    """Docling OCR and layout analysis."""
    
    def __init__(self):
        super().__init__("Docling")
        self.dep_manager = get_dependency_manager()
    
    def _initialize_engine(self):
        """Initialize Docling with version detection."""
        if not self.dep_manager.is_available("docling"):
            raise ImportError("Docling not available")
        
        # Try current API (simplified)
        try:
            from docling.document_converter import DocumentConverter
            self.converter = DocumentConverter()
            self.api_version = "current"
        except Exception as e:
            raise ImportError(f"Docling API not compatible: {e}")
        
        self.version = self.dep_manager.get_version("docling")
    
    def extract_text_from_bytes(self, pdf_bytes: bytes, page_num: int) -> Optional[OCRResult]:
        """Extract text using Docling."""
        if not self.available:
            return None
        
        try:
            import tempfile
            
            # Docling works with file paths
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            
            try:
                result = self.converter.convert(tmp_path)
                text = result.document.export_to_markdown()
                
                if text.strip():
                    return OCRResult(
                        text=text.strip(),
                        engine="Docling",
                        confidence=0.9,
                        metadata={"api_version": self.api_version, "page": page_num}
                    )
            finally:
                import os
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.warning(f"Docling extraction failed: {e}")
        
        return None


class PaddleOCRBackend(OCRBackend):
    """PaddleOCR with automatic version detection."""
    
    def __init__(self):
        super().__init__("PaddleOCR")
        self.dep_manager = get_dependency_manager()
    
    def _initialize_engine(self):
        """Initialize PaddleOCR with version detection."""
        if not self.dep_manager.is_available("paddleocr"):
            raise ImportError("PaddleOCR not available")
        
        from paddleocr import PaddleOCR
        
        # Detect API version by inspecting the ocr method signature
        ocr_signature = inspect.signature(PaddleOCR.ocr)
        
        # Check if 'cls' parameter exists in signature
        has_cls_param = 'cls' in ocr_signature.parameters
        
        # Use new parameter name to avoid deprecation warning
        try:
            self._engine = PaddleOCR(use_textline_orientation=True, lang='en')
        except TypeError:
            # Fallback for older versions
            self._engine = PaddleOCR(use_angle_cls=True, lang='en')
        
        self.api_version = "new" if not has_cls_param else "old"
        self.version = self.dep_manager.get_version("paddleocr")
        
        logger.info(f"PaddleOCR API version: {self.api_version}")
    
    def extract_text(self, image: Image.Image, **kwargs) -> Optional[OCRResult]:
        """Extract text using PaddleOCR."""
        if not self.available:
            return None
        
        try:
            import numpy as np
            img_np = np.array(image)
            
            # Use appropriate API based on version
            if self.api_version == "new":
                # New API - no cls parameter
                result = self._engine.ocr(img_np)
            else:
                # Old API - with cls parameter
                result = self._engine.ocr(img_np, cls=True)
            
            txts = []
            if result and result[0]:
                for line in result[0]:
                    txts.append(line[1][0])
            
            if txts:
                return OCRResult(
                    text=" ".join(txts),
                    engine="PaddleOCR",
                    confidence=0.85,
                    metadata={"api_version": self.api_version}
                )
        except Exception as e:
            logger.warning(f"PaddleOCR extraction failed: {e}")
        
        return None


class EasyOCRBackend(OCRBackend):
    """EasyOCR backend."""
    
    def __init__(self):
        super().__init__("EasyOCR")
        self.dep_manager = get_dependency_manager()
    
    def _initialize_engine(self):
        """Initialize EasyOCR."""
        if not self.dep_manager.is_available("easyocr"):
            raise ImportError("EasyOCR not available")
        
        import easyocr
        self._engine = easyocr.Reader(['en'])
        self.version = self.dep_manager.get_version("easyocr")
    
    def extract_text(self, image: Image.Image, **kwargs) -> Optional[OCRResult]:
        """Extract text using EasyOCR."""
        if not self.available:
            return None
        
        try:
            import numpy as np
            img_np = np.array(image)
            
            result = self._engine.readtext(img_np)
            txts = [text for bbox, text, conf in result if conf > 0.5]
            
            if txts:
                return OCRResult(
                    text=" ".join(txts),
                    engine="EasyOCR",
                    confidence=0.8,
                    metadata={"detections": len(result)}
                )
        except Exception as e:
            logger.warning(f"EasyOCR extraction failed: {e}")
        
        return None


class TesseractBackend(OCRBackend):
    """Tesseract OCR backend."""
    
    def __init__(self):
        super().__init__("Tesseract")
        self.dep_manager = get_dependency_manager()
    
    def _initialize_engine(self):
        """Initialize Tesseract."""
        if not self.dep_manager.is_available("tesseract"):
            raise ImportError("Tesseract not available")
        
        import pytesseract
        self._engine = pytesseract
        self.version = self.dep_manager.get_version("tesseract")
    
    def extract_text(self, image: Image.Image, **kwargs) -> Optional[OCRResult]:
        """Extract text using Tesseract."""
        if not self.available:
            return None
        
        try:
            text = self._engine.image_to_string(image, lang='eng')
            if text.strip():
                return OCRResult(
                    text=text.strip(),
                    engine="Tesseract",
                    confidence=0.75,
                    metadata={}
                )
        except Exception as e:
            logger.warning(f"Tesseract extraction failed: {e}")
        
        return None


class OCRManager:
    """
    Main OCR Manager with robust fallback chain.
    Priority: PyMuPDF → Docling → PaddleOCR → EasyOCR → Tesseract
    """
    
    def __init__(self):
        self.dep_manager = get_dependency_manager()
        self.backends: Dict[str, OCRBackend] = {}
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize all available backends in priority order."""
        backend_classes = [
            ("PyMuPDF", PyMuPDFBackend),
            ("Docling", DoclingBackend),
            ("PaddleOCR", PaddleOCRBackend),
            ("EasyOCR", EasyOCRBackend),
            ("Tesseract", TesseractBackend),
        ]
        
        for name, backend_class in backend_classes:
            backend = backend_class()
            backend.initialize()
            self.backends[name] = backend
    
    def extract_from_page(self, page, page_num: int, min_text_length: int = 50) -> OCRResult:
        """
        Extract text from a PDF page with smart detection.
        Skips OCR if native text is sufficient.
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
                logger.info(f"  Page {page_num}: PyMuPDF text too short ({len(result.text)} chars), trying OCR...")
        
        # Stage 2: Try Docling if available
        docling_backend = self.backends.get("Docling")
        if docling_backend and docling_backend.is_available():
            logger.info(f"  Page {page_num}: Trying Docling...")
            try:
                # Get PDF bytes for Docling
                pdf_bytes = page.parent.tobytes()
                result = docling_backend.extract_text_from_bytes(pdf_bytes, page_num)
                if result:
                    logger.info(f"  Page {page_num}: Docling ✓")
                    return result
            except Exception as e:
                logger.warning(f"  Page {page_num}: Docling ✗ ({e})")
        else:
            logger.info(f"  Page {page_num}: Docling skipped (not installed)")
        
        # Stage 3: Render page to image and try OCR backends
        try:
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))
            
            # Try PaddleOCR
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
            
            # Try EasyOCR
            easyocr_backend = self.backends.get("EasyOCR")
            if easyocr_backend and easyocr_backend.is_available():
                logger.info(f"  Page {page_num}: Trying EasyOCR...")
                result = easyocr_backend.extract_text(image)
                if result:
                    logger.info(f"  Page {page_num}: EasyOCR ✓")
                    return result
                else:
                    logger.info(f"  Page {page_num}: EasyOCR ✗")
            else:
                logger.info(f"  Page {page_num}: EasyOCR skipped (not installed)")
            
            # Try Tesseract
            tesseract_backend = self.backends.get("Tesseract")
            if tesseract_backend and tesseract_backend.is_available():
                logger.info(f"  Page {page_num}: Trying Tesseract...")
                result = tesseract_backend.extract_text(image)
                if result:
                    logger.info(f"  Page {page_num}: Tesseract ✓")
                    return result
                else:
                    logger.info(f"  Page {page_num}: Tesseract ✗")
            else:
                logger.info(f"  Page {page_num}: Tesseract skipped (not installed)")
            
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


if __name__ == "__main__":
    # Test the OCR manager
    manager = OCRManager()
    manager.print_status()
