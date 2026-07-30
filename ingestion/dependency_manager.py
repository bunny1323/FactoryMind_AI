"""
Dependency Manager for FactoryMind AI Ingestion Pipeline.
Automatically detects installed libraries and their versions.
Provides capability detection and installation guidance.
"""
import importlib
import subprocess
import sys
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("factorymind")


@dataclass
class DependencyInfo:
    """Information about a dependency."""
    name: str
    installed: bool
    version: Optional[str] = None
    import_path: Optional[str] = None
    install_command: Optional[str] = None
    capabilities: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


class DependencyManager:
    """Manages dependency detection and capability reporting."""
    
    # Known dependencies with their import paths and install commands
    DEPENDENCIES = {
        "docling": {
            "import_path": "docling",
            "install_command": "pip install docling",
            "capabilities": ["pdf_layout_analysis", "table_extraction", "ocr"]
        },
        "paddleocr": {
            "import_path": "paddleocr",
            "install_command": "pip install paddleocr",
            "capabilities": ["ocr", "multilingual_ocr"]
        },
        "easyocr": {
            "import_path": "easyocr",
            "install_command": "pip install easyocr",
            "capabilities": ["ocr", "multilingual_ocr"]
        },
        "tesseract": {
            "import_path": "pytesseract",
            "install_command": "pip install pytesseract && winget install UB-Mannheim.TesseractOCR",
            "capabilities": ["ocr", "legacy_ocr"]
        },
        "pymupdf": {
            "import_path": "fitz",
            "install_command": "pip install pymupdf",
            "capabilities": ["pdf_text_extraction", "image_extraction", "pdf_rendering"]
        },
        "pdfplumber": {
            "import_path": "pdfplumber",
            "install_command": "pip install pdfplumber",
            "capabilities": ["table_extraction", "pdf_layout_analysis"]
        },
        "camelot": {
            "import_path": "camelot",
            "install_command": "pip install camelot-py[cv]",
            "capabilities": ["table_extraction"]
        },
        "poppler": {
            "import_path": None,  # System library, not Python
            "install_command": "winget install -e --id Ghostscript.Ghostscript",
            "capabilities": ["pdf_rendering", "system_dependency"]
        }
    }
    
    def __init__(self):
        self._cache: Dict[str, DependencyInfo] = {}
        self._detect_all()
    
    def _detect_all(self):
        """Detect all dependencies at initialization."""
        for name, config in self.DEPENDENCIES.items():
            self._cache[name] = self._detect_dependency(name, config)
    
    def _detect_dependency(self, name: str, config: Dict) -> DependencyInfo:
        """Detect a single dependency."""
        import_path = config.get("import_path")
        install_command = config.get("install_command")
        capabilities = config.get("capabilities", [])
        
        if import_path is None:
            # System dependency - check via command
            return self._detect_system_dependency(name, install_command, capabilities)
        
        # PaddleOCR needs a deeper check — the package imports fine but
        # silently fails if paddlepaddle (the inference engine) is absent.
        if name == "paddleocr":
            return self._detect_paddleocr(install_command, capabilities)
        
        try:
            module = importlib.import_module(import_path)
            version = self._get_version(module, import_path)
            return DependencyInfo(
                name=name,
                installed=True,
                version=version,
                import_path=import_path,
                install_command=install_command,
                capabilities=capabilities
            )
        except ImportError:
            return DependencyInfo(
                name=name,
                installed=False,
                import_path=import_path,
                install_command=install_command,
                capabilities=capabilities
            )
    
    def _detect_system_dependency(self, name: str, install_command: str, capabilities: List[str]) -> DependencyInfo:
        """Detect system-level dependencies."""
        try:
            # Try to check if command exists
            result = subprocess.run(
                ["where", name] if sys.platform == "win32" else ["which", name],
                capture_output=True,
                text=True,
                timeout=5
            )
            installed = result.returncode == 0
            version = self._get_system_version(name) if installed else None
            return DependencyInfo(
                name=name,
                installed=installed,
                version=version,
                install_command=install_command,
                capabilities=capabilities
            )
        except Exception:
            return DependencyInfo(
                name=name,
                installed=False,
                install_command=install_command,
                capabilities=capabilities
            )
    
    def _detect_paddleocr(self, install_command: str, capabilities: list) -> DependencyInfo:
        """Detect PaddleOCR with a real instantiation test.
        
        paddleocr 3.x imports fine even when paddlepaddle is missing,
        so we need to actually try constructing the OCR object.
        """
        try:
            import paddleocr  # noqa: F401 — verify package exists
            version_str = getattr(paddleocr, "__version__", "installed")
        except ImportError:
            logger.debug("paddleocr package not found")
            return DependencyInfo(
                name="paddleocr",
                installed=False,
                install_command=install_command,
                capabilities=capabilities
            )
        
        # Now verify the inference backend (paddlepaddle) is importable
        try:
            import paddle  # noqa: F401
            logger.debug("PaddleOCR + PaddlePaddle both available")
            return DependencyInfo(
                name="paddleocr",
                installed=True,
                version=version_str,
                import_path="paddleocr",
                install_command=install_command,
                capabilities=capabilities
            )
        except ImportError as e:
            logger.warning(
                f"PaddleOCR package is installed but PaddlePaddle backend is missing: {e}\n"
                f"  Install PaddlePaddle: pip install paddlepaddle"
            )
            return DependencyInfo(
                name="paddleocr",
                installed=False,
                version=version_str,
                import_path="paddleocr",
                install_command="pip install paddlepaddle && pip install paddleocr",
                capabilities=capabilities
            )
    
    def _get_version(self, module, import_path: str) -> Optional[str]:
        """Extract version from a module."""
        try:
            if hasattr(module, "__version__"):
                return str(module.__version__)
            elif hasattr(module, "version"):
                return str(module.version)
            elif import_path == "fitz":
                # PyMuPDF
                return str(module.version)
            else:
                return "unknown"
        except Exception:
            return "unknown"
    
    def _get_system_version(self, name: str) -> Optional[str]:
        """Get version of system dependency."""
        try:
            result = subprocess.run(
                [name, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip().split()[-1]
        except Exception:
            pass
        return None
    
    def get_info(self, name: str) -> DependencyInfo:
        """Get information about a specific dependency."""
        return self._cache.get(name, DependencyInfo(name, False))
    
    def is_available(self, name: str) -> bool:
        """Check if a dependency is available."""
        return self._cache.get(name, DependencyInfo(name, False)).installed
    
    def get_version(self, name: str) -> Optional[str]:
        """Get version of a dependency."""
        return self._cache.get(name, DependencyInfo(name, False)).version
    
    def has_capability(self, capability: str) -> List[str]:
        """Get all dependencies that have a specific capability."""
        return [
            name for name, info in self._cache.items()
            if info.installed and capability in info.capabilities
        ]
    
    def print_report(self):
        """Print a formatted dependency report."""
        print("\n" + "="*70)
        print("  DEPENDENCY REPORT")
        print("="*70 + "\n")
        
        # Group by status
        installed = []
        missing = []
        
        for name, info in sorted(self._cache.items()):
            if info.installed:
                installed.append(info)
            else:
                missing.append(info)
        
        if installed:
            print("✅ INSTALLED DEPENDENCIES:")
            print("-" * 70)
            for info in installed:
                version_str = f"v{info.version}" if info.version else "version unknown"
                caps_str = ", ".join(info.capabilities) if info.capabilities else "general"
                print(f"  {info.name:20} {version_str:20} [{caps_str}]")
            print()
        
        if missing:
            print("❌ MISSING DEPENDENCIES:")
            print("-" * 70)
            for info in missing:
                caps_str = ", ".join(info.capabilities) if info.capabilities else "general"
                print(f"  {info.name:20} [{caps_str}]")
                print(f"    Install: {info.install_command}")
            print()
        
        # OCR capabilities summary
        print("🔍 OCR CAPABILITIES:")
        print("-" * 70)
        ocr_backends = self.has_capability("ocr")
        if ocr_backends:
            print(f"  Available OCR engines: {', '.join(ocr_backends)}")
        else:
            print("  ⚠️  No OCR engines installed. Install one of:")
            for info in missing:
                if "ocr" in info.capabilities:
                    print(f"    - {info.name}: {info.install_command}")
        
        # Table extraction capabilities
        print("\n📊 TABLE EXTRACTION CAPABILITIES:")
        print("-" * 70)
        table_backends = self.has_capability("table_extraction")
        if table_backends:
            print(f"  Available table extractors: {', '.join(table_backends)}")
        else:
            print("  ⚠️  No table extractors installed.")
        
        print("="*70 + "\n")
    
    def get_missing_install_commands(self) -> List[str]:
        """Get install commands for all missing dependencies."""
        return [
            info.install_command
            for info in self._cache.values()
            if not info.installed and info.install_command
        ]


# Global instance
_dependency_manager = None


def get_dependency_manager() -> DependencyManager:
    """Get the global dependency manager instance."""
    global _dependency_manager
    if _dependency_manager is None:
        _dependency_manager = DependencyManager()
    return _dependency_manager


if __name__ == "__main__":
    # Test the dependency manager
    manager = DependencyManager()
    manager.print_report()
