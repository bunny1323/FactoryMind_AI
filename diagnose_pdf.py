"""
Diagnostic script to check problematic PDF files.
"""
import fitz
import os

problematic_pdfs = [
    "hyundai-r215l-smart-maintenance-standard-manual.pdf",
    "hyundai-r215l-smart-mechatronics-system-manual.pdf",
    "hyundai-r215l-smart-electrical-system-manual.pdf"
]

for filename in problematic_pdfs:
    filepath = f"data/manuals/{filename}"
    print(f"\n{'='*70}")
    print(f"Checking: {filename}")
    print(f"{'='*70}")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found")
        continue
    
    # File size
    size = os.path.getsize(filepath)
    print(f"File size: {size} bytes ({size/1024/1024:.2f} MB)")
    
    # Check header
    with open(filepath, "rb") as f:
        header = f.read(4)
        print(f"Header: {header}")
        print(f"Valid PDF header: {header == b'%PDF'}")
    
    # Try to open with PyMuPDF
    try:
        doc = fitz.open(filepath)
        print(f"PyMuPDF pages: {len(doc)}")
        print(f"Is encrypted: {doc.is_encrypted}")
        
        if len(doc) > 0:
            # Try to read first page
            try:
                page = doc[0]
                text = page.get_text()
                print(f"First page text length: {len(text)}")
                print(f"First page text preview: {text[:200]}")
            except Exception as e:
                print(f"Error reading first page: {e}")
        
        doc.close()
    except Exception as e:
        print(f"❌ PyMuPDF error: {e}")
        import traceback
        traceback.print_exc()
    
    # Try to open with pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            print(f"pdfplumber pages: {len(pdf.pages)}")
            if len(pdf.pages) > 0:
                first_page = pdf.pages[0]
                text = first_page.extract_text()
                print(f"First page text length: {len(text) if text else 0}")
    except Exception as e:
        print(f"❌ pdfplumber error: {e}")
