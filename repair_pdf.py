"""
Attempt to repair corrupted PDFs using pikepdf.
"""
import pikepdf
import os

problematic_pdfs = [
    "hyundai-r215l-smart-maintenance-standard-manual.pdf",
    "hyundai-r215l-smart-mechatronics-system-manual.pdf",
    "hyundai-r215l-smart-electrical-system-manual.pdf"
]

for filename in problematic_pdfs:
    filepath = f"data/manuals/{filename}"
    backup_path = f"data/manuals/{filename}.backup"
    repaired_path = f"data/manuals/{filename}.repaired.pdf"
    
    print(f"\n{'='*70}")
    print(f"Attempting to repair: {filename}")
    print(f"{'='*70}")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found")
        continue
    
    # Create backup
    try:
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy(filepath, backup_path)
            print(f"✅ Backup created: {backup_path}")
    except Exception as e:
        print(f"⚠️  Backup failed: {e}")
    
    # Try to open and repair with pikepdf
    try:
        with pikepdf.open(filepath, allow_overwriting_input=True) as pdf:
            print(f"✅ Opened with pikepdf")
            print(f"Pages: {len(pdf.pages)}")
            
            # Try to save as repaired version
            pdf.save(repaired_path, linearize=True)
            print(f"✅ Repaired version saved: {repaired_path}")
            
            # Verify repaired version
            with pikepdf.open(repaired_path) as repaired_pdf:
                print(f"✅ Repaired PDF pages: {len(repaired_pdf.pages)}")
                
                # Test with PyMuPDF
                import fitz
                doc = fitz.open(repaired_path)
                print(f"✅ PyMuPDF can read repaired: {len(doc)} pages")
                doc.close()
                
                # Replace original with repaired
                import shutil
                shutil.move(repaired_path, filepath)
                print(f"✅ Original replaced with repaired version")
                
    except Exception as e:
        print(f"❌ Repair failed: {e}")
        import traceback
        traceback.print_exc()
