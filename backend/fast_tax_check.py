import pymupdf
from pathlib import Path

uploads = list(Path("data/uploads").glob("*.pdf"))
for pdf_file in uploads:
    doc = pymupdf.open(str(pdf_file))
    print(f"=== File: {pdf_file.name} ({len(doc)} pages) ===")
    tax_pages = []
    effective_tax_pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if "effective tax rate" in text.lower():
            effective_tax_pages.append((i + 1, text))
        elif "provision for income taxes" in text.lower() or "income tax" in text.lower():
            tax_pages.append((i + 1, text))
            
    print(f"  Pages matching 'effective tax rate': {len(effective_tax_pages)}")
    for page_num, text in effective_tax_pages:
        snippet = text.replace('\n', ' ')
        print(f"    Page {page_num}: {snippet[:250]}...")
        
    print(f"  Pages matching 'income tax': {len(tax_pages)}")
    for page_num, text in tax_pages[:3]:
        snippet = text.replace('\n', ' ')
        print(f"    Page {page_num}: {snippet[:150]}...")
