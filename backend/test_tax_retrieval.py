import sys
from pathlib import Path
from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.chunker import chunk_document

uploads_dir = Path("data/uploads")
pdf_files = list(uploads_dir.glob("*.pdf"))
print(f"Found {len(pdf_files)} uploaded PDF files:")
for f in pdf_files:
    print(f" - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")

if pdf_files:
    sample_pdf = pdf_files[-1]
    print(f"\nParsing sample PDF: {sample_pdf.name}...")
    parsed = parse_pdf(str(sample_pdf), "data/extracted_images/test_inspect")
    print(f"Extracted {len(parsed['pages'])} text pages, {len(parsed['tables'])} tables, {len(parsed['images'])} images.")
    
    tax_pages = [p for p in parsed["pages"] if "tax" in p["text"].lower()]
    print(f"Found {len(tax_pages)} pages containing 'tax':")
    for p in tax_pages[:5]:
        print(f" Page {p['page_num']}: {p['text'][:200]}...\n")
        
    chunks = chunk_document(parsed, chunk_size=500, overlap=50)
    print(f"Total chunks created: {len(chunks)}")
    tax_chunks = [c for c in chunks if "tax rate" in c["text"].lower() or "effective tax" in c["text"].lower()]
    print(f"Found {len(tax_chunks)} chunks containing 'effective tax' or 'tax rate':")
    for c in tax_chunks[:5]:
        print(f" Chunk index {c['chunk_index']} (Page {c['page_num']}):\n{c['text']}\n{'-'*50}")
