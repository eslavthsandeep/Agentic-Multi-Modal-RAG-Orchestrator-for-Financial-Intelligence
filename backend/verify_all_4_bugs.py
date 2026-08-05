import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s")

from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import embed_texts
from app.db.qdrant_client import ensure_collections, upsert_text_chunks
from app.agents.supervisor import run_query

async def main():
    doc_id = "test_tax_verify_doc"
    pdf_path = Path("data/uploads/cb1942fe-574.pdf")
    if not pdf_path.exists():
        pdf_path = list(Path("data/uploads").glob("*.pdf"))[0]
        
    print(f"\n========================================================")
    print(f"VERIFYING 4 BUGS WITH FILE: {pdf_path.name}")
    print(f"========================================================\n")
    
    # Step 1: Ingest & Chunk
    ensure_collections()
    parsed = parse_pdf(str(pdf_path), "data/extracted_images/test_verify")
    chunks = chunk_document(parsed, chunk_size=500, overlap=50)
    print(f"[OK] Document parsed: {len(parsed['pages'])} pages, {len(chunks)} chunks created.")
    
    # Verify Bug 4: No mid-word truncations
    bad_starts = [c for c in chunks if c['text'].split()[0].lower() in ['versight', 'ccounting', 'tandards', 'itical']]
    print(f"[OK] Bug 4 Check: Chunks starting with truncated words = {len(bad_starts)} (Expected: 0)")
    
    # Store in Qdrant
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    upsert_text_chunks(doc_id, chunks, embeddings)
    print(f"[OK] {len(chunks)} text chunks stored into persistent Qdrant collection.\n")
    
    # Step 2: Execute Test Query
    query = "What was Apple's effective tax rate in fiscal 2025, and how did it compare to 2024?"
    print(f"Executing query: '{query}'...\n")
    
    result = await run_query(query=query, document_id=doc_id)
    
    print("\n========================================================")
    print("FINAL OUTPUT VERIFICATION")
    print("========================================================")
    print(f"Route Taken: {result.get('route_taken')}")
    print(f"Guardrail Status: {result.get('guardrail_status')}")
    print(f"Citations ({len(result.get('citations', []))}): {result.get('citations')}")
    print(f"Agent Trace: {result.get('agent_trace')}")
    ans = result.get('answer', '')
    print(f"\nANSWER:\n{ans.encode('ascii', errors='ignore').decode('ascii')}")

if __name__ == "__main__":
    asyncio.run(main())
