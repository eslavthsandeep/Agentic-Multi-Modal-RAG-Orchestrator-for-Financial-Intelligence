import logging
logging.basicConfig(level=logging.INFO)

from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import embed_texts, embed_text_for_image_search
from app.db.qdrant_client import get_qdrant_client, search_text, search_images, ensure_collections, upsert_text_chunks
from app.agents.search_agent import _hybrid_rerank

doc_id = "debug_scores_doc"
pdf_path = "data/uploads/cb1942fe-574.pdf"

ensure_collections()
parsed = parse_pdf(pdf_path, "data/extracted_images/debug")
chunks = chunk_document(parsed, chunk_size=500, overlap=50)

texts = [c["text"] for c in chunks]
embeddings = embed_texts(texts)
upsert_text_chunks(doc_id, chunks, embeddings)

query = "What was Apple's effective tax rate in fiscal 2025, and how did it compare to 2024?"
query_emb = embed_texts([query])[0]

text_res = search_text(query_emb, doc_id, top_k=500)
print(f"\nTotal candidate chunks retrieved: {len(text_res)}")

reranked = _hybrid_rerank(query, text_res)

print("\nTOP 10 RERANKED CHUNKS AND THEIR SCORES:")
for i, item in enumerate(reranked[:10]):
    print(f"\n--- Rank {i+1} | Page {item.get('page_num')} | Score: {item.get('score')} ---")
    snippet = item.get("text", "").replace("\n", " ")
    print(f"Snippet: {snippet[:200]}...")
