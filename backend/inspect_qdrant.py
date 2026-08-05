from app.db.qdrant_client import get_qdrant_client

client = get_qdrant_client()
res, _ = client.scroll(collection_name="omnibrain_text", limit=1000)
print(f"Total chunks in Qdrant: {len(res)}")
tax_chunks = [p for p in res if "tax" in p.payload.get("text", "").lower()]
print(f"Tax-related chunks found: {len(tax_chunks)}")
for p in tax_chunks[:10]:
    page = p.payload.get("page_num")
    text = p.payload.get("text", "").replace("\n", " ")
    print(f"Page {page}: {text[:150]}...")
