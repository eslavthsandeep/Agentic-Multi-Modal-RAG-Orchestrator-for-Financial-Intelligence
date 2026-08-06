from app.db.qdrant_client import get_qdrant_client

client = get_qdrant_client()
scroll_res = client.scroll(collection_name="omnibrain_text", limit=200)
points = scroll_res[0]
print(f"Total points returned by scroll: {len(points)}")

pages = set()
matches = []
for p in points:
    payload = p.payload or {}
    pages.add(payload.get("page_num"))
    text = payload.get("text", "")
    t_lower = text.lower()
    if any(k in t_lower for k in ["gross margin", "justice", "doj", "antitrust", "repurchase", "buyback"]):
        matches.append((payload.get("page_num"), text))

print("Indexed Pages:", sorted([p for p in pages if p is not None]))
print(f"\nFound {len(matches)} keyword matches in vector store:")
for page_num, text in matches:
    print(f"=== Page {page_num} ===\n{text[:300]}\n")
