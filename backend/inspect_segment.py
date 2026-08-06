from app.db.qdrant_client import search_text

print("=== CHECKING CASE A: GROSS MARGIN ===")
res_gm = search_text("gross margin percentage fiscal 2025 2024", 5)
print(f"Found {len(res_gm)} chunks:")
for c in res_gm:
    print(f"[Page {c.get('page_num')}] {c.get('text')[:300]}\n")

print("=== CHECKING CASE B: DOJ LAWSUIT ===")
res_doj = search_text("Department of Justice DOJ lawsuit antitrust legal proceedings", 5)
print(f"Found {len(res_doj)} chunks:")
for c in res_doj:
    print(f"[Page {c.get('page_num')}] {c.get('text')[:300]}\n")
