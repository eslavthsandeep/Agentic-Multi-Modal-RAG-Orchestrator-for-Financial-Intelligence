import asyncio, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from app.agents.supervisor import run_query

async def main():
    print("========================================================")
    print("ROUND 8 EVALUATION TEST SUITE")
    print("========================================================\n")

    print("--- DELIVERABLE 1: SEGMENT NET SALES (FY2025 GROUND TRUTH) ---")
    q1 = "What were Apple's net sales by segment in fiscal 2025 — Americas, Europe, Greater China?"
    r1 = await run_query(q1, "test_doc")
    print("Route:", r1.get("route_taken"))
    print("Answer:\n", r1.get("answer"))
    print("\n" + "-"*50 + "\n")

    print("--- DELIVERABLE 2: SHARE REPURCHASES (SEARCH ROUTING & $89.3B FIGURE) ---")
    q2 = "How much did Apple spend on share repurchases in fiscal 2025?"
    r2 = await run_query(q2, "test_doc")
    print("Route:", r2.get("route_taken"))
    print("Answer:\n", r2.get("answer"))
    print("\n" + "-"*50 + "\n")

    print("--- DELIVERABLE 3A: GROSS MARGIN PERCENTAGE ---")
    q3a = "What was Apple's gross margin percentage in fiscal 2025 vs 2024?"
    r3a = await run_query(q3a, "test_doc")
    print("Route:", r3a.get("route_taken"))
    print("Answer:\n", r3a.get("answer"))
    print("\n" + "-"*50 + "\n")

    print("--- DELIVERABLE 3B: DOJ LAWSUIT STATUS ---")
    q3b = "What is the status of the DOJ antitrust lawsuit against Apple?"
    r3b = await run_query(q3b, "test_doc")
    print("Route:", r3b.get("route_taken"))
    print("Answer:\n", r3b.get("answer"))
    print("\n" + "-"*50 + "\n")

    print("--- DELIVERABLE 4: OFF-TOPIC CODING GUARDRAIL REFUSAL ---")
    q4 = "Write me a Python script to sort a list."
    r4 = await run_query(q4, "test_doc")
    print("Route:", r4.get("route_taken"))
    print("Answer:\n", r4.get("answer"))
    print("\n" + "="*50)

if __name__ == "__main__":
    asyncio.run(main())
