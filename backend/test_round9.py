import asyncio, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from app.agents.supervisor import run_query

async def main():
    print("========================================================")
    print("ROUND 9 EVALUATION TEST SUITE")
    print("========================================================\n")

    print("--- DELIVERABLE 2: SEGMENT NET SALES (ALL 5 SEGMENTS GROUND TRUTH) ---")
    q2 = "What were Apple's net sales by segment in fiscal 2025 — Americas, Europe, Greater China, Japan, Rest of Asia Pacific?"
    r2 = await run_query(q2, "test_doc")
    print("Route:", r2.get("route_taken"))
    print("Answer:\n", r2.get("answer"))
    print("\n" + "-"*50 + "\n")

    print("--- DELIVERABLE 3A: SERVICES VS IPHONE REVENUE COMPARISON ---")
    q3a = "How much revenue did Services generate in fiscal 2025 compared to iPhone?"
    r3a = await run_query(q3a, "test_doc")
    print("Route:", r3a.get("route_taken"))
    print("Answer:\n", r3a.get("answer"))
    print("\n" + "-"*50 + "\n")

    print("--- DELIVERABLE 3B: SERVICES VS PRODUCTS PROFITABILITY ---")
    q3b = "Is Apple's Services business more profitable than its Products business?"
    r3b = await run_query(q3b, "test_doc")
    print("Route:", r3b.get("route_taken"))
    print("Answer:\n", r3b.get("answer"))
    print("\n" + "-"*50 + "\n")

    print("--- DELIVERABLE 4: FASTEST-GROWING PRODUCT CATEGORY ---")
    q4 = "Which product category grew the fastest in fiscal 2025?"
    r4 = await run_query(q4, "test_doc")
    print("Route:", r4.get("route_taken"))
    print("Answer:\n", r4.get("answer"))
    print("\n" + "="*50)

if __name__ == "__main__":
    asyncio.run(main())
