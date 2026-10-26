"""
OmniBrain RAG Evaluation Benchmark Suite
Evaluates the entire end-to-end RAG system across:
1. Guardrail Accuracy & Security
2. Supervisor Routing Precision
3. Hybrid Retrieval Quality (Hit Rate, MRR, Relevance Scores)
4. Factual Correctness & Citation Precision
5. SQL Agent Security & Execution Correctness
6. End-to-End Latency & Throughput
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List

# Suppress noisy logs during eval
logging.basicConfig(level=logging.ERROR)

from app.guardrails.guard import check_input
from app.agents.sql_agent import _validate_sql
from app.db.sql_client import execute_query
from app.agents.supervisor import run_query
from app.agents.search_agent import run_search
from app.agents.state import AgentState

# Golden benchmark evaluation dataset
EVAL_DATASET = [
    # ── Category 1: SEC 10-K Document Text Queries ──
    {
        "id": "doc_tax_rate",
        "category": "Document QA",
        "query": "What was Apple's effective tax rate in fiscal 2025, and how did it compare to 2024?",
        "expected_route": "search",
        "expected_pages": [28, 44, 32],
        "expected_facts": ["15.6", "24.1", "20,719", "132,729", "decreased"],
        "is_on_topic": True
    },
    {
        "id": "doc_gross_margin",
        "category": "Document QA",
        "query": "What was Apple's gross margin percentage in fiscal 2025 vs 2024?",
        "expected_route": "search",
        "expected_pages": [27],
        "expected_facts": ["46.9", "46.2", "0.7", "products", "services"],
        "is_on_topic": True
    },
    {
        "id": "doc_net_sales",
        "category": "Document QA",
        "query": "What was Apple's total net sales in fiscal 2025 and fiscal 2024?",
        "expected_route": "search",
        "expected_pages": [32, 27],
        "expected_facts": ["416,161", "391,035"],
        "is_on_topic": True
    },
    {
        "id": "doc_services_iphone",
        "category": "Document QA",
        "query": "How much revenue did Services generate in 2025 compared to iPhone?",
        "expected_route": "search",
        "expected_pages": [27, 32],
        "expected_facts": ["109,158", "209,586"],
        "is_on_topic": True
    },
    {
        "id": "doc_fastest_growth",
        "category": "Document QA",
        "query": "Which category grew the fastest in fiscal 2025?",
        "expected_route": "search",
        "expected_pages": [27],
        "expected_facts": ["services", "14"],
        "is_on_topic": True
    },
    {
        "id": "doc_repurchases",
        "category": "Document QA",
        "query": "How much did Apple spend on share repurchases in fiscal 2025?",
        "expected_route": "search",
        "expected_pages": [48, 35, 46],
        "expected_facts": ["89", "401"],
        "is_on_topic": True
    },
    {
        "id": "doc_doj_lawsuit",
        "category": "Document QA",
        "query": "What is the status of the DOJ antitrust lawsuit against Apple?",
        "expected_route": "search",
        "expected_pages": [17, 21],
        "expected_facts": ["department of justice", "antitrust", "monopoliz"],
        "is_on_topic": True
    },
    {
        "id": "doc_segments",
        "category": "Document QA",
        "query": "What was the segment net sales for Americas and Greater China in 2025?",
        "expected_route": "search",
        "expected_pages": [28, 50],
        "expected_facts": ["178,353", "64,377", "americas"],
        "is_on_topic": True
    },
    {
        "id": "doc_ceo",
        "category": "Document QA",
        "query": "Who is Apple's Chief Executive Officer and CFO?",
        "expected_route": "search",
        "expected_pages": [61, 50],
        "expected_facts": ["cook", "tim"],
        "is_on_topic": True
    },
    {
        "id": "doc_supply_chain",
        "category": "Document QA",
        "query": "What supply chain risks and outsourcing dependencies are described in the filing?",
        "expected_route": "search",
        "expected_pages": [4, 49],
        "expected_facts": ["supplier", "manufactur", "risk"],
        "is_on_topic": True
    },

    # ── Category 2: Structured SQL Stock Queries ──
    {
        "id": "sql_avg_close",
        "category": "Structured SQL",
        "query": "What was AAPL's average closing price in the stock database?",
        "expected_route": "sql",
        "expected_pages": [],
        "expected_facts": ["215", "avg", "closing price"],
        "is_on_topic": True
    },
    {
        "id": "sql_peak_volume",
        "category": "Structured SQL",
        "query": "What was the highest trading volume date recorded for AAPL?",
        "expected_route": "sql",
        "expected_pages": [],
        "expected_facts": ["volume", "132", "2026"],
        "is_on_topic": True
    },

    # ── Category 3: Multi-Agent Chained Queries ──
    {
        "id": "multi_sales_vs_stock",
        "category": "Multi-Agent",
        "query": "What was Apple's total net sales in fiscal 2025, and how does that compare to the stock price trend?",
        "expected_route": "multi",
        "expected_pages": [32],
        "expected_facts": ["416,161", "stock", "close_price"],
        "is_on_topic": True
    },

    # ── Category 4: Vision / Chart Queries ──
    {
        "id": "vision_chart",
        "category": "Vision Agent",
        "query": "Show me the segment revenue breakdown chart and visual figures",
        "expected_route": "vision",
        "expected_pages": [],
        "expected_facts": ["segment", "net sales"],
        "is_on_topic": True
    },

    # ── Category 5: Out of Scope / Missing Future Data ──
    {
        "id": "oos_future_data",
        "category": "Hallucination Defense",
        "query": "What are Apple's projected net sales for fiscal 2030 in the metaverse?",
        "expected_route": "search",
        "expected_pages": [],
        "expected_facts": ["couldn't find", "future period", "fiscal 2030"],
        "is_on_topic": True
    },

    # ── Category 6: Off-Topic Guardrail Queries (Should be blocked) ──
    {
        "id": "guard_coding",
        "category": "Guardrail Defense",
        "query": "Write a Python script to sort a list using quicksort algorithm",
        "expected_route": "guardrails",
        "expected_pages": [],
        "expected_facts": ["outside the scope"],
        "is_on_topic": False
    },
    {
        "id": "guard_ml",
        "category": "Guardrail Defense",
        "query": "Explain how transformer self-attention and deep learning backpropagation work",
        "expected_route": "guardrails",
        "expected_pages": [],
        "expected_facts": ["outside the scope"],
        "is_on_topic": False
    },
    {
        "id": "guard_politics",
        "category": "Guardrail Defense",
        "query": "Who is the president of France and how do elections work there?",
        "expected_route": "guardrails",
        "expected_pages": [],
        "expected_facts": ["outside the scope"],
        "is_on_topic": False
    },
    {
        "id": "guard_sports",
        "category": "Guardrail Defense",
        "query": "Who won the FIFA World Cup and the latest cricket championship?",
        "expected_route": "guardrails",
        "expected_pages": [],
        "expected_facts": ["outside the scope"],
        "is_on_topic": False
    },
    {
        "id": "guard_cooking",
        "category": "Guardrail Defense",
        "query": "Give me a step-by-step recipe to cook homemade pizza and pasta",
        "expected_route": "guardrails",
        "expected_pages": [],
        "expected_facts": ["outside the scope"],
        "is_on_topic": False
    }
]

# SQL injection attack vectors
SQL_SECURITY_TESTS = [
    "SELECT * FROM stock_history; DROP TABLE stock_history;",
    "DROP TABLE stock_history;",
    "DELETE FROM stock_history WHERE ticker='AAPL';",
    "INSERT INTO stock_history (ticker, date, close_price) VALUES ('FAKE', '2025-01-01', 999.0);",
    "UPDATE stock_history SET close_price = 0;",
    "ALTER TABLE stock_history ADD COLUMN hacked TEXT;",
    "TRUNCATE TABLE stock_history;",
    "SELECT date, close_price FROM stock_history WHERE ticker = 'AAPL' LIMIT 5;"  # legitimate
]

async def evaluate_system():
    print("================================================================================")
    print("                OMNIBRAIN AGENTIC RAG SYSTEM EVALUATION BENCHMARK               ")
    print("================================================================================")
    print(f"Total Test Cases: {len(EVAL_DATASET)}")
    print(f"SQL Security Tests: {len(SQL_SECURITY_TESTS)}\n")

    doc_id = "test_tax_verify_doc"
    
    results = []
    latencies = []
    
    # ── 1. Run Benchmark Queries ──
    print("Running Benchmark Queries across Agents and Guardrails...")
    for idx, test in enumerate(EVAL_DATASET, 1):
        q = test["query"]
        t_start = time.perf_counter()
        
        try:
            res = await run_query(query=q, document_id=doc_id)
            elapsed = time.perf_counter() - t_start
            latencies.append(elapsed)
            
            answer = res.get("answer", "")
            route_taken = res.get("route_taken", ["unknown"])[0] if res.get("route_taken") else "unknown"
            guardrail_status = res.get("guardrail_status", "unknown")
            citations = res.get("citations", [])
            cited_pages = [c.get("page") for c in citations if isinstance(c, dict) and c.get("page")]
            
            # Check guardrail outcome
            is_blocked = (guardrail_status == "blocked") or ("outside the scope" in answer.lower())
            guardrail_correct = (not test["is_on_topic"]) == is_blocked
            
            # Check routing outcome
            if not test["is_on_topic"]:
                route_correct = is_blocked or route_taken in ["guardrails", "end"]
            else:
                route_correct = (route_taken == test["expected_route"])
                
            # Check factual correctness
            facts_matched = 0
            ans_lower = answer.lower()
            for fact in test["expected_facts"]:
                if fact.lower() in ans_lower:
                    facts_matched += 1
            fact_score = (facts_matched / len(test["expected_facts"])) if test["expected_facts"] else 1.0
            
            # Check citation accuracy
            citation_hit = False
            if test["expected_pages"]:
                citation_hit = any(p in test["expected_pages"] for p in cited_pages) or any(f"[page {p}" in ans_lower for p in test["expected_pages"])
            else:
                citation_hit = True # Not required for off-topic or SQL
                
            results.append({
                "id": test["id"],
                "category": test["category"],
                "query": q,
                "route_expected": test["expected_route"],
                "route_predicted": route_taken,
                "route_correct": route_correct,
                "guardrail_correct": guardrail_correct,
                "fact_score": fact_score,
                "citation_hit": citation_hit,
                "latency_sec": elapsed,
                "is_on_topic": test["is_on_topic"],
                "is_blocked": is_blocked
            })
            
            print(f"  [{idx:02d}/{len(EVAL_DATASET):02d}] {test['id']:<20} | Route: {route_taken:<10} | Facts: {fact_score*100:>5.1f}% | Time: {elapsed:>5.2f}s")
            
        except Exception as e:
            print(f"  [{idx:02d}/{len(EVAL_DATASET):02d}] {test['id']:<20} | ERROR: {e}")
            results.append({
                "id": test["id"],
                "category": test["category"],
                "query": q,
                "route_correct": False,
                "guardrail_correct": False,
                "fact_score": 0.0,
                "citation_hit": False,
                "latency_sec": 0.0
            })

    # ── 2. Run SQL Security Evaluation ──
    print("\nEvaluating SQL Injection Prevention Guard (_validate_sql)...")
    sql_blocked_count = 0
    sql_total_malicious = len(SQL_SECURITY_TESTS) - 1
    
    for sql in SQL_SECURITY_TESTS:
        is_malicious = any(k in sql.upper() for k in ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'TRUNCATE'])
        try:
            valid_sql = _validate_sql(sql)
            if is_malicious:
                print(f"  [FAIL] Malicious query passed validation: {sql}")
            else:
                print(f"  [PASS] Legitimate SELECT passed validation: {valid_sql}")
        except ValueError:
            if is_malicious:
                sql_blocked_count += 1
                print(f"  [PASS] Successfully blocked malicious statement: {sql[:45]}...")
            else:
                print(f"  [FAIL] Legitimate query incorrectly blocked: {sql}")

    # ── 3. Calculate Aggregated Metrics ──
    total_queries = len(results)
    
    # Guardrail metrics
    true_positives = sum(1 for r in results if not r["is_on_topic"] and r["is_blocked"])  # correctly blocked off-topic
    true_negatives = sum(1 for r in results if r["is_on_topic"] and not r["is_blocked"]) # correctly allowed on-topic
    false_positives = sum(1 for r in results if r["is_on_topic"] and r["is_blocked"])     # incorrectly blocked financial
    false_negatives = sum(1 for r in results if not r["is_on_topic"] and not r["is_blocked"]) # allowed off-topic
    
    guardrail_accuracy = (true_positives + true_negatives) / total_queries
    guardrail_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 1.0
    guardrail_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 1.0
    
    # Route accuracy
    route_accuracy = sum(1 for r in results if r["route_correct"]) / total_queries
    
    # Fact accuracy on on-topic queries
    on_topic_results = [r for r in results if r["is_on_topic"]]
    avg_fact_accuracy = sum(r["fact_score"] for r in on_topic_results) / len(on_topic_results)
    
    # Citation Hit Rate
    doc_qa_results = [r for r in results if r["category"] == "Document QA"]
    citation_hit_rate = sum(1 for r in doc_qa_results if r["citation_hit"]) / len(doc_qa_results) if doc_qa_results else 1.0
    
    # SQL Security Rate
    sql_security_rate = sql_blocked_count / sql_total_malicious
    
    # Latency Stats
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    sorted_latencies = sorted(latencies)
    median_latency = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 0.0
    p95_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)] if sorted_latencies else 0.0

    print("\n" + "="*80)
    print("                    EVALUATION SCORECARD RESULTS SUMMARY                    ")
    print("="*80)
    print(f" 1. Guardrail Off-Topic Defense Accuracy  : {guardrail_accuracy*100:>6.2f}% ({true_positives+true_negatives}/{total_queries})")
    print(f"    - Precision (Off-topic detection)     : {guardrail_precision*100:>6.2f}%")
    print(f"    - Recall (Off-topic catch rate)       : {guardrail_recall*100:>6.2f}%")
    print(f"    - False Positive Rate (Blocked valid) : {(false_positives/total_queries)*100:>6.2f}%")
    print(f" 2. Supervisor Route Classification Acc   : {route_accuracy*100:>6.2f}% ({sum(1 for r in results if r['route_correct'])}/{total_queries})")
    print(f" 3. Document QA Citation Page Hit Rate    : {citation_hit_rate*100:>6.2f}% ({sum(1 for r in doc_qa_results if r['citation_hit'])}/{len(doc_qa_results)})")
    print(f" 4. Factual Knowledge Extraction Accuracy : {avg_fact_accuracy*100:>6.2f}% (Average across all facts)")
    print(f" 5. SQL Injection Attack Defense Rate     : {sql_security_rate*100:>6.2f}% ({sql_blocked_count}/{sql_total_malicious} blocked)")
    print(f" 6. Latency Benchmark:")
    print(f"    - Mean Query Response Time            : {avg_latency:>6.2f}s")
    print(f"    - Median (P50) Response Time          : {median_latency:>6.2f}s")
    print(f"    - 95th Percentile (P95) Response Time : {p95_latency:>6.2f}s")
    print("="*80 + "\n")

    # Output JSON summary report
    summary = {
        "guardrail_accuracy": round(guardrail_accuracy, 4),
        "guardrail_precision": round(guardrail_precision, 4),
        "guardrail_recall": round(guardrail_recall, 4),
        "route_accuracy": round(route_accuracy, 4),
        "citation_hit_rate": round(citation_hit_rate, 4),
        "factual_accuracy": round(avg_fact_accuracy, 4),
        "sql_security_rate": round(sql_security_rate, 4),
        "mean_latency_sec": round(avg_latency, 3),
        "median_latency_sec": round(median_latency, 3),
        "p95_latency_sec": round(p95_latency, 3),
        "total_test_cases": total_queries,
        "details": results
    }
    with open("eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Detailed report saved to backend/eval_results.json")

if __name__ == "__main__":
    asyncio.run(evaluate_system())
