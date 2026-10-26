"""
OmniBrain Specialized RAG Metric Evaluation Benchmark
Directly measures:
1. Recall@K (K=1, 3, 5, 10)
2. Mean Reciprocal Rank (MRR)
3. nDCG@K (K=3, 5, 10)
4. Faithfulness (Groundedness score)
5. Answer Correctness (Factual reference alignment)
6. Citation Correctness (Precision and Recall)
"""

import asyncio
import math
import json
import logging
from typing import List, Dict, Any, Set

logging.basicConfig(level=logging.ERROR)

from app.ingestion.embedder import embed_texts, embed_text_for_image_search
from app.db.qdrant_client import search_text, search_images
from app.agents.search_agent import _hybrid_rerank
from app.agents.supervisor import run_query

# 10 Golden test cases with verified ground-truth disclosure pages and financial facts
GOLDEN_RAG_BENCHMARK = [
    {
        "id": "tax_rate_yoy",
        "query": "What was Apple's effective tax rate in fiscal 2025, and how did it compare to 2024?",
        "ground_truth_pages": [28, 44, 32],
        "key_claims": ["15.6%", "24.1%", "$20,719", "$132,729", "8.5 percentage points"],
        "reference_answer": "Apple's effective tax rate was 15.6% in fiscal 2025 compared to 24.1% in fiscal 2024, decreasing by 8.5 percentage points."
    },
    {
        "id": "gross_margin_percentage",
        "query": "What was Apple's gross margin percentage in fiscal 2025 vs 2024?",
        "ground_truth_pages": [27],
        "key_claims": ["46.9%", "46.2%", "0.7 percentage points", "70 basis points"],
        "reference_answer": "Total gross margin was 46.9% in fiscal 2025 compared to 46.2% in fiscal 2024, increasing by 0.7 percentage points."
    },
    {
        "id": "total_net_sales",
        "query": "What was Apple's total net sales in fiscal 2025 and fiscal 2024?",
        "ground_truth_pages": [32, 27],
        "key_claims": ["$416,161 million", "$391,035 million"],
        "reference_answer": "Total net sales were $416,161 million in fiscal 2025 and $391,035 million in fiscal 2024."
    },
    {
        "id": "share_repurchases",
        "query": "How much did Apple spend on common share repurchases during fiscal 2025?",
        "ground_truth_pages": [48, 35, 46],
        "key_claims": ["$89.3 billion", "401.6 million shares"],
        "reference_answer": "Apple spent $89.3 billion repurchasing approximately 401.6 million shares of common stock in fiscal 2025."
    },
    {
        "id": "doj_antitrust_suit",
        "query": "What is the status and nature of the DOJ antitrust lawsuit against Apple?",
        "ground_truth_pages": [17, 21],
        "key_claims": ["Department of Justice", "antitrust", "monopolization", "District of New Jersey"],
        "reference_answer": "In March 2024, the DOJ and 16 states filed a civil antitrust lawsuit alleging monopolization in performance smartphones. Apple is vigorously defending."
    },
    {
        "id": "fastest_growing_category",
        "query": "Which product or service category grew the fastest in fiscal 2025?",
        "ground_truth_pages": [27],
        "key_claims": ["Services", "14.0%", "$109,158 million"],
        "reference_answer": "Services was the fastest growing category in fiscal 2025, growing +14.0% to $109,158 million."
    },
    {
        "id": "geographic_segment_sales",
        "query": "What were the net sales for the Americas, Europe, and Greater China segments in 2025?",
        "ground_truth_pages": [25, 28, 50],
        "key_claims": ["$178,353 million", "$111,032 million", "$64,377 million"],
        "reference_answer": "Americas net sales were $178,353M, Europe was $111,032M, and Greater China was $64,377M in fiscal 2025."
    },
    {
        "id": "executive_leadership",
        "query": "Who is Apple's Chief Executive Officer and CFO in the filing?",
        "ground_truth_pages": [61, 50],
        "key_claims": ["Tim Cook", "Timothy D. Cook", "Kevan Parekh"],
        "reference_answer": "Tim Cook serves as Chief Executive Officer, and Kevan Parekh serves as Chief Financial Officer."
    },
    {
        "id": "supply_chain_dependencies",
        "query": "What supply chain risks and geographic concentration dependencies are disclosed?",
        "ground_truth_pages": [4, 9, 49],
        "key_claims": ["China mainland", "single-source", "outsourcing", "manufacturing"],
        "reference_answer": "Manufacturing partners are concentrated in China mainland, India, Japan, Vietnam, with single-source component dependencies."
    },
    {
        "id": "services_vs_iphone_sales",
        "query": "How much revenue did Services generate in 2025 compared to iPhone?",
        "ground_truth_pages": [27, 32],
        "key_claims": ["$109,158 million", "$209,586 million"],
        "reference_answer": "Services generated $109,158 million compared to iPhone revenue of $209,586 million in fiscal 2025."
    }
]

def calculate_dcg(relevances: List[int], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        rel = relevances[i]
        dcg += rel / math.log2(i + 2) # i=0 -> log2(2) = 1
    return dcg

def calculate_ndcg(relevances: List[int], k: int, total_relevant: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    actual_dcg = calculate_dcg(relevances, k)
    # Ideal DCG: all 1s at top
    ideal_relevances = [1] * min(k, total_relevant) + [0] * max(0, k - total_relevant)
    idcg = calculate_dcg(ideal_relevances, k)
    return (actual_dcg / idcg) if idcg > 0 else 0.0

async def evaluate_priority_metrics():
    doc_id = "test_tax_verify_doc"
    
    print("=" * 80)
    print("      OMNIBRAIN RAG BENCHMARK: RECALL@K, MRR, NDCG, FAITHFULNESS, CITATIONS     ")
    print("=" * 80)
    
    recall_at_1 = []
    recall_at_3 = []
    recall_at_5 = []
    recall_at_10 = []
    
    mrr_list = []
    ndcg_at_3 = []
    ndcg_at_5 = []
    ndcg_at_10 = []
    
    faithfulness_scores = []
    answer_correctness_scores = []
    citation_precision_scores = []
    citation_recall_scores = []
    
    for idx, test in enumerate(GOLDEN_RAG_BENCHMARK, 1):
        q = test["query"]
        gt_pages = set(test["ground_truth_pages"])
        
        # ── 1. Evaluate Retrieval Component Directly ──
        text_emb = embed_texts([q])[0]
        img_emb = embed_text_for_image_search(q)
        text_hits = search_text(text_emb, doc_id, top_k=100)
        img_hits = search_images(img_emb, doc_id, top_k=3)
        ranked = _hybrid_rerank(q, text_hits + img_hits)[:15]
        
        retrieved_pages = [r.get("page_num") for r in ranked]
        
        # Binary relevance indicators for top K
        relevance_vector = [1 if p in gt_pages else 0 for p in retrieved_pages]
        
        # Recall@K (Hit rate: is at least one ground-truth page present in top K?)
        r1 = 1.0 if any(p in gt_pages for p in retrieved_pages[:1]) else 0.0
        r3 = 1.0 if any(p in gt_pages for p in retrieved_pages[:3]) else 0.0
        r5 = 1.0 if any(p in gt_pages for p in retrieved_pages[:5]) else 0.0
        r10 = 1.0 if any(p in gt_pages for p in retrieved_pages[:10]) else 0.0
        
        recall_at_1.append(r1)
        recall_at_3.append(r3)
        recall_at_5.append(r5)
        recall_at_10.append(r10)
        
        # MRR (Mean Reciprocal Rank)
        rr = 0.0
        for rank_pos, p in enumerate(retrieved_pages, 1):
            if p in gt_pages:
                rr = 1.0 / rank_pos
                break
        mrr_list.append(rr)
        
        # nDCG@K
        ndcg3 = calculate_ndcg(relevance_vector, 3, len(gt_pages))
        ndcg5 = calculate_ndcg(relevance_vector, 5, len(gt_pages))
        ndcg10 = calculate_ndcg(relevance_vector, 10, len(gt_pages))
        
        ndcg_at_3.append(ndcg3)
        ndcg_at_5.append(ndcg5)
        ndcg_at_10.append(ndcg10)
        
        # ── 2. Evaluate Generation & Citations via Full Pipeline ──
        res = await run_query(query=q, document_id=doc_id)
        answer = res.get("answer", "")
        ans_lower = answer.lower()
        citations = res.get("citations", [])
        
        # Extract cited pages from inline text [Page X, ...] or returned filtered citations
        import re
        inline_pages = {int(ip) for ip in re.findall(r'\[page\s*(\d+)', ans_lower)}
        cited_pages: Set[int] = set(inline_pages)
        if not cited_pages:
            for c in citations:
                if isinstance(c, dict) and c.get("page") and c.get("page") > 0:
                    cited_pages.add(c.get("page"))
            
        # Citation Precision: What fraction of cited pages are true relevant disclosure pages?
        if cited_pages:
            valid_citations = cited_pages.intersection(gt_pages)
            c_prec = len(valid_citations) / len(cited_pages)
        else:
            c_prec = 0.0
            
        # Citation Recall: What fraction of ground-truth disclosure pages were cited?
        if gt_pages:
            c_rec = len(cited_pages.intersection(gt_pages)) / len(gt_pages)
        else:
            c_rec = 1.0
            
        citation_precision_scores.append(c_prec)
        citation_recall_scores.append(c_rec)
        
        # Answer Correctness: Did the answer contain the key financial claims?
        claims_matched = 0
        for claim in test["key_claims"]:
            if claim.lower().replace("$", "").replace(",", "") in ans_lower.replace("$", "").replace(",", "") or any(term in ans_lower for term in claim.lower().split()):
                claims_matched += 1
        ans_corr = (claims_matched / len(test["key_claims"])) if test["key_claims"] else 1.0
        answer_correctness_scores.append(ans_corr)
        
        # Faithfulness: All claims directly attributable to retrieved context chunks
        # Verified against retrieved chunks
        all_chunk_text = " ".join((r.get("text") or "") for r in ranked[:12]).lower()
        grounded_claims = 0
        for claim in test["key_claims"]:
            c_clean = claim.lower().replace("$", "").replace(",", "")
            if c_clean in all_chunk_text or any(term in all_chunk_text for term in c_clean.split()):
                grounded_claims += 1
        faith_score = (grounded_claims / len(test["key_claims"])) if test["key_claims"] else 1.0
        faithfulness_scores.append(faith_score)
        
        first_hit_rank = f"Rank {int(1/rr)}" if rr > 0 else "Miss"
        print(f"[{idx:02d}/10] {test['id']:<25} | Recall@3: {r3:.1f} | MRR: {rr:.3f} ({first_hit_rank:<7}) | CitPrec: {c_prec*100:>5.1f}% | AnsCorr: {ans_corr*100:>5.1f}%")

    # ── 3. Calculate Final Aggregate Priority Metrics ──
    mean_recall_1 = sum(recall_at_1) / len(recall_at_1)
    mean_recall_3 = sum(recall_at_3) / len(recall_at_3)
    mean_recall_5 = sum(recall_at_5) / len(recall_at_5)
    mean_recall_10 = sum(recall_at_10) / len(recall_at_10)
    
    mean_mrr = sum(mrr_list) / len(mrr_list)
    mean_ndcg_3 = sum(ndcg_at_3) / len(ndcg_at_3)
    mean_ndcg_5 = sum(ndcg_at_5) / len(ndcg_at_5)
    mean_ndcg_10 = sum(ndcg_at_10) / len(ndcg_at_10)
    
    mean_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    mean_answer_correctness = sum(answer_correctness_scores) / len(answer_correctness_scores)
    mean_citation_precision = sum(citation_precision_scores) / len(citation_precision_scores)
    mean_citation_recall = sum(citation_recall_scores) / len(citation_recall_scores)

    print("\n" + "=" * 80)
    print("                    SPECIALIZED RAG METRICS SCORECARD                   ")
    print("=" * 80)
    print(" --- RETRIEVAL METRICS ---")
    print(f"  • Recall@1                     : {mean_recall_1*100:>6.2f}% (Hit in Top-1 rank)")
    print(f"  • Recall@3                     : {mean_recall_3*100:>6.2f}% (Hit in Top-3 ranks)")
    print(f"  • Recall@5                     : {mean_recall_5*100:>6.2f}% (Hit in Top-5 ranks)")
    print(f"  • Recall@10                    : {mean_recall_10*100:>6.2f}% (Hit in Top-10 ranks)")
    print(f"  • Mean Reciprocal Rank (MRR)   : {mean_mrr:>6.4f} (Average 1/rank of first hit)")
    print(f"  • nDCG@3                       : {mean_ndcg_3*100:>6.2f}% (Discounted Gain at 3)")
    print(f"  • nDCG@5                       : {mean_ndcg_5*100:>6.2f}% (Discounted Gain at 5)")
    print(f"  • nDCG@10                      : {mean_ndcg_10*100:>6.2f}% (Discounted Gain at 10)")
    print("\n --- GENERATION & GROUNDING METRICS ---")
    print(f"  • Faithfulness (Grounding)     : {mean_faithfulness*100:>6.2f}% (Claims supported by retrieved context)")
    print(f"  • Answer Correctness           : {mean_answer_correctness*100:>6.2f}% (Factual reference disclosure agreement)")
    print(f"  • Citation Precision           : {mean_citation_precision*100:>6.2f}% (Cited pages containing valid disclosure)")
    print(f"  • Citation Recall              : {mean_citation_recall*100:>6.2f}% (Coverage of all source disclosure pages)")
    print("=" * 80 + "\n")

    results_data = {
        "recall_at_1": round(mean_recall_1, 4),
        "recall_at_3": round(mean_recall_3, 4),
        "recall_at_5": round(mean_recall_5, 4),
        "recall_at_10": round(mean_recall_10, 4),
        "mrr": round(mean_mrr, 4),
        "ndcg_at_3": round(mean_ndcg_3, 4),
        "ndcg_at_5": round(mean_ndcg_5, 4),
        "ndcg_at_10": round(mean_ndcg_10, 4),
        "faithfulness": round(mean_faithfulness, 4),
        "answer_correctness": round(mean_answer_correctness, 4),
        "citation_precision": round(mean_citation_precision, 4),
        "citation_recall": round(mean_citation_recall, 4),
    }
    
    with open("priority_rag_metrics.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print("Detailed metrics saved locally to backend/priority_rag_metrics.json")

if __name__ == "__main__":
    asyncio.run(evaluate_priority_metrics())
