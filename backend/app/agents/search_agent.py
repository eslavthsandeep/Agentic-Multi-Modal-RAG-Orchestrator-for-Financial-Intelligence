"""Agent responsible for vector search across textual and image metadata embeddings.
Implements the core retrieval logic for conceptual and textual queries against Qdrant."""

from __future__ import annotations

import logging

from langgraph.types import Command
import openai

from app.config import settings
from app.agents.state import AgentState
from app.db.qdrant_client import search_text, search_images
from app.ingestion.embedder import embed_texts, embed_text_for_image_search
from app.observability.langfuse_client import traced

logger = logging.getLogger(__name__)

def _hybrid_rerank(query: str, items: list[dict]) -> list[dict]:
    """Re-rank candidate chunks by combining vector similarity with BM25 keyword/phrase matching."""
    query_lower = query.lower()
    stopwords = {"what", "was", "is", "are", "the", "in", "and", "how", "did", "it", "to", "compare", "of", "for", "a", "an", "as", "its", "this"}
    terms = [t.strip("?.,\"'") for t in query_lower.split() if t.strip("?.,\"'") not in stopwords and len(t.strip("?.,\"'")) > 1]
    
    # Extract dynamic 2-gram and 3-gram phrases from query
    words = [w.strip("?.,\"'") for w in query_lower.split() if len(w.strip("?.,\"'")) > 1]
    phrases = []
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if not all(w in stopwords for w in words[i:i+2]):
            phrases.append(bigram)
    for i in range(len(words) - 2):
        trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
        if not all(w in stopwords for w in words[i:i+3]):
            phrases.append(trigram)

    reranked = []
    for item in items:
        text = (item.get("text") or item.get("image_path") or "").lower()
        base_score = float(item.get("score", 0.5))
        
        phrase_boost = sum(0.25 for p in phrases if p in text)
        term_matches = sum(1 for t in terms if t in text)
        term_boost = (term_matches / len(terms)) * 0.40 if terms else 0.0
        
        final_score = round(min(1.0, base_score * 0.2 + phrase_boost + term_boost), 4)
        item_copy = dict(item)
        item_copy["score"] = final_score
        reranked.append(item_copy)

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked

@traced("search_agent")
def run_search(state: AgentState) -> Command:
    """Retrieve relevant text and image context from the vector store."""
    query = state["query"]
    doc_id = state["document_id"]
    
    logger.info("Executing vector search for query: %s", query)
    
    try:
        text_embeddings = embed_texts([query])
        text_query_embedding = text_embeddings[0]
        
        image_query_embedding = embed_text_for_image_search(query)
        
        text_results = search_text(text_query_embedding, doc_id, top_k=100)
        image_results = search_images(image_query_embedding, doc_id, top_k=3)
        
        # Log top 10 raw search results before filtering/reranking
        top10_raw = text_results[:10]
        raw_log = "\n".join(f"- [Page {r.get('page_num', '?')}] {r.get('text', '')[:120]}..." for r in top10_raw)
        logger.info("Top 10 raw retrieved chunks before reranking:\n%s", raw_log)

        raw_combined = text_results + image_results
        combined_results = _hybrid_rerank(query, raw_combined)[:5]
        
        max_score = max((res.get("score", 0.0) for res in combined_results), default=0.0)
        logger.info("Top hybrid retrieval score: %.4f for query '%s'", max_score, query)
        
        if max_score < settings.RELEVANCE_THRESHOLD and state.get("self_correction_attempts", 0) < settings.MAX_CORRECTION_ATTEMPTS:
            logger.info("Search relevance %.2f below threshold %.2f, attempting correction", max_score, settings.RELEVANCE_THRESHOLD)
            try:
                client = openai.Client(api_key=settings.OPENAI_API_KEY)
                rewrite_prompt = f"Rewrite this query to be more specific for financial document search: {query}"
                
                response = client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[{"role": "user", "content": rewrite_prompt}]
                )
                
                rewritten_query = response.choices[0].message.content or query
                logger.info("Rewritten query: %s", rewritten_query)
                
                return Command(
                    goto="search_agent",
                    update={
                        "query": rewritten_query,
                        "self_correction_attempts": state.get("self_correction_attempts", 0) + 1
                    }
                )
            except Exception as rewrite_err:
                logger.warning(f"Query rewrite skipped due to LLM error ({rewrite_err}). Proceeding with current search results.")
            
        trace_step = {"step": "search", "detail": f"Retrieved {len(combined_results)} items (Max score: {max_score:.2f})"}
        
        next_node = "synthesizer"
        if state.get("route_decision") == "multi":
            next_node = "sql_agent"
            
        return Command(
            goto=next_node,
            update={
                "search_results": combined_results,
                "agent_trace": state.get("agent_trace", []) + [trace_step]
            }
        )
        
    except Exception as exc:
        logger.error("Search agent execution failed: %s", exc)
        return Command(
            goto="synthesizer",
            update={
                "search_results": combined_results if 'combined_results' in locals() else [],
                "agent_trace": state.get("agent_trace", []) + [{"step": "search", "detail": f"Search fallback: {exc}"}]
            }
        )
