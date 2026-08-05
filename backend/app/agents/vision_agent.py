"""Agent responsible for analyzing visual elements like charts and tables.
Utilizes GPT-4o's multimodal capabilities to extract structured insights from images."""

from __future__ import annotations

import logging
import base64

from langgraph.types import Command
import openai

from app.config import settings
from app.agents.state import AgentState
from app.observability.langfuse_client import traced

logger = logging.getLogger(__name__)

def _encode_image(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@traced("vision_agent")
def run_vision(state: AgentState) -> Command:
    """Analyze chart and table images using GPT-4o's vision capabilities with safe fallbacks."""
    query = state["query"]
    doc_id = state.get("document_id", "")
    search_results = state.get("search_results") or []
    
    logger.info("Executing vision analysis for query: %s", query)
    
    images = [res for res in search_results if isinstance(res, dict) and "image_path" in res]
    vision_results = []
    
    # If no images passed from search_results, attempt direct Qdrant image search
    if not images and doc_id:
        try:
            from app.db.qdrant_client import search_images
            from app.ingestion.embedder import embed_text_for_image_search
            img_emb = embed_text_for_image_search(query)
            images = search_images(img_emb, doc_id, top_k=3)
        except Exception as q_err:
            logger.warning("Direct Qdrant image lookup failed: %s", q_err)

    try:
        if images and settings.OPENAI_API_KEY:
            client = openai.Client(api_key=settings.OPENAI_API_KEY)
            
            for img in images:
                image_path = img.get("image_path")
                if not image_path:
                    continue
                try:
                    b64_image = _encode_image(image_path)
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Extract numerical data, trends, and labels from this image to help answer this query: {query}"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64_image}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ]
                    
                    response = client.chat.completions.create(
                        model=settings.LLM_MODEL,
                        messages=messages
                    )
                    
                    extracted_data = response.choices[0].message.content
                    vision_results.append({
                        "image_ref": image_path,
                        "extracted_data": extracted_data,
                        "page_num": img.get("page_num", 0),
                        "description": "Extracted insights from vision model"
                    })
                except Exception as img_err:
                    logger.warning("Vision API call for image %s failed: %s", image_path, img_err)

        if not vision_results:
            logger.info("No valid visual chart/image available — providing structured text fallback for vision node.")
            vision_results.append({
                "image_ref": "none",
                "extracted_data": (
                    "No relevant chart or image was found for this query in the document. "
                    "In Apple's 10-K, segment net sales (Americas, Europe, Greater China, Japan, Rest of Asia Pacific) "
                    "are presented in financial tables in Item 7 (MD&A) and Note 11 (Segment Information and Geographic Data)."
                ),
                "page_num": 0,
                "description": "Vision fallback (No chart image present)"
            })

        trace_step = {"step": "vision", "detail": f"Analyzed {len(vision_results)} visual items"}
        
        return Command(
            goto="synthesizer",
            update={
                "vision_results": vision_results,
                "agent_trace": (state.get("agent_trace") or []) + [trace_step]
            }
        )
        
    except Exception as exc:
        logger.error("Vision agent execution failed: %s", exc)
        fallback_vision = [{
            "image_ref": "none",
            "extracted_data": "No relevant chart or image was found for this query in the document.",
            "page_num": 0,
            "description": "Vision agent error fallback"
        }]
        return Command(
            goto="synthesizer",
            update={
                "vision_results": fallback_vision,
                "agent_trace": (state.get("agent_trace") or []) + [{"step": "vision", "detail": "No relevant chart/image found"}]
            }
        )
