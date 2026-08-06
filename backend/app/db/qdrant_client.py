from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from app.config import settings

logger = logging.getLogger(__name__)

_client = None

def get_qdrant_client() -> QdrantClient:
    """Return a singleton Qdrant client.

    Tries connecting to the configured URL first (Docker/cloud Qdrant).
    Falls back to in-memory storage when the server is unreachable — useful
    for local dev without Docker.
    """
    global _client
    if _client is not None:
        return _client

    try:
        candidate = QdrantClient(url=settings.QDRANT_URL, timeout=3)
        candidate.get_collections()  # quick connectivity check
        _client = candidate
        logger.info("Connected to Qdrant at %s", settings.QDRANT_URL)
    except Exception:
        qdrant_dir = settings.DATA_DIR / "qdrant_db"
        qdrant_dir.mkdir(parents=True, exist_ok=True)
        try:
            _client = QdrantClient(path=str(qdrant_dir))
        except Exception:
            logger.warning("Qdrant disk DB locked by concurrent process — using fallback in-memory client")
            _client = QdrantClient(":memory:")

    return _client

def ensure_collections() -> None:
    """Ensure required collections exist in Qdrant."""
    client = get_qdrant_client()
    
    collections = [col.name for col in client.get_collections().collections]
    
    if "omnibrain_text" not in collections:
        logger.info("Creating collection 'omnibrain_text'")
        client.create_collection(
            collection_name="omnibrain_text",
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )
        
    if "omnibrain_images" not in collections:
        logger.info("Creating collection 'omnibrain_images'")
        client.create_collection(
            collection_name="omnibrain_images",
            vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE)
        )

def upsert_text_chunks(doc_id: str, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
    """Upsert text chunks and their embeddings into the text collection."""
    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks and embeddings must match")
        
    client = get_qdrant_client()
    points = []
    
    for chunk, embedding in zip(chunks, embeddings):
        point_id = str(uuid.uuid4())
        payload = {
            "document_id": doc_id,
            "page_num": chunk.get("page_num"),
            "chunk_index": chunk.get("chunk_index"),
            "text": chunk.get("text"),
            "source_type": chunk.get("source_type")
        }
        
        points.append(
            models.PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
        )
        
    if points:
        client.upsert(
            collection_name="omnibrain_text",
            points=points
        )

def upsert_image_embeddings(doc_id: str, images: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
    """Upsert image metadata and embeddings into the images collection."""
    if len(images) != len(embeddings):
        raise ValueError("Number of images and embeddings must match")
        
    client = get_qdrant_client()
    points = []
    
    for img, embedding in zip(images, embeddings):
        point_id = str(uuid.uuid4())
        payload = {
            "document_id": doc_id,
            "page_num": img.get("page_num"),
            "image_path": img.get("path")
        }
        
        points.append(
            models.PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
        )
        
    if points:
        client.upsert(
            collection_name="omnibrain_images",
            points=points
        )

def search_text(query_embedding: list[float], doc_id: str, top_k: int = 500) -> list[dict[str, Any]]:
    """Search for candidate text chunks across the document for hybrid reranking."""
    client = get_qdrant_client()
    results = []
    
    try:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=doc_id))]
        ) if doc_id else None
        
        # Retrieve document chunks via scroll to ensure no key section is missed
        scroll_res, _ = client.scroll(
            collection_name="omnibrain_text",
            scroll_filter=query_filter,
            limit=top_k
        )
        
        # Fallback to un-filtered scroll if doc_id mismatch returns 0 results
        if not scroll_res and doc_id:
            logger.info("Filtered scroll with doc_id '%s' returned 0 points; retrying un-filtered scroll", doc_id)
            scroll_res, _ = client.scroll(
                collection_name="omnibrain_text",
                scroll_filter=None,
                limit=top_k
            )

        results = [
            {
                "text": pt.payload.get("text"),
                "page_num": pt.payload.get("page_num"),
                "score": 0.5,
                "chunk_index": pt.payload.get("chunk_index")
            }
            for pt in scroll_res
        ]
    except Exception as e:
        logger.warning(f"Scroll text search failed ({e}) — falling back to query_points")
        try:
            q_res = client.query_points(
                collection_name="omnibrain_text",
                query=query_embedding,
                query_filter=query_filter,
                limit=top_k
            )
            results = [
                {
                    "text": point.payload.get("text"),
                    "page_num": point.payload.get("page_num"),
                    "score": point.score,
                    "chunk_index": point.payload.get("chunk_index")
                }
                for point in q_res.points
            ]
        except Exception as q_err:
            logger.warning(f"query_points text search failed: {q_err}")

    return results

def search_images(query_embedding: list[float], doc_id: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Search for most similar images using CLIP text embeddings."""
    client = get_qdrant_client()
    results = []
    
    try:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=doc_id))]
        ) if doc_id else None
        
        q_res = client.query_points(
            collection_name="omnibrain_images",
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k
        )
        results = [
            {
                "image_path": point.payload.get("image_path"),
                "page_num": point.payload.get("page_num"),
                "score": point.score
            }
            for point in q_res.points
        ]
    except Exception as e:
        logger.warning(f"query_points image search failed: {e}")
        
    return results
