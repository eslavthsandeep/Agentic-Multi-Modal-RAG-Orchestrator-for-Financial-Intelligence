from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, UploadFile

from app.config import settings
from app.db.qdrant_client import ensure_collections, upsert_image_embeddings, upsert_text_chunks
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import embed_images, embed_texts
from app.ingestion.pdf_parser import parse_pdf

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory document status tracking
_document_status: dict[str, dict] = {}

def _run_ingestion(doc_id: str, pdf_path: str) -> None:
    """Background task: parse, chunk, embed, store."""
    try:
        _document_status[doc_id] = {"status": "processing", "pages_processed": 0}
        ensure_collections()
        
        # 1. parse_pdf
        images_out_dir = settings.EXTRACTED_IMAGES_DIR / doc_id
        parsed_doc = parse_pdf(pdf_path, str(images_out_dir))
        num_pages = len(parsed_doc.get("pages", []))
        _document_status[doc_id]["pages_processed"] = num_pages
        
        # 2. chunk_document
        chunks = chunk_document(
            parsed_doc, 
            chunk_size=settings.CHUNK_SIZE, 
            overlap=settings.CHUNK_OVERLAP
        )
        
        # 3. embed and 4. upsert texts
        if chunks:
            texts = [c["text"] for c in chunks]
            text_embeddings = embed_texts(texts)
            upsert_text_chunks(doc_id, chunks, text_embeddings)
            
        # 3. embed and 4. upsert images (isolated try-except)
        images = parsed_doc.get("images", [])
        if images:
            try:
                image_paths = [img["path"] for img in images]
                image_embeddings = embed_images(image_paths)
                upsert_image_embeddings(doc_id, images, image_embeddings)
            except Exception as img_err:
                logger.warning(f"Image embedding skipped for {doc_id}: {img_err}")
            
        # 5. Update status
        _document_status[doc_id] = {
            "status": "ready",
            "pages_processed": num_pages,
            "images_extracted": len(images),
            "stats": {
                "chunks_processed": len(chunks),
                "images_processed": len(images)
            }
        }
        logger.info(f"Successfully ingested document {doc_id}")
        
    except Exception as e:
        logger.error(f"Failed to ingest document {doc_id}: {e}")
        _document_status[doc_id] = {
            "status": "failed",
            "pages_processed": 0,
            "error": str(e)
        }

@router.post("/api/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process a PDF document asynchronously."""
    doc_id = str(uuid.uuid4())[:12]
    
    upload_dir = settings.DATA_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = upload_dir / f"{doc_id}.pdf"
    
    try:
        with open(pdf_path, "wb") as f:
            f.write(await file.read())
            
        _document_status[doc_id] = {"status": "processing"}
        background_tasks.add_task(_run_ingestion, doc_id, str(pdf_path))
        
        return {"document_id": doc_id, "status": "processing"}
    except Exception as e:
        logger.error(f"Error handling upload: {e}")
        return {"error": str(e)}

@router.get("/api/upload/{document_id}/status")
async def get_upload_status(document_id: str):
    """Get the ingestion status of an uploaded document."""
    status = _document_status.get(document_id, {"status": "not_found"})
    return status
