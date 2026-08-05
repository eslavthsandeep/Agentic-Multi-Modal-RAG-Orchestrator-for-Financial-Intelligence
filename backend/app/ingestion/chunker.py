from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

def chunk_document(parsed_doc: dict[str, list[dict[str, Any]]], chunk_size: int = 500, overlap: int = 50) -> list[dict[str, Any]]:
    """Split parsed document content into overlapping chunks for embedding."""
    chunks = []
    global_chunk_index = 0
    
    # Process text pages
    for page in parsed_doc.get("pages", []):
        page_num = page.get("page_num")
        text = page.get("text", "")
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if len(current_chunk.strip()) >= 20:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "page_num": page_num,
                        "chunk_index": global_chunk_index,
                        "source_type": "text"
                    })
                    global_chunk_index += 1
                
                # Start new chunk with overlap from previous chunk, snapping to nearest word boundary
                if len(current_chunk) > overlap:
                    raw_overlap = current_chunk[-overlap:]
                    first_space = raw_overlap.find(" ")
                    if first_space != -1 and first_space < len(raw_overlap) - 1:
                        overlap_text = raw_overlap[first_space + 1:]
                    else:
                        overlap_text = raw_overlap
                else:
                    overlap_text = current_chunk
                    
                current_chunk = overlap_text + sentence + " "
                
        if len(current_chunk.strip()) >= 20:
            chunks.append({
                "text": current_chunk.strip(),
                "page_num": page_num,
                "chunk_index": global_chunk_index,
                "source_type": "text"
            })
            global_chunk_index += 1

    # Process tables separately
    for table in parsed_doc.get("tables", []):
        text = table.get("text", "")
        if len(text.strip()) >= 20:
            chunks.append({
                "text": text,
                "page_num": table.get("page_num"),
                "chunk_index": global_chunk_index,
                "source_type": "table"
            })
            global_chunk_index += 1
            
    return chunks
