from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import pdfplumber
import pymupdf
from PIL import Image

logger = logging.getLogger(__name__)

def parse_pdf(pdf_path: str, images_output_dir: str) -> dict[str, list[dict[str, Any]]]:
    """Extract text, tables, and images from a PDF file.
    
    Returns dict with keys 'pages', 'tables', 'images' containing
    the extracted content with page number metadata.
    """
    out_dir = Path(images_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pages_data = []
    tables_data = []
    images_data = []
    
    try:
        # Extract text and images using pymupdf
        with pymupdf.open(pdf_path) as doc:
            for page_index in range(len(doc)):
                page_num = page_index + 1
                page = doc.load_page(page_index)
                
                # Text extraction
                text = page.get_text()
                if text and text.strip():
                    pages_data.append({"page_num": page_num, "text": text.strip()})
                
                # Image extraction
                image_list = page.get_images()
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                        if not base_image:
                            continue
                        
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        colorspace = base_image.get("colorspace", 0)
                        
                        img_path = out_dir / f"{page_num}_{img_index}.png"
                        
                        with open(img_path, "wb") as f:
                            f.write(image_bytes)
                            
                        # Handle colorspace conversion
                        with Image.open(img_path) as img:
                            if img.mode == "CMYK" or colorspace == pymupdf.csCMYK:
                                img = img.convert("RGB")
                                img.save(img_path, "PNG")
                            elif image_ext != "png":
                                img.save(img_path, "PNG")
                                
                        images_data.append({
                            "page_num": page_num, 
                            "path": str(img_path.absolute())
                        })
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} on page {page_num}: {e}")
                        
        # Extract tables using pdfplumber (with individual page fallback)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_index, page in enumerate(pdf.pages):
                    page_num = page_index + 1
                    try:
                        tables = page.extract_tables()
                        for table in tables:
                            if not table:
                                continue
                            
                            cleaned_table = []
                            for row in table:
                                cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                                cleaned_table.append(cleaned_row)
                            
                            text_representation = "\n".join([" | ".join(row) for row in cleaned_table])
                            
                            if text_representation.strip():
                                tables_data.append({
                                    "page_num": page_num,
                                    "data": cleaned_table,
                                    "text": text_representation
                                })
                    except Exception as page_e:
                        logger.warning(f"Skipping table extraction on page {page_num}: {page_e}")
        except Exception as plumber_e:
            logger.warning(f"pdfplumber table extraction skipped: {plumber_e}")
                        
    except Exception as e:
        logger.error(f"Error parsing PDF {pdf_path}: {e}")
        raise
        
    return {
        "pages": pages_data,
        "tables": tables_data,
        "images": images_data
    }
