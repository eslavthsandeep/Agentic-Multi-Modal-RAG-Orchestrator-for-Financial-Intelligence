"""Embedding module providing text and image vector generation.

Three distinct embedding paths serve different parts of the retrieval pipeline:
- embed_texts(): OpenAI text-embedding-3-small (1536-dim) for text chunk storage
- embed_images(): CLIP image encoder (512-dim) for image storage in Qdrant
- embed_text_for_image_search(): CLIP text encoder (512-dim) for querying images

The CLIP text and image encoders share a latent space, so text queries embedded
via CLIP can be compared against CLIP image embeddings. OpenAI text embeddings
are 1536-dim and live in a completely different space — using them against CLIP
images would be dimensionally and semantically meaningless.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# Lazy-loaded CLIP components
_clip_model = None
_clip_processor = None
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

def _load_clip() -> tuple[Any, Any]:
    global _clip_model, _clip_processor
    if _clip_model is None:
        logger.info(f"Loading CLIP model {CLIP_MODEL_ID} (first call, ~600MB download)")
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        _clip_model.eval()
    return _clip_model, _clip_processor

import hashlib
import random

def _generate_synthetic_vector(text: str, dim: int = 1536) -> list[float]:
    """Generate a deterministic 1536-dim unit vector based on text hash for offline/demo fallback."""
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    raw = [rng.gauss(0, 1) for _ in range(dim)]
    norm = sum(x**2 for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts using OpenAI's text-embedding-3-small, with synthetic fallback if quota is exceeded."""
    if not settings.OPENAI_API_KEY:
        logger.warning("No OpenAI API key provided — using synthetic embeddings")
        return [_generate_synthetic_vector(t) for t in texts]

    client = OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=0)
    embeddings = []
    
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            response = client.embeddings.create(
                input=batch,
                model=settings.EMBEDDING_MODEL
            )
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.warning(f"OpenAI embedding call failed ({e}). Falling back to synthetic vector generation for demo resilience.")
            fallback_batch = [_generate_synthetic_vector(t) for t in batch]
            embeddings.extend(fallback_batch)
            
    return embeddings

def embed_images(image_paths: list[str]) -> list[list[float]]:
    """Embed images using CLIP image encoder, with synthetic fallback if CLIP fails."""
    if not image_paths:
        return []
        
    try:
        model, processor = _load_clip()
        images = [Image.open(path).convert("RGB") for path in image_paths]
        inputs = processor(images=images, return_tensors="pt")
        
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            if hasattr(features, "image_embeds"):
                features = features.image_embeds
            elif not isinstance(features, torch.Tensor):
                features = features[0]
            # L2 normalize
            norm_features = F.normalize(features, p=2, dim=-1)
            
        return norm_features.tolist()
    except Exception as e:
        logger.warning(f"CLIP image embedding failed ({e}) — using fallback 512d vectors")
        return [_generate_synthetic_vector(path, dim=512) for path in image_paths]

def embed_text_for_image_search(text: str) -> list[float]:
    """Embed text query using CLIP text encoder for image search, with synthetic fallback."""
    try:
        model, processor = _load_clip()
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            features = model.get_text_features(**inputs)
            if hasattr(features, "text_embeds"):
                features = features.text_embeds
            elif not isinstance(features, torch.Tensor):
                features = features[0]
            # L2 normalize
            norm_features = F.normalize(features, p=2, dim=-1)
            
        return norm_features[0].tolist()
    except Exception as e:
        logger.warning(f"CLIP text embedding failed ({e}) — using fallback 512d vector")
        return _generate_synthetic_vector(text, dim=512)
