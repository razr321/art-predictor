#!/usr/bin/env python3
"""Extract visual features from downloaded artwork images using CLIP.

For each image in data/images/, extracts:
  - CLIP embedding (512-dim) and text-similarity classifications for
    subject, color palette, and style
  - Simple color features: dominant color (RGB), color richness, brightness

Output: data/processed/image_features.csv

Requires: transformers, torch, Pillow, scikit-learn
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.helpers import setup_logger, DATA_PROCESSED, DATA_IMAGES

logger = setup_logger(__name__, "extract_image_features.log")

OUTPUT_FILE = DATA_PROCESSED / "image_features.csv"
BATCH_SIZE = 32

# ── Text prompts for zero-shot classification via CLIP ──────────────────────

SUBJECT_LABELS = [
    "portrait",
    "landscape",
    "abstract composition",
    "figurative scene",
    "still life",
    "religious scene",
    "nude figure",
    "animals",
    "cityscape",
    "geometric pattern",
]

PALETTE_LABELS = [
    "vibrant warm colors",
    "cool blue tones",
    "earth tones",
    "monochrome",
    "high contrast",
    "muted pastels",
]

STYLE_LABELS = [
    "expressionist",
    "realistic",
    "cubist",
    "impressionist",
    "minimalist",
    "folk art",
    "modernist abstract",
]


def load_clip_model():
    """Load CLIP model and processor. Downloads on first run (~600 MB)."""
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        logger.error(
            "transformers and torch are required. Install with:\n"
            "  pip install transformers torch"
        )
        sys.exit(1)

    logger.info("Loading CLIP model (openai/clip-vit-base-patch32)...")
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    logger.info("CLIP model loaded successfully")
    return model, processor


def compute_text_embeddings(model, processor, labels: list[str]):
    """Pre-compute text embeddings for a set of labels."""
    import torch

    prompts = [f"a painting of {label}" for label in labels]
    inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_out = model.text_model(**inputs)
        text_features = model.text_projection(text_out.pooler_output)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def classify_from_embedding(image_embedding, text_embeddings, labels: list[str]):
    """Classify an image embedding against pre-computed text embeddings.

    Returns (top_label, confidence_score).
    """
    import torch

    # image_embedding: (512,) tensor, text_embeddings: (N, 512) tensor
    similarities = (image_embedding @ text_embeddings.T).squeeze()
    # Apply softmax for interpretable confidence scores
    probs = torch.softmax(similarities * 100, dim=-1)  # temperature=100 (CLIP default logit_scale)
    top_idx = probs.argmax().item()
    return labels[top_idx], probs[top_idx].item()


def extract_color_features(img: Image.Image) -> dict:
    """Extract simple color features from a PIL image.

    Returns:
        dominant_color_r, dominant_color_g, dominant_color_b: RGB of dominant color
        color_richness: number of distinct color clusters (KMeans k=5)
        brightness: mean luminance (0-255)
    """
    from sklearn.cluster import KMeans

    # Dominant color: resize to 1x1
    tiny = img.resize((1, 1), Image.LANCZOS)
    r, g, b = tiny.getpixel((0, 0))

    # Brightness: convert to grayscale and get mean
    gray = img.convert("L")
    brightness = np.array(gray).mean()

    # Color richness: KMeans clustering on downsampled image
    small = img.resize((64, 64), Image.LANCZOS)
    pixels = np.array(small).reshape(-1, 3).astype(float)

    try:
        kmeans = KMeans(n_clusters=5, n_init=3, max_iter=100, random_state=42)
        kmeans.fit(pixels)
        # Color richness = measure of how spread out the clusters are
        # Use the average pairwise distance between cluster centers
        centers = kmeans.cluster_centers_
        dists = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dists.append(np.linalg.norm(centers[i] - centers[j]))
        color_richness = np.mean(dists) if dists else 0.0
    except Exception:
        color_richness = 0.0

    return {
        "dominant_color_r": int(r),
        "dominant_color_g": int(g),
        "dominant_color_b": int(b),
        "color_richness": round(float(color_richness), 2),
        "brightness": round(float(brightness), 2),
    }


def process_batch(
    image_paths: list[Path],
    model,
    processor,
    subject_text_emb,
    palette_text_emb,
    style_text_emb,
) -> list[dict]:
    """Process a batch of images: extract CLIP embeddings + color features."""
    import torch

    results = []
    valid_images = []
    valid_paths = []

    # Load images
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            valid_images.append(img)
            valid_paths.append(path)
        except Exception as e:
            logger.warning(f"Could not open {path.name}: {e}")

    if not valid_images:
        return results

    # Extract CLIP embeddings in batch
    try:
        inputs = processor(images=valid_images, return_tensors="pt")
        with torch.no_grad():
            vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
            image_features = model.visual_projection(vision_out.pooler_output)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    except Exception as e:
        logger.warning(f"CLIP batch processing failed: {e}")
        return results

    # Process each image
    for idx, (path, img) in enumerate(zip(valid_paths, valid_images)):
        lot_id = path.stem
        emb = image_features[idx]

        # Zero-shot classification
        subject, subject_score = classify_from_embedding(emb, subject_text_emb, SUBJECT_LABELS)
        palette, palette_score = classify_from_embedding(emb, palette_text_emb, PALETTE_LABELS)
        style, style_score = classify_from_embedding(emb, style_text_emb, STYLE_LABELS)

        # Color features
        color_feats = extract_color_features(img)

        # CLIP embedding as JSON string
        embedding_list = emb.cpu().numpy().tolist()
        clip_embedding_json = json.dumps([round(x, 6) for x in embedding_list])

        record = {
            "lot_id": lot_id,
            "subject": subject,
            "subject_score": round(subject_score, 4),
            "palette": palette,
            "palette_score": round(palette_score, 4),
            "style": style,
            "style_score": round(style_score, 4),
            **color_feats,
            "clip_embedding": clip_embedding_json,
        }
        results.append(record)

    return results


def main():
    logger.info("=" * 60)
    logger.info("Image Feature Extraction Pipeline")
    logger.info("=" * 60)

    # Check for images
    if not DATA_IMAGES.exists():
        logger.error(f"Image directory not found: {DATA_IMAGES}")
        logger.error("Run download_images.py first!")
        sys.exit(1)

    image_paths = sorted(DATA_IMAGES.glob("*.jpg"))
    # Exclude progress file and other non-image files
    image_paths = [p for p in image_paths if not p.name.startswith("_")]
    logger.info(f"Found {len(image_paths)} images to process")

    if not image_paths:
        logger.error("No images found!")
        sys.exit(1)

    # Load existing results to enable resuming
    existing_lot_ids = set()
    existing_records = []
    if OUTPUT_FILE.exists():
        existing_df = pd.read_csv(OUTPUT_FILE)
        existing_lot_ids = set(existing_df["lot_id"].astype(str))
        existing_records = existing_df.to_dict("records")
        logger.info(f"Found {len(existing_lot_ids)} already-processed images, resuming...")

    # Filter to unprocessed images
    remaining_paths = [p for p in image_paths if p.stem not in existing_lot_ids]
    logger.info(f"Remaining to process: {len(remaining_paths)}")

    if not remaining_paths:
        logger.info("All images already processed!")
        return

    # Load CLIP model
    model, processor = load_clip_model()

    # Pre-compute text embeddings for classification
    logger.info("Computing text embeddings for classification prompts...")
    subject_text_emb = compute_text_embeddings(model, processor, SUBJECT_LABELS)
    palette_text_emb = compute_text_embeddings(model, processor, PALETTE_LABELS)
    style_text_emb = compute_text_embeddings(model, processor, STYLE_LABELS)

    # Process in batches
    all_records = list(existing_records)
    total = len(remaining_paths)
    processed = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_paths = remaining_paths[batch_start:batch_end]

        batch_results = process_batch(
            batch_paths, model, processor,
            subject_text_emb, palette_text_emb, style_text_emb,
        )
        all_records.extend(batch_results)
        processed += len(batch_paths)

        # Print progress every 50 images
        if processed % 50 < BATCH_SIZE or batch_end == total:
            logger.info(f"Progress: {processed}/{total} images processed")

            # Save intermediate results
            df_out = pd.DataFrame(all_records)
            df_out.to_csv(OUTPUT_FILE, index=False)

    # Final save
    df_out = pd.DataFrame(all_records)
    df_out.to_csv(OUTPUT_FILE, index=False)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Feature extraction complete!")
    logger.info(f"  Total images processed: {len(df_out)}")
    logger.info(f"  Output: {OUTPUT_FILE}")
    logger.info(f"  Columns: {list(df_out.columns)}")

    # Quick stats
    logger.info(f"\n--- Classification Distribution ---")
    for col in ["subject", "palette", "style"]:
        if col in df_out.columns:
            logger.info(f"\n  {col}:")
            for label, count in df_out[col].value_counts().head(5).items():
                pct = count / len(df_out) * 100
                logger.info(f"    {label}: {count} ({pct:.1f}%)")

    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
