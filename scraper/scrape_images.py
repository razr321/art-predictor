#!/usr/bin/env python3
"""Download artwork images from Christie's for visual feature extraction.

Reads lots.csv and downloads images to data/images/{lot_id}.jpg.
Resumable — skips already-downloaded images.

Output: data/images/*.jpg
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.helpers import setup_logger, DATA_RAW, DATA_IMAGES

logger = setup_logger(__name__, "scrape_images.log")

LOTS_FILE = DATA_RAW / "lots.csv"
TARGET_SIZE = 512  # Resize to 512x512 max dimension
DELAY = 0.5  # Seconds between downloads
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def download_image(url: str, save_path: Path, max_dim: int = TARGET_SIZE) -> bool:
    """Download image, resize, and save as JPEG."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()

        img = Image.open(BytesIO(resp.content))
        img = img.convert("RGB")

        # Resize maintaining aspect ratio
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path, "JPEG", quality=90)
        return True

    except Exception as e:
        logger.warning(f"Failed to download {url[:80]}: {e}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("Downloading artwork images")
    logger.info("=" * 60)

    if not LOTS_FILE.exists():
        logger.error(f"Lots file not found: {LOTS_FILE}")
        logger.error("Run scrape_lots.py first!")
        sys.exit(1)

    df = pd.read_csv(LOTS_FILE)
    # Only rows with image URLs
    df = df[df["image_url"].notna() & (df["image_url"] != "")]
    logger.info(f"Found {len(df)} lots with image URLs")

    DATA_IMAGES.mkdir(parents=True, exist_ok=True)

    # Check what's already downloaded
    existing = {p.stem for p in DATA_IMAGES.glob("*.jpg")}
    to_download = df[~df["lot_id"].astype(str).isin(existing)]
    logger.info(f"Already downloaded: {len(existing)}, remaining: {len(to_download)}")

    success = 0
    failed = 0

    for _, row in tqdm(to_download.iterrows(), total=len(to_download), desc="Downloading images"):
        lot_id = str(row["lot_id"])
        url = row["image_url"]
        save_path = DATA_IMAGES / f"{lot_id}.jpg"

        if download_image(url, save_path):
            success += 1
        else:
            failed += 1

        time.sleep(DELAY)

    logger.info(f"\nDone! Downloaded: {success}, Failed: {failed}")
    logger.info(f"Total images in {DATA_IMAGES}: {len(list(DATA_IMAGES.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
