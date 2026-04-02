#!/usr/bin/env python3
"""Download artwork images for visual feature extraction.

Reads master.csv and downloads images to data/images/{lot_id}.jpg.
Resumable — skips already-downloaded images.
Handles multiple auction house URL formats:
  - Saffronart: direct image URLs (mediacloud.saffronart.com)
  - Sotheby's: direct image URLs (dam.sothebys.com)
  - Christie's: lot page URLs — attempts to extract image via known patterns
  - Pundoles: mostly placeholder SVGs — skipped

Output: data/images/*.jpg
"""

import json
import sys
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.helpers import setup_logger, DATA_PROCESSED, DATA_IMAGES

logger = setup_logger(__name__, "download_images.log")

MASTER_FILE = DATA_PROCESSED / "master.csv"
PROGRESS_FILE = DATA_IMAGES / "_download_progress.json"
TARGET_SIZE = 512  # Max dimension in pixels
DELAY = 0.5  # Seconds between downloads
TIMEOUT = 10  # Request timeout in seconds

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
}

# Known placeholder / non-image URLs to skip
SKIP_PATTERNS = [
    "cdn.userway.org",        # Pundoles placeholder SVGs
    "spin_wh.svg",
    "placeholder",
    ".svg",
]


def is_valid_image_url(url: str) -> bool:
    """Check if a URL is a plausible direct image URL (not a page URL or placeholder)."""
    if not url or not isinstance(url, str):
        return False
    url_lower = url.strip().lower()

    # Must start with http
    if not url_lower.startswith("http"):
        return False

    # Skip known placeholders
    for pattern in SKIP_PATTERNS:
        if pattern in url_lower:
            return False

    return True


def resolve_image_url(url: str) -> str | None:
    """Resolve the actual downloadable image URL based on source.

    - Saffronart URLs are direct image links — use as-is
    - Sotheby's URLs are direct image links — can request larger size
    - Christie's URLs point to lot pages, not images — we try a known
      image API pattern, but these may fail
    """
    if not is_valid_image_url(url):
        return None

    parsed = urlparse(url)
    host = parsed.netloc.lower()

    # Saffronart: direct image URL — use as-is
    if "saffronart.com" in host:
        return url

    # Sotheby's: direct image URL — can swap "Small" for "Medium" for better quality
    if "sothebys.com" in host:
        # URL format: .../primary/Small -> try Medium for better quality, fall back to Small
        return url.replace("/Small", "/Medium")

    # Christie's: lot page URL like https://www.christies.com/en/lot/lot-5417046
    # The actual images are served from a CDN. Try the known image API pattern.
    if "christies.com" in host:
        # Extract lot number from URL
        if "/lot/lot-" in url:
            lot_num = url.split("/lot/lot-")[-1].split("/")[0].split("?")[0]
            # Christie's image CDN pattern (known to work for many lots)
            return f"https://www.christies.com/img/LotImages/{lot_num}/{lot_num}_1_l.jpg"
        return None

    # Other URLs — try as-is
    return url


def download_image(url: str, save_path: Path, max_dim: int = TARGET_SIZE) -> bool:
    """Download image, resize, and save as JPEG."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()

        # Check content type
        content_type = resp.headers.get("Content-Type", "")
        if "html" in content_type or "text" in content_type:
            return False

        content = resp.content
        if len(content) < 1000:
            # Too small — probably an error page or placeholder
            return False

        img = Image.open(BytesIO(content))
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
        logger.debug(f"Failed to download {url[:100]}: {e}")
        return False


def load_progress() -> dict:
    """Load download progress (which lot_ids have been attempted)."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"downloaded": [], "failed": [], "skipped": []}


def save_progress(progress: dict) -> None:
    """Save download progress."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def main():
    logger.info("=" * 60)
    logger.info("Downloading artwork images")
    logger.info("=" * 60)

    if not MASTER_FILE.exists():
        logger.error(f"Master file not found: {MASTER_FILE}")
        logger.error("Run feature_engineering.py first!")
        sys.exit(1)

    df = pd.read_csv(MASTER_FILE)
    logger.info(f"Loaded {len(df)} lots from master.csv")

    # Filter to rows with image URLs
    has_url = df["image_url"].notna() & (df["image_url"].astype(str).str.strip() != "")
    df_with_images = df[has_url].copy()
    logger.info(f"Lots with image_url: {len(df_with_images)}")

    DATA_IMAGES.mkdir(parents=True, exist_ok=True)

    # Check what's already downloaded
    existing = {p.stem for p in DATA_IMAGES.glob("*.jpg")}
    progress = load_progress()
    attempted = set(progress["downloaded"] + progress["failed"] + progress["skipped"])

    # Determine what to download
    to_process = df_with_images[
        ~df_with_images["lot_id"].astype(str).isin(existing | attempted)
    ]
    logger.info(f"Already downloaded: {len(existing)}")
    logger.info(f"Previously failed/skipped: {len(attempted - existing)}")
    logger.info(f"Remaining to process: {len(to_process)}")

    if len(to_process) == 0:
        logger.info("Nothing to download!")
        return

    success = 0
    failed = 0
    skipped = 0
    total = len(to_process)

    for i, (_, row) in enumerate(to_process.iterrows()):
        lot_id = str(row["lot_id"])
        raw_url = str(row["image_url"]).strip()

        # Resolve to a downloadable URL
        image_url = resolve_image_url(raw_url)

        if image_url is None:
            skipped += 1
            progress["skipped"].append(lot_id)
            continue

        save_path = DATA_IMAGES / f"{lot_id}.jpg"

        if download_image(image_url, save_path):
            success += 1
            progress["downloaded"].append(lot_id)
        else:
            # For Sotheby's, try fallback to Small if Medium failed
            fallback = False
            if "sothebys.com" in image_url and "/Medium" in image_url:
                fallback_url = image_url.replace("/Medium", "/Small")
                if download_image(fallback_url, save_path):
                    success += 1
                    progress["downloaded"].append(lot_id)
                    fallback = True
            if not fallback:
                failed += 1
                progress["failed"].append(lot_id)

        # Print progress every 100 images
        processed = success + failed + skipped
        if processed % 100 == 0:
            logger.info(
                f"Progress: {processed}/{total} "
                f"(success={success}, failed={failed}, skipped={skipped})"
            )
            save_progress(progress)

        time.sleep(DELAY)

    # Final save
    save_progress(progress)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Download complete!")
    logger.info(f"  Successfully downloaded: {success}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Skipped (invalid URL): {skipped}")
    logger.info(f"  Total images on disk: {len(list(DATA_IMAGES.glob('*.jpg')))}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
