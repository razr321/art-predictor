#!/usr/bin/env python3
"""Merge Christie's and Sotheby's lot data into a single lots.csv.

Reads:
  data/raw/lots_christies.csv  (renamed from lots.csv)
  data/raw/lots_sothebys.csv

Output:
  data/raw/lots.csv  (combined, deduplicated)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.helpers import setup_logger, DATA_RAW

logger = setup_logger(__name__, "merge_sources.log")

CHRISTIES_FILE = DATA_RAW / "lots_christies.csv"
SOTHEBYS_FILE = DATA_RAW / "lots_sothebys.csv"
SAFFRONART_FILE = DATA_RAW / "lots_saffronart.csv"
OUTPUT_FILE = DATA_RAW / "lots.csv"

# Also support the original lots.csv as Christie's source
CHRISTIES_FALLBACK = DATA_RAW / "lots.csv"

# Individual artist datasets (scraped separately)
ARTIST_FILES = [
    ("lots_tyeb_mehta_clean.csv", "artist_scrape"),
    ("lots_souza_clean.csv", "artist_scrape"),
    ("lots_raza_clean.csv", "artist_scrape"),
]


def main():
    logger.info("=" * 60)
    logger.info("Merging auction house data")
    logger.info("=" * 60)

    frames = []

    # Christie's
    christies_path = CHRISTIES_FILE if CHRISTIES_FILE.exists() else CHRISTIES_FALLBACK
    if christies_path.exists():
        df_c = pd.read_csv(christies_path)
        # Ensure lot_ids don't collide — prefix if needed
        if not df_c["lot_id"].astype(str).str.startswith("christies_").any():
            # Only add prefix if not already prefixed and IDs are numeric
            if df_c["lot_id"].astype(str).str.isnumeric().all():
                df_c["lot_id"] = "christies_" + df_c["lot_id"].astype(str)
        df_c["source"] = "christies"
        frames.append(df_c)
        logger.info(f"Christie's: {len(df_c)} lots from {christies_path.name}")
    else:
        logger.warning("No Christie's data found")

    # Sotheby's
    if SOTHEBYS_FILE.exists():
        df_s = pd.read_csv(SOTHEBYS_FILE)
        df_s["source"] = "sothebys"
        frames.append(df_s)
        logger.info(f"Sotheby's: {len(df_s)} lots from {SOTHEBYS_FILE.name}")
    else:
        logger.warning("No Sotheby's data found")

    # Saffronart
    if SAFFRONART_FILE.exists():
        df_sa = pd.read_csv(SAFFRONART_FILE)
        df_sa["source"] = "saffronart"
        frames.append(df_sa)
        logger.info(f"Saffronart: {len(df_sa)} lots from {SAFFRONART_FILE.name}")
    else:
        logger.warning("No Saffronart data found")

    # Individual artist datasets
    for fname, source_label in ARTIST_FILES:
        fpath = DATA_RAW / fname
        if fpath.exists():
            df_a = pd.read_csv(fpath)
            if "source" not in df_a.columns:
                df_a["source"] = source_label
            frames.append(df_a)
            logger.info(f"Artist file: {len(df_a)} lots from {fname}")
        else:
            logger.debug(f"Artist file not found: {fname}")

    if not frames:
        logger.error("No data to merge!")
        sys.exit(1)

    # Combine
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["lot_id"])
    df = df.sort_values(["auction_date", "lot_number"], ascending=[False, True])

    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"\nMerged: {len(df)} total lots → {OUTPUT_FILE}")
    logger.info(f"By source: {df['source'].value_counts().to_dict()}")
    logger.info(f"Sold: {df['is_sold'].sum()}")
    logger.info(f"Unique artists: {df['artist_name'].nunique()}")


if __name__ == "__main__":
    main()
