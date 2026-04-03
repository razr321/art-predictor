#!/usr/bin/env python3
"""Scrape lot details from AstaGuru past art auctions via their REST API.

Two-phase approach (no Selenium needed):
1. Fetch past auction list, filter to qualifying art auctions.
2. For each auction, fetch all lots via the filter-lots API.

Output: data/raw/lots_astaguru.csv
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.helpers import setup_logger, DATA_RAW, append_ndjson, load_ndjson
from utils.data_cleaning import (
    parse_medium,
    parse_dimensions,
    parse_year_created,
    is_signed,
    is_dated,
    count_provenance_entries,
    count_literature_entries,
    count_exhibition_entries,
)
from utils.currency import to_usd

logger = setup_logger(__name__, "scrape_lots_astaguru.log")

PROGRESS_FILE = DATA_RAW / "lots_astaguru_progress.ndjson"
OUTPUT_FILE = DATA_RAW / "lots_astaguru.csv"

SAVE_INTERVAL = 50
DELAY_BETWEEN_REQUESTS = 0.5

BASE_URL = "https://www.astaguru.com"
AUCTION_API = f"{BASE_URL}/api/auctions/get-auctions-by-status?auctionType=PAST&sortOrder=desc&limit=300"
LOT_API_TEMPLATE = f"{BASE_URL}/api/auctions/filter-lots?auctionId={{auction_id}}&limit=1000"

# Subcategories to include
VALID_SUBCATEGORIES = {"Painting", "Sculpture", "Work on Paper", "Mixed Media"}

# Auction title keywords to exclude (non-art auctions)
EXCLUDE_KEYWORDS = [
    "jewel", "auto", "motor", "timepiece", "treasure", "watch",
    "silver", "car", "stamp", "book", "numismatic", "pashmina",
    "textile", "memorabilia",
]

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
})


# ---------------------------------------------------------------------------
# Dimension & year parsing helpers (AstaGuru-specific formats)
# ---------------------------------------------------------------------------

def parse_astaguru_dimensions(size_str: str) -> tuple[Optional[float], Optional[float]]:
    """Parse AstaGuru size field: '28 x 32.5 in (71 x 82.5 cm)'.

    Prefers cm values when available; falls back to inches converted to cm.
    """
    if not size_str:
        return (None, None)

    # Try cm first: "71 x 82.5 cm" or "71 × 82.5 cm"
    m = re.search(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*cm", size_str, re.I)
    if m:
        return (float(m.group(1)), float(m.group(2)))

    # Try inches, convert to cm
    m = re.search(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*in", size_str, re.I)
    if m:
        return (round(float(m.group(1)) * 2.54, 2), round(float(m.group(2)) * 2.54, 2))

    # Fallback to generic parser
    return parse_dimensions(size_str)


def parse_astaguru_year(year_str: str) -> Optional[int]:
    """Parse AstaGuru creationYearValue: 'Circa 1940', '1965', 'Early 1970s', etc."""
    if not year_str:
        return None
    text = year_str.strip().lower()

    # "early 1970s" -> 1971, "late 1970s" -> 1978, "mid 1970s" -> 1975
    m = re.search(r"(early|mid|late)\s+(\d{4})s", text)
    if m:
        decade = int(m.group(2))
        modifier = m.group(1)
        if modifier == "early":
            return decade + 1
        elif modifier == "mid":
            return decade + 5
        else:  # late
            return decade + 8

    # "1970s" -> 1975
    m = re.search(r"(\d{4})s\b", text)
    if m:
        return int(m.group(1)) + 5

    # "Circa 1940" or "c. 1940"
    m = re.search(r"(?:circa|c\.?)\s*(\d{4})", text)
    if m:
        return int(m.group(1))

    # Range: "1960-1965" -> midpoint
    m = re.search(r"(\d{4})\s*[-–]\s*(\d{4})", text)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if 1800 < y1 < 2030 and 1800 < y2 < 2030:
            return (y1 + y2) // 2

    # Bare year
    m = re.search(r"\b((?:19|20)\d{2})\b", text)
    if m:
        return int(m.group(1))

    # Fallback to generic parser
    return parse_year_created(year_str)


def _clean_html(html_str: str) -> str:
    """Strip HTML tags from a string, preserving line breaks."""
    if not html_str:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", html_str)
    text = re.sub(r"</(?:p|li|div|tr)>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&#\d+;", "", text)
    return text.strip()


def _parse_auction_date(auction_details: dict, auction_meta: dict) -> str:
    """Extract auction date as YYYY-MM-DD from lot or auction metadata."""
    # Try auctionBasicDetails.auctionStartIST: "03/10/2026 10:00:00"
    if auction_details:
        start_ist = auction_details.get("auctionStartIST", "")
        if start_ist:
            try:
                dt = datetime.strptime(start_ist.split(" ")[0], "%m/%d/%Y")
                return dt.strftime("%Y-%m-%d")
            except (ValueError, IndexError):
                pass

    # Try auction-level startDateTime
    start_dt = auction_meta.get("startDateTime", "")
    if start_dt:
        try:
            dt = datetime.fromisoformat(start_dt.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass

    return ""


# ---------------------------------------------------------------------------
# Phase 1: Fetch and filter auctions
# ---------------------------------------------------------------------------

def is_qualifying_auction(auction: dict) -> bool:
    """Check if an auction qualifies for scraping."""
    title = (auction.get("title", "") or "").lower()

    # Exclude non-art auctions by title keywords
    if any(kw in title for kw in EXCLUDE_KEYWORDS):
        return False

    return True


def fetch_qualifying_auctions() -> list[dict]:
    """Fetch all past auctions and filter to qualifying art ones."""
    logger.info(f"Fetching auction list from API: {AUCTION_API}")
    try:
        resp = SESSION.get(AUCTION_API, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch auction API: {e}")
        return []

    all_auctions = data.get("auctions", [])
    total_count = data.get("totalCount", len(all_auctions))
    logger.info(f"  Total past auctions from API: {total_count}")

    qualifying = []
    for auction in all_auctions:
        if not is_qualifying_auction(auction):
            title = auction.get("title", "")
            logger.info(f"  Skipping non-art auction: {title}")
            continue

        auction_id = auction.get("id")
        title = auction.get("title", "")
        slug = auction.get("slug", "")
        start_dt = auction.get("startDateTime", "")
        end_dt = auction.get("endDateTime", "")
        total_sale_inr = auction.get("totalSaleINR")

        # Parse auction date
        auction_date = ""
        if start_dt:
            try:
                dt = datetime.fromisoformat(start_dt.replace("Z", "+00:00"))
                auction_date = dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass

        qualifying.append({
            "auction_id": auction_id,
            "title": title,
            "slug": slug,
            "start_date": auction_date,
            "start_datetime": start_dt,
            "end_datetime": end_dt,
            "total_sale_inr": total_sale_inr,
        })

    logger.info(f"  Qualifying art auctions: {len(qualifying)}")
    for a in qualifying:
        logger.info(f"    {a['start_date']} | {a['title']} (id={a['auction_id']})")

    return qualifying


# ---------------------------------------------------------------------------
# Phase 2: Fetch lots for each auction
# ---------------------------------------------------------------------------

def fetch_lots_for_auction(auction: dict) -> list[dict]:
    """Fetch all lots for a given auction from the API."""
    auction_id = auction["auction_id"]
    url = LOT_API_TEMPLATE.format(auction_id=auction_id)

    try:
        resp = SESSION.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"  Failed to fetch lots for auction {auction_id}: {e}")
        return []

    lots = data.get("lots", [])
    total = data.get("totalCount", len(lots))
    logger.info(f"  Fetched {len(lots)}/{total} lots from API")
    return lots


def build_lot_record(lot: dict, auction: dict) -> Optional[dict]:
    """Build a standardized lot record from AstaGuru API lot data.

    Returns None if the lot should be filtered out.
    """
    # Filter by category and subcategory
    category = (lot.get("category") or "").strip()
    sub_category = (lot.get("subCategory") or "").strip()

    if category != "Art":
        return None
    if sub_category not in VALID_SUBCATEGORIES:
        return None

    auction_id = auction["auction_id"]
    lot_id_raw = lot.get("id") or lot.get("_id", "")
    lot_number = lot.get("lotNumber", 0)
    slug = lot.get("slug", "")

    # Artist
    artist_name = (lot.get("creatorValue") or "").strip()

    # Title
    title = (lot.get("title") or "").strip()

    # Medium
    medium_raw = (lot.get("mediumValue") or "").strip()
    medium_category = parse_medium(medium_raw)

    # Dimensions
    size_str = (lot.get("size") or "").strip()
    height_cm, width_cm = parse_astaguru_dimensions(size_str)

    # Year created
    year_str = (lot.get("creationYearValue") or "").strip()
    year_created = parse_astaguru_year(year_str)

    # Description (contains signed/dated info)
    description_raw = lot.get("description") or ""
    description = _clean_html(description_raw)

    # Signed / Dated from description
    signed = is_signed(description)
    dated = is_dated(description)

    # Provenance
    provenance_raw = lot.get("provenance") or ""
    provenance_text = _clean_html(provenance_raw)
    prov_count = count_provenance_entries(provenance_text)

    # Literature
    literature_raw = lot.get("literature") or ""
    literature_text = _clean_html(literature_raw)
    lit_count = count_literature_entries(literature_text)

    # Exhibition
    exhibited_raw = lot.get("exhibition") or lot.get("exhibited") or ""
    exhibited_text = _clean_html(exhibited_raw)
    exh_count = count_exhibition_entries(exhibited_text)

    # Auction date
    auction_basic = lot.get("auctionBasicDetails") or {}
    auction_date = _parse_auction_date(auction_basic, auction)
    if not auction_date:
        auction_date = auction.get("start_date", "")

    # Auction title from lot-level data or auction-level
    auction_title = auction_basic.get("auctionName") or auction.get("title", "")

    # Estimates (in INR, convert to USD)
    est_low_inr = lot.get("priceMinINR")
    est_high_inr = lot.get("priceMaxINR")
    est_low_usd = to_usd(est_low_inr, "INR", auction_date) if est_low_inr else None
    est_high_usd = to_usd(est_high_inr, "INR", auction_date) if est_high_inr else None

    # Hammer price from auctionState
    auction_state = lot.get("auctionState") or {}
    is_closed = auction_state.get("isClosed", False)
    hammer_with_margin_inr = auction_state.get("hammerWithMarginINR")
    hammer_with_margin_usd = auction_state.get("hammerWithMarginUSD")

    # Determine sold status
    is_sold = False
    is_withdrawn = False
    hammer_price_usd = None
    hammer_currency = ""

    if is_closed and hammer_with_margin_inr and hammer_with_margin_inr > 0:
        is_sold = True
        hammer_currency = "INR"
        # Use to_usd for consistency rather than the API-provided USD
        hammer_price_usd = to_usd(hammer_with_margin_inr, "INR", auction_date)

    # Check withdrawal status
    lot_status = (lot.get("status") or "").lower()
    if "withdrawn" in lot_status or "withdraw" in lot_status:
        is_withdrawn = True
        is_sold = False
        hammer_price_usd = None

    # Image URL
    image_url = ""
    media_collection = lot.get("mediaCollection") or []
    if media_collection:
        # Pick the first image's URL
        first_media = media_collection[0]
        image_url = first_media.get("url") or first_media.get("imageUrl") or ""

    # Lot URL
    lot_url = ""
    if slug:
        lot_url = f"{BASE_URL}/lots/{slug}"

    # Sale type
    auction_title_lower = auction_title.lower()
    if "online" in auction_title_lower:
        sale_type = "online"
    elif "live" in auction_title_lower:
        sale_type = "live"
    else:
        sale_type = "auction"

    return {
        "lot_id": f"astaguru_{lot_id_raw}",
        "lot_number": lot_number,
        "auction_id": f"astaguru_{auction_id}",
        "auction_title": auction_title,
        "auction_date": auction_date,
        "auction_location": "Mumbai",
        "sale_type": sale_type,
        "artist_name": artist_name,
        "artist_birth_year": None,  # AstaGuru API doesn't provide birth/death years
        "artist_death_year": None,
        "title": title,
        "medium_raw": medium_raw[:500] if medium_raw else "",
        "medium_category": medium_category,
        "height_cm": height_cm,
        "width_cm": width_cm,
        "year_created": year_created,
        "is_signed": signed,
        "is_dated": dated,
        "provenance_text": provenance_text[:2000] if provenance_text else "",
        "provenance_count": prov_count,
        "literature_text": literature_text[:2000] if literature_text else "",
        "literature_count": lit_count,
        "exhibited_text": exhibited_text[:2000] if exhibited_text else "",
        "exhibition_count": exh_count,
        "estimate_low_usd": est_low_usd,
        "estimate_high_usd": est_high_usd,
        "estimate_currency": "INR",
        "hammer_price_usd": hammer_price_usd,
        "hammer_currency": hammer_currency,
        "is_sold": is_sold,
        "is_withdrawn": is_withdrawn,
        "image_url": image_url,
        "lot_url": lot_url,
    }


# ---------------------------------------------------------------------------
# Auction-level orchestrator
# ---------------------------------------------------------------------------

def scrape_auction(auction: dict, scraped_lot_ids: set) -> list[dict]:
    """Scrape all lots from a single AstaGuru auction."""
    auction_id = auction["auction_id"]
    logger.info(f"\nScraping auction: {auction['title']} (id={auction_id})")

    lots_data = fetch_lots_for_auction(auction)
    if not lots_data:
        logger.warning(f"  No lots found for auction {auction_id}")
        return []

    records = []
    skipped_category = 0
    new_count = 0

    for lot in tqdm(lots_data, desc=f"  Lots (id={auction_id})", leave=False):
        record = build_lot_record(lot, auction)
        if record is None:
            skipped_category += 1
            continue

        lot_id = record["lot_id"]
        if lot_id in scraped_lot_ids:
            continue

        records.append(record)
        scraped_lot_ids.add(lot_id)
        new_count += 1

        # Save progress periodically
        if new_count % SAVE_INTERVAL == 0:
            for rec in records[-SAVE_INTERVAL:]:
                append_ndjson(PROGRESS_FILE, rec)
            logger.info(f"  Saved progress ({new_count} lots)")

    # Save remaining
    remaining = new_count % SAVE_INTERVAL
    if remaining > 0:
        for rec in records[-remaining:]:
            append_ndjson(PROGRESS_FILE, rec)

    if skipped_category > 0:
        logger.info(f"  Skipped {skipped_category} non-art lots (category/subCategory filter)")
    logger.info(f"  -> Scraped {new_count} new art lots from {auction['title']}")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("Scraping AstaGuru past auction lot details")
    logger.info("=" * 60)

    # Phase 1: Fetch qualifying auctions
    auctions = fetch_qualifying_auctions()
    if not auctions:
        logger.error("No qualifying auctions found!")
        sys.exit(1)

    # Load existing progress
    if "--fresh" in sys.argv:
        existing = []
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        logger.info("Fresh run — ignoring previous progress")
    else:
        existing = load_ndjson(PROGRESS_FILE)
        logger.info(f"Loaded {len(existing)} previously scraped lots")
    scraped_lot_ids = {r["lot_id"] for r in existing}
    all_records = list(existing)

    try:
        for i, auction in enumerate(auctions, 1):
            logger.info(f"\n--- Auction {i}/{len(auctions)} ---")
            new_records = scrape_auction(auction, scraped_lot_ids)
            all_records.extend(new_records)
            time.sleep(DELAY_BETWEEN_REQUESTS)

    except KeyboardInterrupt:
        logger.info("\nInterrupted — saving progress...")

    # Save final CSV
    if all_records:
        df = pd.DataFrame(all_records)
        df = df.drop_duplicates(subset=["lot_id"])
        df = df.sort_values(["auction_date", "lot_number"], ascending=[False, True])
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"\nSaved {len(df)} lots to {OUTPUT_FILE}")
        logger.info(f"Sold: {df['is_sold'].sum()}, Unsold: {(~df['is_sold']).sum()}")
        logger.info(f"Artists: {df['artist_name'].nunique()}")
        if df['hammer_price_usd'].notna().any():
            median_hammer = df.loc[df['is_sold'], 'hammer_price_usd'].median()
            logger.info(f"Median hammer (sold, USD): ${median_hammer:,.0f}")
    else:
        logger.warning("No lots scraped!")


if __name__ == "__main__":
    main()
