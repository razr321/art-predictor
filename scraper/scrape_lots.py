#!/usr/bin/env python3
"""Scrape lot details from Christie's South Asian Modern & Contemporary Art auctions.

Two-phase approach:
1. Bulk: Load auction page → extract all lots from window.chrComponents (fast)
2. Detail: Load individual lot pages for provenance, literature, exhibitions (slow)

Output: data/raw/lots.csv
"""

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.helpers import setup_logger, DATA_RAW, append_ndjson, load_ndjson, save_ndjson_batch
from utils.data_cleaning import (
    parse_artist_name,
    parse_medium,
    parse_dimensions,
    parse_year_created,
    is_signed,
    is_dated,
    count_provenance_entries,
    count_literature_entries,
    count_exhibition_entries,
    parse_currency_amount,
)
from utils.currency import to_usd

logger = setup_logger(__name__, "scrape_lots.log")

PROGRESS_FILE = DATA_RAW / "lots_progress.ndjson"
OUTPUT_FILE = DATA_RAW / "lots.csv"
AUCTIONS_FILE = DATA_RAW / "auctions.csv"

SAVE_INTERVAL = 20  # Save progress every N lots
DELAY_BETWEEN_LOTS = 1.5  # Seconds between lot detail page loads
DELAY_BETWEEN_AUCTIONS = 3  # Seconds between auction page loads


def make_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)


def dismiss_overlays(driver: webdriver.Chrome) -> None:
    """Dismiss cookie banners, popups, age checks."""
    selectors = [
        "#onetrust-accept-btn-handler",
        "[data-testid='cookie-accept']",
        ".chr-cookie-banner button",
        ".chr-modal__close",
        "[aria-label='Close']",
        ".chr-popup__close",
    ]
    for sel in selectors:
        try:
            btn = driver.find_element(By.CSS_SELECTOR, sel)
            if btn.is_displayed():
                btn.click()
                time.sleep(0.5)
        except Exception:
            pass


def extract_bulk_lots(driver: webdriver.Chrome) -> list[dict]:
    """Extract all lots from auction listing page via JS."""
    try:
        data = driver.execute_script("""
            if (!window.chrComponents) return null;
            for (const key of Object.keys(window.chrComponents)) {
                const comp = window.chrComponents[key];
                if (comp && comp.data && comp.data.lots && Array.isArray(comp.data.lots)) {
                    // Also grab sale data
                    let sale = null;
                    for (const k2 of Object.keys(window.chrComponents)) {
                        const c2 = window.chrComponents[k2];
                        if (c2 && c2.data && c2.data.sale_id) {
                            sale = c2.data;
                            break;
                        }
                    }
                    return JSON.stringify({lots: comp.data.lots, sale: sale, total: comp.data.total_lots || comp.data.lots.length});
                }
            }
            return null;
        """)
        if data:
            return json.loads(data)
    except Exception as e:
        logger.warning(f"Failed to extract bulk lots: {e}")
    return {"lots": [], "sale": None, "total": 0}


def extract_lot_detail(driver: webdriver.Chrome) -> dict:
    """Extract detailed lot info from individual lot page."""
    result = {
        "details_text": "",
        "provenance_text": "",
        "literature_text": "",
        "exhibited_text": "",
        "lot_essay": "",
        "image_urls": [],
    }

    try:
        # Wait for content to render
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".chr-lot-details, .chr-lot-header"))
        )
        time.sleep(1)

        # Extract from JS objects (structured data)
        js_data = driver.execute_script("""
            if (!window.chrComponents) return null;
            for (const key of Object.keys(window.chrComponents)) {
                if (key.startsWith('lotHeader')) {
                    const comp = window.chrComponents[key];
                    if (comp && comp.data && comp.data.lots) {
                        return JSON.stringify(comp.data);
                    }
                }
            }
            return null;
        """)

        if js_data:
            parsed = json.loads(js_data)
            lots = parsed.get("lots", [])
            if lots:
                lot = lots[0]
                # Get image URLs from lot_assets
                assets = lot.get("lot_assets", [])
                for asset in assets:
                    img = asset.get("image_src") or asset.get("image_desktop_src", "")
                    if img:
                        result["image_urls"].append(img)

        # Extract accordion sections from HTML
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        # Find all accordion content sections
        sections = soup.select(".chr-lot-section__accordion--content")
        # Also try alternate selectors
        if not sections:
            sections = soup.select("[class*='lot-section'] [class*='accordion']")

        # Map sections by their preceding label
        all_section_divs = soup.select(".chr-lot-details section, .chr-lot-details .chr-lot-section")
        for div in all_section_divs:
            label_el = div.select_one("h3, .chr-lot-section__title, [class*='section__title']")
            content_el = div.select_one(".chr-lot-section__accordion--content, [class*='accordion--content']")
            if not label_el or not content_el:
                continue
            label = label_el.get_text(strip=True).lower()
            content = content_el.get_text(separator="\n", strip=True)

            if "detail" in label:
                result["details_text"] = content
            elif "provenance" in label:
                result["provenance_text"] = content
            elif "literature" in label:
                result["literature_text"] = content
            elif "exhibit" in label:
                result["exhibited_text"] = content

        # If sections found but no labels, use positional mapping (Details, Provenance, Literature, Exhibited)
        if not result["details_text"] and len(sections) >= 1:
            result["details_text"] = sections[0].get_text(separator="\n", strip=True)
        if not result["provenance_text"] and len(sections) >= 2:
            result["provenance_text"] = sections[1].get_text(separator="\n", strip=True)
        if not result["literature_text"] and len(sections) >= 3:
            result["literature_text"] = sections[2].get_text(separator="\n", strip=True)
        if not result["exhibited_text"] and len(sections) >= 4:
            result["exhibited_text"] = sections[3].get_text(separator="\n", strip=True)

        # Lot essay
        essay_el = soup.select_one(".chr-lot-article, [class*='lot-essay']")
        if essay_el:
            result["lot_essay"] = essay_el.get_text(separator="\n", strip=True)

    except Exception as e:
        logger.warning(f"Error extracting lot detail: {e}")

    return result


def parse_lot_record(lot_js: dict, auction_meta: dict, detail: dict | None = None) -> dict:
    """Convert raw JS lot object + detail scrape into a clean record."""
    # Artist info
    raw_artist = lot_js.get("title_primary_txt", "")
    artist_name, birth_year, death_year = parse_artist_name(raw_artist)

    # Title
    title = lot_js.get("title_secondary_txt", "")

    # Description (may contain medium, dimensions)
    desc = lot_js.get("description_txt", "")

    # Use detail text if available, otherwise fall back to description
    details_text = ""
    if detail:
        details_text = detail.get("details_text", "")
    if not details_text:
        details_text = desc

    # Medium
    medium = parse_medium(details_text)

    # Dimensions — try JS lot_assets first, then parse from text
    height_cm, width_cm = None, None
    assets = lot_js.get("lot_assets", [])
    if assets:
        a = assets[0]
        h = a.get("height_cm")
        w = a.get("width_cm")
        if h and w:
            try:
                height_cm, width_cm = float(h), float(w)
            except (ValueError, TypeError):
                pass
    if not height_cm:
        height_cm, width_cm = parse_dimensions(details_text)

    # Year created
    year_created = parse_year_created(details_text)

    # Signed / Dated
    signed = is_signed(details_text)
    dated = is_dated(details_text)

    # Provenance, Literature, Exhibited
    prov_text = detail.get("provenance_text", "") if detail else ""
    lit_text = detail.get("literature_text", "") if detail else ""
    exh_text = detail.get("exhibited_text", "") if detail else ""

    prov_count = count_provenance_entries(prov_text)
    lit_count = count_literature_entries(lit_text)
    exh_count = count_exhibition_entries(exh_text)

    # Estimates
    est_low = lot_js.get("estimate_low")
    est_high = lot_js.get("estimate_high")
    est_txt = lot_js.get("estimate_txt", "")
    _, est_currency = parse_currency_amount(est_txt) if est_txt else (None, "USD")

    # Price realized (hammer)
    price = lot_js.get("price_realised")
    price_txt = lot_js.get("price_realised_txt", "")
    _, price_currency = parse_currency_amount(price_txt) if price_txt else (None, "USD")

    # Convert to USD
    auction_date = auction_meta.get("start_date", "")
    est_low_usd = to_usd(est_low, est_currency, auction_date)
    est_high_usd = to_usd(est_high, est_currency, auction_date)
    hammer_price_usd = to_usd(price, price_currency, auction_date)

    # Sale status
    is_sold = not lot_js.get("is_unsold", True)
    is_withdrawn = lot_js.get("lot_withdrawn", False)

    # Image URL
    image_url = ""
    if detail and detail.get("image_urls"):
        image_url = detail["image_urls"][0]
    elif assets:
        image_url = assets[0].get("image_src", "") or assets[0].get("image_desktop_src", "")
    elif lot_js.get("image"):
        img_obj = lot_js["image"]
        image_url = img_obj.get("image_src", "") or img_obj.get("image_desktop_src", "")

    # Lot URL
    lot_url = lot_js.get("url", "")
    if lot_url and not lot_url.startswith("http"):
        lot_url = f"https://www.christies.com{lot_url}"

    return {
        "lot_id": str(lot_js.get("object_id", "")),
        "lot_number": lot_js.get("lot_id_txt", ""),
        "auction_id": auction_meta.get("auction_id", ""),
        "auction_title": auction_meta.get("title", ""),
        "auction_date": auction_date,
        "auction_location": auction_meta.get("location", ""),
        "sale_type": auction_meta.get("sale_type", ""),
        "artist_name": artist_name,
        "artist_birth_year": birth_year,
        "artist_death_year": death_year,
        "title": title,
        "medium_raw": details_text[:500] if details_text else "",
        "medium_category": medium,
        "height_cm": height_cm,
        "width_cm": width_cm,
        "year_created": year_created,
        "is_signed": signed,
        "is_dated": dated,
        "provenance_text": prov_text[:2000] if prov_text else "",
        "provenance_count": prov_count,
        "literature_text": lit_text[:2000] if lit_text else "",
        "literature_count": lit_count,
        "exhibited_text": exh_text[:2000] if exh_text else "",
        "exhibition_count": exh_count,
        "estimate_low_usd": est_low_usd,
        "estimate_high_usd": est_high_usd,
        "estimate_currency": est_currency,
        "hammer_price_usd": hammer_price_usd,
        "hammer_currency": price_currency,
        "is_sold": is_sold,
        "is_withdrawn": is_withdrawn,
        "image_url": image_url,
        "lot_url": lot_url,
    }


def scrape_auction(driver: webdriver.Chrome, auction: dict, scraped_lot_ids: set) -> list[dict]:
    """Scrape all lots from a single auction."""
    url = auction.get("url", "")
    if not url:
        logger.warning(f"No URL for auction {auction.get('auction_id')}")
        return []

    if not url.startswith("http"):
        url = f"https://www.christies.com{url}"

    auction_id = auction.get("auction_id", "")
    logger.info(f"\nScraping auction: {auction.get('title', '')} ({auction_id})")
    logger.info(f"  URL: {url}")

    # Load auction listing page
    try:
        driver.get(url)
        time.sleep(DELAY_BETWEEN_AUCTIONS)
        dismiss_overlays(driver)
    except Exception as e:
        logger.error(f"Failed to load auction page: {e}")
        return []

    # Phase 1: Bulk extract from listing page
    bulk = extract_bulk_lots(driver)
    lots_js = bulk.get("lots", [])
    sale_data = bulk.get("sale", {})
    total = bulk.get("total", len(lots_js))

    logger.info(f"  Bulk extracted {len(lots_js)} of {total} lots")

    # Build auction metadata
    auction_meta = {
        "auction_id": auction_id,
        "title": auction.get("title", ""),
        "start_date": auction.get("start_date", ""),
        "location": auction.get("location", ""),
        "sale_type": auction.get("sale_type", ""),
    }
    if sale_data:
        auction_meta["start_date"] = sale_data.get("start_date", auction_meta["start_date"])
        auction_meta["location"] = sale_data.get("location_txt", auction_meta["location"])

    records = []
    new_count = 0

    for lot_js in tqdm(lots_js, desc=f"  Lots ({auction_id})", leave=False):
        lot_id = str(lot_js.get("object_id", ""))
        if lot_id in scraped_lot_ids:
            continue

        # Phase 2: Load lot detail page for provenance/literature/exhibitions
        detail = None
        lot_url = lot_js.get("url", "")
        if lot_url:
            if not lot_url.startswith("http"):
                lot_url = f"https://www.christies.com{lot_url}"
            try:
                driver.get(lot_url)
                time.sleep(DELAY_BETWEEN_LOTS)
                dismiss_overlays(driver)
                detail = extract_lot_detail(driver)
            except Exception as e:
                logger.warning(f"  Failed to load lot {lot_id}: {e}")

        record = parse_lot_record(lot_js, auction_meta, detail)
        records.append(record)
        scraped_lot_ids.add(lot_id)
        new_count += 1

        # Save progress periodically
        if new_count % SAVE_INTERVAL == 0:
            for rec in records[-SAVE_INTERVAL:]:
                append_ndjson(PROGRESS_FILE, rec)
            logger.info(f"  Saved progress ({new_count} lots)")

    # Save any remaining unsaved records
    remaining = new_count % SAVE_INTERVAL
    if remaining > 0:
        for rec in records[-remaining:]:
            append_ndjson(PROGRESS_FILE, rec)

    logger.info(f"  → Scraped {new_count} new lots from {auction.get('title', '')}")
    return records


def main():
    logger.info("=" * 60)
    logger.info("Scraping Christie's SA M&CA lot details")
    logger.info("=" * 60)

    # Load auctions list
    if not AUCTIONS_FILE.exists():
        logger.error(f"Auctions file not found: {AUCTIONS_FILE}")
        logger.error("Run discover_auctions.py first!")
        sys.exit(1)

    auctions_df = pd.read_csv(AUCTIONS_FILE)
    auctions = auctions_df.to_dict("records")
    logger.info(f"Found {len(auctions)} auctions to scrape")

    # Load existing progress
    existing = load_ndjson(PROGRESS_FILE)
    scraped_lot_ids = {r["lot_id"] for r in existing}
    all_records = list(existing)
    logger.info(f"Loaded {len(existing)} previously scraped lots")

    driver = make_driver()
    try:
        for i, auction in enumerate(auctions, 1):
            logger.info(f"\n--- Auction {i}/{len(auctions)} ---")
            new_records = scrape_auction(driver, auction, scraped_lot_ids)
            all_records.extend(new_records)

    except KeyboardInterrupt:
        logger.info("\nInterrupted — saving progress...")
    finally:
        driver.quit()

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
            logger.info(f"Price range: ${df['hammer_price_usd'].min():,.0f} - ${df['hammer_price_usd'].max():,.0f}")
    else:
        logger.warning("No lots scraped!")


if __name__ == "__main__":
    main()
