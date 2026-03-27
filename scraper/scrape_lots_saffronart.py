#!/usr/bin/env python3
"""Scrape lot details from Saffronart seasonal art auctions (2015 onwards).

Three-phase approach:
1. Fetch auction list from Saffronart API, filter to qualifying seasonal auctions.
2. Load each auction's PostCatalog page to extract lot listing data.
3. Load individual lot detail pages (PostWork) for provenance, literature, exhibition.

Output: data/raw/lots_saffronart.csv
"""

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.helpers import setup_logger, DATA_RAW, append_ndjson, load_ndjson
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
)
from utils.currency import to_usd

logger = setup_logger(__name__, "scrape_lots_saffronart.log")

PROGRESS_FILE = DATA_RAW / "lots_saffronart_progress.ndjson"
OUTPUT_FILE = DATA_RAW / "lots_saffronart.csv"

SAVE_INTERVAL = 20
DELAY_BETWEEN_LOTS = 2.0
DELAY_BETWEEN_AUCTIONS = 3

BASE_URL = "https://www.saffronart.com"
AUCTION_API = f"{BASE_URL}/Service1.svc/FetchAllSaffronAuctions/?AucType=ART"

# Seasonal keywords to match in auction titles (2015 onwards)
SEASON_KEYWORDS = ["spring", "summer", "autumn", "winter"]

# Minimum year for auctions to scrape
MIN_YEAR = 2015


# ---------------------------------------------------------------------------
# Driver setup
# ---------------------------------------------------------------------------

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
    selectors = [
        "#onetrust-accept-btn-handler",
        "[data-testid='cookie-accept']",
        "button[class*='cookie']",
        "[aria-label='Close']",
        "[class*='modal'] button[class*='close']",
    ]
    for sel in selectors:
        try:
            btn = driver.find_element(By.CSS_SELECTOR, sel)
            if btn.is_displayed():
                btn.click()
                time.sleep(0.5)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Phase 1: Fetch and filter auctions from API
# ---------------------------------------------------------------------------

def _parse_dotnet_date(date_str: str) -> str:
    """Parse .NET /Date(1775061000000-0400)/ format to YYYY-MM-DD."""
    if not date_str:
        return ""
    m = re.search(r"/Date\((\d+)([+-]\d{4})?\)/", date_str)
    if m:
        ts_ms = int(m.group(1))
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")
    return ""


def _extract_year_from_event(event: dict) -> int | None:
    """Extract year from EventDate string like 'April 2026' or from EventStartDate."""
    event_date = event.get("EventDate", "") or ""
    m = re.search(r"(20\d{2})", event_date)
    if m:
        return int(m.group(1))
    # Fallback to start date
    start_date = _parse_dotnet_date(event.get("EventStartDate", "") or "")
    if start_date:
        try:
            return int(start_date[:4])
        except (ValueError, IndexError):
            pass
    return None


def is_qualifying_auction(event: dict) -> bool:
    """Check if an auction is a qualifying seasonal art auction from 2015+."""
    title = (event.get("Title", "") or "").lower()
    year = _extract_year_from_event(event)

    if year is None or year < MIN_YEAR:
        return False

    # Must contain a season keyword
    has_season = any(kw in title for kw in SEASON_KEYWORDS)
    if not has_season:
        return False

    # Exclude non-art auctions
    exclude_keywords = [
        "jewel", "collectible", "fundraiser", "charity", "relief",
        "textile", "watch", "wine", "book", "thread",
    ]
    if any(kw in title for kw in exclude_keywords):
        return False

    # Skip upcoming auctions (status 6) — no results yet
    status = event.get("EventStatus", 0)
    if status != 3:
        return False

    return True


def fetch_qualifying_auctions() -> list[dict]:
    """Fetch all auctions from API and filter to qualifying seasonal ones."""
    logger.info(f"Fetching auction list from API: {AUCTION_API}")
    try:
        resp = requests.get(AUCTION_API, timeout=30)
        resp.raise_for_status()
        all_auctions = resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch auction API: {e}")
        return []

    # API returns {"Events": [[], [], [list of auctions]], "Response": ...}
    if isinstance(all_auctions, dict) and "Events" in all_auctions:
        event_lists = all_auctions["Events"]
        # Flatten all sub-lists
        flat = []
        for sub in event_lists:
            if isinstance(sub, list):
                flat.extend(sub)
        all_auctions = flat

    logger.info(f"  Total auctions from API: {len(all_auctions)}")

    qualifying = []
    for event in all_auctions:
        if not is_qualifying_auction(event):
            continue

        event_id = event.get("EventId")
        title = event.get("Title", "")
        url_path = event.get("URL", "")
        year = _extract_year_from_event(event)
        start_date = _parse_dotnet_date(event.get("EventStartDate", "") or "")
        event_status = event.get("EventStatus", 0)

        # Determine sale type from title
        title_lower = title.lower()
        if "live" in title_lower:
            sale_type = "live"
        elif "online" in title_lower:
            sale_type = "online"
        else:
            sale_type = "auction"

        qualifying.append({
            "event_id": event_id,
            "title": title,
            "url_path": url_path,
            "year": year,
            "start_date": start_date,
            "sale_type": sale_type,
            "event_status": event_status,
        })

    # Sort by year descending
    qualifying.sort(key=lambda x: x.get("year", 0), reverse=True)
    logger.info(f"  Qualifying seasonal auctions (>= {MIN_YEAR}): {len(qualifying)}")
    for a in qualifying:
        logger.info(f"    {a['year']} | {a['title']} (eid={a['event_id']})")

    return qualifying


# ---------------------------------------------------------------------------
# Phase 2: Scrape auction catalog page for lot listing
# ---------------------------------------------------------------------------

def _determine_catalog_url(auction: dict) -> str:
    """Determine whether to use PreCatalog or PostCatalog based on auction status.

    EventStatus values observed:
      6 = upcoming/current (use PreCatalog or the slug URL)
      Other = completed (use PostCatalog)

    For completed auctions, PostCatalog.aspx shows results with winning bids.
    For upcoming, the slug URL (e.g. /auctions/spring-live-auction-2026-4964) works.
    """
    eid = auction["event_id"]
    status = auction.get("event_status", 0)

    # For completed auctions, use PostCatalog to see results
    if status != 6:
        return f"{BASE_URL}/auctions/PostCatalog.aspx?eid={eid}"

    # For upcoming/current auctions, use the slug URL
    url_path = auction.get("url_path", "")
    if url_path:
        if url_path.startswith("http"):
            return url_path
        return f"{BASE_URL}/{url_path}"

    # Fallback to PreCatalog
    return f"{BASE_URL}/auctions/PreCatalog.aspx?eid={eid}"


def _is_post_auction(auction: dict) -> bool:
    """Check if auction has concluded (results available)."""
    return auction.get("event_status", 0) != 6


def _parse_indian_number(s: str) -> float | None:
    """Parse Indian-format numbers like '38,40,000' or '1,20,00,000' to float."""
    if not s:
        return None
    # Remove currency symbols and whitespace
    s = re.sub(r"[Rs₹$€£\s]", "", s.strip())
    # Remove commas
    s = s.replace(",", "")
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def scrape_catalog_page(driver: webdriver.Chrome, auction: dict) -> list[dict]:
    """Load auction catalog page and extract lot listing information.

    Returns a list of dicts with basic lot info from the catalog page.
    Each dict has keys: lot_number, lot_work_id, artist_raw, title,
    medium_dims_text, estimate_text, winning_bid_text, image_url, detail_url.
    """
    catalog_url = _determine_catalog_url(auction)
    is_post = _is_post_auction(auction)
    eid = auction["event_id"]

    logger.info(f"  Loading catalog: {catalog_url}")

    try:
        driver.get(catalog_url)
        time.sleep(DELAY_BETWEEN_AUCTIONS)
        dismiss_overlays(driver)
    except Exception as e:
        logger.error(f"  Failed to load catalog page: {e}")
        return []

    # Check for redirect to storyltd.com (some online auctions)
    current_url = driver.current_url
    if "storyltd.com" in current_url:
        logger.warning(f"  Redirected to StoryLTD ({current_url}) — skipping")
        return []

    # Wait for lot content to load
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='Work.aspx']"))
        )
    except Exception:
        logger.warning("  Timed out waiting for lot links — trying anyway")

    # Paginate through all catalog pages
    seen_work_ids = set()
    lot_entries = []
    page_idx = 1

    while True:
        # Extract lot detail links to get work IDs
        lot_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='Work.aspx']")

        page_count = 0
        for link in lot_links:
            href = link.get_attribute("href") or ""
            m = re.search(r"Work\.aspx\?l=(\d+)&eid=(\d+)&lotno=(\d+)", href)
            if not m:
                continue

            work_id = m.group(1)
            link_eid = m.group(2)
            lot_no = m.group(3)

            if int(link_eid) != eid:
                continue
            if work_id in seen_work_ids:
                continue
            seen_work_ids.add(work_id)

            if is_post:
                detail_url = f"{BASE_URL}/auctions/PostWork.aspx?l={work_id}&eid={eid}&lotno={lot_no}&n={lot_no}"
            else:
                detail_url = f"{BASE_URL}/auctions/PreWork.aspx?l={work_id}&eid={eid}&lotno={lot_no}&n={lot_no}"

            lot_entries.append({
                "lot_number": int(lot_no),
                "lot_work_id": work_id,
                "detail_url": detail_url,
            })
            page_count += 1

        # Check for next page link
        page_idx += 1
        next_page_url = None
        try:
            paging_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='PageIndex']")
            for pl in paging_links:
                href = pl.get_attribute("href") or ""
                if f"PageIndex={page_idx}" in href:
                    next_page_url = href
                    break
        except Exception:
            pass

        if not next_page_url:
            break

        # Load next page
        try:
            driver.get(next_page_url)
            time.sleep(2)
        except Exception:
            break

    # Sort by lot number
    lot_entries.sort(key=lambda x: x["lot_number"])

    logger.info(f"  Found {len(lot_entries)} lots across {page_idx - 1} catalog pages")
    return lot_entries


# ---------------------------------------------------------------------------
# Phase 3: Scrape individual lot detail pages
# ---------------------------------------------------------------------------

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


def scrape_lot_detail(driver: webdriver.Chrome, detail_url: str, auction: dict) -> dict:
    """Load an individual lot page and extract all detail fields.

    Saffronart lot pages (Pre/PostWork.aspx) are server-rendered ASP.NET pages.
    We extract data by reading the page text and specific element patterns.
    """
    result = {
        "artist_raw": "",
        "artist_birth_year": None,
        "artist_death_year": None,
        "title": "",
        "medium_raw": "",
        "details_text": "",
        "dimensions_text": "",
        "year_text": "",
        "signed_dated_text": "",
        "provenance_text": "",
        "literature_text": "",
        "exhibited_text": "",
        "estimate_text": "",
        "estimate_low_inr": None,
        "estimate_high_inr": None,
        "estimate_low_usd": None,
        "estimate_high_usd": None,
        "hammer_price_inr": None,
        "hammer_price_usd": None,
        "hammer_currency": "",
        "is_sold": False,
        "is_withdrawn": False,
        "image_url": "",
    }

    try:
        driver.get(detail_url)
        time.sleep(DELAY_BETWEEN_LOTS)
        dismiss_overlays(driver)
    except Exception as e:
        logger.warning(f"    Failed to load lot page: {e}")
        return result

    # Check for redirect
    if "storyltd.com" in driver.current_url:
        logger.warning(f"    Redirected to StoryLTD — skipping lot")
        return result

    # Wait for page content
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(1)  # Extra wait for dynamic content
    except Exception:
        pass

    page_source = driver.page_source
    page_text = driver.find_element(By.TAG_NAME, "body").text

    # --- Image URL ---
    # Look for the main lot image (high-res)
    img_match = re.search(
        r'(https?://mediacloud\.saffronart\.com/[^"\']+?_(?:big|hires|6_big)\.(?:jpg|png|jpeg))',
        page_source, re.I
    )
    if img_match:
        result["image_url"] = img_match.group(1)
    else:
        # Fallback: any product image
        img_match = re.search(
            r'(https?://mediacloud\.saffronart\.com/sourcingcen/prod/productimages/[^"\']+\.(?:jpg|png|jpeg))',
            page_source, re.I
        )
        if img_match:
            result["image_url"] = img_match.group(1)

    # --- Artist name and dates ---
    # Pattern: "Artist Name (YYYY - YYYY)" or "Artist Name (b. YYYY)"
    artist_match = re.search(
        r'(?:^|\n)([A-Z][A-Za-z\s\.\-\']+?)\s*\((\d{4})\s*[-–]\s*(\d{4})\)',
        page_text
    )
    if artist_match:
        result["artist_raw"] = artist_match.group(1).strip()
        result["artist_birth_year"] = int(artist_match.group(2))
        result["artist_death_year"] = int(artist_match.group(3))
    else:
        artist_match = re.search(
            r'(?:^|\n)([A-Z][A-Za-z\s\.\-\']+?)\s*\(b\.\s*(\d{4})\)',
            page_text
        )
        if artist_match:
            result["artist_raw"] = artist_match.group(1).strip()
            result["artist_birth_year"] = int(artist_match.group(2))

    # --- Try extracting from HTML structure ---
    # Artist links are typically near the top of the page
    try:
        # Look for artist name in links near lot content
        artist_links = driver.find_elements(
            By.CSS_SELECTOR, "a[href*='DefaultController.aspx'], a[href*='artists/']"
        )
        for al in artist_links:
            name_text = al.text.strip()
            if name_text and len(name_text) > 2 and not name_text.startswith("http"):
                if not result["artist_raw"]:
                    result["artist_raw"] = name_text
                break
    except Exception:
        pass

    # --- Extract structured sections from page text ---
    # The page text typically has sections like:
    # ARTWORK DETAILS
    # Signed ...
    # Circa YYYY / YYYY
    # Medium on support
    # H x W in (H x W cm)
    #
    # PROVENANCE
    # ...
    # PUBLISHED
    # ...
    # EXHIBITED
    # ...

    sections = {}
    current_section = "header"
    section_headers = [
        "ARTWORK DETAILS", "PROVENANCE", "PUBLISHED", "LITERATURE",
        "EXHIBITED", "EXHIBITION HISTORY", "CONDITION REPORT",
        "ESTIMATE", "WINNING BID", "LOT NOTES", "LOT ESSAY",
    ]

    for line in page_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        matched_header = None
        for hdr in section_headers:
            if upper == hdr or upper.startswith(hdr):
                matched_header = hdr
                break
        if matched_header:
            current_section = matched_header
            sections[current_section] = []
        else:
            if current_section not in sections:
                sections[current_section] = []
            sections[current_section].append(stripped)

    # --- Title ---
    # Title is typically in the header section, after artist name
    # --- Title from ARTWORK DETAILS section ---
    # The ARTWORK DETAILS section has artist name then title as the next lines
    details_lines_raw = sections.get("ARTWORK DETAILS", [])
    if details_lines_raw:
        for line in details_lines_raw:
            # Skip lines that are the artist name
            if result["artist_raw"] and result["artist_raw"].upper() in line.upper():
                continue
            # Skip signed/dated lines
            if re.match(r"^(Signed|Dated|Inscribed|Stamped)", line, re.I):
                break  # Title comes before signed lines
            # Skip medium lines
            if re.match(r"^(Oil|Acrylic|Tempera|Watercolour|Gouache|Ink|Pen|Pencil|Bronze|Mixed)", line, re.I):
                break
            # Skip year-only lines
            if re.match(r"^(Circa\s+)?\d{4}s?$", line.strip()):
                break
            # Skip dimension lines
            if re.search(r"\d+\.?\d*\s*x\s*\d+", line):
                break
            if not result["title"] and len(line) > 1 and len(line) < 200:
                result["title"] = line
                break

    # Fallback: try header section (skip nav elements)
    if not result["title"]:
        nav_words = {"advanced", "calendar", "contact", "login", "sign up", "home",
                     "auctions", "catalogue", "details", "previous", "next",
                     "quick zoom", "read more", "back to", "subscribe", "about",
                     "view catalogue", "e-catalogue", "view results"}
        header_lines = sections.get("header", [])
        found_artist = False
        for line in header_lines:
            if result["artist_raw"] and result["artist_raw"].upper() in line.upper():
                found_artist = True
                continue
            if not found_artist:
                continue
            if line.lower() in nav_words or len(line) < 2:
                continue
            if re.match(r"^(Lot|LOT)\s+\d+", line):
                continue
            if re.search(r"\(\d{4}\s*[-–]\s*\d{4}\)", line):
                continue
            if re.search(r"\(b\.\s*\d{4}\)", line):
                continue
            if not result["title"] and len(line) < 200:
                result["title"] = line
                break

    # --- Artwork Details ---
    details_lines = sections.get("ARTWORK DETAILS", [])
    details_text = "\n".join(details_lines)
    result["details_text"] = details_text

    # Medium: usually a line like "Oil on canvas" or "Tempera on card"
    for line in details_lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in [
            "oil on", "acrylic on", "tempera on", "watercolour", "watercolor",
            "gouache", "ink on", "pen on", "pencil on", "charcoal",
            "pastel", "mixed media", "bronze", "marble", "lithograph",
            "etching", "screenprint", "serigraph", "photograph",
            "on canvas", "on paper", "on board", "on card", "on silk",
        ]):
            result["medium_raw"] = line.strip()
            break

    # Dimensions line
    for line in details_lines:
        if re.search(r"\d+\.?\d*\s*[x×]\s*\d+\.?\d*\s*(?:in|cm)", line, re.I):
            result["dimensions_text"] = line.strip()
            break

    # Signed/dated info
    signed_lines = []
    for line in details_lines:
        line_lower = line.lower()
        if "signed" in line_lower or "dated" in line_lower or "inscribed" in line_lower:
            signed_lines.append(line.strip())
    result["signed_dated_text"] = "; ".join(signed_lines)

    # Year text
    for line in details_lines:
        if re.search(r"(?:circa\s+)?\d{4}s?\b", line, re.I):
            # Avoid dimension lines and signed lines
            if not re.search(r"\d+\.?\d*\s*[x×]", line):
                result["year_text"] = line.strip()
                break

    # --- Provenance ---
    prov_lines = sections.get("PROVENANCE", [])
    result["provenance_text"] = "\n".join(prov_lines)

    # --- Literature / Published ---
    lit_lines = sections.get("PUBLISHED", []) or sections.get("LITERATURE", [])
    result["literature_text"] = "\n".join(lit_lines)

    # --- Exhibition ---
    exh_lines = sections.get("EXHIBITED", []) or sections.get("EXHIBITION HISTORY", [])
    result["exhibited_text"] = "\n".join(exh_lines)

    # --- Estimate ---
    # Try from page text: "Rs X,XX,XXX - X,XX,XXX" or "$X,XXX - X,XXX"
    estimate_match = re.search(
        r"(?:Estimate|ESTIMATE)[:\s]*(?:Rs\.?\s*)?([\d,]+)\s*[-–]\s*([\d,]+)",
        page_text, re.I
    )
    if estimate_match:
        result["estimate_low_inr"] = _parse_indian_number(estimate_match.group(1))
        result["estimate_high_inr"] = _parse_indian_number(estimate_match.group(2))

    # Also look for USD estimates: "$X,XXX - X,XXX"
    usd_est_match = re.search(
        r"\$\s*([\d,]+)\s*[-–]\s*\$?\s*([\d,]+)",
        page_text
    )
    if usd_est_match:
        result["estimate_low_usd"] = _parse_indian_number(usd_est_match.group(1))
        result["estimate_high_usd"] = _parse_indian_number(usd_est_match.group(2))

    # Try from sections
    est_lines = sections.get("ESTIMATE", [])
    est_text = " ".join(est_lines)
    if est_text and not result["estimate_low_inr"]:
        inr_match = re.search(r"Rs\.?\s*([\d,]+)\s*[-–]\s*([\d,]+)", est_text)
        if inr_match:
            result["estimate_low_inr"] = _parse_indian_number(inr_match.group(1))
            result["estimate_high_inr"] = _parse_indian_number(inr_match.group(2))

    # Broad search for estimate in entire page text
    if not result["estimate_low_inr"] and not result["estimate_low_usd"]:
        # Pattern: "Rs 5,00,000 - 7,00,000 | $5,560 - 7,780"
        broad_est = re.search(
            r"Rs\.?\s*([\d,]+)\s*[-–]\s*([\d,]+)\s*\|\s*\$\s*([\d,]+)\s*[-–]\s*([\d,]+)",
            page_text
        )
        if broad_est:
            result["estimate_low_inr"] = _parse_indian_number(broad_est.group(1))
            result["estimate_high_inr"] = _parse_indian_number(broad_est.group(2))
            result["estimate_low_usd"] = _parse_indian_number(broad_est.group(3))
            result["estimate_high_usd"] = _parse_indian_number(broad_est.group(4))
        else:
            # Just INR
            inr_est = re.search(r"Rs\.?\s*([\d,]+)\s*[-–]\s*([\d,]+)", page_text)
            if inr_est:
                result["estimate_low_inr"] = _parse_indian_number(inr_est.group(1))
                result["estimate_high_inr"] = _parse_indian_number(inr_est.group(2))

    # --- Hammer price / Winning bid ---
    # Pattern: "Winning bid Rs 38,40,000 | $43,146"
    winning_match = re.search(
        r"(?:Winning\s+bid|SOLD\s+FOR|HAMMER\s+PRICE)[:\s]*Rs\.?\s*([\d,]+)\s*(?:\|\s*\$\s*([\d,]+))?",
        page_text, re.I
    )
    if winning_match:
        result["hammer_price_inr"] = _parse_indian_number(winning_match.group(1))
        result["hammer_currency"] = "INR"
        result["is_sold"] = True
        if winning_match.group(2):
            result["hammer_price_usd"] = _parse_indian_number(winning_match.group(2))

    # Also check for USD-only winning bid (some lots are USD-only)
    if not result["hammer_price_inr"]:
        usd_winning = re.search(
            r"(?:Winning\s+bid|SOLD\s+FOR|HAMMER\s+PRICE)[:\s]*\$\s*([\d,]+)",
            page_text, re.I
        )
        if usd_winning:
            result["hammer_price_usd"] = _parse_indian_number(usd_winning.group(1))
            result["hammer_currency"] = "USD"
            result["is_sold"] = True

    # Check sections for winning bid
    win_lines = sections.get("WINNING BID", [])
    if win_lines and not result["is_sold"]:
        win_text = " ".join(win_lines)
        inr_match = re.search(r"Rs\.?\s*([\d,]+)", win_text)
        if inr_match:
            result["hammer_price_inr"] = _parse_indian_number(inr_match.group(1))
            result["hammer_currency"] = "INR"
            result["is_sold"] = True
        usd_match = re.search(r"\$\s*([\d,]+)", win_text)
        if usd_match:
            result["hammer_price_usd"] = _parse_indian_number(usd_match.group(1))
            if not result["hammer_currency"]:
                result["hammer_currency"] = "USD"
            result["is_sold"] = True

    # --- Unsold / Withdrawn detection ---
    text_lower = page_text.lower()
    if "not sold" in text_lower or "unsold" in text_lower or "passed" in text_lower:
        result["is_sold"] = False
        result["hammer_price_inr"] = None
        result["hammer_price_usd"] = None
    if "withdrawn" in text_lower:
        result["is_withdrawn"] = True
        result["is_sold"] = False

    return result


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

def build_lot_record(
    lot_entry: dict,
    detail: dict,
    auction: dict,
) -> dict:
    """Assemble a complete lot record matching the project schema."""

    eid = auction["event_id"]
    work_id = lot_entry["lot_work_id"]
    lot_number = lot_entry["lot_number"]
    auction_date = auction.get("start_date", "")

    # Artist
    artist_raw = detail.get("artist_raw", "")
    artist_name, birth_year, death_year = parse_artist_name(artist_raw)
    if not artist_name:
        artist_name = artist_raw

    # Override birth/death if detail extracted them directly
    if detail.get("artist_birth_year"):
        birth_year = detail["artist_birth_year"]
    if detail.get("artist_death_year"):
        death_year = detail["artist_death_year"]

    # Title
    title = detail.get("title", "")

    # Medium
    medium_raw = detail.get("medium_raw", "") or detail.get("details_text", "")
    medium_category = parse_medium(medium_raw)

    # Dimensions
    dims_text = detail.get("dimensions_text", "") or detail.get("details_text", "")
    height_cm, width_cm = parse_dimensions(dims_text)

    # Year created
    year_text = detail.get("year_text", "") or detail.get("details_text", "")
    year_created = parse_year_created(year_text)

    # Signed / Dated
    signed_text = detail.get("signed_dated_text", "") or detail.get("details_text", "")
    signed = is_signed(signed_text)
    dated = is_dated(signed_text)

    # Provenance, Literature, Exhibition
    prov_text = detail.get("provenance_text", "")
    lit_text = detail.get("literature_text", "")
    exh_text = detail.get("exhibited_text", "")

    prov_count = count_provenance_entries(prov_text)
    lit_count = count_literature_entries(lit_text)
    exh_count = count_exhibition_entries(exh_text)

    # Estimates
    est_low_usd = detail.get("estimate_low_usd")
    est_high_usd = detail.get("estimate_high_usd")
    est_currency = "INR"

    # Convert INR estimates to USD if we don't have USD directly
    if not est_low_usd and detail.get("estimate_low_inr"):
        est_low_usd = to_usd(detail["estimate_low_inr"], "INR", auction_date)
    if not est_high_usd and detail.get("estimate_high_inr"):
        est_high_usd = to_usd(detail["estimate_high_inr"], "INR", auction_date)

    if detail.get("estimate_low_usd") and not detail.get("estimate_low_inr"):
        est_currency = "USD"

    # Hammer price
    hammer_price_usd = detail.get("hammer_price_usd")
    if not hammer_price_usd and detail.get("hammer_price_inr"):
        hammer_price_usd = to_usd(detail["hammer_price_inr"], "INR", auction_date)
    hammer_currency = detail.get("hammer_currency", "INR") or "INR"

    lot_url = detail.get("detail_url", "") or lot_entry.get("detail_url", "")

    return {
        "lot_id": f"saffronart_{work_id}",
        "lot_number": lot_number,
        "auction_id": f"saffronart_{eid}",
        "auction_title": auction.get("title", ""),
        "auction_date": auction_date,
        "auction_location": "Mumbai",  # Saffronart is Mumbai-based
        "sale_type": auction.get("sale_type", ""),
        "artist_name": artist_name,
        "artist_birth_year": birth_year,
        "artist_death_year": death_year,
        "title": title,
        "medium_raw": medium_raw[:500] if medium_raw else "",
        "medium_category": medium_category,
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
        "hammer_currency": hammer_currency,
        "is_sold": detail.get("is_sold", False),
        "is_withdrawn": detail.get("is_withdrawn", False),
        "image_url": detail.get("image_url", ""),
        "lot_url": lot_url,
    }


# ---------------------------------------------------------------------------
# Auction-level orchestrator
# ---------------------------------------------------------------------------

def scrape_auction(driver: webdriver.Chrome, auction: dict, scraped_lot_ids: set) -> list[dict]:
    """Scrape all lots from a single Saffronart auction."""
    eid = auction["event_id"]
    logger.info(f"\nScraping auction: {auction['title']} (eid={eid}, {auction['year']})")

    # Phase 2: Get lot listing from catalog page
    lot_entries = scrape_catalog_page(driver, auction)
    if not lot_entries:
        logger.warning(f"  No lots found for auction {eid}")
        return []

    records = []
    new_count = 0

    for lot_entry in tqdm(lot_entries, desc=f"  Lots (eid={eid})", leave=False):
        work_id = lot_entry["lot_work_id"]
        lot_id = f"saffronart_{work_id}"

        if lot_id in scraped_lot_ids:
            continue

        # Phase 3: Load lot detail page
        detail_url = lot_entry["detail_url"]
        detail = scrape_lot_detail(driver, detail_url, auction)

        # Build record
        record = build_lot_record(lot_entry, detail, auction)
        record["lot_url"] = detail_url
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

    logger.info(f"  -> Scraped {new_count} new lots from {auction['title']}")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("Scraping Saffronart seasonal auction lot details")
    logger.info("=" * 60)

    # Phase 1: Fetch qualifying auctions from API
    auctions = fetch_qualifying_auctions()
    if not auctions:
        logger.error("No qualifying auctions found!")
        sys.exit(1)

    # Load existing progress
    if "--fresh" in sys.argv:
        existing = []
        logger.info("Fresh run — ignoring previous progress")
    else:
        existing = load_ndjson(PROGRESS_FILE)
        logger.info(f"Loaded {len(existing)} previously scraped lots")
    scraped_lot_ids = {r["lot_id"] for r in existing}
    all_records = list(existing)

    driver = make_driver()

    try:
        for i, auction in enumerate(auctions, 1):
            logger.info(f"\n--- Auction {i}/{len(auctions)} ---")
            new_records = scrape_auction(driver, auction, scraped_lot_ids)
            all_records.extend(new_records)
            time.sleep(DELAY_BETWEEN_AUCTIONS)

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
    else:
        logger.warning("No lots scraped!")


if __name__ == "__main__":
    main()
