#!/usr/bin/env python3
"""Scrape lot details from Bonhams South Asian art auctions (2015 onwards).

Three-phase approach:
1. Discover South Asian art auctions from Bonhams search results page.
2. Use Bonhams lot listing API to enumerate lots per auction.
3. Load individual lot detail pages via Selenium for full data
   (estimates, provenance, literature, exhibition, medium, dimensions).

Output: data/raw/lots_bonhams.csv
"""

import re
import sys
import time
from pathlib import Path
from typing import Optional

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

logger = setup_logger(__name__, "scrape_lots_bonhams.log")

PROGRESS_FILE = DATA_RAW / "lots_bonhams_progress.ndjson"
OUTPUT_FILE = DATA_RAW / "lots_bonhams.csv"

SAVE_INTERVAL = 10
DELAY_BETWEEN_LOTS = 2.5
DELAY_BETWEEN_AUCTIONS = 3

BASE_URL = "https://www.bonhams.com"
LOTS_API = BASE_URL + "/api/v1/lots/{sale_no}/"
SEARCH_URL = BASE_URL + "/auctions/results/?query=south+asian+art"

# Minimum year for auctions to scrape
MIN_YEAR = 2015

# Known South Asian art auctions (auction_id, title, date, location, sale_type)
# Discovered from Bonhams search results; the scraper also discovers dynamically.
KNOWN_AUCTIONS = [
    # 2025
    (30642, "Modern and Contemporary South Asian Art", "2025-12-09", "London", "live"),
    (30694, "Modern & Contemporary South Asian Art Online", "2025-09-01", "London", "online"),
    (30553, "Modern and Contemporary South Asian Art", "2025-06-04", "London", "live"),
    (30527, "The Modern & Contemporary Middle Eastern and South Asian Art Online Auction", "2025-01-14", "London", "online"),
    # 2024
    (29352, "Modern and Contemporary South Asian Art", "2024-12-10", "London", "live"),
    (30723, "Diwali Online Only Sale", "2024-10-10", "London", "online"),
    (29291, "Modern and Contemporary South Asian Art Online", "2024-07-26", "London", "online"),
    (29193, "Modern and Contemporary South Asian Art", "2024-06-05", "London", "live"),
    (29191, "The Asia Edit: Contemporary Art from the South Asian Diaspora", "2024-01-17", "London", "online"),
    (29307, "The Jamini Roy Online Sale", "2024-01-05", "London", "online"),
    # 2023
    (28735, "Modern and Contemporary South Asian Art", "2023-11-14", "London", "live"),
    (29197, "Modern and Contemporary South Asian Art Online", "2023-07-21", "London", "online"),
    (28317, "Modern and Contemporary South Asian Art", "2023-06-06", "London", "live"),
    # 2022
    (27947, "Modern and Contemporary South Asian Art for a Pakistani Charity", "2022-05-24", "London", "live"),
    (27432, "Modern and Contemporary South Asian Art", "2022-05-24", "London", "live"),
    # 2021
    (27481, "Modern and Contemporary South Asian Art Online", "2021-11-09", "London", "online"),
    (27430, "Modern and Contemporary South Asian Art", "2021-10-25", "London", "live"),
    # 2020
    (26017, "Islamic and Indian Art", "2020-10-26", "London", "live"),
    (26016, "Islamic and Indian Art", "2020-06-11", "London", "live"),
    # 2019
    (25610, "Islamic and Indian Art", "2019-10-22", "London", "live"),
    (25609, "Islamic and Indian Art", "2019-06-05", "London", "live"),
    # 2018
    (25141, "Islamic and Indian Art", "2018-10-23", "London", "live"),
    (25140, "Islamic and Indian Art", "2018-06-05", "London", "live"),
    # 2017
    (24455, "Islamic and Indian Art", "2017-10-24", "London", "live"),
    (24454, "Islamic and Indian Art", "2017-04-25", "London", "live"),
    # 2016
    (23717, "Islamic and Indian Art", "2016-10-04", "London", "live"),
    (23716, "Islamic and Indian Art", "2016-04-19", "London", "live"),
    # 2015
    (22703, "Islamic and Indian Art", "2015-10-06", "London", "live"),
    (22702, "Islamic and Indian Art", "2015-04-21", "London", "live"),
]


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
        "button[class*='consent']",
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
# Phase 1: Discover auctions from search results
# ---------------------------------------------------------------------------

def discover_auctions_from_search(driver: webdriver.Chrome) -> list[dict]:
    """Load Bonhams search results and extract auction URLs.

    Supplements KNOWN_AUCTIONS with any newly discovered ones.
    """
    known_ids = {a[0] for a in KNOWN_AUCTIONS}
    discovered = []

    for page in range(1, 5):
        url = f"{SEARCH_URL}&page={page}" if page > 1 else SEARCH_URL
        logger.info(f"  Loading search results page {page}: {url}")
        try:
            driver.get(url)
            time.sleep(3)
            dismiss_overlays(driver)
        except Exception as e:
            logger.warning(f"  Failed to load search page {page}: {e}")
            break

        # Find auction links
        links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/auction/']")
        page_found = 0
        for link in links:
            href = link.get_attribute("href") or ""
            m = re.search(r"/auction/(\d+)/", href)
            if not m:
                continue
            auction_id = int(m.group(1))
            if auction_id in known_ids:
                continue
            known_ids.add(auction_id)

            title = link.text.strip() or link.get_attribute("title") or ""
            if not title:
                continue

            # Filter: must be South Asian art related
            title_lower = title.lower()
            if not any(kw in title_lower for kw in [
                "south asian", "indian art", "diwali", "jamini roy",
                "asia edit", "asian diaspora",
            ]):
                continue

            # Determine sale type
            sale_type = "online" if "online" in title_lower else "live"

            discovered.append({
                "auction_id": auction_id,
                "title": title,
                "date": "",  # Will be filled from lot pages
                "location": "London",
                "sale_type": sale_type,
            })
            page_found += 1

        if page_found == 0 and page > 1:
            break

    logger.info(f"  Discovered {len(discovered)} new auctions from search")
    return discovered


def build_auction_list(driver: webdriver.Chrome) -> list[dict]:
    """Build complete auction list from known + discovered auctions."""
    auctions = []

    # Add known auctions
    for aid, title, date, location, sale_type in KNOWN_AUCTIONS:
        year = int(date[:4]) if date else 0
        if year < MIN_YEAR:
            continue
        auctions.append({
            "auction_id": aid,
            "title": title,
            "date": date,
            "location": location,
            "sale_type": sale_type,
        })

    # Discover additional auctions
    try:
        new_auctions = discover_auctions_from_search(driver)
        existing_ids = {a["auction_id"] for a in auctions}
        for a in new_auctions:
            if a["auction_id"] not in existing_ids:
                auctions.append(a)
    except Exception as e:
        logger.warning(f"  Search discovery failed: {e}")

    # Sort by date descending
    auctions.sort(key=lambda x: x.get("date", ""), reverse=True)

    logger.info(f"  Total auctions to scrape: {len(auctions)}")
    for a in auctions:
        logger.info(f"    {a['date']} | {a['title']} (id={a['auction_id']})")

    return auctions


# ---------------------------------------------------------------------------
# Phase 2: Fetch lot listing from Bonhams API
# ---------------------------------------------------------------------------

def _parse_gbp_amount(val: str) -> Optional[float]:
    """Parse '57,550' or '45,000' string to float."""
    if not val:
        return None
    val = re.sub(r"[£€$\s]", "", val.strip())
    val = val.replace(",", "")
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def fetch_lot_listing(auction_id: int) -> list[dict]:
    """Fetch lot listing from Bonhams API for a given auction.

    Returns list of dicts with basic lot info: lot_number, sLotStatus,
    hammer_price_gbp, estimate_low_gbp, estimate_high_gbp, image_url,
    lot_url, artist_title_raw, is_withdrawn.
    """
    all_lots = []
    page = 1
    per_page = 100

    while True:
        api_url = LOTS_API.format(sale_no=auction_id)
        params = {"page": page, "per_page": per_page, "category": "list"}

        try:
            resp = requests.get(api_url, params=params, timeout=30, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                "Accept": "application/json",
            })
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"  API request failed for auction {auction_id} page {page}: {e}")
            break

        lots = data.get("lots", [])
        if not lots:
            break

        for lot in lots:
            lot_no = lot.get("iSaleLotNo")
            if lot_no is None:
                continue

            status = lot.get("sLotStatus", "")
            banners = lot.get("banners", {})
            is_withdrawn = banners.get("withdrawn", False) or banners.get("cancelled", False)

            # Hammer price (without premium) for sold lots
            hammer_prices = lot.get("hammer_prices", {})
            prices_no_premium = hammer_prices.get("prices_without_premium", [])
            hammer_gbp = None
            if prices_no_premium:
                hammer_gbp = _parse_gbp_amount(prices_no_premium[0].get("value", ""))

            # Hammer price with premium
            prices_with_premium = hammer_prices.get("prices", [])
            hammer_premium_gbp = None
            if prices_with_premium:
                hammer_premium_gbp = _parse_gbp_amount(prices_with_premium[0].get("value", ""))

            # Estimates (shown for unsold lots in API, but we get from detail page too)
            estimates = lot.get("high_low_estimates", {})
            est_prices = estimates.get("prices", [])
            est_low_gbp = None
            est_high_gbp = None
            if est_prices:
                est_low_gbp = _parse_gbp_amount(est_prices[0].get("low", ""))
                est_high_gbp = _parse_gbp_amount(est_prices[0].get("high", ""))

            # Image
            image_data = lot.get("image", {})
            image_url = image_data.get("url", "")

            # URL for lot detail page
            lot_url_path = lot.get("url", "")

            # Description (artist + title combined)
            desc = lot.get("sDesc", "")

            # Styled title for parsing
            styled = lot.get("styled_title", "")

            all_lots.append({
                "lot_number": lot_no,
                "lot_status": status,
                "is_withdrawn": is_withdrawn,
                "hammer_price_gbp": hammer_gbp,
                "hammer_premium_gbp": hammer_premium_gbp,
                "estimate_low_gbp": est_low_gbp,
                "estimate_high_gbp": est_high_gbp,
                "image_url": image_url,
                "lot_url_path": lot_url_path,
                "desc": desc,
                "styled_title": styled,
                "sale_no": lot.get("sale_no"),
                "online_only": lot.get("online_only", False),
            })

        total_lots = data.get("total_lots", 0)
        if len(all_lots) >= total_lots or len(lots) < per_page:
            break
        page += 1

    all_lots.sort(key=lambda x: x["lot_number"])
    return all_lots


# ---------------------------------------------------------------------------
# Phase 3: Scrape individual lot detail pages via Selenium
# ---------------------------------------------------------------------------

def _parse_styled_title(styled: str) -> tuple[str, str, Optional[int], Optional[int]]:
    """Parse Bonhams styled_title HTML to extract artist name, title, dates.

    Format: <div class="firstLine">Artist Name</div>
            <div class="secondLine">(Country, YYYY-YYYY)</div>
            <div class="otherLine"><i>Title</i></div>
    """
    artist = ""
    title = ""
    birth_year = None
    death_year = None

    # Extract artist from firstLine
    m = re.search(r'class="firstLine"[^>]*>([^<]+)<', styled)
    if m:
        artist = m.group(1).strip()

    # Extract dates from secondLine
    m = re.search(r'class="secondLine"[^>]*>([^<]+)<', styled)
    if m:
        dates_str = m.group(1).strip()
        dm = re.search(r"\((?:.*?,\s*)?(\d{4})\s*[-–]\s*(\d{4})\)", dates_str)
        if dm:
            birth_year = int(dm.group(1))
            death_year = int(dm.group(2))
        else:
            dm = re.search(r"\((?:.*?[Bb]\.\s*)?(\d{4})\)", dates_str)
            if dm:
                birth_year = int(dm.group(1))

    # Extract title from otherLine (may be in <i> tags)
    m = re.search(r'class="otherLine"[^>]*>(?:<i>)?([^<]+)', styled)
    if m:
        title = m.group(1).strip()

    return artist, title, birth_year, death_year


def _build_lot_detail_url(auction_id: int, lot_number: int) -> str:
    """Build Bonhams lot detail URL."""
    return f"{BASE_URL}/auction/{auction_id}/lot/{lot_number}/"


def scrape_lot_detail(driver: webdriver.Chrome, auction_id: int, lot_number: int) -> dict:
    """Load a Bonhams lot detail page and extract all fields.

    Extracts: medium, dimensions, signed/dated info, estimates,
    provenance, literature, exhibition history from the rendered page text.
    """
    result = {
        "medium_raw": "",
        "dimensions_text": "",
        "signed_dated_text": "",
        "year_text": "",
        "provenance_text": "",
        "literature_text": "",
        "exhibited_text": "",
        "estimate_low_gbp": None,
        "estimate_high_gbp": None,
        "hammer_price_gbp": None,
        "detail_image_url": "",
    }

    detail_url = _build_lot_detail_url(auction_id, lot_number)

    try:
        driver.get(detail_url)
        time.sleep(DELAY_BETWEEN_LOTS)
        dismiss_overlays(driver)
    except Exception as e:
        logger.warning(f"    Failed to load lot page {detail_url}: {e}")
        return result

    # Wait for content
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(1)
    except Exception:
        pass

    page_text = ""
    try:
        page_text = driver.find_element(By.TAG_NAME, "body").text
    except Exception:
        logger.warning(f"    Could not get page text for lot {lot_number}")
        return result

    page_source = driver.page_source

    # --- Extract estimate from page source JSON data ---
    # The page embeds per-lot estimates as dEstimateLow / dEstimateHigh in
    # the main lot JavaScript object.  The "highlights" carousel also
    # contains estimate data for OTHER lots, so we must target the main
    # lot object specifically.  The main lot object has the pattern:
    #   "brand":"bonhams","dEstimateHigh":NNN,"dEstimateLow":NNN
    # We use "brand" as an anchor to avoid matching highlights entries.
    est_main = re.search(
        r'"brand"\s*:\s*"[^"]*"\s*,\s*"dEstimateHigh"\s*:\s*(\d+(?:\.\d+)?)'
        r'\s*,\s*"dEstimateLow"\s*:\s*(\d+(?:\.\d+)?)',
        page_source,
    )
    if est_main:
        result["estimate_high_gbp"] = float(est_main.group(1))
        result["estimate_low_gbp"] = float(est_main.group(2))
    else:
        # Fallback: look for the lot-specific object using iSaleLotNo anchor
        est_lot = re.search(
            r'"iSaleLotNo"\s*:\s*' + str(lot_number)
            + r'\b.*?"dEstimateHigh"\s*:\s*(\d+(?:\.\d+)?)'
            + r'\s*,\s*"dEstimateLow"\s*:\s*(\d+(?:\.\d+)?)',
            page_source,
        )
        if est_lot:
            result["estimate_high_gbp"] = float(est_lot.group(1))
            result["estimate_low_gbp"] = float(est_lot.group(2))
        else:
            # Final fallback: try visible page text with "Estimate" label
            est_match = re.search(
                r"[Ee]stimate[:\s]*£\s*([\d,]+)\s*[-–]\s*£?\s*([\d,]+)",
                page_text,
            )
            if est_match:
                result["estimate_low_gbp"] = _parse_gbp_amount(est_match.group(1))
                result["estimate_high_gbp"] = _parse_gbp_amount(est_match.group(2))

    # --- Hammer price from page text ---
    hammer_match = re.search(
        r"(?:[Ss]old\s+for|[Hh]ammer\s+[Pp]rice)[:\s]*£\s*([\d,]+)",
        page_text,
    )
    if hammer_match:
        result["hammer_price_gbp"] = _parse_gbp_amount(hammer_match.group(1))

    # --- Extract catalog description sections ---
    # Bonhams lot pages have structured sections in the body text.
    # Key sections: medium/dimensions line, Signed info, Provenance,
    # Literature, Exhibited, Footnotes.

    # Medium: look for common medium patterns in page text
    medium_patterns = [
        r"((?:oil|acrylic|gouache|watercolou?r|tempera|ink|pencil|charcoal|pastel|mixed media|bronze|marble|photograph|lithograph|etching|screenprint)\s+(?:on\s+\w+)?[^,\n]*(?:,\s*framed)?)",
    ]
    for pat in medium_patterns:
        mm = re.search(pat, page_text, re.I)
        if mm:
            result["medium_raw"] = mm.group(1).strip()
            break

    # If that didn't work, try to find the catalog description block
    # which usually has format: "medium\ndimensions\nsigned info"
    if not result["medium_raw"]:
        # Look for lines that are typical medium descriptions
        for line in page_text.split("\n"):
            line_s = line.strip().lower()
            if any(kw in line_s for kw in [
                "oil on", "acrylic on", "gouache on", "watercolour on",
                "watercolor on", "ink on", "pencil on", "tempera on",
                "charcoal on", "pastel on", "mixed media", "on canvas",
                "on paper", "on board",
            ]):
                result["medium_raw"] = line.strip()
                break

    # Dimensions: look for "XX x XX cm" or "XX x XX in"
    dim_match = re.search(
        r"(\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?\s*cm)",
        page_text, re.I,
    )
    if dim_match:
        result["dimensions_text"] = dim_match.group(1).strip()
    else:
        dim_match = re.search(
            r"(\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?\s*in)",
            page_text, re.I,
        )
        if dim_match:
            result["dimensions_text"] = dim_match.group(1).strip()

    # Signed/Dated info
    signed_lines = []
    for line in page_text.split("\n"):
        line_s = line.strip()
        line_lower = line_s.lower()
        if re.match(r"^signed\b", line_lower) or re.match(r"^dated\b", line_lower):
            signed_lines.append(line_s)
        elif "signed and dated" in line_lower and len(line_s) < 200:
            signed_lines.append(line_s)
    result["signed_dated_text"] = "; ".join(signed_lines)

    # --- Provenance, Literature, Exhibition ---
    # These appear as sections in the page text, often under "Footnotes"
    # or directly with headers like "Provenance", "Literature", "Exhibited"
    _extract_footnote_sections(page_text, result)

    # Also try to extract from page source (HTML) where sections have bold headers
    if not result["provenance_text"]:
        _extract_sections_from_html(page_source, result)

    # --- Image URL (high res) ---
    img_match = re.search(
        r'(https?://img\d*\.bonhams\.com/image\?src=[^"\'&\s]+)',
        page_source,
    )
    if img_match:
        result["detail_image_url"] = img_match.group(1)

    # Also try images1.bonhams.com
    if not result["detail_image_url"]:
        img_match = re.search(
            r'(https?://images?\d*\.bonhams\.com/image\?src=[^"\'&\s]+)',
            page_source,
        )
        if img_match:
            result["detail_image_url"] = img_match.group(1)

    return result


def _extract_footnote_sections(page_text: str, result: dict) -> None:
    """Extract Provenance, Literature, Exhibited from page text.

    Bonhams lot pages typically have these as labeled sections:
    Provenance
    collector A; collector B; ...

    Literature
    reference 1; reference 2; ...

    Exhibited
    exhibition 1; exhibition 2; ...
    """
    lines = page_text.split("\n")
    sections = {}
    current_section = None

    # Section headers to look for (case-insensitive)
    section_map = {
        "provenance": "provenance",
        "literature": "literature",
        "exhibited": "exhibited",
        "exhibition history": "exhibited",
        "exhibitions": "exhibited",
    }

    # Stop markers: sections that come after provenance/lit/exh
    stop_markers = {
        "condition report", "lot notes", "lot essay", "footnotes",
        "special notice", "important notice", "please note",
        "artist's resale right", "catalogue note", "sale room notice",
        "additional", "vat", "share this lot",
    }

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        lower = stripped.lower()

        # Check if this line is a section header
        matched_section = None
        for header, sec_name in section_map.items():
            if lower == header or lower == header + ":":
                matched_section = sec_name
                break

        if matched_section:
            current_section = matched_section
            sections[current_section] = []
            continue

        # Check for stop markers
        if any(lower.startswith(sm) for sm in stop_markers):
            current_section = None
            continue

        # Append content to current section
        if current_section and current_section in sections:
            # Skip very short lines that are likely UI elements
            if len(stripped) > 1:
                sections[current_section].append(stripped)

    if "provenance" in sections:
        # Provenance entries are often separated by semicolons on one line
        # or as separate lines. Join them with newlines.
        prov_text = "\n".join(sections["provenance"])
        # Also split semicolon-separated entries into lines
        if "\n" not in prov_text and ";" in prov_text:
            prov_text = prov_text.replace("; ", "\n")
        result["provenance_text"] = prov_text.strip()

    if "literature" in sections:
        result["literature_text"] = "\n".join(sections["literature"]).strip()

    if "exhibited" in sections:
        result["exhibited_text"] = "\n".join(sections["exhibited"]).strip()


def _extract_sections_from_html(page_source: str, result: dict) -> None:
    """Fallback: extract sections from HTML source using bold/strong tags.

    Bonhams sometimes uses <strong>Provenance</strong> or
    **Provenance** in their footnote HTML.
    """
    # Try to find provenance in HTML
    prov_match = re.search(
        r'(?:<strong>|<b>|\*\*)Provenance(?:</strong>|</b>|\*\*)[:\s]*(.+?)(?=<strong>|<b>|\*\*|$)',
        page_source, re.I | re.DOTALL,
    )
    if prov_match and not result["provenance_text"]:
        text = re.sub(r"<br\s*/?>", "\n", prov_match.group(1))
        text = re.sub(r"<[^>]+>", "", text)
        text = text.strip()
        if text:
            result["provenance_text"] = text

    # Literature
    lit_match = re.search(
        r'(?:<strong>|<b>|\*\*)Literature(?:</strong>|</b>|\*\*)[:\s]*(.+?)(?=<strong>|<b>|\*\*|$)',
        page_source, re.I | re.DOTALL,
    )
    if lit_match and not result["literature_text"]:
        text = re.sub(r"<br\s*/?>", "\n", lit_match.group(1))
        text = re.sub(r"<[^>]+>", "", text)
        text = text.strip()
        if text:
            result["literature_text"] = text

    # Exhibited
    exh_match = re.search(
        r'(?:<strong>|<b>|\*\*)Exhibited(?:</strong>|</b>|\*\*)[:\s]*(.+?)(?=<strong>|<b>|\*\*|$)',
        page_source, re.I | re.DOTALL,
    )
    if exh_match and not result["exhibited_text"]:
        text = re.sub(r"<br\s*/?>", "\n", exh_match.group(1))
        text = re.sub(r"<[^>]+>", "", text)
        text = text.strip()
        if text:
            result["exhibited_text"] = text


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

def build_lot_record(
    lot_api: dict,
    detail: dict,
    auction: dict,
) -> dict:
    """Assemble a complete lot record matching the project schema (34 columns)."""

    auction_id = auction["auction_id"]
    lot_number = lot_api["lot_number"]
    auction_date = auction.get("date", "")

    # --- Artist, title, dates from styled_title ---
    styled = lot_api.get("styled_title", "")
    artist_name, title, birth_year, death_year = _parse_styled_title(styled)

    # Fallback: parse from desc field
    if not artist_name:
        desc = lot_api.get("desc", "")
        artist_name, birth_year, death_year = parse_artist_name(desc)
        if not title:
            # Title is usually after artist name in desc
            parts = desc.split(artist_name, 1) if artist_name else ["", desc]
            if len(parts) > 1:
                remainder = parts[1].strip()
                # Remove date part like "(1924-2002)"
                remainder = re.sub(r"^\s*\(.*?\)\s*", "", remainder)
                title = remainder.strip()

    # Medium
    medium_raw = detail.get("medium_raw", "")
    medium_category = parse_medium(medium_raw)

    # Dimensions
    dims_text = detail.get("dimensions_text", "")
    height_cm, width_cm = parse_dimensions(dims_text)

    # Year created from signed/dated text or medium line
    year_text = detail.get("signed_dated_text", "") or detail.get("medium_raw", "")
    year_created = parse_year_created(year_text)

    # Signed / Dated
    signed_text = detail.get("signed_dated_text", "")
    signed = is_signed(signed_text)
    dated = is_dated(signed_text)

    # Provenance, Literature, Exhibition
    prov_text = detail.get("provenance_text", "")
    lit_text = detail.get("literature_text", "")
    exh_text = detail.get("exhibited_text", "")

    prov_count = count_provenance_entries(prov_text)
    lit_count = count_literature_entries(lit_text)
    exh_count = count_exhibition_entries(exh_text)

    # --- Estimates ---
    # Prefer detail page estimates, fallback to API
    est_low_gbp = detail.get("estimate_low_gbp") or lot_api.get("estimate_low_gbp")
    est_high_gbp = detail.get("estimate_high_gbp") or lot_api.get("estimate_high_gbp")

    est_low_usd = to_usd(est_low_gbp, "GBP", auction_date)
    est_high_usd = to_usd(est_high_gbp, "GBP", auction_date)

    # --- Hammer price ---
    # API provides hammer without premium; detail page may have it too
    lot_status = lot_api.get("lot_status", "")
    is_lot_sold = lot_status == "SOLD"
    is_lot_withdrawn = lot_api.get("is_withdrawn", False)

    hammer_gbp = None
    if is_lot_sold:
        # Use API hammer price (without premium) — this is the true hammer price
        hammer_gbp = lot_api.get("hammer_price_gbp")
        # Fallback to detail page
        if hammer_gbp is None:
            hammer_gbp = detail.get("hammer_price_gbp")

    hammer_usd = to_usd(hammer_gbp, "GBP", auction_date)

    # Image URL: prefer detail page (higher res), fallback to API
    image_url = detail.get("detail_image_url", "") or lot_api.get("image_url", "")

    # Lot URL
    lot_url = _build_lot_detail_url(auction_id, lot_number)

    return {
        "lot_id": f"bonhams_{auction_id}_{lot_number}",
        "lot_number": lot_number,
        "auction_id": f"bonhams_{auction_id}",
        "auction_title": auction.get("title", ""),
        "auction_date": auction_date,
        "auction_location": auction.get("location", "London"),
        "sale_type": auction.get("sale_type", "live"),
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
        "estimate_currency": "GBP",
        "hammer_price_usd": hammer_usd,
        "hammer_currency": "GBP" if hammer_gbp else "",
        "is_sold": is_lot_sold,
        "is_withdrawn": is_lot_withdrawn,
        "image_url": image_url,
        "lot_url": lot_url,
    }


# ---------------------------------------------------------------------------
# Auction-level orchestrator
# ---------------------------------------------------------------------------

def scrape_auction(
    driver: webdriver.Chrome,
    auction: dict,
    scraped_lot_ids: set,
) -> list[dict]:
    """Scrape all lots from a single Bonhams auction."""
    auction_id = auction["auction_id"]
    logger.info(f"\nScraping auction: {auction['title']} (id={auction_id})")

    # Phase 2: Get lot listing from API
    lot_listing = fetch_lot_listing(auction_id)
    if not lot_listing:
        logger.warning(f"  No lots found for auction {auction_id}")
        return []

    logger.info(f"  Found {len(lot_listing)} lots from API")

    records = []
    new_count = 0

    for lot_api in tqdm(lot_listing, desc=f"  Lots (id={auction_id})", leave=False):
        lot_number = lot_api["lot_number"]
        lot_id = f"bonhams_{auction_id}_{lot_number}"

        if lot_id in scraped_lot_ids:
            continue

        # Skip withdrawn lots
        if lot_api.get("is_withdrawn", False):
            logger.info(f"    Lot {lot_number}: withdrawn — skipping detail page")
            # Still create a minimal record
            record = build_lot_record(lot_api, {}, auction)
            records.append(record)
            scraped_lot_ids.add(lot_id)
            new_count += 1
            continue

        # Phase 3: Load lot detail page
        detail = scrape_lot_detail(driver, auction_id, lot_number)

        # Build record
        record = build_lot_record(lot_api, detail, auction)
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
    logger.info("Scraping Bonhams South Asian art auction lot details")
    logger.info("=" * 60)

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

    driver = make_driver()

    try:
        # Phase 1: Build auction list
        auctions = build_auction_list(driver)
        if not auctions:
            logger.error("No auctions found!")
            sys.exit(1)

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
