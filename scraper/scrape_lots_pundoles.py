#!/usr/bin/env python3
"""Scrape lot details from Pundole's fine art auctions.

Three-phase approach:
1. Load past auctions page, extract auction links, filter to fine art only.
2. For each auction, load the lots listing and collect lot URLs.
3. For each lot, load the detail page and scrape all fields.

The site (auctions.pundoles.com) is an Ember.js SPA on AuctionMobility,
so all pages require Selenium to render JavaScript.

Output: data/raw/lots_pundoles.csv
"""

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
)
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

logger = setup_logger(__name__, "scrape_lots_pundoles.log")

PROGRESS_FILE = DATA_RAW / "lots_pundoles_progress.ndjson"
OUTPUT_FILE = DATA_RAW / "lots_pundoles.csv"

SAVE_INTERVAL = 20
DELAY_BETWEEN_LOTS = 2.5
DELAY_BETWEEN_AUCTIONS = 4
PAGE_LOAD_WAIT = 8

BASE_URL = "https://auctions.pundoles.com"
PAST_AUCTIONS_URL = f"{BASE_URL}/auctions/past"

# ---------------------------------------------------------------------------
# Auction title filtering
# ---------------------------------------------------------------------------

# Titles containing any of these (case-insensitive) are fine art auctions
INCLUDE_KEYWORDS = [
    "fine art sale",
    "fine art auction",
    "summer fine art",
    "works on paper",
    "sculpture sale",
    "art of india",
    "mf husain",
    "m.f. husain",
    "m. f. husain",
    "husain an artist",
]

# Titles containing any of these are NOT fine art auctions — skip them
EXCLUDE_KEYWORDS = [
    "cricket",
    "memorabilia",
    "decorative art",
    "classical painting",
    "classical indian painting",
]


def is_fine_art_auction(title: str) -> bool:
    """Determine if an auction title qualifies as fine art."""
    t = title.lower().strip()

    # Exclude first
    for kw in EXCLUDE_KEYWORDS:
        if kw in t:
            return False

    # Include if matches
    for kw in INCLUDE_KEYWORDS:
        if kw in t:
            return True

    return False


# ---------------------------------------------------------------------------
# Driver setup
# ---------------------------------------------------------------------------

def make_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-gpu")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    driver.set_page_load_timeout(60)
    return driver


def dismiss_overlays(driver: webdriver.Chrome) -> None:
    """Try to close cookie banners, modals, etc."""
    selectors = [
        "#onetrust-accept-btn-handler",
        "[data-testid='cookie-accept']",
        "button[class*='cookie']",
        "[aria-label='Close']",
        "[class*='modal'] button[class*='close']",
        ".close-button",
        "button.accept",
    ]
    for sel in selectors:
        try:
            btn = driver.find_element(By.CSS_SELECTOR, sel)
            if btn.is_displayed():
                btn.click()
                time.sleep(0.3)
        except Exception:
            pass


def safe_get(driver: webdriver.Chrome, url: str, wait_seconds: float = PAGE_LOAD_WAIT) -> None:
    """Navigate to URL with retry logic."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            driver.get(url)
            time.sleep(wait_seconds)
            dismiss_overlays(driver)
            return
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"  Retry {attempt+1} loading {url}: {e}")
                time.sleep(5)
            else:
                raise


def scroll_to_bottom(driver: webdriver.Chrome, max_scrolls: int = 30, pause: float = 2.0) -> None:
    """Scroll down repeatedly to trigger lazy-loading / infinite scroll."""
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


def click_load_more(driver: webdriver.Chrome, max_clicks: int = 30) -> None:
    """Click 'Load More' button repeatedly until it disappears."""
    for _ in range(max_clicks):
        try:
            # AuctionMobility uses various load-more button patterns
            load_more = None
            for sel in [
                "button.load-more",
                "a.load-more",
                "[class*='load-more']",
                "button[class*='LoadMore']",
                "[data-action*='load-more']",
            ]:
                try:
                    el = driver.find_element(By.CSS_SELECTOR, sel)
                    if el.is_displayed():
                        load_more = el
                        break
                except NoSuchElementException:
                    continue

            # Also try by text content
            if not load_more:
                buttons = driver.find_elements(By.TAG_NAME, "button")
                buttons += driver.find_elements(By.TAG_NAME, "a")
                for btn in buttons:
                    try:
                        txt = btn.text.strip().lower()
                        if "load more" in txt:
                            load_more = btn
                            break
                    except StaleElementReferenceException:
                        continue

            if not load_more:
                break

            driver.execute_script("arguments[0].scrollIntoView(true);", load_more)
            time.sleep(0.5)
            load_more.click()
            time.sleep(2.5)
        except Exception:
            break


# ---------------------------------------------------------------------------
# Phase 1: Discover and filter past auctions
# ---------------------------------------------------------------------------

def discover_auctions(driver: webdriver.Chrome) -> list[dict]:
    """Load past auctions page and extract all auction links."""
    logger.info(f"Loading past auctions page: {PAST_AUCTIONS_URL}")
    safe_get(driver, PAST_AUCTIONS_URL, wait_seconds=PAGE_LOAD_WAIT)

    # Scroll and click load-more to reveal all 65 auctions
    click_load_more(driver, max_clicks=40)
    scroll_to_bottom(driver, max_scrolls=20, pause=2.0)
    click_load_more(driver, max_clicks=20)

    time.sleep(2)

    # Extract auction links — on AuctionMobility, auction cards are typically
    # <a> tags linking to /auctions/{id}/{slug}
    auctions = []
    seen_urls = set()

    links = driver.find_elements(By.TAG_NAME, "a")
    for link in links:
        try:
            href = link.get_attribute("href") or ""
            text = link.text.strip()
        except StaleElementReferenceException:
            continue

        if not href or not text:
            continue

        # Match auction URL pattern: /auctions/{id}/{slug}
        m = re.match(
            r"https?://auctions\.pundoles\.com/auctions/([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)",
            href,
        )
        if not m:
            continue

        # Skip the "past" page itself
        if m.group(2) == "past":
            continue

        # Avoid duplicates
        if href in seen_urls:
            continue
        seen_urls.add(href)

        auction_id = m.group(1)
        slug = m.group(2)

        # Try to extract date from visible text near the link
        # The card text usually has title + date
        date_str = _extract_date_from_text(text)

        # Derive title from URL slug (more reliable than link text which is often just "71 LOTS")
        slug_title = slug.replace("-", " ").strip()
        slug_title = re.sub(r"\s+[mt]\d{4}$", "", slug_title, flags=re.I)
        slug_title = slug_title.title()

        auctions.append({
            "auction_id": auction_id,
            "slug": slug,
            "url": href,
            "raw_text": text,
            "title": slug_title,
            "date": date_str,
        })

    logger.info(f"  Found {len(auctions)} total past auctions")
    for a in auctions:
        logger.debug(f"    {a['title']} — {a['url']}")

    # Filter to fine art
    fine_art = [a for a in auctions if is_fine_art_auction(a["title"])]
    logger.info(f"  Fine art auctions after filtering: {len(fine_art)}")
    for a in fine_art:
        logger.info(f"    {a['title']} | {a['date']} | {a['url']}")

    return fine_art


def _extract_title_from_text(text: str) -> str:
    """Extract auction title from card text, stripping date and other info."""
    # The card text often has the title on the first line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return text.strip()

    # Title is typically the first substantive line
    title = lines[0]

    # If first line looks like a date, try second line
    if re.match(r"^\d{1,2}\s+\w+\s+\d{4}$", title):
        title = lines[1] if len(lines) > 1 else title

    return title.strip()


def _extract_date_from_text(text: str) -> str:
    """Extract date from card text, return as YYYY-MM-DD if possible."""
    # Look for patterns like "15 September 2023", "September 15, 2023"
    # or "15 Sep 2023"
    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "jun": "06", "jul": "07", "aug": "08", "sep": "09",
        "oct": "10", "nov": "11", "dec": "12",
    }

    # Pattern: "DD Month YYYY"
    m = re.search(r"(\d{1,2})\s+(\w+)\s+(\d{4})", text)
    if m:
        day = int(m.group(1))
        month_name = m.group(2).lower()
        year = m.group(3)
        month_num = months.get(month_name)
        if month_num:
            return f"{year}-{month_num}-{day:02d}"

    # Pattern: "Month DD, YYYY"
    m = re.search(r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", text)
    if m:
        month_name = m.group(1).lower()
        day = int(m.group(2))
        year = m.group(3)
        month_num = months.get(month_name)
        if month_num:
            return f"{year}-{month_num}-{day:02d}"

    # Fallback: just extract year
    m = re.search(r"(20\d{2})", text)
    if m:
        return f"{m.group(1)}-01-01"

    return ""


# ---------------------------------------------------------------------------
# Phase 2: Scrape lot listing from auction page
# ---------------------------------------------------------------------------

def scrape_lot_listing(driver: webdriver.Chrome, auction: dict) -> list[dict]:
    """Load an auction page and extract all lot entries (links + basic info)."""
    url = auction["url"]
    logger.info(f"  Loading auction page: {url}")
    safe_get(driver, url, wait_seconds=PAGE_LOAD_WAIT)

    # Wait for lots to render
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "[class*='lot-'], a[href*='/lot']")
            )
        )
    except TimeoutException:
        logger.warning("  Timed out waiting for lots — trying anyway")

    # Click load-more to reveal all lots
    click_load_more(driver, max_clicks=50)
    scroll_to_bottom(driver, max_scrolls=30, pause=1.5)
    click_load_more(driver, max_clicks=20)
    time.sleep(2)

    # Extract lot links
    # Pundole's lot detail URLs: /lots/view/{lot_id}/{slug}
    lot_entries = []
    seen_lot_urls = set()

    links = driver.find_elements(By.TAG_NAME, "a")
    for link in links:
        try:
            href = link.get_attribute("href") or ""
            text = link.text.strip()
        except StaleElementReferenceException:
            continue

        if not href:
            continue

        # Match lot URL pattern: /lots/view/{id}/{slug}
        lot_match = re.search(
            r"/lots/view/([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)",
            href,
        )
        if not lot_match:
            continue

        if href in seen_lot_urls:
            continue
        seen_lot_urls.add(href)

        lot_id_str = lot_match.group(1)
        lot_slug = lot_match.group(2)

        # Try to extract lot number from text or slug
        lot_number = _extract_lot_number(text, lot_slug)

        lot_entries.append({
            "lot_url": href,
            "lot_slug": lot_slug,
            "lot_number": lot_number,
            "raw_text": text,
        })

    # Sort by lot number
    lot_entries.sort(key=lambda x: x.get("lot_number") or 9999)

    logger.info(f"  Found {len(lot_entries)} lots")
    return lot_entries


def _extract_lot_number(text: str, slug: str) -> int | None:
    """Extract lot number from visible text or slug."""
    # From text: "Lot 1", "LOT 42", "1.", etc.
    m = re.search(r"(?:lot\s*#?\s*)?(\d+)", text, re.I)
    if m:
        return int(m.group(1))

    # From slug: might just be a number
    m = re.match(r"^(\d+)$", slug)
    if m:
        return int(m.group(1))

    # Slug might contain lot number
    m = re.search(r"(\d+)", slug)
    if m:
        return int(m.group(1))

    return None


# ---------------------------------------------------------------------------
# Phase 3: Scrape individual lot detail pages
# ---------------------------------------------------------------------------

def _parse_indian_number(s: str) -> float | None:
    """Parse Indian-format numbers like '38,40,000' or '1,20,00,000' to float."""
    if not s:
        return None
    # Remove currency symbols and whitespace
    s = re.sub(r"[Rs₹$€£\s]", "", s.strip())
    # Remove commas
    s = s.replace(",", "")
    # Remove any trailing dots or non-numeric
    s = s.strip(".")
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def scrape_lot_detail(driver: webdriver.Chrome, lot_url: str) -> dict:
    """Load an individual lot detail page and extract all fields.

    AuctionMobility lot pages typically show:
    - Lot image
    - Artist name (with birth/death years)
    - Title (often in italics)
    - Medium, dimensions
    - Estimate range
    - Hammer price / result (sold, unsold, withdrawn)
    - Provenance, Literature, Exhibited sections
    """
    result = {
        "artist_raw": "",
        "artist_birth_year": None,
        "artist_death_year": None,
        "title": "",
        "medium_raw": "",
        "dimensions_text": "",
        "year_text": "",
        "signed_dated_text": "",
        "provenance_text": "",
        "literature_text": "",
        "exhibited_text": "",
        "estimate_low_inr": None,
        "estimate_high_inr": None,
        "hammer_price_inr": None,
        "is_sold": False,
        "is_withdrawn": False,
        "image_url": "",
    }

    try:
        safe_get(driver, lot_url, wait_seconds=DELAY_BETWEEN_LOTS)
    except Exception as e:
        logger.warning(f"    Failed to load lot page {lot_url}: {e}")
        return result

    # Wait for lot content to render
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "h1"))
        )
        time.sleep(1)
    except TimeoutException:
        # Try waiting for any content
        time.sleep(3)

    page_text = ""
    page_source = ""
    try:
        page_text = driver.find_element(By.TAG_NAME, "body").text
        page_source = driver.page_source
    except Exception:
        logger.warning("    Could not read page content")
        return result

    # --- Image URL ---
    try:
        # Look for lot image in <img> tags
        imgs = driver.find_elements(By.TAG_NAME, "img")
        for img in imgs:
            src = img.get_attribute("src") or ""
            alt = (img.get_attribute("alt") or "").lower()
            # Look for lot images (usually hosted on cloudfront or similar CDN)
            if src and any(pattern in src.lower() for pattern in [
                "cloudfront", "cdn", "lotimage", "lot_image", "auction",
                "pundoles", ".jpg", ".jpeg", ".png",
            ]):
                # Skip tiny icons, logos, avatars
                if any(skip in src.lower() for skip in [
                    "logo", "icon", "avatar", "placeholder", "spinner",
                    "flag", "social", "footer", "header",
                ]):
                    continue
                # Skip very small images (likely icons)
                try:
                    width = img.size.get("width", 0)
                    if width and width < 50:
                        continue
                except Exception:
                    pass
                result["image_url"] = src
                break
    except Exception:
        pass

    # Also try to find image URL in page source via regex
    if not result["image_url"]:
        img_match = re.search(
            r'(https?://[^"\']+?(?:cloudfront|cdn|auctionmobility)[^"\']*?\.(?:jpg|jpeg|png))',
            page_source, re.I,
        )
        if img_match:
            result["image_url"] = img_match.group(1)

    # --- Parse page text into sections ---
    # AuctionMobility lot pages typically have labeled sections
    sections = _parse_page_sections(page_text)

    # --- Artist name ---
    result["artist_raw"] = _extract_artist(driver, page_text, sections)

    # Parse birth/death years from artist text
    artist_raw = result["artist_raw"]
    m = re.search(r"\((\d{4})\s*[-–]\s*(\d{4})\)", artist_raw)
    if m:
        result["artist_birth_year"] = int(m.group(1))
        result["artist_death_year"] = int(m.group(2))
        result["artist_raw"] = artist_raw[:m.start()].strip()
    else:
        m = re.search(r"\(b\.\s*(\d{4})\)", artist_raw)
        if m:
            result["artist_birth_year"] = int(m.group(1))
            result["artist_raw"] = artist_raw[:m.start()].strip()
        else:
            # Look for years adjacent to artist name in page text
            m = re.search(
                re.escape(artist_raw) + r"\s*\(?\s*(\d{4})\s*[-–]\s*(\d{4})\s*\)?",
                page_text,
            )
            if m:
                result["artist_birth_year"] = int(m.group(1))
                result["artist_death_year"] = int(m.group(2))

    # --- Title ---
    result["title"] = _extract_title(driver, page_text, sections)

    # --- Medium ---
    result["medium_raw"] = _extract_medium(page_text, sections)

    # --- Dimensions ---
    result["dimensions_text"] = _extract_dimensions(page_text, sections)

    # --- Year created ---
    result["year_text"] = _extract_year_text(page_text, sections)

    # --- Signed / Dated ---
    result["signed_dated_text"] = _extract_signed_dated(page_text, sections)

    # --- Provenance ---
    result["provenance_text"] = _extract_section_text(sections, [
        "provenance", "provenence",
    ])

    # --- Literature ---
    result["literature_text"] = _extract_section_text(sections, [
        "literature", "published", "publications",
    ])

    # --- Exhibited ---
    result["exhibited_text"] = _extract_section_text(sections, [
        "exhibited", "exhibition history", "exhibitions",
    ])

    # --- Estimate ---
    _extract_estimate(page_text, result)

    # --- Hammer price / Sold status ---
    _extract_hammer_price(page_text, driver, result)

    return result


def _parse_page_sections(page_text: str) -> dict[str, list[str]]:
    """Parse page text into labeled sections."""
    sections = {}
    current = "header"
    section_headers = [
        "artist", "provenance", "provenence", "literature", "published", "publications",
        "exhibited", "exhibition history", "exhibitions",
        "condition report", "condition", "lot notes", "lot essay",
        "estimate", "artwork details", "description",
        "catalogue note", "catalog note", "note",
    ]

    for line in page_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()

        matched_header = None
        for hdr in section_headers:
            if lower == hdr or lower == hdr + ":":
                matched_header = hdr
                break

        if matched_header:
            current = matched_header
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, [])
            sections[current].append(stripped)

    return sections


def _extract_artist(driver: webdriver.Chrome, page_text: str, sections: dict) -> str:
    """Extract artist name from lot page.

    Pundole's AuctionMobility format:
      ARTIST
      JAMINI ROY (1887-1972)
    """
    # Primary: look in the ARTIST section
    artist_lines = sections.get("artist", [])
    for line in artist_lines:
        line = line.strip()
        if len(line) > 2 and line.lower() not in {"follow", "artist", "my bids", "upcoming auctions"}:
            return line

    # Fallback: find "Name (YYYY-YYYY)" pattern anywhere
    m = re.search(
        r"([A-Z][A-Za-z\.\s\'\-]+?)\s*\(\d{4}\s*[-–]\s*\d{4}\)",
        page_text,
    )
    if m:
        return m.group(0).strip()

    return ""


def _extract_title(driver: webdriver.Chrome, page_text: str, sections: dict) -> str:
    """Extract artwork title from lot page.

    Pundole's format: title is right after "Lot N*" in the header, before "WATCH"/"SOLD".
    """
    header_lines = sections.get("header", [])
    for i, line in enumerate(header_lines):
        # Find "Lot N" line
        if re.match(r"^Lot\s+\d+", line, re.I):
            # Title is the next non-trivial line
            for j in range(i + 1, min(i + 4, len(header_lines))):
                candidate = header_lines[j].strip()
                skip = {"watch", "jump", "watch jump", "sold", "unsold", "passed",
                        "withdrawn", "back to auction", "follow", "live auction",
                        "timed auction", "bidding open"}
                if candidate.lower() in skip:
                    continue
                if re.match(r"^(SOLD|UNSOLD|₹|Rs|EST|LIVE|TIMED|BACK)", candidate, re.I):
                    continue
                if len(candidate) > 1 and len(candidate) < 300:
                    return candidate
            break

    # Fallback: description section first line if it looks like a title
    desc_lines = sections.get("description", [])
    if desc_lines:
        return desc_lines[0]

    return ""


def _extract_medium(page_text: str, sections: dict) -> str:
    """Extract medium from lot page."""
    medium_keywords = [
        "oil on", "acrylic on", "tempera on", "watercolour", "watercolor",
        "gouache", "ink on", "pen on", "pencil on", "charcoal",
        "pastel", "mixed media", "bronze", "marble", "lithograph",
        "etching", "screenprint", "serigraph", "photograph",
        "on canvas", "on paper", "on board", "on card", "on silk",
        "on masonite", "on panel",
    ]

    # Search all sections
    for section_lines in sections.values():
        for line in section_lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in medium_keywords):
                # Make sure it's not a title or provenance entry
                if len(line) < 200:
                    return line.strip()

    # Broad search in page text
    for line in page_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line_lower = line.lower()
        if any(kw in line_lower for kw in medium_keywords):
            if len(line) < 200:
                return line.strip()

    return ""


def _extract_dimensions(page_text: str, sections: dict) -> str:
    """Extract dimensions text from lot page."""
    # Look for dimension patterns: "24 x 36 in", "61 x 91.4 cm"
    for line in page_text.split("\n"):
        line = line.strip()
        if re.search(r"\d+\.?\d*\s*[x×]\s*\d+\.?\d*\s*(?:in|cm|inch|mm)", line, re.I):
            return line.strip()

    # Also match just "H x W" without units
    for line in page_text.split("\n"):
        line = line.strip()
        if re.search(r"\d+\.?\d*\s*[x×]\s*\d+\.?\d*", line):
            # Avoid matching dates or prices
            if not re.search(r"[Rs₹$€£]", line) and "20" not in line[:4]:
                return line.strip()

    return ""


def _extract_year_text(page_text: str, sections: dict) -> str:
    """Extract year-created text from lot page."""
    # Look for standalone year or "Circa YYYY"
    for line in page_text.split("\n"):
        line = line.strip()
        if re.match(r"^(?:circa\s+)?(?:c\.\s*)?(\d{4})s?$", line, re.I):
            return line.strip()
        if re.match(r"^(?:Painted|Executed|Created)\s+(?:in\s+)?(?:circa\s+)?\d{4}", line, re.I):
            return line.strip()

    # Look in description / artwork details sections
    for sec_name in ["description", "artwork details", "header"]:
        for line in sections.get(sec_name, []):
            if re.search(r"(?:circa\s+)?\d{4}s?\b", line, re.I):
                if not re.search(r"\d+\.?\d*\s*[x×]", line):
                    if not re.search(r"[Rs₹$€£]", line):
                        return line.strip()

    return ""


def _extract_signed_dated(page_text: str, sections: dict) -> str:
    """Extract signed/dated information."""
    signed_lines = []
    for line in page_text.split("\n"):
        line = line.strip()
        line_lower = line.lower()
        if any(kw in line_lower for kw in ["signed", "inscribed", "stamped"]):
            if "unsigned" not in line_lower:
                signed_lines.append(line)
        elif line_lower.startswith("dated"):
            signed_lines.append(line)

    return "; ".join(signed_lines[:5])  # Cap at 5 entries


def _extract_section_text(sections: dict, keys: list[str]) -> str:
    """Extract text from a named section."""
    for key in keys:
        lines = sections.get(key, [])
        if lines:
            return "\n".join(lines)
    return ""


def _extract_estimate(page_text: str, result: dict) -> None:
    """Extract estimate range from page text."""
    # Pattern: "Est: Rs 5,00,000 - 7,00,000" or "Estimate ₹5,00,000 - ₹7,00,000"
    # or "INR 5,00,000 - 7,00,000"
    patterns = [
        # "Rs X - Y" or "₹X - Y" or "INR X - Y"
        r"(?:Est(?:imate)?[:\s]*)?(?:Rs\.?|₹|INR)\s*([\d,]+)\s*[-–to]+\s*(?:Rs\.?|₹|INR)?\s*([\d,]+)",
        # Broader: look for two numbers separated by dash near "estimate"
        r"[Ee]stimate[:\s]*[^\d]*([\d,]+)\s*[-–]\s*([\d,]+)",
    ]

    for pat in patterns:
        m = re.search(pat, page_text)
        if m:
            low = _parse_indian_number(m.group(1))
            high = _parse_indian_number(m.group(2))
            if low and high:
                result["estimate_low_inr"] = low
                result["estimate_high_inr"] = high
                return


def _extract_hammer_price(page_text: str, driver: webdriver.Chrome, result: dict) -> None:
    """Extract hammer price and sold status from page text."""
    text_lower = page_text.lower()

    # Check for withdrawn
    if "withdrawn" in text_lower:
        result["is_withdrawn"] = True
        result["is_sold"] = False
        return

    # Check for unsold / not sold / passed
    if any(kw in text_lower for kw in ["not sold", "unsold", "passed", "lot passed",
                                         "bought in", "no sale"]):
        result["is_sold"] = False
        return

    # Look for sold price / hammer price / winning bid
    # Pattern: "Sold for Rs X" or "Hammer Price: ₹X" or "Winning Bid: INR X"
    # or just "Rs X,XX,XXX" near "sold" context
    patterns = [
        r"SOLD\s*(?:Rs\.?|₹|INR)\s*([\d,]+)",
        r"(?:Sold\s+(?:for|at)?|Hammer\s+Price|Winning\s+Bid|Price\s+Realised|Result)[:\s]*(?:Rs\.?|₹|INR)\s*([\d,]+)",
    ]

    for pat in patterns:
        m = re.search(pat, page_text, re.I)
        if m:
            price = _parse_indian_number(m.group(1))
            if price and price > 0:
                result["hammer_price_inr"] = price
                result["is_sold"] = True
                return

    # Check for "Sold" keyword without explicit price
    if "sold" in text_lower and "not sold" not in text_lower and "unsold" not in text_lower:
        result["is_sold"] = True

        # Try to find any price on the page that looks like a hammer price
        # (larger number, likely in INR)
        prices = re.findall(r"(?:Rs\.?|₹|INR)\s*([\d,]+)", page_text)
        if prices:
            parsed_prices = [_parse_indian_number(p) for p in prices]
            parsed_prices = [p for p in parsed_prices if p and p > 1000]
            if parsed_prices:
                # The hammer price is typically the largest price shown
                # (or the one after estimate)
                result["hammer_price_inr"] = max(parsed_prices)


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

def build_lot_record(
    lot_entry: dict,
    detail: dict,
    auction: dict,
) -> dict:
    """Assemble a complete lot record matching the project schema."""

    auction_id = auction["auction_id"]
    lot_slug = lot_entry["lot_slug"]
    lot_number = lot_entry.get("lot_number") or 0
    auction_date = auction.get("date", "")
    lot_url = lot_entry["lot_url"]

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
    medium_raw = detail.get("medium_raw", "")
    medium_category = parse_medium(medium_raw)

    # Dimensions
    dims_text = detail.get("dimensions_text", "")
    height_cm, width_cm = parse_dimensions(dims_text)

    # Year created
    year_text = detail.get("year_text", "")
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

    # Estimates — convert INR to USD
    est_low_inr = detail.get("estimate_low_inr")
    est_high_inr = detail.get("estimate_high_inr")
    est_low_usd = to_usd(est_low_inr, "INR", auction_date) if est_low_inr else None
    est_high_usd = to_usd(est_high_inr, "INR", auction_date) if est_high_inr else None

    # Hammer price — convert INR to USD
    hammer_inr = detail.get("hammer_price_inr")
    hammer_usd = to_usd(hammer_inr, "INR", auction_date) if hammer_inr else None

    # Determine sale type from auction title
    title_lower = auction.get("title", "").lower()
    if "online" in title_lower:
        sale_type = "online"
    elif "live" in title_lower:
        sale_type = "live"
    else:
        sale_type = "auction"

    return {
        "lot_id": f"pundoles_{auction_id}_{lot_slug}",
        "lot_number": lot_number,
        "auction_id": f"pundoles_{auction_id}",
        "auction_title": auction.get("title", ""),
        "auction_date": auction_date,
        "auction_location": "Mumbai",
        "sale_type": sale_type,
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
        "estimate_currency": "INR",
        "hammer_price_usd": hammer_usd,
        "hammer_currency": "INR",
        "is_sold": detail.get("is_sold", False),
        "is_withdrawn": detail.get("is_withdrawn", False),
        "image_url": detail.get("image_url", ""),
        "lot_url": lot_url,
    }


# ---------------------------------------------------------------------------
# Auction-level orchestrator
# ---------------------------------------------------------------------------

def scrape_auction(driver: webdriver.Chrome, auction: dict, scraped_lot_ids: set) -> list[dict]:
    """Scrape all lots from a single Pundole's auction."""
    auction_id = auction["auction_id"]
    logger.info(f"\nScraping auction: {auction['title']} ({auction['date']}) [{auction_id}]")

    # Phase 2: Get lot listing
    lot_entries = scrape_lot_listing(driver, auction)
    if not lot_entries:
        logger.warning(f"  No lots found for auction {auction_id}")
        return []

    records = []
    new_count = 0

    for lot_entry in tqdm(lot_entries, desc=f"  Lots ({auction['title'][:30]})", leave=False):
        lot_id = f"pundoles_{auction_id}_{lot_entry['lot_slug']}"

        if lot_id in scraped_lot_ids:
            continue

        # Phase 3: Load lot detail page
        detail = scrape_lot_detail(driver, lot_entry["lot_url"])

        # Build record
        record = build_lot_record(lot_entry, detail, auction)
        records.append(record)
        scraped_lot_ids.add(lot_id)
        new_count += 1

        # Save progress periodically
        if new_count % SAVE_INTERVAL == 0:
            for rec in records[-SAVE_INTERVAL:]:
                append_ndjson(PROGRESS_FILE, rec)
            logger.info(f"  Saved progress ({new_count} lots so far)")

    # Save remaining unsaved records
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
    logger.info("Scraping Pundole's fine art auction lot details")
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
        # Phase 1: Discover fine art auctions
        auctions = discover_auctions(driver)
        if not auctions:
            logger.error("No qualifying fine art auctions found!")
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
