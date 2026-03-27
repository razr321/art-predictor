#!/usr/bin/env python3
"""Scrape lot details from Sotheby's South Asian Modern & Contemporary Art auctions.

Two-phase approach (same as Christie's scraper):
1. Bulk: Load auction page → extract all lots from __NEXT_DATA__ Algolia JSON (fast)
2. Detail: Load individual lot pages for prices, provenance, dimensions (slow)

Requires login for hammer prices on closed lots.
Set env vars: SOTHEBYS_EMAIL, SOTHEBYS_PASSWORD

Output: data/raw/lots_sothebys.csv
"""

import json
import os
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

logger = setup_logger(__name__, "scrape_lots_sothebys.log")

PROGRESS_FILE = DATA_RAW / "lots_sothebys_progress.ndjson"
OUTPUT_FILE = DATA_RAW / "lots_sothebys.csv"
AUCTIONS_FILE = DATA_RAW / "auctions_sothebys.csv"

SAVE_INTERVAL = 20
DELAY_BETWEEN_LOTS = 2.0
DELAY_BETWEEN_AUCTIONS = 3


def make_driver(use_profile: bool = False) -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    if use_profile:
        # Copy Chrome profile to a temp dir so we don't lock the real profile.
        import shutil
        import tempfile
        chrome_data = os.path.expanduser(
            "~/Library/Application Support/Google/Chrome"
        )
        tmp_dir = tempfile.mkdtemp(prefix="chrome_profile_")
        # Copy Default profile (cookies, local storage, etc.)
        src = os.path.join(chrome_data, "Default")
        dst = os.path.join(tmp_dir, "Default")
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
            "Cache", "Code Cache", "GPUCache", "Service Worker",
            "GrShaderCache", "ShaderCache", "DawnCache",
        ))
        # Copy top-level files needed by Chrome
        for fname in ("Local State",):
            src_f = os.path.join(chrome_data, fname)
            if os.path.exists(src_f):
                shutil.copy2(src_f, os.path.join(tmp_dir, fname))
        opts.add_argument(f"--user-data-dir={tmp_dir}")
        opts.add_argument("--profile-directory=Default")
        logger.info(f"Using copied Chrome profile from {tmp_dir}")

    # Enable performance logging to intercept Algolia API credentials
    opts.set_capability("goog:loggingPrefs", {"performance": "ALL"})

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


def login_sothebys(driver: webdriver.Chrome) -> bool:
    """Login to Sotheby's to access hammer prices."""
    email = os.environ.get("SOTHEBYS_EMAIL", "")
    password = os.environ.get("SOTHEBYS_PASSWORD", "")

    if not email or not password:
        logger.warning("SOTHEBYS_EMAIL / SOTHEBYS_PASSWORD not set — prices may be hidden")
        return False

    logger.info("Logging in to Sotheby's...")
    try:
        driver.get("https://www.sothebys.com/en/login")
        time.sleep(3)
        dismiss_overlays(driver)

        # Wait for login form
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email'], input[name='email'], #email"))
        )

        # Fill email
        email_input = driver.find_element(By.CSS_SELECTOR, "input[type='email'], input[name='email'], #email")
        email_input.clear()
        email_input.send_keys(email)
        time.sleep(0.5)

        # Fill password
        pw_input = driver.find_element(By.CSS_SELECTOR, "input[type='password'], input[name='password'], #password")
        pw_input.clear()
        pw_input.send_keys(password)
        time.sleep(0.5)

        # Submit
        submit_btn = driver.find_element(By.CSS_SELECTOR, "button[type='submit'], input[type='submit']")
        submit_btn.click()
        time.sleep(5)

        # Check if login succeeded (look for account indicator or absence of login form)
        if "login" not in driver.current_url.lower():
            logger.info("  Login successful")
            return True
        else:
            logger.warning("  Login may have failed — still on login page")
            return False

    except Exception as e:
        logger.warning(f"  Login failed: {e}")
        return False


def extract_next_data(driver: webdriver.Chrome) -> dict | None:
    """Extract __NEXT_DATA__ JSON from current page."""
    try:
        data = driver.execute_script("""
            var el = document.getElementById('__NEXT_DATA__');
            return el ? el.textContent : null;
        """)
        if data:
            return json.loads(data)
    except Exception as e:
        logger.warning(f"Failed to extract __NEXT_DATA__: {e}")
    return None


def extract_algolia_credentials(driver: webdriver.Chrome) -> tuple[str, str, str]:
    """Extract Algolia app ID, API key, and index name from network requests.

    Enables performance logging to intercept the Algolia API call headers.
    Returns (app_id, api_key, index_name) or empty strings if not found.
    """
    try:
        logs = driver.get_log("performance")
        for entry in logs:
            msg = json.loads(entry["message"])["message"]
            if msg["method"] != "Network.requestWillBeSent":
                continue
            url = msg["params"]["request"]["url"]
            if "algolia.net" not in url or "prod_lots" not in url:
                continue
            headers = msg["params"]["request"].get("headers", {})
            app_id = headers.get("x-algolia-application-id", "")
            api_key = headers.get("x-algolia-api-key", "")
            return app_id, api_key, "prod_lots"
    except Exception as e:
        logger.debug(f"Could not extract Algolia credentials: {e}")
    return "", "", ""


def fetch_all_algolia_lots(driver: webdriver.Chrome, algolia_json: dict) -> list[dict]:
    """Fetch ALL lots via the Algolia API, paginating if needed.

    Uses the filter params from the initial __NEXT_DATA__ Algolia response,
    but queries the Algolia REST API directly to get all pages.
    """
    import requests as req

    hits = algolia_json.get("hits", [])
    nb_hits = algolia_json.get("nbHits", len(hits))
    nb_pages = algolia_json.get("nbPages", 1)
    hits_per_page = algolia_json.get("hitsPerPage", 48)

    if nb_pages <= 1:
        return hits  # All lots fit in one page

    logger.info(f"    Algolia reports {nb_hits} total lots across {nb_pages} pages — fetching remaining pages")

    # Extract Algolia credentials from intercepted network requests
    app_id, api_key, index = extract_algolia_credentials(driver)
    if not app_id or not api_key:
        logger.warning("    Could not extract Algolia API credentials — using page 1 only")
        return hits

    # Build the filter from the initial response params
    params_str = algolia_json.get("params", "")
    import urllib.parse
    parsed = urllib.parse.parse_qs(params_str)
    filters = parsed.get("filters", [""])[0]

    algolia_url = f"https://{app_id.lower()}-dsn.algolia.net/1/indexes/{index}/query"
    headers = {
        "X-Algolia-Application-Id": app_id,
        "X-Algolia-API-Key": api_key,
        "Content-Type": "application/json",
    }

    all_hits = list(hits)  # Start with page 0 hits we already have

    for page_num in range(1, nb_pages):
        body = {
            "query": "",
            "filters": filters,
            "facetFilters": [["withdrawn:false"], []],
            "hitsPerPage": hits_per_page,
            "page": page_num,
            "facets": ["*"],
            "numericFilters": [],
        }
        try:
            resp = req.post(algolia_url, json=body, headers=headers, timeout=15)
            if resp.status_code == 200:
                page_hits = resp.json().get("hits", [])
                all_hits.extend(page_hits)
                logger.info(f"    Fetched page {page_num + 1}/{nb_pages}: {len(page_hits)} lots")
            else:
                logger.warning(f"    Algolia page {page_num + 1} failed: {resp.status_code}")
        except Exception as e:
            logger.warning(f"    Algolia page {page_num + 1} error: {e}")

    return all_hits


def extract_bulk_lots_from_auction(driver: webdriver.Chrome, next_data: dict) -> list[dict]:
    """Extract lot list from auction page __NEXT_DATA__ (Algolia hits), with pagination."""
    try:
        page_props = next_data.get("props", {}).get("pageProps", {})

        # Try algoliaJson.hits first
        algolia = page_props.get("algoliaJson", {})
        hits = algolia.get("hits", [])
        if hits:
            # Check if there are more pages
            nb_pages = algolia.get("nbPages", 1)
            if nb_pages > 1:
                return fetch_all_algolia_lots(driver, algolia)
            return hits

        # Try other nested structures
        for key in ["lots", "data", "auctionData"]:
            if key in page_props and isinstance(page_props[key], list):
                return page_props[key]

        # Deep search for hits array
        def find_hits(obj, depth=0):
            if depth > 5:
                return []
            if isinstance(obj, dict):
                if "hits" in obj and isinstance(obj["hits"], list) and obj["hits"]:
                    first = obj["hits"][0]
                    if isinstance(first, dict) and ("lotDisplayNumber" in first or "objectID" in first):
                        return obj["hits"]
                for v in obj.values():
                    result = find_hits(v, depth + 1)
                    if result:
                        return result
            return []

        return find_hits(next_data)

    except Exception as e:
        logger.warning(f"Failed to extract bulk lots: {e}")
    return []


def extract_lot_detail_from_page(next_data: dict) -> dict:
    """Extract lot details from individual lot page __NEXT_DATA__ (Apollo cache).

    Key paths (from actual Sotheby's pages):
      - apolloCache contains LotV2 objects with creatorsDisplayTitle, description, etc.
      - estimateV2.lowEstimate.amount / highEstimate.amount
      - bidState.bidAsk = final bid amount
      - bidState.sold.__typename = "ResultHidden" (logged out) or actual amount
      - provenance, literature, exhibition = HTML strings directly on lot object
    """
    result = {
        "hammer_price": None,
        "hammer_currency": "",
        "is_sold": False,
        "details_text": "",
        "dimensions_text": "",
        "provenance_text": "",
        "literature_text": "",
        "exhibited_text": "",
        "image_url": "",
        "artist_raw": "",
        "birth_year": None,
        "death_year": None,
    }

    try:
        page_props = next_data.get("props", {}).get("pageProps", {})
        apollo_cache = page_props.get("apolloCache", {})

        if not apollo_cache:
            return result

        # Find the LotV2 object in Apollo cache
        lot_data = None
        for key, val in apollo_cache.items():
            if not isinstance(val, dict):
                continue
            typename = val.get("__typename", "")
            if typename in ("LotV2", "Lot", "LotDetail"):
                lot_data = val
                break
            if "creatorsDisplayTitle" in val:
                lot_data = val
                break

        if not lot_data:
            return result

        # --- Artist info ---
        artist_raw = lot_data.get("creatorsDisplayTitle", "")
        result["artist_raw"] = artist_raw

        # Birth/death from creatorsDates or scanning cache for creator objects
        dates_str = lot_data.get("creatorsDates", "") or ""
        if not dates_str:
            # Scan cache for creator-like entries
            for key, val in apollo_cache.items():
                if isinstance(val, dict) and val.get("__typename") in ("Creator", "Artist"):
                    dates_str = val.get("dates", "") or val.get("lifespan", "") or ""
                    if dates_str:
                        break

        m = re.search(r"(\d{4})\s*[-–]\s*(\d{4})", dates_str)
        if m:
            result["birth_year"] = int(m.group(1))
            result["death_year"] = int(m.group(2))
        else:
            m = re.search(r"b\.\s*(\d{4})", dates_str)
            if m:
                result["birth_year"] = int(m.group(1))

        # --- Price / sold status ---
        # Follow __ref to resolve BidState from Apollo cache
        bid_state = {}
        bid_state_ref = lot_data.get("bidState", {})
        if isinstance(bid_state_ref, dict) and "__ref" in bid_state_ref:
            bid_state = apollo_cache.get(bid_state_ref["__ref"], {})
        elif isinstance(bid_state_ref, dict):
            bid_state = bid_state_ref

        # Extract from BidState (the primary source)
        bid_ask = bid_state.get("bidAsk") or lot_data.get("bidAsk")
        reserve_met = bid_state.get("reserveMet", False) or lot_data.get("reserveMet", False)
        is_closed = bid_state.get("isClosed", False)

        # Check sold field on BidState or lot object
        sold_data = bid_state.get("sold") or lot_data.get("sold")
        if isinstance(sold_data, dict):
            if sold_data.get("__typename") == "ResultHidden":
                # Price hidden but lot was sold — use bidAsk as hammer price
                if is_closed and reserve_met and bid_ask:
                    try:
                        result["hammer_price"] = float(str(bid_ask).replace(",", ""))
                        result["is_sold"] = True
                    except (ValueError, TypeError):
                        pass
                elif is_closed and bid_ask:
                    # Closed but reserve not met — lot unsold
                    result["is_sold"] = False
            elif sold_data.get("amount"):
                try:
                    result["hammer_price"] = float(str(sold_data["amount"]).replace(",", ""))
                    result["hammer_currency"] = sold_data.get("currency", "")
                    result["is_sold"] = True
                except (ValueError, TypeError):
                    pass

        # Fallback: scan Apollo cache for BidState entries we might have missed
        if not result["hammer_price"]:
            for key, val in apollo_cache.items():
                if not isinstance(val, dict):
                    continue
                if val.get("__typename") == "BidState":
                    ba = val.get("bidAsk")
                    rm = val.get("reserveMet", False)
                    ic = val.get("isClosed", False)
                    if ba and rm and ic:
                        try:
                            result["hammer_price"] = float(str(ba).replace(",", ""))
                            result["is_sold"] = True
                        except (ValueError, TypeError):
                            pass
                        break

        # Fallback: scan for Money/Amount objects with hammer/result in key
        if not result["hammer_price"]:
            for key, val in apollo_cache.items():
                if not isinstance(val, dict):
                    continue
                typename = val.get("__typename", "")
                if typename in ("Money", "Amount", "Price"):
                    amount_str = val.get("amount") or val.get("value")
                    currency = val.get("currency", "")
                    key_lower = key.lower()
                    if amount_str and any(kw in key_lower for kw in ("hammer", "result", "sold", "price")):
                        try:
                            price_val = float(str(amount_str).replace(",", ""))
                            if price_val > 0:
                                result["hammer_price"] = price_val
                                result["hammer_currency"] = currency
                                result["is_sold"] = True
                        except (ValueError, TypeError):
                            pass

        # Get currency from estimateV2 if not set
        if not result["hammer_currency"]:
            for key, val in apollo_cache.items():
                if isinstance(val, dict) and "currency" in val and val.get("__typename") in ("Money", "Amount"):
                    result["hammer_currency"] = val["currency"]
                    break

        # --- Description / medium ---
        description = lot_data.get("description", "") or ""
        if isinstance(description, dict):
            description = description.get("text", "") or str(description)
        # Strip HTML tags
        description = re.sub(r"<[^>]+>", " ", description).strip()
        description = re.sub(r"\s{2,}", " ", description)
        result["details_text"] = description

        # --- Dimensions (often embedded in description HTML) ---
        # Check for dedicated dimensions field first
        dims = lot_data.get("dimensions", "") or ""
        if isinstance(dims, dict):
            dims = dims.get("text", "") or ""
        if not dims:
            dims = description  # Dimensions often in description
        result["dimensions_text"] = dims

        # --- Provenance, Literature, Exhibition ---
        # These are direct HTML string fields on the LotV2 object
        def clean_html(html_str):
            if not html_str:
                return ""
            if isinstance(html_str, dict):
                html_str = html_str.get("text", "") or str(html_str)
            text = re.sub(r"<br\s*/?>", "\n", html_str)
            text = re.sub(r"</(?:p|li|div)>", "\n", text)
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text.strip()

        result["provenance_text"] = clean_html(lot_data.get("provenance", ""))
        result["literature_text"] = clean_html(lot_data.get("literature", ""))
        result["exhibited_text"] = clean_html(lot_data.get("exhibition", ""))

        # Fallback: scan cache for section-like entries
        if not result["provenance_text"] or not result["literature_text"]:
            for key, val in apollo_cache.items():
                if not isinstance(val, dict):
                    continue
                key_lower = key.lower()
                text = clean_html(val.get("text", "") or val.get("content", ""))
                if not text:
                    continue
                if "provenance" in key_lower and not result["provenance_text"]:
                    result["provenance_text"] = text
                elif "literature" in key_lower and not result["literature_text"]:
                    result["literature_text"] = text
                elif "exhibit" in key_lower and not result["exhibited_text"]:
                    result["exhibited_text"] = text

        # --- Image URL ---
        for key, val in apollo_cache.items():
            if not isinstance(val, dict):
                continue
            if val.get("__typename") in ("Image", "LotImage"):
                img_url = val.get("url", "") or val.get("src", "")
                if img_url:
                    result["image_url"] = img_url
                    break
            # Also check renditions
            if "rendition" in key.lower() or "image" in key.lower():
                img_url = val.get("url", "") or val.get("src", "")
                if img_url and ("sothebys" in img_url or "brightspot" in img_url or "dam." in img_url):
                    result["image_url"] = img_url
                    break

        if not result["image_url"]:
            lot_id = lot_data.get("lotId", "") or lot_data.get("objectID", "")
            if lot_id:
                result["image_url"] = f"https://dam.sothebys.com/dam/image/lot/{lot_id}/primary/Small"

    except Exception as e:
        logger.warning(f"Error extracting lot detail: {e}")

    return result


def parse_sothebys_lot(hit: dict, auction_meta: dict, detail: dict | None = None) -> dict:
    """Convert Algolia hit + detail into a record matching Christie's schema."""

    # Artist info from bulk data
    artist_raw = hit.get("creatorsDisplayTitle", "") or ""
    artist_name, birth_year, death_year = parse_artist_name(artist_raw)

    # Override with detail page data if available
    if detail:
        if detail.get("artist_raw"):
            artist_name_d, birth_year_d, death_year_d = parse_artist_name(detail["artist_raw"])
            if artist_name_d:
                artist_name = artist_name_d
            if birth_year_d:
                birth_year = birth_year_d
            if death_year_d:
                death_year = death_year_d
        if detail.get("birth_year"):
            birth_year = detail["birth_year"]
        if detail.get("death_year"):
            death_year = detail["death_year"]

    # Title
    title = hit.get("title", "") or ""

    # Details text (medium, dimensions from detail page)
    details_text = ""
    if detail:
        details_text = detail.get("details_text", "")
    if not details_text:
        details_text = hit.get("description", "") or ""

    # Medium
    medium = parse_medium(details_text)

    # Dimensions — try detail page dimensions text, then details text
    height_cm, width_cm = None, None
    if detail and detail.get("dimensions_text"):
        height_cm, width_cm = parse_dimensions(detail["dimensions_text"])
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

    # Estimates from bulk data
    est_low = hit.get("lowEstimate")
    est_high = hit.get("highEstimate")
    est_currency = hit.get("currency", "USD")

    # Convert estimates
    auction_date = auction_meta.get("start_date", "")
    try:
        est_low = float(est_low) if est_low else None
        est_high = float(est_high) if est_high else None
    except (ValueError, TypeError):
        est_low, est_high = None, None

    est_low_usd = to_usd(est_low, est_currency, auction_date)
    est_high_usd = to_usd(est_high, est_currency, auction_date)

    # Hammer price — try detail page first, then Algolia bulk data
    hammer_price = None
    hammer_currency = est_currency  # Default to same currency as estimates
    is_sold = False

    if detail:
        hammer_price = detail.get("hammer_price")
        if detail.get("hammer_currency"):
            hammer_currency = detail["hammer_currency"]
        is_sold = detail.get("is_sold", False)

    # Algolia bulk data has 'price' field for some lots
    if not hammer_price:
        algolia_price = hit.get("price")
        if algolia_price is not None:
            try:
                hammer_price = float(algolia_price)
                if hammer_price > 0:
                    is_sold = True
            except (ValueError, TypeError):
                pass

    # Lot state from Algolia
    lot_state = hit.get("lotState", "")
    if lot_state in ("Closed", "SoldAfterSale") and not is_sold:
        # Lot closed but we don't have price (likely hidden without login)
        pass
    if hit.get("withdrawn", False):
        is_withdrawn = True

    hammer_price_usd = to_usd(hammer_price, hammer_currency, auction_date)

    # Is withdrawn
    is_withdrawn = hit.get("isWithdrawn", False) or lot_state == "Withdrawn"

    # Image URL
    image_url = ""
    if detail and detail.get("image_url"):
        image_url = detail["image_url"]
    if not image_url:
        obj_id = hit.get("objectID", "")
        if obj_id:
            image_url = f"https://dam.sothebys.com/dam/image/lot/{obj_id}/primary/Small"

    # Lot URL — slug may be a full path or just a name
    lot_url = ""
    slug = hit.get("slug", "") or hit.get("lotSlug", "")
    if slug:
        if slug.startswith("/"):
            # Full path slug like /en/buy/auction/2024/sale-name/lot-name
            lot_url = f"https://www.sothebys.com{slug}"
        else:
            # Just a lot name — append to auction URL
            auction_url = auction_meta.get("auction_url", "")
            if auction_url:
                base = auction_url.rstrip("/").split("?")[0]
                lot_url = f"{base}/{slug}"
            else:
                aid = auction_meta.get("auction_id", "")
                if aid:
                    lot_url = f"https://www.sothebys.com/en/buy/auction/{aid}/{slug}"

    if lot_url and not lot_url.startswith("http"):
        lot_url = f"https://www.sothebys.com{lot_url}"

    return {
        "lot_id": f"sothebys_{hit.get('objectID', '')}",
        "lot_number": hit.get("lotDisplayNumber", "") or hit.get("lotNumber", ""),
        "auction_id": f"sothebys_{auction_meta.get('auction_id', '')}",
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
        "hammer_currency": hammer_currency,
        "is_sold": is_sold,
        "is_withdrawn": is_withdrawn,
        "image_url": image_url,
        "lot_url": lot_url,
    }


def scrape_auction(driver: webdriver.Chrome, auction: dict, scraped_lot_ids: set) -> list[dict]:
    """Scrape all lots from a single Sotheby's auction."""
    url = auction.get("url", "")
    if not url:
        logger.warning(f"No URL for auction {auction.get('auction_id')}")
        return []

    # Ensure we load all lots view
    if "?" in url:
        if "lotFilter" not in url:
            url += "&lotFilter=AllLots"
    else:
        url += "?lotFilter=AllLots"

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

    # Phase 1: Bulk extract from __NEXT_DATA__
    next_data = extract_next_data(driver)
    if not next_data:
        logger.warning("  No __NEXT_DATA__ found on auction page")
        return []

    hits = extract_bulk_lots_from_auction(driver, next_data)
    logger.info(f"  Bulk extracted {len(hits)} lots")

    # Try to get auction date from the page data
    page_props = next_data.get("props", {}).get("pageProps", {})
    auction_date = ""
    # Check hits for auctionDate
    if hits:
        auction_date = hits[0].get("auctionDate", "") or ""
    if not auction_date:
        auction_date = auction.get("start_date", "")

    auction_meta = {
        "auction_id": auction_id,
        "title": auction.get("title", ""),
        "start_date": auction_date,
        "location": auction.get("location", ""),
        "sale_type": auction.get("sale_type", ""),
        "year": auction.get("year", ""),
        "auction_url": auction.get("url", ""),
    }

    records = []
    new_count = 0

    for hit in tqdm(hits, desc=f"  Lots ({auction_id})", leave=False):
        obj_id = hit.get("objectID", "")
        lot_id = f"sothebys_{obj_id}"
        if lot_id in scraped_lot_ids:
            continue

        # Phase 2: Load individual lot page for prices/provenance
        detail = None
        slug = hit.get("slug", "") or hit.get("lotSlug", "")
        if slug:
            # Slug may be a full path (e.g. /en/buy/auction/2024/sale-name/lot-name)
            if slug.startswith("/"):
                lot_url = f"https://www.sothebys.com{slug}"
            else:
                lot_url = f"https://www.sothebys.com/en/buy/auction/{auction_id}/{slug}"
            try:
                driver.get(lot_url)
                time.sleep(DELAY_BETWEEN_LOTS)
                dismiss_overlays(driver)
                lot_next_data = extract_next_data(driver)
                if lot_next_data:
                    detail = extract_lot_detail_from_page(lot_next_data)
            except Exception as e:
                logger.warning(f"  Failed to load lot {obj_id}: {e}")

        record = parse_sothebys_lot(hit, auction_meta, detail)
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

    logger.info(f"  → Scraped {new_count} new lots from {auction.get('title', '')}")
    return records


def main():
    logger.info("=" * 60)
    logger.info("Scraping Sotheby's SA M&CA lot details")
    logger.info("=" * 60)

    if not AUCTIONS_FILE.exists():
        logger.error(f"Auctions file not found: {AUCTIONS_FILE}")
        logger.error("Run discover_auctions_sothebys.py first!")
        sys.exit(1)

    auctions_df = pd.read_csv(AUCTIONS_FILE)
    auctions = auctions_df.to_dict("records")
    logger.info(f"Found {len(auctions)} auctions to scrape")

    # Load existing progress (skip if --fresh to re-scrape with new session)
    if "--fresh" in sys.argv:
        existing = []
        logger.info("Fresh run — ignoring previous progress")
    else:
        existing = load_ndjson(PROGRESS_FILE)
        logger.info(f"Loaded {len(existing)} previously scraped lots")
    scraped_lot_ids = {r["lot_id"] for r in existing}
    all_records = list(existing)

    # Use Chrome profile if --profile flag passed (inherits login cookies)
    use_profile = "--profile" in sys.argv
    driver = make_driver(use_profile=use_profile)
    logged_in = False

    try:
        if use_profile:
            logger.info("Using Chrome profile — skipping programmatic login")
            logged_in = True
        else:
            # Login first
            logged_in = login_sothebys(driver)
            if not logged_in:
                logger.warning("Proceeding without login — hammer prices may be missing")

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
    else:
        logger.warning("No lots scraped!")


if __name__ == "__main__":
    main()
