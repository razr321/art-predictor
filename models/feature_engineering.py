#!/usr/bin/env python3
"""Feature engineering pipeline: raw lots → master.csv → ml_ready.csv.

Computes artist-level rolling features (no future leakage), lot-level features,
and market-level features.

Output:
  - data/processed/master.csv   (full features for dashboard)
  - data/processed/ml_ready.csv (ML features + target)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.helpers import setup_logger, DATA_RAW, DATA_PROCESSED

logger = setup_logger(__name__, "feature_engineering.log")
warnings.filterwarnings("ignore", category=FutureWarning)

LOTS_FILE = DATA_RAW / "lots.csv"
MASTER_FILE = DATA_PROCESSED / "master.csv"
ML_READY_FILE = DATA_PROCESSED / "ml_ready.csv"


def load_data() -> pd.DataFrame:
    """Load and do initial cleaning of lots data."""
    df = pd.read_csv(LOTS_FILE)
    logger.info(f"Loaded {len(df)} lots")

    # Parse dates — handle mixed formats (ISO, with/without timezone)
    df["auction_date"] = pd.to_datetime(df["auction_date"], errors="coerce", format="mixed", utc=True)
    df["auction_date"] = df["auction_date"].dt.tz_localize(None)
    df = df.sort_values("auction_date").reset_index(drop=True)

    # Filter out withdrawn lots
    df = df[~df["is_withdrawn"].fillna(False)].copy()
    logger.info(f"After removing withdrawn: {len(df)} lots")

    # Fix is_sold: if hammer_price_usd > 0, the lot was sold
    has_price = df["hammer_price_usd"].notna() & (df["hammer_price_usd"] > 0)
    fixed = has_price & ~df["is_sold"].fillna(False)
    if fixed.sum() > 0:
        df.loc[has_price, "is_sold"] = True
        logger.info(f"Fixed is_sold for {fixed.sum()} lots with hammer prices")

    # Normalize artist names for grouping
    df["artist_name_clean"] = df["artist_name"].str.strip().str.upper()
    df["artist_name_clean"] = df["artist_name_clean"].str.replace(r"\s+", " ", regex=True)

    return df


def compute_artist_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-artist rolling features using ONLY prior lots (no leakage).

    For each lot, we look at all of the same artist's PRIOR lots (by date)
    to compute rolling stats.
    """
    logger.info("Computing artist rolling features...")

    # Sort by date
    df = df.sort_values("auction_date").reset_index(drop=True)

    # Pre-compute columns
    artist_features = {
        "artist_prior_lots_total": [],
        "artist_prior_lots_sold": [],
        "artist_sell_through_rate": [],
        "artist_avg_price": [],
        "artist_median_price": [],
        "artist_max_price": [],
        "artist_min_price": [],
        "artist_price_std": [],
        "artist_price_trend": [],
        "artist_avg_estimate_accuracy": [],
        "artist_years_in_market": [],
        "artist_days_since_last_sale": [],
        "artist_desirability_score": [],
        # New time-series features
        "artist_recent_avg_price": [],       # avg of last 5 sales
        "artist_recent_median_price": [],    # median of last 5 sales
        "artist_recent_max_price": [],       # max of last 5 sales
        "artist_momentum": [],               # recent avg / all-time avg (>1 = trending up)
        "artist_price_velocity": [],         # slope of last 10 sales (log scale)
        "artist_price_per_cm2": [],          # avg price per sq cm (rolling)
        "artist_recent_lots_12m": [],        # lots sold in last 12 months (supply heat)
        "artist_premium_trend": [],          # avg (hammer/estimate) over last 10 sales
    }

    # Group by artist, iterate chronologically
    for _, group in df.groupby("artist_name_clean"):
        group = group.sort_values("auction_date")
        prices_so_far = []
        sold_count = 0
        total_count = 0
        estimate_ratios = []
        first_date = None
        last_sold_date = None
        dates_so_far = []
        prices_per_cm2 = []

        for idx, row in group.iterrows():
            if first_date is None:
                first_date = row["auction_date"]

            # All features use PRIOR data only (before this lot)
            if total_count == 0:
                # First appearance — no history
                artist_features["artist_prior_lots_total"].append(0)
                artist_features["artist_prior_lots_sold"].append(0)
                artist_features["artist_sell_through_rate"].append(np.nan)
                artist_features["artist_avg_price"].append(np.nan)
                artist_features["artist_median_price"].append(np.nan)
                artist_features["artist_max_price"].append(np.nan)
                artist_features["artist_min_price"].append(np.nan)
                artist_features["artist_price_std"].append(np.nan)
                artist_features["artist_price_trend"].append(np.nan)
                artist_features["artist_avg_estimate_accuracy"].append(np.nan)
                artist_features["artist_years_in_market"].append(0.0)
                artist_features["artist_days_since_last_sale"].append(np.nan)
                artist_features["artist_desirability_score"].append(np.nan)
            else:
                artist_features["artist_prior_lots_total"].append(total_count)
                artist_features["artist_prior_lots_sold"].append(sold_count)
                str_rate = sold_count / total_count if total_count > 0 else 0
                artist_features["artist_sell_through_rate"].append(str_rate)

                if prices_so_far:
                    artist_features["artist_avg_price"].append(np.mean(prices_so_far))
                    artist_features["artist_median_price"].append(np.median(prices_so_far))
                    artist_features["artist_max_price"].append(max(prices_so_far))
                    artist_features["artist_min_price"].append(min(prices_so_far))
                    artist_features["artist_price_std"].append(np.std(prices_so_far) if len(prices_so_far) > 1 else 0)

                    # Price trend: slope of log(price) vs index
                    if len(prices_so_far) >= 3:
                        log_prices = np.log(np.array(prices_so_far) + 1)
                        x = np.arange(len(log_prices))
                        slope = np.polyfit(x, log_prices, 1)[0]
                        artist_features["artist_price_trend"].append(slope)
                    else:
                        artist_features["artist_price_trend"].append(np.nan)
                else:
                    for key in ["artist_avg_price", "artist_median_price", "artist_max_price",
                                "artist_min_price", "artist_price_std", "artist_price_trend"]:
                        artist_features[key].append(np.nan)

                # Estimate accuracy
                if estimate_ratios:
                    artist_features["artist_avg_estimate_accuracy"].append(np.mean(estimate_ratios))
                else:
                    artist_features["artist_avg_estimate_accuracy"].append(np.nan)

                # Years in market
                years = (row["auction_date"] - first_date).days / 365.25
                artist_features["artist_years_in_market"].append(years)

                # Days since last sale
                if last_sold_date is not None:
                    days = (row["auction_date"] - last_sold_date).days
                    artist_features["artist_days_since_last_sale"].append(days)
                else:
                    artist_features["artist_days_since_last_sale"].append(np.nan)

                # Desirability score
                if prices_so_far and total_count > 0:
                    freq = total_count  # More lots = more market presence
                    avg_hammer_ratio = np.mean(estimate_ratios) if estimate_ratios else 1.0
                    score = str_rate * avg_hammer_ratio * np.log1p(freq)
                    artist_features["artist_desirability_score"].append(score)
                else:
                    artist_features["artist_desirability_score"].append(np.nan)

            # --- New time-series features ---
            if total_count == 0:
                for key in ["artist_recent_avg_price", "artist_recent_median_price",
                            "artist_recent_max_price", "artist_momentum",
                            "artist_price_velocity", "artist_price_per_cm2",
                            "artist_recent_lots_12m", "artist_premium_trend"]:
                    artist_features[key].append(np.nan)
            else:
                # Recent prices (last 5 sales)
                recent_5 = prices_so_far[-5:] if prices_so_far else []
                if recent_5:
                    artist_features["artist_recent_avg_price"].append(np.mean(recent_5))
                    artist_features["artist_recent_median_price"].append(np.median(recent_5))
                    artist_features["artist_recent_max_price"].append(max(recent_5))
                    # Momentum: recent avg / all-time avg (>1 = prices rising)
                    all_avg = np.mean(prices_so_far) if prices_so_far else 1
                    artist_features["artist_momentum"].append(np.mean(recent_5) / all_avg if all_avg > 0 else 1.0)
                else:
                    for key in ["artist_recent_avg_price", "artist_recent_median_price",
                                "artist_recent_max_price", "artist_momentum"]:
                        artist_features[key].append(np.nan)

                # Price velocity: slope of last 10 sales (log scale)
                recent_10 = prices_so_far[-10:] if prices_so_far else []
                if len(recent_10) >= 3:
                    log_p = np.log(np.array(recent_10) + 1)
                    x = np.arange(len(log_p))
                    slope = np.polyfit(x, log_p, 1)[0]
                    artist_features["artist_price_velocity"].append(slope)
                else:
                    artist_features["artist_price_velocity"].append(np.nan)

                # Price per sq cm (rolling average)
                if prices_per_cm2:
                    artist_features["artist_price_per_cm2"].append(np.median(prices_per_cm2))
                else:
                    artist_features["artist_price_per_cm2"].append(np.nan)

                # Lots in last 12 months (supply heat)
                if dates_so_far and pd.notna(row["auction_date"]):
                    cutoff = row["auction_date"] - pd.DateOffset(months=12)
                    recent_count = sum(1 for d in dates_so_far if pd.notna(d) and d >= cutoff)
                    artist_features["artist_recent_lots_12m"].append(recent_count)
                else:
                    artist_features["artist_recent_lots_12m"].append(0)

                # Premium trend: avg(hammer/estimate_mid) for last 10 sales
                recent_ratios = estimate_ratios[-10:] if estimate_ratios else []
                if recent_ratios:
                    artist_features["artist_premium_trend"].append(np.mean(recent_ratios))
                else:
                    artist_features["artist_premium_trend"].append(np.nan)

            # Update running stats with THIS lot's data (for next lot)
            total_count += 1
            if row["is_sold"] and pd.notna(row["hammer_price_usd"]) and row["hammer_price_usd"] > 0:
                sold_count += 1
                prices_so_far.append(row["hammer_price_usd"])
                last_sold_date = row["auction_date"]
                dates_so_far.append(row["auction_date"])
                # Track price per sq cm
                if pd.notna(row.get("surface_area_cm2")) and row["surface_area_cm2"] > 0:
                    prices_per_cm2.append(row["hammer_price_usd"] / row["surface_area_cm2"])

                # Estimate accuracy: hammer / estimate_midpoint
                est_mid = None
                if pd.notna(row.get("estimate_low_usd")) and pd.notna(row.get("estimate_high_usd")):
                    est_mid = (row["estimate_low_usd"] + row["estimate_high_usd"]) / 2
                if est_mid and est_mid > 0:
                    estimate_ratios.append(row["hammer_price_usd"] / est_mid)

    # Add features back to dataframe
    # We need to reconstruct the order since we grouped
    feature_df = pd.DataFrame(artist_features)
    # The features are in the same order as iterating through groupby then rows
    # Reconstruct index
    indices = []
    for _, group in df.groupby("artist_name_clean"):
        group = group.sort_values("auction_date")
        indices.extend(group.index.tolist())

    feature_df.index = indices
    feature_df = feature_df.sort_index()

    for col in feature_df.columns:
        df[col] = feature_df[col]

    logger.info("  Done: artist rolling features")
    return df


def compute_lot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute lot-level derived features."""
    logger.info("Computing lot-level features...")

    # Surface area
    df["surface_area_cm2"] = df["height_cm"] * df["width_cm"]

    # Artwork age at time of sale
    df["auction_year"] = df["auction_date"].dt.year
    df["auction_month"] = df["auction_date"].dt.month
    df["artwork_age"] = df["auction_year"] - df["year_created"]
    df.loc[df["artwork_age"] < 0, "artwork_age"] = np.nan  # Bad data
    df.loc[df["artwork_age"] > 200, "artwork_age"] = np.nan

    # Artist age at creation (if birth year known)
    df["artist_age_at_creation"] = df["year_created"] - df["artist_birth_year"]
    df.loc[df["artist_age_at_creation"] < 10, "artist_age_at_creation"] = np.nan
    df.loc[df["artist_age_at_creation"] > 100, "artist_age_at_creation"] = np.nan

    # Is artist deceased at time of sale?
    df["artist_deceased"] = df["artist_death_year"].notna().astype(int)

    # Estimate features
    df["estimate_midpoint"] = (df["estimate_low_usd"] + df["estimate_high_usd"]) / 2
    denom = df["estimate_low_usd"].replace(0, np.nan)
    df["estimate_spread"] = (df["estimate_high_usd"] - df["estimate_low_usd"]) / denom

    # Log estimate (useful as feature)
    df["log_estimate_mid"] = np.log1p(df["estimate_midpoint"])

    # Boolean flags
    df["has_provenance"] = (df["provenance_count"] > 0).astype(int)
    df["has_literature"] = (df["literature_count"] > 0).astype(int)
    df["has_exhibitions"] = (df["exhibition_count"] > 0).astype(int)

    # Is live auction (typically higher prices)
    df["is_live_auction"] = (df["sale_type"] == "live").astype(int)

    # Season indicator
    df["is_spring_sale"] = df["auction_month"].isin([3, 4, 5]).astype(int)
    df["is_fall_sale"] = df["auction_month"].isin([9, 10, 11]).astype(int)

    # --- Auction house prestige (major houses get higher prices) ---
    prestige_map = {"christies": 3, "sothebys": 3, "bonhams": 2,
                    "pundoles": 1, "saffronart": 1, "astaguru": 1, "artist_scrape": 0}
    df["auction_house_prestige"] = df["source"].map(prestige_map).fillna(0).astype(int)

    # --- Title-based subject (works even when no image) ---
    def extract_title_subject(title):
        if pd.isna(title):
            return "unknown"
        t = str(title).lower()
        if "untitled" in t and len(t) < 12:
            return "untitled"
        for subj, keywords in [
            ("horse", ["horse", "equestrian", "gallop"]),
            ("nude", ["nude", "naked"]),
            ("landscape", ["landscape", "village", "cityscape", "city"]),
            ("portrait", ["portrait", "head", "face", "self-portrait"]),
            ("woman", ["woman", "girl", "mother", "lady", "bride"]),
            ("abstract", ["abstract", "composition", "bindu", "untitled ("]),
            ("still_life", ["still life", "flower", "vase", "fruit"]),
            ("animal", ["bull", "cow", "bird", "cat", "dog", "fish", "animal"]),
            ("religious", ["krishna", "shiva", "ganesha", "buddha", "christ", "goddess"]),
            ("figure", ["figure", "dancer", "musician", "seated", "standing"]),
        ]:
            if any(kw in t for kw in keywords):
                return subj
        return "other"

    df["title_subject"] = df["title"].apply(extract_title_subject)

    # --- Provenance quality score ---
    def score_provenance(prov_text):
        if pd.isna(prov_text) or not str(prov_text).strip():
            return 0
        p = str(prov_text).lower()
        score = 0
        if "museum" in p or "national gallery" in p or "tate" in p:
            score += 4
        if "gallery" in p or "galerie" in p:
            score += 2
        if "exhibited" in p or "exhibition" in p:
            score += 2
        if "published" in p or "illustrated" in p:
            score += 1
        if "collection" in p:
            score += 1
        if "acquired directly" in p or "from the artist" in p:
            score += 3
        if "estate" in p or "family" in p:
            score += 2
        return score

    df["provenance_quality"] = df["provenance_text"].apply(score_provenance)

    # --- Creation period (artist-relative) ---
    df["creation_period"] = "unknown"
    has_years = df["year_created"].notna() & df["artist_birth_year"].notna()
    age_at_creation = df.loc[has_years, "year_created"] - df.loc[has_years, "artist_birth_year"]
    df.loc[has_years & (age_at_creation < 30), "creation_period"] = "early"
    df.loc[has_years & (age_at_creation >= 30) & (age_at_creation < 50), "creation_period"] = "mid_career"
    df.loc[has_years & (age_at_creation >= 50) & (age_at_creation < 70), "creation_period"] = "late"
    df.loc[has_years & (age_at_creation >= 70), "creation_period"] = "very_late"

    # --- Rarity score (how often this artist appears at auction per year) ---
    if "artist_prior_lots_total" in df.columns and "artist_years_in_market" in df.columns:
        years_active = df["artist_years_in_market"].replace(0, 1)
        df["artist_rarity"] = df["artist_prior_lots_total"] / years_active
        # Invert: fewer lots per year = more rare = higher score
        df["artist_rarity"] = 1.0 / (df["artist_rarity"] + 0.1)

    logger.info("  Done: lot-level features")
    return df


def compute_comparable_sales(df: pd.DataFrame) -> pd.DataFrame:
    """For each lot, find the most similar prior sale by the same artist.

    This mimics what appraisers do: 'a similar Souza oil, 60x45cm, sold for $X.'
    Uses only PRIOR data to avoid leakage.
    """
    logger.info("Computing comparable sales features...")

    df = df.sort_values("auction_date").reset_index(drop=True)

    comp_price = np.full(len(df), np.nan)
    comp_ratio = np.full(len(df), np.nan)  # current estimate / comp price
    comp_recency_days = np.full(len(df), np.nan)

    # Group by artist for efficiency
    for artist, group in df.groupby("artist_name_clean"):
        group = group.sort_values("auction_date")
        idxs = group.index.tolist()

        # Track prior sold lots for this artist
        prior_sold = []  # list of (idx, price, area, year_created, date, medium)

        for pos, idx in enumerate(idxs):
            row = df.loc[idx]

            if prior_sold:
                # Find best comparable: minimize distance in (size, year, medium)
                current_area = row.get("surface_area_cm2")
                current_year = row.get("year_created")
                current_medium = row.get("medium_category", "")
                current_date = row.get("auction_date")

                best_score = float("inf")
                best_price = None
                best_date = None

                for p_idx, p_price, p_area, p_year, p_date, p_medium in prior_sold:
                    score = 0
                    # Size similarity (log ratio, lower = more similar)
                    if pd.notna(current_area) and pd.notna(p_area) and p_area > 0 and current_area > 0:
                        score += abs(np.log(current_area / p_area)) * 2
                    else:
                        score += 2  # penalty for missing size

                    # Year similarity
                    if pd.notna(current_year) and pd.notna(p_year):
                        score += abs(current_year - p_year) / 10
                    else:
                        score += 1

                    # Medium match bonus
                    if current_medium and p_medium and current_medium == p_medium:
                        score -= 0.5  # bonus for same medium

                    # Recency bonus (prefer recent comps)
                    if pd.notna(current_date) and pd.notna(p_date):
                        days_ago = (current_date - p_date).days
                        score += days_ago / 3650  # slight penalty for old comps

                    if score < best_score:
                        best_score = score
                        best_price = p_price
                        best_date = p_date

                if best_price is not None:
                    comp_price[idx] = best_price
                    if pd.notna(current_date) and pd.notna(best_date):
                        comp_recency_days[idx] = (current_date - best_date).days

            # Add this lot to prior sold if it sold
            if row.get("is_sold") and pd.notna(row.get("hammer_price_usd")) and row["hammer_price_usd"] > 0:
                prior_sold.append((
                    idx, row["hammer_price_usd"],
                    row.get("surface_area_cm2"),
                    row.get("year_created"),
                    row.get("auction_date"),
                    row.get("medium_category", ""),
                ))

    df["comp_price"] = comp_price
    df["log_comp_price"] = np.log1p(comp_price)
    df["comp_recency_days"] = comp_recency_days

    n_with_comp = np.isfinite(comp_price).sum()
    logger.info(f"  Done: comparable sales ({n_with_comp}/{len(df)} lots have comps)")
    return df


def compute_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute market-level features (rolling market index)."""
    logger.info("Computing market features...")

    df = df.sort_values("auction_date").reset_index(drop=True)

    # Compute rolling market index: average log(hammer price) over prior 365 days
    market_idx = []
    for i, row in df.iterrows():
        date = row["auction_date"]
        if pd.isna(date):
            market_idx.append(np.nan)
            continue
        # All sold lots in prior 365 days
        mask = (
            (df.index < i)
            & df["is_sold"]
            & df["hammer_price_usd"].notna()
            & (df["hammer_price_usd"] > 0)
            & ((date - df["auction_date"]).dt.days <= 365)
            & ((date - df["auction_date"]).dt.days > 0)
        )
        prior = df.loc[mask, "hammer_price_usd"]
        if len(prior) >= 5:
            market_idx.append(np.log(prior.median()))
        else:
            market_idx.append(np.nan)

    df["market_index"] = market_idx

    logger.info("  Done: market features")
    return df


def create_ml_ready(df: pd.DataFrame) -> pd.DataFrame:
    """Create ML-ready dataset with only numeric features + target."""
    # Target: log hammer price (only for sold lots with valid prices)
    sold = df[df["is_sold"] & df["hammer_price_usd"].notna() & (df["hammer_price_usd"] > 0)].copy()
    sold["log_hammer_price"] = np.log(sold["hammer_price_usd"])

    # Merge image features if available
    image_features_file = DATA_PROCESSED / "image_features.csv"
    if image_features_file.exists():
        img_feats = pd.read_csv(image_features_file)
        # Drop the large clip_embedding column
        if "clip_embedding" in img_feats.columns:
            img_feats = img_feats.drop(columns=["clip_embedding"])
        img_feats["lot_id"] = img_feats["lot_id"].astype(str)
        sold["lot_id"] = sold["lot_id"].astype(str)
        sold = sold.merge(img_feats, on="lot_id", how="left")
        n_with_img = sold["subject"].notna().sum()
        logger.info(f"Merged image features: {n_with_img}/{len(sold)} lots have image data")
    else:
        logger.info("No image features found — skipping (run models/extract_image_features.py)")

    logger.info(f"ML-ready: {len(sold)} sold lots with valid prices")

    # Feature columns
    numeric_features = [
        # Artist rolling
        "artist_prior_lots_total",
        "artist_prior_lots_sold",
        "artist_sell_through_rate",
        "artist_avg_price",
        "artist_median_price",
        "artist_max_price",
        "artist_min_price",
        "artist_price_std",
        "artist_price_trend",
        "artist_avg_estimate_accuracy",
        "artist_years_in_market",
        "artist_days_since_last_sale",
        "artist_desirability_score",
        # Lot features
        "height_cm",
        "width_cm",
        "surface_area_cm2",
        "year_created",
        "artwork_age",
        "artist_age_at_creation",
        "artist_deceased",
        "is_signed",
        "is_dated",
        "provenance_count",
        "literature_count",
        "exhibition_count",
        "has_provenance",
        "has_literature",
        "has_exhibitions",
        "estimate_midpoint",
        "estimate_spread",
        "log_estimate_mid",
        "is_live_auction",
        "is_spring_sale",
        "is_fall_sale",
        "auction_year",
        # Market
        "market_index",
        # Time-series / momentum features
        "artist_recent_avg_price",
        "artist_recent_median_price",
        "artist_recent_max_price",
        "artist_momentum",
        "artist_price_velocity",
        "artist_price_per_cm2",
        "artist_recent_lots_12m",
        "artist_premium_trend",
        # Derived lot features
        "auction_house_prestige",
        "provenance_quality",
        "artist_rarity",
        # Comparable sales
        "comp_price",
        "log_comp_price",
        "comp_recency_days",
        # Image features (from CLIP + color analysis)
        "color_richness",
        "brightness",
        "subject_score",
        "palette_score",
        "style_score",
    ]

    # Categorical features (for CatBoost native handling)
    cat_features = ["medium_category", "artist_name_clean", "subject", "palette", "style",
                     "title_subject", "creation_period"]

    # Keep only available columns
    available_numeric = [c for c in numeric_features if c in sold.columns]
    available_cat = [c for c in cat_features if c in sold.columns]

    feature_cols = available_numeric + available_cat
    target_col = "log_hammer_price"

    # Build ML-ready DataFrame
    ml = sold[feature_cols + [target_col, "auction_date", "hammer_price_usd",
                               "artist_name", "title", "lot_id"]].copy()

    # Fill categoricals
    for col in available_cat:
        ml[col] = ml[col].fillna("unknown")

    logger.info(f"ML features: {len(available_numeric)} numeric + {len(available_cat)} categorical")
    logger.info(f"Target: {target_col}")

    return ml


def main():
    logger.info("=" * 60)
    logger.info("Feature Engineering Pipeline")
    logger.info("=" * 60)

    if not LOTS_FILE.exists():
        logger.error(f"Lots file not found: {LOTS_FILE}")
        logger.error("Run scrape_lots.py first!")
        sys.exit(1)

    # Load
    df = load_data()

    # Engineer features
    df = compute_artist_rolling_features(df)
    df = compute_lot_features(df)
    df = compute_comparable_sales(df)
    df = compute_market_features(df)

    # Save master
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(MASTER_FILE, index=False)
    logger.info(f"\nSaved master.csv: {len(df)} rows × {len(df.columns)} cols → {MASTER_FILE}")

    # Create ML-ready
    ml = create_ml_ready(df)
    ml.to_csv(ML_READY_FILE, index=False)
    logger.info(f"Saved ml_ready.csv: {len(ml)} rows × {len(ml.columns)} cols → {ML_READY_FILE}")

    # Summary stats
    logger.info(f"\n--- Summary ---")
    logger.info(f"Total lots: {len(df)}")
    logger.info(f"Sold lots: {df['is_sold'].sum()}")
    logger.info(f"Unique artists: {df['artist_name_clean'].nunique()}")
    if df['hammer_price_usd'].notna().any():
        logger.info(f"Price range: ${df['hammer_price_usd'].min():,.0f} - ${df['hammer_price_usd'].max():,.0f}")
        logger.info(f"Median price: ${df['hammer_price_usd'].median():,.0f}")
    logger.info(f"Date range: {df['auction_date'].min()} to {df['auction_date'].max()}")


if __name__ == "__main__":
    main()
