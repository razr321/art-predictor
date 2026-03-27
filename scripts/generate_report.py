#!/usr/bin/env python3
"""
Generate a comprehensive HTML analytics report for the Indian Art Market dataset.
Outputs a single self-contained HTML file with embedded CSS, JS, and Chart.js charts.
"""

import json
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = "/Users/sidkumar/Desktop/Art/art_predictor"
MASTER_CSV = os.path.join(BASE, "data/processed/master.csv")
ML_CSV = os.path.join(BASE, "data/processed/ml_ready.csv")
MANIFEST = os.path.join(BASE, "models/saved/model_manifest.json")
OUT_HTML = os.path.join(BASE, "reports/market_report.html")

# ── star artists mapping (display name → list of raw names to match) ─────────
STAR_ARTISTS = {
    "Tyeb Mehta": ["TYEB MEHTA"],
    "F.N. Souza": ["FRANCIS NEWTON SOUZA", "F N SOUZA"],
    "M.F. Husain": ["MAQBOOL FIDA HUSAIN", "M F HUSAIN"],
    "Ram Kumar": ["RAM KUMAR"],
    "Akbar Padamsee": ["AKBAR PADAMSEE"],
    "S.H. Raza": ["SAYED HAIDER RAZA", "S H RAZA"],
    "V.S. Gaitonde": ["VASUDEO S. GAITONDE", "V S GAITONDE"],
}

STAR_COLORS = {
    "Tyeb Mehta": "#e74c3c",
    "F.N. Souza": "#3498db",
    "M.F. Husain": "#2ecc71",
    "Ram Kumar": "#f39c12",
    "Akbar Padamsee": "#9b59b6",
    "S.H. Raza": "#1abc9c",
    "V.S. Gaitonde": "#e67e22",
}

# ── helpers ────────────────────────────────────────────────────────────────────

def fmt_price(v):
    if pd.isna(v) or v is None:
        return "N/A"
    return f"${v:,.0f}"

def fmt_pct(v):
    if pd.isna(v) or v is None:
        return "N/A"
    return f"{v:.1f}%"

def fmt_num(v):
    if pd.isna(v) or v is None:
        return "N/A"
    return f"{v:,.0f}"

def safe_json(obj):
    """Convert to JSON-safe value."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def to_json(obj):
    """Recursively make json-safe then dump."""
    if isinstance(obj, dict):
        return json.dumps({k: to_json_val(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return json.dumps([to_json_val(v) for v in obj])
    return json.dumps(safe_json(obj))

def to_json_val(obj):
    if isinstance(obj, dict):
        return {k: to_json_val(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_val(v) for v in obj]
    return safe_json(obj)


# ── load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
master = pd.read_csv(MASTER_CSV, low_memory=False)
ml = pd.read_csv(ML_CSV, low_memory=False)
with open(MANIFEST) as f:
    manifest = json.load(f)

# parse dates
master["auction_date"] = pd.to_datetime(
    master["auction_date"], errors="coerce", format="mixed", utc=True
).dt.tz_localize(None)
ml["auction_date"] = pd.to_datetime(
    ml["auction_date"], errors="coerce", format="mixed", utc=True
).dt.tz_localize(None)

# ensure auction_year
if "auction_year" not in master.columns or master["auction_year"].isna().all():
    master["auction_year"] = master["auction_date"].dt.year
else:
    master["auction_year"] = master["auction_year"].fillna(master["auction_date"].dt.year)

if "auction_year" not in ml.columns or ml["auction_year"].isna().all():
    ml["auction_year"] = ml["auction_date"].dt.year

# normalise artist name
master["artist_upper"] = master["artist_name"].str.strip().str.upper()
ml["artist_upper"] = ml["artist_name"].str.strip().str.upper()

# map star artists
def map_star(name):
    if pd.isna(name):
        return None
    for display, variants in STAR_ARTISTS.items():
        if name in variants:
            return display
    return None

master["star_artist"] = master["artist_upper"].apply(map_star)
ml["star_artist"] = ml["artist_upper"].apply(map_star)

# sold subset of master
sold_master = master[master["is_sold"] == True].copy()

print(f"  master rows: {len(master)}, sold: {len(sold_master)}, ml_ready: {len(ml)}")

# ── Section 1: Executive Summary ──────────────────────────────────────────────
total_lots = len(master)
sold_lots = int(master["is_sold"].sum())
sources = master["source"].dropna().unique().tolist() if "source" in master.columns else []
date_min = master["auction_date"].min()
date_max = master["auction_date"].max()
model_r2 = manifest["metrics"]["r2_log"]
model_median_ae = manifest["metrics"]["median_ae_usd"]
model_rmse = manifest["metrics"]["rmse_log"]
model_mape = manifest["metrics"]["mape_pct"]
total_hammer = sold_master["hammer_price_usd"].sum()

# ── Section 2: Market Overview ─────────────────────────────────────────────────
yearly_volume = (
    sold_master.groupby("auction_year")
    .agg(lots_sold=("lot_id", "count"), total_hammer=("hammer_price_usd", "sum"))
    .sort_index()
)
yearly_volume = yearly_volume[yearly_volume.index.notna()].copy()
yearly_volume.index = yearly_volume.index.astype(int)
market_years = yearly_volume.index.tolist()
market_lots = yearly_volume["lots_sold"].tolist()
market_hammer = yearly_volume["total_hammer"].tolist()

# ── Section 3: Star Artists YoY ───────────────────────────────────────────────
star_yoy = {}
for display_name in STAR_ARTISTS:
    variants = STAR_ARTISTS[display_name]
    mask = master["artist_upper"].isin(variants)
    df_artist = master[mask].copy()
    sold_artist = df_artist[df_artist["is_sold"] == True].copy()

    if sold_artist.empty:
        continue

    grp = sold_artist.groupby("auction_year")["hammer_price_usd"]
    total_grp = df_artist.groupby("auction_year")["lot_id"].count()
    sold_grp = sold_artist.groupby("auction_year")["lot_id"].count()

    stats = pd.DataFrame({
        "median": grp.median(),
        "mean": grp.mean(),
        "max": grp.max(),
        "count_sold": sold_grp,
    })
    stats["count_total"] = total_grp
    stats = stats.fillna(0)
    stats["sell_through"] = (stats["count_sold"] / stats["count_total"] * 100).round(1)
    stats = stats[stats.index.notna()].copy()
    stats.index = stats.index.astype(int)
    stats = stats.sort_index()

    star_yoy[display_name] = {
        "years": stats.index.tolist(),
        "median": stats["median"].tolist(),
        "mean": stats["mean"].tolist(),
        "max": stats["max"].tolist(),
        "count_sold": stats["count_sold"].astype(int).tolist(),
        "count_total": stats["count_total"].astype(int).tolist(),
        "sell_through": stats["sell_through"].tolist(),
    }

# ── Section 4: Price Distribution ─────────────────────────────────────────────
# log10 bins for histogram
sold_prices = sold_master["hammer_price_usd"].dropna()
sold_prices = sold_prices[sold_prices > 0]
log_prices = np.log10(sold_prices)
bins = np.arange(0, math.ceil(log_prices.max()) + 0.5, 0.25)
hist_all, _ = np.histogram(log_prices, bins=bins)
hist_labels = [f"${10**b:,.0f}" for b in bins[:-1]]

star_hists = {}
for display_name in STAR_ARTISTS:
    variants = STAR_ARTISTS[display_name]
    mask = sold_master["artist_upper"].isin(variants)
    sp = sold_master.loc[mask, "hammer_price_usd"].dropna()
    sp = sp[sp > 0]
    if len(sp) == 0:
        continue
    h, _ = np.histogram(np.log10(sp), bins=bins)
    star_hists[display_name] = h.tolist()

# ── Section 5: Model Performance ──────────────────────────────────────────────
# We need predictions. Load models and predict on ml_ready.
try:
    import catboost as cb

    feature_cols = manifest["feature_cols"]
    cat_cols = manifest["cat_cols"]
    target = manifest["target"]

    X = ml[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)
    y_true_log = ml[target].values
    y_true = ml["hammer_price_usd"].values

    # ensemble predict
    preds = []
    for mf in manifest["catboost_models"]:
        model_path = os.path.join(BASE, "models/saved", mf)
        if os.path.exists(model_path):
            m = cb.CatBoostRegressor()
            m.load_model(model_path)
            preds.append(m.predict(X))

    # Try xgboost
    try:
        import xgboost as xgb
        for mf in manifest["xgboost_models"]:
            model_path = os.path.join(BASE, "models/saved", mf)
            if os.path.exists(model_path):
                m = xgb.XGBRegressor()
                m.load_model(model_path)
                # need to handle cat cols for xgb - encode them
                X_xgb = X.copy()
                for c in cat_cols:
                    X_xgb[c] = X_xgb[c].astype("category").cat.codes
                preds.append(m.predict(X_xgb))
    except ImportError:
        print("  xgboost not available, skipping xgb models")

    if preds:
        y_pred_log = np.mean(preds, axis=0)
        y_pred = np.exp(y_pred_log)

        # accuracy metrics
        def accuracy_stats(y_t, y_p):
            pct_err = np.abs(y_t - y_p) / np.where(y_t > 0, y_t, 1)
            return {
                "within_10": float((pct_err <= 0.10).mean() * 100),
                "within_25": float((pct_err <= 0.25).mean() * 100),
                "within_50": float((pct_err <= 0.50).mean() * 100),
                "median_ae": float(np.median(np.abs(y_t - y_p))),
                "count": int(len(y_t)),
            }

        overall_acc = accuracy_stats(y_true, y_pred)

        # per star artist
        star_acc = {}
        ml_star = ml["star_artist"].values
        twelve_months_ago = pd.Timestamp.now() - pd.DateOffset(months=12)
        ml_dates = ml["auction_date"].values

        for display_name in STAR_ARTISTS:
            mask = ml["star_artist"] == display_name
            if mask.sum() == 0:
                continue
            acc_all = accuracy_stats(y_true[mask], y_pred[mask])

            # last 12 months
            mask_recent = mask & (ml["auction_date"] >= twelve_months_ago)
            if mask_recent.sum() > 0:
                acc_recent = accuracy_stats(y_true[mask_recent.values], y_pred[mask_recent.values])
            else:
                acc_recent = None

            star_acc[display_name] = {"all_time": acc_all, "recent": acc_recent}

        # overall recent
        mask_recent_all = ml["auction_date"] >= twelve_months_ago
        if mask_recent_all.sum() > 0:
            overall_acc_recent = accuracy_stats(y_true[mask_recent_all.values], y_pred[mask_recent_all.values])
        else:
            overall_acc_recent = None

        has_model_perf = True
    else:
        has_model_perf = False
except Exception as e:
    print(f"  Model loading failed: {e}")
    has_model_perf = False

# ── Section 6: Top Sales ──────────────────────────────────────────────────────
top_sales = (
    sold_master.nlargest(25, "hammer_price_usd")[
        ["artist_name", "title", "hammer_price_usd", "auction_date", "auction_year", "medium_category"]
    ]
    .copy()
)
top_sales["auction_date_str"] = top_sales["auction_date"].dt.strftime("%Y-%m-%d")
top_sales_list = []
for _, row in top_sales.iterrows():
    top_sales_list.append({
        "artist": row["artist_name"],
        "title": str(row["title"])[:80] if pd.notna(row["title"]) else "Untitled",
        "price": fmt_price(row["hammer_price_usd"]),
        "price_raw": float(row["hammer_price_usd"]),
        "date": row["auction_date_str"] if pd.notna(row["auction_date_str"]) else "",
        "medium": str(row["medium_category"]) if pd.notna(row["medium_category"]) else "",
    })

# ── Section 7: Source Comparison ──────────────────────────────────────────────
source_stats = []
if "source" in master.columns:
    for src in sorted(master["source"].dropna().unique()):
        src_df = master[master["source"] == src]
        src_sold = src_df[src_df["is_sold"] == True]
        source_stats.append({
            "source": src,
            "total_lots": int(len(src_df)),
            "sold_lots": int(len(src_sold)),
            "sell_through": round(len(src_sold) / len(src_df) * 100, 1) if len(src_df) > 0 else 0,
            "median_price": float(src_sold["hammer_price_usd"].median()) if len(src_sold) > 0 else 0,
            "mean_price": float(src_sold["hammer_price_usd"].mean()) if len(src_sold) > 0 else 0,
            "total_hammer": float(src_sold["hammer_price_usd"].sum()) if len(src_sold) > 0 else 0,
        })

# ── Load models and compute fair values on ml_ready data ─────────────────────
print("Loading models for fair value predictions...")
try:
    from catboost import CatBoostRegressor
    import xgboost as xgb

    _models = []
    for i in range(1, 6):
        _m = CatBoostRegressor()
        _m.load_model(os.path.join(BASE, f"models/saved/model_{i}.cbm"))
        _models.append(("cb", _m))
    for i in range(1, 3):
        _m = xgb.Booster()
        _m.load_model(os.path.join(BASE, f"models/saved/xgb_model_{i}.json"))
        _models.append(("xgb", _m))

    _feature_cols = manifest["feature_cols"]
    _cat_cols = manifest.get("cat_cols", [])

    def _predict_ensemble(df_sub):
        X = df_sub[_feature_cols].copy()
        preds = []
        for mt, m in _models:
            if mt == "cb":
                preds.append(m.predict(X))
            else:
                Xx = X.copy()
                for c in _cat_cols:
                    if c in Xx.columns:
                        Xx[c] = Xx[c].astype("category")
                preds.append(m.predict(xgb.DMatrix(Xx, enable_categorical=True)))
        return np.mean(preds, axis=0)

    ml["predicted_log"] = _predict_ensemble(ml)
    ml["predicted_usd"] = np.expm1(ml["predicted_log"])
    has_predictions = True
    print(f"  Predictions computed for {len(ml)} lots")
except Exception as e:
    print(f"  Warning: Could not load models for fair value: {e}")
    has_predictions = False

# ── Section 8 & 9: Index Performance ─────────────────────────────────────────

# Core 4 Index: Souza, Raza, Tyeb Mehta, Husain
CORE_4 = ["F.N. Souza", "S.H. Raza", "Tyeb Mehta", "M.F. Husain"]
# Full Star Index: anyone with a sale >= $500K in last 10 years
cutoff_10yr = pd.Timestamp("2016-01-01")
_recent_sold = sold_master[(sold_master["auction_date"] >= cutoff_10yr) & (sold_master["hammer_price_usd"].notna())]
_artist_max_10yr = _recent_sold.groupby("star_artist")["hammer_price_usd"].max()
# Also check non-star artists by artist_upper
_all_artist_max = _recent_sold.groupby("artist_upper")["hammer_price_usd"].max()
_big_artists_upper = set(_all_artist_max[_all_artist_max >= 500000].index.tolist())
# Map to display names where possible, keep raw names otherwise
_star_upper_to_display = {}
for display, variants in STAR_ARTISTS.items():
    for v in variants:
        _star_upper_to_display[v] = display
full_star_names = set()
for au in _big_artists_upper:
    full_star_names.add(_star_upper_to_display.get(au, au))

def _build_index_yoy(artist_set, label):
    """Build year-over-year index data for a set of artist display names."""
    # Determine mask using both star_artist column and artist_upper
    mask = pd.Series(False, index=sold_master.index)
    for name in artist_set:
        if name in STAR_ARTISTS:
            for v in STAR_ARTISTS[name]:
                mask = mask | (sold_master["artist_upper"] == v)
        else:
            mask = mask | (sold_master["artist_upper"] == name)
    idx_df = sold_master[mask & sold_master["hammer_price_usd"].notna()].copy()

    # Also build "rest of market" for comparison
    rest_df = sold_master[~mask & sold_master["hammer_price_usd"].notna()].copy()

    # Build ml mask for fair value (ml_ready has predictions)
    ml_mask = pd.Series(False, index=ml.index)
    for name in artist_set:
        if name in STAR_ARTISTS:
            for v in STAR_ARTISTS[name]:
                ml_mask = ml_mask | (ml["artist_upper"] == v)
        else:
            ml_mask = ml_mask | (ml["artist_upper"] == name)
    ml_idx = ml[ml_mask].copy()
    ml_rest = ml[~ml_mask].copy()

    years = sorted(idx_df["auction_year"].dropna().unique().astype(int).tolist())
    idx_data = {"years": [], "median": [], "mean": [], "count": [], "total": [],
                "fair_median": [], "fair_mean": []}
    rest_data = {"years": [], "median": [], "mean": [], "count": [], "total": [],
                 "fair_median": [], "fair_mean": []}

    for yr in years:
        yr_idx = idx_df[idx_df["auction_year"] == yr]["hammer_price_usd"]
        yr_rest = rest_df[rest_df["auction_year"] == yr]["hammer_price_usd"]
        if len(yr_idx) >= 1:
            idx_data["years"].append(yr)
            idx_data["median"].append(round(float(yr_idx.median()), 0))
            idx_data["mean"].append(round(float(yr_idx.mean()), 0))
            idx_data["count"].append(int(len(yr_idx)))
            idx_data["total"].append(round(float(yr_idx.sum()), 0))
            # Fair value from model predictions
            if has_predictions and "auction_year" in ml_idx.columns:
                yr_ml = ml_idx[ml_idx["auction_year"] == yr]["predicted_usd"]
                idx_data["fair_median"].append(round(float(yr_ml.median()), 0) if len(yr_ml) else None)
                idx_data["fair_mean"].append(round(float(yr_ml.mean()), 0) if len(yr_ml) else None)
            else:
                idx_data["fair_median"].append(None)
                idx_data["fair_mean"].append(None)
        if len(yr_rest) >= 1:
            if yr not in rest_data["years"]:
                rest_data["years"].append(yr)
                rest_data["median"].append(round(float(yr_rest.median()), 0))
                rest_data["mean"].append(round(float(yr_rest.mean()), 0))
                rest_data["count"].append(int(len(yr_rest)))
                rest_data["total"].append(round(float(yr_rest.sum()), 0))
                if has_predictions and "auction_year" in ml_rest.columns:
                    yr_mlr = ml_rest[ml_rest["auction_year"] == yr]["predicted_usd"]
                    rest_data["fair_median"].append(round(float(yr_mlr.median()), 0) if len(yr_mlr) else None)
                    rest_data["fair_mean"].append(round(float(yr_mlr.mean()), 0) if len(yr_mlr) else None)
                else:
                    rest_data["fair_median"].append(None)
                    rest_data["fair_mean"].append(None)

    # Build constituent artists table
    artist_stats = []
    for name in sorted(artist_set):
        a_mask = pd.Series(False, index=sold_master.index)
        if name in STAR_ARTISTS:
            for v in STAR_ARTISTS[name]:
                a_mask = a_mask | (sold_master["artist_upper"] == v)
        else:
            a_mask = a_mask | (sold_master["artist_upper"] == name)
        a_df = sold_master[a_mask & sold_master["hammer_price_usd"].notna()]
        if len(a_df) == 0:
            continue
        # Recent 12m
        a_recent = a_df[a_df["auction_date"] >= pd.Timestamp("2024-03-26")]
        artist_stats.append({
            "name": name,
            "n": int(len(a_df)),
            "n_12m": int(len(a_recent)),
            "median_all": float(a_df["hammer_price_usd"].median()),
            "median_12m": float(a_recent["hammer_price_usd"].median()) if len(a_recent) else None,
            "mean_all": float(a_df["hammer_price_usd"].mean()),
            "max_all": float(a_df["hammer_price_usd"].max()),
        })

    return idx_data, rest_data, artist_stats

core4_idx, core4_rest, core4_artists = _build_index_yoy(set(CORE_4), "Core 4")
fullstar_idx, fullstar_rest, fullstar_artists = _build_index_yoy(full_star_names, "Full Star")

print(f"  Core 4 Index: {sum(core4_idx['count'])} lots, {len(core4_artists)} artists")
print(f"  Full Star Index: {sum(fullstar_idx['count'])} lots, {len(fullstar_artists)} artists")

# ── Section 10: CAGR & Hedonic Price Index ───────────────────────────────────
print("Computing CAGRs and hedonic indices...")

def _compute_cagr_and_index(artist_mask_fn, ml_df):
    """Compute hedonic price index and CAGR.

    Three approaches:
    1. Simple CAGR: log-linear regression of log(price) ~ year (all lots)
    2. Top-tier CAGR: regression on the top quartile (75th pctile+) per year
       — tracks the market for quality work, not dragged down by minor pieces
    3. Hedonic index: ratio of actual/predicted per year, rebased to 100
       — model normalizes for size, medium, provenance → isolates pure appreciation
    """
    from scipy import stats

    sub = ml_df[artist_mask_fn(ml_df)].copy()
    if len(sub) < 5:
        return None

    sub = sub[sub["auction_year"].notna() & sub["hammer_price_usd"].notna()].copy()
    sub["log_price"] = np.log(sub["hammer_price_usd"].clip(lower=1))
    sub["year"] = sub["auction_year"].astype(int)

    years = sorted(sub["year"].unique())
    if len(years) < 2:
        return None

    # 1. Simple CAGR from log-linear regression (all lots)
    slope, intercept, r_value, _, _ = stats.linregress(sub["year"], sub["log_price"])
    simple_cagr = (np.exp(slope) - 1) * 100

    # 2. Top-tier CAGR: use 75th percentile price per year
    p75_per_year = sub.groupby("year")["hammer_price_usd"].quantile(0.75)
    if len(p75_per_year) >= 2:
        log_p75 = np.log(p75_per_year.clip(lower=1))
        slope_top, _, r_top, _, _ = stats.linregress(p75_per_year.index, log_p75)
        top_cagr = (np.exp(slope_top) - 1) * 100
        r_sq_top = r_top ** 2
    else:
        top_cagr = simple_cagr
        r_sq_top = r_value ** 2

    # 3. Hedonic CAGR using model quality-adjustment
    if has_predictions and "predicted_usd" in sub.columns:
        sub["quality_ratio"] = sub["hammer_price_usd"] / sub["predicted_usd"].clip(lower=1)
        sub["log_ratio"] = np.log(sub["quality_ratio"].clip(lower=0.01))
        slope_h, _, r_h, _, _ = stats.linregress(sub["year"], sub["log_ratio"])
        hedonic_cagr = (np.exp(slope_h) - 1) * 100
    else:
        hedonic_cagr = top_cagr

    # 4. Build rebased indices (base year = earliest year with enough data = 100)
    base_year = min(years)
    index_simple = []
    index_top = []
    index_hedonic = []

    base_med = sub[sub["year"] == base_year]["hammer_price_usd"].median()
    base_p75 = sub[sub["year"] == base_year]["hammer_price_usd"].quantile(0.75)
    if has_predictions and "quality_ratio" in sub.columns:
        base_qr = sub[sub["year"] == base_year]["quality_ratio"].median()
    else:
        base_qr = 1

    for yr in years:
        yr_sub = sub[sub["year"] == yr]
        if len(yr_sub) == 0:
            index_simple.append(None)
            index_top.append(None)
            index_hedonic.append(None)
            continue

        # Simple median
        med = yr_sub["hammer_price_usd"].median()
        index_simple.append(round(med / base_med * 100, 1) if base_med > 0 else 100)

        # Top-tier (75th percentile)
        p75 = yr_sub["hammer_price_usd"].quantile(0.75)
        index_top.append(round(p75 / base_p75 * 100, 1) if base_p75 > 0 else 100)

        # Hedonic (quality-adjusted ratio)
        if has_predictions and "quality_ratio" in yr_sub.columns:
            qr = yr_sub["quality_ratio"].median()
            index_hedonic.append(round(qr / base_qr * 100, 1) if base_qr > 0 else 100)
        else:
            index_hedonic.append(index_top[-1])

    return {
        "years": [int(y) for y in years],
        "simple_cagr": round(float(simple_cagr), 1),
        "top_cagr": round(float(top_cagr), 1),
        "hedonic_cagr": round(float(hedonic_cagr), 1),
        "index_simple": [float(x) if x is not None else None for x in index_simple],
        "index_top": [float(x) if x is not None else None for x in index_top],
        "index_hedonic": [float(x) if x is not None else None for x in index_hedonic],
        "n_lots": int(len(sub)),
        "n_years": int(len(years)),
        "r_squared": round(float(r_sq_top), 3),
    }

def _make_artist_mask(artist_set):
    def mask_fn(df):
        mask = pd.Series(False, index=df.index)
        for name in artist_set:
            if name in STAR_ARTISTS:
                for v in STAR_ARTISTS[name]:
                    mask = mask | (df["artist_upper"] == v)
            else:
                mask = mask | (df["artist_upper"] == name)
        return mask
    return mask_fn

# Individual artist CAGRs
artist_cagrs = {}
for display_name in STAR_ARTISTS:
    result = _compute_cagr_and_index(_make_artist_mask({display_name}), ml)
    if result:
        artist_cagrs[display_name] = result

# Core 4 and Full Star index CAGRs
core4_cagr = _compute_cagr_and_index(_make_artist_mask(set(CORE_4)), ml)
fullstar_cagr = _compute_cagr_and_index(_make_artist_mask(full_star_names), ml)
market_cagr = _compute_cagr_and_index(lambda df: pd.Series(True, index=df.index), ml)

print(f"  Artist CAGRs computed: {len(artist_cagrs)}")
if core4_cagr:
    print(f"  Core 4 CAGR: {core4_cagr['simple_cagr']:.1f}% simple, {core4_cagr['top_cagr']:.1f}% top-tier, {core4_cagr['hedonic_cagr']:.1f}% hedonic")
if fullstar_cagr:
    print(f"  Full Star CAGR: {fullstar_cagr['simple_cagr']:.1f}% simple, {fullstar_cagr['top_cagr']:.1f}% top-tier, {fullstar_cagr['hedonic_cagr']:.1f}% hedonic")
if market_cagr:
    print(f"  Full Market CAGR: {market_cagr['simple_cagr']:.1f}% simple, {market_cagr['top_cagr']:.1f}% top-tier, {market_cagr['hedonic_cagr']:.1f}% hedonic")

# ── Build HTML ─────────────────────────────────────────────────────────────────
print("Building HTML report...")

# Build star artist tables HTML
star_tables_html = ""
for display_name in STAR_ARTISTS:
    if display_name not in star_yoy:
        continue
    data = star_yoy[display_name]
    color = STAR_COLORS[display_name]
    rows = ""
    for i, yr in enumerate(data["years"]):
        rows += f"""<tr>
            <td>{yr}</td>
            <td>{data['count_sold'][i]}</td>
            <td>{fmt_price(data['median'][i])}</td>
            <td>{fmt_price(data['mean'][i])}</td>
            <td>{fmt_price(data['max'][i])}</td>
            <td>{data['sell_through'][i]:.1f}%</td>
        </tr>"""
    star_tables_html += f"""
    <div class="star-artist-block" id="block-{display_name.replace(' ', '-').replace('.', '')}">
        <h3 style="color:{color};border-bottom:2px solid {color};padding-bottom:8px;">{display_name}</h3>
        <div class="chart-row">
            <div class="chart-box">
                <canvas id="chart-median-{display_name.replace(' ', '-').replace('.', '')}"></canvas>
            </div>
            <div class="chart-box">
                <canvas id="chart-mean-{display_name.replace(' ', '-').replace('.', '')}"></canvas>
            </div>
        </div>
        <div class="table-scroll">
        <table>
            <thead><tr><th>Year</th><th># Sold</th><th>Median</th><th>Mean</th><th>Max</th><th>STR</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        </div>
    </div>"""

# Model perf table
model_perf_html = ""
if has_model_perf:
    def perf_row(name, acc_all, acc_recent, is_overall=False):
        cls = ' class="overall-row"' if is_overall else ""
        recent_10 = fmt_pct(acc_recent["within_10"]) if acc_recent else "N/A"
        recent_25 = fmt_pct(acc_recent["within_25"]) if acc_recent else "N/A"
        recent_50 = fmt_pct(acc_recent["within_50"]) if acc_recent else "N/A"
        recent_mae = fmt_price(acc_recent["median_ae"]) if acc_recent else "N/A"
        recent_n = fmt_num(acc_recent["count"]) if acc_recent else "N/A"
        return f"""<tr{cls}>
            <td>{name}</td>
            <td>{acc_all['count']}</td>
            <td>{fmt_pct(acc_all['within_10'])}</td>
            <td>{fmt_pct(acc_all['within_25'])}</td>
            <td>{fmt_pct(acc_all['within_50'])}</td>
            <td>{fmt_price(acc_all['median_ae'])}</td>
            <td>{recent_n}</td>
            <td>{recent_10}</td>
            <td>{recent_25}</td>
            <td>{recent_50}</td>
            <td>{recent_mae}</td>
        </tr>"""

    perf_rows = ""
    for dn in STAR_ARTISTS:
        if dn in star_acc:
            perf_rows += perf_row(dn, star_acc[dn]["all_time"], star_acc[dn]["recent"])
    perf_rows += perf_row("Overall", overall_acc, overall_acc_recent, is_overall=True)

    model_perf_html = f"""
    <div class="card">
        <h2>5. Model Performance</h2>
        <p class="subtitle">Ensemble: {manifest['metrics']['n_catboost']} CatBoost + {manifest['metrics']['n_xgboost']} XGBoost models
        &nbsp;|&nbsp; R&sup2; (log): {model_r2:.4f} &nbsp;|&nbsp; RMSE (log): {model_rmse:.4f} &nbsp;|&nbsp; MAPE: {model_mape:.1f}%</p>
        <div class="table-scroll">
        <table>
            <thead>
                <tr>
                    <th rowspan="2">Artist</th>
                    <th colspan="5" style="text-align:center;border-bottom:1px solid #444;">All Time</th>
                    <th colspan="5" style="text-align:center;border-bottom:1px solid #444;">Last 12 Months</th>
                </tr>
                <tr>
                    <th>N</th><th>&le;10%</th><th>&le;25%</th><th>&le;50%</th><th>Med AE</th>
                    <th>N</th><th>&le;10%</th><th>&le;25%</th><th>&le;50%</th><th>Med AE</th>
                </tr>
            </thead>
            <tbody>{perf_rows}</tbody>
        </table>
        </div>
    </div>"""

# Top sales table
top_sales_rows = ""
for i, s in enumerate(top_sales_list, 1):
    top_sales_rows += f"""<tr>
        <td>{i}</td>
        <td>{s['artist']}</td>
        <td>{s['title']}</td>
        <td class="price">{s['price']}</td>
        <td>{s['date']}</td>
        <td>{s['medium']}</td>
    </tr>"""

# Source comparison rows
source_rows = ""
for s in source_stats:
    source_rows += f"""<tr>
        <td>{s['source']}</td>
        <td>{fmt_num(s['total_lots'])}</td>
        <td>{fmt_num(s['sold_lots'])}</td>
        <td>{fmt_pct(s['sell_through'])}</td>
        <td>{fmt_price(s['median_price'])}</td>
        <td>{fmt_price(s['mean_price'])}</td>
        <td>{fmt_price(s['total_hammer'])}</td>
    </tr>"""

# ── Chart.js JavaScript ───────────────────────────────────────────────────────
# Star artist charts JS
star_charts_js = ""
for display_name in STAR_ARTISTS:
    if display_name not in star_yoy:
        continue
    data = star_yoy[display_name]
    color = STAR_COLORS[display_name]
    safe_id = display_name.replace(" ", "-").replace(".", "")
    years_json = json.dumps(data["years"])
    median_json = json.dumps([to_json_val(v) for v in data["median"]])
    mean_json = json.dumps([to_json_val(v) for v in data["mean"]])

    star_charts_js += f"""
    // {display_name} median
    new Chart(document.getElementById('chart-median-{safe_id}'), {{
        type: 'line',
        data: {{
            labels: {years_json},
            datasets: [{{
                label: 'Median Hammer Price',
                data: {median_json},
                borderColor: '{color}',
                backgroundColor: '{color}33',
                fill: true,
                tension: 0.3,
                pointRadius: 3,
            }}]
        }},
        options: {{
            responsive: true,
            plugins: {{
                title: {{ display: true, text: '{display_name} — Median Price by Year', color: '#e0e0e0', font: {{ size: 14 }} }},
                tooltip: {{ callbacks: {{ label: ctx => '$' + ctx.parsed.y.toLocaleString() }} }},
                legend: {{ labels: {{ color: '#ccc' }} }}
            }},
            scales: {{
                x: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#333' }} }},
                y: {{ ticks: {{ color: '#aaa', callback: v => '$' + v.toLocaleString() }}, grid: {{ color: '#333' }} }}
            }}
        }}
    }});
    // {display_name} mean
    new Chart(document.getElementById('chart-mean-{safe_id}'), {{
        type: 'line',
        data: {{
            labels: {years_json},
            datasets: [{{
                label: 'Mean Hammer Price',
                data: {mean_json},
                borderColor: '{color}',
                backgroundColor: '{color}33',
                borderDash: [5, 5],
                fill: true,
                tension: 0.3,
                pointRadius: 3,
            }}]
        }},
        options: {{
            responsive: true,
            plugins: {{
                title: {{ display: true, text: '{display_name} — Mean Price by Year', color: '#e0e0e0', font: {{ size: 14 }} }},
                tooltip: {{ callbacks: {{ label: ctx => '$' + ctx.parsed.y.toLocaleString() }} }},
                legend: {{ labels: {{ color: '#ccc' }} }}
            }},
            scales: {{
                x: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#333' }} }},
                y: {{ ticks: {{ color: '#aaa', callback: v => '$' + v.toLocaleString() }}, grid: {{ color: '#333' }} }}
            }}
        }}
    }});
    """

# Price distribution chart datasets
hist_all_json = json.dumps(hist_all.tolist())
hist_labels_json = json.dumps(hist_labels)
star_hist_datasets = ""
for dn, h in star_hists.items():
    color = STAR_COLORS[dn]
    star_hist_datasets += f"""{{
        label: '{dn}',
        data: {json.dumps(h)},
        backgroundColor: '{color}88',
        borderColor: '{color}',
        borderWidth: 1,
    }},"""


# ── Full HTML ──────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Indian Art Market — Analytics Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root {{
    --bg: #0f0f1a;
    --card-bg: #1a1a2e;
    --card-border: #2a2a45;
    --accent: #e94560;
    --accent2: #0f3460;
    --text: #e0e0e0;
    --text-dim: #8888aa;
    --success: #2ecc71;
    --warn: #f39c12;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 0;
}}
.container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
header {{
    background: linear-gradient(135deg, #16213e 0%, #0f3460 50%, #1a1a2e 100%);
    padding: 48px 24px;
    text-align: center;
    border-bottom: 3px solid var(--accent);
}}
header h1 {{ font-size: 2.4rem; font-weight: 700; letter-spacing: -0.5px; }}
header h1 span {{ color: var(--accent); }}
header p {{ color: var(--text-dim); margin-top: 8px; font-size: 1rem; }}
.card {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 28px;
}}
.card h2 {{
    font-size: 1.5rem;
    margin-bottom: 18px;
    color: #fff;
    border-left: 4px solid var(--accent);
    padding-left: 14px;
}}
.card .subtitle {{
    color: var(--text-dim);
    font-size: 0.9rem;
    margin-bottom: 16px;
    margin-top: -10px;
}}
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 10px;
}}
.kpi {{
    background: #16213e;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    border: 1px solid #253a5e;
}}
.kpi .value {{ font-size: 1.8rem; font-weight: 700; color: #fff; }}
.kpi .label {{ font-size: 0.82rem; color: var(--text-dim); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}}
th, td {{
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid #2a2a40;
}}
th {{
    background: #16213e;
    color: #8ea4cc;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.78rem;
    letter-spacing: 0.5px;
    position: sticky;
    top: 0;
}}
tr:hover {{ background: #1e1e38; }}
.overall-row {{ background: #16213e !important; font-weight: 700; }}
.price {{ font-family: 'JetBrains Mono', 'Fira Code', monospace; color: var(--success); }}
.chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
.chart-box {{ background: #12122a; border-radius: 10px; padding: 16px; border: 1px solid #2a2a40; }}
.chart-full {{ background: #12122a; border-radius: 10px; padding: 16px; border: 1px solid #2a2a40; margin-bottom: 20px; }}
.star-artist-block {{ margin-bottom: 36px; }}
.table-scroll {{ overflow-x: auto; }}
.nav {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 24px;
    padding: 16px;
    background: var(--card-bg);
    border-radius: 10px;
    border: 1px solid var(--card-border);
}}
.nav a {{
    color: var(--text-dim);
    text-decoration: none;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 0.85rem;
    transition: all 0.2s;
    border: 1px solid transparent;
}}
.nav a:hover {{ background: #16213e; color: #fff; border-color: var(--accent); }}
footer {{
    text-align: center;
    padding: 32px;
    color: var(--text-dim);
    font-size: 0.8rem;
    border-top: 1px solid var(--card-border);
    margin-top: 40px;
}}
@media (max-width: 768px) {{
    .chart-row {{ grid-template-columns: 1fr; }}
    .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
    header h1 {{ font-size: 1.6rem; }}
    .card {{ padding: 18px; }}
}}
</style>
</head>
<body>

<header>
    <h1>Indian Art Market <span>Analytics Report</span></h1>
    <p>Comprehensive market analysis &nbsp;|&nbsp; Data range: {date_min.strftime('%b %Y') if pd.notna(date_min) else 'N/A'} &mdash; {date_max.strftime('%b %Y') if pd.notna(date_max) else 'N/A'} &nbsp;|&nbsp; Generated: {pd.Timestamp.now().strftime('%Y-%m-%d')}</p>
</header>

<div class="container">

<nav class="nav">
    <a href="#exec-summary">Executive Summary</a>
    <a href="#market-overview">Market Overview</a>
    <a href="#star-artists">Star Artists</a>
    <a href="#price-dist">Price Distribution</a>
    <a href="#model-perf">Model Performance</a>
    <a href="#top-sales">Top Sales</a>
    <a href="#source-comp">Source Comparison</a>
    <a href="#core4-index">Core 4 Index</a>
    <a href="#fullstar-index">Full Star Index</a>
    <a href="#cagr-index">CAGR &amp; Price Index</a>
</nav>

<!-- 1. Executive Summary -->
<div class="card" id="exec-summary">
    <h2>1. Executive Summary</h2>
    <div class="kpi-grid">
        <div class="kpi"><div class="value">{fmt_num(total_lots)}</div><div class="label">Total Lots</div></div>
        <div class="kpi"><div class="value">{fmt_num(sold_lots)}</div><div class="label">Sold Lots</div></div>
        <div class="kpi"><div class="value">{fmt_pct(sold_lots/total_lots*100 if total_lots else 0)}</div><div class="label">Sell-Through Rate</div></div>
        <div class="kpi"><div class="value">{fmt_price(total_hammer)}</div><div class="label">Total Hammer Value</div></div>
        <div class="kpi"><div class="value">{len(sources)}</div><div class="label">Data Sources</div></div>
        <div class="kpi"><div class="value">{model_r2:.4f}</div><div class="label">Model R&sup2; (log)</div></div>
        <div class="kpi"><div class="value">{fmt_price(model_median_ae)}</div><div class="label">Median Abs Error</div></div>
        <div class="kpi"><div class="value">{model_mape:.1f}%</div><div class="label">MAPE</div></div>
    </div>
</div>

<!-- 2. Market Overview -->
<div class="card" id="market-overview">
    <h2>2. Market Overview</h2>
    <div class="chart-row">
        <div class="chart-box"><canvas id="chart-yearly-lots"></canvas></div>
        <div class="chart-box"><canvas id="chart-yearly-hammer"></canvas></div>
    </div>
</div>

<!-- 3. Star Artists -->
<div class="card" id="star-artists">
    <h2>3. Star Artists Year-over-Year</h2>
    <p class="subtitle">Median and mean hammer prices, volume, and sell-through rate for 7 key artists</p>
    {star_tables_html}
</div>

<!-- 4. Price Distribution -->
<div class="card" id="price-dist">
    <h2>4. Price Distribution</h2>
    <p class="subtitle">Hammer price histogram on log&#8322;&#8321;&#8320; scale</p>
    <div class="chart-full"><canvas id="chart-hist-all" height="100"></canvas></div>
    <div class="chart-full"><canvas id="chart-hist-stars" height="120"></canvas></div>
</div>

<!-- 5. Model Performance -->
{model_perf_html}

<!-- 6. Top Sales -->
<div class="card" id="top-sales">
    <h2>6. Top 25 Sales</h2>
    <div class="table-scroll">
    <table>
        <thead><tr><th>#</th><th>Artist</th><th>Title</th><th>Price (USD)</th><th>Date</th><th>Medium</th></tr></thead>
        <tbody>{top_sales_rows}</tbody>
    </table>
    </div>
</div>

<!-- 7. Source Comparison -->
<div class="card" id="source-comp">
    <h2>7. Source Comparison</h2>
    <div class="table-scroll">
    <table>
        <thead><tr><th>Source</th><th>Total Lots</th><th>Sold</th><th>STR</th><th>Median Price</th><th>Mean Price</th><th>Total Hammer</th></tr></thead>
        <tbody>{source_rows}</tbody>
    </table>
    </div>
</div>

<!-- 10. CAGR & Price Index -->
<div class="card" id="cagr-index">
    <h2>10. CAGR &amp; Quality-Adjusted Price Index</h2>
    <p class="subtitle">Three ways to measure price growth: <strong>Simple</strong> (raw median, noisy), <strong>Top-Tier</strong> (75th percentile, tracks quality work), <strong>Hedonic</strong> (model quality-adjusted, purest signal)</p>

    <div class="kpi-grid">
        <div class="kpi"><div class="value" style="color:#e94560">{f'{core4_cagr["top_cagr"]:+.1f}%' if core4_cagr else 'N/A'}</div><div class="label">Core 4 Top-Tier CAGR</div></div>
        <div class="kpi"><div class="value" style="color:#f39c12">{f'{fullstar_cagr["top_cagr"]:+.1f}%' if fullstar_cagr else 'N/A'}</div><div class="label">Full Star Top-Tier CAGR</div></div>
        <div class="kpi"><div class="value" style="color:#aaa">{f'{market_cagr["top_cagr"]:+.1f}%' if market_cagr else 'N/A'}</div><div class="label">Full Market Top-Tier CAGR</div></div>
        <div class="kpi"><div class="value" style="color:#00e5ff">{f'{core4_cagr["hedonic_cagr"]:+.1f}%' if core4_cagr else 'N/A'}</div><div class="label">Core 4 Hedonic CAGR</div></div>
        <div class="kpi"><div class="value" style="color:#00e5ff">{f'{fullstar_cagr["hedonic_cagr"]:+.1f}%' if fullstar_cagr else 'N/A'}</div><div class="label">Full Star Hedonic CAGR</div></div>
        <div class="kpi"><div class="value" style="color:#00e5ff">{f'{market_cagr["hedonic_cagr"]:+.1f}%' if market_cagr else 'N/A'}</div><div class="label">Full Market Hedonic CAGR</div></div>
    </div>

    <div class="chart-row">
        <div class="chart-box"><canvas id="chart-cagr-hedonic"></canvas></div>
        <div class="chart-box"><canvas id="chart-cagr-simple"></canvas></div>
    </div>

    <h3 style="margin-top:24px;">CAGR by Artist</h3>
    <div class="table-scroll">
    <table>
        <thead><tr><th>Artist / Index</th><th>Period</th><th># Lots</th><th>Simple CAGR</th><th>Top-Tier CAGR</th><th>Hedonic CAGR</th><th>R&sup2;</th></tr></thead>
        <tbody>
        {"".join(f'<tr style="font-weight:bold;color:#e94560"><td>Core 4 Index</td><td>{core4_cagr["years"][0]}&ndash;{core4_cagr["years"][-1]}</td><td>{core4_cagr["n_lots"]}</td><td>{core4_cagr["simple_cagr"]:+.1f}%</td><td>{core4_cagr["top_cagr"]:+.1f}%</td><td>{core4_cagr["hedonic_cagr"]:+.1f}%</td><td>{core4_cagr["r_squared"]:.3f}</td></tr>' if core4_cagr else '')}
        {"".join(f'<tr style="font-weight:bold;color:#f39c12"><td>Full Star Index</td><td>{fullstar_cagr["years"][0]}&ndash;{fullstar_cagr["years"][-1]}</td><td>{fullstar_cagr["n_lots"]}</td><td>{fullstar_cagr["simple_cagr"]:+.1f}%</td><td>{fullstar_cagr["top_cagr"]:+.1f}%</td><td>{fullstar_cagr["hedonic_cagr"]:+.1f}%</td><td>{fullstar_cagr["r_squared"]:.3f}</td></tr>' if fullstar_cagr else '')}
        {"".join(f'<tr style="color:#888"><td>Full Market</td><td>{market_cagr["years"][0]}&ndash;{market_cagr["years"][-1]}</td><td>{market_cagr["n_lots"]}</td><td>{market_cagr["simple_cagr"]:+.1f}%</td><td>{market_cagr["top_cagr"]:+.1f}%</td><td>{market_cagr["hedonic_cagr"]:+.1f}%</td><td>{market_cagr["r_squared"]:.3f}</td></tr>' if market_cagr else '')}
        {"".join(f'<tr><td style="color:{STAR_COLORS.get(name, "#ccc")}">{name}</td><td>{d["years"][0]}&ndash;{d["years"][-1]}</td><td>{d["n_lots"]}</td><td>{d["simple_cagr"]:+.1f}%</td><td>{d["top_cagr"]:+.1f}%</td><td>{d["hedonic_cagr"]:+.1f}%</td><td>{d["r_squared"]:.3f}</td></tr>' for name, d in sorted(artist_cagrs.items(), key=lambda x: -x[1]["top_cagr"]))}
        </tbody>
    </table>
    </div>
</div>

<!-- 8. Core 4 Index -->
<div class="card" id="core4-index">
    <h2>8. Core 4 Index</h2>
    <p class="subtitle">Souza &bull; Raza &bull; Tyeb Mehta &bull; Husain &mdash; Year-over-year performance vs rest of market</p>
    <div class="chart-row">
        <div class="chart-box"><canvas id="chart-core4-median"></canvas></div>
        <div class="chart-box"><canvas id="chart-core4-value"></canvas></div>
    </div>
    <div class="table-scroll">
    <table>
        <thead><tr><th>Year</th><th># Sold</th><th>Median</th><th>Fair Value (Med)</th><th>Mean</th><th>Total Value</th><th>Rest Median</th></tr></thead>
        <tbody>{"".join(f'<tr><td>{core4_idx["years"][i]}</td><td>{core4_idx["count"][i]}</td><td>{fmt_price(core4_idx["median"][i])}</td><td>{fmt_price(core4_idx["fair_median"][i])}</td><td>{fmt_price(core4_idx["mean"][i])}</td><td>{fmt_price(core4_idx["total"][i])}</td><td>{fmt_price(core4_rest["median"][i]) if i < len(core4_rest["median"]) else "N/A"}</td></tr>' for i in range(len(core4_idx["years"])))}</tbody>
    </table>
    </div>
    <h3 style="margin-top:24px;">Constituent Artists</h3>
    <div class="table-scroll">
    <table>
        <thead><tr><th>Artist</th><th>Total Sales</th><th>Sales (12m)</th><th>Median (All)</th><th>Median (12m)</th><th>Mean (All)</th><th>Max</th></tr></thead>
        <tbody>{"".join(f'<tr><td>{a["name"]}</td><td>{a["n"]}</td><td>{a["n_12m"]}</td><td>{fmt_price(a["median_all"])}</td><td>{fmt_price(a["median_12m"])}</td><td>{fmt_price(a["mean_all"])}</td><td>{fmt_price(a["max_all"])}</td></tr>' for a in core4_artists)}</tbody>
    </table>
    </div>
</div>

<!-- 9. Full Star Index -->
<div class="card" id="fullstar-index">
    <h2>9. Full Star Index</h2>
    <p class="subtitle">All artists with a sale &ge; $500K in the last 10 years &mdash; {len(fullstar_artists)} artists</p>
    <div class="chart-row">
        <div class="chart-box"><canvas id="chart-fullstar-median"></canvas></div>
        <div class="chart-box"><canvas id="chart-fullstar-value"></canvas></div>
    </div>
    <div class="table-scroll">
    <table>
        <thead><tr><th>Year</th><th># Sold</th><th>Median</th><th>Fair Value (Med)</th><th>Mean</th><th>Total Value</th><th>Rest Median</th></tr></thead>
        <tbody>{"".join(f'<tr><td>{fullstar_idx["years"][i]}</td><td>{fullstar_idx["count"][i]}</td><td>{fmt_price(fullstar_idx["median"][i])}</td><td>{fmt_price(fullstar_idx["fair_median"][i])}</td><td>{fmt_price(fullstar_idx["mean"][i])}</td><td>{fmt_price(fullstar_idx["total"][i])}</td><td>{fmt_price(fullstar_rest["median"][i]) if i < len(fullstar_rest["median"]) else "N/A"}</td></tr>' for i in range(len(fullstar_idx["years"])))}</tbody>
    </table>
    </div>
    <h3 style="margin-top:24px;">Constituent Artists</h3>
    <div class="table-scroll">
    <table>
        <thead><tr><th>Artist</th><th>Total Sales</th><th>Sales (12m)</th><th>Median (All)</th><th>Median (12m)</th><th>Mean (All)</th><th>Max</th></tr></thead>
        <tbody>{"".join(f'<tr><td>{a["name"]}</td><td>{a["n"]}</td><td>{a["n_12m"]}</td><td>{fmt_price(a["median_all"])}</td><td>{fmt_price(a["median_12m"])}</td><td>{fmt_price(a["mean_all"])}</td><td>{fmt_price(a["max_all"])}</td></tr>' for a in sorted(fullstar_artists, key=lambda x: -x["max_all"]))}</tbody>
    </table>
    </div>
</div>

</div><!-- /container -->

<footer>
    Indian Art Market Analytics &nbsp;|&nbsp; Data: master.csv ({fmt_num(total_lots)} lots) &amp; ml_ready.csv ({fmt_num(len(ml))} lots)
    &nbsp;|&nbsp; Models: {manifest['metrics']['n_catboost']} CatBoost + {manifest['metrics']['n_xgboost']} XGBoost
</footer>

<script>
Chart.defaults.color = '#aaa';
Chart.defaults.borderColor = '#333';

// 2. Market Overview Charts
new Chart(document.getElementById('chart-yearly-lots'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(market_years)},
        datasets: [{{
            label: 'Lots Sold',
            data: {json.dumps(market_lots)},
            backgroundColor: '#e9456088',
            borderColor: '#e94560',
            borderWidth: 1,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{
            title: {{ display: true, text: 'Auction Volume (Lots Sold) by Year', color: '#e0e0e0', font: {{ size: 14 }} }},
            legend: {{ labels: {{ color: '#ccc' }} }},
            tooltip: {{ callbacks: {{ label: ctx => ctx.parsed.y.toLocaleString() + ' lots' }} }}
        }},
        scales: {{
            x: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }} }},
            y: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }} }}
        }}
    }}
}});

new Chart(document.getElementById('chart-yearly-hammer'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(market_years)},
        datasets: [{{
            label: 'Total Hammer (USD)',
            data: {json.dumps([to_json_val(v) for v in market_hammer])},
            backgroundColor: '#0f346088',
            borderColor: '#0f3460',
            borderWidth: 1,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{
            title: {{ display: true, text: 'Total Hammer Value (USD) by Year', color: '#e0e0e0', font: {{ size: 14 }} }},
            legend: {{ labels: {{ color: '#ccc' }} }},
            tooltip: {{ callbacks: {{ label: ctx => '$' + ctx.parsed.y.toLocaleString() }} }}
        }},
        scales: {{
            x: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }} }},
            y: {{ ticks: {{ color: '#aaa', callback: v => '$' + (v/1e6).toFixed(1) + 'M' }}, grid: {{ color: '#222' }} }}
        }}
    }}
}});

// 3. Star artist charts
{star_charts_js}

// 4. Price Distribution
new Chart(document.getElementById('chart-hist-all'), {{
    type: 'bar',
    data: {{
        labels: {hist_labels_json},
        datasets: [{{
            label: 'All Artists',
            data: {hist_all_json},
            backgroundColor: '#e9456066',
            borderColor: '#e94560',
            borderWidth: 1,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{
            title: {{ display: true, text: 'Hammer Price Distribution (All Artists, Log Scale Bins)', color: '#e0e0e0', font: {{ size: 14 }} }},
            legend: {{ labels: {{ color: '#ccc' }} }}
        }},
        scales: {{
            x: {{ ticks: {{ color: '#aaa', maxRotation: 60 }}, grid: {{ color: '#222' }} }},
            y: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }} }}
        }}
    }}
}});

new Chart(document.getElementById('chart-hist-stars'), {{
    type: 'bar',
    data: {{
        labels: {hist_labels_json},
        datasets: [{star_hist_datasets}]
    }},
    options: {{
        responsive: true,
        plugins: {{
            title: {{ display: true, text: 'Hammer Price Distribution by Star Artist (Log Scale Bins)', color: '#e0e0e0', font: {{ size: 14 }} }},
            legend: {{ labels: {{ color: '#ccc' }} }}
        }},
        scales: {{
            x: {{ ticks: {{ color: '#aaa', maxRotation: 60 }}, grid: {{ color: '#222' }}, stacked: false }},
            y: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }}, stacked: false }}
        }}
    }}
}});

// 10. CAGR & Price Index Charts
new Chart(document.getElementById('chart-cagr-hedonic'), {{
    type: 'line',
    data: {{
        labels: {json.dumps(core4_cagr["years"] if core4_cagr else [])},
        datasets: [
            {{ label: 'Core 4 (Top-Tier P75)', data: {json.dumps(core4_cagr["index_top"] if core4_cagr else [])}, borderColor: '#e94560', backgroundColor: '#e9456022', fill: true, tension: 0.3, borderWidth: 2.5 }},
            {{ label: 'Full Star (Top-Tier P75)', data: {json.dumps(fullstar_cagr["index_top"] if fullstar_cagr else [])}, borderColor: '#f39c12', fill: false, tension: 0.3, borderWidth: 2.5 }},
            {{ label: 'Full Market (Top-Tier P75)', data: {json.dumps(market_cagr["index_top"] if market_cagr else [])}, borderColor: '#888', borderDash: [5,5], fill: false, tension: 0.3, borderWidth: 1.5 }},
            {','.join(f'{{ label: "{name} (P75)", data: {json.dumps(d["index_top"])}, borderColor: "{STAR_COLORS.get(name, "#ccc")}", fill: false, tension: 0.3, borderWidth: 1.5, hidden: true }}' for name, d in artist_cagrs.items())}
        ]
    }},
    options: {{
        responsive: true,
        plugins: {{
            title: {{ display: true, text: 'Top-Tier Price Index (75th Percentile) — Base Year = 100', color: '#e0e0e0', font: {{ size: 14 }} }},
            subtitle: {{ display: true, text: 'Tracks quality work — not dragged down by small/minor pieces', color: '#888', font: {{ size: 11 }} }},
            tooltip: {{ callbacks: {{ label: function(ctx) {{ return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1); }} }} }},
            legend: {{ labels: {{ color: '#ccc', usePointStyle: true }} }}
        }},
        scales: {{
            x: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }} }},
            y: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#333' }}, title: {{ display: true, text: 'Index (base=100)', color: '#aaa' }} }}
        }}
    }}
}});

new Chart(document.getElementById('chart-cagr-simple'), {{
    type: 'line',
    data: {{
        labels: {json.dumps(core4_cagr["years"] if core4_cagr else [])},
        datasets: [
            {{ label: 'Core 4 (Hedonic)', data: {json.dumps(core4_cagr["index_hedonic"] if core4_cagr else [])}, borderColor: '#e94560', backgroundColor: '#e9456022', fill: true, tension: 0.3, borderWidth: 2.5 }},
            {{ label: 'Full Star (Hedonic)', data: {json.dumps(fullstar_cagr["index_hedonic"] if fullstar_cagr else [])}, borderColor: '#f39c12', fill: false, tension: 0.3, borderWidth: 2.5 }},
            {{ label: 'Full Market (Hedonic)', data: {json.dumps(market_cagr["index_hedonic"] if market_cagr else [])}, borderColor: '#888', borderDash: [5,5], fill: false, tension: 0.3, borderWidth: 1.5 }},
            {','.join(f'{{ label: "{name} (Hedonic)", data: {json.dumps(d["index_hedonic"])}, borderColor: "{STAR_COLORS.get(name, "#ccc")}", fill: false, tension: 0.3, borderWidth: 1.5, hidden: true }}' for name, d in artist_cagrs.items())}
        ]
    }},
    options: {{
        responsive: true,
        plugins: {{
            title: {{ display: true, text: 'Hedonic (Quality-Adjusted) Price Index — Base Year = 100', color: '#e0e0e0', font: {{ size: 14 }} }},
            subtitle: {{ display: true, text: 'Model normalizes for size, medium, provenance → isolates pure price appreciation', color: '#888', font: {{ size: 11 }} }},
            tooltip: {{ callbacks: {{ label: function(ctx) {{ return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1); }} }} }},
            legend: {{ labels: {{ color: '#ccc', usePointStyle: true }} }}
        }},
        scales: {{
            x: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }} }},
            y: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#333' }}, title: {{ display: true, text: 'Index (base=100)', color: '#aaa' }} }}
        }}
    }}
}});

// 8. Core 4 Index Charts
new Chart(document.getElementById('chart-core4-median'), {{
    type: 'line',
    data: {{
        labels: {json.dumps(core4_idx["years"])},
        datasets: [
            {{ label: 'Core 4 Median (Actual)', data: {json.dumps(core4_idx["median"])}, borderColor: '#e94560', backgroundColor: '#e9456033', fill: true, tension: 0.3, borderWidth: 2 }},
            {{ label: 'Core 4 Fair Value (Model)', data: {json.dumps(core4_idx["fair_median"])}, borderColor: '#00e5ff', borderDash: [8,4], fill: false, tension: 0.3, borderWidth: 2, pointStyle: 'rectRot', pointRadius: 5 }},
            {{ label: 'Rest of Market Median', data: {json.dumps(core4_rest["median"])}, borderColor: '#666', borderDash: [3,3], fill: false, tension: 0.3, borderWidth: 1 }}
        ]
    }},
    options: {{
        responsive: true,
        plugins: {{
            title: {{ display: true, text: 'Core 4 — Actual Median vs Model Fair Value', color: '#e0e0e0', font: {{ size: 14 }} }},
            tooltip: {{ callbacks: {{ label: function(ctx) {{ return ctx.dataset.label + ': $' + ctx.parsed.y.toLocaleString(); }} }} }}
        }},
        scales: {{
            x: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }} }},
            y: {{ ticks: {{ color: '#aaa', callback: function(v) {{ return '$' + (v/1000).toFixed(0) + 'K'; }} }}, grid: {{ color: '#222' }} }}
        }}
    }}
}});

new Chart(document.getElementById('chart-core4-value'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(core4_idx["years"])},
        datasets: [
            {{ label: 'Core 4 Total Value', data: {json.dumps(core4_idx["total"])}, backgroundColor: '#e9456088', borderColor: '#e94560', borderWidth: 1 }},
            {{ label: '# Lots Sold', data: {json.dumps(core4_idx["count"])}, type: 'line', borderColor: '#3498db', yAxisID: 'y1', tension: 0.3 }}
        ]
    }},
    options: {{
        responsive: true,
        plugins: {{
            title: {{ display: true, text: 'Core 4 — Total Hammer Value & Volume', color: '#e0e0e0', font: {{ size: 14 }} }},
            tooltip: {{ callbacks: {{ label: function(ctx) {{ if(ctx.dataset.type==='line') return ctx.dataset.label + ': ' + ctx.parsed.y; return ctx.dataset.label + ': $' + ctx.parsed.y.toLocaleString(); }} }} }}
        }},
        scales: {{
            x: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }} }},
            y: {{ ticks: {{ color: '#aaa', callback: function(v) {{ return '$' + (v/1e6).toFixed(1) + 'M'; }} }}, grid: {{ color: '#222' }}, position: 'left' }},
            y1: {{ ticks: {{ color: '#3498db' }}, grid: {{ display: false }}, position: 'right' }}
        }}
    }}
}});

// 9. Full Star Index Charts
new Chart(document.getElementById('chart-fullstar-median'), {{
    type: 'line',
    data: {{
        labels: {json.dumps(fullstar_idx["years"])},
        datasets: [
            {{ label: 'Star Index Median (Actual)', data: {json.dumps(fullstar_idx["median"])}, borderColor: '#f39c12', backgroundColor: '#f39c1233', fill: true, tension: 0.3, borderWidth: 2 }},
            {{ label: 'Star Index Fair Value (Model)', data: {json.dumps(fullstar_idx["fair_median"])}, borderColor: '#00e5ff', borderDash: [8,4], fill: false, tension: 0.3, borderWidth: 2, pointStyle: 'rectRot', pointRadius: 5 }},
            {{ label: 'Rest of Market Median', data: {json.dumps(fullstar_rest["median"])}, borderColor: '#666', borderDash: [3,3], fill: false, tension: 0.3, borderWidth: 1 }}
        ]
    }},
    options: {{
        responsive: true,
        plugins: {{
            title: {{ display: true, text: 'Full Star Index — Actual Median vs Model Fair Value', color: '#e0e0e0', font: {{ size: 14 }} }},
            tooltip: {{ callbacks: {{ label: function(ctx) {{ return ctx.dataset.label + ': $' + ctx.parsed.y.toLocaleString(); }} }} }}
        }},
        scales: {{
            x: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }} }},
            y: {{ ticks: {{ color: '#aaa', callback: function(v) {{ return '$' + (v/1000).toFixed(0) + 'K'; }} }}, grid: {{ color: '#222' }} }}
        }}
    }}
}});

new Chart(document.getElementById('chart-fullstar-value'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(fullstar_idx["years"])},
        datasets: [
            {{ label: 'Star Index Total Value', data: {json.dumps(fullstar_idx["total"])}, backgroundColor: '#f39c1288', borderColor: '#f39c12', borderWidth: 1 }},
            {{ label: '# Lots Sold', data: {json.dumps(fullstar_idx["count"])}, type: 'line', borderColor: '#3498db', yAxisID: 'y1', tension: 0.3 }}
        ]
    }},
    options: {{
        responsive: true,
        plugins: {{
            title: {{ display: true, text: 'Full Star Index — Total Hammer Value & Volume', color: '#e0e0e0', font: {{ size: 14 }} }},
            tooltip: {{ callbacks: {{ label: function(ctx) {{ if(ctx.dataset.type==='line') return ctx.dataset.label + ': ' + ctx.parsed.y; return ctx.dataset.label + ': $' + ctx.parsed.y.toLocaleString(); }} }} }}
        }},
        scales: {{
            x: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }} }},
            y: {{ ticks: {{ color: '#aaa', callback: function(v) {{ return '$' + (v/1e6).toFixed(1) + 'M'; }} }}, grid: {{ color: '#222' }}, position: 'left' }},
            y1: {{ ticks: {{ color: '#3498db' }}, grid: {{ display: false }}, position: 'right' }}
        }}
    }}
}});

</script>
</body>
</html>"""

os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
with open(OUT_HTML, "w") as f:
    f.write(html)

print(f"\nReport written to {OUT_HTML}")
print(f"File size: {os.path.getsize(OUT_HTML) / 1024:.1f} KB")
