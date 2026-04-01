"""
Comprehensive backtest of the art price prediction model against all auctions.
Compares model predictions vs actual hammer prices and auction house estimates.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "saved"
DATA_DIR = BASE_DIR / "data" / "processed"


def load_models_and_manifest():
    """Load manifest, CatBoost, and XGBoost models."""
    with open(MODELS_DIR / "model_manifest.json") as f:
        manifest = json.load(f)

    cb_models = []
    for fname in manifest["catboost_models"]:
        m = CatBoostRegressor()
        m.load_model(str(MODELS_DIR / fname))
        cb_models.append(m)

    xgb_models = []
    for fname in manifest["xgboost_models"]:
        m = XGBRegressor()
        m.load_model(str(MODELS_DIR / fname))
        xgb_models.append(m)

    return manifest, cb_models, xgb_models


def run_predictions(df, manifest, cb_models, xgb_models):
    """Run ensemble predictions on all rows. Returns array of log predictions."""
    feature_cols = manifest["feature_cols"]
    cat_cols = manifest["cat_cols"]
    cat_indices = manifest["cat_indices"]

    X = df[feature_cols].copy()

    # CatBoost predictions
    X_cb = X.copy()
    for col in cat_cols:
        X_cb[col] = X_cb[col].fillna("unknown").astype(str)
    pool = Pool(X_cb, cat_features=cat_indices)

    cb_preds = np.column_stack([m.predict(pool) for m in cb_models])

    # XGBoost predictions
    X_xgb = X.copy()
    for col in cat_cols:
        X_xgb[col] = X_xgb[col].astype("category")
    xgb_preds = np.column_stack([m.predict(X_xgb) for m in xgb_models])

    # Ensemble: average all
    all_preds = np.concatenate([cb_preds, xgb_preds], axis=1)
    mean_log_pred = all_preds.mean(axis=1)

    return mean_log_pred


def compute_metrics(actual_usd, predicted_usd, estimate_mid_usd=None):
    """Compute error metrics. All inputs in USD (not log)."""
    # Percentage error
    pct_err = np.abs(predicted_usd - actual_usd) / actual_usd * 100

    result = {
        "n": int(len(actual_usd)),
        "median_pct_err": round(float(np.median(pct_err)), 2),
        "mean_pct_err": round(float(np.mean(pct_err)), 2),
        "within_10": round(float((pct_err <= 10).mean() * 100), 1),
        "within_25": round(float((pct_err <= 25).mean() * 100), 1),
        "within_50": round(float((pct_err <= 50).mean() * 100), 1),
        "median_abs_err_usd": round(float(np.median(np.abs(predicted_usd - actual_usd))), 0),
        "r2_log": round(float(1 - np.sum((np.log1p(actual_usd) - np.log1p(predicted_usd))**2) /
                                np.sum((np.log1p(actual_usd) - np.mean(np.log1p(actual_usd)))**2)), 4),
    }

    if estimate_mid_usd is not None:
        # Filter out NaN/zero/inf estimates
        valid = np.isfinite(estimate_mid_usd) & (estimate_mid_usd > 0) & np.isfinite(actual_usd) & (actual_usd > 0)
        if valid.sum() > 0:
            est_act = actual_usd[valid]
            est_pred = estimate_mid_usd[valid]
            est_pct_err = np.abs(est_pred - est_act) / est_act * 100
            result["est_n"] = int(valid.sum())
            result["est_median_pct_err"] = round(float(np.median(est_pct_err)), 2)
            result["est_mean_pct_err"] = round(float(np.mean(est_pct_err)), 2)
            result["est_within_10"] = round(float((est_pct_err <= 10).mean() * 100), 1)
            result["est_within_25"] = round(float((est_pct_err <= 25).mean() * 100), 1)
            result["est_within_50"] = round(float((est_pct_err <= 50).mean() * 100), 1)

    return result


def main():
    print("=" * 80)
    print("ART PRICE PREDICTION MODEL - COMPREHENSIVE BACKTEST")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    ml_df = pd.read_csv(DATA_DIR / "ml_ready.csv")
    master_df = pd.read_csv(DATA_DIR / "master.csv")
    ml_df["auction_date"] = pd.to_datetime(
        ml_df["auction_date"], errors="coerce", format="mixed", utc=True
    ).dt.tz_localize(None)

    master_df["auction_date"] = pd.to_datetime(
        master_df["auction_date"], errors="coerce", format="mixed", utc=True
    ).dt.tz_localize(None)

    print(f"  ml_ready rows: {len(ml_df)}")
    print(f"  master rows:   {len(master_df)}")

    # Merge source from master into ml_df via lot_id
    source_map = master_df[["lot_id", "source"]].drop_duplicates("lot_id")
    source_map["lot_id"] = source_map["lot_id"].astype(str)
    ml_df["lot_id"] = ml_df["lot_id"].astype(str)
    ml_df = ml_df.merge(source_map, on="lot_id", how="left")
    print(f"  Source distribution: {ml_df['source'].value_counts().to_dict()}")

    # Load models
    print("\nLoading models...")
    manifest, cb_models, xgb_models = load_models_and_manifest()
    print(f"  CatBoost models: {len(cb_models)}")
    print(f"  XGBoost models:  {len(xgb_models)}")

    # Run predictions
    print("\nRunning predictions on all {0} lots...".format(len(ml_df)))
    log_preds = run_predictions(ml_df, manifest, cb_models, xgb_models)

    # Convert from log space to USD
    predicted_usd = np.expm1(log_preds)
    actual_usd = np.expm1(ml_df["log_hammer_price"].values)
    estimate_mid_usd = ml_df["estimate_midpoint"].values

    # Chronological split: last 10% is test
    n = len(ml_df)
    split_idx = int(n * 0.9)
    train_mask = np.arange(n) < split_idx
    test_mask = ~train_mask

    # --- Overall metrics ---
    overall = compute_metrics(actual_usd, predicted_usd, estimate_mid_usd)
    print(f"\n{'=' * 80}")
    print("OVERALL RESULTS (all {0} lots)".format(n))
    print(f"{'=' * 80}")
    print(f"  Model median % error:    {overall['median_pct_err']:.1f}%")
    print(f"  Estimate median % error: {overall.get('est_median_pct_err', 'N/A')}")
    print(f"  Model within 25%:        {overall['within_25']:.1f}%")
    print(f"  Estimate within 25%:     {overall.get('est_within_25', 'N/A')}")
    print(f"  Model within 50%:        {overall['within_50']:.1f}%")
    print(f"  Estimate within 50%:     {overall.get('est_within_50', 'N/A')}")
    print(f"  R-squared (log):         {overall['r2_log']:.4f}")

    # --- Train vs Test ---
    train_metrics = compute_metrics(actual_usd[train_mask], predicted_usd[train_mask],
                                     estimate_mid_usd[train_mask])
    test_metrics = compute_metrics(actual_usd[test_mask], predicted_usd[test_mask],
                                    estimate_mid_usd[test_mask])

    print(f"\n{'=' * 80}")
    print("TRAIN vs TEST SPLIT")
    print(f"{'=' * 80}")
    header = f"{'Set':<12} {'N':>6} {'Med%Err':>8} {'W/in25':>8} {'W/in50':>8} {'R2(log)':>8} | {'Est Med%':>8} {'EstW25':>8} {'EstW50':>8}"
    print(header)
    print("-" * len(header))
    for label, m in [("Train(90%)", train_metrics), ("Test(10%)", test_metrics)]:
        est_med = f"{m['est_median_pct_err']:>7.1f}%" if 'est_median_pct_err' in m else "    N/A "
        est_25 = f"{m['est_within_25']:>7.1f}%" if 'est_within_25' in m else "    N/A "
        est_50 = f"{m['est_within_50']:>7.1f}%" if 'est_within_50' in m else "    N/A "
        print(f"{label:<12} {m['n']:>6} {m['median_pct_err']:>7.1f}% {m['within_25']:>7.1f}% {m['within_50']:>7.1f}% {m['r2_log']:>8.4f} | {est_med} {est_25} {est_50}")

    # --- By Source ---
    print(f"\n{'=' * 80}")
    print("BY AUCTION SOURCE")
    print(f"{'=' * 80}")
    by_source = []
    print(header)
    print("-" * len(header))
    for src in sorted(ml_df["source"].dropna().unique()):
        mask = (ml_df["source"] == src).values
        if mask.sum() < 5:
            continue
        m = compute_metrics(actual_usd[mask], predicted_usd[mask], estimate_mid_usd[mask])
        m["source"] = src
        by_source.append(m)
        est_med = f"{m['est_median_pct_err']:>7.1f}%" if 'est_median_pct_err' in m else "    N/A "
        est_25 = f"{m['est_within_25']:>7.1f}%" if 'est_within_25' in m else "    N/A "
        est_50 = f"{m['est_within_50']:>7.1f}%" if 'est_within_50' in m else "    N/A "
        print(f"{src:<12} {m['n']:>6} {m['median_pct_err']:>7.1f}% {m['within_25']:>7.1f}% {m['within_50']:>7.1f}% {m['r2_log']:>8.4f} | {est_med} {est_25} {est_50}")

    # --- By Year ---
    print(f"\n{'=' * 80}")
    print("BY AUCTION YEAR")
    print(f"{'=' * 80}")
    by_year = []
    years = sorted(ml_df["auction_year"].dropna().unique())
    print(header)
    print("-" * len(header))
    for yr in years:
        mask = (ml_df["auction_year"] == yr).values
        if mask.sum() < 3:
            continue
        m = compute_metrics(actual_usd[mask], predicted_usd[mask], estimate_mid_usd[mask])
        m["year"] = int(yr)
        by_year.append(m)
        label = str(int(yr))
        est_med = f"{m['est_median_pct_err']:>7.1f}%" if 'est_median_pct_err' in m else "    N/A "
        est_25 = f"{m['est_within_25']:>7.1f}%" if 'est_within_25' in m else "    N/A "
        est_50 = f"{m['est_within_50']:>7.1f}%" if 'est_within_50' in m else "    N/A "
        print(f"{label:<12} {m['n']:>6} {m['median_pct_err']:>7.1f}% {m['within_25']:>7.1f}% {m['within_50']:>7.1f}% {m['r2_log']:>8.4f} | {est_med} {est_25} {est_50}")

    # --- By Star Artist ---
    STAR_ARTISTS = [
        "Francis Newton Souza",
        "Maqbool Fida Husain",
        "Sayed Haider Raza",
        "Tyeb Mehta",
        "Vasudeo S. Gaitonde",
        "Akbar Padamsee",
        "Ram Kumar",
    ]
    print(f"\n{'=' * 80}")
    print("BY STAR ARTIST")
    print(f"{'=' * 80}")
    by_artist = []
    header2 = f"{'Artist':<24} {'N':>5} {'Med%Err':>8} {'W/in25':>8} {'W/in50':>8} {'R2(log)':>8} | {'Est Med%':>8} {'EstW25':>8}"
    print(header2)
    print("-" * len(header2))
    for artist in STAR_ARTISTS:
        mask = (ml_df["artist_name"] == artist).values
        if mask.sum() < 3:
            continue
        m = compute_metrics(actual_usd[mask], predicted_usd[mask], estimate_mid_usd[mask])
        m["artist"] = artist
        by_artist.append(m)
        short = artist[:23]
        est_med = f"{m['est_median_pct_err']:>7.1f}%" if 'est_median_pct_err' in m else "    N/A "
        est_25 = f"{m['est_within_25']:>7.1f}%" if 'est_within_25' in m else "    N/A "
        print(f"{short:<24} {m['n']:>5} {m['median_pct_err']:>7.1f}% {m['within_25']:>7.1f}% {m['within_50']:>7.1f}% {m['r2_log']:>8.4f} | {est_med} {est_25}")

    # --- Build output JSON ---
    output = {
        "overall": overall,
        "train_set": train_metrics,
        "test_set": test_metrics,
        "by_source": by_source,
        "by_year": by_year,
        "by_artist": by_artist,
        "notes": {
            "target": "log_hammer_price (np.log1p of USD)",
            "prediction_method": "Ensemble of 5 CatBoost + 2 XGBoost, averaged in log space",
            "train_test_split": "Chronological: first 90% train, last 10% test",
            "in_sample_warning": "Overall/by_year/by_source/by_artist include training data. Train set performance is in-sample (optimistic). Test set is true out-of-sample.",
            "estimate_baseline": "Auction house estimate midpoint = (estimate_low + estimate_high) / 2",
            "pct_err_definition": "abs(predicted - actual) / actual * 100",
            "backtest_date": "2026-03-26",
            "total_lots": int(n),
        },
    }

    out_path = DATA_DIR / "backtest_full.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'=' * 80}")
    print("KEY TAKEAWAYS")
    print(f"{'=' * 80}")
    print(f"  Total lots evaluated:          {n}")
    print(f"  Out-of-sample (test) lots:     {test_metrics['n']}")
    print(f"")
    print(f"  Model test median % error:     {test_metrics['median_pct_err']:.1f}%")
    if 'est_median_pct_err' in test_metrics:
        print(f"  Estimate test median % error:  {test_metrics['est_median_pct_err']:.1f}%")
    print(f"  Model test R2 (log):           {test_metrics['r2_log']:.4f}")
    print(f"")
    if 'est_median_pct_err' in test_metrics:
        delta = test_metrics['est_median_pct_err'] - test_metrics['median_pct_err']
        if delta > 0:
            print(f"  Model beats estimates by {delta:.1f} percentage points (median error) on test set")
        else:
            print(f"  Estimates beat model by {-delta:.1f} percentage points (median error) on test set")


if __name__ == "__main__":
    main()
