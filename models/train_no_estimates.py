#!/usr/bin/env python3
"""Train ensemble model WITHOUT auction house estimates.

This isolates the model's ability to predict price purely from
artwork characteristics (artist, medium, size, provenance, etc.)
without the auction house's price guidance.

Output: models/saved/no_est_model_*.cbm, no_est_xgb_model_*.json, no_est_manifest.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.helpers import setup_logger

logger = setup_logger(__name__, "train_no_estimates.log")

BASE = Path(__file__).resolve().parent.parent
ML_READY_FILE = BASE / "data" / "processed" / "ml_ready.csv"
MODELS_DIR = BASE / "models" / "saved"

# Estimate-related features to EXCLUDE
ESTIMATE_FEATURES = [
    "estimate_midpoint", "estimate_spread", "log_estimate_mid",
    "artist_avg_estimate_accuracy",
]


def main():
    logger.info("=" * 60)
    logger.info("Training NO-ESTIMATES Ensemble")
    logger.info("=" * 60)

    df = pd.read_csv(ML_READY_FILE)
    df["auction_date"] = pd.to_datetime(df["auction_date"], errors="coerce", format="mixed", utc=True)
    df["auction_date"] = df["auction_date"].dt.tz_localize(None)
    df = df.sort_values("auction_date").reset_index(drop=True)

    logger.info(f"Loaded {len(df)} rows")

    target_col = "log_hammer_price"
    meta_cols = ["auction_date", "hammer_price_usd", "artist_name", "title", "lot_id"]
    cat_cols = ["medium_category", "artist_name_clean", "subject", "palette", "style"]

    # All features MINUS estimate features
    feature_cols = [c for c in df.columns if c not in [target_col] + meta_cols + ESTIMATE_FEATURES]
    available_cat = [c for c in cat_cols if c in feature_cols]

    logger.info(f"Features: {len(feature_cols)} (excluded {len(ESTIMATE_FEATURES)} estimate features)")
    logger.info(f"Excluded: {ESTIMATE_FEATURES}")

    # Chronological split
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Fill categoricals
    for col in available_cat:
        for frame in [X_train, X_val, X_test]:
            frame[col] = frame[col].fillna("unknown").astype(str)

    cat_indices = [i for i, c in enumerate(feature_cols) if c in available_cat]
    w_train = np.linspace(1.0, 1.5, len(X_train))

    # Train CatBoost models
    cb_models = []
    for seed in range(42, 47):
        logger.info(f"\n--- CatBoost (no-est) seed={seed} ---")
        model = CatBoostRegressor(
            iterations=2000, learning_rate=0.05, depth=6,
            l2_leaf_reg=3, random_seed=seed,
            early_stopping_rounds=100, verbose=100,
        )
        pool_train = Pool(X_train, y_train, cat_features=cat_indices, weight=w_train)
        pool_val = Pool(X_val, y_val, cat_features=cat_indices)
        model.fit(pool_train, eval_set=pool_val, use_best_model=True)

        path = MODELS_DIR / f"no_est_model_{seed - 41}.cbm"
        model.save_model(str(path))
        cb_models.append(model)
        logger.info(f"  Saved: {path}")

    # Train XGBoost models (use label encoding for categoricals)
    xgb_models = []
    # Build label encoders from train+val combined
    label_maps = {}
    for col in available_cat:
        all_vals = pd.concat([X_train[col], X_val[col], X_test[col]]).fillna("unknown").unique()
        label_maps[col] = {v: i for i, v in enumerate(all_vals)}

    for seed in [42, 43]:
        logger.info(f"\n--- XGBoost (no-est) seed={seed} ---")
        X_tr_xgb = X_train.copy()
        X_va_xgb = X_val.copy()
        for col in available_cat:
            X_tr_xgb[col] = X_tr_xgb[col].map(label_maps[col]).fillna(-1).astype(int)
            X_va_xgb[col] = X_va_xgb[col].map(label_maps[col]).fillna(-1).astype(int)

        dtrain = xgb.DMatrix(X_tr_xgb, label=y_train, weight=w_train)
        dval = xgb.DMatrix(X_va_xgb, label=y_val)

        params = {
            "objective": "reg:squarederror", "eval_metric": "rmse",
            "max_depth": 6, "learning_rate": 0.05, "seed": seed,
            "tree_method": "hist",
        }
        model = xgb.train(params, dtrain, num_boost_round=2000,
                         evals=[(dval, "val")], early_stopping_rounds=100, verbose_eval=False)
        path = MODELS_DIR / f"no_est_xgb_model_{seed - 41}.json"
        model.save_model(str(path))
        xgb_models.append(model)
        logger.info(f"  Saved: {path}")

    # Evaluate ensemble on test set
    preds = []
    for m in cb_models:
        preds.append(m.predict(X_test))
    for m in xgb_models:
        X_te_xgb = X_test.copy()
        for col in available_cat:
            X_te_xgb[col] = X_te_xgb[col].map(label_maps[col]).fillna(-1).astype(int)
        preds.append(m.predict(xgb.DMatrix(X_te_xgb)))

    ensemble = np.mean(preds, axis=0)

    # Metrics
    rmse = np.sqrt(np.mean((y_test - ensemble) ** 2))
    ss_res = np.sum((y_test - ensemble) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    actual_usd = np.expm1(y_test.values)
    pred_usd = np.expm1(ensemble)
    ae = np.abs(actual_usd - pred_usd)
    pct_err = ae / actual_usd

    logger.info("\n" + "=" * 50)
    logger.info("NO-ESTIMATES ENSEMBLE TEST METRICS")
    logger.info("=" * 50)
    logger.info(f"RMSE (log scale):     {rmse:.4f}")
    logger.info(f"R² (log scale):       {r2:.4f}")
    logger.info(f"MAE (USD):            ${ae.mean():,.0f}")
    logger.info(f"MAPE:                 {pct_err.mean() * 100:.1f}%")
    logger.info(f"Median AE (USD):      ${np.median(ae):,.0f}")
    logger.info(f"Within 25%:           {(pct_err <= 0.25).mean() * 100:.1f}%")
    logger.info(f"Within 50%:           {(pct_err <= 0.50).mean() * 100:.1f}%")

    # Save manifest
    manifest = {
        "feature_cols": feature_cols,
        "cat_cols": available_cat,
        "cat_indices": cat_indices,
        "excluded_features": ESTIMATE_FEATURES,
        "target": target_col,
        "metrics": {
            "rmse_log": round(float(rmse), 4),
            "r2_log": round(float(r2), 4),
            "mae_usd": round(float(ae.mean()), 2),
            "mape_pct": round(float(pct_err.mean() * 100), 2),
            "median_ae_usd": round(float(np.median(ae)), 2),
            "within_25_pct": round(float((pct_err <= 0.25).mean() * 100), 1),
            "within_50_pct": round(float((pct_err <= 0.50).mean() * 100), 1),
            "test_size": int(len(y_test)),
        },
    }
    manifest_path = MODELS_DIR / "no_est_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"\nSaved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
