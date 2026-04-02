#!/usr/bin/env python3
"""Train price prediction ensemble: 5 CatBoost + 2 XGBoost regressors.

Target: log(hammer_price_usd)
Split: Chronological train/val/test (no future leakage)

Output: models/saved/model_*.cbm, xgb_model_*.json, model_manifest.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.helpers import setup_logger, MODELS_DIR, DATA_PROCESSED

logger = setup_logger(__name__, "train.log")

ML_READY_FILE = DATA_PROCESSED / "ml_ready.csv"
MANIFEST_FILE = MODELS_DIR / "model_manifest.json"

# Seeds for ensemble diversity
CB_SEEDS = [42, 43, 44, 45, 46]
XGB_SEEDS = [42, 43]

# Default hyperparameters
CB_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "min_data_in_leaf": 5,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "early_stopping_rounds": 100,
    "verbose": 100,
}

XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "reg_lambda": 3.0,
    "min_child_weight": 5,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "early_stopping_rounds": 100,
    "verbosity": 0,
}


def load_data():
    """Load ML-ready data and split chronologically."""
    df = pd.read_csv(ML_READY_FILE)
    df["auction_date"] = pd.to_datetime(df["auction_date"])
    df = df.sort_values("auction_date").reset_index(drop=True)

    logger.info(f"Loaded {len(df)} rows")

    # Identify feature columns
    target_col = "log_hammer_price"
    meta_cols = ["auction_date", "hammer_price_usd", "artist_name", "title", "lot_id"]
    cat_cols = ["medium_category", "artist_name_clean", "subject", "palette", "style"]
    feature_cols = [c for c in df.columns if c not in [target_col] + meta_cols]

    # Chronological split: train 80%, val 10%, test 10%
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    logger.info(f"Train period: {train_df['auction_date'].min()} to {train_df['auction_date'].max()}")
    logger.info(f"Test period:  {test_df['auction_date'].min()} to {test_df['auction_date'].max()}")

    # Separate features and target
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Sample weights: linear increasing to favor recent data
    w_train = np.linspace(1.0, 1.5, len(X_train))

    # Identify categorical column indices for CatBoost
    cat_indices = [i for i, c in enumerate(feature_cols) if c in cat_cols]

    return {
        "X_train": X_train, "y_train": y_train, "w_train": w_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_cols": feature_cols, "cat_cols": cat_cols, "cat_indices": cat_indices,
        "test_df": test_df,
    }


def train_catboost(data: dict, seed: int, model_idx: int) -> CatBoostRegressor:
    """Train a single CatBoost model."""
    logger.info(f"\n--- CatBoost model {model_idx} (seed={seed}) ---")

    params = CB_PARAMS.copy()
    params["random_seed"] = seed

    # Fill NaN in categoricals for CatBoost
    X_train = data["X_train"].copy()
    X_val = data["X_val"].copy()
    for col in data["cat_cols"]:
        if col in X_train.columns:
            X_train[col] = X_train[col].fillna("unknown").astype(str)
            X_val[col] = X_val[col].fillna("unknown").astype(str)

    train_pool = Pool(X_train, data["y_train"], cat_features=data["cat_indices"], weight=data["w_train"])
    val_pool = Pool(X_val, data["y_val"], cat_features=data["cat_indices"])

    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # Save
    save_path = MODELS_DIR / f"model_{model_idx}.cbm"
    model.save_model(str(save_path))
    logger.info(f"  Saved: {save_path}")

    return model


def train_xgboost(data: dict, seed: int, model_idx: int) -> XGBRegressor:
    """Train a single XGBoost model."""
    logger.info(f"\n--- XGBoost model {model_idx} (seed={seed}) ---")

    params = XGB_PARAMS.copy()
    params["random_state"] = seed

    # XGBoost needs numeric only — encode categoricals
    X_train = data["X_train"].copy()
    X_val = data["X_val"].copy()
    for col in data["cat_cols"]:
        if col in X_train.columns:
            # Simple label encoding
            combined = pd.concat([X_train[col], X_val[col]]).fillna("unknown").astype(str)
            codes = {v: i for i, v in enumerate(combined.unique())}
            X_train[col] = X_train[col].fillna("unknown").astype(str).map(codes).astype(float)
            X_val[col] = X_val[col].fillna("unknown").astype(str).map(codes).astype(float)

    model = XGBRegressor(**params)
    model.fit(
        X_train, data["y_train"],
        sample_weight=data["w_train"],
        eval_set=[(X_val, data["y_val"])],
        verbose=False,
    )

    # Save
    save_path = MODELS_DIR / f"xgb_model_{model_idx}.json"
    model.save_model(str(save_path))
    logger.info(f"  Saved: {save_path}")

    return model


def evaluate_ensemble(models_cb, models_xgb, data):
    """Evaluate the full ensemble on test set."""
    X_test = data["X_test"].copy()
    y_test = data["y_test"]

    # CatBoost predictions
    X_test_cb = X_test.copy()
    for col in data["cat_cols"]:
        if col in X_test_cb.columns:
            X_test_cb[col] = X_test_cb[col].fillna("unknown").astype(str)

    test_pool = Pool(X_test_cb, cat_features=data["cat_indices"])
    preds_cb = np.array([m.predict(test_pool) for m in models_cb])

    # XGBoost predictions
    X_test_xgb = X_test.copy()
    for col in data["cat_cols"]:
        if col in X_test_xgb.columns:
            combined = pd.concat([data["X_train"][col], data["X_val"][col], X_test_xgb[col]]).fillna("unknown").astype(str)
            codes = {v: i for i, v in enumerate(combined.unique())}
            X_test_xgb[col] = X_test_xgb[col].fillna("unknown").astype(str).map(codes).astype(float)

    preds_xgb = np.array([m.predict(X_test_xgb) for m in models_xgb])

    # Ensemble average
    all_preds = np.concatenate([preds_cb, preds_xgb], axis=0)
    ensemble_pred = all_preds.mean(axis=0)

    # Metrics on log scale
    rmse_log = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    r2_log = r2_score(y_test, ensemble_pred)

    # Metrics on dollar scale
    y_test_usd = np.exp(y_test)
    pred_usd = np.exp(ensemble_pred)
    mae_usd = mean_absolute_error(y_test_usd, pred_usd)
    mape = np.mean(np.abs((y_test_usd - pred_usd) / y_test_usd)) * 100
    median_ae = np.median(np.abs(y_test_usd - pred_usd))

    # Within-estimate accuracy
    test_df = data["test_df"]
    within_est = 0
    for i, (_, row) in enumerate(test_df.iterrows()):
        est_low = row.get("estimate_low_usd", np.nan) if "estimate_low_usd" in test_df.columns else np.nan
        est_high = row.get("estimate_high_usd", np.nan) if "estimate_high_usd" in test_df.columns else np.nan
        p = pred_usd.iloc[i] if hasattr(pred_usd, "iloc") else pred_usd[i]
        if pd.notna(est_low) and pd.notna(est_high):
            if est_low <= p <= est_high:
                within_est += 1

    metrics = {
        "rmse_log": round(rmse_log, 4),
        "r2_log": round(r2_log, 4),
        "mae_usd": round(mae_usd, 2),
        "mape_pct": round(mape, 2),
        "median_ae_usd": round(median_ae, 2),
        "test_size": len(y_test),
        "train_size": len(data["y_train"]),
        "n_catboost": len(models_cb),
        "n_xgboost": len(models_xgb),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"ENSEMBLE TEST METRICS")
    logger.info(f"{'='*50}")
    logger.info(f"RMSE (log scale):     {rmse_log:.4f}")
    logger.info(f"R² (log scale):       {r2_log:.4f}")
    logger.info(f"MAE (USD):            ${mae_usd:,.0f}")
    logger.info(f"MAPE:                 {mape:.1f}%")
    logger.info(f"Median AE (USD):      ${median_ae:,.0f}")

    return metrics


def main():
    logger.info("=" * 60)
    logger.info("Training Price Prediction Ensemble")
    logger.info("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Try to load HPO params
    hpo_cb = MODELS_DIR / "best_params_catboost.json"
    hpo_xgb = MODELS_DIR / "best_params_xgboost.json"
    if hpo_cb.exists():
        with hpo_cb.open() as f:
            CB_PARAMS.update(json.load(f))
        logger.info("Loaded CatBoost HPO params")
    if hpo_xgb.exists():
        with hpo_xgb.open() as f:
            XGB_PARAMS.update(json.load(f))
        logger.info("Loaded XGBoost HPO params")

    data = load_data()

    # Train CatBoost models
    models_cb = []
    for i, seed in enumerate(CB_SEEDS, 1):
        model = train_catboost(data, seed, i)
        models_cb.append(model)

    # Train XGBoost models
    models_xgb = []
    for i, seed in enumerate(XGB_SEEDS, 1):
        model = train_xgboost(data, seed, i)
        models_xgb.append(model)

    # Evaluate
    metrics = evaluate_ensemble(models_cb, models_xgb, data)

    # Save manifest
    manifest = {
        "feature_cols": data["feature_cols"],
        "cat_cols": data["cat_cols"],
        "cat_indices": data["cat_indices"],
        "target": "log_hammer_price",
        "catboost_models": [f"model_{i}.cbm" for i in range(1, len(CB_SEEDS) + 1)],
        "xgboost_models": [f"xgb_model_{i}.json" for i in range(1, len(XGB_SEEDS) + 1)],
        "cb_seeds": CB_SEEDS,
        "xgb_seeds": XGB_SEEDS,
        "metrics": metrics,
    }
    with MANIFEST_FILE.open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"\nSaved manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
