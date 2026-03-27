#!/usr/bin/env python3
"""Evaluate trained models: metrics, confusion plots, feature importance.

Output: models/saved/evaluation_metrics.json, *.png plots, feature_importance.csv
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.helpers import setup_logger, MODELS_DIR, DATA_PROCESSED

logger = setup_logger(__name__, "evaluate.log")

ML_READY_FILE = DATA_PROCESSED / "ml_ready.csv"
MANIFEST_FILE = MODELS_DIR / "model_manifest.json"
METRICS_FILE = MODELS_DIR / "evaluation_metrics.json"


def load_models(manifest):
    """Load all trained models."""
    cb_models = []
    for fname in manifest["catboost_models"]:
        model = CatBoostRegressor()
        model.load_model(str(MODELS_DIR / fname))
        cb_models.append(model)

    xgb_models = []
    for fname in manifest["xgboost_models"]:
        model = XGBRegressor()
        model.load_model(str(MODELS_DIR / fname))
        xgb_models.append(model)

    return cb_models, xgb_models


def main():
    logger.info("=" * 60)
    logger.info("Model Evaluation")
    logger.info("=" * 60)

    with MANIFEST_FILE.open() as f:
        manifest = json.load(f)

    # Load data
    df = pd.read_csv(ML_READY_FILE)
    df["auction_date"] = pd.to_datetime(df["auction_date"])
    df = df.sort_values("auction_date").reset_index(drop=True)

    feature_cols = manifest["feature_cols"]
    cat_cols = manifest["cat_cols"]
    cat_indices = manifest["cat_indices"]

    # Split same as training
    n = len(df)
    test_df = df.iloc[int(n * 0.9):]
    X_test = test_df[feature_cols].copy()
    y_test = test_df["log_hammer_price"]

    # Load models
    cb_models, xgb_models = load_models(manifest)

    # CatBoost predictions
    X_cb = X_test.copy()
    for col in cat_cols:
        if col in X_cb.columns:
            X_cb[col] = X_cb[col].fillna("unknown").astype(str)
    test_pool = Pool(X_cb, cat_features=cat_indices)
    preds_cb = np.array([m.predict(test_pool) for m in cb_models])

    # XGBoost predictions
    X_xgb = X_test.copy()
    train_df = df.iloc[:int(n * 0.9)]
    for col in cat_cols:
        if col in X_xgb.columns:
            combined = pd.concat([train_df[col], X_xgb[col]]).fillna("unknown").astype(str)
            codes = {v: i for i, v in enumerate(combined.unique())}
            X_xgb[col] = X_xgb[col].fillna("unknown").astype(str).map(codes).astype(float)
    preds_xgb = np.array([m.predict(X_xgb) for m in xgb_models])

    # Ensemble
    all_preds = np.concatenate([preds_cb, preds_xgb], axis=0)
    ensemble_pred = all_preds.mean(axis=0)

    y_test_usd = np.exp(y_test.values)
    pred_usd = np.exp(ensemble_pred)

    # Metrics
    metrics = {
        "rmse_log": round(float(np.sqrt(mean_squared_error(y_test, ensemble_pred))), 4),
        "r2_log": round(float(r2_score(y_test, ensemble_pred)), 4),
        "mae_usd": round(float(mean_absolute_error(y_test_usd, pred_usd)), 2),
        "mape_pct": round(float(np.mean(np.abs((y_test_usd - pred_usd) / y_test_usd)) * 100), 2),
        "median_ae_usd": round(float(np.median(np.abs(y_test_usd - pred_usd))), 2),
        "test_size": int(len(y_test)),
        "test_period_start": str(test_df["auction_date"].min().date()),
        "test_period_end": str(test_df["auction_date"].max().date()),
    }

    with METRICS_FILE.open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics: {METRICS_FILE}")

    # --- Plots ---

    # 1. Predicted vs Actual scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test_usd, pred_usd, alpha=0.5, s=30)
    max_val = max(y_test_usd.max(), pred_usd.max())
    ax.plot([0, max_val], [0, max_val], "r--", label="Perfect prediction")
    ax.set_xlabel("Actual Hammer Price (USD)")
    ax.set_ylabel("Predicted Hammer Price (USD)")
    ax.set_title("Predicted vs Actual Hammer Prices")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(MODELS_DIR / "predicted_vs_actual.png", dpi=150)
    plt.close()

    # 2. Residual distribution
    residuals = pred_usd - y_test_usd
    pct_residuals = (pred_usd - y_test_usd) / y_test_usd * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(residuals, bins=50, alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Residual (USD)")
    axes[0].set_title("Residual Distribution")
    axes[0].axvline(0, color="red", linestyle="--")
    axes[1].hist(pct_residuals, bins=50, alpha=0.7, edgecolor="black", range=(-200, 200))
    axes[1].set_xlabel("Residual (%)")
    axes[1].set_title("Percentage Residual Distribution")
    axes[1].axvline(0, color="red", linestyle="--")
    plt.tight_layout()
    fig.savefig(MODELS_DIR / "residuals.png", dpi=150)
    plt.close()

    # 3. Feature importance (from first CatBoost model)
    importances = cb_models[0].get_feature_importance()
    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(MODELS_DIR / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = min(25, len(fi_df))
    top = fi_df.head(top_n)
    ax.barh(range(top_n), top["importance"].values, align="center")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Feature Importances (CatBoost)")
    plt.tight_layout()
    fig.savefig(MODELS_DIR / "feature_importance.png", dpi=150)
    plt.close()

    logger.info("Saved plots: predicted_vs_actual.png, residuals.png, feature_importance.png")
    logger.info(f"\nTest R²: {metrics['r2_log']}")
    logger.info(f"Test MAPE: {metrics['mape_pct']}%")
    logger.info(f"Test MAE: ${metrics['mae_usd']:,.0f}")


if __name__ == "__main__":
    main()
