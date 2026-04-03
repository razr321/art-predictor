#!/usr/bin/env python3
"""Hyperparameter tuning with Optuna for CatBoost + XGBoost.

Saves best params to models/saved/best_params_catboost.json and
models/saved/best_params_xgboost.json, which train.py auto-loads.

Usage:
    python3 models/tune.py              # 100 trials each (default)
    python3 models/tune.py --trials 50  # fewer trials for quick run
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.helpers import setup_logger, MODELS_DIR, DATA_PROCESSED

logger = setup_logger(__name__, "tune.log")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ML_READY_FILE = DATA_PROCESSED / "ml_ready.csv"


def load_data():
    """Same chronological split as train.py."""
    df = pd.read_csv(ML_READY_FILE)
    df["auction_date"] = pd.to_datetime(df["auction_date"])
    df = df.sort_values("auction_date").reset_index(drop=True)

    target_col = "log_hammer_price"
    meta_cols = ["auction_date", "hammer_price_usd", "artist_name", "title", "lot_id"]
    cat_cols = ["medium_category", "artist_name_clean", "subject", "palette", "style", "title_subject", "creation_period"]
    feature_cols = [c for c in df.columns if c not in [target_col] + meta_cols]

    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    w_train = np.linspace(1.0, 1.5, len(X_train))
    cat_indices = [i for i, c in enumerate(feature_cols) if c in cat_cols]

    return {
        "X_train": X_train, "y_train": y_train, "w_train": w_train,
        "X_val": X_val, "y_val": y_val,
        "feature_cols": feature_cols, "cat_cols": cat_cols, "cat_indices": cat_indices,
    }


def objective_catboost(trial, data):
    params = {
        "iterations": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 10.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),
        "random_strength": trial.suggest_float("random_strength", 0.1, 5.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "early_stopping_rounds": 50,
        "verbose": 0,
        "random_seed": 42,
    }

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

    preds = model.predict(val_pool)
    return np.sqrt(mean_squared_error(data["y_val"], preds))


def objective_xgboost(trial, data):
    params = {
        "n_estimators": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "early_stopping_rounds": 50,
        "verbosity": 0,
        "random_state": 42,
    }

    X_train = data["X_train"].copy()
    X_val = data["X_val"].copy()
    for col in data["cat_cols"]:
        if col in X_train.columns:
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

    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(data["y_val"], preds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100, help="Trials per model type")
    args = parser.parse_args()

    logger.info(f"Starting HPO with {args.trials} trials per model")
    data = load_data()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- CatBoost ---
    logger.info("Tuning CatBoost...")
    study_cb = optuna.create_study(direction="minimize", study_name="catboost")
    study_cb.optimize(lambda t: objective_catboost(t, data), n_trials=args.trials, show_progress_bar=True)

    best_cb = study_cb.best_params
    best_cb_score = study_cb.best_value
    logger.info(f"CatBoost best RMSE: {best_cb_score:.4f}")
    logger.info(f"CatBoost best params: {best_cb}")

    out_cb = MODELS_DIR / "best_params_catboost.json"
    with out_cb.open("w") as f:
        json.dump(best_cb, f, indent=2)
    logger.info(f"Saved: {out_cb}")

    # --- XGBoost ---
    logger.info("Tuning XGBoost...")
    study_xgb = optuna.create_study(direction="minimize", study_name="xgboost")
    study_xgb.optimize(lambda t: objective_xgboost(t, data), n_trials=args.trials, show_progress_bar=True)

    best_xgb = study_xgb.best_params
    best_xgb_score = study_xgb.best_value
    logger.info(f"XGBoost best RMSE: {best_xgb_score:.4f}")
    logger.info(f"XGBoost best params: {best_xgb}")

    out_xgb = MODELS_DIR / "best_params_xgboost.json"
    with out_xgb.open("w") as f:
        json.dump(best_xgb, f, indent=2)
    logger.info(f"Saved: {out_xgb}")

    # Summary
    print(f"\n{'='*50}")
    print(f"HPO COMPLETE — {args.trials} trials each")
    print(f"{'='*50}")
    print(f"CatBoost  val RMSE: {best_cb_score:.4f}  (was ~0.72 with defaults)")
    print(f"XGBoost   val RMSE: {best_xgb_score:.4f}")
    print(f"\nSaved to:")
    print(f"  {out_cb}")
    print(f"  {out_xgb}")
    print(f"\nNow re-run:  python3 models/train.py")


if __name__ == "__main__":
    main()
