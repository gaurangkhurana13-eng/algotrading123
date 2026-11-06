"""
Strategy Module for Bhavishya V25: Continuous Features + Platt Calibration
Trains CatBoost with 35 continuous features (including CLV and Upper Wick Ratio).
Evaluates out-of-sample bullish test bars with Platt Sigmoid Probability Calibration.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from .features import FEATURE_COLS, compute_features
from .labels import apply_triple_barrier_labels
from .nifty_filter import compute_nifty_regime

def build_dataset(cached_path: str, config: dict) -> pd.DataFrame:
    raw_data = pd.read_pickle(cached_path)
    min_atr   = config.get("min_atr_norm", 0.005)
    hold_bars = config.get("hold_bars", 12)

    nifty_allowed = compute_nifty_regime(raw_data)

    dfs = []
    for ticker in raw_data.columns.levels[0]:
        if ticker == "^NSEI": continue
        try:
            df_s = raw_data[ticker].dropna().copy()
            if len(df_s) < 200: continue

            df_s.columns = [c.lower() for c in df_s.columns]
            df_feat = compute_features(df_s)
            df_lab  = apply_triple_barrier_labels(df_feat, min_atr=min_atr, hold_bars=hold_bars)
            df_lab["ticker"] = ticker.replace(".NS", "")
            
            # Reindex lagged Nifty soft gate to stock timestamps
            df_lab["nifty_allowed"] = nifty_allowed.reindex(df_lab.index).fillna(True).astype(int)
            dfs.append(df_lab)
        except Exception:
            continue

    if not dfs:
        raise ValueError("No stocks processed successfully!")

    master_df = pd.concat(dfs).sort_index()
    return master_df

def run_walk_forward_pipeline(df_pool: pd.DataFrame, config: dict):
    df_pool = df_pool.sort_index()
    unique_dates = pd.Series(df_pool.index.date).drop_duplicates().tolist()
    n_dates = len(unique_dates)

    wf_cfg = config.get("walk_forward", {})
    n_folds = wf_cfg.get("n_folds", 5)
    embargo_days = wf_cfg.get("embargo_days", 2)
    min_train_days = wf_cfg.get("min_train_days", 25)
    adx_min = config.get("adx_min_trend", 20.0)

    fold_dates = n_dates // (n_folds + 1)
    oos_records = []

    for fold in range(n_folds):
        train_end_idx = min_train_days + fold * fold_dates
        test_start_idx = train_end_idx + embargo_days
        test_end_idx = min(n_dates, test_start_idx + fold_dates)

        if test_start_idx >= n_dates:
            break

        train_dates = set(unique_dates[:train_end_idx])
        test_dates  = set(unique_dates[test_start_idx:test_end_idx])

        train_mask = df_pool.index.map(lambda x: x.date() in train_dates)
        test_mask  = df_pool.index.map(lambda x: x.date() in test_dates)

        train_df = df_pool[train_mask].dropna(subset=FEATURE_COLS + ["y_buy"])
        test_df  = df_pool[test_mask].dropna(subset=FEATURE_COLS + ["y_buy"])

        if len(train_df) < 500 or len(test_df) < 100:
            continue

        X_train, y_train = train_df[FEATURE_COLS], train_df["y_buy"]

        # Chronological validation split (last 20% of training data) for Platt Calibration
        val_split_idx = int(len(train_df) * 0.8)
        X_tr, y_tr = X_train.iloc[:val_split_idx], y_train.iloc[:val_split_idx]
        X_val, y_val = X_train.iloc[val_split_idx:], y_train.iloc[val_split_idx:]

        # Filter test bars for bullish intraday trend momentum
        bullish_test = test_df[
            (test_df["above_vwap"] == 1) &
            (test_df["adx"] >= adx_min) &
            (test_df["nifty_allowed"] == 1)
        ].copy()

        if len(bullish_test) == 0:
            continue

        X_test = bullish_test[FEATURE_COLS]

        cb = CatBoostClassifier(
            iterations=wf_cfg.get("iterations", 400),
            depth=wf_cfg.get("depth", 4),
            l2_leaf_reg=wf_cfg.get("l2_reg", 15.0),
            learning_rate=wf_cfg.get("learning_rate", 0.04),
            verbose=False,
            random_seed=42
        )
        cb.fit(X_tr, y_tr)

        # Platt Sigmoid Calibration
        val_raw_probs = cb.predict_proba(X_val)[:, 1].reshape(-1, 1)
        test_raw_probs = cb.predict_proba(X_test)[:, 1].reshape(-1, 1)

        platt = LogisticRegression(C=1.0, solver="lbfgs")
        platt.fit(val_raw_probs, y_val)
        calibrated_test_probs = platt.predict_proba(test_raw_probs)[:, 1]

        bullish_test["prob"] = calibrated_test_probs
        bullish_test["fold"] = fold + 1
        oos_records.append(bullish_test)

    if not oos_records:
        raise ValueError("No out-of-sample predictions generated!")

    master_oos = pd.concat(oos_records).sort_index()
    return master_oos
