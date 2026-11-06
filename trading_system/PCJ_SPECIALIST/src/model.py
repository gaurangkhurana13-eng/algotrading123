"""
Dedicated CatBoost Walk-Forward Modeling Pipeline for PCJEWELLER.
Performs 5 chronological expanding walk-forward folds with a 3-day embargo.
Uses holdout calibration split to calculate fold-relative mu_f and sigma_f for z-score scaling.
Zero data leakage, 100% out-of-sample purity.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from .features import FEATURE_COLS, compute_pcj_features
from .labels import apply_pcj_triple_barrier_labels

def prepare_pcj_dataset(raw_pkl_path: str, config: dict) -> pd.DataFrame:
    raw_df = pd.read_pickle(raw_pkl_path)
    raw_df.columns = [c.lower() for c in raw_df.columns]
    
    tp_mult = config.get("tp_atr_mult", 2.1)
    sl_mult = config.get("sl_atr_mult", 1.45)
    hold_bars = config.get("hold_bars", 12)

    print("  Computing PCJEWELLER specific technical features...")
    feat_df = compute_pcj_features(raw_df)
    
    print("  Applying triple-barrier labels...")
    labeled_df = apply_pcj_triple_barrier_labels(feat_df, tp_mult=tp_mult, sl_mult=sl_mult, hold_bars=hold_bars)
    labeled_df["ticker"] = "PCJEWELLER"
    return labeled_df

def run_pcj_walk_forward(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.sort_index()
    unique_dates = pd.Series(df.index.date).drop_duplicates().tolist()
    n_dates = len(unique_dates)

    wf_cfg = config.get("walk_forward", {})
    n_folds = wf_cfg.get("n_folds", 5)
    embargo_days = wf_cfg.get("embargo_days", 3)
    min_train_days = wf_cfg.get("min_train_days", 45)

    fold_step = (n_dates - min_train_days) // n_folds
    oos_list = []

    print(f"Starting Walk-Forward across {n_dates} total trading days ({n_folds} folds, {embargo_days}-day embargo)...")

    for fold in range(1, n_folds + 1):
        train_end_idx = min_train_days + (fold - 1) * fold_step
        test_start_idx = train_end_idx + embargo_days
        test_end_idx = min_train_days + fold * fold_step if fold < n_folds else n_dates

        if test_start_idx >= n_dates:
            break

        train_cutoff    = unique_dates[train_end_idx]
        test_start_date = unique_dates[test_start_idx]
        test_end_date   = unique_dates[test_end_idx - 1]

        tr = df[df.index.date < train_cutoff]
        te = df[(df.index.date >= test_start_date) & (df.index.date <= test_end_date)]

        # Training filter: exclude near close, require ADX >= 18 and positive volume
        tr_f = tr[(tr["near_close"] == 0) & (tr["adx"] >= 18.0) & (tr["volume"] > 0)].dropna(subset=FEATURE_COLS)
        pos_labels = tr_f["y_buy"].sum()
        if len(tr_f) < 200 or pos_labels < 20:
            continue

        scale_pos = max(1.2, min(3.5, (len(tr_f) - pos_labels) / max(pos_labels, 1)))

        te_f = te[(te["near_close"] == 0) & (te["adx"] >= 18.0) & (te["volume"] > 0)].dropna(subset=FEATURE_COLS).copy()
        if len(te_f) == 0:
            continue

        # Hold out the last 20% of train dates as calibration split for mu_f and sigma_f
        dates_tr = sorted(list(set(tr_f.index.date)))
        cal_split_idx = int(len(dates_tr) * 0.80)
        cal_split_date = dates_tr[cal_split_idx]

        train_sub = tr_f[tr_f.index.date < cal_split_date]
        cal_sub   = tr_f[tr_f.index.date >= cal_split_date]

        X_tr = train_sub[FEATURE_COLS].values
        y_tr = train_sub["y_buy"].values
        X_cal = cal_sub[FEATURE_COLS].values
        y_cal = cal_sub["y_buy"].values
        X_te  = te_f[FEATURE_COLS].values

        # Dedicated CatBoostClassifier
        cb = CatBoostClassifier(
            iterations=wf_cfg.get("iterations", 450),
            depth=wf_cfg.get("depth", 5),
            l2_leaf_reg=wf_cfg.get("l2_reg", 6.0),
            learning_rate=wf_cfg.get("learning_rate", 0.035),
            random_seed=42,
            eval_metric="Logloss",
            verbose=0,
            scale_pos_weight=scale_pos
        )
        cb.fit(X_tr, y_tr, eval_set=(X_cal, y_cal), early_stopping_rounds=40, verbose=False)

        # Probabilities on calibration holdout and unseen test fold
        p_cal = cb.predict_proba(X_cal)[:, 1]
        p_te  = cb.predict_proba(X_te)[:, 1]

        mu_f = float(np.mean(p_cal))
        sigma_f = float(np.std(p_cal))
        if sigma_f < 1e-4:
            sigma_f = 0.05

        z_scores = (p_te - mu_f) / sigma_f

        te_f["prob"] = p_te
        te_f["mu_f"] = mu_f
        te_f["sigma_f"] = sigma_f
        te_f["z_score"] = z_scores
        te_f["fold"] = fold

        test_days_count = len(set(te_f.index.date))
        print(f"  Fold {fold}: Train={len(train_sub)} samples, Test={test_start_date} to {test_end_date} ({test_days_count} days, {len(te_f)} bars), mu_f={mu_f:.4f}, sigma_f={sigma_f:.4f}")
        oos_list.append(te_f)

    if not oos_list:
        raise ValueError("Walk-forward produced 0 out-of-sample data!")

    oos = pd.concat(oos_list).sort_index()
    total_test_days = len(set(oos.index.date))
    print(f"Total Out-Of-Sample Test Days: {total_test_days} days across {len(oos)} bars.")
    return oos
