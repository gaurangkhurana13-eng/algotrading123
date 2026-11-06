"""
Strategy Module: CatBoost Walk-Forward Purged Cross-Validation
Strict chronological splitting with 2-day embargo to eliminate lookahead leakage.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from .features import FEATURE_COLS, compute_features
from .labels import apply_triple_barrier_labels

def build_dataset(cached_path: str, config: dict) -> pd.DataFrame:
    raw_data = pd.read_pickle(cached_path)
    tp_mult   = config.get("tp_atr_mult", 2.0)
    sl_mult   = config.get("sl_atr_mult", 1.5)
    min_atr   = config.get("min_atr_norm", 0.005)
    hold_bars = config.get("hold_bars", 12)

    dfs = []
    for ticker in raw_data.columns.levels[0]:
        if ticker == "^NSEI": continue
        try:
            df_s = raw_data[ticker].dropna().copy()
            if len(df_s) < 200: continue
            df_s.columns = [c.lower() for c in df_s.columns]
            df_feat = compute_features(df_s)
            df_lab  = apply_triple_barrier_labels(df_feat, tp_mult=tp_mult, sl_mult=sl_mult, min_atr=min_atr, hold_bars=hold_bars)
            df_lab["ticker"] = ticker.replace(".NS", "")
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
    adx_min = config.get("adx_min_trend", 22.0)

    fold_dates = n_dates // (n_folds + 1)
    oos_list = []

    for fold in range(1, n_folds + 1):
        train_end_idx = fold * fold_dates
        test_start_idx = train_end_idx + embargo_days
        test_end_idx = min(test_start_idx + fold_dates, n_dates)

        if train_end_idx < min_train_days or test_start_idx >= n_dates:
            continue

        train_cutoff    = unique_dates[train_end_idx]
        test_start_date = unique_dates[test_start_idx]
        test_end_date   = unique_dates[test_end_idx - 1]

        tr = df_pool[df_pool.index.date < train_cutoff]
        te = df_pool[(df_pool.index.date >= test_start_date) & (df_pool.index.date <= test_end_date)]

        tr_f = tr[(tr["near_close"] == 0) & (tr["adx"] >= adx_min) & (tr["above_vwap"] == 1)]
        pos_labels = tr_f["y_buy"].sum()
        if len(tr_f) < 400 or pos_labels < 40:
            continue

        pos = tr_f["y_buy"].sum()
        neg = len(tr_f) - pos
        cw  = {0: 1, 1: max(2, min(5, neg // max(pos, 1)))}

        model = CatBoostClassifier(
            iterations    = wf_cfg.get("iterations", 400),
            depth         = wf_cfg.get("depth", 4),
            l2_leaf_reg   = wf_cfg.get("l2_reg", 15.0),
            learning_rate = wf_cfg.get("learning_rate", 0.04),
            random_seed   = 42,
            eval_metric   = "AUC",
            od_type       = "Iter",
            od_wait       = 35,
            verbose       = 0,
            class_weights = cw
        )
        te_f = te[(te["near_close"] == 0) & (te["adx"] >= adx_min) & (te["above_vwap"] == 1)].copy()
        if len(te_f) == 0:
            continue

        dates_tr = sorted(list(set(tr_f.index.date)))
        n_dates_tr = len(dates_tr)
        cal_split_idx = int(n_dates_tr * 0.80)
        cal_split_date = dates_tr[cal_split_idx]

        train_sub = tr_f[tr_f.index.date < cal_split_date]
        cal_sub   = tr_f[tr_f.index.date >= cal_split_date]

        if len(train_sub) < 300 or len(cal_sub) < 50 or train_sub["y_buy"].sum() < 20 or cal_sub["y_buy"].sum() < 5:
            model.fit(tr_f[FEATURE_COLS].values, tr_f["y_buy"].values)
            prob_preds = model.predict_proba(te_f[FEATURE_COLS].values)[:, 1]
        else:
            model.fit(train_sub[FEATURE_COLS].values, train_sub["y_buy"].values)
            from sklearn.linear_model import LogisticRegression
            p_val = model.predict_proba(cal_sub[FEATURE_COLS].values)[:, 1].reshape(-1, 1)
            platt = LogisticRegression(C=1.0)
            platt.fit(p_val, cal_sub["y_buy"].values)
            p_test = model.predict_proba(te_f[FEATURE_COLS].values)[:, 1].reshape(-1, 1)
            prob_preds = platt.predict_proba(p_test)[:, 1]

        te_f["prob"] = prob_preds
        te_f["fold"] = fold
        oos_list.append(te_f)

    if not oos_list:
        raise ValueError("Walk-forward produced 0 out-of-sample data!")

    oos = pd.concat(oos_list).sort_index()
    return oos
