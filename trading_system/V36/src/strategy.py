"""
Strategy Module for Bhavishya V36:
Orthogonal 3-Model Stacking Ensemble with Claude Sonnet 7-Factor High-Conviction Filter.
Processes Nifty 50 (^NSEI) for Market Regime Alignment (Factor 6).
Strict chronological walk-forward splitting with 2-day embargo guarantees 100% out-of-sample realism.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from .features import FEATURE_COLS, compute_stock_features, augment_cross_sectional_features
from .labels import apply_triple_barrier_labels

def build_dataset(cached_path: str, config: dict) -> pd.DataFrame:
    raw_data = pd.read_pickle(cached_path)
    tp_mult   = config.get("tp_atr_mult", 2.0)
    sl_mult   = config.get("stop_d_mult", 1.4)
    min_atr   = config.get("min_atr_norm", 0.0035)
    hold_bars = config.get("hold_bars", 12)

    # Process Nifty 50 (^NSEI) for Factor 6 (Index Gate)
    nifty_df = None
    if "^NSEI" in raw_data.columns.levels[0]:
        df_n = raw_data["^NSEI"].dropna().copy()
        df_n.columns = [c.lower() for c in df_n.columns]
        df_n["_date"] = df_n.index.date
        df_n["_tpv"] = ((df_n["high"] + df_n["low"] + df_n["close"]) / 3) * df_n["volume"]
        df_n["_vcum"] = df_n.groupby("_date")["_tpv"].cumsum()
        df_n["_volcum"] = df_n.groupby("_date")["volume"].cumsum()
        df_n["vwap"] = (df_n["_vcum"] / (df_n["_volcum"] + 1e-9)).shift(1)
        nifty_df = df_n[["close", "vwap"]].copy()

    dfs = []
    for ticker in raw_data.columns.levels[0]:
        if ticker == "^NSEI": continue
        try:
            df_s = raw_data[ticker].dropna().copy()
            if len(df_s) < 200: continue
            df_s.columns = [c.lower() for c in df_s.columns]
            df_feat = compute_stock_features(df_s, nifty_df=nifty_df)
            df_lab  = apply_triple_barrier_labels(df_feat, tp_mult=tp_mult, sl_mult=sl_mult, min_atr=min_atr, hold_bars=hold_bars)
            df_lab["ticker"] = ticker.replace(".NS", "")
            dfs.append(df_lab)
        except Exception:
            continue

    if not dfs:
        raise ValueError("No stocks processed successfully!")

    raw_pool = pd.concat(dfs).sort_index()
    print("  Adding Cross-Sectional Relative Strength (CS_RS) across universe...")
    master_df = augment_cross_sectional_features(raw_pool)
    return master_df

def run_walk_forward_pipeline(df_pool: pd.DataFrame, config: dict):
    df_pool = df_pool.sort_index()
    unique_dates = pd.Series(df_pool.index.date).drop_duplicates().tolist()
    n_dates = len(unique_dates)

    wf_cfg = config.get("walk_forward", {})
    n_folds = wf_cfg.get("n_folds", 5)
    embargo_days = wf_cfg.get("embargo_days", 2)
    min_train_days = wf_cfg.get("min_train_days", 15)
    adx_min = config.get("adx_min_trend", 20.0)

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

        tr_f = tr[(tr["near_close"] == 0) & (tr["adx"] >= adx_min) & (tr["above_vwap"] == 1)].dropna(subset=FEATURE_COLS)
        pos_labels = tr_f["y_buy"].sum()
        if len(tr_f) < 400 or pos_labels < 40:
            continue

        pos = tr_f["y_buy"].sum()
        neg = len(tr_f) - pos
        scale_pos = max(1.5, min(4.0, neg / max(pos, 1)))

        te_f = te[(te["near_close"] == 0) & (te["adx"] >= adx_min) & (te["above_vwap"] == 1)].dropna(subset=FEATURE_COLS).copy()
        if len(te_f) == 0:
            continue

        # Split train into train_sub (80%) and cal_sub (20%) for stacking and calibration
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

        # Model 1: CatBoost
        cb = CatBoostClassifier(
            iterations=wf_cfg.get("iterations", 350),
            depth=wf_cfg.get("depth", 5),
            l2_leaf_reg=wf_cfg.get("l2_reg", 5.0),
            learning_rate=wf_cfg.get("learning_rate", 0.04),
            random_seed=42,
            eval_metric="AUC",
            verbose=0,
            scale_pos_weight=scale_pos
        )
        cb.fit(X_tr, y_tr)

        # Model 2: LightGBM
        lgb = LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            num_leaves=15,
            colsample_bytree=0.7,
            subsample=0.8,
            min_child_samples=25,
            learning_rate=0.04,
            random_state=42,
            scale_pos_weight=scale_pos,
            verbose=-1
        )
        lgb.fit(X_tr, y_tr)

        # Model 3: XGBoost
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            colsample_bytree=0.7,
            reg_lambda=4.0,
            reg_alpha=1.0,
            learning_rate=0.04,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos,
            verbosity=0
        )
        xgb.fit(X_tr, y_tr)

        # Out-of-fold predictions on calibration split
        p_cal_cb  = cb.predict_proba(X_cal)[:, 1]
        p_cal_lgb = lgb.predict_proba(X_cal)[:, 1]
        p_cal_xgb = xgb.predict_proba(X_cal)[:, 1]
        P_cal_matrix = np.column_stack([p_cal_cb, p_cal_lgb, p_cal_xgb])

        # Test predictions from 3 models
        p_te_cb  = cb.predict_proba(X_te)[:, 1]
        p_te_lgb = lgb.predict_proba(X_te)[:, 1]
        p_te_xgb = xgb.predict_proba(X_te)[:, 1]
        P_te_matrix = np.column_stack([p_te_cb, p_te_lgb, p_te_xgb])

        # Stacking Calibrator: Logistic Regression on calibration split
        if len(np.unique(y_cal)) < 2 or y_cal.sum() < 5:
            calibrated_probs_cal = np.mean(P_cal_matrix, axis=1)
            calibrated_probs = np.mean(P_te_matrix, axis=1)
        else:
            stacker = LogisticRegression(C=1.0)
            stacker.fit(P_cal_matrix, y_cal)
            calibrated_probs_cal = stacker.predict_proba(P_cal_matrix)[:, 1]
            calibrated_probs = stacker.predict_proba(P_te_matrix)[:, 1]

        mu_f = float(np.mean(calibrated_probs_cal))
        sigma_f = float(np.std(calibrated_probs_cal))
        if sigma_f < 1e-4:
            sigma_f = 0.05

        te_f["prob"] = calibrated_probs
        te_f["mu_f"] = mu_f
        te_f["sigma_f"] = sigma_f
        te_f["z_score"] = (calibrated_probs - mu_f) / sigma_f
        te_f["prob_cb"] = p_te_cb
        te_f["prob_lgb"] = p_te_lgb
        te_f["prob_xgb"] = p_te_xgb
        te_f["fold"] = fold

        def calc_rank_score(g):
            if len(g) <= 1:
                g["rank_score"] = 0.50
            else:
                g["rank_score"] = g["prob"].rank(pct=True).values
            return g

        te_f = te_f.groupby(te_f.index, group_keys=False).apply(calc_rank_score)
        print(f"  Fold {fold} ({test_start_date} to {test_end_date}): mu_f={mu_f:.4f}, sigma_f={sigma_f:.4f}, Bars={len(te_f)}")
        oos_list.append(te_f)

    if not oos_list:
        raise ValueError("Walk-forward produced 0 out-of-sample data!")

    oos = pd.concat(oos_list).sort_index()
    return oos
