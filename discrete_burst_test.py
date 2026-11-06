import sys
import os
import warnings
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from scipy.stats import rankdata

from src.data_fetcher import DataFetcher
from burst_quant_ai import build_burst_features, create_high_density_labels, HighCapacityAlphaNet

fetcher = DataFetcher(data_dir="data")

def evaluate_discrete_trades(ticker="PCJEWELLER", tf="1h", conf_threshold=0.65):
    raw = fetcher.get_data(ticker, tf)
    df, feat_cols = build_burst_features(raw)
    df = create_high_density_labels(df, tp_mult=2.0, sl_mult=1.2, max_bars=10)
    df = df.dropna(subset=feat_cols).copy()
    
    n = len(df)
    split = int(n * 0.70)
    train = df.iloc[:split]
    test = df.iloc[split:].copy()
    
    X_train, y_train = train[feat_cols].values, train["label"].values
    X_test, y_test = test[feat_cols].values, test["label"].values
    
    # Train Ensemble
    pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-5)
    params_lgb = {
        "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.04, 
        "num_leaves": 45, "scale_pos_weight": pos_weight * 0.8, "verbose": -1
    }
    m_lgb = lgb.train(params_lgb, lgb.Dataset(X_train, label=y_train), num_boost_round=180)
    p_lgb = m_lgb.predict(X_test)
    
    m_xgb = xgb.XGBClassifier(n_estimators=140, learning_rate=0.03, max_depth=6, scale_pos_weight=pos_weight * 0.8, eval_metric="logloss")
    m_xgb.fit(X_train, y_train)
    p_xgb = m_xgb.predict_proba(X_test)[:, 1]
    
    ensemble_prob = (0.55 * p_lgb) + (0.45 * p_xgb)
    calibrated_conf = rankdata(ensemble_prob) / len(ensemble_prob)
    test["conf"] = calibrated_conf
    
    # Simulate Real Discrete Trade Execution (One position at a time)
    c = test["close"].values
    h = test["high"].values
    l = test["low"].values
    atr = test["atr_10"].values
    conf = test["conf"].values
    
    capital = 10000.0
    pos = 0 # 0 or 1
    entry_p = 0.0
    tp = 0.0
    sl = 0.0
    trades = []
    
    for i in range(len(test)):
        cur_c = c[i]
        cur_h = h[i]
        cur_l = l[i]
        cur_atr = atr[i]
        cur_conf = conf[i]
        
        if pos == 0:
            if cur_conf >= conf_threshold:
                pos = 1
                entry_p = cur_c
                tp = entry_p + (cur_atr * 2.2) # Target +2.2 ATR
                sl = entry_p - (cur_atr * 1.3) # Stop Loss -1.3 ATR
                entry_idx = i
        elif pos == 1:
            exit_triggered = False
            exit_p = 0.0
            
            if cur_h >= tp:
                exit_triggered = True
                exit_p = tp
            elif cur_l <= sl:
                exit_triggered = True
                exit_p = sl
            elif (i - entry_idx) >= 12: # Time limit
                exit_triggered = True
                exit_p = cur_c
                
            if exit_triggered:
                pnl_pct = ((exit_p - entry_p) / entry_p) * 100.0 - 0.06
                pnl_abs = capital * (pnl_pct / 100.0)
                capital += pnl_abs
                trades.append({"pnl_pct": pnl_pct, "capital": capital})
                pos = 0
                
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["pnl_pct"] > 0]
    losses = tdf[tdf["pnl_pct"] <= 0]
    total_trades = len(tdf)
    win_rate = (len(wins) / total_trades) * 100.0 if total_trades > 0 else 0.0
    net_profit = ((capital - 10000.0) / 10000.0) * 100.0
    gp = wins["pnl_pct"].sum()
    gl = abs(losses["pnl_pct"].sum())
    pf = (gp / gl) if gl > 0 else 99.0
    
    eq = tdf["capital"]
    peak = eq.cummax()
    dd = (eq - peak) / peak * 100.0
    max_dd = abs(dd.min()) if not dd.empty else 0.0
    
    return {
        "Ticker": ticker,
        "Conf Threshold": f"Top {(1.0-conf_threshold)*100:.0f}% (P >= {conf_threshold:.2f})",
        "Total Trades": total_trades,
        "Win Rate (%)": f"{win_rate:.1f}%",
        "Profit Factor": f"{pf:.2f}",
        "Net Return (%)": f"{net_profit:+.2f}%",
        "Max Drawdown (%)": f"{max_dd:.2f}%"
    }

print("Running Discrete Burst Backtest:")
res = []
for tkr in ["PCJEWELLER", "SUZLON"]:
    for th in [0.55, 0.60, 0.65, 0.70]:
        r = evaluate_discrete_trades(ticker=tkr, tf="1h", conf_threshold=th)
        res.append(r)

rdf = pd.DataFrame(res)
print(rdf.to_string(index=False))
