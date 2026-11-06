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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from tabulate import tabulate

from src.data_fetcher import DataFetcher
from src.strategy import calculate_atr, calculate_rsi

fetcher = DataFetcher(data_dir="data")

# 1. Feature Engineering
def build_burst_features(df):
    d = df.copy()
    c, h, l, o, v = d["close"], d["high"], d["low"], d["open"], d["volume"]
    
    # Returns
    d["ret_1"] = c.pct_change(1)
    d["ret_2"] = c.pct_change(2)
    d["ret_4"] = c.pct_change(4)
    d["ret_8"] = c.pct_change(8)
    
    # Candlestick Mechanics
    hl_range = h - l + 1e-9
    d["body_ratio"] = (c - o) / hl_range
    d["wick_upper"] = (h - np.maximum(c, o)) / hl_range
    d["wick_lower"] = (np.minimum(c, o) - l) / hl_range
    
    # EMAs & Ribbons
    d["ema_8"] = c.ewm(span=8, adjust=False).mean()
    d["ema_21"] = c.ewm(span=21, adjust=False).mean()
    d["ema_55"] = c.ewm(span=55, adjust=False).mean()
    d["ema_200"] = c.ewm(span=200, adjust=False).mean()
    d["spread_fast"] = (d["ema_8"] - d["ema_21"]) / c
    d["spread_slow"] = (d["ema_21"] - d["ema_55"]) / c
    d["macro_trend"] = (c - d["ema_200"]) / c
    
    # Oscillators
    d["rsi_6"] = calculate_rsi(c, 6)
    d["rsi_14"] = calculate_rsi(c, 14)
    d["rsi_slope"] = d["rsi_6"] - d["rsi_6"].shift(2)
    
    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"] = (macd - signal) / c
    d["macd_hist_diff"] = d["macd_hist"] - d["macd_hist"].shift(1)
    
    # Volatility & Bands
    d["atr_10"] = calculate_atr(d, 10)
    d["atr_pct"] = d["atr_10"] / c
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    d["bb_squeeze"] = (2 * std20) / sma20
    d["bb_pos"] = (c - (sma20 - 2 * std20)) / (4 * std20 + 1e-9)
    
    # Volume Dynamics & VWAP
    vol_sma = v.rolling(20).mean()
    d["vol_surge"] = v / (vol_sma + 1e-9)
    cum_v = v.rolling(20).sum()
    cum_pv = (c * v).rolling(20).sum()
    d["vwap"] = cum_pv / (cum_v + 1e-9)
    d["vwap_spread"] = (c - d["vwap"]) / (d["vwap"] + 1e-9)
    
    feat_cols = [
        "ret_1", "ret_2", "ret_4", "ret_8", "body_ratio", "wick_upper", "wick_lower",
        "spread_fast", "spread_slow", "macro_trend", "rsi_6", "rsi_14", "rsi_slope",
        "macd_hist", "macd_hist_diff", "atr_pct", "bb_squeeze", "bb_pos", "vol_surge", "vwap_spread"
    ]
    return d, feat_cols

# 2. Optimized Barrier Formulation (High Signal Density)
def create_high_density_labels(df, tp_mult=1.8, sl_mult=1.2, max_bars=8):
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    atr = df["atr_10"].values
    n = len(df)
    
    labels = np.zeros(n)
    returns = np.zeros(n)
    
    for i in range(n - max_bars):
        entry_p = c[i]
        curr_atr = atr[i]
        tp = entry_p + (curr_atr * tp_mult)
        sl = entry_p - (curr_atr * sl_mult)
        
        hit_tp = False
        hit_sl = False
        
        for k in range(1, max_bars + 1):
            if h[i + k] >= tp:
                hit_tp = True
                break
            if l[i + k] <= sl:
                hit_sl = True
                break
                
        if hit_tp and not hit_sl:
            labels[i] = 1 # Winner
            returns[i] = (tp - entry_p) / entry_p
        elif hit_sl:
            labels[i] = 0 # Loss
            returns[i] = (sl - entry_p) / entry_p
        else:
            labels[i] = 0 # Chop / Timeout
            returns[i] = (c[i + max_bars] - entry_p) / entry_p
            
    df["label"] = labels
    df["trade_ret"] = returns
    return df

# 3. Deep PyTorch Network
class HighCapacityAlphaNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(), # Modern Swish activation
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def run_burst_optimization(ticker="PCJEWELLER", tf="1h"):
    raw = fetcher.get_data(ticker, tf)
    df, feat_cols = build_burst_features(raw)
    df = create_high_density_labels(df, tp_mult=1.8, sl_mult=1.2, max_bars=8)
    df = df.dropna(subset=feat_cols).copy()
    
    n = len(df)
    split = int(n * 0.70)
    train = df.iloc[:split]
    test = df.iloc[split:]
    
    X_train, y_train = train[feat_cols].values, train["label"].values
    X_test, y_test = test[feat_cols].values, test["label"].values
    
    # Model 1: LightGBM with Balanced Focal Boosting
    lgb_train = lgb.Dataset(X_train, label=y_train)
    pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-5)
    params_lgb = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 45,
        "learning_rate": 0.04,
        "scale_pos_weight": pos_weight * 0.8,
        "feature_fraction": 0.85,
        "verbose": -1
    }
    m_lgb = lgb.train(params_lgb, lgb_train, num_boost_round=180)
    raw_p_lgb = m_lgb.predict(X_test)
    
    # Model 2: XGBoost with Scaled Weights
    m_xgb = xgb.XGBClassifier(
        n_estimators=140, learning_rate=0.03, max_depth=6, 
        scale_pos_weight=pos_weight * 0.8, eval_metric="logloss"
    )
    m_xgb.fit(X_train, y_train)
    raw_p_xgb = m_xgb.predict_proba(X_test)[:, 1]
    
    # Model 3: PyTorch Deep Neural Network
    mean, std = np.mean(X_train, axis=0), np.std(X_train, axis=0) + 1e-7
    X_tr_t = torch.FloatTensor((X_train - mean) / std)
    y_tr_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_te_t = torch.FloatTensor((X_test - mean) / std)
    
    torch_model = HighCapacityAlphaNet(len(feat_cols))
    opt = torch.optim.AdamW(torch_model.parameters(), lr=0.003, weight_decay=1e-3)
    crit = nn.BCELoss()
    
    torch_model.train()
    for ep in range(70):
        opt.zero_grad()
        loss = crit(torch_model(X_tr_t), y_tr_t)
        loss.backward()
        opt.step()
        
    torch_model.eval()
    with torch.no_grad():
        raw_p_torch = torch_model(X_te_t).squeeze().numpy()
        
    # Consensus Ensemble
    ensemble_prob = (0.45 * raw_p_lgb) + (0.35 * raw_p_xgb) + (0.20 * raw_p_torch)
    
    # Isotonic Calibration on In-Sample -> Transform Out-of-Sample
    # Rank Percentile Mapping for Burst Mode
    # Convert probabilities to empirical percentiles (0 to 100%)
    from scipy.stats import rankdata
    calibrated_conf = rankdata(ensemble_prob) / len(ensemble_prob)
    
    rows = []
    for conf_th in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = calibrated_conf >= conf_th
        cnt = np.sum(mask)
        if cnt > 0:
            actual_wins = np.sum(y_test[mask] == 1)
            wr = (actual_wins / cnt) * 100.0
            pnl = test["trade_ret"].values[mask]
            cum_ret = np.sum(pnl) * 100.0
            win_p = pnl[pnl > 0].sum()
            loss_p = abs(pnl[pnl <= 0].sum())
            pf = (win_p / loss_p) if loss_p > 0 else 99.0
            rows.append({
                "Confidence Gate": f"Top {(1.0-conf_th)*100:.0f}% Signals (Conf >= {conf_th:.2f})",
                "Trades": cnt,
                "Win Rate (%)": f"{wr:.1f}%",
                "Profit Factor": f"{pf:.2f}",
                "Unseen Return": f"{cum_ret:+.1f}%"
            })
            
    print("=" * 85)
    print(f"  BURST QUANT AI ENGINE: {ticker.upper()} [{tf.upper()}] (100% UNSEEN FORWARD TEST)")
    print(f"  Total Test Bars: {len(test)} | Target: +1.8x ATR | SL: -1.2x ATR (RR = 1.5)")
    print("=" * 85)
    res_df = pd.DataFrame(rows)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    run_burst_optimization("PCJEWELLER", "1h")
    run_burst_optimization("SUZLON", "1h")
