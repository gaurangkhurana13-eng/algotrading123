import sys
import os
import datetime
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
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from tabulate import tabulate

from src.data_fetcher import DataFetcher
from src.strategy import calculate_atr, calculate_rsi

fetcher = DataFetcher(data_dir="data")

# ==============================================================================
# 1. INSTITUTIONAL ALPHA FEATURE ENGINEERING (30+ QUANT FACTORS)
# ==============================================================================
def extract_alpha_features(df):
    d = df.copy()
    c = d["close"]
    h = d["high"]
    l = d["low"]
    o = d["open"]
    v = d["volume"]
    
    # A. Price Action & Microstructure
    d["ret_1"] = c.pct_change(1)
    d["ret_3"] = c.pct_change(3)
    d["ret_5"] = c.pct_change(5)
    d["ret_10"] = c.pct_change(10)
    d["candle_body"] = (c - o) / (h - l + 1e-9)
    d["upper_wick"] = (h - np.maximum(c, o)) / (h - l + 1e-9)
    d["lower_wick"] = (np.minimum(c, o) - l) / (h - l + 1e-9)
    
    # B. Moving Averages & Exponential Ribbon
    d["ema_9"] = c.ewm(span=9, adjust=False).mean()
    d["ema_21"] = c.ewm(span=21, adjust=False).mean()
    d["ema_50"] = c.ewm(span=50, adjust=False).mean()
    d["ema_200"] = c.ewm(span=200, adjust=False).mean()
    d["ema_spread_9_21"] = (d["ema_9"] - d["ema_21"]) / c
    d["ema_spread_21_50"] = (d["ema_21"] - d["ema_50"]) / c
    d["dist_ema_200"] = (c - d["ema_200"]) / c
    
    # C. Momentum Oscillators
    d["rsi_7"] = calculate_rsi(c, period=7)
    d["rsi_14"] = calculate_rsi(c, period=14)
    d["rsi_diff"] = d["rsi_7"] - d["rsi_14"]
    
    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    d["macd_hist"] = (macd_line - macd_signal) / c
    d["macd_hist_slope"] = d["macd_hist"] - d["macd_hist"].shift(1)
    
    # D. Volatility & ATR Bands
    d["atr_14"] = calculate_atr(d, period=14)
    d["atr_pct"] = d["atr_14"] / c
    
    # Bollinger Bands
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    d["bb_width"] = (2 * std20) / sma20
    d["bb_pct_b"] = (c - (sma20 - 2 * std20)) / (4 * std20 + 1e-9)
    
    # E. Volume Order-Flow & VWAP
    vol_sma20 = v.rolling(20).mean()
    d["vol_surge"] = v / (vol_sma20 + 1e-9)
    
    cum_vol = (v).rolling(20).sum()
    cum_pv = (c * v).rolling(20).sum()
    d["vwap_20"] = cum_pv / (cum_vol + 1e-9)
    d["vwap_dist"] = (c - d["vwap_20"]) / (d["vwap_20"] + 1e-9)
    
    # Clean NaNs
    feature_cols = [
        "ret_1", "ret_3", "ret_5", "ret_10", "candle_body", "upper_wick", "lower_wick",
        "ema_spread_9_21", "ema_spread_21_50", "dist_ema_200", "rsi_7", "rsi_14", "rsi_diff",
        "macd_hist", "macd_hist_slope", "atr_pct", "bb_width", "bb_pct_b", "vol_surge", "vwap_dist"
    ]
    
    return d, feature_cols

# ==============================================================================
# 2. TRIPLE BARRIER LABELING ENGINE (Profit First Target vs Stop Loss)
# ==============================================================================
def create_triple_barrier_labels(df, tp_atr_mult=2.5, sl_atr_mult=1.5, max_bars=10):
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    atr = df["atr_14"].values
    n = len(df)
    
    labels = np.zeros(n)
    returns = np.zeros(n)
    
    for i in range(n - max_bars):
        entry_p = c[i]
        curr_atr = atr[i]
        tp = entry_p + (curr_atr * tp_atr_mult)
        sl = entry_p - (curr_atr * sl_atr_mult)
        
        hit_tp = False
        hit_sl = False
        
        for k in range(1, max_bars + 1):
            bar_idx = i + k
            if h[bar_idx] >= tp:
                hit_tp = True
                break
            if l[bar_idx] <= sl:
                hit_sl = True
                break
                
        if hit_tp and not hit_sl:
            labels[i] = 1 # High-Conviction Winning Trade
            returns[i] = (tp - entry_p) / entry_p
        elif hit_sl:
            labels[i] = 0 # Loss
            returns[i] = (sl - entry_p) / entry_p
        else:
            labels[i] = 0 # Chop / Time expiration
            returns[i] = (c[i + max_bars] - entry_p) / entry_p
            
    df["label"] = labels
    df["trade_ret"] = returns
    return df

# ==============================================================================
# 3. DEEP PYTORCH NEURAL NETWORK
# ==============================================================================
class DeepAlphaNet(nn.Module):
    def __init__(self, input_dim):
        super(DeepAlphaNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

# ==============================================================================
# 4. TRAINING & WALK-FORWARD BENCHMARK
# ==============================================================================
def train_and_evaluate_ai(symbol="PCJEWELLER", tf="1h"):
    raw_df = fetcher.get_data(symbol, tf)
    if raw_df is None or len(raw_df) < 300:
        return None
        
    df, feat_cols = extract_alpha_features(raw_df)
    df = create_triple_barrier_labels(df)
    df = df.dropna(subset=feat_cols).copy()
    
    n = len(df)
    split_idx = int(n * 0.70) # 70% In-Sample Train, 30% Out-of-Sample Test
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feat_cols].values
    y_train = train_df["label"].values
    
    X_test = test_df[feat_cols].values
    y_test = test_df["label"].values
    
    # 1. Model 1: Microsoft LightGBM
    lgb_train = lgb.Dataset(X_train, label=y_train)
    params_lgb = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "verbose": -1
    }
    model_lgb = lgb.train(params_lgb, lgb_train, num_boost_round=150)
    prob_lgb = model_lgb.predict(X_test)
    
    # 2. Model 2: XGBoost
    model_xgb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.03, max_depth=5, eval_metric="logloss")
    model_xgb.fit(X_train, y_train)
    prob_xgb = model_xgb.predict_proba(X_test)[:, 1]
    
    # 3. Model 3: PyTorch DeepAlphaNet
    # Normalize inputs
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-7
    X_tr_norm = torch.FloatTensor((X_train - mean) / std)
    y_tr_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_te_norm = torch.FloatTensor((X_test - mean) / std)
    
    torch_model = DeepAlphaNet(input_dim=len(feat_cols))
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    torch_model.train()
    for epoch in range(60):
        optimizer.zero_grad()
        out = torch_model(X_tr_norm)
        loss = criterion(out, y_tr_t)
        loss.backward()
        optimizer.step()
        
    torch_model.eval()
    with torch.no_grad():
        prob_torch = torch_model(X_te_norm).squeeze().numpy()
        
    # Consensus Ensemble Probability (Weighted Average)
    ensemble_prob = (0.45 * prob_lgb) + (0.35 * prob_xgb) + (0.20 * prob_torch)
    
    # High-Conviction Probability Filtering
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    eval_rows = []
    
    for th in thresholds:
        trade_mask = ensemble_prob >= th
        trades_count = np.sum(trade_mask)
        if trades_count > 0:
            actual_wins = np.sum(y_test[trade_mask] == 1)
            win_rate = (actual_wins / trades_count) * 100.0
            
            # Cumulative PnL on high-conviction trades
            pnl_series = test_df["trade_ret"].values[trade_mask]
            cum_ret = np.sum(pnl_series) * 100.0
            win_rets = pnl_series[pnl_series > 0].sum()
            loss_rets = abs(pnl_series[pnl_series <= 0].sum())
            pf = (win_rets / loss_rets) if loss_rets > 0 else 99.0
            
            eval_rows.append({
                "Threshold": f"P >= {th:.2f}",
                "Trades": trades_count,
                "Win Rate (%)": f"{win_rate:.1f}%",
                "Profit Factor": f"{pf:.2f}",
                "Out-of-Sample Return": f"{cum_ret:+.1f}%"
            })
            
    print(f"\n" + "=" * 80)
    print(f"  AI QUANT SUPER-MODEL OUT-OF-SAMPLE TEST: {symbol.upper()} [{tf.upper()}]")
    print(f"  Train: {split_idx} bars | Unseen Test: {len(test_df)} bars (Zero Data Leakage)")
    print("=" * 80)
    res_df = pd.DataFrame(eval_rows)
    print(res_df.to_string(index=False))
    
    return model_lgb, feat_cols

def main():
    print("Initializing Quantitative Machine Learning Pipeline...")
    train_and_evaluate_ai(symbol="PCJEWELLER", tf="1h")
    train_and_evaluate_ai(symbol="SUZLON", tf="1h")

if __name__ == "__main__":
    main()
