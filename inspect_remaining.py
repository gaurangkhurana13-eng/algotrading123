import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT / "strategies"))
from framework import calculate_atr
from strategies.V36.run_v36 import (
    compute_vwap_shifted, compute_er10, compute_adx_di
)

with open(ROOT / "data" / "2022-23" / "universe_30_15m.pkl", "rb") as f:
    p2022 = pickle.load(f)

df_trades = pd.read_csv(ROOT / "strategies" / "V36" / "fold_2022" / "trades_intraday.csv")

for idx, r in df_trades.iterrows():
    sym = r['symbol']
    t_str = r['entry_time']
    k = sym + ".NS" if sym + ".NS" in p2022 else sym
    df = p2022[k]
    ts = pd.Timestamp(t_str)
    
    v = df["volume"]
    v_sma20 = v.rolling(20).mean()
    rvol = v / (v_sma20 + 1e-9)
    rvol_val = rvol.loc[ts]
    
    if rvol_val < 1.50:
        c, h, l, o = df["close"], df["high"], df["low"], df["open"]
        macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        sig = macd.ewm(span=9).mean()
        macd_cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
        er10 = compute_er10(c)
        adx, pdi, mdi = compute_adx_di(df, 10)
        ema20 = c.ewm(span=20).mean()
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(7).mean()
        loss = (-delta.clip(upper=0)).rolling(7).mean()
        rsi7 = 100.0 - (100.0 / (1.0 + gain / (loss + 1e-9)))
        is_t1a = bool(macd_cross_up.loc[ts] and macd.loc[ts] > 0 and c.loc[ts] > ema20.loc[ts] and rsi7.loc[ts] > 50.0 and adx.loc[ts] >= 18.0 and pdi.loc[ts] > mdi.loc[ts] and er10.loc[ts] >= 0.35 and rvol.loc[ts] >= 1.20)
        
        # Check t3 squeeze
        atr14 = calculate_atr(df, 14)
        vwap_lag = compute_vwap_shifted(df)
        h_box = h.rolling(4).max().shift(1)
        l_box = l.rolling(4).min().shift(1)
        box_range = h_box - l_box
        ema9 = c.ewm(span=9).mean()
        is_t3 = bool((box_range.loc[ts] <= 1.65 * atr14.loc[ts]) and (l_box.loc[ts] >= vwap_lag.loc[ts] - 0.15 * atr14.loc[ts]) and (c.loc[ts] > h_box.loc[ts]) and (c.shift(1).loc[ts] <= h_box.loc[ts]) and (c.loc[ts] > ema9.loc[ts]) and (rvol.loc[ts] >= 1.30) and (((c.loc[ts] - l.loc[ts]) / (h.loc[ts] - l.loc[ts] + 1e-6)) >= 0.60) and (macd.loc[ts] >= sig.loc[ts]) and (rsi7.loc[ts] >= 54.0) and (rsi7.loc[ts] <= 78.0))

        # Check t2
        sup_prev = np.maximum(ema9.shift(1), vwap_lag) + (0.22 * atr14)
        l_min2 = np.minimum(l.shift(1), l.shift(2))
        v_min2 = np.minimum(v.shift(1), v.shift(2))
        v_max2_prev = np.maximum(v.shift(2), v.shift(3))
        is_t2 = bool((macd.loc[ts] > sig.loc[ts]) and (macd.loc[ts] > 0) and (pdi.loc[ts] > mdi.loc[ts]) and (c.loc[ts] > ema9.loc[ts]) and (l_min2.loc[ts] <= sup_prev.loc[ts]) and (v_min2.loc[ts] < v_max2_prev.loc[ts]) and (v.loc[ts] >= 1.12 * v.shift(1).loc[ts]) and (rvol.loc[ts] >= 1.15) and (c.loc[ts] > h.shift(1).loc[ts]) and (c.loc[ts] > o.loc[ts]) and (c.shift(1).loc[ts] <= h.shift(2).loc[ts]) and (rsi7.loc[ts] >= 52.0) and (rsi7.loc[ts] <= 75.0))

        if not is_t1a and not is_t2:
            print(f"Row {idx} ({sym} at {t_str}, RVOL={rvol_val:.2f}): is_t1a={is_t1a}, is_t2={is_t2}, is_t3={is_t3}")
