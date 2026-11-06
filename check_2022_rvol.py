import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT / "strategies"))
from framework import calculate_atr
from strategies.V36.run_v36 import (
    V29_30_UNIVERSE, compute_vwap_shifted, compute_er10, compute_adx_di
)

# Load 2022 data
with open(ROOT / "data" / "2022-23" / "universe_30_15m.pkl", "rb") as f:
    p2022 = pickle.load(f)

# Inspect how many bars in 2022 had t1_b with RVOL in [1.30, 1.50) vs RVOL >= 1.50
t1b_low_rvol_count = 0
t1b_high_rvol_count = 0

for k, df in p2022.items():
    c, h, l, v, o = df["close"], df["high"], df["low"], df["volume"], df["open"]
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()
    hist = macd - sig
    hist_prev = hist.shift(1)
    v_sma20 = v.rolling(20).mean()
    rvol = v / (v_sma20 + 1e-9)
    er10 = compute_er10(c)
    atr14 = calculate_atr(df, 14)
    ema9 = c.ewm(span=9).mean()
    ema20 = c.ewm(span=20).mean()
    ema50 = c.ewm(span=50).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(7).mean()
    loss = (-delta.clip(upper=0)).rolling(7).mean()
    rsi7 = 100.0 - (100.0 / (1.0 + gain / (loss + 1e-9)))

    h_max2 = np.maximum(h.shift(1), h.shift(2))
    h_max2_prev = np.maximum(h.shift(2), h.shift(3))

    cond_base = (
        (macd > sig) & (sig > 0) & (hist > hist_prev) &
        (c > h_max2) & (c.shift(1) <= h_max2_prev) &
        (ema9 > ema20) & (ema20 > ema50) &
        (rsi7 >= 54.0) & (rsi7 <= 76.0) &
        (er10 >= 0.40) &
        (((c - ema9) / (atr14 + 1e-9)) <= 1.80)
    )

    t1b_low = cond_base & (rvol >= 1.30) & (rvol < 1.50)
    t1b_high = cond_base & (rvol >= 1.50)

    t1b_low_rvol_count += t1b_low.sum()
    t1b_high_rvol_count += t1b_high.sum()

print(f"2022 Full Year (30 stocks across 1 year):")
print(f"  Bars where T1B fired with RVOL in [1.30, 1.50): {t1b_low_rvol_count}")
print(f"  Bars where T1B fired with RVOL >= 1.50:          {t1b_high_rvol_count}")
