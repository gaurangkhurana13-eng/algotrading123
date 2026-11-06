import pickle
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
data_path = ROOT / "data" / "2024-25" / "universe_30_15m.pkl"
with open(data_path, "rb") as f:
    pool = pickle.load(f)

# Test real Day Trader Skill Time Machine target/stop ratio
sample = pool["TATASTEEL"].loc["2024-01-01":"2024-12-31"].copy()
c = sample["close"]
h = sample["high"]
l = sample["low"]

# Next 8 bars max high vs min low
# Let's say entry at c, target = c + 1.2*atr, stop = c - 0.9*atr
atr14 = (h - l).rolling(14).mean()

# Let's compute forward target hit before stop hit
fwd_hit_target = pd.Series(False, index=c.index)
fwd_hit_stop = pd.Series(False, index=c.index)

for i in range(len(c) - 8):
    cur_c = c.iloc[i]
    cur_atr = atr14.iloc[i]
    target = cur_c + 1.2 * cur_atr
    stop = cur_c - 0.9 * cur_atr
    
    # check next 8 bars
    future_bars = sample.iloc[i+1 : i+9]
    hit_t = False
    hit_s = False
    for _, row in future_bars.iterrows():
        if row["low"] <= stop:
            hit_s = True
            break
        if row["high"] >= target:
            hit_t = True
            break
    fwd_hit_target.iloc[i] = hit_t
    fwd_hit_stop.iloc[i] = hit_s

# Rolling odds across past 40 bars (shifted by 8 to prevent lookahead)
# target_share = targets / (targets + stops)
past_targets = fwd_hit_target.astype(float).shift(8).rolling(40, min_periods=10).sum()
past_stops = fwd_hit_stop.astype(float).shift(8).rolling(40, min_periods=10).sum()
tm_odds = past_targets / (past_targets + past_stops + 1e-6)

print("TM Odds distribution:")
print(tm_odds.describe())
