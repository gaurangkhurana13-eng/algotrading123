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

# Let us check which trigger was true for each of the 35 trades
t1a_count = 0
t2_count = 0
t1b_only_count = 0

for idx, r in df_trades.iterrows():
    sym = r['symbol']
    t_str = r['entry_time']
    k = sym + ".NS" if sym + ".NS" in p2022 else sym
    df = p2022[k]
    ts = pd.Timestamp(t_str)
    
    c, h, l, v, o = df["close"], df["high"], df["low"], df["volume"], df["open"]
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()
    macd_cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    v_sma20 = v.rolling(20).mean()
    rvol = v / (v_sma20 + 1e-9)
    er10 = compute_er10(c)
    adx, pdi, mdi = compute_adx_di(df, 10)
    ema20 = c.ewm(span=20).mean()
    
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(7).mean()
    loss = (-delta.clip(upper=0)).rolling(7).mean()
    rsi7 = 100.0 - (100.0 / (1.0 + gain / (loss + 1e-9)))

    rvol_val = rvol.loc[ts]
    if rvol_val < 1.50:
        # Check if t1_a was true
        is_t1a = bool(macd_cross_up.loc[ts] and macd.loc[ts] > 0 and c.loc[ts] > ema20.loc[ts] and rsi7.loc[ts] > 50.0 and adx.loc[ts] >= 18.0 and pdi.loc[ts] > mdi.loc[ts] and er10.loc[ts] >= 0.35 and rvol.loc[ts] >= 1.20)
        
        # Check if t2 was true
        ema9 = c.ewm(span=9).mean()
        vwap_lag = compute_vwap_shifted(df)
        atr14 = calculate_atr(df, 14)
        sup_prev = np.maximum(ema9.shift(1), vwap_lag) + (0.22 * atr14)
        l_min2 = np.minimum(l.shift(1), l.shift(2))
        v_min2 = np.minimum(v.shift(1), v.shift(2))
        v_max2_prev = np.maximum(v.shift(2), v.shift(3))
        is_t2 = bool((macd.loc[ts] > sig.loc[ts]) and (macd.loc[ts] > 0) and (pdi.loc[ts] > mdi.loc[ts]) and (c.loc[ts] > ema9.loc[ts]) and (l_min2.loc[ts] <= sup_prev.loc[ts]) and (v_min2.loc[ts] < v_max2_prev.loc[ts]) and (v.loc[ts] >= 1.12 * v.shift(1).loc[ts]) and (rvol.loc[ts] >= 1.15) and (c.loc[ts] > h.shift(1).loc[ts]) and (c.loc[ts] > o.loc[ts]) and (c.shift(1).loc[ts] <= h.shift(2).loc[ts]) and (rsi7.loc[ts] >= 52.0) and (rsi7.loc[ts] <= 75.0))

        if is_t1a:
            t1a_count += 1
        elif is_t2:
            t2_count += 1
        else:
            t1b_only_count += 1

print(f"Of the 35 trades with RVOL < 1.50 in 2022:")
print(f"  Fired via Primary T1A (MACD cross): {t1a_count}")
print(f"  Fired via T2 (Structural Pullback): {t2_count}")
print(f"  Fired ONLY via T1B:                 {t1b_only_count}")
