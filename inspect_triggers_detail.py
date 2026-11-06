import pickle
import pandas as pd
import numpy as np

# Load data for 2026
with open(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\data\universe_78_15m_1y.pkl", "rb") as f:
    u1 = pickle.load(f)

# Let us check TEXRAIL on 2026-09-23 11:30:00, 2026-09-24 11:45:00, 2026-09-25 10:00:00
df = u1['TEXRAIL.NS']
c, h, l, v, o = df['close'], df['high'], df['low'], df['volume'], df['open']
macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
sig = macd.ewm(span=9).mean()
hist = macd - sig
macd_cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))

v_sma20 = v.rolling(20).mean()
rvol = v / (v_sma20 + 1e-9)

h_max2 = np.maximum(h.shift(1), h.shift(2))
h_max2_prev = np.maximum(h.shift(2), h.shift(3))

for ts_str in ['2026-09-23 11:30:00', '2026-09-24 11:45:00', '2026-09-25 10:00:00']:
    ts = pd.Timestamp(ts_str)
    print(f"\n--- {ts_str} ---")
    print(f"Close: {c.loc[ts]}, MACD: {macd.loc[ts]:.3f}, Sig: {sig.loc[ts]:.3f}, MACD Cross: {macd_cross_up.loc[ts]}")
    print(f"RVOL: {rvol.loc[ts]:.2f}")
    print(f"C > h_max2: {c.loc[ts] > h_max2.loc[ts]} (C={c.loc[ts]}, h_max2={h_max2.loc[ts]})")
