import pickle
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    d_beta = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
    d_unseen = pickle.load(f)

all_pools = {}
all_pools.update(univ)
all_pools.update(d_beta)
all_pools.update(d_unseen)

tickers_core = [
    "PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS", "RITES.NS",
    "IDEA.NS", "MTNL.NS", "ZEEL.NS", "UTKARSHBNK.NS", "IFCI.NS", "NCC.NS"
]

print("=== MONTHLY PRICE & VOL SPIKES FOR ALL 10 STOCKS ===")
for t in tickers_core:
    df = all_pools[t].copy()
    df["month"] = df.index.to_period("M")
    m_agg = df.groupby("month").agg(
        low=("low", "min"),
        high=("high", "max"),
        close=("close", "last"),
        vol=("volume", "sum")
    )
    # Check July vs June volume expansion
    vol_jun = m_agg.loc["2026-06", "vol"] if "2026-06" in m_agg.index else 1
    vol_jul = m_agg.loc["2026-07", "vol"] if "2026-07" in m_agg.index else 1
    vol_aug = m_agg.loc["2026-08", "vol"] if "2026-08" in m_agg.index else 1
    print(f"{t:>15} | Jun Vol: {vol_jun/1e6:>6.1f}M -> Jul Vol: {vol_jul/1e6:>7.1f}M ({(vol_jul/vol_jun):.1f}x) | Aug Vol: {vol_aug/1e6:>7.1f}M ({(vol_aug/vol_jun):.1f}x) | Aug High: Rs {m_agg.loc['2026-08', 'high']:.2f}")
