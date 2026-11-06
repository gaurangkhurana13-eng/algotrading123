import os
import sys
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure stdout uses UTF-8
sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr
from scripts.test_scanner_patterns_v42 import simulate_with_scanner_modes, V29_30_UNIVERSE, core_27

folds = [
    ("2026_BULL", "COMPOSITE_1", "2026-01-01", "2026-10-31", V29_30_UNIVERSE),
    ("2023_FULL", ROOT_DIR / "data/2023-24/universe_30_15m.pkl", "2023-01-02", "2023-12-29", V29_30_UNIVERSE),
    ("2022_BEAR", ROOT_DIR / "data/2022-23/universe_30_15m.pkl", "2022-01-01", "2022-12-31", V29_30_UNIVERSE),
    ("2021_FULL", ROOT_DIR / "data/2021-22/universe_30_15m.pkl", "2021-01-01", "2021-12-31", {'CENTRALBK', 'DELTACORP', 'GSFC', 'IDEA', 'IOB', 'JYOTHYLAB', 'NBCC', 'NCC', 'NIACL', 'NMDC', 'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'SUZLON', 'SWSOLAR', 'TATASTEEL', 'TTML', 'ZEEL'}),
    ("2020_FULL", ROOT_DIR / "data/2020_FULL/universe_30_15m.pkl", "2020-01-01", "2020-12-31", core_27),
]

def load_pool(spec):
    if spec == "COMPOSITE_1":
        u1 = pickle.load(open(ROOT_DIR / "data/universe_78_15m_1y.pkl", "rb"))
        u2 = pickle.load(open(ROOT_DIR / "data/unseen_high_beta_10_15m_1y.pkl", "rb"))
        u3 = pickle.load(open(ROOT_DIR / "data/unseen_10_15m_1y.pkl", "rb"))
        return {**u1, **u2, **u3}
    return pickle.load(open(spec, "rb"))

pools = {}
for label, path, st, ed, u_filt in folds:
    pools[label] = load_pool(path)

print("Analyzing trade attribution for V42_PLUS_SCANNER...")

all_trades = []
for label, path, st, ed, u_filt in folds:
    res = simulate_with_scanner_modes(
        raw_pool=pools[label],
        start=st,
        end=ed,
        universe_filter=u_filt,
        mode="v42_plus_scanner"
    )
    for t in res["trades_list"]:
        t["regime"] = label
        all_trades.append(t)

df_trades = pd.DataFrame(all_trades)
print(f"Total trades logged: {len(df_trades)}")

print("\n--- ATTRIBUTION BY PATTERN ---")
grouped = df_trades.groupby("pattern").agg(
    trades=("net_pnl", "count"),
    wins=("net_pnl", lambda x: (x > 0).sum()),
    win_rate=("net_pnl", lambda x: (x > 0).mean() * 100.0),
    net_pnl=("net_pnl", "sum"),
    gross_pnl=("gross_pnl", "sum"),
    charges=("charges", "sum"),
    avg_trade=("net_pnl", "mean"),
    profit_factor=("net_pnl", lambda x: round(x[x > 0].sum() / abs(x[x < 0].sum() + 1e-9), 2))
)
print(grouped.to_string())

print("\n--- ATTRIBUTION BY PATTERN AND REGIME ---")
reg_pat = df_trades.groupby(["pattern", "regime"]).agg(
    trades=("net_pnl", "count"),
    win_rate=("net_pnl", lambda x: round((x > 0).mean() * 100.0, 1)),
    net_pnl=("net_pnl", lambda x: round(x.sum(), 2)),
    pf=("net_pnl", lambda x: round(x[x > 0].sum() / abs(x[x < 0].sum() + 1e-9), 2))
)
print(reg_pat.to_string())
