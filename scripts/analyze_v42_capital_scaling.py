import sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

import importlib.util
spec = importlib.util.spec_from_file_location('v42', str(ROOT / 'strategies/V42/run_v42.py'))
mod_v42 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod_v42)

# Load 2026 data
p_2026 = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
p_highbeta = pickle.load(open(ROOT / 'data/unseen_high_beta_10_15m_1y.pkl', 'rb'))
p_unseen10 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))
pool_2026 = {**p_2026, **p_highbeta, **p_unseen10}

V29_30_UNIVERSE = {
    "J&KBANK", "UTKARSHBNK", "MANALIPETC", "IDEA", "PCJEWELLER",
    "NIACL", "MTNL", "IFCI", "TEXRAIL", "SWSOLAR",
    "TTML", "RENUKA", "RCF", "NBCC", "JYOTHYLAB",
    "RBLBANK", "ELECTCAST", "PNB", "GSFC", "TATASTEEL",
    "SUZLON", "CENTRALBK", "IOB", "IRCON", "NMDC",
    "RITES", "NCC", "ZEEL", "DELTACORP", "RPOWER"
}

# 1. Run 2026 with different initial capital: ₹500, ₹5,000, ₹10,000, ₹50,000
print("=" * 80)
print("TESTING IMPACT OF STARTING CAPITAL ON V42 (SAME IDENTICAL ENGINE)")
print("=" * 80)

# Let's inspect run_simulation in V42 to see how capital scales
# In run_v42.py: initial_equity = 500.0 is inside.
# Let's see the trades of V42 in 2026
cap_2026, trades_2026 = mod_v42.run_simulation(pool_2026, '2026-01-01', '2026-12-31', 'V42_2026', universe_filter=V29_30_UNIVERSE)

df_t = pd.DataFrame(trades_2026)
print(f"\nV42 2026 Breakdown:")
print(f"Total Trades: {len(df_t)}")
print(f"Net PnL: ₹{df_t['net_pnl'].sum():.2f} (Starting Capital: ₹500.00)")
print(f"Return on Capital: {df_t['net_pnl'].sum() / 500.0 * 100.0:.1f}%")

# Capital allocation analysis:
print(f"\nPosition Sizing & Capital Allocation Analysis:")
print(f"Min Shares traded: {df_t['shares'].min()} | Max Shares traded: {df_t['shares'].max()} | Median Shares: {df_t['shares'].median():.0f}")
print(f"Min Trade Net PnL: ₹{df_t['net_pnl'].min():.2f} | Max Trade Net PnL: ₹{df_t['net_pnl'].max():.2f}")
print(f"Top 5 Winning Trades:")
print(df_t.sort_values(by='net_pnl', ascending=False)[['symbol', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'shares', 'net_pnl', 'exit_reason']].head(5).to_string(index=False))

print(f"\nTop 5 Losing Trades:")
print(df_t.sort_values(by='net_pnl', ascending=True)[['symbol', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'shares', 'net_pnl', 'exit_reason']].head(5).to_string(index=False))
