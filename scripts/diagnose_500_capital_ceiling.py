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

cap_2026, trades_2026 = mod_v42.run_simulation(pool_2026, '2026-01-01', '2026-12-31', 'V42_2026', universe_filter=V29_30_UNIVERSE)
df_t = pd.DataFrame(trades_2026)

print("=" * 80)
print("DIAGNOSING EXACT QUANT BOTTLENECK ON ₹500 CAPITAL")
print("=" * 80)

# 1. Exit Reason distribution
print("\n1. Exit Reasons Distribution:")
print(df_t['exit_reason'].value_counts())

# 2. Breakdown of profitable trades
win_trades = df_t[df_t['net_pnl'] > 0]
loss_trades = df_t[df_t['net_pnl'] < 0]
be_trades = df_t[df_t['net_pnl'] == 0]

print(f"\n2. Trades Breakdown:")
print(f"Wins: {len(win_trades)} | Losses: {len(loss_trades)} | Scratch/BE: {len(be_trades)}")
print(f"Total Net PnL: ₹{df_t['net_pnl'].sum():.2f}")
print(f"Total Gross PnL: ₹{df_t['gross_pnl'].sum():.2f}")
print(f"Total Dhan Charges Paid: ₹{df_t['charges'].sum():.2f}")
print(f"Charges as % of Gross PnL: {df_t['charges'].sum() / df_t['gross_pnl'].sum() * 100:.1f}%")

# 3. Capital utilization per trade
df_t['notional_value'] = df_t['shares'] * df_t['entry_price']
df_t['margin_used'] = df_t['notional_value'] / 5.0
print(f"\n3. Margin & Notional Utilization per Trade:")
print(f"Mean Margin Used: ₹{df_t['margin_used'].mean():.2f} (out of ₹500)")
print(f"Min Margin Used:  ₹{df_t['margin_used'].min():.2f}")
print(f"Max Margin Used:  ₹{df_t['margin_used'].max():.2f}")
print(f"Mean Notional Traded: ₹{df_t['notional_value'].mean():.2f}")

# 4. Trades by month
df_t['month'] = pd.to_datetime(df_t['entry_time']).dt.month
print(f"\n4. Trades & Net PnL By Month in 2026:")
monthly = df_t.groupby('month').agg(trades=('net_pnl', 'count'), net=('net_pnl', 'sum'), win_rate=('net_pnl', lambda x: (x > 0).mean() * 100))
print(monthly.to_string())
