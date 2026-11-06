import sys, pickle
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from strategies.V42.run_v42 import run_simulation

# Check SUZLON and SOUTHBANK in universe_78_15m_1y.pkl
pkl_path = ROOT / "data" / "universe_78_15m_1y.pkl"
with open(pkl_path, "rb") as f:
    pool = pickle.load(f)

print("Checking SUZLON and SOUTHBANK in 15m pool:")
for sym in ["SUZLON.NS", "SOUTHBANK.NS", "PCJEWELLER.NS"]:
    if sym in pool:
        df = pool[sym]
        print(f"  {sym}: {len(df)} bars ({df.index.min()} to {df.index.max()})")
    else:
        print(f"  {sym}: NOT in pool")

# Let's run V42 simulation on SUZLON and SOUTHBANK specifically or check all trades taken by V42 in 2026
from strategies.V42.run_v42 import run_simulation
print("\nRunning V42 Turbo on full 2026 universe to inspect trades on SUZLON and SOUTHBANK:")
cap, trades = run_simulation(pool, "2026-01-01", "2026-10-29", "2026_FULL", universe_filter=None)

suzlon_trades = [t for t in trades if "SUZLON" in t["symbol"]]
southbank_trades = [t for t in trades if "SOUTHBANK" in t["symbol"]]
pcj_trades = [t for t in trades if "PCJEWELLER" in t["symbol"]]

print(f"\nV42 Trades in 2026 across entire 78-stock universe:")
print(f"  SUZLON trades     : {len(suzlon_trades)}")
for t in suzlon_trades:
    print(f"    {t['entry_time']} | In: {t['entry_price']} | Out: {t['exit_price']} | Qty: {t['shares']} | Net: {t['net_pnl']} | {t['exit_reason']}")

print(f"  SOUTHBANK trades  : {len(southbank_trades)}")
for t in southbank_trades:
    print(f"    {t['entry_time']} | In: {t['entry_price']} | Out: {t['exit_price']} | Qty: {t['shares']} | Net: {t['net_pnl']} | {t['exit_reason']}")

print(f"  PCJEWELLER trades : {len(pcj_trades)}")
for t in pcj_trades:
    print(f"    {t['entry_time']} | In: {t['entry_price']} | Out: {t['exit_price']} | Qty: {t['shares']} | Net: {t['net_pnl']} | {t['exit_reason']}")

