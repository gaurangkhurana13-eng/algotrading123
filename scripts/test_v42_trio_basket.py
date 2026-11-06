import sys, pickle
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from strategies.V42.run_v42 import run_simulation

pkl_path = ROOT / "data" / "universe_78_15m_1y.pkl"
with open(pkl_path, "rb") as f:
    pool = pickle.load(f)

# Filter pool to ONLY the trio basket
trio_pool = {k: v for k, v in pool.items() if any(s in k for s in ["PCJEWELLER", "SOUTHBANK", "SUZLON"])}
print(f"Trio Pool symbols: {list(trio_pool.keys())}")

cap, trades = run_simulation(trio_pool, "2026-01-01", "2026-10-29", "V42_TRIO_BASKET_2026")
print(f"\nV42 Trio Basket (15m) Results:")
print(f"Total Trades: {len(trades)}")
for t in trades:
    print(f"  {t['entry_time']} | {t['symbol']:<12} | In: {t['entry_price']} | Out: {t['exit_price']} | Net: Rs {t['net_pnl']:+6.2f} | {t['exit_reason']}")
