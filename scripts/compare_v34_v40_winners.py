import os, sys, pickle, importlib.util
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from strategies.V40.run_v40 import run_simulation as run_v40_sim, V29_30_UNIVERSE

spec34 = importlib.util.spec_from_file_location("run_v34_3", ROOT / "strategies/V34.3/run_v34_3.py")
mod34 = importlib.util.module_from_spec(spec34)
spec34.loader.exec_module(mod34)
run_v34_sim = mod34.run_simulation

from scripts.test_v41_refinement import load_pool_by_spec
p1 = load_pool_by_spec('COMPOSITE_1')

_, t40 = run_v40_sim(p1, '2026-01-01', '2026-12-31', 'F1 V40', universe_filter=V29_30_UNIVERSE)
_, t34 = run_v34_sim(p1, '2026-01-01', '2026-12-31', 'F1 V34.3', universe_filter=V29_30_UNIVERSE)

df40 = pd.DataFrame(t40)
df34 = pd.DataFrame(t34)

print('V40 F1: Trades =', len(df40), 'Net PnL = Rs', df40['net_pnl'].sum())
print('V34.3 F1: Trades =', len(df34), 'Net PnL = Rs', df34['net_pnl'].sum())

df34_big = df34[df34['net_pnl'] > 50]
print(f"\nV34.3 BIG WINNERS (> Rs 50) count: {len(df34_big)}")
for _, r in df34_big.iterrows():
    m40 = df40[df40['entry_time'] == r['entry_time']]
    net40 = f"Rs {m40['net_pnl'].iloc[0]:.2f} (Exit: {m40['exit_reason'].iloc[0]})" if not m40.empty else 'NO ENTRY IN V40'
    print(f"{r['symbol']:12s} {r['entry_time']} | V34.3 Net: Rs {r['net_pnl']:7.2f} (Exit: {r['exit_reason']}) | V40: {net40}")
