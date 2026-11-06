import os, sys, pickle, subprocess, shutil
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from scripts.test_v43_synthesis import simulate_v43_era
from scripts.multi_slot_sim_engine import V29_30_UNIVERSE

GATE1_SCRIPT = ROOT / "strategies/strategy_evaluate_gate_v2.py"
GATE2_SCRIPT = ROOT / "strategies/verify_simulation_integrity.py"

cert_dir = ROOT / "strategies/certified/V43_CERT_2026"
cert_dir.mkdir(parents=True, exist_ok=True)

pool_2026 = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))

print("Running V43 on 2026 data...")
res = simulate_v43_era(pool_2026, "2026-01-01", "2026-10-29", universe_filter=V29_30_UNIVERSE, initial_equity=500.0, include_ma20_reclaim=True)

trades = res['trades']
print(f"2026 Trades: {len(trades)}, Net: Rs {res['net_pnl']:+.2f}, WR: {res['win_rate']:.1f}%, PF: {res['profit_factor']:.2f}, DD: {res['max_dd']:.1f}%")

cols = ['symbol', 'direction', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'shares', 'sl', 'gross_pnl', 'charges', 'net_pnl', 'exit_reason']
df = pd.DataFrame(trades)[cols]
df.to_csv(cert_dir / "trades_intraday.csv", index=False)
pd.DataFrame(columns=cols).to_csv(cert_dir / "trades_swing.csv", index=False)

with open(cert_dir / "results.txt", "w", encoding="utf-8") as rf:
    rf.write(f"V43_CERT_2026 Results:\nTrades: {len(trades)}\nWin Rate: {res['win_rate']:.1f}%\nNet PnL: Rs {res['net_pnl']:+.2f}\n")

# Copy pine
shutil.copyfile(ROOT / 'strategies/V43/v43.pine', cert_dir / "v43_cert_2026.pine")

# Write runner
runner_code = f"""import pickle, pandas as pd, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))
from scripts.test_v43_synthesis import simulate_v43_era
from scripts.multi_slot_sim_engine import V29_30_UNIVERSE

pool = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
res = simulate_v43_era(pool, "2026-01-01", "2026-10-29", universe_filter=V29_30_UNIVERSE, initial_equity=500.0, include_ma20_reclaim=True)
cols = ['symbol', 'direction', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'shares', 'sl', 'gross_pnl', 'charges', 'net_pnl', 'exit_reason']
df = pd.DataFrame(res['trades'])[cols]
df.to_csv(Path(__file__).parent / 'trades_intraday.csv', index=False)
"""
with open(cert_dir / "run_v43_cert_2026.py", "w", encoding="utf-8") as rf:
    rf.write(runner_code)

print("\n--- AUDITING GATE 1 ---")
p_g1 = subprocess.run([sys.executable, str(GATE1_SCRIPT), str(cert_dir), '--data-dir', str(ROOT / 'data')], capture_output=True, text=True, cwd=str(ROOT))
print("Gate 1 Return Code:", p_g1.returncode)
print("Gate 1 Summary Lines:")
for line in p_g1.stdout.splitlines():
    if "VERDICT" in line or "CHECK" in line or "[" in line and "]" in line:
        print(" ", line)

print("\n--- AUDITING GATE 2 ---")
p_g2 = subprocess.run([sys.executable, str(GATE2_SCRIPT), str(cert_dir), '--data-dir', str(ROOT / 'data')], capture_output=True, text=True, cwd=str(ROOT))
print("Gate 2 Return Code:", p_g2.returncode)
for line in p_g2.stdout.splitlines()[-10:]:
    print(" ", line)
