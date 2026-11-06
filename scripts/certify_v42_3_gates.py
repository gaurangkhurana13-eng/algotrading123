import os, sys, subprocess, shutil, pickle
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
DATA_DIR = ROOT / "data" / "2016_H1_UNSEEN"
GATE1_SCRIPT = ROOT / "strategies" / "strategy_evaluate_gate_v2.py"
GATE2_SCRIPT = ROOT / "strategies" / "verify_simulation_integrity.py"

target_dir = ROOT / "strategies" / "V42_3_CERT"
if target_dir.exists():
    shutil.rmtree(target_dir)
target_dir.mkdir(parents=True, exist_ok=True)

# Run V42.3 on 2016 H1 and save results
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))
from strategies.V42_3.run_v42_3 import run_simulation_v42_3

stocks_30_legacy = {
    'CANBK', 'PNB', 'BANKBARODA', 'SBIN', 'IOC',
    'BPCL', 'GAIL', 'COALINDIA', 'TATASTEEL', 'TATAPOWER',
    'TATAMOTORS', 'VEDL', 'HINDALCO', 'JINDALSTEL', 'JSWSTEEL',
    'BEL', 'ADANIPOWER', 'ADANIENT', 'RECLTD', 'PFC',
    'DLF', 'MOTHERSON', 'ITC', 'ICICIBANK', 'AXISBANK',
    'RELIANCE', 'INFY', 'SUNPHARMA', 'TECHM', 'CIPLA'
}

with open(DATA_DIR / "universe_30_15m.pkl", "rb") as f:
    pool_2016 = pickle.load(f)

cap, trades, wr, dd, pf = run_simulation_v42_3(
    pool_2016, "2016-01-01", "2016-06-30", "2016_H1", universe_filter=stocks_30_legacy, base_risk=25.0
)

net_pnl = cap - 500.0
wins = [t["net_pnl"] for t in trades if t["net_pnl"] > 0]
losses = [t["net_pnl"] for t in trades if t["net_pnl"] < 0]
avg_win = sum(wins)/len(wins) if wins else 0.0
avg_loss = sum(losses)/len(losses) if losses else 0.0
max_win = max(wins) if wins else 0.0
max_loss = min(losses) if losses else 0.0

# 1. Save trades.csv
df_trades = pd.DataFrame(trades)
df_trades.to_csv(target_dir / "trades.csv", index=False)

# 2. Save results.txt
with open(target_dir / "results.txt", "w", encoding="utf-8") as f:
    f.write(f"Total Trades: {len(trades)}\n")
    f.write(f"Win Rate: {wr:.2f}%\n")
    f.write(f"Net Profit: Rs {net_pnl:.2f}\n")
    f.write(f"Profit Factor: {pf:.2f}\n")
    f.write(f"Max Drawdown: {dd:.2f}%\n")
    f.write(f"Average Win: Rs {avg_win:.2f}\n")
    f.write(f"Average Loss: Rs -{abs(avg_loss):.2f}\n")
    f.write(f"Largest Win: Rs {max_win:.2f}\n")
    f.write(f"Largest Loss: Rs -{abs(max_loss):.2f}\n")

# 3. Save run.py for Gate 2
run_script_content = f'''
import sys, pickle
from pathlib import Path
ROOT = Path(r"{ROOT}")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))
from strategies.V42_3.run_v42_3 import run_simulation_v42_3

stocks_30_legacy = {
    'CANBK', 'PNB', 'BANKBARODA', 'SBIN', 'IOC',
    'BPCL', 'GAIL', 'COALINDIA', 'TATASTEEL', 'TATAPOWER',
    'TATAMOTORS', 'VEDL', 'HINDALCO', 'JINDALSTEL', 'JSWSTEEL',
    'BEL', 'ADANIPOWER', 'ADANIENT', 'RECLTD', 'PFC',
    'DLF', 'MOTHERSON', 'ITC', 'ICICIBANK', 'AXISBANK',
    'RELIANCE', 'INFY', 'SUNPHARMA', 'TECHM', 'CIPLA'
}

with open(Path(r"{DATA_DIR}") / "universe_30_15m.pkl", "rb") as f:
    pool = pickle.load(f)

cap, trades, wr, dd, pf = run_simulation_v42_3(
    pool, "2016-01-01", "2016-06-30", "2016_H1", universe_filter=stocks_30_legacy, base_risk=25.0
)
'''
with open(target_dir / "run.py", "w", encoding="utf-8") as f:
    f.write(run_script_content)

print("Running Gate 1 Evaluation...")
p_g1 = subprocess.run([sys.executable, str(GATE1_SCRIPT), str(target_dir), '--data-dir', str(DATA_DIR)],
                      capture_output=True, text=True)

print("Running Gate 2 Integrity Verification...")
p_g2 = subprocess.run([sys.executable, str(GATE2_SCRIPT), str(target_dir), '--data-dir', str(DATA_DIR)],
                      capture_output=True, text=True)

print("\n" + "=" * 80)
print("DUAL-GATE CERTIFICATION REPORT FOR V42.3")
print("=" * 80)
print(f"Gate 1 Returncode: {p_g1.returncode} ({'PASS' if p_g1.returncode == 0 else 'FAIL'})")
print(p_g1.stdout)
if p_g1.stderr:
    print("Gate 1 Errors:", p_g1.stderr)

print("-" * 80)
print(f"Gate 2 Returncode: {p_g2.returncode} ({'PASS' if p_g2.returncode == 0 else 'FAIL'})")
print(p_g2.stdout)
if p_g2.stderr:
    print("Gate 2 Errors:", p_g2.stderr)
print("=" * 80)
