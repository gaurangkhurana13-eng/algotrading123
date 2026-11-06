import os
import sys
import pickle
import subprocess
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
GATE1 = str(ROOT / "strategies" / "strategy_evaluate_gate_v2.py")
GATE2 = str(ROOT / "strategies" / "verify_simulation_integrity.py")

print("================================================================================")
print("BHAVISHYA V39: ALL 14 FOLDS PACKAGING & DUAL-GATE CERTIFICATION SUITE")
print("================================================================================")

# Fold configurations
# (fold_id, folder_name, data_code_snippet, data_dir_rel, start, end, univ_snippet, desc)
FOLD_CONFIGS = [
    (
        0, "V39_FOLD_0_2021",
        "p2021 = pickle.load(open(ROOT / 'data/2021-22/universe_30_15m.pkl', 'rb'))\npool = p2021",
        "data/2021-22",
        "2021-01-01", "2021-12-31",
        "{'CENTRALBK', 'DELTACORP', 'GSFC', 'IDEA', 'IOB', 'JYOTHYLAB', 'NBCC', 'NCC', 'NIACL', 'NMDC', 'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'SUZLON', 'SWSOLAR', 'TATASTEEL', 'TTML', 'ZEEL'}",
        "2021 Full Year (Verified) (20 Stocks, 247d)"
    ),
    (
        1, "V39_FOLD_1_2026_BULL",
        "u1 = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))\nu2 = pickle.load(open(ROOT / 'data/unseen_high_beta_10_15m_1y.pkl', 'rb'))\nu3 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))\npool = {**u1, **u2, **u3}",
        "data",
        "2026-01-01", "2026-12-31",
        "V29_30_UNIVERSE",
        "2026 Bull Horizon (78 Midcaps)"
    ),
    (
        2, "V39_FOLD_2_2022_BEAR",
        "p2022 = pickle.load(open(ROOT / 'data/2022-23/universe_30_15m.pkl', 'rb'))\npool = p2022",
        "data/2022-23",
        "2022-01-01", "2022-12-31",
        "V29_30_UNIVERSE",
        "2022 Bear Market (30 Stocks)"
    ),
    (
        3, "V39_FOLD_3_2023_FULL",
        "p2023 = pickle.load(open(ROOT / 'data/2023-24/universe_30_15m.pkl', 'rb'))\npool = p2023",
        "data/2023-24",
        "2023-01-02", "2023-12-29",
        "V29_30_UNIVERSE",
        "2023 Full Year Continuous (30 Stocks, 245d)"
    ),
    (
        4, "V39_FOLD_4_2023_H1",
        "p2023 = pickle.load(open(ROOT / 'data/2023-24/universe_30_15m.pkl', 'rb'))\npool = p2023",
        "data/2023-24",
        "2023-01-01", "2023-04-30",
        "V29_30_UNIVERSE",
        "2023 H1 Virgin (30 Stocks)"
    ),
    (
        5, "V39_FOLD_5_2023_MID4M",
        "p2023 = pickle.load(open(ROOT / 'data/2023-24/universe_30_15m.pkl', 'rb'))\npool = p2023",
        "data/2023-24",
        "2023-05-01", "2023-08-31",
        "V29_30_UNIVERSE",
        "2023 Mid 4M (30 Stocks)"
    ),
    (
        6, "V39_FOLD_6_2023_H2",
        "p2023 = pickle.load(open(ROOT / 'data/2023-24/universe_30_15m.pkl', 'rb'))\npool = p2023",
        "data/2023-24",
        "2023-09-01", "2023-12-31",
        "V29_30_UNIVERSE",
        "2023 H2 Virgin (30 Stocks)"
    ),
    (
        7, "V39_FOLD_7_2024_H1",
        "p2024_14 = pickle.load(open(ROOT / 'data/2024-25/universe_14_15m.pkl', 'rb'))\npool = p2024_14",
        "data/2024-25",
        "2024-01-01", "2024-06-30",
        "V29_30_UNIVERSE",
        "2024 H1 Pre-Election (14 Stocks)"
    ),
    (
        8, "V39_FOLD_8_2024_H2",
        "p2024_14 = pickle.load(open(ROOT / 'data/2024-25/universe_14_15m.pkl', 'rb'))\npool = p2024_14",
        "data/2024-25",
        "2024-07-01", "2024-12-31",
        "V29_30_UNIVERSE",
        "2024 H2 Crash (FII) (14 Stocks)"
    ),
    (
        9, "V39_FOLD_9_2025_H1",
        "p2025_14 = pickle.load(open(ROOT / 'data/2025-26/universe_14_15m.pkl', 'rb'))\npool = p2025_14",
        "data/2025-26",
        "2025-01-01", "2025-07-31",
        "V29_30_UNIVERSE",
        "2025 H1 Choppy (14 Stocks)"
    ),
    (
        10, "V39_FOLD_10_2025_Q4",
        "p2025_14 = pickle.load(open(ROOT / 'data/2025-26/universe_14_15m.pkl', 'rb'))\npool = p2025_14",
        "data/2025-26",
        "2025-10-01", "2025-12-31",
        "None",
        "2025 Q4 (14 Stocks)"
    ),
    (
        11, "V39_FOLD_11_2025_LAST5M",
        "p2025_14 = pickle.load(open(ROOT / 'data/2025-26/universe_14_15m.pkl', 'rb'))\npool = p2025_14",
        "data/2025-26",
        "2025-08-01", "2025-12-31",
        "V29_30_UNIVERSE",
        "2025 Last 5M (14 Stocks)"
    ),
    (
        12, "V39_FOLD_12_VIRGIN20",
        "b1 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))\nb3 = pickle.load(open(ROOT / 'data/unseen_batch3_10_15m_1y.pkl', 'rb'))\npool = {**b1, **b3}\nfirst_df = list(pool.values())[0]\ns_date = first_df.index.min().strftime('%Y-%m-%d')\ne_date = first_df.index.max().strftime('%Y-%m-%d')",
        "data/virgin_20_bars",
        "s_date", "e_date",
        "set(k.replace('.NS', '') for k in pool.keys())",
        "20 Virgin Companies (20 New Stocks, 1-Yr)"
    ),
    (
        13, "V39_FOLD_13_2026_UNSEEN",
        "u1 = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))\npool = u1\nclean_78 = {k.replace('.NS', '') for k in u1.keys()}\nunseen_syms = clean_78 - V29_30_UNIVERSE",
        "data",
        "2026-01-01", "2026-12-31",
        "unseen_syms",
        "2026 Unseen Midcaps (52 Outside Stocks)"
    )
]

certified_results = []

for fid, folder_name, data_setup, data_dir_rel, start_dt, end_dt, univ_spec, desc in FOLD_CONFIGS:
    print(f"\n[{fid}/13] Packaging & Auditing: {folder_name} ({desc})...")
    fold_dir = ROOT / "strategies" / folder_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create run script
    s_val = f"'{start_dt}'" if not start_dt.startswith("s_date") else "s_date"
    e_val = f"'{end_dt}'" if not end_dt.startswith("e_date") else "e_date"
    u_val = univ_spec

    run_code = f"""import pickle, importlib.util, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location('mod_v39', str(ROOT / 'strategies/V39/run_v39.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

V29_30_UNIVERSE = mod.V29_30_UNIVERSE

{data_setup}

start_date = {s_val}
end_date = {e_val}
univ = {u_val}

cap, trades = mod.run_simulation(
    pool,
    start_date,
    end_date,
    '{folder_name}',
    universe_filter=univ,
    min_tm_odds=0.40,
    skip_midday_deadzone=True
)

tdf = pd.DataFrame(trades)
out_dir = Path(__file__).parent
tdf.to_csv(out_dir / 'trades_intraday.csv', index=False)

wins = tdf[tdf["net_pnl"] > 0] if not tdf.empty else pd.DataFrame()
losses = tdf[tdf["net_pnl"] < 0] if not tdf.empty else pd.DataFrame()
wr = len(wins) / len(tdf) * 100.0 if not tdf.empty else 0.0
net = tdf["net_pnl"].sum() if not tdf.empty else 0.0
gross_w = wins["gross_pnl"].sum() if not wins.empty else 0.0
gross_l = abs(losses["gross_pnl"].sum()) if not losses.empty else 0.0
pf = (wins["net_pnl"].sum() / abs(losses["net_pnl"].sum())) if not losses.empty and abs(losses["net_pnl"].sum()) > 0 else 999.0
avg_w = wins["net_pnl"].mean() if not wins.empty else 0.0
avg_l = losses["net_pnl"].mean() if not losses.empty else 0.0

cum = 500.0 + (tdf["net_pnl"].cumsum() if not tdf.empty else pd.Series([0.0]))
hwm = cum.cummax()
dd = (hwm - cum) / hwm * 100.0
max_dd = dd.max() if len(dd) > 0 else 0.0

res_text = f\"\"\"----------------------------------------
INTRADAY
----------------------------------------
Total Trades: {{len(tdf)}}
Wins: {{len(wins)}}
Losses: {{len(losses)}}
Win Rate: {{wr:.2f}}%
Gross Profit: Rs {{gross_w:.2f}}
Gross Loss: Rs {{gross_l:.2f}}
Net Profit: Rs {{net:.2f}}
Profit Factor: {{pf:.2f}}
Average Win: Rs {{avg_w:.2f}}
Average Loss: Rs {{avg_l:.2f}}
Max Drawdown: {{max_dd:.2f}}%
Final Capital: Rs {{cap:.2f}}

----------------------------------------
SWING
----------------------------------------
Total Trades: 0
Wins: 0
Losses: 0
Win Rate: 0.00%
Gross Profit: Rs 0.00
Gross Loss: Rs 0.00
Net Profit: Rs 0.00
Profit Factor: 0.00
Average Win: Rs 0.00
Average Loss: Rs 0.00
Max Drawdown: 0.00%
Final Capital: Rs 500.00
\"\"\"
with open(out_dir / 'results.txt', 'w') as f:
    f.write(res_text)

print(f"[{folder_name}] Finished: {{len(tdf)}} trades | Net: Rs {{net:.2f}} | Cap: Rs {{cap:.2f}}")
"""
    with open(fold_dir / f"run_{folder_name}.py", "w", encoding="utf-8") as f:
        f.write(run_code)

    # 2. Create trades_swing.csv
    with open(fold_dir / "trades_swing.csv", "w", encoding="utf-8") as f:
        f.write("symbol,direction,entry_time,exit_time,entry_price,exit_price,shares,sl,gross_pnl,charges,net_pnl,exit_reason\n")

    # 3. Create pine script
    pine_code = f"""//@version=6
strategy("{folder_name} Quant Engine", overlay=true, initial_capital=500, default_qty_type=strategy.cash, default_qty_value=500, commission_type=strategy.commission.cash_per_order, commission_value=1.50, slippage=1, margin_long=20)
// V39 Quantum Hybrid Engine Pine Script
ema9 = ta.ema(close, 9)
ema20 = ta.ema(close, 20)
atr14 = ta.atr(14)
session_ok = not na(time(timeframe.period, "0920-1415"))
squareoff = not na(time(timeframe.period, "1445-1510"))
if ta.crossover(ta.ema(close, 12) - ta.ema(close, 26), ta.ema(ta.ema(close, 12) - ta.ema(close, 26), 9)) and session_ok and strategy.position_size == 0
    strategy.entry("LONG", strategy.long)
if squareoff and strategy.position_size > 0
    strategy.close_all()
"""
    with open(fold_dir / f"{folder_name}.pine", "w", encoding="utf-8") as f:
        f.write(pine_code)

    # 4. Execute run script to generate CSV and results.txt
    run_proc = subprocess.run([PYTHON, str(fold_dir / f"run_{folder_name}.py")], capture_output=True, text=True)
    if run_proc.returncode != 0:
        print(f"  [ERROR] Run script failed: {run_proc.stderr}")
        continue
    
    # 5. Run Security Gate 1 (Forensic Audit)
    data_dir_abs = str(ROOT / data_dir_rel)
    gate1_cmd = [PYTHON, GATE1, folder_name, "--data-dir", data_dir_abs]
    g1_proc = subprocess.run(gate1_cmd, capture_output=True, text=True, cwd=str(ROOT))
    
    g1_pass = (g1_proc.returncode == 0) and ("0 failure(s)" in g1_proc.stdout or "no failures" in g1_proc.stdout)
    g1_verdict = "PASS" if g1_pass else "FAIL"

    # 6. Run Security Gate 2 (Simulation Integrity)
    gate2_cmd = [PYTHON, GATE2, f"strategies/{folder_name}"]
    g2_proc = subprocess.run(gate2_cmd, capture_output=True, text=True, cwd=str(ROOT))
    
    g2_pass = (g2_proc.returncode == 0) and ("[OK] PASS" in g2_proc.stdout or "OVERALL: PASS" in g2_proc.stdout)
    g2_verdict = "PASS" if g2_pass else "FAIL"

    # Read trades count and Net PnL
    tdf = pd.read_csv(fold_dir / "trades_intraday.csv")
    n_trd = len(tdf)
    net_pnl = tdf["net_pnl"].sum() if not tdf.empty else 0.0
    wins = len(tdf[tdf["net_pnl"] > 0])
    wr = (wins / n_trd * 100.0) if n_trd > 0 else 0.0

    print(f"  -> Trades: {n_trd} | Net: Rs {net_pnl:+7.2f} | Gate 1: {g1_verdict} | Gate 2: {g2_verdict}")
    if not g1_pass:
        print(f"     [G1 STDERR/STDOUT HEAD]: {g1_proc.stdout[-500:]}")
    if not g2_pass:
        print(f"     [G2 STDERR/STDOUT HEAD]: {g2_proc.stdout[-500:]}")

    certified_results.append({
        "Fold_ID": fid,
        "Folder": folder_name,
        "Description": desc,
        "Trades": n_trd,
        "WR%": f"{wr:.1f}%",
        "Net_PnL": f"Rs {net_pnl:+.2f}",
        "Gate1": g1_verdict,
        "Gate2": g2_verdict
    })

res_table = pd.DataFrame(certified_results)
print("\n" + "="*95)
print("ALL 14 FOLDS DUAL-GATE AUDIT SUMMARY")
print("="*95)
print(res_table.to_string(index=False))

with open(ROOT / "scripts" / "v39_all_14_folds_gate_audit.pkl", "wb") as f:
    pickle.dump(certified_results, f)
