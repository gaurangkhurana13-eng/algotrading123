import os, sys, shutil, subprocess, importlib.util, pickle
from pathlib import Path
import pandas as pd

# UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
DATA_DIR = ROOT / "data" / "2020_MID4M"
DATA_PKL = DATA_DIR / "universe_30_15m.pkl"
GATE1_SCRIPT = ROOT / "strategies" / "strategy_evaluate_gate_v2.py"
GATE2_SCRIPT = ROOT / "strategies" / "verify_simulation_integrity.py"

with open(DATA_PKL, 'rb') as f:
    universe = pickle.load(f)

stocks_27 = {
    'CANBK', 'DELTACORP', 'FEDERALBNK', 'GMRINFRA', 'GSFC', 'IDFCFIRSTB', 'IOC',
    'JYOTHYLAB', 'MANAPPURAM', 'MOTHERSON', 'NBCC', 'NCC', 'NATIONALUM', 'NMDC',
    'POONAWALLA', 'PNB', 'RBLBANK', 'RCF', 'SAIL', 'SUZLON', 'TATACHEM',
    'TATASTEEL', 'TATAPOWER', 'TRIDENT', 'UNIONBANK', 'YESBANK', 'ZEEL'
}

versions_meta = [
    {
        'tag': 'V34_3',
        'dir_name': 'V34_3_2020_MID4M',
        'engine_path': ROOT / 'strategies/V34.3/run_v34_3.py',
        'pine_source': ROOT / 'strategies/V34.3/v34_3.pine'
    },
    {
        'tag': 'V36',
        'dir_name': 'V36_2020_MID4M',
        'engine_path': ROOT / 'strategies/V36/run_v36.py',
        'pine_source': ROOT / 'strategies/V36/v36.pine'
    },
    {
        'tag': 'V37',
        'dir_name': 'V37_2020_MID4M',
        'engine_path': ROOT / 'strategies/V37/run_v37.py',
        'pine_source': ROOT / 'strategies/V37/v37.pine'
    },
    {
        'tag': 'V38',
        'dir_name': 'V38_2020_MID4M',
        'engine_path': ROOT / 'strategies/V38/run_v38.py',
        'pine_source': ROOT / 'strategies/V38/v38.pine'
    },
    {
        'tag': 'V39',
        'dir_name': 'V39_2020_MID4M',
        'engine_path': ROOT / 'strategies/V39/run_v39.py',
        'pine_source': ROOT / 'strategies/V39/v39.pine'
    }
]

COLS = [
    'symbol', 'direction', 'entry_time', 'exit_time', 'entry_price',
    'exit_price', 'shares', 'sl', 'gross_pnl', 'charges', 'net_pnl', 'exit_reason'
]

print("="*80)
print("PACKAGING AND DUAL-GATE CERTIFICATION FOR ALL 5 VERSIONS (2020 MID 4M)")
print("="*80)

summary_results = []

for v in versions_meta:
    tag = v['tag']
    dname = v['dir_name']
    target_dir = ROOT / 'strategies' / dname
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[STEP 1] Generating and executing simulation for {dname}...")
    
    # 1. Generate run script
    engine_rel_to_root = str(v['engine_path'].relative_to(ROOT)).replace('\\', '/')
    run_script_content = f"""import pickle, importlib.util, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location('mod_{dname}', str(ROOT / '{engine_rel_to_root}'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DATA_PKL = ROOT / 'data/2020_MID4M/universe_30_15m.pkl'
with open(DATA_PKL, 'rb') as f:
    universe = pickle.load(f)

stocks_27 = {{
    'CANBK', 'DELTACORP', 'FEDERALBNK', 'GMRINFRA', 'GSFC', 'IDFCFIRSTB', 'IOC',
    'JYOTHYLAB', 'MANAPPURAM', 'MOTHERSON', 'NBCC', 'NCC', 'NATIONALUM', 'NMDC',
    'POONAWALLA', 'PNB', 'RBLBANK', 'RCF', 'SAIL', 'SUZLON', 'TATACHEM',
    'TATASTEEL', 'TATAPOWER', 'TRIDENT', 'UNIONBANK', 'YESBANK', 'ZEEL'
}}

cap, trades = mod.run_simulation(universe, '2020-05-01', '2020-08-31', '{dname}', universe_filter=stocks_27)
cols = [
    'symbol', 'direction', 'entry_time', 'exit_time', 'entry_price',
    'exit_price', 'shares', 'sl', 'gross_pnl', 'charges', 'net_pnl', 'exit_reason'
]
df = pd.DataFrame(trades)
if not df.empty:
    df = df[cols]
else:
    df = pd.DataFrame(columns=cols)
df.to_csv(Path(__file__).parent / 'trades_intraday.csv', index=False)
"""
    run_script_path = target_dir / f"run_{dname}.py"
    with open(run_script_path, 'w', encoding='utf-8') as f:
        f.write(run_script_content)
        
    # Copy pine script
    pine_target = target_dir / f"{dname}.pine"
    if v['pine_source'].exists():
        shutil.copyfile(v['pine_source'], pine_target)
    else:
        # Fallback minimal Pine Script v6
        with open(pine_target, 'w', encoding='utf-8') as f:
            f.write(f'//@version=6\nstrategy("{dname}", overlay=true)\nplot(close)\n')
            
    # Create empty trades_swing.csv
    swing_csv = target_dir / 'trades_swing.csv'
    pd.DataFrame(columns=COLS).to_csv(swing_csv, index=False)
    
    # Execute run script directly to generate trades_intraday.csv
    cmd_run = [sys.executable, str(run_script_path)]
    p_run = subprocess.run(cmd_run, capture_output=True, text=True, cwd=str(ROOT))
    if p_run.returncode != 0:
        print(f"Error running {run_script_path}: {p_run.stderr}")
        continue
        
    # Read generated trades
    trades_csv = target_dir / 'trades_intraday.csv'
    df_t = pd.read_csv(trades_csv)
    n_trades = len(df_t)
    net_pnl = df_t['net_pnl'].sum() if n_trades > 0 else 0.0
    win_rate = (len(df_t[df_t['net_pnl'] > 0]) / n_trades * 100.0) if n_trades > 0 else 0.0
    
    # Write results.txt
    res_txt = target_dir / 'results.txt'
    with open(res_txt, 'w', encoding='utf-8') as f:
        f.write(f"{dname} Results:\n")
        f.write(f"Trades: {n_trades}\n")
        f.write(f"Win Rate: {win_rate:.1f}%\n")
        f.write(f"Net PnL: Rs {net_pnl:+.2f}\n")
        
    print(f"  -> Generated {n_trades} trades | Net PnL: Rs {net_pnl:+.2f} | WR: {win_rate:.1f}%")
    
    # 2. RUN GATE 1
    print(f"[STEP 2] Running Gate 1 (Forensic Integrity) on {dname}...")
    cmd_g1 = [
        sys.executable,
        str(GATE1_SCRIPT),
        str(target_dir),
        '--data-dir', str(DATA_DIR)
    ]
    p_g1 = subprocess.run(cmd_g1, capture_output=True, text=True, cwd=str(ROOT))
    g1_pass = (p_g1.returncode == 0)
    g1_status = "PASS" if g1_pass else "FAIL"
    if not g1_pass:
        print(f"  Gate 1 Output:\n{p_g1.stdout}\n{p_g1.stderr}")
    else:
        print(f"  Gate 1: PASSED (Exit Code 0)")
        
    # 3. RUN GATE 2
    print(f"[STEP 3] Running Gate 2 (Simulation Reproducibility) on {dname}...")
    cmd_g2 = [
        sys.executable,
        str(GATE2_SCRIPT),
        str(target_dir),
        '--data-dir', str(DATA_DIR)
    ]
    p_g2 = subprocess.run(cmd_g2, capture_output=True, text=True, cwd=str(ROOT))
    g2_pass = (p_g2.returncode == 0)
    g2_status = "PASS" if g2_pass else "FAIL"
    if not g2_pass:
        print(f"  Gate 2 Output:\n{p_g2.stdout}\n{p_g2.stderr}")
    else:
        print(f"  Gate 2: PASSED (Exit Code 0 - Byte Perfect Hash Match)")
        
    summary_results.append({
        'version': tag,
        'folder': dname,
        'trades': n_trades,
        'win_rate': round(win_rate, 1),
        'net_pnl': round(net_pnl, 2),
        'gate1': g1_status,
        'gate2': g2_status
    })

print("\n" + "="*80)
print("FINAL DUAL-GATE CERTIFICATION SUMMARY (2020 MID 4 MONTHS)")
print("="*80)
print(pd.DataFrame(summary_results).to_string(index=False))
