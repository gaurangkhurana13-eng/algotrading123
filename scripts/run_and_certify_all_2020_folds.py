import os, sys, shutil, subprocess, importlib.util, pickle
from pathlib import Path
import pandas as pd

# UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
GATE1_SCRIPT = ROOT / "strategies" / "strategy_evaluate_gate_v2.py"
GATE2_SCRIPT = ROOT / "strategies" / "verify_simulation_integrity.py"

core_27 = {
    'CANBK', 'DELTACORP', 'FEDERALBNK', 'GMRINFRA', 'GSFC', 'IDFCFIRSTB', 'IOC',
    'JYOTHYLAB', 'MANAPPURAM', 'MOTHERSON', 'NBCC', 'NCC', 'NATIONALUM', 'NMDC',
    'POONAWALLA', 'PNB', 'RBLBANK', 'RCF', 'SAIL', 'SUZLON', 'TATACHEM',
    'TATASTEEL', 'TATAPOWER', 'TRIDENT', 'UNIONBANK', 'YESBANK', 'ZEEL'
}

folds = [
    {
        'tag': '2020_FIRST4M',
        'data_dir': ROOT / 'data/2020_FIRST4M',
        'start_date': '2020-01-01',
        'end_date': '2020-04-30',
        'desc': 'Jan-Apr 2020 (Crash Crucible)'
    },
    {
        'tag': '2020_LAST4M',
        'data_dir': ROOT / 'data/2020_LAST4M',
        'start_date': '2020-09-01',
        'end_date': '2020-12-31',
        'desc': 'Sep-Dec 2020 (Recovery Bull)'
    },
    {
        'tag': '2020_FULL',
        'data_dir': ROOT / 'data/2020_FULL',
        'start_date': '2020-01-01',
        'end_date': '2020-12-31',
        'desc': 'Full Year 2020 Continuous (250d)'
    }
]

versions_meta = [
    {
        'tag': 'V34_3',
        'engine_path': ROOT / 'strategies/V34.3/run_v34_3.py',
        'pine_source': ROOT / 'strategies/V34.3/v34_3.pine'
    },
    {
        'tag': 'V36',
        'engine_path': ROOT / 'strategies/V36/run_v36.py',
        'pine_source': ROOT / 'strategies/V36/v36.pine'
    },
    {
        'tag': 'V37',
        'engine_path': ROOT / 'strategies/V37/run_v37.py',
        'pine_source': ROOT / 'strategies/V37/v37.pine'
    },
    {
        'tag': 'V38',
        'engine_path': ROOT / 'strategies/V38/run_v38.py',
        'pine_source': ROOT / 'strategies/V38/v38.pine'
    },
    {
        'tag': 'V39',
        'engine_path': ROOT / 'strategies/V39/run_v39.py',
        'pine_source': ROOT / 'strategies/V39/v39.pine'
    }
]

COLS = [
    'symbol', 'direction', 'entry_time', 'exit_time', 'entry_price',
    'exit_price', 'shares', 'sl', 'gross_pnl', 'charges', 'net_pnl', 'exit_reason'
]

master_results = []

print("="*90)
print("STARTING 3-FOLD BATCH EXECUTION & DUAL-GATE CERTIFICATION ACROSS ALL 5 VERSIONS")
print("="*90)

for f_info in folds:
    fold_tag = f_info['tag']
    data_dir = f_info['data_dir']
    start_date = f_info['start_date']
    end_date = f_info['end_date']
    data_pkl = data_dir / "universe_30_15m.pkl"
    
    print(f"\n==================== FOLD: {fold_tag} ({f_info['desc']}) ====================")
    
    for v_info in versions_meta:
        v_tag = v_info['tag']
        dname = f"{v_tag}_{fold_tag}"
        target_dir = ROOT / 'strategies' / dname
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Write run_<dname>.py
        engine_rel = str(v_info['engine_path'].relative_to(ROOT)).replace('\\', '/')
        data_pkl_rel = str(data_pkl.relative_to(ROOT)).replace('\\', '/')
        
        run_content = f"""import pickle, importlib.util, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location('mod_{dname}', str(ROOT / '{engine_rel}'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DATA_PKL = ROOT / '{data_pkl_rel}'
with open(DATA_PKL, 'rb') as f:
    universe = pickle.load(f)

stocks_27 = {{
    'CANBK', 'DELTACORP', 'FEDERALBNK', 'GMRINFRA', 'GSFC', 'IDFCFIRSTB', 'IOC',
    'JYOTHYLAB', 'MANAPPURAM', 'MOTHERSON', 'NBCC', 'NCC', 'NATIONALUM', 'NMDC',
    'POONAWALLA', 'PNB', 'RBLBANK', 'RCF', 'SAIL', 'SUZLON', 'TATACHEM',
    'TATASTEEL', 'TATAPOWER', 'TRIDENT', 'UNIONBANK', 'YESBANK', 'ZEEL'
}}

cap, trades = mod.run_simulation(universe, '{start_date}', '{end_date}', '{dname}', universe_filter=stocks_27)
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
            f.write(run_content)
            
        # Copy pine script
        pine_target = target_dir / f"{dname}.pine"
        if v_info['pine_source'].exists():
            shutil.copyfile(v_info['pine_source'], pine_target)
        else:
            with open(pine_target, 'w', encoding='utf-8') as f:
                f.write(f'//@version=6\nstrategy("{dname}", overlay=true)\nplot(close)\n')
                
        # Empty swing trades
        pd.DataFrame(columns=COLS).to_csv(target_dir / 'trades_swing.csv', index=False)
        
        # 2. Run simulation
        p_run = subprocess.run([sys.executable, str(run_script_path)], capture_output=True, text=True, cwd=str(ROOT))
        if p_run.returncode != 0:
            print(f"[-] Execution error in {dname}: {p_run.stderr}")
            continue
            
        # Parse trades
        trades_csv = target_dir / 'trades_intraday.csv'
        df_t = pd.read_csv(trades_csv)
        n_trades = len(df_t)
        if n_trades > 0:
            win_trades = df_t[df_t['net_pnl'] > 0]
            loss_trades = df_t[df_t['net_pnl'] < 0]
            win_rate = (len(win_trades) / n_trades) * 100.0
            gross_pnl = df_t['gross_pnl'].sum()
            charges = df_t['charges'].sum()
            net_pnl = df_t['net_pnl'].sum()
            gross_wins = win_trades['net_pnl'].sum()
            gross_losses = abs(loss_trades['net_pnl'].sum())
            pf = (gross_wins / gross_losses) if gross_losses > 0 else 999.0
            
            # Max DD
            equity_curve = [500.0]
            cur = 500.0
            for pnl in df_t['net_pnl']:
                cur += pnl
                equity_curve.append(cur)
            peak = equity_curve[0]
            max_dd = 0.0
            for eq in equity_curve:
                if eq > peak: peak = eq
                dd = (peak - eq) / peak * 100.0
                if dd > max_dd: max_dd = dd
        else:
            win_rate = gross_pnl = charges = net_pnl = pf = max_dd = 0.0
            
        # Write results.txt
        with open(target_dir / 'results.txt', 'w', encoding='utf-8') as f:
            f.write(f"{dname} Results:\n")
            f.write(f"Trades: {n_trades}\n")
            f.write(f"Win Rate: {win_rate:.1f}%\n")
            f.write(f"Net PnL: Rs {net_pnl:+.2f}\n")
            
        # 3. GATE 1
        p_g1 = subprocess.run(
            [sys.executable, str(GATE1_SCRIPT), str(target_dir), '--data-dir', str(data_dir)],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        g1_pass = (p_g1.returncode == 0)
        
        # 4. GATE 2
        p_g2 = subprocess.run(
            [sys.executable, str(GATE2_SCRIPT), str(target_dir), '--data-dir', str(data_dir)],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        g2_pass = (p_g2.returncode == 0)
        
        status_line = f"[{dname:24s}] Trades: {n_trades:3d} | WR: {win_rate:5.1f}% | Net: Rs {net_pnl:+7.2f} | PF: {pf:4.2f} | MaxDD: {max_dd:4.1f}% | Gate 1: {'PASS' if g1_pass else 'FAIL'} | Gate 2: {'PASS' if g2_pass else 'FAIL'}"
        print(status_line)
        
        master_results.append({
            'fold': fold_tag,
            'version': v_tag,
            'folder': dname,
            'trades': n_trades,
            'win_rate': round(win_rate, 1),
            'gross_pnl': round(gross_pnl, 2),
            'charges': round(charges, 2),
            'net_pnl': round(net_pnl, 2),
            'pf': round(pf, 2),
            'max_dd': round(max_dd, 1),
            'gate1': 'PASS' if g1_pass else 'FAIL',
            'gate2': 'PASS' if g2_pass else 'FAIL'
        })

print("\n" + "="*90)
print("ALL RUNS COMPLETE. COMPILING MATRIX...")
print("="*90)
df_all = pd.DataFrame(master_results)
print(df_all.to_string(index=False))
