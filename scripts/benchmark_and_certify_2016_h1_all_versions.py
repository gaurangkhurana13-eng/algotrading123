import os, sys, shutil, subprocess, importlib.util, pickle
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

DATA_DIR = ROOT / "data" / "2016_H1_UNSEEN"
DATA_PKL = DATA_DIR / "universe_30_15m.pkl"
GATE1_SCRIPT = ROOT / "strategies" / "strategy_evaluate_gate_v2.py"
GATE2_SCRIPT = ROOT / "strategies" / "verify_simulation_integrity.py"

print("="*115)
print("BENCHMARK AND DUAL-GATE CERTIFICATION ACROSS ALL VERSIONS ON 2016 H1 UNSEEN DATA (CHINA CRASH + BREXIT)")
print("="*115)

with open(DATA_PKL, 'rb') as f:
    universe = pickle.load(f)

stocks_30_2016 = {
    'CANBK', 'PNB', 'BANKBARODA', 'SBIN', 'IOC',
    'BPCL', 'GAIL', 'COALINDIA', 'TATASTEEL', 'TATAPOWER',
    'TATAMOTORS', 'VEDL', 'HINDALCO', 'JINDALSTEL', 'JSWSTEEL',
    'BEL', 'ADANIPOWER', 'ADANIENT', 'RECLTD', 'PFC',
    'DLF', 'MOTHERSON', 'ITC', 'ICICIBANK', 'AXISBANK',
    'RELIANCE', 'INFY', 'SUNPHARMA', 'TECHM', 'CIPLA'
}

versions_meta = [
    {
        'tag': 'V34.3',
        'dir_name': 'V34_3_2016_H1',
        'engine_path': ROOT / 'strategies/V34.3/run_v34_3.py',
        'pine_source': ROOT / 'strategies/V34.3/v34_3.pine'
    },
    {
        'tag': 'V36',
        'dir_name': 'V36_2016_H1',
        'engine_path': ROOT / 'strategies/V36/run_v36.py',
        'pine_source': ROOT / 'strategies/V36/v36.pine'
    },
    {
        'tag': 'V37',
        'dir_name': 'V37_2016_H1',
        'engine_path': ROOT / 'strategies/V37/run_v37.py',
        'pine_source': ROOT / 'strategies/V37/v37.pine'
    },
    {
        'tag': 'V38',
        'dir_name': 'V38_2016_H1',
        'engine_path': ROOT / 'strategies/V38/run_v38.py',
        'pine_source': ROOT / 'strategies/V38/v38.pine'
    },
    {
        'tag': 'V39',
        'dir_name': 'V39_2016_H1',
        'engine_path': ROOT / 'strategies/V39/run_v39.py',
        'pine_source': ROOT / 'strategies/V39/v39.pine'
    },
    {
        'tag': 'V39.1',
        'dir_name': 'V39_1_2016_H1',
        'engine_path': ROOT / 'strategies/V39.1/run_v39_1.py',
        'pine_source': ROOT / 'strategies/V39.1/v39_1.pine'
    },
    {
        'tag': 'V40',
        'dir_name': 'V40_2016_H1',
        'engine_path': ROOT / 'strategies/V40/run_v40.py',
        'pine_source': ROOT / 'strategies/V40/v40.pine'
    },
    {
        'tag': 'V41',
        'dir_name': 'V41_2016_H1',
        'engine_path': ROOT / 'strategies/V41/run_v41.py',
        'pine_source': ROOT / 'strategies/V41/v41.pine'
    },
    {
        'tag': 'V42',
        'dir_name': 'V42_2016_H1',
        'engine_path': ROOT / 'strategies/V42/run_v42.py',
        'pine_source': ROOT / 'strategies/V42/v42.pine'
    }
]

COLS = [
    'symbol', 'direction', 'entry_time', 'exit_time', 'entry_price',
    'exit_price', 'shares', 'sl', 'gross_pnl', 'charges', 'net_pnl', 'exit_reason'
]

def eval_trades(df_t, initial_cap=500.0):
    n = len(df_t)
    if n > 0:
        win_trades = df_t[df_t['net_pnl'] > 0]
        loss_trades = df_t[df_t['net_pnl'] < 0]
        wr = (len(win_trades) / n) * 100.0
        net = df_t['net_pnl'].sum()
        gw = win_trades['net_pnl'].sum()
        gl = abs(loss_trades['net_pnl'].sum())
        pf = (gw / gl) if gl > 0 else 999.0
        avg_w = win_trades['net_pnl'].mean() if len(win_trades) > 0 else 0.0
        avg_l = abs(loss_trades['net_pnl'].mean()) if len(loss_trades) > 0 else 0.0
        
        eq = [initial_cap]
        cur = initial_cap
        for p in df_t['net_pnl']:
            cur += p
            eq.append(cur)
        pk = eq[0]
        mdd = 0.0
        for e in eq:
            if e > pk: pk = e
            dd = (pk - e) / pk * 100.0
            if dd > mdd: mdd = dd
    else:
        wr = net = pf = mdd = avg_w = avg_l = 0.0
    return n, round(wr, 1), round(net, 2), round(pf, 2), round(mdd, 1), round(avg_w, 2), round(avg_l, 2)

summary_results = []

for v in versions_meta:
    tag = v['tag']
    dname = v['dir_name']
    target_dir = ROOT / 'strategies' / dname
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n---> Benchmarking {tag} ({dname}) on H1 2016 Unseen Data...")
    
    # 1. Load module
    spec = importlib.util.spec_from_file_location(f"mod_{dname}", str(v['engine_path']))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    # 2. Run simulation
    cap, trades = mod.run_simulation(universe, '2016-01-01', '2016-06-30', dname, universe_filter=stocks_30_2016)
    df = pd.DataFrame(trades)
    n, wr, net, pf, dd, avg_w, avg_l = eval_trades(df, initial_cap=500.0)
    print(f"[{tag}] Trades: {n} | WinRate: {wr}% | Net PnL: Rs {net:+.2f} | Final Cap: Rs {500+net:.2f} | PF: {pf} | DD: {dd}% | AvgWin: Rs {avg_w} | AvgLoss: Rs {avg_l}")
    
    # 3. Save trades and results
    if not df.empty:
        df[COLS].to_csv(target_dir / 'trades_intraday.csv', index=False)
        charges_total = df['charges'].sum()
        gross_pnl = df['gross_pnl'].sum()
    else:
        pd.DataFrame(columns=COLS).to_csv(target_dir / 'trades_intraday.csv', index=False)
        charges_total = gross_pnl = 0.0
        
    pd.DataFrame(columns=COLS).to_csv(target_dir / 'trades_swing.csv', index=False)
    
    with open(target_dir / 'results.txt', 'w', encoding='utf-8') as rf:
        rf.write(f"{dname} Results:\nTrades: {n}\nWin Rate: {wr}%\nNet PnL: Rs {net:+.2f}\nFinal Capital: Rs {500+net:.2f}\nProfit Factor: {pf}\nMax Drawdown: {dd}%\nAverage Win: Rs {avg_w}\nAverage Loss: Rs {avg_l}\nGross PnL: Rs {gross_pnl:.2f}\nCharges: Rs {charges_total:.2f}\n")
        
    # Copy pine script
    pine_target = target_dir / f"{dname}.pine"
    if v['pine_source'].exists():
        shutil.copyfile(v['pine_source'], pine_target)
    else:
        with open(pine_target, 'w', encoding='utf-8') as pf:
            pf.write(f'//@version=6\nstrategy("{dname}", overlay=true)\nplot(close)\n')
            
    # 4. Write run script for Gate 2 verification
    engine_rel_to_root = str(v['engine_path'].relative_to(ROOT)).replace('\\', '/')
    run_script_content = f"""import pickle, importlib.util, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location('mod_{dname}', str(ROOT / '{engine_rel_to_root}'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DATA_PKL = ROOT / 'data/2016_H1_UNSEEN/universe_30_15m.pkl'
with open(DATA_PKL, 'rb') as f:
    universe = pickle.load(f)

stocks_30_2016 = {{
    'CANBK', 'PNB', 'BANKBARODA', 'SBIN', 'IOC',
    'BPCL', 'GAIL', 'COALINDIA', 'TATASTEEL', 'TATAPOWER',
    'TATAMOTORS', 'VEDL', 'HINDALCO', 'JINDALSTEL', 'JSWSTEEL',
    'BEL', 'ADANIPOWER', 'ADANIENT', 'RECLTD', 'PFC',
    'DLF', 'MOTHERSON', 'ITC', 'ICICIBANK', 'AXISBANK',
    'RELIANCE', 'INFY', 'SUNPHARMA', 'TECHM', 'CIPLA'
}}

cap, trades = mod.run_simulation(universe, '2016-01-01', '2016-06-30', '{dname}', universe_filter=stocks_30_2016)
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
    with open(target_dir / f"run_{dname}.py", 'w', encoding='utf-8') as rf:
        rf.write(run_script_content)
        
    # 5. Run Gate 1 Check
    p_g1 = subprocess.run([sys.executable, str(GATE1_SCRIPT), str(target_dir), '--data-dir', str(DATA_DIR)],
                          capture_output=True, text=True, cwd=str(ROOT))
    g1_status = 'PASS' if p_g1.returncode == 0 else 'FAIL'
    
    # 6. Run Gate 2 Check
    p_g2 = subprocess.run([sys.executable, str(GATE2_SCRIPT), str(target_dir), '--data-dir', str(DATA_DIR)],
                          capture_output=True, text=True, cwd=str(ROOT))
    g2_status = 'PASS' if p_g2.returncode == 0 else 'FAIL'
    
    print(f"[{tag}] Gate 1: {g1_status} | Gate 2: {g2_status}")
    if g1_status == 'FAIL':
        print(f"  Gate 1 Output:\n{p_g1.stdout}\n{p_g1.stderr}")
    if g2_status == 'FAIL':
        print(f"  Gate 2 Output:\n{p_g2.stdout}\n{p_g2.stderr}")
        
    summary_results.append({
        'version': tag,
        'dir_name': dname,
        'trades': n,
        'win_rate': wr,
        'gross_pnl': round(gross_pnl, 2),
        'charges': round(charges_total, 2),
        'net_pnl': net,
        'ending_capital': round(500 + net, 2),
        'pf': pf,
        'max_dd': dd,
        'avg_win': avg_w,
        'avg_loss': avg_l,
        'gate1': g1_status,
        'gate2': g2_status
    })

print("\n" + "="*120)
print("FINAL 2016 H1 UNSEEN BENCHMARK & AUDIT TABLE")
print("="*120)
df_res = pd.DataFrame(summary_results)
print(df_res.to_string(index=False))

with open(ROOT / 'scripts' / '2016_h1_unseen_benchmark_results.pkl', 'wb') as f:
    pickle.dump(df_res, f)
