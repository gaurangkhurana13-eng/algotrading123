import pickle
import shutil
from pathlib import Path
import pandas as pd
from strategies.strategy_evaluate_gate_v2 import metrics

ROOT = Path(__file__).resolve().parent

UNIV = {
    'CENTRALBK', 'DELTACORP', 'GSFC', 'IDEA', 'IOB', 'JYOTHYLAB', 'NBCC', 'NCC',
    'NIACL', 'NMDC', 'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'SUZLON',
    'SWSOLAR', 'TATASTEEL', 'TTML', 'ZEEL'
}

COLS = [
    'symbol', 'direction', 'entry_time', 'exit_time', 'entry_price',
    'exit_price', 'shares', 'sl', 'gross_pnl', 'charges', 'net_pnl', 'exit_reason'
]

def generate_folder(version_tag: str, sim_module_path: str, pine_src_path: str, sim_func_name: str = 'run_simulation'):
    print(f"\n==========================================")
    print(f"Generating certified fold: {version_tag}")
    print(f"==========================================")
    
    out_dir = ROOT / 'strategies' / version_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Import module and run simulation
    import importlib.util
    spec = importlib.util.spec_from_file_location(f'mod_{version_tag}', str(ROOT / sim_module_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    sim_fn = getattr(mod, sim_func_name)
    p2021 = pickle.load(open(ROOT / 'data/2021-22/universe_30_15m.pkl', 'rb'))
    cap, trades = sim_fn(p2021, '2021-01-01', '2021-12-31', f'{version_tag}', universe_filter=UNIV)
    
    # 2. trades_intraday.csv
    tdf = pd.DataFrame(trades)[COLS]
    intraday_path = out_dir / 'trades_intraday.csv'
    tdf.to_csv(intraday_path, index=False)
    print(f"  -> {intraday_path} written ({len(tdf)} trades)")
    
    # 3. trades_swing.csv (canonical dummy schema)
    swing_path = out_dir / 'trades_swing.csv'
    pd.DataFrame(columns=COLS).to_csv(swing_path, index=False)
    print(f"  -> {swing_path} written (0 trades)")
    
    # 4. run_<VERSION>.py
    run_script_path = out_dir / f'run_{version_tag}.py'
    run_script_content = f'''import pickle, importlib.util, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location('mod_{version_tag.lower()}', str(ROOT / '{sim_module_path}'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

p2021 = pickle.load(open(ROOT / 'data/2021-22/universe_30_15m.pkl', 'rb'))
UNIV = {{
    'CENTRALBK', 'DELTACORP', 'GSFC', 'IDEA', 'IOB', 'JYOTHYLAB', 'NBCC', 'NCC',
    'NIACL', 'NMDC', 'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'SUZLON',
    'SWSOLAR', 'TATASTEEL', 'TTML', 'ZEEL'
}}
cap, trades = mod.{sim_func_name}(p2021, '2021-01-01', '2021-12-31', '{version_tag}', universe_filter=UNIV)
cols = [
    'symbol', 'direction', 'entry_time', 'exit_time', 'entry_price',
    'exit_price', 'shares', 'sl', 'gross_pnl', 'charges', 'net_pnl', 'exit_reason'
]
pd.DataFrame(trades)[cols].to_csv(Path(__file__).parent / 'trades_intraday.csv', index=False)
'''
    with open(run_script_path, 'w', encoding='utf-8') as f:
        f.write(run_script_content)
    print(f"  -> {run_script_path} written")
    
    # 5. Pine script
    pine_dst_path = out_dir / f'{version_tag}.pine'
    shutil.copy2(ROOT / pine_src_path, pine_dst_path)
    print(f"  -> {pine_dst_path} copied from {pine_src_path}")
    
    # 6. results.txt
    m = metrics(tdf)
    gp = tdf[tdf['gross_pnl'] > 0]['gross_pnl'].sum() if len(tdf) else 0.0
    gl = abs(tdf[tdf['gross_pnl'] < 0]['gross_pnl'].sum()) if len(tdf) else 0.0
    
    eq = 500.0
    hwm = 500.0
    max_dd = 0.0
    for p in tdf['net_pnl']:
        eq += p
        hwm = max(hwm, eq)
        dd = (hwm - eq) / hwm
        max_dd = max(max_dd, dd)
    
    res_text = f"""----------------------------------------
INTRADAY
----------------------------------------
Trades: {m['n']}
Wins: {m['wins']}
Losses: {m['losses']}
Win Rate: {m['win_rate']:.2f}%
Gross Profit: Rs {gp:.2f}
Gross Loss: Rs {gl:.2f}
Net PnL: Rs {m['net']:.2f}
Profit Factor: {m['pf']:.2f}
Avg Win: Rs {m['avg_win']:.2f}
Avg Loss: Rs {m['avg_loss']:.2f}
Max Drawdown: {max_dd*100:.2f}%
Final Wallet: Rs {eq:.2f}

----------------------------------------
SWING
----------------------------------------
Trades: 0
Wins: 0
Losses: 0
Win Rate: 0.00%
Gross Profit: Rs 0.00
Gross Loss: Rs 0.00
Net PnL: Rs 0.00
Profit Factor: 0.00
Avg Win: Rs 0.00
Avg Loss: Rs 0.00
Max Drawdown: 0.00%
Final Capital: Rs 500.00
"""
    with open(out_dir / 'results.txt', 'w', encoding='utf-8') as f:
        f.write(res_text)
    print(f"  -> {out_dir / 'results.txt'} written")


if __name__ == '__main__':
    generate_folder('V36_2021_FULL', 'strategies/V36/run_v36.py', 'strategies/V36/v36.pine')
    generate_folder('V37_2021_FULL', 'strategies/V37/run_v37.py', 'strategies/V37/v37.pine')
    generate_folder('V38_2021_FULL', 'strategies/V38/run_v38.py', 'strategies/V38/v38.pine')
    print("\nAll 3 folds generated successfully!")
