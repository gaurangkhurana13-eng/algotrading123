import os, sys, pickle, importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
DATA_PKL = ROOT / "data" / "2020_MID4M" / "universe_30_15m.pkl"

with open(DATA_PKL, 'rb') as f:
    universe = pickle.load(f)

# The 27 available stocks
stocks_27 = {
    'CANBK', 'DELTACORP', 'FEDERALBNK', 'GMRINFRA', 'GSFC', 'IDFCFIRSTB', 'IOC',
    'JYOTHYLAB', 'MANAPPURAM', 'MOTHERSON', 'NBCC', 'NCC', 'NATIONALUM', 'NMDC',
    'POONAWALLA', 'PNB', 'RBLBANK', 'RCF', 'SAIL', 'SUZLON', 'TATACHEM',
    'TATASTEEL', 'TATAPOWER', 'TRIDENT', 'UNIONBANK', 'YESBANK', 'ZEEL'
}

print(f"Loaded universe with {len(universe)//2} stocks. Testing 2020 Mid 4M (2020-05-01 to 2020-08-31)...")

versions = [
    ('V34.3', ROOT / 'strategies/V34.3/run_v34_3.py'),
    ('V36', ROOT / 'strategies/V36/run_v36.py'),
    ('V37', ROOT / 'strategies/V37/run_v37.py'),
    ('V38', ROOT / 'strategies/V38/run_v38.py'),
    ('V39', ROOT / 'strategies/V39/run_v39.py')
]

results = []

for name, script_path in versions:
    print(f"\n---> Running {name} on 2020 Mid 4M...")
    spec = importlib.util.spec_from_file_location(f"mod_{name}", str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    cap, trades = mod.run_simulation(
        universe,
        '2020-05-01',
        '2020-08-31',
        f'{name}_2020_MID4M',
        universe_filter=stocks_27
    )
    
    df_t = pd.DataFrame(trades)
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
        
        # Drawdown calculation
        equity_curve = [500.0]
        cur = 500.0
        for pnl in df_t['net_pnl']:
            cur += pnl
            equity_curve.append(cur)
        
        peak = equity_curve[0]
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
    else:
        win_rate = 0.0
        gross_pnl = 0.0
        charges = 0.0
        net_pnl = 0.0
        pf = 0.0
        max_dd = 0.0
        
    results.append({
        'version': name,
        'trades': n_trades,
        'win_rate': round(win_rate, 1),
        'gross_pnl': round(gross_pnl, 2),
        'charges': round(charges, 2),
        'net_pnl': round(net_pnl, 2),
        'end_cap': round(cap, 2),
        'profit_factor': round(pf, 2),
        'max_dd_pct': round(max_dd, 2)
    })
    print(f"[{name}] Trades: {n_trades} | WR: {win_rate:.1f}% | Net PnL: Rs {net_pnl:+.2f} | End Cap: Rs {cap:.2f} | PF: {pf:.2f} | MaxDD: {max_dd:.1f}%")

res_df = pd.DataFrame(results)
print("\n" + "="*80)
print("FINAL 2020 MID 4 MONTHS COMPARATIVE AUDIT TABLE (01 MAY 2020 - 31 AUG 2020)")
print("="*80)
print(res_df.to_string(index=False))
