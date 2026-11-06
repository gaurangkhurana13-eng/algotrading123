import pickle
import importlib.util
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
spec = importlib.util.spec_from_file_location('mod_v34_3', str(ROOT / 'strategies/V34.3/run_v34_3.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

data_path = ROOT / 'data/2021-22/universe_30_15m.pkl'
with open(data_path, 'rb') as f:
    p2021 = pickle.load(f)

print(f"Loaded 2021 dataset with {len(p2021)} entries ({len(p2021)//2} unique stocks).")

# Run V34.3 simulation on 2021 (Full Year: 2021-01-01 to 2021-12-31)
cap, trades = mod.run_simulation(
    p2021,
    '2021-01-01',
    '2021-12-31',
    'V34.3 2021 FULL'
)

tdf = pd.DataFrame(trades)
print(f"\n=== SIMULATION COMPLETED ===")
print(f"Total Trades: {len(tdf)}")
if not tdf.empty:
    wins = tdf[tdf['net_pnl'] > 0]
    losses = tdf[tdf['net_pnl'] < 0]
    wr = len(wins) / len(tdf) * 100
    net_pnl = tdf['net_pnl'].sum()
    gross_win = wins['gross_pnl'].sum() if not wins.empty else 0.0
    gross_loss = abs(losses['gross_pnl'].sum()) if not losses.empty else 0.0
    pf = (wins['net_pnl'].sum() / abs(losses['net_pnl'].sum())) if not losses.empty and abs(losses['net_pnl'].sum()) > 0 else 999.0
    
    initial_equity = 500.0
    cum = initial_equity + tdf['net_pnl'].cumsum()
    hwm = cum.cummax()
    dd = (hwm - cum) / hwm * 100.0
    max_dd = dd.max() if len(dd) > 0 else 0.0
    
    print(f"Wins: {len(wins)} | Losses: {len(losses)} | Win Rate: {wr:.2f}%")
    print(f"Gross Profit: Rs {gross_win:.2f}")
    print(f"Gross Loss: Rs {gross_loss:.2f}")
    print(f"Net PnL: Rs {net_pnl:.2f}")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Final Capital: Rs {cap:.2f}")
    print(f"\nTop 5 traded symbols:\n{tdf['symbol'].value_counts().head(5)}")
    print(f"\nFirst 3 trades:\n{tdf[['symbol', 'direction', 'entry_time', 'exit_time', 'net_pnl', 'exit_reason']].head(3)}")
    print(f"\nLast 3 trades:\n{tdf[['symbol', 'direction', 'entry_time', 'exit_time', 'net_pnl', 'exit_reason']].tail(3)}")
