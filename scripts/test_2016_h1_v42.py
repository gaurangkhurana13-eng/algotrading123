import sys, pickle, importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from scripts.test_base_risk_sensitivity import simulate_v42_risk_tuning

stocks_30_2016 = {
    'CANBK', 'PNB', 'BANKBARODA', 'SBIN', 'IOC',
    'BPCL', 'GAIL', 'COALINDIA', 'TATASTEEL', 'TATAPOWER',
    'TATAMOTORS', 'VEDL', 'HINDALCO', 'JINDALSTEL', 'JSWSTEEL',
    'BEL', 'ADANIPOWER', 'ADANIENT', 'RECLTD', 'PFC',
    'DLF', 'MOTHERSON', 'ITC', 'ICICIBANK', 'AXISBANK',
    'RELIANCE', 'INFY', 'SUNPHARMA', 'TECHM', 'CIPLA'
}

DATA_PKL = ROOT / "data/2016_H1_UNSEEN/universe_30_15m.pkl"
with open(DATA_PKL, 'rb') as f:
    universe = pickle.load(f)

print("="*90)
print("TESTING BHAVISHYA V42 ON 2016 H1 UNSEEN DATA (2016-01-01 to 2016-06-30)")
print("="*90)

# 1. Standard V42 (Base Risk = ₹14.0)
trades14, net14, wr14, dd14, tr14 = simulate_v42_risk_tuning(universe, '2016-01-01', '2016-06-30', stocks_30_2016, base_risk=14.0)
print(f"V42 Standard (Base ₹14.0): Trades={tr14:3d} | WR={wr14:5.1f}% | Net PnL=Rs {net14:+8.2f} | Max DD={dd14:4.1f}% | Final Wallet=Rs {500+net14:.2f}")

# 2. Turbo V42 (Base Risk = ₹25.0)
trades25, net25, wr25, dd25, tr25 = simulate_v42_risk_tuning(universe, '2016-01-01', '2016-06-30', stocks_30_2016, base_risk=25.0)
print(f"V42 Turbo    (Base ₹25.0): Trades={tr25:3d} | WR={wr25:5.1f}% | Net PnL=Rs {net25:+8.2f} | Max DD={dd25:4.1f}% | Final Wallet=Rs {500+net25:.2f}")

df14 = pd.DataFrame(trades14)
if not df14.empty:
    wins = df14[df14['net_pnl'] > 0]
    losses = df14[df14['net_pnl'] < 0]
    avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0.0
    avg_loss = abs(losses['net_pnl'].mean()) if len(losses) > 0 else 0.0
    gross_pnl = df14['gross_pnl'].sum()
    charges = df14['charges'].sum()
    print(f"\n--- Detailed V42 Standard Breakdown ---")
    print(f"  Gross PnL: Rs {gross_pnl:+.2f}")
    print(f"  Dhan Charges + Slippage: Rs {charges:.2f}")
    print(f"  Net PnL: Rs {net14:+.2f}")
    print(f"  Avg Win: Rs {avg_win:.2f}")
    print(f"  Avg Loss: Rs {avg_loss:.2f}")
    print(f"  Win/Loss Ratio: {avg_win/(avg_loss+1e-9):.2f}x")
    print("\nSample Trades:")
    print(df14[['symbol', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'shares', 'net_pnl', 'exit_reason']].head(10).to_string(index=False))
