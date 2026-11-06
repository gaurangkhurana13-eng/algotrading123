import pickle
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
with open(ROOT / "data" / "2022-23" / "universe_30_15m.pkl", "rb") as f:
    p2022 = pickle.load(f)

df_trades = pd.read_csv(ROOT / "strategies" / "V36" / "fold_2022" / "trades_intraday.csv")

rvols = []
for idx, r in df_trades.iterrows():
    sym = r['symbol']
    t_str = r['entry_time']
    # find key in p2022
    k = sym + ".NS" if sym + ".NS" in p2022 else sym
    df = p2022[k]
    v = df['volume']
    v_sma20 = v.rolling(20).mean()
    rvol_val = (v / (v_sma20 + 1e-9)).loc[t_str]
    rvols.append(rvol_val)

df_trades['rvol'] = rvols
print("2022 Trades RVOL Distribution:")
print(f"Min RVOL across all 273 trades: {min(rvols):.2f}")
print(f"Trades with RVOL < 1.50: {sum(r < 1.50 for r in rvols)}")
print(f"Trades with RVOL >= 1.50: {sum(r >= 1.50 for r in rvols)}")

low_rvol_trades = df_trades[df_trades['rvol'] < 1.50]
print("\nTrades with RVOL < 1.50 in 2022:")
print(low_rvol_trades[['symbol', 'entry_time', 'rvol', 'net_pnl']])
