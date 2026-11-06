import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

# Let's inspect run_v17.py trades vs master_engine trades
df_v17 = pd.read_csv(ROOT_DIR / "strategies" / "V17" / "trades_intraday.csv")
print("=== V17 WINNERS SUMMARY ===")
v17_wins = df_v17[df_v17["net_pnl"] > 0]
print("V17 Wins count:", len(v17_wins))
print("V17 Wins Avg Net PnL:", v17_wins["net_pnl"].mean())
print("V17 Wins Max Net PnL:", v17_wins["net_pnl"].max())
print("V17 Exit Reasons:\n", df_v17["exit_reason"].value_counts())
print("\nV17 Top 10 Wins:\n", v17_wins.sort_values("net_pnl", ascending=False)[["symbol", "net_pnl", "exit_reason", "shares"]].head(10).to_string())
