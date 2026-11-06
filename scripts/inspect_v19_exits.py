import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "scripts"))
from test_v19_master_synthesis import run_v19_simulation

res = run_v19_simulation(hard_sl_mult=0.85, single_risk_cap=95.0, enable_pyramid=False)
df = res["df_trades"]

print("=== V19 EXIT REASONS ===")
print(df["exit_reason"].value_counts())

print("\n=== SAMPLE TRADES (FIRST 15) ===")
print(df[["symbol", "net_pnl", "gross_pnl", "exit_reason", "entry_price", "exit_price", "shares"]].head(15).to_string())
