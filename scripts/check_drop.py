import pandas as pd
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
df_old = pd.read_csv(ROOT_DIR / "strategies" / "V14" / "trades_intraday.csv")
print("Trades count:", len(df_old))
print("Exit reasons:")
print(df_old["exit_reason"].value_counts())
