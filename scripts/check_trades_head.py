import pandas as pd
from pathlib import Path
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
tdf = pd.read_csv(ROOT / "strategies/V39_2024_FULL/trades_intraday.csv")
print("First 10 trades:")
print(tdf[["symbol", "entry_time", "exit_time", "entry_price", "exit_price"]].head(10))
