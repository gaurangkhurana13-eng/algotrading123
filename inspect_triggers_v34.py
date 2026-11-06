import os, sys, pickle
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies" / "V34"))

# Let's inspect which triggers generated which trades in V34 2026
df34 = pd.read_csv(ROOT_DIR / "strategies" / "V34" / "fold_2026" / "trades_intraday.csv")
print("Total trades in V34 2026:", len(df34))

# Let's check overlap: how many trades were concurrent (Slot 2)?
df34["entry_dt"] = pd.to_datetime(df34["entry_time"])
df34["exit_dt"] = pd.to_datetime(df34["exit_time"])

slot2_count = 0
slot1_count = 0
for i in range(len(df34)):
    e_i = df34.iloc[i]["entry_dt"]
    # Check if any prior trade j < i was still open when trade i entered
    prior_open = df34.iloc[:i][df34.iloc[:i]["exit_dt"] > e_i]
    if len(prior_open) > 0:
        slot2_count += 1
    else:
        slot1_count += 1

print(f"Slot 1 trades: {slot1_count}, Slot 2 (concurrent) trades: {slot2_count}")
