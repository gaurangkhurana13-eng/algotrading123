import pickle
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
data_path = ROOT / "data" / "2024-25" / "universe_30_15m.pkl"

with open(data_path, "rb") as f:
    pool_2024 = pickle.load(f)

print(f"Loaded {len(pool_2024)} entries.")

# Let's inspect session structure:
# Indian market 15-minute bars per day: 09:15 to 15:15 = 25 bars per day
# Time slots: 09:15, 09:30, 09:45, 10:00, 10:15, 10:30, 10:45, 11:00, 11:15, 11:30, 11:45, 12:00,
# 12:15, 12:30, 12:45, 13:00, 13:15, 13:30, 13:45, 14:00, 14:15, 14:30, 14:45, 15:00, 15:15

sample_df = pool_2024["TATASTEEL"]
print("Sample bars per day:")
print(sample_df.groupby(sample_df.index.date).size().value_counts())
