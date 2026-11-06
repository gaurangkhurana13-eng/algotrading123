import pandas as pd
import pickle
from pathlib import Path

df = pd.read_csv("strategies/V14/trades_intraday.csv")
print("Total rows:", len(df))

for idx in [14, 87]:
    row = df.iloc[idx]
    print(f"\n--- Row {idx+1} ---")
    print(row.to_dict())
    
    # Check bars
    sym = row["symbol"]
    with open(f"data/{sym.lower()}_15m_1y.pkl", "rb") as f:
        bars = pickle.load(f)
    
    e_t = pd.Timestamp(row["entry_time"])
    x_t = pd.Timestamp(row["exit_time"])
    seg = bars.loc[e_t:x_t]
    print("Segment bars count:", len(seg))
    print(seg[["open", "high", "low", "close"]])
    print("Initial SL:", row["sl"])
    print("Min low in seg:", seg["low"].min())
