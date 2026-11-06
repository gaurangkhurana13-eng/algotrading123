import pickle
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
p = ROOT / "data" / "2024-25" / "suzlon_15m_1y.pkl"
with open(p, "rb") as f:
    bars = pickle.load(f)

tdf = pd.read_csv(ROOT / "strategies" / "V39_2024_FULL" / "trades_intraday.csv")
s_trades = tdf[tdf["symbol"] == "SUZLON"]
print("Sample SUZLON trades:")
print(s_trades[["symbol", "entry_time", "exit_time"]].head(3))

times_e = pd.DatetimeIndex(s_trades["entry_time"]).astype("datetime64[ns]")
print("bars.index dtype:", bars.index.dtype)
print("times_e dtype:", times_e.dtype)

try:
    idx = bars.index.get_indexer(times_e)
    print("indexer result:", idx)
except Exception as e:
    print("indexer error:", e)

# Test converting bars.index to ns
bars.index = pd.to_datetime(bars.index).astype("datetime64[ns]")
idx2 = bars.index.get_indexer(times_e)
print("After bars.index -> ns, indexer result:", idx2)
