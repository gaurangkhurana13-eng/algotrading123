import pickle
import importlib.util
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
data_path = ROOT / "data" / "2024-25" / "universe_30_15m.pkl"
with open(data_path, "rb") as f:
    pool_2024 = pickle.load(f)

spec = importlib.util.spec_from_file_location("mod_v34_3", str(ROOT / "strategies/V34.3/run_v34_3.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

cap, trades = mod.run_simulation(pool_2024, "2024-01-01", "2024-12-31", "V34.3 2024")
tdf = pd.DataFrame(trades)

print(f"Total trades: {len(tdf)}")
print(f"Wins: {len(tdf[tdf['net_pnl'] > 0])}, Losses: {len(tdf[tdf['net_pnl'] < 0])}")
print(f"Net PnL: Rs {tdf['net_pnl'].sum():.2f}")

tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
tdf["hour_min"] = tdf["entry_time"].dt.strftime("%H:%M")

print("\nPnL by Entry Time Slot:")
slot_pnl = tdf.groupby("hour_min")["net_pnl"].agg(["count", "sum", lambda x: (x > 0).mean() * 100])
slot_pnl.columns = ["Trades", "Net PnL", "Win Rate %"]
print(slot_pnl.to_string())

print("\nPnL by Symbol (Worst 5 and Best 5):")
sym_pnl = tdf.groupby("symbol")["net_pnl"].agg(["count", "sum", lambda x: (x > 0).mean() * 100])
sym_pnl.columns = ["Trades", "Net PnL", "Win Rate %"]
print("\n--- WORST 5 SYMBOLS ---")
print(sym_pnl.sort_values(by="Net PnL").head(5).to_string())
print("\n--- BEST 5 SYMBOLS ---")
print(sym_pnl.sort_values(by="Net PnL", ascending=False).head(5).to_string())
