import sys
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR))
from scripts.test_v19_final_calibration import run_v19 as run_10_core
from scripts.test_v19_all_weather import df_all as df_209

res_10 = run_10_core(trail_dist=0.28, climax_dist=0.20, risk_cap=110.0)
df_162 = res_10["df_trades"]

pnl_162 = df_162.groupby("symbol")["net_pnl"].sum()
pnl_209 = df_209.groupby("symbol")["net_pnl"].sum()

comparison = pd.DataFrame({"Original_162": pnl_162, "AllWeather_209": pnl_209}).fillna(0)
comparison["Diff"] = comparison["AllWeather_209"] - comparison["Original_162"]
print("=== PER-SYMBOL PROFIT DIFFERENCE (WHERE DID THE PROFIT GO?) ===")
print(comparison.sort_values("Diff", ascending=True))
