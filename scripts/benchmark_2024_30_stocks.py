import pickle
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
data_path = ROOT / "data" / "2024-25" / "universe_30_15m.pkl"

with open(data_path, "rb") as f:
    pool_2024 = pickle.load(f)

print(f"=== LOADED 2024 DATASET: {len(pool_2024)} entries ({len(pool_2024)//2} unique stocks) ===")

versions = [
    ("V34.3", ROOT / "strategies" / "V34.3" / "run_v34_3.py"),
    ("V36", ROOT / "strategies" / "V36" / "run_v36.py"),
    ("V37", ROOT / "strategies" / "V37" / "run_v37.py"),
    ("V38", ROOT / "strategies" / "V38" / "run_v38.py"),
]

results = []

for v_name, v_path in versions:
    print(f"\n>>> Running {v_name} on 2024 Full Year (All 30 Stocks)...")
    spec = importlib.util.spec_from_file_location(f"mod_{v_name}", str(v_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    cap, trades = mod.run_simulation(
        pool_2024,
        "2024-01-01",
        "2024-12-31",
        f"{v_name} 2024 FULL (30 STOCKS)"
    )
    
    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        wins = tdf[tdf["net_pnl"] > 0]
        losses = tdf[tdf["net_pnl"] < 0]
        wr = len(wins) / len(tdf) * 100.0
        net_pnl = tdf["net_pnl"].sum()
        gross_win = wins["gross_pnl"].sum() if not wins.empty else 0.0
        gross_loss = abs(losses["gross_pnl"].sum()) if not losses.empty else 0.0
        pf = (wins["net_pnl"].sum() / abs(losses["net_pnl"].sum())) if not losses.empty and abs(losses["net_pnl"].sum()) > 0 else 999.0
        
        cum = 500.0 + tdf["net_pnl"].cumsum()
        hwm = cum.cummax()
        dd = (hwm - cum) / hwm * 100.0
        max_dd = dd.max() if len(dd) > 0 else 0.0
        
        results.append({
            "Version": v_name,
            "Trades": len(tdf),
            "Wins": len(wins),
            "Losses": len(losses),
            "WR%": f"{wr:.2f}%",
            "Net_PnL": f"+Rs {net_pnl:.2f}" if net_pnl >= 0 else f"-Rs {abs(net_pnl):.2f}",
            "Net_Val": net_pnl,
            "Final_Cap": f"Rs {cap:.2f}",
            "PF": f"{pf:.2f}",
            "Max_DD%": f"{max_dd:.2f}%"
        })
        print(f"  Result: {len(tdf)} trades | {wr:.2f}% WR | Net: Rs {net_pnl:.2f} | End Cap: Rs {cap:.2f} | PF: {pf:.2f} | Max DD: {max_dd:.2f}%")
    else:
        print(f"  Result: 0 trades")

res_df = pd.DataFrame(results)
print("\n" + "="*80)
print("=== 2024 FULL YEAR (30 STOCKS) BENCHMARK SUMMARY TABLE ===")
print("="*80)
print(res_df.to_string(index=False))
