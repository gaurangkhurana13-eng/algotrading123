import sys, os, pickle
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from strategies.V42_3.run_v42_3 import run_simulation_v42_3

def resample_pool(pool_1m, timeframe="5min"):
    pool_resampled = {}
    for k, df in pool_1m.items():
        df_r = df.resample(timeframe).agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        pool_resampled[k] = df_r
    return pool_resampled

def main():
    pkl_1m = ROOT / "data" / "2026_1M_V29_30" / "universe_v29_30_1m.pkl"
    with open(pkl_1m, "rb") as f:
        pool_1m = pickle.load(f)
    print(f"Loaded 1m pool: {len(pool_1m)//2} stocks")

    # Resample to 5m
    print("Resampling 1m data to 5m...")
    pool_5m = resample_pool(pool_1m, "5min")

    # Resample to 15m
    print("Resampling 1m data to 15m...")
    pool_15m = resample_pool(pool_1m, "15min")

    start_dt = "2026-10-12"
    end_dt = "2026-11-05"

    print("\n" + "=" * 95)
    print("V42.3 MULTI-TIMEFRAME AUDIT ON 2026 OCT-NOV DATA (ALL 32 STOCKS)")
    print("=" * 95)

    # 1. 15m V42.3
    print("\n--- RUNNING V42.3 ON 15-MINUTE CANDLES ---")
    cap_15m, trades_15m, wr_15m, dd_15m, pf_15m = run_simulation_v42_3(
        pool_15m, start_dt, end_dt, "V42_3_15M", base_risk=25.0
    )

    # 2. 5m V42.3
    print("\n--- RUNNING V42.3 ON 5-MINUTE CANDLES ---")
    cap_5m, trades_5m, wr_5m, dd_5m, pf_5m = run_simulation_v42_3(
        pool_5m, start_dt, end_dt, "V42_3_5M", base_risk=25.0
    )

    print("\n" + "=" * 95)
    print("SUMMARY COMPARISON ACROSS TIMEFRAMES (OCT 12 TO NOV 05, 2026):")
    print(f"15-Minute: Trades: {len(trades_15m):2d} | WR: {wr_15m:5.1f}% | Net: Rs {cap_15m - 500.0:+7.2f} | PF: {pf_15m:5.2f} | Max DD: {dd_15m:4.1f}%")
    print(f" 5-Minute: Trades: {len(trades_5m):2d} | WR: {wr_5m:5.1f}% | Net: Rs {cap_5m - 500.0:+7.2f} | PF: {pf_5m:5.2f} | Max DD: {dd_5m:4.1f}%")
    print("=" * 95)

if __name__ == "__main__":
    main()
