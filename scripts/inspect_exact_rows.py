from pathlib import Path
import sys
import argparse
import pandas as pd
ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))

import strategy_evaluate_gate_v2 as gate

ap = argparse.ArgumentParser()
ap.add_argument("version")
ap.add_argument("--allow-missing-data", action="store_true")
ap.add_argument("--ts-convention", default="start")
ap.add_argument("--data-dir", default=None)
args = ap.parse_args(["V14"])

ctx = gate.Ctx(ROOT_DIR / "strategies" / "V14", args, ROOT_DIR)
gate.check1(ctx)
gate.prepare_book(ctx, "intraday", ctx.books["intraday"])
gate.check2(ctx)
df = ctx.books["intraday"]
g = ctx.ground["intraday"]

for idx in [13, 86]:
    row = df.iloc[idx]
    ei = g["e_idx"].iloc[idx]
    xi = g["x_idx"].iloc[idx]
    sym = row["_sym"]
    bars = ctx.get_bars(sym)
    print(f"\n--- CSV Line {row['_row']}: {sym} ---")
    print(f"entry_time: {row['entry_time']}, exit_time: {row['exit_time']}")
    print(f"entry_price: {row['entry_price']}, sl: {row['sl']}, exit_reason: {row['exit_reason']}")
    print(f"ei: {ei}, xi: {xi}")
    fill_at_close = abs(row["entry_price"] - g["e_close"].iloc[idx]) < abs(row["entry_price"] - g["e_open"].iloc[idx])
    a = int(ei) + (1 if fill_at_close else 0)
    b = int(xi)
    print(f"fill_at_close: {fill_at_close}, a={a}, b={b}")
    seg = bars.iloc[a:b]
    print(f"Segment bars between a and b ({len(seg)} bars):")
    for t, r in seg.iterrows():
        print(f"  {t}: open={r['open']}, low={r['low']}, high={r['high']}, close={r['close']}")
    print(f"seg low min: {seg['low'].min()} vs sl: {row['sl']}")
