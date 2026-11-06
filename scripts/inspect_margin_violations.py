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
L = gate.build_ledger(ctx)

for d in ctx.rep.discs:
    if d["check"] == 5 and "notional" in d["field"]:
        print(d)

viol = (L["used"] + L["req"]) > L["avail"] + 0.01
print(f"Total margin violations: {viol.sum()}")
bad_L = L[viol]
for idx, r in bad_L.iterrows():
    print(f"\nViolation Trade {idx}: {r['sym']} {r['entry']} -> {r['exit']}")
    print(f"  notional={r['notional']:.2f}, req={r['req']:.2f}, used={r['used']:.2f}, avail={r['avail']:.2f}, E={r['E']:.2f}")
    # Show what trades were in 'used'
    ent_k = r["entry"].value
    k = idx
    prev = L.iloc[:k]
    open_trades = prev[prev["exit"].astype("int64") >= ent_k]
    print(f"  Open trades contributing to used ({len(open_trades)} trades):")
    for _, ot in open_trades.iterrows():
        print(f"    {ot['sym']}: {ot['entry']} -> {ot['exit']}, notional={ot['notional']:.2f}, req={ot['req']:.2f}")
