from pathlib import Path
import sys
import argparse
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
gate.check3(ctx)

for c, s, msg in ctx.rep.findings:
    if c == 3:
        print(f"[{s}] {msg}")

for d in ctx.rep.discs:
    if d["check"] == 3:
        print("Disc:", d)
