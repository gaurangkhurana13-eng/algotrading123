import pickle
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\data")

print("Checking subfolder aggregates:")
for f in ["2022-23", "2023-24", "2024-25", "2024_Q4", "2025-26"]:
    folder = ROOT / f
    if folder.exists():
        pkls = list(folder.glob("*.pkl"))
        print(f"\nFolder: {f} (contains {len(pkls)} pkl files)")
        for p in pkls:
            if "universe" in p.name:
                with open(p, "rb") as fl:
                    d = pickle.load(fl)
                fk = list(d.keys())[0]
                df = d[fk]
                print(f"  {p.name}: {len(d)} tickers, range {df.index.min()} to {df.index.max()}")
