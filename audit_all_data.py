import pickle
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\data")

print("=== CHECKING ALL DATASETS AND AVAILABLE FOLDS ===")
for p in ROOT.rglob("*.pkl"):
    # check if file size > 50KB to skip tiny files
    if p.stat().st_size > 50000:
        rel = p.relative_to(ROOT)
        # if in root, only check aggregate pkls
        if p.parent == ROOT and not p.name.startswith("universe_") and not p.name.startswith("unseen_"):
            continue
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                first_k = list(data.keys())[0]
                first_df = data[first_k]
                if isinstance(first_df, pd.DataFrame):
                    min_dt = first_df.index.min()
                    max_dt = first_df.index.max()
                    print(f"File: {str(rel):<35} | Tickers: {len(data):3d} | Range: {str(min_dt)[:10]} to {str(max_dt)[:10]} ({len(first_df)} bars)")
            elif isinstance(data, pd.DataFrame):
                min_dt = data.index.min()
                max_dt = data.index.max()
                print(f"DataFrame: {str(rel):<30} | Range: {str(min_dt)[:10]} to {str(max_dt)[:10]} ({len(data)} bars)")
        except Exception as e:
            pass
