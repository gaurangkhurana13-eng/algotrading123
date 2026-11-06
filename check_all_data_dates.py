import pickle
import pandas as pd
from pathlib import Path

root = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\data")
pkl_files = list(root.glob("*.pkl"))

print(f"Found {len(pkl_files)} pkl files in {root}")
results = []
for p in pkl_files:
    try:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            for k, df in obj.items():
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    results.append({"file": p.name, "symbol": k, "start": str(df.index.min()), "end": str(df.index.max()), "len": len(df)})
                    break
        elif isinstance(obj, pd.DataFrame) and len(obj) > 0:
            results.append({"file": p.name, "symbol": p.stem, "start": str(obj.index.min()), "end": str(obj.index.max()), "len": len(obj)})
    except Exception as e:
        pass

df_res = pd.DataFrame(results)
print(df_res.head(20).to_string())
print("\nEarliest date found in any file:", df_res["start"].min())
print("Latest date found in any file:", df_res["end"].max())
