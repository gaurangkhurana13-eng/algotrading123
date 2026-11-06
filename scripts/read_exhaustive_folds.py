import pickle
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
pkl_path = ROOT / "scripts/exhaustive_19_folds_v34_v41_v42.pkl"

data = pickle.load(open(pkl_path, "rb"))
if isinstance(data, list):
    df = pd.DataFrame(data)
    print(df.to_string())
else:
    print(data)
