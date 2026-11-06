import pickle
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)

k = list(univ.keys())[0]
df = univ[k]
print("First key:", k)
print("Dataframe shape:", df.shape)
print("Columns:", list(df.columns))
print("Index type:", type(df.index))
print(df.head(2))
