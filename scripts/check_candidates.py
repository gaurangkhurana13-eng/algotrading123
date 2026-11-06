import pickle
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\data")

used = set()
for pkl in ["universe_78_15m_1y.pkl", "unseen_10_15m_1y.pkl", "unseen_high_beta_10_15m_1y.pkl", "unseen_batch3_10_15m_1y.pkl"]:
    path = ROOT / pkl
    if path.exists():
        with open(path, "rb") as f:
            d = pickle.load(f)
            for k in d.keys():
                used.add(k.replace(".NS", "").upper())

pool = [
    "CESC", "CASTROLIND", "MAHABANK", "PSB", "FSL", 
    "GMDCLTD", "PPLPHARMA", "IDFC", "NETWORK18", "JKTYRE",
    "EIDPARRY", "HBLPOWER", "BSOFT", "GRAPHITE", "TRITURBINE"
]

clean = [s for s in pool if s not in used]
print("Verified 100% Unseen Candidates:", clean[:10])
print(f"Count: {len(clean)}")
