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

print(f"Total Unique Stocks Already Downloaded/Used: {len(used)}")
print("Sample of used stocks:", sorted(list(used))[:15])

# Check potential fresh liquid mid/small cap NSE candidates
fresh_candidates = [
    "NBCC", "HFCL", "NATIONALUM", "MANAPPURAM", "ENGINERSIN", 
    "BSOFT", "ECLERX", "EXIDEIND", "CESC", "GRAPHITE", 
    "HEG", "AMARAJABAT", "CASTROLIND", "GLENMARK", "APOLLOTYRE"
]

truly_unseen = [s for s in fresh_candidates if s not in used]
print("\nTruly Unseen Candidate Stocks (Never touched):")
print(truly_unseen)
