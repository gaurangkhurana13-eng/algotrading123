import pickle
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
data_dir = ROOT_DIR / "data"

u78 = set(pickle.load(open(data_dir / "universe_78_15m_1y.pkl", "rb")).keys())
u10_1 = set(pickle.load(open(data_dir / "unseen_10_15m_1y.pkl", "rb")).keys())
u10_2 = set(pickle.load(open(data_dir / "unseen_high_beta_10_15m_1y.pkl", "rb")).keys())

all_used = u78 | u10_1 | u10_2
clean_used = {s.replace(".NS", "").replace("NSE:", "").upper() for s in all_used}

print(f"Total used symbols: {len(clean_used)}")
print("Sample of used symbols:", sorted(list(clean_used))[:20])

# High-Beta, volatile, active retail momentum / smallcap candidates under Rs 450:
candidates = [
    "JPASSOCIAT", "GTLINFRA", "VIKASLIFE", "ORIENTPPR", "HBLPOWER",
    "SJVN", "SUBEXLTD", "FCSSOFT", "RTNPOWER", "INFIBEAM",
    "SOUTHBANK", "UCOBANK", "CENTRALBK", "MAHABANK", "IOB",
    "TRIDENT", "ALOKINDS", "BCG", "DISHTV", "SEPC",
    "TV18BRDCST", "DEN", "HATHWAY", "NETWORK18", "JINDALSTEL",
    "BODALCHEM", "HEG", "GRAPHITE", "RCF", "NFL", "FACT",
    "SUZLON", "ZOMATO", "TATASTEEL", "IREDA", "HUDCO", "IRFC", "RVNL"
]

fresh = [c for c in candidates if c not in clean_used]
print(f"\nFresh untouched candidates not in any previous test ({len(fresh)}):")
print(fresh)
