import pickle
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")

# Check universe_78
with open(ROOT / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)

# Check unseen_high_beta_10
with open(ROOT / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    d_beta = pickle.load(f)

# Check unseen_10
with open(ROOT / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
    d_unseen = pickle.load(f)

all_available = set(list(univ.keys()) + list(d_beta.keys()) + list(d_unseen.keys()))

candidates_20 = [
    "PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS", "RITES.NS", "IDEA.NS",
    "MTNL.NS", "ZEEL.NS", "UTKARSHBNK.NS", "IFCI.NS", "NCC.NS",
    "NBCC.NS", "HFCL.NS", "MANAPPURAM.NS", "NATIONALUM.NS", "HUDCO.NS",
    "BHEL.NS", "SAIL.NS", "IRFC.NS", "NLCINDIA.NS", "CESC.NS"
]

print("=== V18 CANDIDATE STOCKS AVAILABILITY CHECK ===")
available = []
missing = []
for sym in candidates_20:
    if sym in all_available or (ROOT / "data" / f"{sym.replace('.NS', '').lower()}_15m_1y.pkl").exists():
        available.append(sym)
    else:
        missing.append(sym)

print(f"Available ({len(available)}/20):", available)
print(f"Missing ({len(missing)}/20):", missing)
