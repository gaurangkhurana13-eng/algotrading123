import pickle, sys, importlib.util
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
# Load V34.3
p_v34 = ROOT / "strategies" / "V34.3" / "run_v34_3.py"
spec34 = importlib.util.spec_from_file_location("v34_3", p_v34)
mod_v34 = importlib.util.module_from_spec(spec34)
spec34.loader.exec_module(mod_v34)

# Load V37
sys.path.insert(0, str(ROOT / "strategies" / "V37"))
import run_v37 as mod_v37
UNIV = mod_v37.V29_30_UNIVERSE

# Load Data files
with open(ROOT / "data" / "2022-23" / "universe_30_15m.pkl", "rb") as f:
    p2022 = pickle.load(f)
with open(ROOT / "data" / "2023-24" / "universe_30_15m.pkl", "rb") as f:
    p2023 = pickle.load(f)
with open(ROOT / "data" / "2024-25" / "universe_14_15m.pkl", "rb") as f:
    p2024 = pickle.load(f)
with open(ROOT / "data" / "2025-26" / "universe_14_15m.pkl", "rb") as f:
    p2025 = pickle.load(f)

print("=== RUNNING V34.3 vs V37 ACROSS ALL ACCESSIBLE FOLDS ===")
folds = [
    ("Fold 1: 2022 Bear", p2022, "2022-01-01", "2022-12-31"),
    ("Fold 4: 2023 Mid 4M", p2023, "2023-05-01", "2023-08-31"),
    ("Fold 7: 2024 H1", p2024, "2024-01-01", "2024-06-30"),
    ("Fold 8: 2024 H2", p2024, "2024-07-01", "2024-12-31"),
    ("Fold 9: 2025 H1", p2025, "2025-01-01", "2025-07-31"),
    ("Fold 2: 2025 Last 5M", p2025, "2025-08-01", "2025-12-31"),
]

res = []
for name, pool, s, e in folds:
    cap34, tr34 = mod_v34.run_simulation(pool, s, e, f"{name} (V34.3)", universe_filter=UNIV)
    cap37, tr37 = mod_v37.run_simulation(pool, s, e, f"{name} (V37)", universe_filter=UNIV)
    
    df34 = pd.DataFrame(tr34)
    df37 = pd.DataFrame(tr37)
    
    n34 = df34['net_pnl'].sum() if len(df34) > 0 else 0.0
    n37 = df37['net_pnl'].sum() if len(df37) > 0 else 0.0
    
    wr34 = (df34['net_pnl'] > 0).mean() * 100 if len(df34) > 0 else 0.0
    wr37 = (df37['net_pnl'] > 0).mean() * 100 if len(df37) > 0 else 0.0
    
    res.append({
        "Fold": name,
        "V34.3 Trades": len(df34),
        "V34.3 WR": f"{wr34:.1f}%",
        "V34.3 Net": f"Rs {n34:+.2f}",
        "V34.3 End": f"Rs {cap34:.2f}",
        "V37 Trades": len(df37),
        "V37 WR": f"{wr37:.1f}%",
        "V37 Net": f"Rs {n37:+.2f}",
        "V37 End": f"Rs {cap37:.2f}",
    })

print("\n" + "="*80)
print(pd.DataFrame(res).to_string(index=False))
