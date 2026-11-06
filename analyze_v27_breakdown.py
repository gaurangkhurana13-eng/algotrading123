import pandas as pd
from pathlib import Path

csv_path = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V27\trades_intraday.csv")
df = pd.read_csv(csv_path)

print("=== V27 FORENSIC AUTOPSY: LONG VS SHORT ===")
for direction, grp in df.groupby("direction"):
    wins = grp[grp["net_pnl"] > 0]
    losses = grp[grp["net_pnl"] < 0]
    wr = len(wins) / len(grp) * 100.0 if len(grp) > 0 else 0
    gross = grp["gross_pnl"].sum()
    chg = grp["charges"].sum()
    net = grp["net_pnl"].sum()
    print(f"\nDIRECTION: {direction}")
    print(f"Total Trades: {len(grp)}")
    print(f"Win Rate: {wr:.2f}% ({len(wins)} W / {len(losses)} L)")
    print(f"Gross PnL: Rs {gross:.2f}")
    print(f"Dhan Friction: Rs {chg:.2f}")
    print(f"Net PnL: Rs {net:.2f}")
