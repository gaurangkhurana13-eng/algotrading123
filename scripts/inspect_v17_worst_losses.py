import pandas as pd
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
df = pd.read_csv(ROOT_DIR / "strategies" / "V17" / "trades_intraday.csv")

losses = df[df["net_pnl"] < 0].sort_values("net_pnl")
print(f"Total Losing Trades in V17: {len(losses)} / {len(df)}")
print(f"Average Loss: Rs {losses['net_pnl'].mean():.2f}")
print("\nTop 15 Worst Losses in V17:")
for _, r in losses.head(15).iterrows():
    sh = r['shares']
    ep = r['entry_price']
    xp = r['exit_price']
    sl = r['sl']
    notional = sh * ep
    print(f"Date: {r['entry_time'][:10]} | {r['symbol']:<12} | Notional: Rs {notional:>8.2f} | Shares: {sh:>5} | Entry: Rs {ep:>6.2f} | Exit: Rs {xp:>6.2f} | SL: Rs {sl:>6.2f} | Loss: Rs {r['net_pnl']:>7.2f} | Reason: {r['exit_reason']}")
