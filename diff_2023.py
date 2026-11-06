import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies")
df36 = pd.read_csv(ROOT / "V36" / "unseen_2023_mid4m" / "trades_intraday.csv")
df37 = pd.read_csv(ROOT / "V37" / "unseen_2023_mid4m" / "trades_intraday.csv")

# Find which trades in df36 are not in df37
diff_trades = df36[~df36['entry_time'].isin(df37['entry_time'])]
print("Trades dropped in 2023 Mid 4M in V37:")
print(diff_trades[['symbol', 'entry_time', 'exit_time', 'net_pnl', 'exit_reason']])
