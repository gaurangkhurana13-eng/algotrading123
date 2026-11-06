import pickle
import importlib.util
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location('mod_v39', str(ROOT / 'strategies/V39/run_v39.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

pool_path = ROOT / 'data/2024-25/universe_30_15m.pkl'
with open(pool_path, 'rb') as f:
    pool_2024 = pickle.load(f)

UNIV_30 = {
    "CENTRALBK", "DELTACORP", "ELECTCAST", "GSFC", "IDEA", "IFCI", "IOB", "IRCON",
    "J&KBANK", "JYOTHYLAB", "MANALIPETC", "MTNL", "NBCC", "NCC", "NIACL", "NMDC",
    "PCJEWELLER", "PNB", "RBLBANK", "RCF", "RENUKA", "RITES", "RPOWER", "SUZLON",
    "SWSOLAR", "TATASTEEL", "TEXRAIL", "TTML", "UTKARSHBNK", "ZEEL"
}

cap, trades = mod.run_simulation(
    pool_2024,
    '2024-01-01',
    '2024-12-31',
    'V39 2024 FULL (30 STOCKS)',
    universe_filter=UNIV_30,
    min_tm_odds=0.40,
    skip_midday_deadzone=True
)

tdf = pd.DataFrame(trades)
out_dir = Path(__file__).parent
tdf.to_csv(out_dir / 'trades_intraday.csv', index=False)

# Write summary to results.txt
wins = tdf[tdf["net_pnl"] > 0]
losses = tdf[tdf["net_pnl"] < 0]
wr = len(wins) / len(tdf) * 100.0 if not tdf.empty else 0.0
net_pnl = tdf["net_pnl"].sum() if not tdf.empty else 0.0
pf = (wins["net_pnl"].sum() / abs(losses["net_pnl"].sum())) if not losses.empty and abs(losses["net_pnl"].sum()) > 0 else 999.0

cum = 500.0 + tdf["net_pnl"].cumsum() if not tdf.empty else pd.Series([500.0])
hwm = cum.cummax()
dd = (hwm - cum) / hwm * 100.0
max_dd = dd.max() if len(dd) > 0 else 0.0

res_txt = f"""BHAVISHYA V39 2024 FULL YEAR (30 STOCKS) RESULTS:
Trades: {len(tdf)}
Wins: {len(wins)} | Losses: {len(losses)}
Win Rate: {wr:.2f}%
Net PnL: Rs {net_pnl:.2f}
Final Capital: Rs {cap:.2f}
Profit Factor: {pf:.2f}
Max Drawdown: {max_dd:.2f}%
"""
with open(out_dir / 'results.txt', 'w') as f:
    f.write(res_txt)

print(res_txt)
