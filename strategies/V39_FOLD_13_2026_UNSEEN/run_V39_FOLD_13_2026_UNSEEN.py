import pickle, importlib.util, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location('mod_v39', str(ROOT / 'strategies/V39/run_v39.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

V29_30_UNIVERSE = mod.V29_30_UNIVERSE

u1 = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
pool = u1
clean_78 = {k.replace('.NS', '') for k in u1.keys()}
unseen_syms = clean_78 - V29_30_UNIVERSE

start_date = '2026-01-01'
end_date = '2026-12-31'
univ = unseen_syms

cap, trades = mod.run_simulation(
    pool,
    start_date,
    end_date,
    'V39_FOLD_13_2026_UNSEEN',
    universe_filter=univ,
    min_tm_odds=0.40,
    skip_midday_deadzone=True
)

tdf = pd.DataFrame(trades)
out_dir = Path(__file__).parent
tdf.to_csv(out_dir / 'trades_intraday.csv', index=False)

wins = tdf[tdf["net_pnl"] > 0] if not tdf.empty else pd.DataFrame()
losses = tdf[tdf["net_pnl"] < 0] if not tdf.empty else pd.DataFrame()
wr = len(wins) / len(tdf) * 100.0 if not tdf.empty else 0.0
net = tdf["net_pnl"].sum() if not tdf.empty else 0.0
gross_w = wins["gross_pnl"].sum() if not wins.empty else 0.0
gross_l = abs(losses["gross_pnl"].sum()) if not losses.empty else 0.0
pf = (wins["net_pnl"].sum() / abs(losses["net_pnl"].sum())) if not losses.empty and abs(losses["net_pnl"].sum()) > 0 else 999.0
avg_w = wins["net_pnl"].mean() if not wins.empty else 0.0
avg_l = losses["net_pnl"].mean() if not losses.empty else 0.0

cum = 500.0 + (tdf["net_pnl"].cumsum() if not tdf.empty else pd.Series([0.0]))
hwm = cum.cummax()
dd = (hwm - cum) / hwm * 100.0
max_dd = dd.max() if len(dd) > 0 else 0.0

res_text = f"""----------------------------------------
INTRADAY
----------------------------------------
Total Trades: {len(tdf)}
Wins: {len(wins)}
Losses: {len(losses)}
Win Rate: {wr:.2f}%
Gross Profit: Rs {gross_w:.2f}
Gross Loss: Rs {gross_l:.2f}
Net Profit: Rs {net:.2f}
Profit Factor: {pf:.2f}
Average Win: Rs {avg_w:.2f}
Average Loss: Rs {avg_l:.2f}
Max Drawdown: {max_dd:.2f}%
Final Capital: Rs {cap:.2f}

----------------------------------------
SWING
----------------------------------------
Total Trades: 0
Wins: 0
Losses: 0
Win Rate: 0.00%
Gross Profit: Rs 0.00
Gross Loss: Rs 0.00
Net Profit: Rs 0.00
Profit Factor: 0.00
Average Win: Rs 0.00
Average Loss: Rs 0.00
Max Drawdown: 0.00%
Final Capital: Rs 500.00
"""
with open(out_dir / 'results.txt', 'w') as f:
    f.write(res_text)

print(f"[V39_FOLD_13_2026_UNSEEN] Finished: {len(tdf)} trades | Net: Rs {net:.2f} | Cap: Rs {cap:.2f}")
