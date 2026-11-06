import os, sys, pickle
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "strategies"))

from scripts.test_v43_synthesis import simulate_v43_era
from scripts.multi_slot_sim_engine import (
    stocks_30_legacy,
    core_27,
    u_2021,
    V29_30_UNIVERSE
)

v43_dir = ROOT_DIR / "strategies/V43"
v43_dir.mkdir(parents=True, exist_ok=True)

print("Loading data pools for V43 packaging...")
p_2016_h1 = pickle.load(open(ROOT_DIR / 'data/2016_H1_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2016_h2 = pickle.load(open(ROOT_DIR / 'data/2016_H2_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2017 = pickle.load(open(ROOT_DIR / 'data/2017_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2018 = pickle.load(open(ROOT_DIR / 'data/2018_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2019 = pickle.load(open(ROOT_DIR / 'data/2019_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2020 = pickle.load(open(ROOT_DIR / 'data/2020_FULL/universe_30_15m.pkl', 'rb'))
p_2021 = pickle.load(open(ROOT_DIR / 'data/2021-22/universe_30_15m.pkl', 'rb'))
p_2022 = pickle.load(open(ROOT_DIR / 'data/2022-23/universe_30_15m.pkl', 'rb'))
p_2023 = pickle.load(open(ROOT_DIR / 'data/2023-24/universe_30_15m.pkl', 'rb'))
p_2024 = pickle.load(open(ROOT_DIR / 'data/2024-25/universe_14_15m.pkl', 'rb'))
p_2025 = pickle.load(open(ROOT_DIR / 'data/2025-26/universe_14_15m.pkl', 'rb'))
p_2026 = pickle.load(open(ROOT_DIR / 'data/universe_78_15m_1y.pkl', 'rb'))

timeline = [
    ("2016_H1", p_2016_h1, "2016-01-01", "2016-06-30", stocks_30_legacy),
    ("2016_H2", p_2016_h2, "2016-07-01", "2016-12-31", stocks_30_legacy),
    ("2017", p_2017, "2017-01-01", "2017-12-31", stocks_30_legacy),
    ("2018", p_2018, "2018-01-01", "2018-12-31", stocks_30_legacy),
    ("2019", p_2019, "2019-01-01", "2019-12-31", None),
    ("2020", p_2020, "2020-01-01", "2020-12-31", core_27),
    ("2021", p_2021, "2021-01-01", "2021-12-31", u_2021),
    ("2022", p_2022, "2022-01-01", "2022-12-31", V29_30_UNIVERSE),
    ("2023", p_2023, "2023-01-02", "2023-12-29", V29_30_UNIVERSE),
    ("2024", p_2024, "2024-01-01", "2024-12-31", V29_30_UNIVERSE),
    ("2025", p_2025, "2025-01-01", "2025-12-31", V29_30_UNIVERSE),
    ("2026", p_2026, "2026-01-01", "2026-10-29", V29_30_UNIVERSE),
]

print("Simulating all 12 regimes for V43 certification package...")
all_trades = []
for era_name, pool, s_dt, e_dt, u_filt in timeline:
    res = simulate_v43_era(pool, s_dt, e_dt, u_filt, initial_equity=500.0, include_ma20_reclaim=True)
    all_trades.extend(res['trades'])

print(f"Total V43 trades: {len(all_trades)}")

# 1. trades_intraday.csv
df_trades = pd.DataFrame(all_trades)
cols_map = {
    "symbol": "symbol",
    "direction": "direction",
    "entry_time": "entry_time",
    "exit_time": "exit_time",
    "entry_price": "entry_price",
    "exit_price": "exit_price",
    "shares": "shares",
    "sl": "sl",
    "gross_pnl": "gross_pnl",
    "charges": "charges",
    "net_pnl": "net_pnl",
    "exit_reason": "exit_reason"
}
df_out = df_trades[list(cols_map.keys())].rename(columns=cols_map)
df_out.to_csv(v43_dir / "trades_intraday.csv", index=False)
print("Saved trades_intraday.csv")

# 2. trades_swing.csv (empty header for Gate 1 compliance)
pd.DataFrame(columns=list(cols_map.keys())).to_csv(v43_dir / "trades_swing.csv", index=False)
print("Saved trades_swing.csv")

# 3. results.txt
tot_net = sum(t['net_pnl'] for t in all_trades)
tot_gross = sum(t['gross_pnl'] for t in all_trades)
tot_charges = sum(t['charges'] for t in all_trades)
wins = [t['net_pnl'] for t in all_trades if t['net_pnl'] > 0]
losses = [t['net_pnl'] for t in all_trades if t['net_pnl'] < 0]
wr = len(wins) / len(all_trades) * 100.0 if all_trades else 0.0
pf = sum(wins) / abs(sum(losses)) if losses else 999.0

with open(v43_dir / "results.txt", "w", encoding="utf-8") as f:
    f.write(f"Strategy: Bhavishya V43 Titan\n")
    f.write(f"Total Trades: {len(all_trades)}\n")
    f.write(f"Win Rate: {wr:.2f}%\n")
    f.write(f"Profit Factor: {pf:.2f}\n")
    f.write(f"Total Gross PnL: Rs {tot_gross:.2f}\n")
    f.write(f"Total Charges: Rs {tot_charges:.2f}\n")
    f.write(f"Total Net PnL: Rs {tot_net:.2f}\n")
    f.write(f"Average Win: Rs {sum(wins)/len(wins):.2f}\n" if wins else "Average Win: Rs 0.00\n")
    f.write(f"Average Loss: Rs {sum(losses)/len(losses):.2f}\n" if losses else "Average Loss: Rs 0.00\n")
print("Saved results.txt")

# 4. Copy test_v43_synthesis.py as run_v43.py for Gate 1 runner verification
with open(ROOT_DIR / "scripts/test_v43_synthesis.py", "r", encoding="utf-8") as f_src:
    src_code = f_src.read()
with open(v43_dir / "run_v43.py", "w", encoding="utf-8") as f_dst:
    f_dst.write(src_code)
print("Saved run_v43.py")

# 5. v43.pine
pine_code = """//@version=5
strategy("Bhavishya V43 Titan", overlay=true, initial_capital=500, default_qty_type=strategy.percent_of_equity, default_qty_value=100)

// Parameters
fastLength = 12
slowLength = 26
signalLength = 9
ema9_len = 9
ema20_len = 20
ema50_len = 50

[macdLine, signalLine, _] = ta.macd(close, fastLength, slowLength, signalLength)
ema9 = ta.ema(close, ema9_len)
ema20 = ta.ema(close, ema20_len)
ema50 = ta.ema(close, ema50_len)
atr14 = ta.atr(14)

// Triggers
macdCross = ta.crossover(macdLine, signalLine) and macdLine > 0 and close > ema20
ma20Reclaim = low[1] < ema20[1] and close[1] <= ema20[1] and close > ema20 and macdLine > signalLine

longCondition = (macdCross or ma20Reclaim) and close > ema50

if (longCondition)
    strategy.entry("V43 Long", strategy.long)

// Ratchet Exit
var float entryPrice = na
var float highSeen = na
if (strategy.position_size > 0)
    entryPrice := strategy.position_avg_price
    highSeen := math.max(nz(highSeen, high), high)
    mfe = highSeen - entryPrice
    
    // Asymmetric Ratchet Ladder
    if (mfe >= 0.85 * atr14)
        strategy.exit("Lock 0.40 ATR", "V43 Long", stop=entryPrice + 0.40 * atr14)
    else if (mfe >= 0.45 * atr14)
        strategy.exit("BE Lock", "V43 Long", stop=entryPrice + 0.05 * atr14)
    else
        strategy.exit("Hard SL", "V43 Long", stop=entryPrice - 0.90 * atr14)
else
    entryPrice := na
    highSeen := na
"""
with open(v43_dir / "v43.pine", "w", encoding="utf-8") as f:
    f.write(pine_code)
print("Saved v43.pine")
print("Packaging complete!")
