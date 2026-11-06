"""
Deep Search: SuperTrend + Squeeze Release + VWAP Anchor setups
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from framework import calc_dhan_charges, calculate_atr, compute_metrics

data_file = Path("C:/Users/admin/.gemini/antigravity/scratch/new_trade/data/pcjeweller_15m_1y.pkl")
df = pd.read_pickle(data_file)
c = df["close"]
h = df["high"]
l = df["low"]
o = df["open"]
v = df["volume"]

atr14 = calculate_atr(df, 14)
ema50 = c.ewm(span=50).mean()
ema20 = c.ewm(span=20).mean()
ema9 = c.ewm(span=9).mean()
ema21 = c.ewm(span=21).mean()

# Daily VWAP Shifted
df_temp = df.copy()
df_temp["_date"] = df_temp.index.date
tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
df_temp["_tpv"] = tp * df_temp["volume"]
cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
cum_vol = df_temp.groupby("_date")["volume"].cumsum()
vwap_lag = (cum_tpv / (cum_vol + 1e-9)).shift(1)

# SuperTrend
def compute_supertrend(df, period=10, mult=2.5):
    atr = calculate_atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2.0
    basic_upper = hl2 + mult * atr
    basic_lower = hl2 - mult * atr
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    curr_dir = 1
    for i in range(1, len(df)):
        if basic_lower.iloc[i] > final_lower.iloc[i-1] or df["close"].iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]
            
        if basic_upper.iloc[i] < final_upper.iloc[i-1] or df["close"].iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]
            
        if curr_dir == 1:
            if df["close"].iloc[i] < final_lower.iloc[i]:
                curr_dir = -1
                st.iloc[i] = final_upper.iloc[i]
            else:
                st.iloc[i] = final_lower.iloc[i]
        else:
            if df["close"].iloc[i] > final_upper.iloc[i]:
                curr_dir = 1
                st.iloc[i] = final_lower.iloc[i]
            else:
                st.iloc[i] = final_upper.iloc[i]
        direction.iloc[i] = curr_dir
    return st, direction

st_10_25, dir_10_25 = compute_supertrend(df, period=10, mult=2.5)
st_7_20, dir_7_20 = compute_supertrend(df, period=7, mult=2.0)
st_10_20, dir_10_20 = compute_supertrend(df, period=10, mult=2.0)

# Squeeze calculations:
# We test different KC multipliers: 1.5, 1.25, 1.0
bb_mid = c.rolling(20).mean()
bb_std = c.rolling(20).std()
bb_up = bb_mid + 2.0 * bb_std
bb_dn = bb_mid - 2.0 * bb_std

# KC multipliers
kc_mults = [1.5, 1.25, 1.0]
squeezes = {}
for km in kc_mults:
    kc_up = bb_mid + km * atr14
    kc_dn = bb_mid - km * atr14
    sq_on = (bb_up < kc_up) & (bb_dn > kc_dn)
    squeezes[km] = {
        "on": sq_on,
        "rel_1": sq_on.shift(1) & (~sq_on),
        "rel_2": (sq_on.shift(1) | sq_on.shift(2)) & (~sq_on),
        "rel_3": (sq_on.shift(1) | sq_on.shift(2) | sq_on.shift(3)) & (~sq_on),
        "rel_4": (sq_on.shift(1) | sq_on.shift(2) | sq_on.shift(3) | sq_on.shift(4)) & (~sq_on),
    }

# Carter Squeeze Momentum:
# Momentum is linear regression of delta from average of donchian and sma, or simply delta from ema20
mom = c - ( (h.rolling(20).max() + l.rolling(20).min()) / 2.0 + bb_mid ) / 2.0
mom_pos = mom > 0
mom_rising = mom > mom.shift(1)

# RSI 7 & MACD & ADX
delta = c.diff()
gain = delta.clip(lower=0).rolling(7).mean()
loss = (-delta.clip(upper=0)).rolling(7).mean()
rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))

macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
sig = macd.ewm(span=9).mean()

tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
atr10 = tr.rolling(10).mean()
up_move = h - h.shift(1)
down_move = l.shift(1) - l
plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(10).mean() / (atr10 + 1e-9))
minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(10).mean() / (atr10 + 1e-9))
dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
adx = dx.rolling(10).mean()

# Simulator
def simulate(entry_mask, sl_mult, p1_trig, p1_lock, p2_trig, trail_dist, tp_mult=4.0):
    capital = 500.0
    notional_cap = 2450.0
    r_cap = 15.50
    slippage = 0.0005
    trades = []
    pos = 0
    entry_p = 0.0
    qty = 0
    active_sl = 0.0
    active_tp = 0.0
    high_seen = 0.0
    phase = 0
    entry_atr = 0.0
    entry_t = None
    
    n = len(df)
    for i in range(100, n):
        ts = df.index[i]
        t = ts.time()
        cur_c = c.iloc[i]
        cur_h = h.iloc[i]
        cur_l = l.iloc[i]
        
        if pos == 1 and t >= pd.Timestamp("15:10").time():
            exit_p = cur_c * (1.0 - slippage)
            gross = (exit_p - entry_p) * qty
            chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=True)
            net = round(gross - chg, 2)
            trades.append({"net_pnl": net, "charges": chg})
            pos = 0
            continue
            
        if pos == 1:
            high_seen = max(high_seen, cur_h)
            mfe_atr = (high_seen - entry_p) / entry_atr if entry_atr > 0 else 0.0
            
            if mfe_atr >= p2_trig:
                phase = 2
                active_sl = max(active_sl, high_seen - (trail_dist * entry_atr))
            elif mfe_atr >= p1_trig and phase < 1:
                phase = 1
                active_sl = max(active_sl, entry_p + (p1_lock * entry_atr))
                
            if tp_mult > 0 and cur_h >= active_tp:
                exit_p = active_tp * (1.0 - slippage)
                gross = (exit_p - entry_p) * qty
                chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=True)
                net = round(gross - chg, 2)
                trades.append({"net_pnl": net, "charges": chg})
                pos = 0
            elif cur_l <= active_sl:
                exit_p = active_sl * (1.0 - slippage)
                gross = (exit_p - entry_p) * qty
                chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=True)
                net = round(gross - chg, 2)
                trades.append({"net_pnl": net, "charges": chg})
                pos = 0
                
        if pos == 0:
            time_ok = (t >= pd.Timestamp("09:20").time()) and (t <= pd.Timestamp("13:30").time())
            if entry_mask.iloc[i] and time_ok:
                pos = 1
                entry_t = ts
                entry_p = cur_c * (1.0 + slippage)
                entry_atr = max(atr14.iloc[i], 0.05)
                
                stop_dist = sl_mult * entry_atr
                friction_share = 0.002067 * entry_p
                denom = stop_dist + friction_share
                
                qty_risk = int(r_cap / denom) if denom > 0 else 1
                qty_notional = int(notional_cap / entry_p)
                qty = max(1, min(qty_risk, qty_notional))
                
                active_sl = entry_p - stop_dist
                active_tp = entry_p + tp_mult * entry_atr if tp_mult > 0 else 999999.0
                high_seen = entry_p
                phase = 0
                
    tdf = pd.DataFrame(trades)
    return compute_metrics(tdf, capital=500.0)

# Build search set
results = []

st_options = [
    ("ST10_2.5", dir_10_25, st_10_25),
    ("ST7_2.0", dir_7_20, st_7_20),
]

ratchets = [
    # Mission exact: Phase 1 lock at +0.85x ATR, Phase 2 trail at highest high - 0.65x ATR
    # Let's test different initial SL: 0.85, 0.90, 1.00, 1.10
    # Let's test P1 lock value: 0.35, 0.40, 0.50, 0.85
    # Let's test P2 trigger: 1.60, 1.70, 1.80, 2.00
    (0.85, 0.85, 0.35, 1.80, 0.65, 4.0),
    (0.90, 0.85, 0.35, 1.80, 0.65, 4.0),
    (1.00, 0.85, 0.35, 1.80, 0.65, 4.0),
    (0.85, 0.85, 0.40, 1.60, 0.65, 4.0),
    (0.90, 0.85, 0.40, 1.60, 0.65, 4.0),
    (1.00, 0.85, 0.40, 1.60, 0.65, 4.0),
    (0.85, 0.85, 0.50, 1.80, 0.65, 4.0),
    (0.90, 0.85, 0.50, 1.80, 0.65, 4.0),
    (1.00, 0.85, 0.50, 1.80, 0.65, 4.0),
    (0.85, 0.85, 0.85, 1.80, 0.65, 4.0),
    (0.90, 0.85, 0.85, 1.80, 0.65, 4.0),
    (1.00, 0.85, 0.85, 1.80, 0.65, 4.0),
]

print("Scanning combinations...", flush=True)

# Generate combinations:
for st_label, st_dir, st_val in st_options:
    st_bull = (st_dir == 1)
    st_cross = (st_dir == 1) & (st_dir.shift(1) == -1)
    st_cross_recent = st_cross | st_cross.shift(1) | st_cross.shift(2)
    
    for km in [1.5, 1.25, 1.0]:
        sq_data = squeezes[km]
        for sq_rel_type, sq_rel_series in [
            ("Rel1", sq_data["rel_1"]),
            ("Rel2", sq_data["rel_2"]),
            ("Rel3", sq_data["rel_3"]),
            ("Rel4", sq_data["rel_4"]),
        ]:
            # Set A: Squeeze Release triggers while SuperTrend is Bullish + VWAP Anchor
            sig_a = sq_rel_series & st_bull & (c > vwap_lag) & (c > ema50) & (c > o)
            
            # Set B: Squeeze Release + ST Bull + VWAP + Mom > 0
            sig_b = sq_rel_series & st_bull & (c > vwap_lag) & mom_pos & (c > ema50)
            
            # Set C: Squeeze Release + ST Bull + VWAP + MACD Bullish
            sig_c = sq_rel_series & st_bull & (c > vwap_lag) & (macd > sig) & (macd > 0)
            
            # Set D: ST Cross while Squeeze was coiled + VWAP
            sig_d = st_cross_recent & (sq_data["on"].shift(1) | sq_data["on"].shift(2) | sq_data["on"].shift(3)) & (c > vwap_lag) & (c > ema50)
            
            # Set E: V4 signals filtered by ST Bull + Squeeze release/coiled
            sig_e = (macd > sig) & (macd.shift(1) <= sig.shift(1)) & (macd > 0) & (c > ema50) & (rsi7 > 52.0) & (adx > 18.0) & (plus_di > minus_di) & (c > vwap_lag) & st_bull & (sq_rel_series | sq_data["on"].shift(1) | sq_data["on"].shift(2))
            
            # Set F: V4 signals + Squeeze Release fired
            sig_f = (macd > sig) & (macd.shift(1) <= sig.shift(1)) & (macd > 0) & (c > ema50) & (rsi7 > 52.0) & (adx > 18.0) & (plus_di > minus_di) & (c > vwap_lag) & st_bull & sq_rel_series

            # Set G: ST Cross + Squeeze Release + VWAP
            sig_g = (st_cross | st_cross.shift(1)) & sq_rel_series & (c > vwap_lag)

            setups = [
                (f"SigA_{st_label}_KM{km}_{sq_rel_type}", sig_a),
                (f"SigB_{st_label}_KM{km}_{sq_rel_type}", sig_b),
                (f"SigC_{st_label}_KM{km}_{sq_rel_type}", sig_c),
                (f"SigD_{st_label}_KM{km}_{sq_rel_type}", sig_d),
                (f"SigE_{st_label}_KM{km}_{sq_rel_type}", sig_e),
                (f"SigF_{st_label}_KM{km}_{sq_rel_type}", sig_f),
                (f"SigG_{st_label}_KM{km}_{sq_rel_type}", sig_g),
            ]
            
            for s_name, s_mask in setups:
                if s_mask.sum() < 5:
                    continue
                for sl_m, p1_t, p1_l, p2_t, tr_d, tp_m in ratchets:
                    m = simulate(s_mask, sl_m, p1_t, p1_l, p2_t, tr_d, tp_m)
                    if m["total_trades"] >= 8:
                        results.append((m["profit_factor"], m["win_rate"], m["net_pnl"], m["total_trades"], s_name, (sl_m, p1_t, p1_l, p2_t, tr_d), m))

print(f"Total simulated runs evaluated: {len(results)}", flush=True)

# Sort by profit factor descending
results.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

print("\n================== TOP CANDIDATES (PF >= 3.0 OR WR >= 75%) ==================")
qualifying = [r for r in results if r[0] >= 3.0 or r[1] >= 75.0]
print(f"Total qualifying candidates meeting WR >= 75% OR PF >= 3.0: {len(qualifying)}")

for pf, wr, pnl, n, name, r_cfg, m in qualifying[:30]:
    sl_m, p1_t, p1_l, p2_t, tr_d = r_cfg
    print(f"PF={pf:5.2f} | WR={wr:5.1f}% | Net=Rs {pnl:7.2f} | N={n:2d} | AvgW=Rs {m['avg_win']:5.2f} | AvgL=Rs {m['avg_loss']:6.2f} | {name} | SL={sl_m} P1={p1_t}/{p1_l} P2={p2_t}/{tr_d}")
