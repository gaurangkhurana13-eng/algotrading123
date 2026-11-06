import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

# 1. Load 2026 Pools
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    u1 = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    u2 = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
    u3 = pickle.load(f)

all_pools = {}
all_pools.update(u1)
all_pools.update(u2)
all_pools.update(u3)

tickers_combined = [
    "J&KBANK.NS", "UTKARSHBNK.NS", "MANALIPETC.NS", "IDEA.NS", "PCJEWELLER.NS",
    "NIACL.NS", "MTNL.NS", "IFCI.NS", "TEXRAIL.NS", "SWSOLAR.NS",
    "TTML.NS", "RENUKA.NS", "RCF.NS", "NBCC.NS", "JYOTHYLAB.NS",
    "RBLBANK.NS", "ELECTCAST.NS", "PNB.NS", "GSFC.NS", "TATASTEEL.NS",
    "SUZLON.NS", "CENTRALBK.NS", "IOB.NS", "IRCON.NS", "NMDC.NS",
    "RITES.NS", "NCC.NS", "ZEEL.NS", "DELTACORP.NS", "RPOWER.NS"
]

def compute_vwap_shifted(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    df_temp["_date"] = df_temp.index.date
    tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
    df_temp["_tpv"] = tp * df_temp["volume"]
    cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
    cum_vol = df_temp.groupby("_date")["volume"].cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).shift(1)

def compute_er10(series_close: pd.Series) -> pd.Series:
    change = (series_close - series_close.shift(10)).abs()
    volatility = series_close.diff().abs().rolling(10).sum()
    return change / (volatility + 1e-9)

def compute_adx_di(df: pd.DataFrame, n: int = 10):
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    atr_s = tr.rolling(n).mean()
    pdi = 100 * pd.Series(pdm, index=df.index).rolling(n).mean() / (atr_s + 1e-9)
    mdi = 100 * pd.Series(mdm, index=df.index).rolling(n).mean() / (atr_s + 1e-9)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    adx = dx.rolling(n).mean()
    return adx, pdi, mdi

data_map = {}
for t in tickers_combined:
    if t not in all_pools:
        continue
    df = all_pools[t].copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    atr14 = calculate_atr(df, 14)
    atr_pct = (atr14 / c) * 100.0
    ema9 = c.ewm(span=9).mean()
    ema20 = c.ewm(span=20).mean()
    ema50 = c.ewm(span=50).mean()
    ema110 = c.ewm(span=110).mean()
    
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(7).mean()
    loss = (-delta.clip(upper=0)).rolling(7).mean()
    rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))
    
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()
    macd_cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    macd_slope = macd - macd.shift(1)
    
    adx, pdi, mdi = compute_adx_di(df, 10)
    vwap_lag = compute_vwap_shifted(df)
    v_sma20 = v.rolling(20).mean()
    rvol = v / (v_sma20 + 1e-9)
    er10 = compute_er10(c)
    
    high_5 = h.rolling(5).max().shift(1)
    low_2 = l.rolling(2).min()
    lookback = min(300, len(c) - 1)
    mom_cond = (c > c.shift(lookback))
    
    # GOAL 1: Balanced Entry Filter (Sweet Spot)
    # Allows early liftoff runners, rejects flat chop
    entry_sweet_spot = (
        (macd_cross_up | ((macd > sig) & (macd_slope > 0))) &
        (macd > 0) &
        (c > ema50) & (c > ema110) &
        (c > vwap_lag * 1.0010) &
        (rsi7 >= 48.0) & (rsi7 <= 75.0) &
        (adx >= 18.0) & (pdi > mdi) &
        (er10 >= 0.32) &
        (rvol >= 1.20) &
        (atr_pct >= 0.45) &
        mom_cond &
        (c <= 450.0) & (c >= 10.0)
    )
    
    # Strict filter for severe bleed days
    g4_bleed = (
        entry_sweet_spot &
        (rvol >= 1.25) &
        (c > high_5)
    )
    
    # GOAL 2: Intraday Trend Day Detection Flag
    is_trend_day_bar = (
        (c > ema9) &
        (adx >= 22.0) &
        (pdi > mdi) &
        (c > vwap_lag * 1.0015) &
        (c > c.shift(3))
    )
    
    df["_date"] = df.index.date
    day_open = df.groupby("_date")["open"].transform("first")
    df["day_ret"] = (df["close"] / day_open) - 1.0
    df["1h_ret"] = (df["close"] / df["close"].shift(4)) - 1.0
    above_20dma = (c > ema20).astype(float)
    
    data_map[t] = {
        "df": df, "c": c, "h": h, "l": l, "v": v, "atr14": atr14, "atr_pct": atr_pct,
        "ema9": ema9, "rsi7": rsi7, "vwap_lag": vwap_lag, "rvol": rvol, "er10": er10,
        "entry_sweet_spot": entry_sweet_spot, "g4_bleed": g4_bleed,
        "is_trend_day_bar": is_trend_day_bar, "low_2": low_2,
        "day_ret": df["day_ret"], "1h_ret": df["1h_ret"],
        "above_20dma": above_20dma
    }

bench_day_ret_series = pd.concat([d["day_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
bench_1h_ret_series = pd.concat([d["1h_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
breadth_series = pd.concat([d["above_20dma"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)

for t, d in data_map.items():
    b_day = bench_day_ret_series.reindex(d["day_ret"].index)
    b_1h = bench_1h_ret_series.reindex(d["1h_ret"].index)
    d["g1_cond"] = (d["day_ret"] > b_day)
    d["g2_cond"] = (d["1h_ret"] > b_1h)
    d["excess_1h"] = d["1h_ret"] - b_1h

all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
slippage = 0.0005
hard_sl_mult = 0.85
initial_equity = 500.0

positions = {}
trades = []
daily_traded = {}
last_exit_times = set()

# GOAL 4: DUAL-SLOT ARCHITECTURE (Max 2 concurrent positions)
MAX_SLOTS = 2
SLOT_EQUITY_FACTOR = 0.45  # 45% margin per slot -> total 90% (Gate V2 cap is 95%)

for ts in all_ts:
    if ts.weekday() >= 5:
        continue
    cur_date = ts.date()
    if cur_date not in daily_traded:
        daily_traded[cur_date] = set()
    t_time = ts.time()
    
    # 1. Manage Active Positions
    for ticker in list(positions.keys()):
        p = positions[ticker]
        t_data = data_map[ticker]
        if ts not in t_data["df"].index:
            continue
        row = t_data["df"].loc[ts]
        cur_c, cur_h, cur_l = row["close"], row["high"], row["low"]
        cur_rsi = t_data["rsi7"].loc[ts]
        cur_vwap = t_data["vwap_lag"].loc[ts]
        cur_ema9 = t_data["ema9"].loc[ts]
        low_2 = t_data["low_2"].loc[ts]
        is_trend_bar = t_data["is_trend_day_bar"].loc[ts]
        
        p["bars_held"] += 1
        p["high_seen"] = max(p["high_seen"], cur_h)
        mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
        
        # GOAL 2 & 3: Noise-Immune Dynamic Ratchet & Trend-Day Holding
        # 1) Base Structural Stop: 2-bar swing low minus buffer
        base_structural_stop = low_2 - (0.80 * p["entry_atr"])
        
        # 2) High-Water Ratchet
        if mfe_atr >= 4.00:
            if is_trend_bar:
                # Let the runner milk the maximum trend day move!
                high_ratchet = p["high_seen"] - (0.40 * p["entry_atr"])
            else:
                high_ratchet = p["high_seen"] - (0.15 * p["entry_atr"])
        elif mfe_atr >= 3.00:
            high_ratchet = p["high_seen"] - (0.25 * p["entry_atr"])
        elif mfe_atr >= 2.00:
            high_ratchet = p["high_seen"] - (0.45 * p["entry_atr"])
        elif mfe_atr >= 1.20:
            # Breakeven + profit cushion
            high_ratchet = p["entry_p"] + (0.35 * p["entry_atr"])
        elif mfe_atr >= 0.75:
            # Breakeven lock
            high_ratchet = p["entry_p"]
        else:
            high_ratchet = p["entry_p"] - (hard_sl_mult * p["entry_atr"])
            
        new_active_sl = max(p["active_sl"], round(high_ratchet, 2))
        # If in early stage, ensure structural stop doesn't override entry stop
        p["active_sl"] = new_active_sl
        
        def finalize_trade(exit_fill, exit_reason):
            tot_qty = p["qty"]
            gross = round((exit_fill - p["entry_p"]) * tot_qty, 2)
            chg = round(calc_dhan_charges(p["entry_p"], exit_fill, tot_qty, is_intraday=True), 2)
            net = round(gross - chg, 2)
            trades.append({
                "symbol": ticker,
                "direction": "LONG",
                "entry_time": p["entry_t"].strftime("%Y-%m-%d %H:%M:%S"),
                "exit_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "entry_price": p["entry_p"],
                "exit_price": exit_fill,
                "shares": tot_qty,
                "sl": p["initial_sl"],
                "gross_pnl": gross,
                "charges": chg,
                "net_pnl": net,
                "exit_reason": exit_reason
            })
            last_exit_times.add(ts)
            del positions[ticker]
            
        # Check Stop Loss
        if cur_l <= p["active_sl"]:
            finalize_trade(p["active_sl"] * (1.0 - slippage), "STOP")
            continue
            
        # Noise-Immune Scratch rule: only exit if held >= 5 bars and price genuinely collapsing
        if p["bars_held"] >= 5 and mfe_atr < 0.35 and cur_c < p["entry_p"] and (cur_rsi < 44.0 and cur_c < cur_vwap):
            finalize_trade(cur_c * (1.0 - slippage), "SCRATCH")
            continue
            
        # EOD Square-off at 14:45 bar close
        if t_time >= pd.Timestamp("14:45").time():
            finalize_trade(cur_c * (1.0 - slippage), "EOD")
            continue

    ek = initial_equity + sum(t["net_pnl"] for t in trades)
    hwm = initial_equity
    cum = initial_equity
    for t in trades:
        cum += t["net_pnl"]
        if cum > hwm:
            hwm = cum
    dd_pct = (hwm - ek) / max(hwm, initial_equity)
    
    # Check concurrent position capacity (Max 2 slots)
    if len(positions) >= MAX_SLOTS or ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
        continue
        
    current_notional = sum(p["entry_p"] * p["qty"] for p in positions.values())
    avail_margin = (0.95 * ek) - (current_notional / 5.0)
    slot_margin = (SLOT_EQUITY_FACTOR * ek)
    usable_margin = min(avail_margin, slot_margin)
    if usable_margin < 10.0:
        continue
    usable_notional = usable_margin * 5.0
    
    # Streak scaling per slot
    consec_losses = 0
    for t in reversed(trades):
        if t["net_pnl"] < 0:
            consec_losses += 1
        else:
            break
            
    if consec_losses >= 2 or (ek > 2500.0 and dd_pct > 0.16):
        alpha = 0.020
    elif consec_losses == 1:
        alpha = 0.040
    else:
        alpha = 0.088
        
    # Per-slot risk formula
    raw_r = 14.0 + (alpha * max(0.0, ek - 500.0) * 0.5)
    # Gate V2 Constitution: R_slot <= min(0.045 * ek, 130.0)
    gate_v2_slot_cap = min(0.045 * ek, 130.0)
    if dd_pct > 0.15:
        gate_v2_slot_cap *= 0.5
    target_risk = min(raw_r, gate_v2_slot_cap)
    
    # Dynamic Regime Switch 2.0
    b_day_now = bench_day_ret_series.get(ts, 0.0)
    is_severe_bleed_day = (b_day_now < -0.006)
    
    candidates = []
    for ticker in tickers_combined:
        if ticker in positions or ticker in daily_traded[cur_date] or ticker not in data_map:
            continue
        t_data = data_map[ticker]
        if ts not in t_data["df"].index:
            continue
        idx = t_data["df"].index.get_loc(ts)
        if idx < 100:
            continue
            
        gate_to_use = t_data["g4_bleed"].iloc[idx] if is_severe_bleed_day else t_data["entry_sweet_spot"].iloc[idx]
        
        if not (gate_to_use and t_data["g1_cond"].iloc[idx] and t_data["g2_cond"].iloc[idx]):
            continue
            
        cur_c = t_data["c"].iloc[idx]
        cur_h = t_data["h"].iloc[idx]
        cur_l = t_data["l"].iloc[idx]
        cur_v = t_data["v"].iloc[idx]
        entry_atr = max(t_data["atr14"].iloc[idx], 0.05)
        excess_1h = t_data["excess_1h"].iloc[idx]
        candidates.append((ticker, cur_c, cur_h, cur_l, cur_v, entry_atr, excess_1h))
        
    if not candidates:
        continue
    candidates.sort(key=lambda x: x[6], reverse=True)
    best = candidates[0]
    ticker, cur_c, cur_h, cur_l, cur_v, entry_atr, _ = best
    
    raw_entry = cur_c * (1.0 + slippage)
    entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
    active_sl = round(entry_p - (hard_sl_mult * entry_atr), 2)
    sl_dist = max(entry_p - active_sl, 0.05)
    friction = 0.002067 * entry_p
    denom = sl_dist + friction
    
    qty_r = int(target_risk / denom)
    qty_m = int((usable_notional * 0.90) / entry_p)
    qty = min(qty_r, qty_m)
    if cur_v > 0:
        qty = min(qty, int(0.045 * cur_v))
        
    if qty >= 1:
        positions[ticker] = {
            "entry_p": entry_p, "qty": qty, "entry_atr": entry_atr,
            "active_sl": active_sl, "initial_sl": active_sl,
            "high_seen": entry_p, "bars_held": 0, "entry_t": ts
        }
        daily_traded[cur_date].add(ticker)

# Export Outputs for Gate V2 Audit
tdf = pd.DataFrame(trades)
v31_dir = ROOT_DIR / "strategies" / "V31"
v31_dir.mkdir(parents=True, exist_ok=True)
tdf.to_csv(v31_dir / "trades_intraday.csv", index=False)

swing_cols = ["symbol", "direction", "entry_time", "exit_time", "entry_price", "exit_price", "shares", "sl", "gross_pnl", "charges", "net_pnl", "exit_reason"]
pd.DataFrame(columns=swing_cols).to_csv(v31_dir / "trades_swing.csv", index=False)

wins = tdf[tdf["net_pnl"] > 0]
losses = tdf[tdf["net_pnl"] < 0]
wr = len(wins) / len(tdf) * 100.0 if len(tdf) > 0 else 0
gross = tdf["gross_pnl"].sum()
chg = tdf["charges"].sum()
net = tdf["net_pnl"].sum()
cum = initial_equity + tdf["net_pnl"].cumsum()
hwm = cum.cummax()
dd = (hwm - cum) / hwm * 100.0
max_dd = dd.max() if len(dd) > 0 else 0.0

net_win = wins["net_pnl"].sum()
net_loss = abs(losses["net_pnl"].sum())
pf = round(net_win / net_loss, 2) if net_loss > 0 else 999.0
avg_win = round(wins["net_pnl"].mean(), 2) if len(wins) > 0 else 0.0
avg_loss = round(losses["net_pnl"].mean(), 2) if len(losses) > 0 else 0.0

res_text = f"""----------------------------------------
INTRADAY
----------------------------------------
Total Trades: {len(tdf)}
Wins: {len(wins)}
Losses: {len(losses)}
Win Rate: {wr:.2f}%
Gross Profit: Rs {wins['gross_pnl'].sum():.2f}
Gross Loss: Rs {abs(losses['gross_pnl'].sum()):.2f}
Net Profit: Rs {net:.2f}
Profit Factor: {pf}
Average Win: Rs {avg_win:.2f}
Average Loss: Rs {avg_loss:.2f}
Max Drawdown: {max_dd:.2f}%
Final Capital: Rs {cum.iloc[-1] if len(cum) > 0 else initial_equity:.2f}

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

with open(v31_dir / "results.txt", "w") as f:
    f.write(res_text)

print(f"Bhavishya V31 Engine Complete: {len(tdf)} trades, WR {wr:.2f}%, Gross Rs {gross:.2f}, Friction Rs {chg:.2f}, Net Rs {net:.2f}, PF {pf}, DD {max_dd:.2f}%, Final Wallet Rs {cum.iloc[-1] if len(cum) > 0 else initial_equity:.2f}")
