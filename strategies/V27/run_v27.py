import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

# 1. Load Pools
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

data_map = {}
for t in tickers_combined:
    if t not in all_pools:
        continue
    df = all_pools[t].copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    atr14 = calculate_atr(df, 14)
    atr_pct = (atr14 / c) * 100.0
    ema9 = c.ewm(span=9).mean()
    ema20 = c.ewm(span=20).mean()
    ema50 = c.ewm(span=50).mean()
    ema60 = c.ewm(span=60).mean()
    ema110 = c.ewm(span=110).mean()
    
    # RSI 7
    diff = c.diff()
    gain = diff.where(diff > 0, 0.0).rolling(7).mean()
    loss = (-diff.where(diff < 0, 0.0)).rolling(7).mean()
    rs = gain / (loss + 1e-9)
    rsi7 = 100 - (100 / (1 + rs))
    
    vwap_lag = compute_vwap_shifted(df)
    vol_sma20 = v.rolling(20).mean()
    rvol = v / (vol_sma20 + 1e-9)
    er10 = compute_er10(c)
    
    lookback_mom = min(300, len(c) - 1)
    mom_long = c > c.shift(lookback_mom)
    mom_short = c < c.shift(lookback_mom)
    
    g4_long = (
        (c > ema50) & (c > ema110) & (c > ema9) & (c > ema60) &
        (c > vwap_lag) &
        (er10 >= 0.35) &
        (rsi7 >= 48.0) & (rsi7 <= 78.0) &
        (rvol >= 1.2) &
        (atr_pct >= 0.50) &
        mom_long &
        (c <= 450.0) & (c >= 10.0)
    )
    
    g4_short = (
        (c < ema50) & (c < ema110) & (c < ema9) & (c < ema60) &
        (c < vwap_lag) &
        (er10 >= 0.35) &
        (rsi7 <= 52.0) & (rsi7 >= 22.0) &
        (rvol >= 1.2) &
        (atr_pct >= 0.50) &
        mom_short &
        (c <= 450.0) & (c >= 10.0)
    )
    
    df["_date"] = df.index.date
    day_open = df.groupby("_date")["open"].transform("first")
    df["day_ret"] = (df["close"] / day_open) - 1.0
    df["1h_ret"] = (df["close"] / df["close"].shift(4)) - 1.0
    
    data_map[t] = {
        "df": df, "c": c, "h": h, "l": l, "v": v, "atr14": atr14, "atr_pct": atr_pct,
        "ema9": ema9, "rsi7": rsi7, "vwap_lag": vwap_lag, "rvol": rvol, "er10": er10,
        "g4_long": g4_long, "g4_short": g4_short,
        "day_ret": df["day_ret"], "1h_ret": df["1h_ret"]
    }

bench_day_ret_series = pd.concat([d["day_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
bench_1h_ret_series = pd.concat([d["1h_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)

for t, d in data_map.items():
    b_day = bench_day_ret_series.reindex(d["day_ret"].index)
    b_1h = bench_1h_ret_series.reindex(d["1h_ret"].index)
    d["g1_long"] = (d["day_ret"] > b_day)
    d["g2_long"] = (d["1h_ret"] > b_1h)
    d["excess_1h_long"] = d["1h_ret"] - b_1h
    
    d["g1_short"] = (d["day_ret"] < b_day)
    d["g2_short"] = (d["1h_ret"] < b_1h)
    d["excess_1h_short"] = b_1h - d["1h_ret"]

all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
slippage = 0.0005
hard_sl_mult = 0.90
initial_equity = 500.0

positions = {}
trades = []
daily_traded = {}
last_exit_times = set()

alpha_streak = 0.082
beta_streak = 0.035

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
        
        p["bars_held"] += 1
        direction = p["direction"]
        
        def finalize_trade(exit_fill_price, exit_reason):
            exit_fill = round(exit_fill_price, 2)
            tot_qty = p["qty"]
            if direction == "LONG":
                gross = round((exit_fill - p["entry_p"]) * tot_qty, 2)
            else:
                gross = round((p["entry_p"] - exit_fill) * tot_qty, 2)
                
            chg = round(calc_dhan_charges(p["entry_p"], exit_fill, tot_qty, is_intraday=True), 2)
            net = round(gross - chg, 2)
            trades.append({
                "symbol": ticker,
                "direction": direction,
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
            
        if direction == "LONG":
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
            
            # V26 Extended Climax Ladder
            if mfe_atr >= 3.00:
                stop_ladder = p["high_seen"] - (0.14 * p["entry_atr"])
            elif mfe_atr >= 2.00:
                stop_ladder = p["high_seen"] - (0.20 * p["entry_atr"])
            elif mfe_atr >= 1.15:
                stop_ladder = p["high_seen"] - (0.28 * p["entry_atr"])
            elif mfe_atr >= 0.85:
                stop_ladder = p["entry_p"] + (0.40 * p["entry_atr"])
            else:
                stop_ladder = p["entry_p"] - (0.90 * p["entry_atr"])
                
            if mfe_atr >= 0.85 and cur_c < cur_ema9 and cur_rsi < 55.0:
                stop_ladder += 0.05 * p["entry_atr"]
                
            p["active_sl"] = max(p["active_sl"], round(stop_ladder, 2))
            
            # Check stop loss
            if cur_l <= p["active_sl"]:
                finalize_trade(p["active_sl"] * (1.0 - slippage), "STOP")
                continue
            # Stagnation scratch
            if p["bars_held"] >= 4 and mfe_atr < 0.40 and cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                finalize_trade(cur_c * (1.0 - slippage), "SCRATCH")
                continue
            # EOD square-off at 14:45
            if t_time >= pd.Timestamp("14:45").time():
                finalize_trade(cur_c * (1.0 - slippage), "EOD")
                continue
                
        elif direction == "SHORT":
            p["low_seen"] = min(p["low_seen"], cur_l)
            mfe_atr = (p["entry_p"] - p["low_seen"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
            
            # Symmetric Short Climax Ladder
            if mfe_atr >= 3.00:
                stop_ladder = p["low_seen"] + (0.14 * p["entry_atr"])
            elif mfe_atr >= 2.00:
                stop_ladder = p["low_seen"] + (0.20 * p["entry_atr"])
            elif mfe_atr >= 1.15:
                stop_ladder = p["low_seen"] + (0.28 * p["entry_atr"])
            elif mfe_atr >= 0.85:
                stop_ladder = p["entry_p"] - (0.40 * p["entry_atr"])
            else:
                stop_ladder = p["entry_p"] + (0.90 * p["entry_atr"])
                
            if mfe_atr >= 0.85 and cur_c > cur_ema9 and cur_rsi > 45.0:
                stop_ladder -= 0.05 * p["entry_atr"]
                
            p["active_sl"] = min(p["active_sl"], round(stop_ladder, 2))
            
            # Check stop loss
            if cur_h >= p["active_sl"]:
                finalize_trade(p["active_sl"] * (1.0 + slippage), "STOP")
                continue
            # Stagnation scratch
            if p["bars_held"] >= 4 and mfe_atr < 0.40 and cur_c > p["entry_p"] and (cur_rsi > 54.0 or cur_c > cur_vwap):
                finalize_trade(cur_c * (1.0 + slippage), "SCRATCH")
                continue
            # EOD square-off at 14:45
            if t_time >= pd.Timestamp("14:45").time():
                finalize_trade(cur_c * (1.0 + slippage), "EOD")
                continue

    # 2. Equity & Margin Management
    ek = initial_equity + sum(t["net_pnl"] for t in trades)
    hwm = initial_equity
    cum = initial_equity
    for t in trades:
        cum += t["net_pnl"]
        if cum > hwm:
            hwm = cum
    dd_pct = (hwm - ek) / max(hwm, initial_equity)
    
    if len(positions) >= 1 or ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
        continue
        
    current_notional = sum(p["entry_p"] * p["qty"] for p in positions.values())
    avail_margin = (0.95 * ek) - (current_notional / 5.0)
    if avail_margin < 10.0:
        continue
    avail_notional = avail_margin * 5.0
    
    consec_losses = 0
    for t in reversed(trades):
        if t["net_pnl"] < 0:
            consec_losses += 1
        else:
            break
            
    if consec_losses >= 2 or (ek > 2500.0 and dd_pct > 0.16):
        alpha, beta = 0.015, 0.010
    elif consec_losses == 1:
        alpha, beta = 0.035, 0.020
    else:
        alpha, beta = alpha_streak, beta_streak
        
    if ek <= 2500.0:
        raw_r = 14.0 + (alpha * max(0.0, ek - 500.0))
    else:
        raw_r = 14.0 + (alpha * 2000.0) + (beta * (ek - 2500.0))
        
    gate_v2_cap = min(0.09 * ek, 258.0)
    if dd_pct > 0.15:
        gate_v2_cap *= 0.5
    target_risk = min(raw_r, gate_v2_cap)
    
    # 3. Candidate Selection (Bi-Directional Long + Short)
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
            
        cur_c = t_data["c"].iloc[idx]
        cur_h = t_data["h"].iloc[idx]
        cur_l = t_data["l"].iloc[idx]
        cur_v = t_data["v"].iloc[idx]
        entry_atr = max(t_data["atr14"].iloc[idx], 0.05)
        
        # Check Long candidate
        if t_data["g4_long"].iloc[idx] and t_data["g1_long"].iloc[idx] and t_data["g2_long"].iloc[idx]:
            excess = t_data["excess_1h_long"].iloc[idx]
            candidates.append({
                "ticker": ticker, "direction": "LONG", "c": cur_c, "h": cur_h, "l": cur_l, "v": cur_v,
                "atr": entry_atr, "score": excess
            })
        # Check Short candidate
        elif t_data["g4_short"].iloc[idx] and t_data["g1_short"].iloc[idx] and t_data["g2_short"].iloc[idx]:
            excess = t_data["excess_1h_short"].iloc[idx]
            candidates.append({
                "ticker": ticker, "direction": "SHORT", "c": cur_c, "h": cur_h, "l": cur_l, "v": cur_v,
                "atr": entry_atr, "score": excess
            })
            
    if not candidates:
        continue
        
    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]
    ticker = best["ticker"]
    direction = best["direction"]
    cur_c = best["c"]
    cur_h = best["h"]
    cur_l = best["l"]
    cur_v = best["v"]
    entry_atr = best["atr"]
    
    if direction == "LONG":
        raw_entry = cur_c * (1.0 + slippage)
        entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
        active_sl = round(entry_p - (hard_sl_mult * entry_atr), 2)
        sl_dist = max(entry_p - active_sl, 0.05)
    else:
        raw_entry = cur_c * (1.0 - slippage)
        entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
        active_sl = round(entry_p + (hard_sl_mult * entry_atr), 2)
        sl_dist = max(active_sl - entry_p, 0.05)
        
    friction = 0.002067 * entry_p
    denom = sl_dist + friction
    
    qty_r = int(target_risk / denom)
    qty_m = int((avail_notional * 0.90) / entry_p)
    qty = min(qty_r, qty_m)
    if cur_v > 0:
        qty = min(qty, int(0.045 * cur_v))
        
    if qty >= 1:
        if direction == "LONG":
            positions[ticker] = {
                "direction": "LONG", "entry_p": entry_p, "qty": qty, "entry_atr": entry_atr,
                "active_sl": active_sl, "initial_sl": active_sl,
                "high_seen": entry_p, "bars_held": 0, "entry_t": ts
            }
        else:
            positions[ticker] = {
                "direction": "SHORT", "entry_p": entry_p, "qty": qty, "entry_atr": entry_atr,
                "active_sl": active_sl, "initial_sl": active_sl,
                "low_seen": entry_p, "bars_held": 0, "entry_t": ts
            }
        daily_traded[cur_date].add(ticker)

# 4. Export Outputs for Gate V2 Audit
tdf = pd.DataFrame(trades)
v27_dir = ROOT_DIR / "strategies" / "V27"
v27_dir.mkdir(parents=True, exist_ok=True)
tdf.to_csv(v27_dir / "trades_intraday.csv", index=False)

swing_cols = ["symbol", "direction", "entry_time", "exit_time", "entry_price", "exit_price", "shares", "sl", "gross_pnl", "charges", "net_pnl", "exit_reason"]
pd.DataFrame(columns=swing_cols).to_csv(v27_dir / "trades_swing.csv", index=False)

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

with open(v27_dir / "results.txt", "w") as f:
    f.write(res_text)

print(f"Bhavishya V27 Engine: {len(tdf)} trades, WR {wr:.2f}%, Gross Rs {gross:.2f}, Friction Rs {chg:.2f}, Net Rs {net:.2f}, PF {pf}, DD {max_dd:.2f}%, Final Wallet Rs {cum.iloc[-1] if len(cum) > 0 else initial_equity:.2f}")
