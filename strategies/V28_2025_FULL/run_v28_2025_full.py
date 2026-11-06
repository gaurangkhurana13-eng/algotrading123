import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

# Load 2025-26 unseen pool
p_2025 = ROOT_DIR / "data" / "2025-26"
files_2025 = list(p_2025.glob("*_15m_1y.pkl"))

data_map = {}
for f in files_2025:
    sym = f.name.replace("_15m_1y.pkl", "").upper() + ".NS"
    with open(f, "rb") as fp:
        df = pickle.load(fp)
    df = df.loc["2025-01-01":"2025-12-31"].copy()
    if len(df) < 200:
        continue
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
    
    diff = c.diff()
    gain = diff.where(diff > 0, 0.0).rolling(7).mean()
    loss = (-diff.where(diff < 0, 0.0)).rolling(7).mean()
    rs = gain / (loss + 1e-9)
    rsi7 = 100 - (100 / (1 + rs))
    
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    macd_cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1)) & (macd_line > 0)
    
    up = h.diff()
    dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    atr_s = tr.rolling(10).mean()
    pdi = 100 * pd.Series(pdm, index=df.index).rolling(10).mean() / (atr_s + 1e-9)
    mdi = 100 * pd.Series(mdm, index=df.index).rolling(10).mean() / (atr_s + 1e-9)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    adx = dx.rolling(10).mean()
    
    df["_date"] = df.index.date
    tp = (h + l + c) / 3.0
    df["_tpv"] = tp * v
    cum_tpv = df.groupby("_date")["_tpv"].cumsum()
    cum_vol = df.groupby("_date")["volume"].cumsum()
    vwap_lag = (cum_tpv / (cum_vol + 1e-9)).shift(1)
    
    vol_sma20 = v.rolling(20).mean()
    rvol = v / (vol_sma20 + 1e-9)
    change10 = (c - c.shift(10)).abs()
    volat10 = c.diff().abs().rolling(10).sum()
    er10 = change10 / (volat10 + 1e-9)
    
    high_5 = h.rolling(5).max().shift(1)
    lookback = min(300, len(c) - 1)
    mom_long = c > c.shift(lookback)
    
    g4_long = (
        macd_cross_up &
        (c > ema50) & (c > ema110) & (c > ema9) & (c > ema60) &
        (c > vwap_lag) &
        (er10 >= 0.35) &
        (rsi7 >= 48.0) & (rsi7 <= 78.0) &
        (rvol >= 1.25) &
        (atr_pct >= 0.50) &
        (adx >= 18.0) & (pdi > mdi) &
        mom_long &
        (c > high_5) &
        (c <= 450.0) & (c >= 10.0)
    )
    
    day_open = df.groupby("_date")["open"].transform("first")
    df["day_ret"] = (c / day_open) - 1.0
    df["1h_ret"] = (c / c.shift(4)) - 1.0
    above_20dma = (c > c.rolling(20).mean()).astype(float)
    
    data_map[sym] = {
        "df": df, "c": c, "h": h, "l": l, "v": v, "atr14": atr14, "atr_pct": atr_pct,
        "ema9": ema9, "rsi7": rsi7, "vwap_lag": vwap_lag, "rvol": rvol, "er10": er10,
        "g4_long": g4_long, "day_ret": df["day_ret"], "1h_ret": df["1h_ret"],
        "above_20dma": above_20dma
    }

bench_day_ret_series = pd.concat([d["day_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
bench_1h_ret_series = pd.concat([d["1h_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
market_breadth_series = pd.concat([d["above_20dma"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)

for t, d in data_map.items():
    b_day = bench_day_ret_series.reindex(d["day_ret"].index)
    b_1h = bench_1h_ret_series.reindex(d["1h_ret"].index)
    d["g1_long"] = (d["day_ret"] > b_day)
    d["g2_long"] = (d["1h_ret"] > b_1h)
    d["excess_1h_long"] = d["1h_ret"] - b_1h

all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
slippage = 0.0005
hard_sl_mult = 0.90
initial_equity = 500.0

positions = {}
trades = []
daily_traded = {}
last_exit_times = set()

for ts in all_ts:
    if ts.weekday() >= 5:
        continue
    cur_date = ts.date()
    if cur_date not in daily_traded:
        daily_traded[cur_date] = set()
    t_time = ts.time()
    
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
        p["high_seen"] = max(p["high_seen"], cur_h)
        mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
        
        if mfe_atr >= 4.00:
            stop_ladder = p["high_seen"] - (0.10 * p["entry_atr"])
        elif mfe_atr >= 3.00:
            stop_ladder = p["high_seen"] - (0.14 * p["entry_atr"])
        elif mfe_atr >= 2.00:
            stop_ladder = p["high_seen"] - (0.20 * p["entry_atr"])
        elif mfe_atr >= 1.50:
            stop_ladder = p["high_seen"] - (0.25 * p["entry_atr"])
        elif mfe_atr >= 1.15:
            stop_ladder = p["high_seen"] - (0.28 * p["entry_atr"])
        elif mfe_atr >= 0.85:
            stop_ladder = p["entry_p"] + (0.40 * p["entry_atr"])
        else:
            stop_ladder = p["entry_p"] - (0.90 * p["entry_atr"])
            
        if mfe_atr >= 0.85 and cur_c < cur_ema9 and cur_rsi < 55.0:
            stop_ladder += 0.05 * p["entry_atr"]
            
        p["active_sl"] = max(p["active_sl"], round(stop_ladder, 2))
        
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
            
        if cur_l <= p["active_sl"]:
            finalize_trade(p["active_sl"] * (1.0 - slippage), "STOP")
            continue
        if p["bars_held"] >= 4 and mfe_atr < 0.40 and cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
            finalize_trade(cur_c * (1.0 - slippage), "SCRATCH")
            continue
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
    
    if len(positions) >= 1 or ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
        continue
        
    current_notional = sum(p["entry_p"] * p["qty"] for p in positions.values())
    avail_margin = (0.95 * ek) - (current_notional / 5.0)
    if avail_margin < 10.0:
        continue
    avail_notional = avail_margin * 5.0
    
    br = market_breadth_series.get(ts, 0.5)
    f_size = 1.0 if br >= 0.40 else (0.50 if br >= 0.20 else 0.30)
    
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
        alpha, beta = 0.082, 0.035
        
    if ek <= 2500.0:
        raw_r = 14.0 + (alpha * max(0.0, ek - 500.0))
    else:
        raw_r = 14.0 + (alpha * 2000.0) + (beta * (ek - 2500.0))
        
    raw_r *= f_size
    target_risk = min(raw_r, min(0.09 * ek, 258.0))
    if dd_pct > 0.15:
        target_risk *= 0.5
        
    candidates = []
    for ticker, t_data in data_map.items():
        if ticker in positions or ticker in daily_traded[cur_date]:
            continue
        if ts not in t_data["df"].index:
            continue
        idx = t_data["df"].index.get_loc(ts)
        if idx < 100 or not t_data["g4_long"].iloc[idx]:
            continue
        if not (t_data["g1_long"].iloc[idx] and t_data["g2_long"].iloc[idx]):
            continue
        candidates.append((ticker, t_data["c"].iloc[idx], t_data["h"].iloc[idx], t_data["l"].iloc[idx], t_data["v"].iloc[idx], max(t_data["atr14"].iloc[idx], 0.05), t_data["excess_1h_long"].iloc[idx]))
        
    if not candidates:
        continue
    candidates.sort(key=lambda x: x[6], reverse=True)
    ticker, cur_c, cur_h, cur_l, cur_v, entry_atr, _ = candidates[0]
    
    raw_entry = cur_c * (1.0 + slippage)
    entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
    active_sl = round(entry_p - (hard_sl_mult * entry_atr), 2)
    sl_dist = max(entry_p - active_sl, 0.05)
    friction = 0.002067 * entry_p
    denom = sl_dist + friction
    
    qty_r = int(target_risk / denom)
    qty_m = int((avail_notional * 0.90) / entry_p)
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

tdf = pd.DataFrame(trades)
v28_full_dir = ROOT_DIR / "strategies" / "V28_2025_FULL"
v28_full_dir.mkdir(parents=True, exist_ok=True)
tdf.to_csv(v28_full_dir / "trades_intraday.csv", index=False)

swing_cols = ["symbol", "direction", "entry_time", "exit_time", "entry_price", "exit_price", "shares", "sl", "gross_pnl", "charges", "net_pnl", "exit_reason"]
pd.DataFrame(columns=swing_cols).to_csv(v28_full_dir / "trades_swing.csv", index=False)

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

with open(v28_full_dir / "results.txt", "w") as f:
    f.write(res_text)

print(f"V28 2025 Full Engine: {len(tdf)} trades, WR {wr:.2f}%, Net Rs {net:.2f}, PF {pf}, DD {max_dd:.2f}%, Wallet Rs {cum.iloc[-1]:.2f}")
