import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

# 1. Load H1 2025 Data (January 1, 2025 to June 30, 2025)
p_2025 = ROOT_DIR / "data" / "2025-26"
files = list(p_2025.glob("*_15m_1y.pkl"))

data_h1_2025 = {}
for f in files:
    sym = f.name.replace("_15m_1y.pkl", "").upper() + ".NS"
    with open(f, "rb") as fp:
        df = pickle.load(fp)
    df_h1 = df.loc["2025-01-01":"2025-06-30"].copy()
    if len(df_h1) > 500:
        data_h1_2025[sym] = df_h1

print(f"Loaded {len(data_h1_2025)} stocks for H1 2025 (Jan 1, 2025 to Jun 30, 2025)")

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
for t, df in data_h1_2025.items():
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    atr14 = calculate_atr(df, 14)
    atr_pct = (atr14 / c) * 100.0
    ema50 = c.ewm(span=50).mean()
    ema110 = c.ewm(span=110).mean()
    ema9 = c.ewm(span=9).mean()
    vwap_lag = compute_vwap_shifted(df)
    v_sma20 = v.rolling(20).mean()
    rvol = v / (v_sma20 + 1e-9)
    er10 = compute_er10(c)
    
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(7).mean()
    loss = (-delta.clip(upper=0)).rolling(7).mean()
    rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))
    
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()
    macd_cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    
    adx, pdi, mdi = compute_adx_di(df, 10)
    lookback = min(300, len(c) - 1)
    mom_cond = (c > c.shift(lookback))
    
    g4_cond = (
        macd_cross_up & (macd > 0) &
        (c > ema50) & (c > ema110) &
        (c > vwap_lag) &
        (rsi7 > 50.0) & (adx >= 18.0) & (pdi > mdi) &
        (er10 >= 0.35) &
        (rvol >= 1.2) &
        (atr_pct >= 0.50) &
        mom_cond &
        (c <= 450.0) & (c >= 10.0)
    )
    
    df["_date"] = df.index.date
    day_open = df.groupby("_date")["open"].transform("first")
    df["day_ret"] = (df["close"] / day_open) - 1.0
    df["1h_ret"] = (df["close"] / df["close"].shift(4)) - 1.0
    
    data_map[t] = {
        "df": df, "c": c, "h": h, "l": l, "v": v, "atr14": atr14, "atr_pct": atr_pct,
        "ema9": ema9, "rsi7": rsi7, "vwap_lag": vwap_lag, "rvol": rvol, "er10": er10,
        "g4_cond": g4_cond, "day_ret": df["day_ret"], "1h_ret": df["1h_ret"]
    }

bench_day_ret_series = pd.concat([d["day_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
bench_1h_ret_series = pd.concat([d["1h_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)

for t, d in data_map.items():
    b_day = bench_day_ret_series.reindex(d["day_ret"].index)
    b_1h = bench_1h_ret_series.reindex(d["1h_ret"].index)
    d["g1_cond"] = (d["day_ret"] > b_day)
    d["g2_cond"] = (d["1h_ret"] > b_1h)
    d["excess_1h"] = d["1h_ret"] - b_1h

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
    t_time_int = t_time.hour * 100 + t_time.minute
    
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
        cur_v = row["volume"]
        cur_ema9 = t_data["ema9"].loc[ts]
        
        p["bars_held"] += 1
        p["high_seen"] = max(p["high_seen"], cur_h)
        mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
        
        # Exact Bhavishya V26 Extended Exit Ladder
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
            
        # Velocity-gated decay protection
        if mfe_atr >= 0.85 and cur_c < cur_ema9 and cur_rsi < 55.0:
            stop_ladder += 0.05 * p["entry_atr"]
            
        p["active_sl"] = max(p["active_sl"], round(stop_ladder, 2))
        
        def finalize_trade(exit_price_raw, exit_reason):
            tot_qty = p["qty"]
            if cur_v > 0 and tot_qty > int(0.045 * cur_v):
                tot_qty = max(1, int(0.045 * cur_v))
            exit_fill = round(max(cur_l, min(cur_h, exit_price_raw)), 2)
            gross = round(tot_qty * (exit_fill - p["entry_p"]), 2)
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
                "exit_reason": exit_reason,
                "bars": p["bars_held"],
                "mfe": mfe_atr
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
            
    # 2. Equity & Margin
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
    
    # 3. Candidate Selection
    candidates = []
    for ticker in data_map.keys():
        if ticker in positions or ticker in daily_traded[cur_date]:
            continue
        t_data = data_map[ticker]
        if ts not in t_data["df"].index:
            continue
        idx = t_data["df"].index.get_loc(ts)
        if idx < 100 or not t_data["g4_cond"].iloc[idx]:
            continue
        if not (t_data["g1_cond"].iloc[idx] and t_data["g2_cond"].iloc[idx]):
            continue
            
        cur_c = t_data["c"].iloc[idx]
        cur_h = t_data["h"].iloc[idx]
        cur_l = t_data["l"].iloc[idx]
        cur_v = t_data["v"].iloc[idx]
        entry_atr = max(t_data["atr14"].iloc[idx], 0.05)
        excess_1h = t_data["excess_1h"].iloc[idx]
        
        candidates.append({
            "ticker": ticker, "c": cur_c, "h": cur_h, "l": cur_l, "v": cur_v,
            "atr": entry_atr, "score": excess_1h
        })
        
    if not candidates:
        continue
        
    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]
    ticker = best["ticker"]
    cur_c = best["c"]
    cur_h = best["h"]
    cur_l = best["l"]
    cur_v = best["v"]
    entry_atr = best["atr"]
    
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
print("\n" + "="*70)
print("BHAVISHYA V26 PURE UNSEEN AUDIT: H1 2025 (JAN 1, 2025 - JUN 30, 2025)")
print("="*70)
print(f"Total Trades: {len(tdf)}")
if len(tdf) > 0:
    wins = tdf[tdf["net_pnl"] > 0]
    losses = tdf[tdf["net_pnl"] < 0]
    wr = len(wins) / len(tdf) * 100.0
    net = tdf["net_pnl"].sum()
    gross = tdf["gross_pnl"].sum()
    chg = tdf["charges"].sum()
    cum = initial_equity + tdf["net_pnl"].cumsum()
    hwm = cum.cummax()
    dd = (hwm - cum) / hwm * 100.0
    max_dd = dd.max()
    net_win = wins["net_pnl"].sum()
    net_loss = abs(losses["net_pnl"].sum())
    pf = round(net_win / net_loss, 2) if net_loss > 0 else 999.0
    
    print(f"Wins: {len(wins)} | Losses: {len(losses)} | Win Rate: {wr:.2f}%")
    print(f"Gross Profit: Rs {gross:.2f} | Dhan Charges: Rs {chg:.2f}")
    print(f"Net Realized Profit: Rs {net:.2f} | Final Wallet: Rs {cum.iloc[-1]:.2f} ({(cum.iloc[-1]/initial_equity):.2f}x)")
    print(f"Profit Factor: {pf} | Max Drawdown: {max_dd:.2f}%")
    print(f"Avg Win: Rs {wins['net_pnl'].mean():.2f} | Avg Loss: Rs {losses['net_pnl'].mean():.2f}")
    print(f"Exit Reasons: {tdf['exit_reason'].value_counts().to_dict()}")
    print("\nSymbol Breakdown in H1 2025:")
    print(tdf.groupby("symbol")["net_pnl"].agg(["count", "sum", lambda x: (x>0).mean()*100]).rename(columns={"<lambda_0>": "WR%"}).sort_values("sum", ascending=False).to_string())
else:
    print("Zero trades fired in H1 2025!")
