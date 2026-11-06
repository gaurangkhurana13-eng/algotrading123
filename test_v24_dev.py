import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

# Load Data Pools
with open(ROOT_DIR / "data" / "2024-25" / "universe_14_15m.pkl", "rb") as f:
    univ_2024 = pickle.load(f)
with open(ROOT_DIR / "data" / "2025-26" / "universe_14_15m.pkl", "rb") as f:
    univ_2025 = pickle.load(f)

with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    u1 = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    u2 = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
    u3 = pickle.load(f)

pool_2026 = {}
pool_2026.update(u1)
pool_2026.update(u2)
pool_2026.update(u3)

tickers_combined = [
    "PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS", "RITES.NS",
    "IDEA.NS", "MTNL.NS", "ZEEL.NS", "UTKARSHBNK.NS", "IFCI.NS", "NCC.NS",
    "TEXRAIL.NS", "JPPOWER.NS", "PTC.NS", "NIACL.NS"
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

print("Preparing data...")
data_map = {}
for t in tickers_combined:
    raw_s = t.replace(".NS", "")
    df24 = univ_2024.get(t, univ_2024.get(raw_s))
    df25 = univ_2025.get(t, univ_2025.get(raw_s))
    df26 = pool_2026.get(t, pool_2026.get(raw_s))
    
    parts = [p for p in [df24, df25, df26] if p is not None]
    if not parts:
        continue
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    atr14 = calculate_atr(df, 14)
    atr_pct = (atr14 / c) * 100.0
    ema50 = c.ewm(span=50).mean()
    ema110 = c.ewm(span=110).mean()
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
    
    lookback = min(300, len(c) - 1)
    mom_cond = (c > c.shift(lookback))
    
    g4_cond = (
        macd_cross_up & (macd > 0) &
        (c > ema50) & (c > ema110) &
        (c > vwap_lag) &
        (rsi7 > 50.0) & (adx >= 18.0) & (plus_di > minus_di) &
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
    
    df_daily = df.resample("1D").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    data_map[t] = {
        "df": df, "c": c, "h": h, "l": l, "v": v, "atr14": atr14, "atr_pct": atr_pct,
        "rsi7": rsi7, "vwap_lag": vwap_lag, "rvol": rvol, "er10": er10,
        "g4_cond": g4_cond, "df_daily": df_daily,
        "day_ret": df["day_ret"], "1h_ret": df["1h_ret"]
    }

all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
bench_day_ret_series = pd.concat([d["day_ret"] for d in data_map.values()], axis=1).mean(axis=1)
bench_1h_ret_series = pd.concat([d["1h_ret"] for d in data_map.values()], axis=1).mean(axis=1)

for t, d in data_map.items():
    b_day = bench_day_ret_series.reindex(d["day_ret"].index)
    b_1h = bench_1h_ret_series.reindex(d["1h_ret"].index)
    d["g1_cond"] = (d["day_ret"] > b_day)
    d["g2_cond"] = (d["1h_ret"] > b_1h)
    d["excess_1h"] = d["1h_ret"] - b_1h

all_months = sorted(list(set(ts.strftime("%Y-%m") for ts in all_ts)))
monthly_universe = {}

for m_idx, m_str in enumerate(all_months):
    if m_idx == 0:
        continue
    prev_m_str = all_months[m_idx - 1]
    prev_bars = [ts for ts in all_ts if ts.strftime("%Y-%m") == prev_m_str]
    if not prev_bars:
        continue
    t_m = prev_bars[-1]
    
    scores = []
    bench_r20_list = []
    for t, d in data_map.items():
        sub_d = d["df_daily"].loc[:t_m]
        if len(sub_d) >= 21:
            P = sub_d["close"].iloc[-1]
            bench_r20_list.append(P / sub_d["close"].iloc[-21] - 1.0)
    bench_r20 = np.mean(bench_r20_list) if bench_r20_list else 0.0
    
    for t, d in data_map.items():
        sub_d = d["df_daily"].loc[:t_m]
        if len(sub_d) < 70:
            continue
        P = sub_d["close"].iloc[-1]
        if not (10.0 <= P <= 450.0):
            continue
        adv20 = (sub_d["close"] * sub_d["volume"]).tail(20).median()
        if adv20 < 5e7:
            continue
        h_d, l_d, c_prev_d = sub_d["high"], sub_d["low"], sub_d["close"].shift(1)
        tr_d = np.maximum(h_d - l_d, np.maximum((h_d - c_prev_d).abs(), (l_d - c_prev_d).abs()))
        atr14_d = tr_d.ewm(alpha=1/14).mean().iloc[-1]
        if (atr14_d / P) < 0.005:
            continue
            
        r20 = P / sub_d["close"].iloc[-21] - 1.0
        r60 = P / sub_d["close"].iloc[-61] - 1.0
        ema20 = sub_d["close"].ewm(span=20).mean().iloc[-1]
        e = P / ema20 - 1.0
        x20 = r20 - bench_r20
        scores.append({"ticker": t, "r20": r20, "r60": r60, "e": e, "x20": x20})
        
    N = len(scores)
    if N > 0:
        for feat in ["r20", "r60", "e", "x20"]:
            vals = sorted([s[feat] for s in scores])
            for s in scores:
                rank = (vals.index(s[feat]) + len(vals) - 1 - list(reversed(vals)).index(s[feat])) / 2.0 + 1.0
                s[f"p_{feat}"] = (rank - 0.5) / N
        for s in scores:
            s["RS"] = 0.35 * s["p_r20"] + 0.30 * s["p_r60"] + 0.15 * s["p_e"] + 0.20 * s["p_x20"]
        scores.sort(key=lambda x: x["RS"], reverse=True)
        cutoff_k = min(20, max(7, int(0.60 * N)))
        monthly_universe[m_str] = {s["ticker"]: s["RS"] for s in scores[:cutoff_k]}
    else:
        monthly_universe[m_str] = {t: 0.5 for t in data_map.keys()}

def run_simulation(start_date, end_date, sizing_mode="deepseek_1_1", initial_equity=500.0):
    ts_range = [ts for ts in all_ts if start_date <= ts <= end_date and ts.weekday() < 5]
    positions = {}
    trades = []
    daily_traded = {}
    last_exit_times = set()
    slippage = 0.0005
    hard_sl_mult = 0.90
    alpha_streak = 0.082
    beta_streak = 0.035
    
    for ts in ts_range:
        cur_date = ts.date()
        m_str = ts.strftime("%Y-%m")
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
            
            p["bars_held"] += 1
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
            
            # DeepSeek Structural Accelerated Ratchet
            if mfe_atr >= 2.50:
                p["active_sl"] = max(p["active_sl"], round(p["high_seen"] - (0.12 * p["entry_atr"]), 2))
            elif mfe_atr >= 1.50 or (mfe_atr >= 1.15 and t_time_int >= 1415):
                p["active_sl"] = max(p["active_sl"], round(p["high_seen"] - (0.18 * p["entry_atr"]), 2))
            elif mfe_atr >= 1.15:
                p["active_sl"] = max(p["active_sl"], round(p["high_seen"] - (0.28 * p["entry_atr"]), 2))
            elif mfe_atr >= 0.85:
                p["active_sl"] = max(p["active_sl"], round(p["entry_p"] + (0.40 * p["entry_atr"]), 2))
                
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
                
            # Stop Hit
            if cur_l <= p["active_sl"]:
                finalize_trade(p["active_sl"] * (1.0 - slippage), "STOP")
                continue
                
            # Stagnation scratch
            if p["bars_held"] >= 3 and mfe_atr < 0.40 and cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                finalize_trade(cur_c * (1.0 - slippage), "SCRATCH")
                continue
                
            # 14:45 EOD
            if t_time >= pd.Timestamp("14:45").time():
                finalize_trade(cur_c * (1.0 - slippage), "EOD")
                continue
                
        # 2. Equity & Throttle Check
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
        
        if sizing_mode == "deepseek_1_1":
            f_eff = 0.011 if dd_pct < 0.10 else (0.009 if dd_pct < 0.15 else 0.005)
            target_risk = f_eff * ek
        else:
            # Dynamic Compounding Engine (like V23)
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
            
        # Candidates
        candidates = []
        allowed_universe = monthly_universe.get(m_str, {})
        
        for ticker in tickers_combined:
            if ticker in positions or ticker in daily_traded[cur_date] or ticker not in allowed_universe:
                continue
            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            idx = t_data["df"].index.get_loc(ts)
            if idx < 100 or not t_data["g4_cond"].iloc[idx]:
                continue
                
            # G1 & G2 check
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
                "atr": entry_atr, "excess_1h": excess_1h
            })
            
        if not candidates:
            continue
            
        candidates.sort(key=lambda x: x["excess_1h"], reverse=True)
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
    return tdf

def print_metrics(tdf, label, initial_equity=500.0):
    if len(tdf) == 0:
        print(f"{label}: 0 trades!")
        return
    wins = tdf[tdf["net_pnl"] > 0]
    losses = tdf[tdf["net_pnl"] <= 0]
    wr = len(wins) / len(tdf) * 100.0
    gross = tdf["gross_pnl"].sum()
    chg = tdf["charges"].sum()
    net = tdf["net_pnl"].sum()
    cum = initial_equity + tdf["net_pnl"].cumsum()
    hwm = cum.cummax()
    dd = (hwm - cum) / hwm * 100.0
    max_dd = dd.max() if len(dd) > 0 else 0.0
    print(f"{label}: Trades={len(tdf)}, WR={wr:.1f}%, Gross=+Rs.{gross:.2f}, Chg=-Rs.{chg:.2f}, Net=+Rs.{net:.2f}, MaxDD={max_dd:.2f}%, Final=Rs.{cum.iloc[-1]:.2f}")

print("\n=== JUNE - DEC 2025 (UNSEEN MONSOON CHOP) ===")
tdf_2025_ds = run_simulation(pd.Timestamp("2025-06-01 09:15:00"), pd.Timestamp("2025-12-31 15:30:00"), sizing_mode="deepseek_1_1")
tdf_2025_dyn = run_simulation(pd.Timestamp("2025-06-01 09:15:00"), pd.Timestamp("2025-12-31 15:30:00"), sizing_mode="dynamic_compounding")
print_metrics(tdf_2025_ds, "2025 Chop [DeepSeek 1.1% Risk]")
print_metrics(tdf_2025_dyn, "2025 Chop [Dynamic Compounding]")

print("\n=== JAN - OCT 2026 (MODERN MARKET) ===")
tdf_2026_ds = run_simulation(pd.Timestamp("2026-01-01 09:15:00"), pd.Timestamp("2026-10-29 15:30:00"), sizing_mode="deepseek_1_1")
tdf_2026_dyn = run_simulation(pd.Timestamp("2026-01-01 09:15:00"), pd.Timestamp("2026-10-29 15:30:00"), sizing_mode="dynamic_compounding")
print_metrics(tdf_2026_ds, "2026 Modern [DeepSeek 1.1% Risk]")
print_metrics(tdf_2026_dyn, "2026 Modern [Dynamic Compounding]")
