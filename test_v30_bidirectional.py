import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

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

def run_v30_simulation(data_map, bench_day_ret_series, market_breadth_series, start_date, end_date, name):
    all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
    target_ts = [ts for ts in all_ts if pd.Timestamp(start_date) <= ts <= pd.Timestamp(end_date)]
    
    slippage = 0.0005
    initial_equity = 500.0
    
    positions = {}
    trades = []
    daily_traded = {}
    last_exit_times = set()
    daily_short_count = {}
    
    alpha_streak = 0.082
    beta_streak = 0.035
    
    for ts in target_ts:
        if ts.weekday() >= 5:
            continue
        cur_date = ts.date()
        if cur_date not in daily_traded:
            daily_traded[cur_date] = set()
            daily_short_count[cur_date] = 0
        t_time = ts.time()
        
        # 1. Manage Active Positions (Long and Short)
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
            
            def finalize_trade(exit_fill, exit_reason):
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
                
                # V29 Macro Climax Ladder for Longs
                if mfe_atr >= 4.00:
                    stop_ladder = p["high_seen"] - (0.10 * p["entry_atr"])
                elif mfe_atr >= 3.00:
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
                
                if cur_l <= p["active_sl"]:
                    finalize_trade(p["active_sl"] * (1.0 - slippage), "STOP")
                    continue
                if p["bars_held"] >= 4 and mfe_atr < 0.40 and cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                    finalize_trade(cur_c * (1.0 - slippage), "SCRATCH")
                    continue
                if t_time >= pd.Timestamp("14:45").time():
                    finalize_trade(cur_c * (1.0 - slippage), "EOD")
                    continue
            else:
                # SHORT POSITION MANAGEMENT (Tight 1.5 ATR Ratchet)
                p["low_seen"] = min(p["low_seen"], cur_l)
                mfe_atr_short = (p["entry_p"] - p["low_seen"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
                
                # Rapid Dynamic Ratchet
                if mfe_atr_short >= 2.00:
                    stop_ladder_short = p["low_seen"] + (0.15 * p["entry_atr"])
                elif mfe_atr_short >= 1.00:
                    stop_ladder_short = p["entry_p"] - (0.30 * p["entry_atr"])
                elif mfe_atr_short >= 0.50:
                    stop_ladder_short = p["entry_p"]
                else:
                    stop_ladder_short = p["entry_p"] + (0.90 * p["entry_atr"])
                    
                p["active_sl"] = min(p["active_sl"], round(stop_ladder_short, 2))
                
                if cur_h >= p["active_sl"]:
                    finalize_trade(p["active_sl"] * (1.0 + slippage), "STOP")
                    continue
                if mfe_atr_short >= 2.20:
                    finalize_trade(cur_c * (1.0 + slippage), "TARGET_2ATR")
                    continue
                if p["bars_held"] >= 4 and mfe_atr_short < 0.30 and cur_c > p["entry_p"]:
                    finalize_trade(cur_c * (1.0 + slippage), "SCRATCH")
                    continue
                # Avoid 14:00+ short covering squeeze!
                if t_time >= pd.Timestamp("13:45").time():
                    finalize_trade(cur_c * (1.0 + slippage), "TIME_EXIT_SHORT")
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
        
        # Streak scaling for longs
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
        target_risk_long = min(raw_r, gate_v2_cap)
        
        # Short risk budget: fixed 2.5% of equity (controlled asymmetric risk)
        target_risk_short = min(0.025 * ek, 25.0)
        
        b_day_now = bench_day_ret_series.get(ts, 0.0)
        cur_breadth = market_breadth_series.get(ts, 0.5)
        
        # 3 REGIME CLASSIFIER:
        # R1: Bull Expansion -> b_day_now >= -0.006 and breadth >= 0.40
        # R2: Chop / Bleed -> b_day_now < -0.006 or breadth between 0.25 and 0.40
        # R3: Institutional Panic Crash Breakdown -> b_day_now < -0.012 and breadth < 0.25
        is_panic_crash = (b_day_now < -0.012 and cur_breadth < 0.25)
        is_severe_bleed = (b_day_now < -0.006)
        
        # Candidate selection
        long_candidates = []
        short_candidates = []
        
        for ticker in data_map.keys():
            if ticker in positions or ticker in daily_traded[cur_date]:
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
            excess_1h = t_data["excess_1h"].iloc[idx]
            
            # LONG GATE (V29)
            gate_long = t_data["g4_v28"].iloc[idx] if is_severe_bleed else t_data["g4_v26"].iloc[idx]
            if gate_long and t_data["g1_cond"].iloc[idx] and t_data["g2_cond"].iloc[idx] and not is_panic_crash:
                long_candidates.append((ticker, cur_c, cur_h, cur_l, cur_v, entry_atr, excess_1h))
                
            # SHORT GATE (BQS - Panic Breakdown only)
            if is_panic_crash and daily_short_count[cur_date] == 0 and t_time <= pd.Timestamp("12:45").time():
                # BQS Conditions:
                # 1. Red volume RVOL >= 1.50
                # 2. Close < 5-bar Low
                # 3. Relative Weakness: 1h_ret < bench_1h and excess_1h < -0.01
                # 4. ER10 >= 0.40
                # 5. Price below EMA50 and VWAP
                rvol = t_data["rvol"].iloc[idx]
                low_5 = t_data["df"]["low"].rolling(5).min().shift(1).iloc[idx]
                er10 = t_data["er10"].iloc[idx]
                vwap = t_data["vwap_lag"].iloc[idx]
                ema50 = t_data["df"]["close"].ewm(span=50).mean().iloc[idx]
                
                bqs_pass = (
                    (rvol >= 1.50) &
                    (cur_c < low_5) &
                    (excess_1h < -0.010) &
                    (er10 >= 0.38) &
                    (cur_c < vwap) &
                    (cur_c < ema50) &
                    (cur_c <= 450.0) & (cur_c >= 10.0)
                )
                if bqs_pass:
                    short_candidates.append((ticker, cur_c, cur_h, cur_l, cur_v, entry_atr, excess_1h))
                    
        # Execute Best Candidate
        if short_candidates:
            # Sort by highest relative weakness (lowest excess_1h)
            short_candidates.sort(key=lambda x: x[6])
            best = short_candidates[0]
            ticker, cur_c, cur_h, cur_l, cur_v, entry_atr, _ = best
            
            raw_entry = cur_c * (1.0 - slippage)
            entry_p = round(min(cur_h, max(cur_l, raw_entry)), 2)
            active_sl = round(entry_p + (0.90 * entry_atr), 2)
            sl_dist = max(active_sl - entry_p, 0.05)
            friction = 0.002067 * entry_p
            denom = sl_dist + friction
            
            qty_r = int(target_risk_short / denom)
            qty_m = int((avail_notional * 0.90) / entry_p)
            qty = min(qty_r, qty_m)
            if cur_v > 0:
                qty = min(qty, int(0.045 * cur_v))
                
            if qty >= 1:
                positions[ticker] = {
                    "direction": "SHORT", "entry_p": entry_p, "qty": qty, "entry_atr": entry_atr,
                    "active_sl": active_sl, "initial_sl": active_sl,
                    "low_seen": entry_p, "bars_held": 0, "entry_t": ts
                }
                daily_traded[cur_date].add(ticker)
                daily_short_count[cur_date] += 1
                continue
                
        if long_candidates:
            long_candidates.sort(key=lambda x: x[6], reverse=True)
            best = long_candidates[0]
            ticker, cur_c, cur_h, cur_l, cur_v, entry_atr, _ = best
            
            raw_entry = cur_c * (1.0 + slippage)
            entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
            active_sl = round(entry_p - (0.90 * entry_atr), 2)
            sl_dist = max(entry_p - active_sl, 0.05)
            friction = 0.002067 * entry_p
            denom = sl_dist + friction
            
            qty_r = int(target_risk_long / denom)
            qty_m = int((avail_notional * 0.90) / entry_p)
            qty = min(qty_r, qty_m)
            if cur_v > 0:
                qty = min(qty, int(0.045 * cur_v))
                
            if qty >= 1:
                positions[ticker] = {
                    "direction": "LONG", "entry_p": entry_p, "qty": qty, "entry_atr": entry_atr,
                    "active_sl": active_sl, "initial_sl": active_sl,
                    "high_seen": entry_p, "bars_held": 0, "entry_t": ts
                }
                daily_traded[cur_date].add(ticker)
                
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["net_pnl"] > 0]
    losses = tdf[tdf["net_pnl"] < 0]
    wr = len(wins) / len(tdf) * 100.0 if len(tdf) > 0 else 0
    gross = tdf["gross_pnl"].sum() if len(tdf) > 0 else 0
    chg = tdf["charges"].sum() if len(tdf) > 0 else 0
    net = tdf["net_pnl"].sum() if len(tdf) > 0 else 0
    cum = initial_equity + tdf["net_pnl"].cumsum() if len(tdf) > 0 else pd.Series([initial_equity])
    hwm = cum.cummax()
    dd = (hwm - cum) / hwm * 100.0
    max_dd = dd.max() if len(dd) > 0 else 0.0
    pf = round(wins["net_pnl"].sum() / abs(losses["net_pnl"].sum()), 2) if len(losses) > 0 and abs(losses["net_pnl"].sum()) > 0 else 999.0
    
    n_long = len(tdf[tdf["direction"] == "LONG"]) if len(tdf) > 0 else 0
    n_short = len(tdf[tdf["direction"] == "SHORT"]) if len(tdf) > 0 else 0
    net_long = tdf[tdf["direction"] == "LONG"]["net_pnl"].sum() if n_long > 0 else 0.0
    net_short = tdf[tdf["direction"] == "SHORT"]["net_pnl"].sum() if n_short > 0 else 0.0
    
    print(f"=== {name} ({start_date} to {end_date}) ===")
    print(f"Trades: {len(tdf)} (Long: {n_long}, Short: {n_short}) | WR: {wr:.2f}%")
    print(f"Gross: Rs {gross:.2f} | Charges: Rs {chg:.2f} | Net: Rs {net:.2f} (Long Net: Rs {net_long:.2f}, Short Net: Rs {net_short:.2f})")
    print(f"Profit Factor: {pf} | Max DD: {max_dd:.2f}% | Final Capital: Rs {cum.iloc[-1]:.2f}\n")
    return tdf

# Load datasets
def prepare_2024_data():
    with open(ROOT_DIR / "data" / "2024-25" / "universe_14_15m.pkl", "rb") as f:
        raw_u = pickle.load(f)
    tickers = ["DELTACORP", "IFCI", "IDEA", "JPPOWER", "PCJEWELLER", "MTNL", "NCC", "NIACL", "RPOWER", "PTC", "RITES", "TEXRAIL", "UTKARSHBNK", "ZEEL"]
    data_map = {}
    for t in tickers:
        df = raw_u[t].copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        df = df.sort_index()
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
        atr14 = calculate_atr(df, 14)
        atr_pct = (atr14 / c) * 100.0
        ema9 = c.ewm(span=9).mean()
        ema20 = c.ewm(span=20).mean()
        ema50 = c.ewm(span=50).mean()
        ema60 = c.ewm(span=60).mean()
        ema110 = c.ewm(span=110).mean()
        
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(7).mean()
        loss = (-delta.clip(upper=0)).rolling(7).mean()
        rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))
        
        macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        sig = macd.ewm(span=9).mean()
        macd_cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
        
        adx, pdi, mdi = compute_adx_di(df, 10)
        vwap_lag = compute_vwap_shifted(df)
        v_sma20 = v.rolling(20).mean()
        rvol = v / (v_sma20 + 1e-9)
        er10 = compute_er10(c)
        
        high_5 = h.rolling(5).max().shift(1)
        lookback = min(300, len(c) - 1)
        mom_cond = (c > c.shift(lookback))
        
        g4_v26 = (
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
        
        g4_v28 = (
            macd_cross_up & (macd > 0) &
            (c > ema50) & (c > ema110) & (c > ema9) & (c > ema60) &
            (c > vwap_lag) &
            (er10 >= 0.35) &
            (rsi7 >= 48.0) & (rsi7 <= 78.0) &
            (rvol >= 1.25) &
            (atr_pct >= 0.50) &
            (adx >= 18.0) & (pdi > mdi) &
            mom_cond &
            (c > high_5) &
            (c <= 450.0) & (c >= 10.0)
        )
        
        df["_date"] = df.index.date
        day_open = df.groupby("_date")["open"].transform("first")
        df["day_ret"] = (c / day_open) - 1.0
        df["1h_ret"] = (c / c.shift(4)) - 1.0
        above_20dma = (c > ema20).astype(float)
        
        data_map[t + ".NS"] = {
            "df": df, "c": c, "h": h, "l": l, "v": v, "atr14": atr14, "atr_pct": atr_pct,
            "ema9": ema9, "rsi7": rsi7, "vwap_lag": vwap_lag, "rvol": rvol, "er10": er10,
            "g4_v26": g4_v26, "g4_v28": g4_v28,
            "day_ret": df["day_ret"], "1h_ret": df["1h_ret"],
            "above_20dma": above_20dma
        }
    b_day = pd.concat([d["day_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    b_1h = pd.concat([d["1h_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    breadth = pd.concat([d["above_20dma"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    
    for t, d in data_map.items():
        b_d = b_day.reindex(d["day_ret"].index)
        b_1 = b_1h.reindex(d["1h_ret"].index)
        d["g1_cond"] = (d["day_ret"] > b_d)
        d["g2_cond"] = (d["1h_ret"] > b_1)
        d["excess_1h"] = d["1h_ret"] - b_1
    return data_map, b_day, breadth

def prepare_2026_data():
    with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f: u1 = pickle.load(f)
    with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f: u2 = pickle.load(f)
    with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f: u3 = pickle.load(f)
    all_p = {}; all_p.update(u1); all_p.update(u2); all_p.update(u3)
    tickers = [
        "J&KBANK.NS", "UTKARSHBNK.NS", "MANALIPETC.NS", "IDEA.NS", "PCJEWELLER.NS",
        "NIACL.NS", "MTNL.NS", "IFCI.NS", "TEXRAIL.NS", "SWSOLAR.NS",
        "TTML.NS", "RENUKA.NS", "RCF.NS", "NBCC.NS", "JYOTHYLAB.NS",
        "RBLBANK.NS", "ELECTCAST.NS", "PNB.NS", "GSFC.NS", "TATASTEEL.NS",
        "SUZLON.NS", "CENTRALBK.NS", "IOB.NS", "IRCON.NS", "NMDC.NS",
        "RITES.NS", "NCC.NS", "ZEEL.NS", "DELTACORP.NS", "RPOWER.NS"
    ]
    data_map = {}
    for t in tickers:
        if t not in all_p: continue
        df = all_p[t].copy()
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
        atr14 = calculate_atr(df, 14)
        atr_pct = (atr14 / c) * 100.0
        ema9 = c.ewm(span=9).mean()
        ema20 = c.ewm(span=20).mean()
        ema50 = c.ewm(span=50).mean()
        ema60 = c.ewm(span=60).mean()
        ema110 = c.ewm(span=110).mean()
        
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(7).mean()
        loss = (-delta.clip(upper=0)).rolling(7).mean()
        rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))
        
        macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        sig = macd.ewm(span=9).mean()
        macd_cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
        
        adx, pdi, mdi = compute_adx_di(df, 10)
        vwap_lag = compute_vwap_shifted(df)
        v_sma20 = v.rolling(20).mean()
        rvol = v / (v_sma20 + 1e-9)
        er10 = compute_er10(c)
        
        high_5 = h.rolling(5).max().shift(1)
        lookback = min(300, len(c) - 1)
        mom_cond = (c > c.shift(lookback))
        
        g4_v26 = (
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
        g4_v28 = (
            macd_cross_up & (macd > 0) &
            (c > ema50) & (c > ema110) & (c > ema9) & (c > ema60) &
            (c > vwap_lag) &
            (er10 >= 0.35) &
            (rsi7 >= 48.0) & (rsi7 <= 78.0) &
            (rvol >= 1.25) &
            (atr_pct >= 0.50) &
            (adx >= 18.0) & (pdi > mdi) &
            mom_cond &
            (c > high_5) &
            (c <= 450.0) & (c >= 10.0)
        )
        df["_date"] = df.index.date
        day_open = df.groupby("_date")["open"].transform("first")
        df["day_ret"] = (c / day_open) - 1.0
        df["1h_ret"] = (c / c.shift(4)) - 1.0
        above_20dma = (c > ema20).astype(float)
        
        data_map[t] = {
            "df": df, "c": c, "h": h, "l": l, "v": v, "atr14": atr14, "atr_pct": atr_pct,
            "ema9": ema9, "rsi7": rsi7, "vwap_lag": vwap_lag, "rvol": rvol, "er10": er10,
            "g4_v26": g4_v26, "g4_v28": g4_v28,
            "day_ret": df["day_ret"], "1h_ret": df["1h_ret"],
            "above_20dma": above_20dma
        }
    b_day = pd.concat([d["day_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    b_1h = pd.concat([d["1h_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    breadth = pd.concat([d["above_20dma"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    for t, d in data_map.items():
        b_d = b_day.reindex(d["day_ret"].index)
        b_1 = b_1h.reindex(d["1h_ret"].index)
        d["g1_cond"] = (d["day_ret"] > b_d)
        d["g2_cond"] = (d["1h_ret"] > b_1)
        d["excess_1h"] = d["1h_ret"] - b_1
    return data_map, b_day, breadth

if __name__ == "__main__":
    print("Testing Bhavishya V30 Bi-Directional on Q4 2024 Crash...")
    dm24, bd24, br24 = prepare_2024_data()
    run_v30_simulation(dm24, bd24, br24, "2024-10-01", "2024-12-31", "V30 Q4 2024 UNSEEN CRASH")
    
    print("Testing Bhavishya V30 Bi-Directional on 2026 Modern Horizon...")
    dm26, bd26, br26 = prepare_2026_data()
    run_v30_simulation(dm26, bd26, br26, "2026-01-01", "2026-10-31", "V30 2026 MODERN HORIZON")
