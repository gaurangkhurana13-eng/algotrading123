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

def run_pure_short_simulation(data_map, bench_day_ret_series, market_breadth_series, start_date, end_date, name):
    all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
    target_ts = [ts for ts in all_ts if pd.Timestamp(start_date) <= ts <= pd.Timestamp(end_date)]
    
    slippage = 0.0005
    initial_equity = 500.0
    
    positions = {}
    trades = []
    daily_short_count = {}
    last_exit_times = set()
    
    for ts in target_ts:
        if ts.weekday() >= 5:
            continue
        cur_date = ts.date()
        if cur_date not in daily_short_count:
            daily_short_count[cur_date] = 0
        t_time = ts.time()
        
        # 1. Manage Active Short Positions
        for ticker in list(positions.keys()):
            p = positions[ticker]
            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            row = t_data["df"].loc[ts]
            cur_c, cur_h, cur_l = row["close"], row["high"], row["low"]
            
            p["bars_held"] += 1
            p["low_seen"] = min(p["low_seen"], cur_l)
            mfe_atr = (p["entry_p"] - p["low_seen"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
            
            def finalize_trade(exit_fill, exit_reason):
                tot_qty = p["qty"]
                gross = round((p["entry_p"] - exit_fill) * tot_qty, 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_fill, tot_qty, is_intraday=True), 2)
                net = round(gross - chg, 2)
                trades.append({
                    "symbol": ticker,
                    "direction": "SHORT",
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
                    "mfe_atr": round(mfe_atr, 2)
                })
                last_exit_times.add(ts)
                del positions[ticker]
                
            # Dynamic Ratchet Rules for Short:
            # Stage 3: Hard TP at 2.0 ATR
            if mfe_atr >= 2.00:
                finalize_trade(p["entry_p"] - (2.00 * p["entry_atr"]), "TARGET_2ATR")
                continue
            # Stage 2: Trail by 0.25 ATR above lowest price reached once in 1.0 ATR profit
            elif mfe_atr >= 1.00:
                ratchet_sl = p["low_seen"] + (0.25 * p["entry_atr"])
                # Lock at least breakeven
                ratchet_sl = min(ratchet_sl, p["entry_p"] - (0.20 * p["entry_atr"]))
            elif mfe_atr >= 0.50:
                ratchet_sl = p["entry_p"]  # Breakeven lock
            else:
                ratchet_sl = p["initial_sl"]
                
            p["active_sl"] = min(p["active_sl"], round(ratchet_sl, 2))
            
            # Check Stop Loss
            if cur_h >= p["active_sl"]:
                finalize_trade(p["active_sl"] * (1.0 + slippage), "STOP")
                continue
                
            # Scratch rule: after 4 bars, no momentum downwards and price above entry
            if p["bars_held"] >= 4 and mfe_atr < 0.30 and cur_c > p["entry_p"]:
                finalize_trade(cur_c * (1.0 + slippage), "SCRATCH")
                continue
                
            # COMPULSORY EXIT DEADLINE (14:10 IST) to escape 14:15-15:00 short covering squeeze
            if t_time >= pd.Timestamp("14:10").time():
                finalize_trade(cur_c * (1.0 + slippage), "TIME_EXIT_1410")
                continue
                
        ek = initial_equity + sum(t["net_pnl"] for t in trades)
        hwm = initial_equity
        cum = initial_equity
        for t in trades:
            cum += t["net_pnl"]
            if cum > hwm:
                hwm = cum
        dd_pct = (hwm - ek) / max(hwm, initial_equity)
        
        # Entry restrictions:
        # 1. Single-position focus: max 1 active position
        # 2. Strict Anti-churn: max 1 short trade per day!
        # 3. Time windows: ONLY 09:55-11:30 and 13:10-13:50
        if len(positions) >= 1 or daily_short_count[cur_date] >= 1 or ts in last_exit_times:
            continue
            
        is_window_1 = (pd.Timestamp("09:55").time() <= t_time <= pd.Timestamp("11:30").time())
        is_window_2 = (pd.Timestamp("13:10").time() <= t_time <= pd.Timestamp("13:50").time())
        if not (is_window_1 or is_window_2):
            continue
            
        current_notional = sum(p["entry_p"] * p["qty"] for p in positions.values())
        avail_margin = (0.95 * ek) - (current_notional / 5.0)
        if avail_margin < 10.0:
            continue
        avail_notional = avail_margin * 5.0
        
        # Risk Sizing: 2.0% of equity per trade
        target_risk = min(0.025 * ek, 20.0)
        
        # Macro Down check
        b_day_now = bench_day_ret_series.get(ts, 0.0)
        cur_breadth = market_breadth_series.get(ts, 0.5)
        # Macro downtrend: daily return negative or breadth < 0.40
        macro_down = (b_day_now < -0.003 or cur_breadth < 0.40)
        if not macro_down:
            continue
            
        candidates = []
        for ticker, t_data in data_map.items():
            if ts not in t_data["df"].index:
                continue
            idx = t_data["df"].index.get_loc(ts)
            if idx < 100:
                continue
                
            cur_c = t_data["c"].iloc[idx]
            cur_o = t_data["df"]["open"].iloc[idx]
            cur_h = t_data["h"].iloc[idx]
            cur_l = t_data["l"].iloc[idx]
            cur_v = t_data["v"].iloc[idx]
            entry_atr = max(t_data["atr14"].iloc[idx], 0.05)
            
            # Indicator Checks
            rsi = t_data["rsi7"].iloc[idx]
            vwap = t_data["vwap_lag"].iloc[idx]
            ema20 = t_data["ema20"].iloc[idx]
            ema50 = t_data["ema50"].iloc[idx]
            ema20_prev = t_data["ema20"].iloc[idx-1]
            ema20_slope_down = (ema20 < ema20_prev)
            
            # 1. Bearish Rejection RSI (40 <= RSI <= 55) - not oversold!
            rsi_cond = (40.0 <= rsi <= 55.0)
            
            # 2. Dynamic Resistance: Price <= EMA20 and EMA20 < EMA50
            trend_down = (cur_c <= ema20) and (ema20 < ema50) and ema20_slope_down and (cur_c < vwap)
            
            # 3. Upper Wick Rejection: upper wick >= 35% of total candle range
            candle_range = max(cur_h - cur_l, 0.01)
            upper_wick = cur_h - max(cur_o, cur_c)
            wick_ratio = upper_wick / candle_range
            wick_cond = (wick_ratio >= 0.30)
            
            # 4. Breakdown condition: Close < 5-bar Low
            low_5 = t_data["df"]["low"].rolling(5).min().shift(1).iloc[idx]
            bd_cond = (cur_c < low_5)
            
            # 5. Volume surge
            rvol = t_data["rvol"].iloc[idx]
            vol_cond = (rvol >= 1.25)
            
            # 6. Relative Weakness
            excess_1h = t_data["excess_1h"].iloc[idx]
            rel_weak = (excess_1h < -0.005)
            
            # Price range constraint
            price_cond = (10.0 <= cur_c <= 450.0)
            
            if rsi_cond and trend_down and bd_cond and vol_cond and rel_weak and price_cond:
                candidates.append((ticker, cur_c, cur_h, cur_l, cur_v, entry_atr, excess_1h, cur_h))
                
        if not candidates:
            continue
            
        # Select stock with deepest relative weakness
        candidates.sort(key=lambda x: x[6])
        best = candidates[0]
        ticker, cur_c, cur_h, cur_l, cur_v, entry_atr, _, prior_h = best
        
        raw_entry = cur_c * (1.0 - slippage)
        entry_p = round(min(cur_h, max(cur_l, raw_entry)), 2)
        initial_sl = round(max(prior_h + (0.15 * entry_atr), entry_p + (0.75 * entry_atr)), 2)
        sl_dist = max(initial_sl - entry_p, 0.05)
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
                "active_sl": initial_sl, "initial_sl": initial_sl,
                "low_seen": entry_p, "bars_held": 0, "entry_t": ts
            }
            daily_short_count[cur_date] += 1
            
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
    pf = round(wins["net_pnl"].sum() / abs(losses["net_pnl"].sum()), 2) if len(losses) > 0 and abs(losses["net_pnl"].sum()) > 0 else (999.0 if len(wins) > 0 else 0.0)
    
    print(f"==================================================")
    print(f" {name} ({start_date} to {end_date})")
    print(f"==================================================")
    print(f"Total Short Trades: {len(tdf)} | Wins: {len(wins)} | Losses: {len(losses)} | Win Rate: {wr:.2f}%")
    print(f"Gross PnL: Rs {gross:.2f} | Dhan Charges: Rs {chg:.2f} | Net Cash: Rs {net:.2f}")
    print(f"Profit Factor: {pf} | Max Drawdown: {max_dd:.2f}% | Final Capital: Rs {cum.iloc[-1]:.2f}")
    if len(tdf) > 0:
        print("\nSample Trade Log (first 5 trades):")
        print(tdf[['symbol', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'net_pnl', 'exit_reason']].head(5))
    print()
    return tdf

def load_data_2024():
    with open(ROOT_DIR / "data" / "2024-25" / "universe_14_15m.pkl", "rb") as f: raw_u = pickle.load(f)
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
        ema20 = c.ewm(span=20).mean()
        ema50 = c.ewm(span=50).mean()
        
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(7).mean()
        loss = (-delta.clip(upper=0)).rolling(7).mean()
        rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))
        
        vwap_lag = compute_vwap_shifted(df)
        v_sma20 = v.rolling(20).mean()
        rvol = v / (v_sma20 + 1e-9)
        er10 = compute_er10(c)
        
        df["_date"] = df.index.date
        day_open = df.groupby("_date")["open"].transform("first")
        df["day_ret"] = (c / day_open) - 1.0
        df["1h_ret"] = (c / c.shift(4)) - 1.0
        above_20dma = (c > ema20).astype(float)
        
        data_map[t + ".NS"] = {
            "df": df, "c": c, "h": h, "l": l, "v": v, "atr14": atr14,
            "ema20": ema20, "ema50": ema50, "rsi7": rsi7,
            "vwap_lag": vwap_lag, "rvol": rvol, "er10": er10,
            "day_ret": df["day_ret"], "1h_ret": df["1h_ret"],
            "above_20dma": above_20dma
        }
    b_day = pd.concat([d["day_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    b_1h = pd.concat([d["1h_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    breadth = pd.concat([d["above_20dma"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    for t, d in data_map.items():
        b_d = b_day.reindex(d["day_ret"].index)
        b_1 = b_1h.reindex(d["1h_ret"].index)
        d["excess_1h"] = d["1h_ret"] - b_1
    return data_map, b_day, breadth

def load_data_2025():
    p_2025 = ROOT_DIR / "data" / "2025-26"
    files = list(p_2025.glob("*_15m_1y.pkl"))
    data_map = {}
    for f in files:
        sym = f.name.replace("_15m_1y.pkl", "").upper() + ".NS"
        with open(f, "rb") as fp: df = pickle.load(fp)
        df = df.loc["2025-01-01":"2025-12-31"].copy()
        if len(df) < 100: continue
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
        atr14 = calculate_atr(df, 14)
        ema20 = c.ewm(span=20).mean()
        ema50 = c.ewm(span=50).mean()
        
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(7).mean()
        loss = (-delta.clip(upper=0)).rolling(7).mean()
        rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))
        
        vwap_lag = compute_vwap_shifted(df)
        v_sma20 = v.rolling(20).mean()
        rvol = v / (v_sma20 + 1e-9)
        er10 = compute_er10(c)
        
        df["_date"] = df.index.date
        day_open = df.groupby("_date")["open"].transform("first")
        df["day_ret"] = (c / day_open) - 1.0
        df["1h_ret"] = (c / c.shift(4)) - 1.0
        above_20dma = (c > ema20).astype(float)
        
        data_map[sym] = {
            "df": df, "c": c, "h": h, "l": l, "v": v, "atr14": atr14,
            "ema20": ema20, "ema50": ema50, "rsi7": rsi7,
            "vwap_lag": vwap_lag, "rvol": rvol, "er10": er10,
            "day_ret": df["day_ret"], "1h_ret": df["1h_ret"],
            "above_20dma": above_20dma
        }
    b_day = pd.concat([d["day_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    b_1h = pd.concat([d["1h_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    breadth = pd.concat([d["above_20dma"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    for t, d in data_map.items():
        b_d = b_day.reindex(d["day_ret"].index)
        b_1 = b_1h.reindex(d["1h_ret"].index)
        d["excess_1h"] = d["1h_ret"] - b_1
    return data_map, b_day, breadth

def load_data_2026():
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
        ema20 = c.ewm(span=20).mean()
        ema50 = c.ewm(span=50).mean()
        
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(7).mean()
        loss = (-delta.clip(upper=0)).rolling(7).mean()
        rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))
        
        vwap_lag = compute_vwap_shifted(df)
        v_sma20 = v.rolling(20).mean()
        rvol = v / (v_sma20 + 1e-9)
        er10 = compute_er10(c)
        
        df["_date"] = df.index.date
        day_open = df.groupby("_date")["open"].transform("first")
        df["day_ret"] = (c / day_open) - 1.0
        df["1h_ret"] = (c / c.shift(4)) - 1.0
        above_20dma = (c > ema20).astype(float)
        
        data_map[t] = {
            "df": df, "c": c, "h": h, "l": l, "v": v, "atr14": atr14,
            "ema20": ema20, "ema50": ema50, "rsi7": rsi7,
            "vwap_lag": vwap_lag, "rvol": rvol, "er10": er10,
            "day_ret": df["day_ret"], "1h_ret": df["1h_ret"],
            "above_20dma": above_20dma
        }
    b_day = pd.concat([d["day_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    b_1h = pd.concat([d["1h_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    breadth = pd.concat([d["above_20dma"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    for t, d in data_map.items():
        b_d = b_day.reindex(d["day_ret"].index)
        b_1 = b_1h.reindex(d["1h_ret"].index)
        d["excess_1h"] = d["1h_ret"] - b_1
    return data_map, b_day, breadth

if __name__ == "__main__":
    print(">>> TESTING BHAVISHYA PURE-SHORT ENGINE ACROSS ALL HORIZONS <<<\n")
    
    # 1. Q4 2024 Crash
    dm24, bd24, br24 = load_data_2024()
    run_pure_short_simulation(dm24, bd24, br24, "2024-10-01", "2024-12-31", "BHAVISHYA PURE-SHORT: Q4 2024 CRASH")
    
    # 2. Full Year 2025
    dm25, bd25, br25 = load_data_2025()
    run_pure_short_simulation(dm25, bd25, br25, "2025-01-01", "2025-12-31", "BHAVISHYA PURE-SHORT: FULL YEAR 2025")
    
    # 3. Modern 2026 Horizon
    dm26, bd26, br26 = load_data_2026()
    run_pure_short_simulation(dm26, bd26, br26, "2026-01-01", "2026-10-31", "BHAVISHYA PURE-SHORT: 2026 MODERN HORIZON")
