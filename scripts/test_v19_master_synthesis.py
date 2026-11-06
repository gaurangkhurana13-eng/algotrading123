import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

# Load Data Pools
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    d_beta = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
    d_unseen = pickle.load(f)

all_pools = {}
all_pools.update(univ)
all_pools.update(d_beta)
all_pools.update(d_unseen)

tickers_core = [
    "PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS", "RITES.NS",
    "IDEA.NS", "MTNL.NS", "ZEEL.NS", "UTKARSHBNK.NS", "IFCI.NS", "NCC.NS",
    "PTC.NS", "NIACL.NS"
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
for t in tickers_core:
    df = all_pools[t].copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    o = df["open"]
    atr14 = calculate_atr(df, 14)
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
    
    bar_range = (h - l).replace(0, 1e-9)
    body_size = (c - o).abs()
    wick_size = (bar_range - body_size).clip(lower=0.0)
    clv = (c - l) / bar_range
    wbr20 = (wick_size.shift(1).rolling(20).sum() / bar_range.shift(1).rolling(20).sum()).fillna(0.50)
    
    # Sniper Entry with Pre-Trade Toxicity Filters
    entry_cond = (
        macd_cross_up & (macd > 0) &
        (c > ema50) & (c > ema110) &
        (c > vwap_lag) &
        (rsi7 > 50.0) & (adx > 18.0) & (plus_di > minus_di) &
        (c <= 450.0) & (c >= 10.0) &
        ((0.05 / np.maximum(atr14, 0.01)) <= 0.15) &  # Tick ratio <= 15%
        ((atr14 / c) * 100.0 >= 0.40) &               # NATR >= 0.40%
        (clv >= 0.55) &                               # Upper 45% close (no upper wick grab)
        (wbr20 <= 0.60)                               # Anti-PSU chop filter
    )
    
    data_map[t] = {
        "df": df, "c": c, "h": h, "l": l, "o": o, "v": v, "atr14": atr14,
        "rsi7": rsi7, "vwap_lag": vwap_lag, "rvol": rvol, "er10": er10,
        "clv": clv, "wbr20": wbr20, "entry_cond": entry_cond
    }

all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
slippage = 0.0005

def run_v19_simulation(
    hard_sl_mult=0.85,
    single_risk_cap=95.0,
    enable_pyramid=True,
    enable_zero_risk_concurrency=True
):
    positions = {}
    trades = []
    daily_traded = {}
    last_exit_times = set()

    for ts in all_ts:
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
            cur_c, cur_h, cur_l, cur_o, cur_v = row["close"], row["high"], row["low"], row["open"], row["volume"]
            cur_rsi = t_data["rsi7"].loc[ts]
            cur_vwap = t_data["vwap_lag"].loc[ts]
            cur_atr = t_data["atr14"].loc[ts]
            
            p["bars_held"] += 1
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
            
            # Fast Scratch Decoupling (Only if not yet risk-free)
            if not p["is_risk_free"]:
                # Tier 1: Bar 1 or 2 severe rejection below entry mid and vwap
                if p["bars_held"] in (1, 2) and mfe_atr < 0.25:
                    if cur_c < p["entry_mid"] and (cur_c < cur_vwap or (cur_c < cur_o and cur_v >= 0.85 * p["entry_vol"])):
                        raw_exit = cur_c * (1.0 - slippage)
                        exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                        gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                        chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                        net = round(gross - chg, 2)
                        trades.append({
                            "symbol": ticker.replace(".NS", ""), "direction": "long",
                            "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                            "entry_price": p["entry_p"], "exit_price": exit_p,
                            "shares": p["qty"], "sl": p["active_sl"],
                            "gross_pnl": gross, "charges": chg, "net_pnl": net,
                            "exit_reason": "FAST_SCRATCH_TIER1"
                        })
                        last_exit_times.add(ts)
                        del positions[ticker]
                        continue
                # Tier 3: Stagnation Scratch at bar 3+
                if p["bars_held"] >= 3 and mfe_atr < 0.40:
                    if cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                        raw_exit = cur_c * (1.0 - slippage)
                        exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                        gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                        chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                        net = round(gross - chg, 2)
                        trades.append({
                            "symbol": ticker.replace(".NS", ""), "direction": "long",
                            "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                            "entry_price": p["entry_p"], "exit_price": exit_p,
                            "shares": p["qty"], "sl": p["active_sl"],
                            "gross_pnl": gross, "charges": chg, "net_pnl": net,
                            "exit_reason": "STAGNATION_SCRATCH"
                        })
                        last_exit_times.add(ts)
                        del positions[ticker]
                        continue
                        
            # 4-Phase Structural Trailing Corridor
            idx_now = t_data["df"].index.get_loc(ts)
            low_t1 = t_data["l"].iloc[idx_now - 1] if idx_now >= 1 else cur_l
            low_t2 = t_data["l"].iloc[idx_now - 2] if idx_now >= 2 else low_t1
            high_t1 = t_data["h"].iloc[idx_now - 1] if idx_now >= 1 else cur_h
            
            # Phase 1: Risk Elimination Lock (0.80x ATR)
            if mfe_atr >= 0.80:
                p1_lock = p["entry_p"] + 0.20 * p["entry_atr"]
                p["active_sl"] = max(p["active_sl"], round(p1_lock, 2))
                p["is_risk_free"] = True
                p["phase"] = 1
                
            # Phase 2: Structural Swing Low Corridor (1.40x ATR)
            if mfe_atr >= 1.40:
                swing_floor = min(low_t1, low_t2) - 0.10 * cur_atr
                ch_trail = p["high_seen"] - 0.85 * cur_atr
                profit_floor = p["entry_p"] + 0.65 * p["entry_atr"]
                p["active_sl"] = max(p["active_sl"], round(max(profit_floor, min(swing_floor, ch_trail)), 2))
                p["phase"] = 2
                
            # Phase 3: Trend Acceleration Lock (2.80x ATR)
            if mfe_atr >= 2.80:
                fast_trail = p["high_seen"] - 0.55 * cur_atr
                p["active_sl"] = max(p["active_sl"], round(fast_trail, 2))
                p["phase"] = 3
                
            # Phase 4: Climax Harvester (4.00x ATR or late afternoon)
            if mfe_atr >= 4.00 or (mfe_atr >= 2.00 and t_time_int >= 1415):
                climax_trail = p["high_seen"] - 0.35 * cur_atr
                p["active_sl"] = max(p["active_sl"], round(climax_trail, 2))
                p["phase"] = 4
                
            # House-Money Pyramid Tranche on confirmed runners
            if enable_pyramid and not p["pyramided"] and mfe_atr >= 1.40 and cur_c > high_t1 and t_time_int <= 1400:
                add_qty = int(0.50 * p["qty"])
                if add_qty >= 1:
                    raw_pyr_entry = cur_c * (1.0 + slippage)
                    pyr_p = round(max(cur_l, min(cur_h, raw_pyr_entry)), 2)
                    p["total_buy_notional"] += (add_qty * pyr_p)
                    p["qty"] += add_qty
                    p["pyramided"] = True
                    # Lock unified stop to ensure net trade is still in guaranteed profit
                    p["active_sl"] = max(p["active_sl"], round(p["entry_p"] + 0.50 * p["entry_atr"], 2))

            # Stop hit check
            if cur_l <= p["active_sl"]:
                raw_exit = p["active_sl"] * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                avg_entry = round(p["total_buy_notional"] / p["qty"], 2)
                gross = round((exit_p - avg_entry) * p["qty"], 2)
                num_legs = 2 if p["pyramided"] else 1
                chg = round(calc_dhan_charges(avg_entry, exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                trades.append({
                    "symbol": ticker.replace(".NS", ""), "direction": "long",
                    "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                    "entry_price": avg_entry, "exit_price": exit_p,
                    "shares": p["qty"], "sl": p["active_sl"],
                    "gross_pnl": gross, "charges": chg, "net_pnl": net,
                    "exit_reason": f"PHASE_{p['phase']}_STOP_HIT"
                })
                last_exit_times.add(ts)
                del positions[ticker]
                continue
                
            # 14:45 Square-off
            if t_time >= pd.Timestamp("14:45").time():
                raw_exit = cur_c * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                avg_entry = round(p["total_buy_notional"] / p["qty"], 2)
                gross = round((exit_p - avg_entry) * p["qty"], 2)
                chg = round(calc_dhan_charges(avg_entry, exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                trades.append({
                    "symbol": ticker.replace(".NS", ""), "direction": "long",
                    "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                    "entry_price": avg_entry, "exit_price": exit_p,
                    "shares": p["qty"], "sl": p["active_sl"],
                    "gross_pnl": gross, "charges": chg, "net_pnl": net,
                    "exit_reason": "INTRADAY_1445_FLAT"
                })
                last_exit_times.add(ts)
                del positions[ticker]
                continue
                
        closed_before = trades
        ek = 500.0 + sum(t["net_pnl"] for t in closed_before)
        hwm = 500.0
        cum = 500.0
        for t in closed_before:
            cum += t["net_pnl"]
            if cum > hwm:
                hwm = cum
                
        consec_losses = 0
        for t in reversed(closed_before):
            if t["net_pnl"] < 0:
                consec_losses += 1
            else:
                break
                
        dd_pct = (hwm - ek) / max(hwm, 500.0)
        
        # Max slots: 1 under Rs 2,200, 2 above Rs 2,200
        allowed_slots = 1 if ek < 2200.0 or consec_losses >= 2 or dd_pct > 0.14 else 2
        
        if len(positions) >= allowed_slots or ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
            continue
            
        # Zero-Risk Concurrency Law: Slot 2 only opens if Slot 1 is already risk-free!
        if enable_zero_risk_concurrency and len(positions) >= 1:
            if any(not p["is_risk_free"] for p in positions.values()):
                continue
                
        current_notional = sum(p["total_buy_notional"] for p in positions.values())
        avail_margin = (0.95 * ek) - (current_notional / 5.0)
        if avail_margin < 10.0:
            continue
        avail_notional = avail_margin * 5.0
        
        # Sizing Spline
        if consec_losses >= 2 or dd_pct > 0.14:
            alpha, beta = 0.015, 0.010
        elif consec_losses == 1:
            alpha, beta = 0.035, 0.020
        else:
            alpha, beta = 0.082, 0.035
            
        if ek <= 2500.0:
            raw_r = 14.0 + (alpha * max(0.0, ek - 500.0))
        else:
            raw_r = 14.0 + (alpha * 2000.0) + (beta * (ek - 2500.0))
            
        for ticker in tickers_core:
            if len(positions) >= allowed_slots:
                break
            if ticker in positions or ticker in daily_traded[cur_date]:
                continue
            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            idx = t_data["df"].index.get_loc(ts)
            if idx < 100 or not t_data["entry_cond"].iloc[idx]:
                continue
                
            cur_c = t_data["c"].iloc[idx]
            cur_h = t_data["h"].iloc[idx]
            cur_l = t_data["l"].iloc[idx]
            cur_o = t_data["o"].iloc[idx]
            cur_v = t_data["v"].iloc[idx]
            entry_atr = max(t_data["atr14"].iloc[idx], 0.05)
            cur_rvol = t_data["rvol"].iloc[idx]
            cur_er = t_data["er10"].iloc[idx]
            
            # Ticker-Agnostic Dynamic Conviction
            er_bonus = 0.25 * max(0.0, cur_er - 0.35)
            rvol_bonus = 0.05 * min(max(0.0, cur_rvol - 1.5), 3.0)
            weight = min(1.20, max(0.85, 0.85 + er_bonus + rvol_bonus))
            
            target_risk = min(raw_r * weight, single_risk_cap)
            
            raw_entry = cur_c * (1.0 + slippage)
            entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
            
            # Decoupled Sizing: Hard SL at 0.85x ATR, sizing denom at 1.00x ATR + friction
            active_sl = round(entry_p - (hard_sl_mult * entry_atr), 2)
            friction = 0.002067 * entry_p
            denom = entry_atr + friction
            
            qty_r = int(target_risk / denom)
            slots_free = max(1, allowed_slots - len(positions))
            slot_notional = avail_notional / slots_free
            qty_m = int((slot_notional * 0.90) / entry_p)
            qty = min(qty_r, qty_m)
            if cur_v > 0:
                qty = min(qty, int(0.045 * cur_v))
                
            if qty >= 1:
                positions[ticker] = {
                    "entry_p": entry_p, "qty": qty, "entry_atr": entry_atr,
                    "active_sl": active_sl, "high_seen": entry_p, "bars_held": 0,
                    "entry_t": ts, "phase": 0, "is_risk_free": False, "pyramided": False,
                    "entry_mid": 0.5 * (cur_o + cur_c), "entry_vol": cur_v,
                    "total_buy_notional": qty * entry_p
                }
                daily_traded[cur_date].add(ticker)
                current_notional = sum(p["total_buy_notional"] for p in positions.values())
                avail_margin = (0.95 * ek) - (current_notional / 5.0)
                avail_notional = max(0.0, avail_margin * 5.0)

    df_res = pd.DataFrame(trades)
    n = len(df_res)
    wins = (df_res["net_pnl"] > 0).sum()
    wr = (wins / n * 100.0) if n else 0.0
    net = df_res["net_pnl"].sum()
    gp = df_res[df_res["net_pnl"] > 0]["net_pnl"].sum()
    gl = abs(df_res[df_res["net_pnl"] < 0]["net_pnl"].sum())
    pf = (gp / gl) if gl > 0 else 0.0
    avg_win = gp / wins if wins else 0.0
    avg_loss = -gl / (n - wins) if (n - wins) else 0.0
    worst = df_res["net_pnl"].min() if n else 0.0
    path = 500.0 + np.cumsum(df_res["net_pnl"].to_numpy(float))
    peak = np.maximum.accumulate(np.maximum(path, 500.0))
    mdd = float(((peak - path) / peak).max()) * 100.0 if len(path) else 0.0
    return {
        "n": n, "wr": wr, "net": net, "pf": pf,
        "final": path[-1] if len(path) else 500.0,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "worst": worst, "mdd": mdd, "df_trades": df_res
    }

print("=== TESTING BHAVISHYA V19 MASTER SYNTHESIS ===")
for cap in [85.0, 95.0, 110.0]:
    for pyr in [False, True]:
        res = run_v19_simulation(hard_sl_mult=0.85, single_risk_cap=cap, enable_pyramid=pyr)
        print(f"Cap=Rs {cap:>4.1f} | Pyramid={pyr} => Trades={res['n']:>3} | Net=Rs {res['net']:>7.2f} | Final=Rs {res['final']:>7.2f} | PF={res['pf']:.2f} | WR={res['wr']:.1f}% | AvgWin=Rs {res['avg_win']:>5.2f} | AvgLoss=Rs {res['avg_loss']:>6.2f} | Worst=Rs {res['worst']:>6.2f} | MDD={res['mdd']:.1f}%")
