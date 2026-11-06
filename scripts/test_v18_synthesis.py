import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr
import strategy_evaluate_gate_v2 as gate

# Load data pools
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

# If nlcindia is available as separate pkl
nlc_path = ROOT_DIR / "data" / "nlcindia_15m_1y.pkl"
if nlc_path.exists():
    with open(nlc_path, "rb") as f:
        all_pools["NLCINDIA.NS"] = pickle.load(f)

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

basket_top10 = [
    "PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS", "RITES.NS", "IDEA.NS",
    "MTNL.NS", "ZEEL.NS", "UTKARSHBNK.NS", "IFCI.NS", "NCC.NS"
]

basket_verified15 = [
    "PCJEWELLER.NS", "ZEEL.NS", "DELTACORP.NS", "RITES.NS", "UTKARSHBNK.NS",
    "RPOWER.NS", "IFCI.NS", "MTNL.NS", "IDEA.NS", "NCC.NS",
    "PTC.NS", "NIACL.NS", "HUDCO.NS", "ELECTCAST.NS", "CANBK.NS"
]

basket_argon20 = [
    "PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS", "RITES.NS", "IDEA.NS",
    "MTNL.NS", "ZEEL.NS", "UTKARSHBNK.NS", "IFCI.NS", "NCC.NS",
    "NBCC.NS", "HFCL.NS", "MANAPPURAM.NS", "NATIONALUM.NS", "HUDCO.NS",
    "BHEL.NS", "SAIL.NS", "IRFC.NS", "NLCINDIA.NS", "CESC.NS"
]

def prepare_data(ticker_list):
    data_map = {}
    for t in ticker_list:
        if t not in all_pools:
            continue
        df = all_pools[t].copy()
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df["volume"]
        atr14 = calculate_atr(df, 14)
        ema50 = c.ewm(span=50).mean()
        ema110 = c.ewm(span=110).mean()
        vwap_lag = compute_vwap_shifted(df)
        v_sma20 = v.rolling(20).mean()
        rvol = v / (v_sma20 + 1e-9)
        er10 = compute_er10(c)
        turnover_med20 = (c * v).rolling(20).median()
        
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
        
        entry_cond = (
            macd_cross_up & (macd > 0) &
            (c > ema50) & (c > ema110) &
            (c > vwap_lag) &
            (rsi7 > 50.0) & (adx > 18.0) & (plus_di > minus_di) &
            (c <= 450.0) & (c >= 10.0)
        )
        data_map[t] = {
            "df": df, "c": c, "h": h, "l": l, "v": v, "atr14": atr14,
            "rsi7": rsi7, "vwap_lag": vwap_lag, "rvol": rvol, "er10": er10,
            "turnover_med20": turnover_med20, "entry_cond": entry_cond
        }
    return data_map

def run_simulation(
    data_map,
    ticker_list,
    hard_sl_mult=0.75,
    sizing_denom_mult=1.00,
    max_risk_cap=135.0,
    max_slots=2,
    use_microstructure_filter=True,
    tick_ratio_max=0.12,
    natr_min=0.45,
    turnover_min=1_000_000.0,
    er10_min=0.32,
    use_2bar_scratch=True,
    use_structural_trail=True
):
    all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
    slippage = 0.0005
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
            cur_c, cur_h, cur_l = row["close"], row["high"], row["low"]
            cur_rsi = t_data["rsi7"].loc[ts]
            cur_vwap = t_data["vwap_lag"].loc[ts]
            cur_atr = t_data["atr14"].loc[ts]
            
            p["bars_held"] += 1
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
            
            # Mechanic 1: Fast Time-Decay Scratch
            if use_2bar_scratch:
                # 2-Bar VWAP Failure Scratch
                if p["bars_held"] == 2 and mfe_atr < 0.35:
                    if cur_c < cur_vwap or (cur_c < p["entry_p"] - 0.15 * p["entry_atr"] and cur_rsi < 49.0):
                        raw_exit = cur_c * (1.0 - slippage)
                        exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                        gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                        chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                        net = round(gross - chg, 2)
                        trades.append({"ticker": ticker, "net_pnl": net, "exit_p": exit_p, "qty": p["qty"], "exit_reason": "BAR2_VWAP_SCRATCH", "entry_p": p["entry_p"]})
                        last_exit_times.add(ts)
                        del positions[ticker]
                        continue
                # 3-Bar Stagnation Scratch
                if p["bars_held"] >= 3 and mfe_atr < 0.45:
                    if cur_c < p["entry_p"] and (cur_rsi < 47.0 or cur_c < cur_vwap):
                        raw_exit = cur_c * (1.0 - slippage)
                        exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                        gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                        chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                        net = round(gross - chg, 2)
                        trades.append({"ticker": ticker, "net_pnl": net, "exit_p": exit_p, "qty": p["qty"], "exit_reason": "BAR3_STAGNATION_SCRATCH", "entry_p": p["entry_p"]})
                        last_exit_times.add(ts)
                        del positions[ticker]
                        continue
                        
            # Mechanic 2: 3-Gear Structural Corridor & Chandelier Super-Trail
            if use_structural_trail:
                idx_now = t_data["df"].index.get_loc(ts)
                low_t1 = t_data["l"].iloc[idx_now - 1] if idx_now >= 1 else cur_l
                low_t2 = t_data["l"].iloc[idx_now - 2] if idx_now >= 2 else low_t1
                
                # Gear 1: Shield at +0.75x ATR
                if mfe_atr >= 0.75:
                    p["active_sl"] = max(p["active_sl"], round(p["entry_p"] + (0.30 * p["entry_atr"]), 2))
                # Gear 2: Structural Super-trend corridor
                if mfe_atr >= 1.35:
                    profit_floor = p["entry_p"] + (0.85 * p["entry_atr"])
                    structural_low = min(low_t1, low_t2) - (0.08 * cur_atr)
                    wide_ch = p["high_seen"] - (0.80 * cur_atr)
                    p["active_sl"] = max(p["active_sl"], round(max(profit_floor, structural_low, wide_ch), 2))
                # Gear 3: Parabolic Climax Lock
                if mfe_atr >= 2.40 or (mfe_atr >= 1.35 and t_time_int >= 1415):
                    climax_ch = p["high_seen"] - (0.38 * cur_atr)
                    retention = p["entry_p"] + 0.72 * (p["high_seen"] - p["entry_p"])
                    p["active_sl"] = max(p["active_sl"], round(max(climax_ch, retention), 2))
            else:
                # V17 Standard
                if mfe_atr >= 1.35:
                    p["active_sl"] = max(p["active_sl"], round(p["high_seen"] - (0.45 * p["entry_atr"]), 2))
                elif mfe_atr >= 1.70:
                    p["active_sl"] = max(p["active_sl"], round(p["high_seen"] - (0.65 * p["entry_atr"]), 2))
                elif mfe_atr >= 0.85:
                    p["active_sl"] = max(p["active_sl"], round(p["entry_p"] + (0.40 * p["entry_atr"]), 2))

            # Stop hit check
            if cur_l <= p["active_sl"]:
                raw_exit = p["active_sl"] * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                trades.append({"ticker": ticker, "net_pnl": net, "exit_p": exit_p, "qty": p["qty"], "exit_reason": "STOP_HIT", "entry_p": p["entry_p"]})
                last_exit_times.add(ts)
                del positions[ticker]
                continue
                
            # 14:45 Square-off
            if t_time >= pd.Timestamp("14:45").time():
                raw_exit = cur_c * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                trades.append({"ticker": ticker, "net_pnl": net, "exit_p": exit_p, "qty": p["qty"], "exit_reason": "1445_FLAT", "entry_p": p["entry_p"]})
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
        
        # Concurrency slots
        if ek < 2200.0 or consec_losses >= 2 or dd_pct > 0.14:
            allowed_slots = 1
        elif ek < 6500.0:
            allowed_slots = min(max_slots, 2)
        else:
            allowed_slots = min(max_slots, 3)
            
        if len(positions) >= allowed_slots or ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
            continue
            
        current_notional = sum(p["entry_p"] * p["qty"] for p in positions.values())
        avail_margin = (0.95 * ek) - (current_notional / 5.0)
        if avail_margin < 10.0:
            continue
        avail_notional = avail_margin * 5.0
        
        # 3-Stage Concave Spline
        if dd_pct > 0.14 or consec_losses >= 2:
            alpha, beta, gamma = 0.010, 0.0025, 0.0015
        elif consec_losses == 1:
            alpha, beta, gamma = 0.020, 0.0045, 0.0025
        else:
            alpha, beta, gamma = 0.034, 0.0075, 0.0040
            
        if ek <= 2500.0:
            raw_risk = 14.0 + alpha * max(0.0, ek - 500.0)
        elif ek <= 7500.0:
            r_2500 = 14.0 + alpha * 2000.0
            raw_risk = r_2500 + beta * (ek - 2500.0)
        else:
            r_2500 = 14.0 + alpha * 2000.0
            r_7500 = r_2500 + beta * 5000.0
            raw_risk = r_7500 + gamma * (ek - 7500.0)
            
        for ticker in ticker_list:
            if len(positions) >= allowed_slots:
                break
            if ticker not in data_map or ticker in positions or ticker in daily_traded[cur_date]:
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
            bar_vol = t_data["v"].iloc[idx]
            cur_atr = max(t_data["atr14"].iloc[idx], 0.01)
            cur_rvol = t_data["rvol"].iloc[idx]
            cur_er = t_data["er10"].iloc[idx]
            cur_turnover = t_data["turnover_med20"].iloc[idx]
            
            # Mechanic 3: Microstructure Admissions Gate
            if use_microstructure_filter:
                # Law 1: Tick quantization
                if (0.05 / cur_atr) > tick_ratio_max:
                    continue
                # Law 2: NATR
                natr = (cur_atr / cur_c) * 100.0
                if natr < natr_min:
                    continue
                # Law 3: Turnover
                if cur_turnover < turnover_min:
                    continue
                # Law 4: ER
                if cur_er < er10_min:
                    continue
                # Dynamic conviction weight
                er_bonus = 0.25 * max(0.0, cur_er - er10_min)
                rvol_bonus = 0.05 * min(max(0.0, cur_rvol - 1.5), 3.0)
                weight = min(1.20, max(0.85, 0.85 + er_bonus + rvol_bonus))
            else:
                weight = 1.0
                
            target_risk = min(raw_risk * weight, max_risk_cap)
            
            raw_entry = cur_c * (1.0 + slippage)
            entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
            
            # Decoupled Sizing: hard_sl at 0.75x, denominator at 1.00x
            active_sl = round(entry_p - (hard_sl_mult * cur_atr), 2)
            friction = 0.0019 * entry_p
            denom = (sizing_denom_mult * cur_atr) + friction
            
            qty_r = int(target_risk / denom)
            slots_free = max(1, allowed_slots - len(positions))
            slot_notional = avail_notional / slots_free
            qty_m = int((slot_notional * 0.95) / entry_p)
            qty = min(qty_r, qty_m)
            if bar_vol > 0:
                qty = min(qty, int(0.045 * bar_vol))
                
            if qty >= 1:
                positions[ticker] = {
                    "entry_p": entry_p, "qty": qty, "entry_atr": cur_atr,
                    "active_sl": active_sl, "high_seen": entry_p, "bars_held": 0
                }
                daily_traded[cur_date].add(ticker)
                current_notional = sum(p["entry_p"] * p["qty"] for p in positions.values())
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
        "worst": worst, "mdd": mdd
    }

print("=== RUNNING V18 COMPARATIVE CONFIGURATIONS ===")
dm_top10 = prepare_data(basket_top10)
dm_ver15 = prepare_data(basket_verified15)
dm_arg20 = prepare_data(basket_argon20)

configs = [
    ("Top-10 Basket, No Micro, Decoupled 0.75, 2-Slots", dm_top10, basket_top10, False, 0.75, 1.00, 2),
    ("Top-10 Basket, Micro Calibrated, Decoupled 0.75, 2-Slots", dm_top10, basket_top10, True, 0.75, 1.00, 2),
    ("Verified-15 Basket, Micro Calibrated, Decoupled 0.75, 2-Slots", dm_ver15, basket_verified15, True, 0.75, 1.00, 2),
    ("Verified-15 Basket, Micro Calibrated, Decoupled 0.75, 3-Slots", dm_ver15, basket_verified15, True, 0.75, 1.00, 3),
    ("Argon-20 Basket, Micro Calibrated, Decoupled 0.75, 2-Slots", dm_arg20, basket_argon20, True, 0.75, 1.00, 2),
]

for label, dm, b_list, use_micro, sl_m, den_m, slots in configs:
    res = run_simulation(
        dm, b_list,
        hard_sl_mult=sl_m, sizing_denom_mult=den_m,
        max_risk_cap=135.0, max_slots=slots,
        use_microstructure_filter=use_micro
    )
    print(f"\n{label}:")
    print(f"Trades={res['n']} | WR={res['wr']:.1f}% | PF={res['pf']:.2f} | Net=Rs {res['net']:.2f} | Final=Rs {res['final']:.2f}")
    print(f"AvgWin=Rs {res['avg_win']:.2f} | AvgLoss=Rs {res['avg_loss']:.2f} | Worst=Rs {res['worst']:.2f} | MDD={res['mdd']:.1f}%")
