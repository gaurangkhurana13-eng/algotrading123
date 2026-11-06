import sys, pickle, time
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from framework import calc_dhan_charges, calculate_atr
from scripts.test_v42_prototype import compute_vwap_shifted, compute_er10, compute_adx_di

SECTOR_MAP_103 = {
    'FIN': {
        'PNB', 'CENTRALBK', 'IOB', 'J&KBANK', 'UTKARSHBNK', 'RBLBANK', 'CANBK', 'BANKBARODA', 
        'BANKINDIA', 'SBIN', 'ICICIBANK', 'AXISBANK', 'FEDERALBNK', 'IDFCFIRSTB', 'UNIONBANK', 
        'YESBANK', 'EQUITASBNK', 'UCOBANK', 'UJJIVANSFB', 'MAHABANK', 'INDIANB', 'IDBI', 
        'SOUTHBANK', 'BANDHANBNK', 'IFCI', 'PFC', 'RECLTD', 'POONAWALLA', 'MANAPPURAM', 
        'IRFC', 'HUDCO', 'NIACL'
    },
    'INFRA': {
        'IRCON', 'RITES', 'TEXRAIL', 'RVNL', 'NBCC', 'NCC', 'HCC', 'GMRINFRA', 'GMRAIRPORT', 
        'JAICORPLTD', 'LLOYDSENGG', 'DLF', 'BEL', 'MOTHERSON'
    },
    'COMMODITIES': {
        'TATASTEEL', 'NMDC', 'ELECTCAST', 'VEDL', 'HINDALCO', 'JINDALSTEL', 'JSWSTEEL', 'SAIL', 
        'NATIONALUM', 'RCF', 'GSFC', 'MANALIPETC', 'TATACHEM', 'NFL', 'SPIC', 'BCLIND', 
        'RENUKA', 'PATANJALI', 'TRIDENT', 'ALOKINDS', 'MMTC'
    },
    'ENERGY': {
        'SUZLON', 'RPOWER', 'SWSOLAR', 'TATAPOWER', 'ADANIPOWER', 'NTPC', 'NHPC', 'SJVN', 
        'JPPOWER', 'PTC', 'IOC', 'BPCL', 'GAIL', 'COALINDIA', 'ONGC', 'MRPL', 'RELIANCE', 
        'IDEA', 'MTNL', 'TTML'
    },
    'CONSUMER': {
        'PCJEWELLER', 'JYOTHYLAB', 'ITC', 'ZEEL', 'DELTACORP', 'NETWORK18', 'HATHWAY'
    },
    'TECH': {
        'INFY', 'TECHM', 'SUBEXLTD'
    },
    'PHARMA': {
        'SUNPHARMA', 'CIPLA', 'MOREPENLAB'
    },
    'AUTO': {
        'TATAMOTORS', 'ASHOKLEY'
    },
    'CONGLOMERATE': {
        'ADANIENT'
    }
}

SYM_TO_SECTOR_103 = {}
for sec_name, syms in SECTOR_MAP_103.items():
    for s in syms:
        SYM_TO_SECTOR_103[s] = sec_name

V29_30_UNIVERSE = {
    "J&KBANK", "UTKARSHBNK", "MANALIPETC", "IDEA", "PCJEWELLER",
    "NIACL", "MTNL", "IFCI", "TEXRAIL", "SWSOLAR",
    "TTML", "RENUKA", "RCF", "NBCC", "JYOTHYLAB",
    "RBLBANK", "ELECTCAST", "PNB", "GSFC", "TATASTEEL",
    "SUZLON", "CENTRALBK", "IOB", "IRCON", "NMDC",
    "RITES", "NCC", "ZEEL", "DELTACORP", "RPOWER"
}

stocks_30_legacy = {
    'CANBK', 'PNB', 'BANKBARODA', 'SBIN', 'IOC',
    'BPCL', 'GAIL', 'COALINDIA', 'TATASTEEL', 'TATAPOWER',
    'TATAMOTORS', 'VEDL', 'HINDALCO', 'JINDALSTEL', 'JSWSTEEL',
    'BEL', 'ADANIPOWER', 'ADANIENT', 'RECLTD', 'PFC',
    'DLF', 'MOTHERSON', 'ITC', 'ICICIBANK', 'AXISBANK',
    'RELIANCE', 'INFY', 'SUNPHARMA', 'TECHM', 'CIPLA'
}

core_27 = {
    'CANBK', 'DELTACORP', 'FEDERALBNK', 'GMRINFRA', 'GSFC', 'IDFCFIRSTB', 'IOC',
    'JYOTHYLAB', 'MANAPPURAM', 'MOTHERSON', 'NBCC', 'NCC', 'NATIONALUM', 'NMDC',
    'POONAWALLA', 'PNB', 'RBLBANK', 'RCF', 'SAIL', 'SUZLON', 'TATACHEM',
    'TATASTEEL', 'TATAPOWER', 'TRIDENT', 'UNIONBANK', 'YESBANK', 'ZEEL'
}

u_2021 = {'CENTRALBK', 'DELTACORP', 'GSFC', 'IDEA', 'IOB', 'JYOTHYLAB', 'NBCC', 'NCC', 'NIACL', 'NMDC', 'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'SUZLON', 'SWSOLAR', 'TATASTEEL', 'TTML', 'ZEEL'}

def run_multi_slot_sim(
    raw_pool,
    start,
    end,
    universe_filter=None,
    base_risk=25.0,
    initial_equity=500.0,
    max_slots=1,
    slot_unlock_equity=1000.0,
    formula_mode="formula_b",
    enforce_sector_diversification=True
):
    ticker_map = {}
    for k in raw_pool.keys():
        clean = k.replace(".NS", "")
        if universe_filter is not None and clean not in universe_filter:
            continue
        if clean not in ticker_map or not k.endswith(".NS"):
            ticker_map[clean] = k

    data_map = {}
    for clean, raw_k in ticker_map.items():
        df_full = raw_pool[raw_k].loc[start:end].copy()
        if len(df_full) < 150:
            continue
        c, h, l, v, o = df_full["close"], df_full["high"], df_full["low"], df_full["volume"], df_full["open"]
        atr14 = calculate_atr(df_full, 14)
        atr_pct = (atr14 / c) * 100.0

        ema9 = c.ewm(span=9).mean()
        ema20 = c.ewm(span=20).mean()
        ema50 = c.ewm(span=50).mean()
        ema110 = c.ewm(span=110).mean()

        delta = c.diff()
        gain = delta.clip(lower=0).rolling(7).mean()
        loss = (-delta.clip(upper=0)).rolling(7).mean()
        rsi7 = 100.0 - (100.0 / (1.0 + gain / (loss + 1e-9)))

        macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        sig = macd.ewm(span=9).mean()
        macd_cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))

        adx, pdi, mdi = compute_adx_di(df_full, 10)
        vwap_lag = compute_vwap_shifted(df_full)
        v_sma20 = v.rolling(20).mean()
        rvol = v / (v_sma20 + 1e-9)
        er10 = compute_er10(c)

        high_5 = h.rolling(5).max().shift(1)
        lookback = min(300, len(c) - 1)
        c_300 = c.shift(lookback)
        mom_cond = (c > c_300)

        df_full["_date"] = df_full.index.date
        day_open = df_full.groupby("_date")["open"].transform("first")
        daily_close = df_full.groupby("_date")["close"].last()
        prev_close_map = daily_close.shift(1).to_dict()
        df_full["prev_close"] = df_full["_date"].map(prev_close_map)
        df_full["open_gap_pct"] = (day_open - df_full["prev_close"]) / (df_full["prev_close"] + 1e-9) * 100.0
        df_full["day_ret"] = (df_full["close"] / day_open) - 1.0
        df_full["1h_ret"] = (df_full["close"] / df_full["close"].shift(4)) - 1.0

        univ_cond = (
            (c > ema50) & (c > ema110) &
            (c > vwap_lag) &
            (atr_pct >= 0.50) &
            mom_cond &
            (c <= 450.0) & (c >= 10.0)
        )

        t1_base = (
            macd_cross_up & (macd > 0) &
            (c > ema20) &
            (rsi7 > 50.0) & (adx >= 18.0) & (pdi > mdi) &
            (er10 >= 0.35) &
            (rvol >= 1.30)
        )
        t1_tqs = (t1_base & (rvol >= 1.35) & (c > high_5))

        sup_prev = np.maximum(ema9.shift(1), vwap_lag.shift(1)) + (0.15 * atr14)
        t2_pullback = (
            (macd > sig) & (macd > 0) & (pdi > mdi) &
            (ema9 > ema20) & (ema20 > ema50) &
            (c > ema9) &
            (l.shift(1) <= sup_prev) &
            (v.shift(1) < v.shift(2)) & (v >= 1.15 * v.shift(1)) & (rvol >= 1.30) &
            (c > h.shift(1)) & (c.shift(1) <= h.shift(2)) & (c > o) &
            (rsi7 >= 52.0) & (rsi7 <= 74.0) &
            (((c - ema20) / (atr14 + 1e-9)) <= 2.10)
        )

        data_map[clean] = {
            "df": df_full, "c": c, "h": h, "l": l, "v": v, "o": o, "atr14": atr14,
            "ema9": ema9, "ema20": ema20, "rsi7": rsi7, "adx": adx, "vwap_lag": vwap_lag,
            "rvol": rvol, "er10": er10,
            "univ_cond": univ_cond, "t1_base": t1_base, "t1_tqs": t1_tqs, "t2_pullback": t2_pullback,
            "day_ret": df_full["day_ret"], "1h_ret": df_full["1h_ret"],
            "c_300": c_300, "day_open": day_open, "open_gap_pct": df_full["open_gap_pct"]
        }

    bench_day = pd.concat([d["day_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    bench_1h  = pd.concat([d["1h_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    c300_breadth = pd.concat([(d["c"] > d["c_300"]).astype(float) for d in data_map.values()], axis=1, sort=False).mean(axis=1)

    for clean, d in data_map.items():
        b_d = bench_day.reindex(d["day_ret"].index)
        b_1 = bench_1h.reindex(d["1h_ret"].index)
        d["g1_cond"] = (d["day_ret"] > b_d)
        d["g2_cond"] = (d["1h_ret"] > b_1)
        d["excess_1h"] = d["1h_ret"] - b_1

    all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
    ticker_list = sorted(list(data_map.keys()))
    ticker_to_idx = {sym: i for i, sym in enumerate(ticker_list)}
    N_stocks = len(ticker_list)

    bullish_matrix = np.zeros((len(all_ts), N_stocks), dtype=bool)
    for i, s in enumerate(ticker_list):
        s_c = data_map[s]["c"].reindex(all_ts).ffill().to_numpy()
        s_vwap = data_map[s]["vwap_lag"].reindex(all_ts).ffill().to_numpy()
        s_ex = data_map[s]["excess_1h"].reindex(all_ts).fillna(0.0).to_numpy()
        bullish_matrix[:, i] = (s_c > s_vwap) & (s_ex > 0)

    sec_p_indices = {}
    for sec, syms in SECTOR_MAP_103.items():
        sec_p_indices[sec] = {}
        for s in syms:
            if s in ticker_to_idx:
                sec_p_indices[sec][s] = [ticker_to_idx[p] for p in syms if p != s and p in ticker_to_idx]

    ek = initial_equity
    hwm = ek
    positions = {}
    trades = []
    daily_traded = {}
    daily_losses = {}
    symbol_lockouts = {}
    last_exit_times = set()
    slippage = 0.0005
    hard_sl_mult = 0.90

    t2_start_t = pd.Timestamp("09:45").time()
    t2_end_t = pd.Timestamp("13:15").time()

    # Metrics
    total_market_bars = 0
    bars_with_pos = 0
    bars_with_2_pos = 0
    sum_margin_util = 0.0
    max_margin_util = 0.0

    for ts_idx, ts in enumerate(all_ts):
        if ts.weekday() >= 5:
            continue
        cur_date = ts.date()
        if cur_date not in daily_traded:
            daily_traded[cur_date] = set()
        if cur_date not in daily_losses:
            daily_losses[cur_date] = 0
        t_time = ts.time()

        b_c300 = c300_breadth.get(ts, 0.50)
        b_day_now = bench_day.get(ts, 0.0)

        is_bear_env = (b_c300 < 0.40) or (b_day_now < -0.0060)
        is_severe_bleed_day = (b_day_now < -0.0120)

        # 1. Manage Active Positions
        for ticker in list(positions.keys()):
            p = positions[ticker]
            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            row = t_data["df"].loc[ts]
            cur_c = row["close"]
            cur_h = row["high"]
            cur_l = row["low"]
            cur_rsi = t_data["rsi7"].loc[ts]
            cur_vwap = t_data["vwap_lag"].loc[ts]
            cur_ema9 = t_data["ema9"].loc[ts]
            cur_ema20 = t_data["ema20"].loc[ts]
            cur_adx = t_data["adx"].loc[ts]
            cur_rvol = t_data["rvol"].loc[ts]

            p["bars_held"] += 1
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0

            shooting_star = (cur_rvol >= 2.60) and (((cur_h - cur_c) / (cur_h - cur_l + 1e-6)) >= 0.45)
            extreme_rsi = (cur_rsi >= 86.0)
            break_ema9 = (cur_c < p.get("prev_l", cur_l)) and (cur_c < cur_ema9)
            circuit_prox = (mfe_atr >= 8.00)
            psi_climax = shooting_star or extreme_rsi or break_ema9 or circuit_prox

            theta_expand = (cur_c > cur_ema9 > cur_ema20) and (cur_adx >= 24.0 or cur_rvol >= 1.80) and (cur_rsi >= 56.0) and (not psi_climax)
            prev_l = p.get("prev_l", cur_l)

            if mfe_atr >= 3.50:
                if psi_climax:
                    stop_ladder = p["high_seen"] - (0.10 * p["entry_atr"])
                elif theta_expand:
                    stop_ladder = max(p["entry_p"] + (2.00 * p["entry_atr"]), prev_l - (0.08 * p["entry_atr"]), p["high_seen"] - (0.35 * p["entry_atr"]))
                else:
                    stop_ladder = p["high_seen"] - (0.18 * p["entry_atr"])
            elif mfe_atr >= 2.25:
                if psi_climax:
                    stop_ladder = p["high_seen"] - (0.10 * p["entry_atr"])
                elif theta_expand:
                    stop_ladder = max(p["entry_p"] + (1.20 * p["entry_atr"]), prev_l - (0.10 * p["entry_atr"]), p["high_seen"] - (0.35 * p["entry_atr"]))
                else:
                    stop_ladder = p["high_seen"] - (0.20 * p["entry_atr"])
            elif mfe_atr >= 1.25:
                if theta_expand:
                    stop_ladder = max(p["entry_p"] + (0.55 * p["entry_atr"]), prev_l - (0.12 * p["entry_atr"]), p["high_seen"] - (0.38 * p["entry_atr"]))
                else:
                    stop_ladder = p["high_seen"] - (0.28 * p["entry_atr"])
            elif mfe_atr >= 0.85:
                stop_ladder = p["entry_p"] + (0.40 * p["entry_atr"])
            elif mfe_atr >= 0.45:
                stop_ladder = p["entry_p"] + (0.05 * p["entry_atr"])
            else:
                stop_ladder = p["entry_p"] - (0.90 * p["entry_atr"])

            if mfe_atr >= 0.85 and cur_c < cur_ema9 and cur_rsi < 55.0:
                stop_ladder += 0.05 * p["entry_atr"]

            p["active_sl"] = max(p["active_sl"], round(stop_ladder, 2))
            p["prev_l"] = cur_l

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
                    "exit_reason": exit_reason,
                    "slot": p.get("slot", 1)
                })
                last_exit_times.add(ts)
                if net < 0:
                    daily_losses[cur_date] += 1
                if exit_reason == "STOP" and exit_fill <= p["entry_p"]:
                    symbol_lockouts[ticker] = cur_date + pd.Timedelta(days=2)
                del positions[ticker]

            if cur_l <= p["active_sl"]:
                raw_fill = min(cur_h, p["active_sl"]) * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                finalize_trade(exit_fill, "STOP")
                continue
            if p["bars_held"] >= 4 and mfe_atr < 0.40 and cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                raw_fill = cur_c * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                finalize_trade(exit_fill, "SCRATCH")
                continue
            if t_time >= pd.Timestamp("14:45").time():
                raw_fill = cur_c * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                finalize_trade(exit_fill, "EOD")
                continue

        # Dynamic Equity & Risk Budget
        ek = initial_equity + sum(t["net_pnl"] for t in trades)
        hwm = max(hwm, ek)
        dd_pct = (hwm - ek) / max(hwm, initial_equity)

        # Track Capital Utilization across all trading bars
        if pd.Timestamp("09:15").time() <= t_time <= pd.Timestamp("15:30").time():
            total_market_bars += 1
            n_pos = len(positions)
            if n_pos >= 1:
                bars_with_pos += 1
            if n_pos >= 2:
                bars_with_2_pos += 1
            cur_margin = sum((p["entry_p"] * p["qty"]) / 5.0 for p in positions.values())
            util_frac = cur_margin / max(ek, 1.0)
            sum_margin_util += util_frac
            max_margin_util = max(max_margin_util, util_frac)

        # Slot Management Gate
        if ek >= slot_unlock_equity and max_slots >= 2:
            current_max_slots = 2
        else:
            current_max_slots = 1

        if len(positions) >= current_max_slots or ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
            continue

        if pd.Timestamp("11:30").time() <= t_time <= pd.Timestamp("12:30").time():
            continue
        if pd.Timestamp("10:00").time() <= t_time <= pd.Timestamp("10:05").time():
            continue

        if daily_losses.get(cur_date, 0) >= 2:
            continue

        avail_margin_total = (0.95 * ek) - sum((p["entry_p"] * p["qty"]) / 5.0 for p in positions.values())
        if avail_margin_total < 10.0:
            continue

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
            alpha, beta = 0.085, 0.038

        # Sizing and Risk Calculation per Slot
        if current_max_slots == 1:
            if ek <= 2500.0:
                raw_r = base_risk + (alpha * max(0.0, ek - 500.0))
            else:
                raw_r = base_risk + (alpha * 2000.0) + (beta * (ek - 2500.0))
            gate_v2_cap = min(0.090 * ek, 258.0)
            if dd_pct > 0.15:
                gate_v2_cap *= 0.50
            target_risk_slot = min(raw_r, gate_v2_cap)
            margin_limit_slot = avail_margin_total
        else:
            # Multi-slot sizing formulas
            if formula_mode == "formula_a":
                # Dual-Subwallet (Half-Equity Sizing)
                ek_sub = ek / 2.0
                if ek_sub <= 2500.0:
                    raw_r = base_risk + (alpha * max(0.0, ek_sub - 500.0))
                else:
                    raw_r = base_risk + (alpha * 2000.0) + (beta * (ek_sub - 2500.0))
                gate_v2_cap_slot = min(0.090 * ek_sub, 129.0)
                if dd_pct > 0.15:
                    gate_v2_cap_slot *= 0.50
                target_risk_slot = min(raw_r, gate_v2_cap_slot)
                margin_limit_slot = min(avail_margin_total, (0.95 * ek) * 0.50)

            elif formula_mode == "formula_b":
                # Full Slot Risk Capped at 4.5% Equity per Slot (Combined <= 9.0%)
                if ek <= 2500.0:
                    raw_r = base_risk + (alpha * max(0.0, ek - 500.0))
                else:
                    raw_r = base_risk + (alpha * 2000.0) + (beta * (ek - 2500.0))
                gate_v2_cap_slot = min(0.045 * ek, 129.0)
                if dd_pct > 0.15:
                    gate_v2_cap_slot *= 0.50
                target_risk_slot = min(raw_r, gate_v2_cap_slot)
                margin_limit_slot = min(avail_margin_total, (0.95 * ek) * 0.50)

            elif formula_mode == "formula_c":
                # Dynamic Slot Sizing: Slot 1 full risk (up to 60% margin); Slot 2 uses remaining margin & remaining Gate V2 cap
                if len(positions) == 0:
                    # Slot 1
                    if ek <= 2500.0:
                        raw_r = base_risk + (alpha * max(0.0, ek - 500.0))
                    else:
                        raw_r = base_risk + (alpha * 2000.0) + (beta * (ek - 2500.0))
                    gate_v2_cap = min(0.060 * ek, 180.0) # 6% cap for Slot 1
                    if dd_pct > 0.15:
                        gate_v2_cap *= 0.50
                    target_risk_slot = min(raw_r, gate_v2_cap)
                    margin_limit_slot = min(avail_margin_total, (0.95 * ek) * 0.60)
                else:
                    # Slot 2
                    # Remaining risk budget up to 9% daily cap
                    risk_pos1 = sum(p["qty"] * max(p["entry_p"] - p["initial_sl"], 0.05) for p in positions.values())
                    max_total_risk = min(0.090 * ek, 258.0)
                    if dd_pct > 0.15:
                        max_total_risk *= 0.50
                    rem_risk = max(0.0, max_total_risk - risk_pos1)
                    if ek <= 2500.0:
                        raw_r = base_risk + (alpha * max(0.0, ek - 500.0))
                    else:
                        raw_r = base_risk + (alpha * 2000.0) + (beta * (ek - 2500.0))
                    target_risk_slot = min(raw_r, rem_risk)
                    margin_limit_slot = avail_margin_total

            elif formula_mode == "formula_d":
                # Unconstrained Standard Risk per Slot (Full Risk on both slots)
                if ek <= 2500.0:
                    raw_r = base_risk + (alpha * max(0.0, ek - 500.0))
                else:
                    raw_r = base_risk + (alpha * 2000.0) + (beta * (ek - 2500.0))
                gate_v2_cap = min(0.090 * ek, 258.0)
                if dd_pct > 0.15:
                    gate_v2_cap *= 0.50
                target_risk_slot = min(raw_r, gate_v2_cap)
                margin_limit_slot = min(avail_margin_total, (0.95 * ek) * 0.50)

        t_bullish = bullish_matrix[ts_idx, :]
        sum_bullish = int(np.sum(t_bullish))
        k_cluster = {}
        for clean in data_map.keys():
            sec = SYM_TO_SECTOR_103.get(clean)
            tick_i = ticker_to_idx[clean]
            if sec and clean in sec_p_indices.get(sec, {}):
                p_idxs = sec_p_indices[sec][clean]
                if p_idxs:
                    k_cluster[clean] = float(np.mean(t_bullish[p_idxs]))
                else:
                    k_cluster[clean] = 0.0
            else:
                k_cluster[clean] = float(sum_bullish - t_bullish[tick_i]) / max(1, N_stocks - 1)

        candidates = []
        for ticker in data_map.keys():
            if ticker in positions or ticker in daily_traded[cur_date]:
                continue
            if ticker in symbol_lockouts and cur_date <= symbol_lockouts[ticker]:
                continue
            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            idx = t_data["df"].index.get_loc(ts)
            if idx < 100:
                continue

            if not (t_data["univ_cond"].iloc[idx] and t_data["g1_cond"].iloc[idx] and t_data["g2_cond"].iloc[idx]):
                continue

            gap_pct = t_data["open_gap_pct"].iloc[idx]
            if not np.isnan(gap_pct) and gap_pct > 1.50:
                continue

            if is_bear_env:
                k_cl = k_cluster.get(ticker, 0.0)
                decoupling = ((t_data["c"].iloc[idx] - t_data["day_open"].iloc[idx]) / (t_data["day_open"].iloc[idx] + 1e-9)) - b_day_now
                auth_pass = (k_cl >= 0.25) or (decoupling >= 0.0120 and t_data["rvol"].iloc[idx] >= 1.50)
                if not auth_pass:
                    continue

            if is_severe_bleed_day:
                pass_trigger = t_data["t1_tqs"].iloc[idx]
            else:
                pass_t1 = t_data["t1_base"].iloc[idx]
                pass_t2 = (t_data["t2_pullback"].iloc[idx] and (b_day_now >= -0.0040) and (t2_start_t <= t_time <= t2_end_t))
                pass_trigger = pass_t1 or pass_t2

            if not pass_trigger:
                continue

            z_mom = t_data["excess_1h"].iloc[idx]
            z_vol = min(3.0, t_data["rvol"].iloc[idx]) / 3.0
            z_er = min(1.0, t_data["er10"].iloc[idx])
            bar_h = t_data["h"].iloc[idx]
            bar_l = t_data["l"].iloc[idx]
            bar_c = t_data["c"].iloc[idx]
            z_spread = (bar_c - bar_l) / (bar_h - bar_l + 1e-6)
            base_score = (0.45 * z_mom) + (0.25 * z_vol) + (0.15 * z_er) + (0.15 * z_spread)
            final_alpha = base_score * (1.0 + 0.35 * k_cluster.get(ticker, 0.0))

            candidates.append((
                ticker,
                bar_c,
                bar_h,
                bar_l,
                t_data["v"].iloc[idx],
                max(t_data["atr14"].iloc[idx], 0.05),
                final_alpha
            ))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[6], reverse=True)
        open_sectors = {SYM_TO_SECTOR_103.get(p, p) for p in positions.keys()}

        for cand in candidates:
            if len(positions) >= current_max_slots:
                break
            ticker, cur_c, cur_h, cur_l, cur_v, entry_atr, _ = cand

            cand_sector = SYM_TO_SECTOR_103.get(ticker, ticker)
            if enforce_sector_diversification and cand_sector in open_sectors:
                continue  # Sector collision! Must be from independent sector

            cur_avail_margin = (0.95 * ek) - sum((p["entry_p"] * p["qty"]) / 5.0 for p in positions.values())
            if cur_avail_margin < 10.0:
                break

            margin_for_trade = min(cur_avail_margin, margin_limit_slot)
            avail_notional = margin_for_trade * 5.0

            raw_entry = cur_c * (1.0 + slippage)
            entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
            active_sl = round(entry_p - (hard_sl_mult * entry_atr), 2)
            sl_dist = max(entry_p - active_sl, 0.05)
            friction = 0.002067 * entry_p
            denom = sl_dist + friction

            qty_r = int(target_risk_slot / denom)
            qty_m = int((avail_notional * 0.90) / entry_p)
            qty_v = int(cur_v * 0.05)
            qty = min(qty_r, qty_m, qty_v)

            if qty < 1:
                continue

            slot_idx = len(positions) + 1
            positions[ticker] = {
                "entry_t": ts,
                "entry_p": entry_p,
                "initial_sl": active_sl,
                "active_sl": active_sl,
                "entry_atr": entry_atr,
                "qty": qty,
                "high_seen": cur_h,
                "prev_l": cur_l,
                "bars_held": 0,
                "sector": cand_sector,
                "slot": slot_idx
            }
            daily_traded[cur_date].add(ticker)
            open_sectors.add(cand_sector)

    final_cap = initial_equity + sum(t["net_pnl"] for t in trades)
    wr = (sum(1 for t in trades if t["net_pnl"] > 0) / len(trades) * 100.0) if trades else 0.0
    net = sum(t["net_pnl"] for t in trades)

    eq_curve = [initial_equity]
    for t in trades:
        eq_curve.append(eq_curve[-1] + t["net_pnl"])
    eq_s = pd.Series(eq_curve)
    pk = eq_s.cummax()
    max_dd = ((pk - eq_s) / pk).max() * 100.0

    time_util_pct = (bars_with_pos / max(1, total_market_bars)) * 100.0
    two_slot_time_pct = (bars_with_2_pos / max(1, total_market_bars)) * 100.0
    avg_margin_util_pct = (sum_margin_util / max(1, total_market_bars)) * 100.0
    peak_margin_util_pct = max_margin_util * 100.0

    return {
        "final_cap": final_cap,
        "net_pnl": net,
        "trades_count": len(trades),
        "win_rate": wr,
        "max_dd": max_dd,
        "time_util_pct": time_util_pct,
        "two_slot_time_pct": two_slot_time_pct,
        "avg_margin_util_pct": avg_margin_util_pct,
        "peak_margin_util_pct": peak_margin_util_pct,
        "trades": trades
    }
