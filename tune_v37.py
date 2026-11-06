import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

from strategies.V36.run_v36 import (
    V29_30_UNIVERSE, SECTOR_MAP, SYM_TO_SECTOR,
    compute_vwap_shifted, compute_er10, compute_adx_di
)

# Load 2026 data
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    u1 = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    u2 = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
    u3 = pickle.load(f)
p2026 = {}
p2026.update(u1); p2026.update(u2); p2026.update(u3)

def run_sim_candidate(
    raw_pool, start, end, label,
    universe_filter=None,
    max_slots=2,
    dd_guard_thresh=0.10,          # candidate 1: if dd > dd_guard_thresh, restrict secondary triggers
    symbol_loss_cooldown_days=1,    # candidate 2: if symbol lost via HARD_SL, skip for N days
    t1b_rvol_min=1.30,              # candidate 3: t1b rvol requirement
    t1b_adx_min=0.0                 # candidate 3: t1b adx requirement
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
        hist = macd - sig
        hist_prev = hist.shift(1)
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
        df_full["day_ret"] = (df_full["close"] / day_open) - 1.0
        df_full["1h_ret"] = (df_full["close"] / df_full["close"].shift(4)) - 1.0

        univ_cond = (
            (c > ema50) & (c > ema110) &
            (c > vwap_lag) &
            (atr_pct >= 0.50) &
            mom_cond &
            (c <= 450.0) & (c >= 10.0)
        )
        if clean not in V29_30_UNIVERSE:
            phi_friction = (0.05 + 0.00206 * c) / (atr14 + 1e-9)
            unseen_ok = (c >= 30.0) & (atr14 >= 0.25) & (phi_friction <= 0.55)
            univ_cond = univ_cond & unseen_ok

        t1_a = (
            macd_cross_up & (macd > 0) &
            (c > ema20) &
            (rsi7 > 50.0) & (adx >= 18.0) & (pdi > mdi) &
            (er10 >= 0.35) &
            (rvol >= 1.20)
        )
        t1_tqs = (t1_a & (rvol >= 1.25) & (c > high_5))

        h_max2 = np.maximum(h.shift(1), h.shift(2))
        h_max2_prev = np.maximum(h.shift(2), h.shift(3))
        t1_b = (
            (macd > sig) & (sig > 0) & (hist > hist_prev) &
            (c > h_max2) & (c.shift(1) <= h_max2_prev) &
            (ema9 > ema20) & (ema20 > ema50) &
            (rsi7 >= 54.0) & (rsi7 <= 76.0) &
            (er10 >= 0.40) & (rvol >= t1b_rvol_min) & (adx >= t1b_adx_min) &
            (((c - ema9) / (atr14 + 1e-9)) <= 1.80)
        )

        sup_prev = np.maximum(ema9.shift(1), vwap_lag) + (0.22 * atr14)
        l_min2 = np.minimum(l.shift(1), l.shift(2))
        v_min2 = np.minimum(v.shift(1), v.shift(2))
        v_max2_prev = np.maximum(v.shift(2), v.shift(3))
        t2_pullback = (
            (macd > sig) & (macd > 0) & (pdi > mdi) &
            (c > ema9) &
            (l_min2 <= sup_prev) &
            (v_min2 < v_max2_prev) & (v >= 1.12 * v.shift(1)) & (rvol >= 1.15) &
            (c > h.shift(1)) & (c > o) & (c.shift(1) <= h.shift(2)) &
            (rsi7 >= 52.0) & (rsi7 <= 75.0) &
            (((c - ema20) / (atr14 + 1e-9)) <= 2.15)
        )

        h_box = h.rolling(4).max().shift(1)
        l_box = l.rolling(4).min().shift(1)
        box_range = h_box - l_box
        t3_squeeze = (
            (box_range <= 1.65 * atr14) & (l_box >= vwap_lag - 0.15 * atr14) &
            (c > h_box) & (c.shift(1) <= h_box) & (c > ema9) &
            (rvol >= 1.30) & (((c - l) / (h - l + 1e-6)) >= 0.60) &
            (macd >= sig) & (rsi7 >= 54.0) & (rsi7 <= 78.0)
        )

        data_map[clean] = {
            "df": df_full, "c": c, "h": h, "l": l, "v": v, "o": o, "atr14": atr14,
            "ema9": ema9, "ema20": ema20, "rsi7": rsi7, "adx": adx, "vwap_lag": vwap_lag,
            "univ_cond": univ_cond,
            "t1_a": t1_a, "t1_tqs": t1_tqs, "t1_b": t1_b,
            "t2_pullback": t2_pullback, "t3_squeeze": t3_squeeze,
            "day_ret": df_full["day_ret"], "1h_ret": df_full["1h_ret"],
            "c_300": c_300, "day_open": day_open, "rvol": rvol, "er10": er10
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
    ret_matrix = np.zeros((len(all_ts), N_stocks), dtype=float)

    for i, s in enumerate(ticker_list):
        s_c = data_map[s]["c"].reindex(all_ts).values
        s_vwap = data_map[s]["vwap_lag"].reindex(all_ts).values
        s_ex = data_map[s]["excess_1h"].reindex(all_ts).values
        bullish_matrix[:, i] = (s_c > s_vwap) & (s_ex > 0)
        ret_matrix[:, i] = data_map[s]["1h_ret"].reindex(all_ts).fillna(0.0).values

    sec_p_indices = {}
    for sec, syms in SECTOR_MAP.items():
        sec_p_indices[sec] = {}
        for s in syms:
            if s in ticker_to_idx:
                sec_p_indices[sec][s] = [ticker_to_idx[p] for p in syms if p != s and p in ticker_to_idx]

    unseen_peers = {}
    if any(s not in V29_30_UNIVERSE for s in ticker_list) and N_stocks > 5:
        cov_mat = np.corrcoef(ret_matrix.T)
        cov_mat = np.nan_to_num(cov_mat, nan=0.0)
        for i, s in enumerate(ticker_list):
            if s not in V29_30_UNIVERSE:
                peers = [j for j in np.argsort(-cov_mat[i]) if j != i and cov_mat[i, j] >= 0.15][:5]
                unseen_peers[s] = peers

    initial_equity = 500.0
    ek = initial_equity
    hwm = ek
    positions = {}
    trades = []
    daily_traded = {}
    symbol_last_loss_date = {}      # symbol cooldown tracker
    last_exit_times = set()
    slippage = 0.0005

    alpha_streak = 0.082
    beta_streak = 0.035

    t1b_end_t = pd.Timestamp("11:00").time()
    t2_start_t = pd.Timestamp("09:45").time()
    t2_end_t = pd.Timestamp("13:15").time()
    t3_start_t = pd.Timestamp("11:00").time()
    t3_end_t = pd.Timestamp("14:15").time()

    for ts_idx, ts in enumerate(all_ts):
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
            cur_ema20 = t_data["ema20"].loc[ts]
            cur_adx = t_data["adx"].loc[ts]
            cur_rvol = t_data["rvol"].loc[ts]
            cur_atr = t_data["atr14"].loc[ts]

            p["bars_held"] += 1
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0

            shooting_star = (cur_rvol >= 2.60) and (((cur_h - cur_c) / (cur_h - cur_l + 1e-6)) >= 0.45)
            extreme_rsi = (cur_rsi >= 86.0)
            break_ema9 = (cur_c < p.get("prev_l", cur_l)) and (cur_c < cur_ema9)
            circuit_prox = (mfe_atr >= 8.50)
            psi_climax = shooting_star or extreme_rsi or break_ema9 or circuit_prox

            theta_expand = (cur_c > cur_ema9 > cur_ema20) and (cur_adx >= 25.0 or cur_rvol >= 2.0) and (cur_rsi >= 58.0) and (not psi_climax)
            nu = min(1.65, max(1.0, cur_atr / (p["entry_atr"] + 1e-9)))

            prev_l1 = p.get("prev_l1", cur_l)
            prev_l2 = p.get("prev_l2", cur_l)

            if mfe_atr >= 2.50:
                if psi_climax:
                    stop_ladder = p["high_seen"] - (0.08 * p["entry_atr"])
                elif theta_expand:
                    stop_ladder = max(
                        p["entry_p"] + 1.00 * p["entry_atr"],
                        min(prev_l1, prev_l2) - 0.08 * p["entry_atr"],
                        p["high_seen"] - (0.36 * nu * p["entry_atr"])
                    )
                else:
                    stop_ladder = p["high_seen"] - (0.18 * p["entry_atr"])
            elif mfe_atr >= 1.25:
                if theta_expand:
                    stop_ladder = max(
                        p["entry_p"] + 0.55 * p["entry_atr"],
                        min(prev_l1, prev_l2) - 0.10 * p["entry_atr"],
                        p["high_seen"] - (0.50 * p["entry_atr"])
                    )
                else:
                    stop_ladder = p["high_seen"] - (0.28 * p["entry_atr"])
            elif mfe_atr >= 0.85:
                stop_ladder = p["entry_p"] + (0.40 * p["entry_atr"])
            else:
                stop_ladder = p["entry_p"] - (0.90 * p["entry_atr"])

            p["active_sl"] = max(p["active_sl"], stop_ladder)
            p["prev_l2"] = prev_l1
            p["prev_l1"] = cur_l
            p["prev_l"] = cur_l

            def finalize_trade(exit_price, reason):
                nonlocal ek, hwm
                gross = round((exit_price - p["entry_p"]) * p["qty"], 2)
                chgs = round(calc_dhan_charges(p["entry_p"], exit_price, p["qty"], is_intraday=True), 2)
                net = round(gross - chgs, 2)
                t_record = {
                    "symbol": ticker,
                    "direction": "LONG",
                    "entry_time": str(p["entry_time"]),
                    "exit_time": str(ts),
                    "entry_price": p["entry_p"],
                    "exit_price": exit_fill,
                    "shares": p["qty"],
                    "sl": p["initial_sl"],
                    "gross_pnl": gross,
                    "charges": chgs,
                    "net_pnl": net,
                    "exit_reason": reason
                }
                trades.append(t_record)
                ek += net
                hwm = max(hwm, ek)
                daily_traded[cur_date].add(ticker)
                last_exit_times.add(ts)
                if net < 0 and reason == "HARD_SL":
                    symbol_last_loss_date[ticker] = cur_date
                del positions[ticker]

            if cur_l <= p["active_sl"]:
                raw_fill = min(cur_h, p["active_sl"]) * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                reason = "TRAIL_SL" if p["active_sl"] >= p["entry_p"] else "HARD_SL"
                finalize_trade(exit_fill, reason)
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

        ek = initial_equity + sum(t["net_pnl"] for t in trades)
        hwm = max(hwm, ek)
        dd_pct = (hwm - ek) / max(hwm, initial_equity)

        k_open = len(positions)
        if k_open >= max_slots or ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
            continue

        b_day_now = bench_day.get(ts, 0.0)
        b300_now = c300_breadth.get(ts, 0.5)
        is_severe_bleed_day = (b_day_now < -0.006)
        is_bear_env = (b300_now < 0.35) and (b_day_now < 0.0)

        if k_open == 1:
            p1 = list(positions.values())[0]
            mfe_1 = (p1["high_seen"] - p1["entry_p"]) / p1["entry_atr"]
            p1_c = data_map[p1["symbol"]]["c"].get(ts, p1["entry_p"])
            p1_ema9 = data_map[p1["symbol"]]["ema9"].get(ts, p1["entry_p"])
            is_p1_freeroll = (mfe_1 >= 0.85) and (p1["active_sl"] >= p1["entry_p"] + 0.40 * p1["entry_atr"]) and (p1_c > p1_ema9)
            if not (is_p1_freeroll and b_day_now >= -0.0030):
                continue
        elif k_open == 2:
            p1, p2 = list(positions.values())
            mfe_1 = (p1["high_seen"] - p1["entry_p"]) / p1["entry_atr"]
            mfe_2 = (p2["high_seen"] - p2["entry_p"]) / p2["entry_atr"]
            is_p1_free = (mfe_1 >= 0.85) and (p1["active_sl"] >= p1["entry_p"] + 0.40 * p1["entry_atr"])
            is_p2_free = (mfe_2 >= 0.85) and (p2["active_sl"] >= p2["entry_p"] + 0.40 * p2["entry_atr"])
            if not (is_p1_free and is_p2_free and b_day_now >= 0.0):
                continue

        avail_margin = (0.95 * ek) - sum((p["entry_p"] * p["qty"]) / 5.0 for p in positions.values())
        if avail_margin < 15.0:
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
            gate_v2_cap = 0.090 * ek
        else:
            raw_r = 14.0 + (alpha * 2000.0) + (beta * (ek - 2500.0))
            gate_v2_cap = min(210.0 + 0.040 * (ek - 2500.0), 250.0)

        if dd_pct > 0.15:
            gate_v2_cap *= 0.50
        target_risk = min(raw_r, gate_v2_cap)

        t_bullish = bullish_matrix[ts_idx]
        sum_bullish = int(np.sum(t_bullish))

        k_cluster = {}
        for clean in data_map.keys():
            sec = SYM_TO_SECTOR.get(clean)
            tick_i = ticker_to_idx[clean]
            if sec and clean in sec_p_indices.get(sec, {}):
                p_idxs = sec_p_indices[sec][clean]
                if p_idxs:
                    k_cluster[clean] = float(np.mean(t_bullish[p_idxs]))
                else:
                    k_cluster[clean] = 0.0
            elif clean in unseen_peers and unseen_peers[clean]:
                p_idxs = unseen_peers[clean]
                k_cluster[clean] = float(np.mean(t_bullish[p_idxs]))
            else:
                k_cluster[clean] = float(sum_bullish - t_bullish[tick_i]) / max(1, N_stocks - 1)

        candidates = []
        for ticker in data_map.keys():
            if ticker in positions or ticker in daily_traded[cur_date]:
                continue

            # Candidate 2: Symbol Loss Cooldown check
            if symbol_loss_cooldown_days > 0 and ticker in symbol_last_loss_date:
                days_since_loss = (cur_date - symbol_last_loss_date[ticker]).days
                if days_since_loss <= symbol_loss_cooldown_days:
                    continue

            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            idx = t_data["df"].index.get_loc(ts)
            if idx < 100:
                continue

            if not (t_data["univ_cond"].iloc[idx] and t_data["g1_cond"].iloc[idx] and t_data["g2_cond"].iloc[idx]):
                continue

            k_cl = k_cluster.get(ticker, 0.0)
            if ticker not in V29_30_UNIVERSE:
                if k_cl < 0.25:
                    continue

            if is_bear_env:
                decoupling = ((t_data["c"].iloc[idx] - t_data["day_open"].iloc[idx]) / (t_data["day_open"].iloc[idx] + 1e-9)) - b_day_now
                auth_pass = (k_cl >= 0.25) or (decoupling >= 0.0120 and t_data["rvol"].iloc[idx] >= 1.50)
                if not auth_pass:
                    continue

            if is_severe_bleed_day:
                pass_trigger = t_data["t1_tqs"].iloc[idx]
            else:
                pass_t1a = t_data["t1_a"].iloc[idx]
                pass_t1b = (t_data["t1_b"].iloc[idx] and (b_day_now >= -0.0030) and (t_time <= t1b_end_t))
                pass_t2  = (t_data["t2_pullback"].iloc[idx] and (b_day_now >= -0.0040) and (t2_start_t <= t_time <= t2_end_t))
                pass_t3  = (t_data["t3_squeeze"].iloc[idx] and (b_day_now >= -0.0020) and (t3_start_t <= t_time <= t3_end_t))

                # Candidate 1: DD Guard
                if dd_guard_thresh > 0 and dd_pct >= dd_guard_thresh:
                    # When in drawdown >= dd_guard_thresh, only permit primary setups
                    pass_trigger = pass_t1a or pass_t2
                else:
                    pass_trigger = pass_t1a or pass_t1b or pass_t2 or pass_t3

            if not pass_trigger:
                continue

            z_mom = t_data["excess_1h"].iloc[idx]
            z_vol = min(3.0, t_data["rvol"].iloc[idx]) / 3.0
            z_er = min(1.0, t_data["er10"].iloc[idx])
            bar_h, bar_l, bar_c = t_data["h"].iloc[idx], t_data["l"].iloc[idx], t_data["c"].iloc[idx]
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
        slots_needed = max_slots - len(positions)
        selected = candidates[:slots_needed]

        for best in selected:
            b_tick, b_c, b_h, b_l, b_v, b_atr, _ = best
            raw_fill = b_c * (1.0 + slippage)
            entry_fill = round(min(b_h, max(b_l, raw_fill)), 2)

            planned_sl = entry_fill - (0.90 * b_atr)
            sl_dist = max(entry_fill - planned_sl, 0.05)
            friction = 0.002067 * entry_fill
            risk_per_share = sl_dist + friction

            q_risk = int(target_risk / risk_per_share)
            q_margin = int((avail_notional * 0.90) / entry_fill)
            q_volume = int(0.05 * b_v)

            q = min(q_risk, q_margin, q_volume)
            if q < 1:
                continue

            pos_margin = (entry_fill * q) / 5.0
            avail_margin -= pos_margin
            avail_notional = avail_margin * 5.0

            positions[b_tick] = {
                "symbol": b_tick,
                "entry_time": ts,
                "entry_p": entry_fill,
                "entry_atr": b_atr,
                "initial_sl": planned_sl,
                "qty": q,
                "active_sl": planned_sl,
                "high_seen": b_h,
                "bars_held": 0,
                "prev_l": b_l,
                "prev_l1": b_l,
                "prev_l2": b_l
            }

    tdf = pd.DataFrame(trades)
    final_cap = initial_equity + sum(t["net_pnl"] for t in trades)
    wr = (len(tdf[tdf['net_pnl'] > 0]) / len(tdf)) * 100.0 if not tdf.empty else 0.0
    net = tdf['net_pnl'].sum() if not tdf.empty else 0.0
    wins = tdf[tdf['net_pnl'] > 0]
    losses = tdf[tdf['net_pnl'] < 0]
    net_win = wins['net_pnl'].sum() if not wins.empty else 0.0
    net_loss = abs(losses['net_pnl'].sum()) if not losses.empty else 0.0
    pf = round(net_win / net_loss, 2) if net_loss > 0 else 99.99

    cum = initial_equity + (tdf['net_pnl'].cumsum() if not tdf.empty else pd.Series([0.0]))
    hwm_s = cum.cummax()
    dd_s = (hwm_s - cum) / hwm_s
    max_dd = dd_s.max() * 100.0 if len(dd_s) > 0 else 0.0

    print(f"  [{label}] -> Trades: {len(tdf):3d} | WR: {wr:5.1f}% | Net: Rs {net:+8.2f} | PF: {pf:5.2f} | Max DD: {max_dd:4.1f}% | End: Rs {final_cap:.2f}")
    return final_cap, trades, max_dd

# Now let us test candidates on 2026 Bull
print("Testing Candidates on 2026 Bull Horizon:")
print("\n--- BASELINE V36 ---")
run_sim_candidate(p2026, "2026-01-01", "2026-12-31", "Baseline V36", universe_filter=V29_30_UNIVERSE, dd_guard_thresh=0.0, symbol_loss_cooldown_days=0)

print("\n--- CANDIDATE 1: DD Guard at 10% (restricts secondary triggers when DD >= 10%) ---")
run_sim_candidate(p2026, "2026-01-01", "2026-12-31", "DD Guard 10%", universe_filter=V29_30_UNIVERSE, dd_guard_thresh=0.10, symbol_loss_cooldown_days=0)

print("\n--- CANDIDATE 2: Symbol Loss Cooldown 1 Day (skip same symbol for 1 day after HARD_SL) ---")
run_sim_candidate(p2026, "2026-01-01", "2026-12-31", "Symbol Cooldown 1D", universe_filter=V29_30_UNIVERSE, dd_guard_thresh=0.0, symbol_loss_cooldown_days=1)

print("\n--- CANDIDATE 3: Stricter T1B (RVOL >= 1.50) ---")
run_sim_candidate(p2026, "2026-01-01", "2026-12-31", "T1B RVOL>=1.50", universe_filter=V29_30_UNIVERSE, dd_guard_thresh=0.0, symbol_loss_cooldown_days=0, t1b_rvol_min=1.50)

print("\n--- CANDIDATE 4: Symbol Cooldown 1D + DD Guard 12% ---")
run_sim_candidate(p2026, "2026-01-01", "2026-12-31", "Cooldown + DD Guard 12%", universe_filter=V29_30_UNIVERSE, dd_guard_thresh=0.12, symbol_loss_cooldown_days=1)
