import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT / "strategies" / "V37"))
import run_v37 as v37_base

# Copy functions from run_v37
calc_dhan_charges = v37_base.calc_dhan_charges
calculate_atr = v37_base.calculate_atr
compute_vwap_shifted = v37_base.compute_vwap_shifted
compute_er10 = v37_base.compute_er10
compute_adx_di = v37_base.compute_adx_di
SECTOR_MAP = v37_base.SECTOR_MAP
SYM_TO_SECTOR = v37_base.SYM_TO_SECTOR
V29_30_UNIVERSE = v37_base.V29_30_UNIVERSE

def run_v38_simulation(
    raw_pool,
    start: str, end: str,
    fold_label: str,
    universe_filter=None,
    max_slots: int = 3,
    cooldown_sessions: int = 2,
    consec_loss_brake: bool = True
):
    # Same data loading & indicator computation as V37
    ticker_map = {}
    for k in raw_pool.keys():
        clean = k.replace(".NS", "")
        if universe_filter is not None and clean not in universe_filter:
            continue
        if clean not in ticker_map or not k.endswith(".NS"):
            ticker_map[clean] = k

    data_map = {}
    for clean, orig_k in ticker_map.items():
        df_raw = raw_pool[orig_k].copy()
        if not isinstance(df_raw.index, pd.DatetimeIndex):
            df_raw.index = pd.to_datetime(df_raw.index)
        df_full = df_raw.sort_index()
        if df_full.empty:
            continue

        c, h, l, v, o = df_full["close"], df_full["high"], df_full["low"], df_full["volume"], df_full["open"]
        atr14 = calculate_atr(df_full, 14)
        atr_pct = (atr14 / c) * 100.0

        ema9 = c.ewm(span=9).mean()
        ema20 = c.ewm(span=20).mean()
        ema50 = c.ewm(span=50).mean()
        ema110 = c.ewm(span=110).mean()
        c_300 = c.shift(300)

        delta = c.diff()
        gain = delta.clip(lower=0).rolling(7).mean()
        loss = (-delta.clip(upper=0)).rolling(7).mean()
        rsi7 = 100.0 - (100.0 / (1.0 + gain / (loss + 1e-9)))

        macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        sig = macd.ewm(span=9).mean()
        macd_cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))

        vwap_lag = compute_vwap_shifted(df_full)
        rvol = v / (v.rolling(20).mean() + 1e-9)
        er10 = compute_er10(c)
        adx, pdi, mdi = compute_adx_di(df_full, 10)

        mom_cond = (c > c.shift(1)) & (c.shift(1) > c.shift(2))
        high_5 = h.rolling(5).max().shift(1)

        df_dates = pd.Series(df_full.index.date, index=df_full.index)
        first_bar_mask = df_dates != df_dates.shift(1)
        day_open = df_full["open"].where(first_bar_mask).ffill()
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
            penny_trap = (c >= 30.0) & (atr14 >= 0.30)
            univ_cond = univ_cond & penny_trap

        t1_base = (
            macd_cross_up & (macd > 0) &
            (c > ema20) &
            (rsi7 > 50.0) & (adx >= 18.0) & (pdi > mdi) &
            (er10 >= 0.35)
        )
        t1_a = (t1_base & (rvol >= 1.50) & (c > high_5))
        t1_tqs = (t1_base & (rvol >= 1.25) & (c > high_5))
        t1_b = (t1_base & (rvol >= 1.20) & (c > high_5))

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
        for i, s in enumerate(ticker_list):
            if s not in V29_30_UNIVERSE:
                corrs = cov_mat[i]
                sorted_idx = np.argsort(corrs)[::-1]
                top_peers = [idx for idx in sorted_idx if idx != i and corrs[idx] > 0.15][:3]
                unseen_peers[s] = top_peers

    initial_equity = 500.0
    ek = initial_equity
    hwm = ek
    positions = {}
    trades = []
    daily_traded = {}
    last_exit_times = set()
    slippage = 0.0005

    alpha_streak = 0.082
    beta_streak = 0.035

    t1b_end_t = pd.Timestamp("11:30").time()
    t2_start_t = pd.Timestamp("11:30").time()
    t2_end_t = pd.Timestamp("14:15").time()
    t3_start_t = pd.Timestamp("13:15").time()
    t3_end_t = pd.Timestamp("14:15").time()

    # V38 Universal Defense Architecture
    symbol_sl_lockout = {}
    unique_dates = sorted(list(set(ts.date() for ts in all_ts)))
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}

    for ts_idx, ts in enumerate(all_ts):
        t_str = str(ts)
        cur_date = ts.date()
        cur_d_idx = date_to_idx[cur_date]
        t_time = ts.time()

        if cur_date not in daily_traded:
            daily_traded[cur_date] = set()

        # 1. Manage active positions
        active_tickers = list(positions.keys())
        for ticker in active_tickers:
            p = positions[ticker]
            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            idx = t_data["df"].index.get_loc(ts)

            cur_c = t_data["c"].iloc[idx]
            cur_h = t_data["h"].iloc[idx]
            cur_l = t_data["l"].iloc[idx]
            cur_atr = max(t_data["atr14"].iloc[idx], 0.05)
            cur_rsi = t_data["rsi7"].iloc[idx]
            cur_adx = t_data["adx"].iloc[idx]
            cur_rvol = t_data["rvol"].iloc[idx]
            cur_vwap = t_data["vwap_lag"].iloc[idx]
            cur_ema9 = t_data["ema9"].iloc[idx]
            cur_ema20 = t_data["ema20"].iloc[idx]

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

            # V37/V38 Asymmetric Elastic Runner Channel
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
                    "entry_time": p["entry_time"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(p["entry_time"], pd.Timestamp) else str(p["entry_time"]),
                    "exit_time": ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, pd.Timestamp) else str(ts),
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

                # V38 Symbol Lockout on Hard Stop
                if reason == "HARD_SL" and cooldown_sessions > 0:
                    symbol_sl_lockout[ticker] = cur_d_idx + cooldown_sessions

                del positions[ticker]

            # Hard stop
            if cur_l <= p["active_sl"]:
                raw_fill = min(cur_h, p["active_sl"]) * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                reason = "TRAIL_SL" if p["active_sl"] >= p["entry_p"] else "HARD_SL"
                finalize_trade(exit_fill, reason)
                continue

            # Time scratch
            if p["bars_held"] >= 4 and mfe_atr < 0.40 and cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                raw_fill = cur_c * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                finalize_trade(exit_fill, "SCRATCH")
                continue

            # EOD
            if t_time >= pd.Timestamp("14:45").time():
                raw_fill = cur_c * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                finalize_trade(exit_fill, "EOD")
                continue

        # 2. Dynamic Equity & Risk Budget
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

        # Freeroll condition for unlocking Slot 2 / 3
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

        dd_mult = 0.45 if dd_pct > 0.16 else (0.70 if dd_pct > 0.10 else (0.85 if dd_pct > 0.06 else 1.0))
        r_budget = min(raw_r * dd_mult, gate_v2_cap)

        t_bullish = bullish_matrix[ts_idx]
        sum_bullish = np.sum(t_bullish)

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
            
            # V38 Symbol Lockout Check
            if ticker in symbol_sl_lockout and cur_d_idx < symbol_sl_lockout[ticker]:
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

            # V38 Consecutive Loss Brake: If consec_losses >= 2, disable secondary triggers (t1b & t3)
            pass_t1a = t_data["t1_a"].iloc[idx]
            if is_severe_bleed_day:
                pass_trigger = t_data["t1_tqs"].iloc[idx]
            elif consec_loss_brake and consec_losses >= 2:
                # Require A+ morning breakout OR prime structural pullback
                pass_t2 = (t_data["t2_pullback"].iloc[idx] and (b_day_now >= -0.0020) and (t2_start_t <= t_time <= t2_end_t))
                pass_trigger = pass_t1a or pass_t2
            else:
                pass_t1b = (t_data["t1_b"].iloc[idx] and (b_day_now >= -0.0030) and (t_time <= t1b_end_t))
                pass_t2  = (t_data["t2_pullback"].iloc[idx] and (b_day_now >= -0.0040) and (t2_start_t <= t_time <= t2_end_t))
                pass_t3  = (t_data["t3_squeeze"].iloc[idx] and (b_day_now >= -0.0020) and (t3_start_t <= t_time <= t3_end_t))
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
        chosen = candidates[:slots_needed]

        for cand in chosen:
            ticker, cur_c, cur_h, cur_l, cur_v, cur_atr, alpha_score = cand

            fill_price = round(cur_c * (1.0 + slippage), 2)
            init_sl = round(fill_price - 0.90 * cur_atr, 2)
            risk_unit = max(fill_price - init_sl, 0.05)

            friction_per_share = 0.002067 * fill_price
            effective_risk_per_share = risk_unit + friction_per_share

            shares_by_risk = int(r_budget // effective_risk_per_share)
            shares_by_margin = int(avail_notional // fill_price)

            final_shares = min(shares_by_risk, shares_by_margin)
            if final_shares < 1:
                final_shares = 1

            positions[ticker] = {
                "symbol": ticker,
                "entry_p": fill_price,
                "entry_time": ts,
                "entry_atr": cur_atr,
                "initial_sl": init_sl,
                "active_sl": init_sl,
                "qty": final_shares,
                "bars_held": 0,
                "high_seen": cur_h
            }
            daily_traded[cur_date].add(ticker)
            avail_notional -= (fill_price * final_shares)
            if avail_notional < 15.0 * 5.0:
                break

    df_tr = pd.DataFrame(trades)
    net = df_tr["net_pnl"].sum() if len(df_tr) > 0 else 0.0
    wr = (df_tr["net_pnl"] > 0).mean() * 100.0 if len(df_tr) > 0 else 0.0
    wins = df_tr[df_tr["net_pnl"] > 0]["net_pnl"].sum() if len(df_tr) > 0 else 0.0
    losses = abs(df_tr[df_tr["net_pnl"] < 0]["net_pnl"].sum()) if len(df_tr) > 0 else 0.0
    pf = (wins / losses) if losses > 0 else 999.0
    print(f"[{fold_label}] -> Trades: {len(df_tr):3d} | WR: {wr:5.1f}% | Net: Rs {net:+8.2f} | PF: {pf:4.2f} | End Wallet: Rs {ek:7.2f}")
    return ek, trades

if __name__ == "__main__":
    with open(ROOT / "data" / "2024-25" / "universe_14_15m.pkl", "rb") as f:
        p2024 = pickle.load(f)
    with open(ROOT / "data" / "2025-26" / "universe_14_15m.pkl", "rb") as f:
        p2025 = pickle.load(f)

    print("=== TESTING V38 DEFENSES ON 2024 H2 (UNSEEN DEFICIT FOLD) ===")
    print("Baseline V37 on 2024 H2:")
    run_v38_simulation(p2024, "2024-07-01", "2024-12-31", "V37 Baseline (CD=0, Brake=False)", universe_filter=V29_30_UNIVERSE, cooldown_sessions=0, consec_loss_brake=False)
    
    for cd in [1, 2, 3]:
        print(f"\nV38 Candidate (Cooldown={cd} days, Brake=True):")
        run_v38_simulation(p2024, "2024-07-01", "2024-12-31", f"V38 (CD={cd}, Brake=True)", universe_filter=V29_30_UNIVERSE, cooldown_sessions=cd, consec_loss_brake=True)

    print("\n=== TESTING V38 DEFENSES ON 2025 H1 (UNSEEN B) ===")
    print("Baseline V37 on 2025 H1:")
    run_v38_simulation(p2025, "2025-01-01", "2025-07-31", "V37 Baseline (CD=0, Brake=False)", universe_filter=V29_30_UNIVERSE, cooldown_sessions=0, consec_loss_brake=False)
    for cd in [1, 2, 3]:
        print(f"\nV38 Candidate (Cooldown={cd} days, Brake=True):")
        run_v38_simulation(p2025, "2025-01-01", "2025-07-31", f"V38 (CD={cd}, Brake=True)", universe_filter=V29_30_UNIVERSE, cooldown_sessions=cd, consec_loss_brake=True)
