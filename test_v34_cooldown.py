import pickle, sys, importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
p_v34 = ROOT / "strategies" / "V34.3" / "run_v34_3.py"
spec34 = importlib.util.spec_from_file_location("v34_3", p_v34)
mod_v34 = importlib.util.module_from_spec(spec34)
spec34.loader.exec_module(mod_v34)

calc_dhan_charges = mod_v34.calc_dhan_charges
calculate_atr = mod_v34.calculate_atr
compute_vwap_shifted = mod_v34.compute_vwap_shifted
compute_er10 = mod_v34.compute_er10
compute_adx_di = mod_v34.compute_adx_di
SECTOR_MAP = mod_v34.SECTOR_MAP
V29_30_UNIVERSE = mod_v34.V29_30_UNIVERSE

def run_v34_with_cooldown(
    raw_pool,
    start: str, end: str,
    fold_label: str,
    universe_filter=None,
    cooldown_sessions: int = 2
):
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
            (er10 >= 0.35) &
            (rvol >= 1.20)
        )
        t1_tqs = (t1_base & (rvol >= 1.25) & (c > high_5))

        sup_prev = np.maximum(ema9.shift(1), vwap_lag.shift(1)) + (0.15 * atr14)
        t2_pullback = (
            (macd > sig) & (macd > 0) & (pdi > mdi) &
            (ema9 > ema20) & (ema20 > ema50) &
            (c > ema9) &
            (l.shift(1) <= sup_prev) &
            (v.shift(1) < v.shift(2)) & (v >= 1.15 * v.shift(1)) & (rvol >= 1.20) &
            (c > h.shift(1)) & (c.shift(1) <= h.shift(2)) & (c > o) &
            (rsi7 >= 52.0) & (rsi7 <= 74.0) &
            (((c - ema20) / (atr14 + 1e-9)) <= 2.10)
        )

        data_map[clean] = {
            "df": df_full, "c": c, "h": h, "l": l, "v": v, "o": o, "atr14": atr14,
            "ema9": ema9, "rsi7": rsi7, "vwap_lag": vwap_lag,
            "univ_cond": univ_cond,
            "t1_base": t1_base, "t1_tqs": t1_tqs,
            "t2_pullback": t2_pullback,
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
    for i, s in enumerate(ticker_list):
        s_c = data_map[s]["c"].reindex(all_ts).ffill().to_numpy()
        s_vwap = data_map[s]["vwap_lag"].reindex(all_ts).ffill().to_numpy()
        s_ex = data_map[s]["excess_1h"].reindex(all_ts).fillna(0.0).to_numpy()
        bullish_matrix[:, i] = (s_c > s_vwap) & (s_ex > 0)

    sec_p_indices = {}
    for sec, syms in SECTOR_MAP.items():
        sec_p_indices[sec] = {}
        for s in syms:
            if s in ticker_to_idx:
                sec_p_indices[sec][s] = [ticker_to_idx[p] for p in syms if p != s and p in ticker_to_idx]

    initial_equity = 500.0
    ek = initial_equity
    hwm = ek
    positions = {}
    trades = []
    daily_traded = {}
    last_exit_times = set()
    slippage = 0.0005
    hard_sl_mult = 0.90

    alpha_streak = 0.082
    beta_streak = 0.035

    symbol_loss_lockout = {}
    unique_dates = sorted(list(set(ts.date() for ts in all_ts)))
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}

    for ts_idx, ts in enumerate(all_ts):
        ts_str = str(ts)
        cur_date = ts.date()
        cur_d_idx = date_to_idx[cur_date]
        t_time = ts.time()

        if cur_date not in daily_traded:
            daily_traded[cur_date] = set()

        # 1. Active Positions
        active_tickers = list(positions.keys())
        for ticker in active_tickers:
            p = positions[ticker]
            t_data = data_map[ticker]
            if ts not in t_data["c"].index:
                continue

            cur_c = t_data["c"].loc[ts]
            cur_h = t_data["h"].loc[ts]
            cur_l = t_data["l"].loc[ts]
            cur_rsi = t_data["rsi7"].loc[ts]
            cur_ema9 = t_data["ema9"].loc[ts]
            cur_vwap = t_data["vwap_lag"].loc[ts]

            p["bars_held"] += 1
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0

            cur_v = t_data["v"].loc[ts]
            cur_rvol = t_data["rvol"].loc[ts]
            shooting_star = (cur_rvol >= 2.60) and (((cur_h - cur_c) / (cur_h - cur_l + 1e-6)) >= 0.45)
            extreme_rsi = (cur_rsi >= 86.0)
            break_ema9 = (cur_c < p.get("prev_l", cur_l)) and (cur_c < cur_ema9)
            circuit_prox = (mfe_atr >= 8.00)
            psi_climax = shooting_star or extreme_rsi or break_ema9 or circuit_prox

            if mfe_atr >= 3.00:
                if psi_climax:
                    stop_ladder = p["high_seen"] - (0.08 * p["entry_atr"])
                else:
                    prev_l = p.get("prev_l", cur_l)
                    stop_ladder = max(p["high_seen"] - (0.26 * p["entry_atr"]), prev_l - (0.08 * p["entry_atr"]))
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
                    "exit_reason": exit_reason
                })
                last_exit_times.add(ts)

                # Cooldown Lockout on negative PnL / Stop
                if net < 0 and cooldown_sessions > 0:
                    symbol_loss_lockout[ticker] = cur_d_idx + cooldown_sessions

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

        # 2. Risk Sizing & Entries
        ek = initial_equity + sum(t["net_pnl"] for t in trades)
        hwm = max(hwm, ek)
        dd_pct = (hwm - ek) / max(hwm, initial_equity)

        k_open = len(positions)
        if k_open >= 3 or ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
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

        dd_mult = 0.45 if dd_pct > 0.16 else (0.70 if dd_pct > 0.10 else (0.85 if dd_pct > 0.06 else 1.0))
        r_budget = min(raw_r * dd_mult, gate_v2_cap)

        t_bullish = bullish_matrix[ts_idx]
        sum_bullish = np.sum(t_bullish)

        k_cluster = {}
        for clean in data_map.keys():
            sec = mod_v34.SYM_TO_SECTOR.get(clean)
            tick_i = ticker_to_idx[clean]
            if sec and clean in sec_p_indices.get(sec, {}):
                p_idxs = sec_p_indices[sec][clean]
                k_cluster[clean] = float(np.mean(t_bullish[p_idxs])) if p_idxs else 0.0
            else:
                k_cluster[clean] = float(sum_bullish - t_bullish[tick_i]) / max(1, N_stocks - 1)

        candidates = []
        for ticker in data_map.keys():
            if ticker in positions or ticker in daily_traded[cur_date]:
                continue
            if ticker in symbol_loss_lockout and cur_d_idx < symbol_loss_lockout[ticker]:
                continue

            t_data = data_map[ticker]
            if ts not in t_data["c"].index:
                continue

            univ_ok = t_data["univ_cond"].loc[ts]
            g1 = t_data["g1_cond"].loc[ts]
            g2 = t_data["g2_cond"].loc[ts]
            if not (univ_ok and g1 and g2):
                continue

            k_cl = k_cluster.get(ticker, 0.0)
            if is_bear_env:
                decoupling = ((t_data["c"].loc[ts] - t_data["day_open"].loc[ts]) / (t_data["day_open"].loc[ts] + 1e-9)) - b_day_now
                auth_pass = (k_cl >= 0.25) or (decoupling >= 0.0120 and t_data["rvol"].loc[ts] >= 1.50)
                if not auth_pass:
                    continue

            if is_severe_bleed_day:
                pass_trigger = t_data["t1_tqs"].loc[ts]
            else:
                pass_t1 = t_data["t1_base"].loc[ts]
                pass_t2 = t_data["t2_pullback"].loc[ts]
                pass_trigger = pass_t1 or pass_t2

            if not pass_trigger:
                continue

            z_mom = t_data["excess_1h"].loc[ts]
            z_vol = min(3.0, t_data["rvol"].loc[ts]) / 3.0
            z_er = min(1.0, t_data["er10"].loc[ts])
            bar_h, bar_l, bar_c = t_data["h"].loc[ts], t_data["l"].loc[ts], t_data["c"].loc[ts]
            z_spread = (bar_c - bar_l) / (bar_h - bar_l + 1e-6)
            base_score = (0.45 * z_mom) + (0.25 * z_vol) + (0.15 * z_er) + (0.15 * z_spread)
            final_alpha = base_score * (1.0 + 0.35 * k_cluster.get(ticker, 0.0))

            candidates.append((
                ticker,
                bar_c,
                bar_h,
                bar_l,
                t_data["v"].loc[ts],
                max(t_data["atr14"].loc[ts], 0.05),
                final_alpha
            ))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[6], reverse=True)
        slots_needed = 3 - len(positions)
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
                "entry_t": ts,
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
    print(f"[{fold_label}] -> Trades: {len(df_tr):3d} | WR: {wr:5.1f}% | Net: Rs {net:+8.2f} | End Wallet: Rs {ek:7.2f}")
    return ek, trades

if __name__ == "__main__":
    with open(ROOT / "data" / "2024-25" / "universe_14_15m.pkl", "rb") as f:
        p2024 = pickle.load(f)
    with open(ROOT / "data" / "2025-26" / "universe_14_15m.pkl", "rb") as f:
        p2025 = pickle.load(f)
    with open(ROOT / "data" / "2022-23" / "universe_30_15m.pkl", "rb") as f:
        p2022 = pickle.load(f)

    print("--- TESTING COOLDOWN ON V34 ARCHITECTURE ---")
    for cd in [0, 1, 2, 3]:
        print(f"\nCooldown = {cd} Sessions:")
        run_v34_with_cooldown(p2024, "2024-07-01", "2024-12-31", f"2024 H2 (CD={cd})", universe_filter=V29_30_UNIVERSE, cooldown_sessions=cd)
        run_v34_with_cooldown(p2025, "2025-01-01", "2025-07-31", f"2025 H1 (CD={cd})", universe_filter=V29_30_UNIVERSE, cooldown_sessions=cd)
        run_v34_with_cooldown(p2022, "2022-01-01", "2022-12-31", f"2022 Bear (CD={cd})", universe_filter=V29_30_UNIVERSE, cooldown_sessions=cd)
