import os
import sys
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT / "strategies" / "V37"))
from run_v37 import (
    calc_dhan_charges, calculate_atr, compute_vwap_shifted,
    compute_er10, compute_adx_di, SECTOR_MAP, V29_30_UNIVERSE
)

def run_v38_simulation(
    raw_pool,
    start: str, end: str,
    fold_label: str,
    universe_filter=None,
    max_slots: int = 2,
    symbol_cooldown_days: int = 3,
    consecutive_loss_brake: bool = True
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

        # Triggers
        t1_base = (
            macd_cross_up & (macd > 0) &
            (c > ema20) &
            (rsi7 > 50.0) & (adx >= 18.0) & (pdi > mdi) &
            (er10 >= 0.35)
        )
        t1_a = (t1_base & (rvol >= 1.50) & (c > high_5))
        t1_tqs = (t1_base & (rvol >= 1.25) & (c > high_5))
        t1_b = (t1_base & (rvol >= 1.20) & (c > high_5))

        # Trigger 2 (Sharpened Pullback)
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
            (rsi7 >= 52.0) & (rsi7 <= 75.0)
        )

        # Trigger 3 (Afternoon Squeeze Box)
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

    # V38 Risk Control State
    # Map ticker -> date until which it is locked out
    symbol_sl_lockout = {} 
    consecutive_hard_stops = 0

    unique_dates = sorted(list(set(ts.date() for ts in all_ts)))
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}

    for ts_idx, ts in enumerate(all_ts):
        ts_str = str(ts)
        cur_date = ts.date()
        cur_time = ts.time()
        cur_d_idx = date_to_idx[cur_date]

        if cur_date not in daily_traded:
            daily_traded[cur_date] = set()

        # 1. Manage Positions
        active_tickers = list(positions.keys())
        for ticker in active_tickers:
            p = positions[ticker]
            d = data_map[ticker]
            if ts not in d["c"].index:
                continue

            cur_c = d["c"].loc[ts]
            cur_h = d["h"].loc[ts]
            cur_l = d["l"].loc[ts]
            cur_atr = d["atr14"].loc[ts]
            cur_rsi = d["rsi7"].loc[ts]
            cur_ema9 = d["ema9"].loc[ts]
            cur_ema20 = d["ema20"].loc[ts]
            cur_adx = d["adx"].loc[ts]
            cur_rvol = d["rvol"].loc[ts]

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

            # Asymmetric Elastic Runner Channel
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
                nonlocal ek, hwm, consecutive_hard_stops
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

                if reason == "HARD_SL":
                    consecutive_hard_stops += 1
                    # Lockout symbol for N future trading sessions
                    symbol_sl_lockout[ticker] = cur_d_idx + symbol_cooldown_days
                else:
                    if net > 0:
                        consecutive_hard_stops = 0

                del positions[ticker]

            # Hard stop
            if cur_l <= p["active_sl"]:
                raw_fill = min(cur_h, p["active_sl"]) * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                reason = "TRAIL_SL" if p["active_sl"] >= p["entry_p"] else "HARD_SL"
                finalize_trade(exit_fill, reason)
                continue

            # Time scratch
            if p["bars_held"] >= 5 and mfe_atr < 0.35 and cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < d["vwap_lag"].loc[ts]):
                raw_fill = cur_c * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                finalize_trade(exit_fill, "SCRATCH")
                continue

            # EOD Square-off at 15:00
            if cur_time >= pd.Timestamp("15:00").time():
                raw_fill = cur_c * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                finalize_trade(exit_fill, "EOD")
                continue

        # 2. Entries
        if ts_str < start or ts_str > end:
            continue
        if cur_time < pd.Timestamp("09:30").time() or cur_time > pd.Timestamp("14:30").time():
            continue
        if len(positions) >= max_slots:
            continue
        if ts in last_exit_times:
            continue

        b_day_val = bench_day.asof(ts)
        is_severe_bleed_day = (b_day_val < -0.006)
        c300_val = c300_breadth.asof(ts)

        candidates = []
        for ticker in ticker_list:
            if ticker in positions or ticker in daily_traded[cur_date]:
                continue
            
            # V38 Symbol Lockout Check
            if ticker in symbol_sl_lockout and cur_d_idx < symbol_sl_lockout[ticker]:
                continue

            d = data_map[ticker]
            if ts not in d["c"].index:
                continue

            univ_ok = d["univ_cond"].loc[ts]
            if not univ_ok:
                continue

            g1 = d["g1_cond"].loc[ts]
            g2 = d["g2_cond"].loc[ts]
            if not (g1 and g2):
                continue

            # Sector Breadth
            idx_i = ticker_to_idx[ticker]
            sec_name = None
            for sc, sm_list in SECTOR_MAP.items():
                if ticker in sm_list:
                    sec_name = sc
                    break

            if sec_name and ticker in sec_p_indices.get(sec_name, {}):
                p_idxs = sec_p_indices[sec_name][ticker]
                if p_idxs:
                    bm_row = bullish_matrix[ts_idx]
                    sec_breadth = np.mean(bm_row[p_idxs])
                    if sec_breadth < 0.33:
                        continue
            elif ticker in unseen_peers:
                p_idxs = unseen_peers[ticker]
                if p_idxs:
                    bm_row = bullish_matrix[ts_idx]
                    sec_breadth = np.mean(bm_row[p_idxs])
                    if sec_breadth < 0.33:
                        continue

            t_val = cur_time
            is_morning = (t_val <= pd.Timestamp("11:30").time())
            is_afternoon = (t_val >= pd.Timestamp("13:15").time())

            t1a = d["t1_a"].loc[ts]
            t1tqs = d["t1_tqs"].loc[ts]
            t1b = d["t1_b"].loc[ts]
            t2 = d["t2_pullback"].loc[ts]
            t3 = d["t3_squeeze"].loc[ts]

            # V38 Consecutive Loss Circuit Breaker
            # If 2 consecutive hard stops occurred, only allow A+ morning breakouts (t1_a)
            if consecutive_loss_brake and consecutive_hard_stops >= 2:
                if not (is_morning and t1a):
                    continue

            # Normal regime filtering
            if is_severe_bleed_day:
                trigger_ok = (is_morning and t1a)
            elif c300_val < 0.40:
                trigger_ok = (is_morning and (t1a or t1tqs)) or (is_afternoon and t2)
            else:
                trigger_ok = (
                    (is_morning and (t1a or t1tqs or t1b)) or
                    (is_afternoon and (t2 or t3))
                )

            if trigger_ok:
                candidates.append((ticker, d["excess_1h"].loc[ts]))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[1], reverse=True)
        available_slots = max_slots - len(positions)

        for ticker, _ in candidates[:available_slots]:
            d = data_map[ticker]
            cur_c = d["c"].loc[ts]
            cur_atr = d["atr14"].loc[ts]
            cur_rvol = d["rvol"].loc[ts]
            cur_er = d["er10"].loc[ts]

            # Dynamic Kelly Fraction
            streak_factor = 1.0 + (alpha_streak * min(3, consecutive_hard_stops * -1 if consecutive_hard_stops > 0 else 1)) # simplified
            dd_current = (hwm - ek) / max(hwm, initial_equity)
            throttle = 0.50 if dd_current >= 0.12 else (0.75 if dd_current >= 0.08 else 1.0)

            # Volatility-Targeted Sizing
            risk_budget = ek * 0.02 * throttle
            fill_price = round(cur_c * (1.0 + slippage), 2)
            sl_price = round(fill_price - 0.90 * cur_atr, 2)
            risk_per_share = max(0.10, fill_price - sl_price + (0.002067 * fill_price))
            qty = int(max(1, risk_budget // risk_per_share))

            # Max capital check per trade (5x leverage on intraday cash)
            max_shares = int((ek * 5.0) // fill_price)
            qty = min(qty, max_shares)

            positions[ticker] = {
                "entry_p": fill_price,
                "initial_sl": sl_price,
                "active_sl": sl_price,
                "qty": qty,
                "entry_atr": cur_atr,
                "entry_time": ts,
                "bars_held": 0,
                "high_seen": fill_price
            }
            daily_traded[cur_date].add(ticker)

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

    print("--- TESTING DIFFERENT LOCKOUT WINDOWS ON 2024 H2 ---")
    for cd in [1, 2, 3, 4]:
        print(f"\nTesting symbol_cooldown_days = {cd} (consecutive_loss_brake = True):")
        run_v38_simulation(p2024, "2024-07-01", "2024-12-31", f"2024 H2 (CD={cd})", universe_filter=V29_30_UNIVERSE, symbol_cooldown_days=cd, consecutive_loss_brake=True)

    print("\n--- TESTING ON 2025 H1 (UNSEEN B) ---")
    for cd in [1, 2, 3]:
        print(f"\nTesting symbol_cooldown_days = {cd} on 2025 H1:")
        run_v38_simulation(p2025, "2025-01-01", "2025-07-31", f"2025 H1 (CD={cd})", universe_filter=V29_30_UNIVERSE, symbol_cooldown_days=cd, consecutive_loss_brake=True)
