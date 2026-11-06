import sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from framework import calc_dhan_charges, calculate_atr
from scripts.test_v42_prototype import compute_vwap_shifted, compute_er10, compute_adx_di, SECTOR_MAP, SYM_TO_SECTOR, V29_30_UNIVERSE

def simulate_v42_evolved(
    raw_pool, start, end, universe_filter=None,
    base_risk=14.0,
    adaptive_be=True,       # Bull: 0.75 ATR, Bear/Chop: 0.45 ATR
    smart_strike=True       # Scratches (loss < 0.35R) don't count towards daily strike limit
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

        if clean not in V29_30_UNIVERSE:
            phi_friction = (0.05 + 0.00206 * c) / (atr14 + 1e-9)
            unseen_ok = (c >= 30.0) & (atr14 >= 0.25) & (phi_friction <= 0.45)
            univ_cond = univ_cond & unseen_ok

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
    daily_strikes = {}
    symbol_lockouts = {}
    slippage = 0.0005
    hard_sl_mult = 0.90

    t2_start_t = pd.Timestamp("09:45").time()
    t2_end_t = pd.Timestamp("13:15").time()

    for ts_idx, ts in enumerate(all_ts):
        if ts.weekday() >= 5:
            continue
        cur_date = ts.date()
        if cur_date not in daily_traded:
            daily_traded[cur_date] = set()
        if cur_date not in daily_strikes:
            daily_strikes[cur_date] = 0
        t_time = ts.time()

        b_c300 = c300_breadth.get(ts, 0.50)
        b_day_now = bench_day.get(ts, 0.0)

        is_bull_expansion = (b_c300 >= 0.52) and (b_day_now >= 0.0)
        is_bear_env = (b_c300 < 0.40) or (b_day_now < -0.0060)
        is_severe_bleed_day = (b_day_now < -0.0120)

        # Adaptive BE threshold
        if adaptive_be:
            current_be_mfe = 0.75 if is_bull_expansion else 0.45
        else:
            current_be_mfe = 0.45

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

            p["bars_held"] += 1
            if cur_h > p["high_seen"]:
                p["high_seen"] = cur_h

            mfe = p["high_seen"] - p["entry_p"]
            mfe_atr = mfe / p["entry_atr"]

            # Weapon 1: Breakeven lock
            if mfe_atr >= current_be_mfe:
                be_level = round(p["entry_p"] + (0.05 * p["entry_atr"]), 2)
                if be_level > p["active_sl"]:
                    p["active_sl"] = be_level

            # Stage 2 (+0.85 ATR)
            if mfe_atr >= 0.85:
                lock_p = round(p["entry_p"] + (0.40 * p["entry_atr"]), 2)
                if lock_p > p["active_sl"]:
                    p["active_sl"] = lock_p

            # Stage 3 (+1.25 ATR)
            if mfe_atr >= 1.25:
                buf = 0.38 if is_bull_expansion else 0.28
                cand_sl = round(p["prev_l"] - (buf * p["entry_atr"]), 2)
                if cand_sl > p["active_sl"]:
                    p["active_sl"] = cand_sl

            # Stage 4 (+2.25 ATR)
            if mfe_atr >= 2.25:
                min_l = round(p["entry_p"] + (1.20 * p["entry_atr"]), 2)
                cand_sl = max(min_l, round(p["high_seen"] - (0.35 * p["entry_atr"]), 2))
                if cand_sl > p["active_sl"]:
                    p["active_sl"] = cand_sl

            # Stage 5 Climax (+3.50 ATR)
            if mfe_atr >= 3.50:
                is_climax = (cur_c < cur_vwap) or (cur_rsi < 48.0)
                if is_climax:
                    p["active_sl"] = max(p["active_sl"], round(p["high_seen"] - (0.10 * p["entry_atr"]), 2))
                else:
                    cand_sl = max(round(p["entry_p"] + 2.00 * p["entry_atr"], 2), round(p["high_seen"] - 0.08 * p["entry_atr"], 2))
                    p["active_sl"] = max(p["active_sl"], cand_sl)

            exit_flag = False
            exit_p = 0.0
            reason = ""

            if cur_l <= p["active_sl"]:
                exit_flag = True
                exit_p = round(max(cur_l, min(cur_h, p["active_sl"] * (1.0 - slippage))), 2)
                reason = "STOP"
            elif t_time >= pd.Timestamp("15:10").time():
                exit_flag = True
                exit_p = round(cur_c * (1.0 - slippage), 2)
                reason = "EOD"

            if exit_flag:
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chgs = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chgs, 2)
                ek += net
                if ek > hwm:
                    hwm = ek

                # Smart Strike check: is it a real loss or just a scratch?
                planned_loss = p["qty"] * (p["entry_p"] - p["initial_sl"])
                is_real_loss = (net < 0) and (not smart_strike or abs(net) >= 0.35 * planned_loss)

                if is_real_loss:
                    daily_strikes[cur_date] += 1
                    symbol_lockouts[ticker] = cur_date + pd.Timedelta(days=2)

                trades.append({
                    "symbol": ticker, "direction": "LONG",
                    "entry_time": p["entry_t"], "exit_time": ts,
                    "entry_price": p["entry_p"], "exit_price": exit_p,
                    "shares": p["qty"], "sl": p["active_sl"],
                    "gross_pnl": gross, "charges": chgs, "net_pnl": net,
                    "exit_reason": reason
                })
                del positions[ticker]
            else:
                p["prev_l"] = cur_l

        # 2. Check Daily Strike Circuit Breaker
        if daily_strikes[cur_date] >= 2:
            continue

        if len(positions) >= 1:
            continue

        if not (pd.Timestamp("09:30").time() <= t_time <= pd.Timestamp("14:30").time()):
            continue
        if pd.Timestamp("10:00").time() <= t_time < pd.Timestamp("10:05").time():
            continue
        if pd.Timestamp("11:30").time() <= t_time < pd.Timestamp("12:30").time():
            continue

        margin_cap = ek * 0.90
        avail_notional = margin_cap * 5.0
        if avail_notional < 100.0:
            continue

        dd_pct = (hwm - ek) / (hwm + 1e-9)

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

        if ek <= 2500.0:
            raw_r = base_risk + (alpha * max(0.0, ek - 500.0))
        else:
            raw_r = base_risk + (alpha * 2000.0) + (beta * (ek - 2500.0))

        gate_v2_cap = min(0.090 * ek, 258.0)
        if dd_pct > 0.15:
            gate_v2_cap *= 0.50
        target_risk = min(raw_r, gate_v2_cap)

        t_bullish = bullish_matrix[ts_idx, :]
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
            bar_h, bar_l, bar_c = t_data["h"].iloc[idx], t_data["l"].iloc[idx], t_data["c"].iloc[idx]
            z_spread = (bar_c - bar_l) / (bar_h - bar_l + 1e-6)
            base_score = (0.45 * z_mom) + (0.25 * z_vol) + (0.15 * z_er) + (0.15 * z_spread)
            final_alpha = base_score * (1.0 + 0.35 * k_cluster.get(ticker, 0.0))

            candidates.append((
                ticker, bar_c, bar_h, bar_l, t_data["v"].iloc[idx],
                max(t_data["atr14"].iloc[idx], 0.05), final_alpha
            ))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[6], reverse=True)
        best = candidates[0]
        ticker, cur_c, cur_h, cur_l, cur_v, entry_atr, _ = best

        raw_entry = cur_c * (1.0 + slippage)
        entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
        active_sl = round(entry_p - (hard_sl_mult * entry_atr), 2)
        sl_dist = max(entry_p - active_sl, 0.05)
        friction = 0.002067 * entry_p
        denom = sl_dist + friction

        qty_r = int(target_risk / denom)
        qty_m = int((avail_notional * 0.90) / entry_p)
        qty_v = int(cur_v * 0.05)
        qty = min(qty_r, qty_m, qty_v)

        if qty < 1:
            continue

        positions[ticker] = {
            "entry_t": ts, "entry_p": entry_p, "initial_sl": active_sl,
            "active_sl": active_sl, "entry_atr": entry_atr, "qty": qty,
            "high_seen": cur_h, "prev_l": cur_l, "bars_held": 0
        }
        daily_traded[cur_date].add(ticker)

    final_cap = initial_equity + sum(t["net_pnl"] for t in trades)
    wr = (sum(1 for t in trades if t["net_pnl"] > 0) / len(trades) * 100.0) if trades else 0.0
    net = sum(t["net_pnl"] for t in trades)

    eq_curve = [initial_equity]
    c_eq = initial_equity
    for t in trades:
        c_eq += t["net_pnl"]
        eq_curve.append(c_eq)
    eq_s = pd.Series(eq_curve)
    pk = eq_s.cummax()
    max_dd = ((pk - eq_s) / pk).max() * 100.0

    return final_cap, net, wr, max_dd, len(trades)

# Test across 2026 Bull Horizon
p_2026 = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
p_hb = pickle.load(open(ROOT / 'data/unseen_high_beta_10_15m_1y.pkl', 'rb'))
p_u10 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))
pool_2026 = {**p_2026, **p_hb, **p_u10}

print("="*95)
print("TESTING EVOLVED V42 HYPOTHESES (ADAPTIVE BE + SMART STRIKES)")
print("="*95)

# Baseline V42
c_base, net_base, wr_base, dd_base, tr_base = simulate_v42_evolved(
    pool_2026, "2026-01-01", "2026-12-31", V29_30_UNIVERSE,
    base_risk=14.0, adaptive_be=False, smart_strike=False
)
print(f"1. V42 Baseline (2026):     Trades={tr_base:2d} | WR={wr_base:4.1f}% | Net=Rs {net_base:+8.2f} | Max DD={dd_base:4.1f}%")

# Evolved V42 (Base Risk 14.0)
c_ev14, net_ev14, wr_ev14, dd_ev14, tr_ev14 = simulate_v42_evolved(
    pool_2026, "2026-01-01", "2026-12-31", V29_30_UNIVERSE,
    base_risk=14.0, adaptive_be=True, smart_strike=True
)
print(f"2. V42 Evolved Base 14 (2026): Trades={tr_ev14:2d} | WR={wr_ev14:4.1f}% | Net=Rs {net_ev14:+8.2f} | Max DD={dd_ev14:4.1f}%")

# Evolved V42 (Base Risk 25.0)
c_ev25, net_ev25, wr_ev25, dd_ev25, tr_ev25 = simulate_v42_evolved(
    pool_2026, "2026-01-01", "2026-12-31", V29_30_UNIVERSE,
    base_risk=25.0, adaptive_be=True, smart_strike=True
)
print(f"3. V42 Evolved Base 25 (2026): Trades={tr_ev25:2d} | WR={wr_ev25:4.1f}% | Net=Rs {net_ev25:+8.2f} | Max DD={dd_ev25:4.1f}%")
