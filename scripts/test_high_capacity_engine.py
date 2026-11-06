import sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from framework import calc_dhan_charges, calculate_atr
from strategies.V42.run_v42 import compute_vwap_shifted, compute_er10, compute_adx_di, SECTOR_MAP, SYM_TO_SECTOR, V29_30_UNIVERSE

def run_high_capacity_simulation(
    raw_pool, start, end, fold_label, universe_filter=None,
    base_risk=25.0, max_concurrent_positions=2,
    target_r_multiple=2.2, min_reward_risk=1.5
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

        high_20 = h.rolling(20).max().shift(1)
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

        # SETUP 1: V42 Base Breakout
        t1_base = (
            macd_cross_up & (macd > 0) &
            (c > ema20) &
            (rsi7 > 50.0) & (adx >= 18.0) & (pdi > mdi) &
            (er10 >= 0.35) &
            (rvol >= 1.30)
        )
        t1_tqs = (t1_base & (rvol >= 1.35) & (c > high_5))

        # SETUP 2: Pullback to VWAP / EMA
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

        # SETUP 3 (From trader-scanner): High-Tight Consolidation Breakout (Bull Flag)
        # 10 bars expansion, last 3 bars tight range < 1.5% ATR, now expanding with volume spike
        flag_range = (h.rolling(3).max() - l.rolling(3).min()) / (c + 1e-9)
        t3_bull_flag = (
            (c > ema20) & (ema20 > ema50) &
            (c > vwap_lag) &
            (c > high_20) & # 20-bar breakout
            (flag_range.shift(1) <= 0.02) & # tight compression
            (v >= 1.40 * v_sma20) & # volume surge
            (rsi7 >= 55.0) & (rsi7 <= 80.0)
        )

        data_map[clean] = {
            "df": df_full, "c": c, "h": h, "l": l, "v": v, "o": o, "atr14": atr14,
            "ema9": ema9, "ema20": ema20, "rsi7": rsi7, "adx": adx, "vwap_lag": vwap_lag,
            "rvol": rvol, "er10": er10,
            "univ_cond": univ_cond, "t1_base": t1_base, "t1_tqs": t1_tqs, "t2_pullback": t2_pullback,
            "t3_bull_flag": t3_bull_flag,
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

    initial_equity = 500.0
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

    for ts_idx, ts in enumerate(all_ts):
        if ts.weekday() >= 5:
            continue
        cur_date = ts.date()
        if cur_date not in daily_traded:
            daily_traded[cur_date] = set()
            daily_losses[cur_date] = 0
        t_time = ts.time()

        b_c300 = c300_breadth.get(ts, 0.50)
        b_day_now = bench_day.get(ts, 0.0)
        is_bear_env = (b_c300 < 0.40) or (b_day_now < -0.0060)

        # 1. Manage Positions
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

            p["bars_held"] += 1
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0

            # Dynamic R-Multiple & Ratchet from trader-exit
            # Target = 2.2R (where 1R = entry_atr)
            target_price = p["entry_p"] + (target_r_multiple * p["entry_atr"])
            
            # Weapon 1: Breakeven lock at +0.45 ATR
            if mfe_atr >= 2.0:
                stop_ladder = p["entry_p"] + (1.20 * p["entry_atr"])
            elif mfe_atr >= 1.2:
                stop_ladder = p["entry_p"] + (0.60 * p["entry_atr"])
            elif mfe_atr >= 0.45:
                stop_ladder = p["entry_p"] + (0.05 * p["entry_atr"])
            else:
                stop_ladder = p["initial_sl"]

            p["active_sl"] = max(p["active_sl"], round(stop_ladder, 2))

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
                    "net_pnl": net,
                    "exit_reason": exit_reason
                })
                last_exit_times.add(ts)
                if net < 0:
                    daily_losses[cur_date] += 1
                if exit_reason == "STOP" and exit_fill <= p["entry_p"]:
                    symbol_lockouts[ticker] = cur_date + pd.Timedelta(days=2)
                del positions[ticker]

            # Check Target Hit
            if cur_h >= target_price:
                exit_fill = round(min(cur_h, target_price * (1.0 - slippage)), 2)
                finalize_trade(exit_fill, "TARGET_2.2R")
                continue

            # Check SL Hit
            if cur_l <= p["active_sl"]:
                raw_fill = min(cur_h, p["active_sl"]) * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                finalize_trade(exit_fill, "STOP")
                continue

            # Scratch if stagnant after 5 bars
            if p["bars_held"] >= 5 and mfe_atr < 0.40 and cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                raw_fill = cur_c * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                finalize_trade(exit_fill, "SCRATCH")
                continue

            if t_time >= pd.Timestamp("15:15").time():
                raw_fill = cur_c * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                finalize_trade(exit_fill, "EOD")
                continue

        # 2. Capital & Concurrency Sizing (Multi-Slot Support from trader-budget)
        ek = initial_equity + sum(t["net_pnl"] for t in trades)
        hwm = max(hwm, ek)
        dd_pct = (hwm - ek) / max(hwm, initial_equity)

        # Dynamic Slots: Allow 2 concurrent positions if equity > Rs 1,000; else 1 slot
        allowed_slots = 2 if ek >= 1000.0 else 1
        if len(positions) >= allowed_slots or ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:30").time():
            continue

        if pd.Timestamp("11:30").time() <= t_time <= pd.Timestamp("12:30").time():
            continue
        if pd.Timestamp("10:00").time() <= t_time <= pd.Timestamp("10:05").time():
            continue

        # Daily 2-strike breaker
        if daily_losses.get(cur_date, 0) >= 2:
            continue

        avail_margin = (0.95 * ek) - sum((p["entry_p"] * p["qty"]) / 5.0 for p in positions.values())
        if avail_margin < 10.0:
            continue
        avail_notional = avail_margin * 5.0

        # Sizing per slot
        slot_margin = avail_notional / max(1, (allowed_slots - len(positions)))
        raw_r = base_risk + (0.085 * max(0.0, ek - 500.0))
        target_risk = min(raw_r, min(0.090 * ek, 258.0)) / max(1, allowed_slots - len(positions))

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

            is_t1 = t_data["t1_tqs"].iloc[idx] or t_data["t1_base"].iloc[idx]
            is_t2 = t_data["t2_pullback"].iloc[idx]
            is_t3 = t_data["t3_bull_flag"].iloc[idx]
            if not (is_t1 or is_t2 or is_t3):
                continue

            bar_c = t_data["c"].iloc[idx]
            bar_h = t_data["h"].iloc[idx]
            bar_l = t_data["l"].iloc[idx]
            bar_atr = t_data["atr14"].iloc[idx]
            if bar_atr <= 0:
                continue

            tqs_bonus = 2.5 if is_t3 else (2.0 if t_data["t1_tqs"].iloc[idx] else (1.0 if is_t1 else 0.5))
            excess = t_data["excess_1h"].iloc[idx]
            base_score = (excess / (bar_atr / bar_c)) * tqs_bonus

            candidates.append((
                ticker, bar_c, bar_h, bar_l, t_data["v"].iloc[idx],
                max(bar_atr, 0.05), base_score
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
        qty_m = int((slot_margin * 0.90) / entry_p)
        qty_v = int(cur_v * 0.05)
        qty = min(qty_r, qty_m, qty_v)

        if qty < 1:
            continue

        positions[ticker] = {
            "entry_t": ts,
            "entry_p": entry_p,
            "initial_sl": active_sl,
            "active_sl": active_sl,
            "entry_atr": entry_atr,
            "qty": qty,
            "high_seen": cur_h,
            "bars_held": 0
        }
        daily_traded[cur_date].add(ticker)

    final_cap = initial_equity + sum(t["net_pnl"] for t in trades)
    wr = (sum(1 for t in trades if t["net_pnl"] > 0) / len(trades) * 100.0) if trades else 0.0
    net = sum(t["net_pnl"] for t in trades)
    wins = [t["net_pnl"] for t in trades if t["net_pnl"] > 0]
    losses = [t["net_pnl"] for t in trades if t["net_pnl"] < 0]
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    pf = round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0 else 999.0

    equity_curve = [initial_equity]
    for t in trades:
        equity_curve.append(equity_curve[-1] + t["net_pnl"])
    eq_series = pd.Series(equity_curve)
    peaks = eq_series.cummax()
    dds = (peaks - eq_series) / peaks
    max_dd_pct = dds.max() * 100.0

    print(f"[{fold_label:<10}] Trades: {len(trades):3d} | WR: {wr:5.1f}% | AvgWin: Rs {avg_win:+6.2f} | AvgLoss: Rs {avg_loss:+6.2f} | Net: Rs {net:+8.2f} | PF: {pf:5.2f} | MaxDD: {max_dd_pct:4.1f}%")
    return final_cap, trades, wr, max_dd_pct, pf, avg_win, avg_loss

def main():
    p_2026 = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
    p_2020 = pickle.load(open(ROOT / 'data/2020_FULL/universe_30_15m.pkl', 'rb'))
    p_2022 = pickle.load(open(ROOT / 'data/2022-23/universe_30_15m.pkl', 'rb'))

    print("=" * 110)
    print("HIGH CAPACITY DAY TRADER ENGINE TEST (2 SLOTS + BULL FLAG + 2.2R ASYMMETRIC TARGET)")
    print("=" * 110)
    run_high_capacity_simulation(p_2026, "2026-01-01", "2026-10-29", "2026_TEST", universe_filter=V29_30_UNIVERSE)
    run_high_capacity_simulation(p_2020, "2020-01-01", "2020-12-31", "2020_TEST", universe_filter=None)
    run_high_capacity_simulation(p_2022, "2022-01-01", "2022-12-31", "2022_TEST", universe_filter=V29_30_UNIVERSE)
    print("=" * 110)

if __name__ == "__main__":
    main()
