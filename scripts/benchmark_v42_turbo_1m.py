import sys, os, pickle
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from framework import calc_dhan_charges, calculate_atr
from strategies.V42.run_v42 import (
    compute_vwap_shifted, compute_er10, compute_adx_di,
    SECTOR_MAP, SYM_TO_SECTOR, V29_30_UNIVERSE
)

def run_v42_turbo_1m_engine(raw_pool, mode="raw_1m", base_risk=25.0, initial_equity=500.0, slippage=0.0005):
    """
    Simulates V42 Turbo on 1-minute intraday data.
    mode: 'raw_1m' (literal 1m bar indicators) or 'mtf_15m_1m' (15m MTF signals with 1m execution)
    """
    ticker_map = {}
    for k in raw_pool.keys():
        clean = k.replace(".NS", "")
        if clean not in ticker_map or not k.endswith(".NS"):
            ticker_map[clean] = k

    data_map = {}
    for clean, raw_k in ticker_map.items():
        df_full = raw_pool[raw_k].copy()
        if len(df_full) < 300:
            continue
            
        c = df_full["close"]
        h = df_full["high"]
        l = df_full["low"]
        v = df_full["volume"]
        o = df_full["open"]

        if mode == "raw_1m":
            # Literal 1m indicators
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
            # Literal 4-bar return on 1m
            df_full["1h_ret"] = (df_full["close"] / df_full["close"].shift(60)) - 1.0 # 60 bars = 1 hr

            univ_cond = (
                (c > ema50) & (c > ema110) &
                (c > vwap_lag) &
                (atr_pct >= 0.15) & # 1m ATR is naturally smaller in %
                mom_cond &
                (c <= 450.0) & (c >= 10.0)
            )

            rvol_thresh = 1.30
            t1_base = (
                macd_cross_up & (macd > 0) &
                (c > ema20) &
                (rsi7 > 50.0) & (adx >= 18.0) & (pdi > mdi) &
                (er10 >= 0.35) &
                (rvol >= rvol_thresh)
            )
            t1_tqs = (t1_base & (rvol >= max(1.25, rvol_thresh + 0.05)) & (c > high_5))

            sup_prev = np.maximum(ema9.shift(1), vwap_lag.shift(1)) + (0.15 * atr14)
            t2_pullback = (
                (macd > sig) & (macd > 0) & (pdi > mdi) &
                (ema9 > ema20) & (ema20 > ema50) &
                (c > ema9) &
                (l.shift(1) <= sup_prev) &
                (v.shift(1) < v.shift(2)) & (v >= 1.15 * v.shift(1)) & (rvol >= rvol_thresh) &
                (c > h.shift(1)) & (c.shift(1) <= h.shift(2)) & (c > o) &
                (rsi7 >= 52.0) & (rsi7 <= 74.0) &
                (((c - ema20) / (atr14 + 1e-9)) <= 2.10)
            )

        elif mode == "mtf_15m_1m":
            # Resample 1m data to 15m to get robust institutional signals, forward-fill back to 1m
            df_15m = df_full.resample("15min").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).dropna()

            atr14_15m = calculate_atr(df_15m, 14)
            atr_pct_15m = (atr14_15m / df_15m["close"]) * 100.0
            ema9_15m = df_15m["close"].ewm(span=9).mean()
            ema20_15m = df_15m["close"].ewm(span=20).mean()
            ema50_15m = df_15m["close"].ewm(span=50).mean()
            ema110_15m = df_15m["close"].ewm(span=110).mean()

            delta_15m = df_15m["close"].diff()
            gain_15m = delta_15m.clip(lower=0).rolling(7).mean()
            loss_15m = (-delta_15m.clip(upper=0)).rolling(7).mean()
            rsi7_15m = 100.0 - (100.0 / (1.0 + gain_15m / (loss_15m + 1e-9)))

            macd_15m = df_15m["close"].ewm(span=12).mean() - df_15m["close"].ewm(span=26).mean()
            sig_15m = macd_15m.ewm(span=9).mean()
            macd_cross_up_15m = (macd_15m > sig_15m) & (macd_15m.shift(1) <= sig_15m.shift(1))

            adx_15m, pdi_15m, mdi_15m = compute_adx_di(df_15m, 10)
            vwap_lag_15m = compute_vwap_shifted(df_15m)
            v_sma20_15m = df_15m["volume"].rolling(20).mean()
            rvol_15m = df_15m["volume"] / (v_sma20_15m + 1e-9)
            er10_15m = compute_er10(df_15m["close"])

            high_5_15m = df_15m["high"].rolling(5).max().shift(1)
            lookback_15m = min(300, len(df_15m) - 1)
            c_300_15m = df_15m["close"].shift(lookback_15m)
            mom_cond_15m = (df_15m["close"] > c_300_15m)

            df_15m["_date"] = df_15m.index.date
            day_open_15m = df_15m.groupby("_date")["open"].transform("first")
            daily_close_15m = df_15m.groupby("_date")["close"].last()
            prev_close_map_15m = daily_close_15m.shift(1).to_dict()
            df_15m["prev_close"] = df_15m["_date"].map(prev_close_map_15m)
            df_15m["open_gap_pct"] = (day_open_15m - df_15m["prev_close"]) / (df_15m["prev_close"] + 1e-9) * 100.0
            df_15m["day_ret"] = (df_15m["close"] / day_open_15m) - 1.0
            df_15m["1h_ret"] = (df_15m["close"] / df_15m["close"].shift(4)) - 1.0

            univ_cond_15m = (
                (df_15m["close"] > ema50_15m) & (df_15m["close"] > ema110_15m) &
                (df_15m["close"] > vwap_lag_15m) &
                (atr_pct_15m >= 0.50) &
                mom_cond_15m &
                (df_15m["close"] <= 450.0) & (df_15m["close"] >= 10.0)
            )

            rvol_thresh = 1.30
            t1_base_15m = (
                macd_cross_up_15m & (macd_15m > 0) &
                (df_15m["close"] > ema20_15m) &
                (rsi7_15m > 50.0) & (adx_15m >= 18.0) & (pdi_15m > mdi_15m) &
                (er10_15m >= 0.35) &
                (rvol_15m >= rvol_thresh)
            )
            t1_tqs_15m = (t1_base_15m & (rvol_15m >= max(1.25, rvol_thresh + 0.05)) & (df_15m["close"] > high_5_15m))

            sup_prev_15m = np.maximum(ema9_15m.shift(1), vwap_lag_15m.shift(1)) + (0.15 * atr14_15m)
            t2_pullback_15m = (
                (macd_15m > sig_15m) & (macd_15m > 0) & (pdi_15m > mdi_15m) &
                (ema9_15m > ema20_15m) & (ema20_15m > ema50_15m) &
                (df_15m["close"] > ema9_15m) &
                (df_15m["low"].shift(1) <= sup_prev_15m) &
                (df_15m["volume"].shift(1) < df_15m["volume"].shift(2)) & 
                (df_15m["volume"] >= 1.15 * df_15m["volume"].shift(1)) & 
                (rvol_15m >= rvol_thresh) &
                (df_15m["close"] > df_15m["high"].shift(1)) & 
                (df_15m["close"].shift(1) <= df_15m["high"].shift(2)) & 
                (df_15m["close"] > df_15m["open"]) &
                (rsi7_15m >= 52.0) & (rsi7_15m <= 74.0) &
                (((df_15m["close"] - ema20_15m) / (atr14_15m + 1e-9)) <= 2.10)
            )

            # Reindex / forward-fill 15m signals onto 1m timestamps
            idx_1m = df_full.index
            atr14 = atr14_15m.reindex(idx_1m, method="ffill")
            ema9 = ema9_15m.reindex(idx_1m, method="ffill")
            ema20 = ema20_15m.reindex(idx_1m, method="ffill")
            ema50 = ema50_15m.reindex(idx_1m, method="ffill")
            ema110 = ema110_15m.reindex(idx_1m, method="ffill")
            rsi7 = rsi7_15m.reindex(idx_1m, method="ffill")
            adx = adx_15m.reindex(idx_1m, method="ffill")
            vwap_lag = vwap_lag_15m.reindex(idx_1m, method="ffill")
            rvol = rvol_15m.reindex(idx_1m, method="ffill")
            er10 = er10_15m.reindex(idx_1m, method="ffill")
            c_300 = c_300_15m.reindex(idx_1m, method="ffill")
            univ_cond = univ_cond_15m.reindex(idx_1m, method="ffill").fillna(False)
            t1_base = t1_base_15m.reindex(idx_1m, method="ffill").fillna(False)
            t1_tqs = t1_tqs_15m.reindex(idx_1m, method="ffill").fillna(False)
            t2_pullback = t2_pullback_15m.reindex(idx_1m, method="ffill").fillna(False)

            df_full["_date"] = df_full.index.date
            day_open = df_full.groupby("_date")["open"].transform("first")
            df_full["open_gap_pct"] = df_15m["open_gap_pct"].reindex(idx_1m, method="ffill")
            df_full["day_ret"] = df_15m["day_ret"].reindex(idx_1m, method="ffill")
            df_full["1h_ret"] = df_15m["1h_ret"].reindex(idx_1m, method="ffill")

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

    positions = {}
    trades = []
    symbol_lockouts = {}
    daily_traded = {}
    daily_losses = {}
    last_exit_times = set()

    hwm = initial_equity
    hard_sl_mult = 1.0
    max_gap_up = 1.50
    symbol_lockout_days = 2
    be_mfe_atr = 0.45
    be_lock_offset = 0.05
    daily_loss_limit = 2
    scratch_bar_threshold = 4 if mode == "raw_1m" else 60

    print(f"\nSimulation Starting ({mode})... Total 1m timestamps: {len(all_ts)}")

    c300_dict = c300_breadth.to_dict()
    bench_day_dict = bench_day.to_dict()

    for ts_idx, ts in enumerate(all_ts):
        cur_date = ts.date()
        t_time = ts.time()

        if cur_date not in daily_traded:
            daily_traded[cur_date] = set()
            daily_losses[cur_date] = 0

        # Market Regime
        b_c300_now = c300_dict.get(ts, 0.50)
        b_day_now = bench_day_dict.get(ts, 0.0)
        is_crisis_env = (b_c300_now < 0.40) or (b_day_now < -0.006)
        is_bear_env = is_crisis_env or (b_c300_now < 0.48)

        # 1. Update Positions
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

            # V42 Ratchet
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
            elif be_mfe_atr > 0 and mfe_atr >= be_mfe_atr:
                stop_ladder = p["entry_p"] + (be_lock_offset * p["entry_atr"])
            else:
                stop_ladder = p["initial_sl"]

            p["active_sl"] = max(p["active_sl"], stop_ladder)
            p["prev_l"] = cur_l

            def finalize_trade(exit_fill, exit_reason):
                buy_val = p["entry_p"] * p["qty"]
                sell_val = exit_fill * p["qty"]
                gross = round(sell_val - buy_val, 2)
                chg = calc_dhan_charges(p["entry_p"], exit_fill, p["qty"], is_intraday=True)
                slip_val = round((p["entry_p"] * slippage + exit_fill * slippage) * p["qty"], 2)
                net = round(gross - chg - slip_val, 2)

                trades.append({
                    "ticker": ticker,
                    "entry_t": p["entry_t"],
                    "exit_t": ts,
                    "entry_p": p["entry_p"],
                    "exit_p": exit_fill,
                    "qty": p["qty"],
                    "gross_pnl": gross,
                    "dhan_charges": chg,
                    "slippage": slip_val,
                    "net_pnl": net,
                    "reason": exit_reason,
                    "mfe_atr": round(mfe_atr, 2),
                    "bars_held": p["bars_held"]
                })
                last_exit_times.add(ts)
                if net < 0:
                    daily_losses[cur_date] += 1
                if exit_reason == "STOP" and exit_fill <= p["entry_p"]:
                    symbol_lockouts[ticker] = cur_date + pd.Timedelta(days=symbol_lockout_days)
                del positions[ticker]

            if cur_l <= p["active_sl"]:
                raw_fill = min(cur_h, p["active_sl"]) * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                finalize_trade(exit_fill, "STOP")
                continue
            if p["bars_held"] >= scratch_bar_threshold and mfe_atr < 0.40 and cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                raw_fill = cur_c * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                finalize_trade(exit_fill, "SCRATCH")
                continue
            if t_time >= pd.Timestamp("15:15").time(): # Intraday square-off time
                raw_fill = cur_c * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                finalize_trade(exit_fill, "EOD")
                continue

        # 2. Capital & Risk Allocation
        ek = initial_equity + sum(t["net_pnl"] for t in trades)
        hwm = max(hwm, ek)
        dd_pct = (hwm - ek) / max(hwm, initial_equity)

        # Single Slot Discipline & Session Filter
        if len(positions) >= 1 or ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:30").time():
            continue

        if pd.Timestamp("11:30").time() <= t_time <= pd.Timestamp("12:30").time():
            continue
        if pd.Timestamp("10:00").time() <= t_time <= pd.Timestamp("10:05").time():
            continue

        if daily_loss_limit > 0 and daily_losses.get(cur_date, 0) >= daily_loss_limit:
            continue

        avail_margin = (0.95 * ek) - sum((p["entry_p"] * p["qty"]) / 5.0 for p in positions.values())
        if avail_margin < 10.0:
            continue
        avail_notional = avail_margin * 5.0

        consec_losses = 0
        for t in reversed(trades):
            if t["net_pnl"] < 0:
                consec_losses += 1
            else:
                break

        # Turbo base risk = 25.0 (5% risk per trade)
        alpha, beta = (0.085, 0.038) if consec_losses == 0 else (0.035, 0.020)
        raw_r = base_risk + (alpha * max(0.0, ek - 500.0))
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
                k_cluster[clean] = float(np.mean(t_bullish[p_idxs])) if p_idxs else 0.0
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
            if not np.isnan(gap_pct) and gap_pct > max_gap_up:
                continue

            if is_bear_env:
                k_cl = k_cluster.get(ticker, 0.0)
                decoupling = ((t_data["c"].iloc[idx] - t_data["day_open"].iloc[idx]) / (t_data["day_open"].iloc[idx] + 1e-9)) - b_day_now
                auth_pass = (k_cl >= 0.25) or (decoupling >= 0.0120 and t_data["rvol"].iloc[idx] >= 1.50)
                if not auth_pass:
                    continue

            is_t1 = t_data["t1_tqs"].iloc[idx] or t_data["t1_base"].iloc[idx]
            is_t2 = t_data["t2_pullback"].iloc[idx]
            if not (is_t1 or is_t2):
                continue

            bar_c = t_data["c"].iloc[idx]
            bar_h = t_data["h"].iloc[idx]
            bar_l = t_data["l"].iloc[idx]
            bar_atr = t_data["atr14"].iloc[idx]
            if bar_atr <= 0:
                continue

            tqs_bonus = 2.0 if t_data["t1_tqs"].iloc[idx] else (1.0 if is_t1 else 0.5)
            excess = t_data["excess_1h"].iloc[idx]
            base_score = (excess / (bar_atr / bar_c)) * tqs_bonus
            final_alpha = base_score * (1.0 + 0.35 * k_cluster.get(ticker, 0.0))

            candidates.append((
                ticker, bar_c, bar_h, bar_l, t_data["v"].iloc[idx],
                max(bar_atr, 0.05), final_alpha
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
            "entry_t": ts,
            "entry_p": entry_p,
            "initial_sl": active_sl,
            "active_sl": active_sl,
            "entry_atr": entry_atr,
            "qty": qty,
            "high_seen": cur_h,
            "prev_l": cur_l,
            "bars_held": 0
        }
        daily_traded[cur_date].add(ticker)

    # Compile Detailed Metrics
    final_cap = initial_equity + sum(t["net_pnl"] for t in trades)
    net_pnl = sum(t["net_pnl"] for t in trades)
    gross_pnl = sum(t["gross_pnl"] for t in trades)
    total_dhan = sum(t["dhan_charges"] for t in trades)
    total_slip = sum(t["slippage"] for t in trades)
    
    wins = [t["net_pnl"] for t in trades if t["net_pnl"] > 0]
    losses = [t["net_pnl"] for t in trades if t["net_pnl"] < 0]
    wr = (len(wins) / len(trades) * 100.0) if trades else 0.0
    gp = sum(wins)
    gl = abs(sum(losses))
    pf = round(gp / gl, 2) if gl > 0 else (999.0 if gp > 0 else 0.0)
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    max_win = max(wins) if wins else 0.0
    max_loss = min(losses) if losses else 0.0

    # Max Drawdown
    equity_curve = [initial_equity]
    for t in trades:
        equity_curve.append(equity_curve[-1] + t["net_pnl"])
    eq_series = pd.Series(equity_curve)
    peaks = eq_series.cummax()
    dds = (peaks - eq_series) / peaks
    max_dd_pct = dds.max() * 100.0
    max_dd_inr = (peaks - eq_series).max()

    reasons = {}
    for t in trades:
        r = t["reason"]
        reasons[r] = reasons.get(r, 0) + 1

    return {
        "mode": mode,
        "trades_count": len(trades),
        "win_rate": wr,
        "gross_pnl": gross_pnl,
        "dhan_charges": total_dhan,
        "slippage": total_slip,
        "net_pnl": net_pnl,
        "final_cap": final_cap,
        "roi_pct": (net_pnl / initial_equity) * 100.0,
        "profit_factor": pf,
        "max_dd_pct": max_dd_pct,
        "max_dd_inr": max_dd_inr,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_win": max_win,
        "max_loss": max_loss,
        "reasons": reasons,
        "trades": trades
    }

def print_audit_report(res, title):
    print("=" * 80)
    print(f"AUDIT REPORT: {title}")
    print("=" * 80)
    print(f"Initial Capital   : Rs 500.00")
    print(f"Final Capital     : Rs {res['final_cap']:.2f}")
    print(f"Net P&L           : Rs {res['net_pnl']:+.2f} ({res['roi_pct']:+.2f}%)")
    print(f"Gross P&L         : Rs {res['gross_pnl']:+.2f}")
    print(f"Dhan Brokerage/Tax: Rs {res['dhan_charges']:.2f}")
    print(f"Execution Slippage: Rs {res['slippage']:.2f}")
    print(f"Total Trades      : {res['trades_count']}")
    print(f"Win Rate          : {res['win_rate']:.2f}%")
    print(f"Profit Factor     : {res['profit_factor']:.2f}")
    print(f"Max Drawdown      : Rs {res['max_dd_inr']:.2f} ({res['max_dd_pct']:.2f}%)")
    print(f"Average Win       : Rs {res['avg_win']:+.2f}")
    print(f"Average Loss      : Rs {res['avg_loss']:+.2f}")
    print(f"Win / Loss Ratio  : {abs(res['avg_win'] / (res['avg_loss'] - 1e-9)):.2f}x")
    print(f"Largest Win       : Rs {res['max_win']:+.2f}")
    print(f"Largest Loss      : Rs {res['max_loss']:+.2f}")
    print(f"Exit Reasons      : {res['reasons']}")
    
    if res['trades']:
        print("\nFirst 5 Trades:")
        for t in res['trades'][:5]:
            print(f"  {t['entry_t']} | {t['ticker']:<12} | In: Rs {t['entry_p']:6.2f} | Out: Rs {t['exit_p']:6.2f} | Qty: {t['qty']:4d} | Net: Rs {t['net_pnl']:+6.2f} | {t['reason']}")
        if len(res['trades']) > 5:
            print("\nLast 5 Trades:")
            for t in res['trades'][-5:]:
                print(f"  {t['entry_t']} | {t['ticker']:<12} | In: Rs {t['entry_p']:6.2f} | Out: Rs {t['exit_p']:6.2f} | Qty: {t['qty']:4d} | Net: Rs {t['net_pnl']:+6.2f} | {t['reason']}")
    print("=" * 80 + "\n")

def main():
    # Load Universe 1: 30 small/midcaps of V29_30
    pkl_v29_30 = ROOT / "data" / "2026_1M_V29_30" / "universe_v29_30_1m.pkl"
    with open(pkl_v29_30, "rb") as f:
        pool_v29 = pickle.load(f)
    print(f"Loaded Universe V29_30 (32 stocks, 1m bars): {len(pool_v29)//2} stocks")

    # Load Universe 2: 30 large-caps
    pkl_30 = ROOT / "data" / "2026_1M_30_STOCKS" / "universe_30_1m.pkl"
    with open(pkl_30, "rb") as f:
        pool_30 = pickle.load(f)
    print(f"Loaded Universe 30 Large-Caps (30 stocks, 1m bars): {len(pool_30)//2} stocks")

    # TEST 1: V42 Turbo on V29_30 Universe (Raw 1m execution)
    res_v29_raw = run_v42_turbo_1m_engine(pool_v29, mode="raw_1m", base_risk=25.0)
    print_audit_report(res_v29_raw, "V42 TURBO ON V29_30 UNIVERSE (RAW 1-MINUTE EXECUTION)")

    # TEST 2: V42 Turbo on V29_30 Universe (MTF 15m Signal + 1m Tick/Bar Execution)
    res_v29_mtf = run_v42_turbo_1m_engine(pool_v29, mode="mtf_15m_1m", base_risk=25.0)
    print_audit_report(res_v29_mtf, "V42 TURBO ON V29_30 UNIVERSE (MTF 15m SIGNAL + 1-MINUTE PRECISION EXIT)")

    # TEST 3: V42 Turbo on 30 Large-Caps (Raw 1m execution)
    res_30_raw = run_v42_turbo_1m_engine(pool_30, mode="raw_1m", base_risk=25.0)
    print_audit_report(res_30_raw, "V42 TURBO ON 30 LARGE-CAPS (RAW 1-MINUTE EXECUTION)")

    # Save all results to disk
    out_results = {
        "v29_raw": res_v29_raw,
        "v29_mtf": res_v29_mtf,
        "30_raw": res_30_raw
    }
    with open(ROOT / "data" / "v42_turbo_1m_full_audit_results.pkl", "wb") as f:
        pickle.dump(out_results, f)
    print("All results saved to data/v42_turbo_1m_full_audit_results.pkl")

if __name__ == "__main__":
    main()
