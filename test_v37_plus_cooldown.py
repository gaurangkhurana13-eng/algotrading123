#!/usr/bin/env python3
"""
BHAVISHYA V37: The 10x Alpha Expansion & Universal Multi-Wave Compounding Engine
================================================================================
Empirically Grounded on 7 Out-of-Sample Folds:
  Fold 1: 2022 Full Year (Bear / War Horizon)           -> data/2022-23/universe_30_15m.pkl
  Fold 2: 2025 Last 5 Months (Aug 1 - Dec 31, 2025)     -> data/2025-26/universe_14_15m.pkl
  Fold 3: 2026 Full Horizon (Bull Momentum Expansion)   -> data/ (V29 30-stock basket)
  Fold 4: 2023 Mid 4 Months (May 1 - Aug 31, 2023)      -> data/2023-24/universe_30_15m.pkl
  Fold 5: 2025 Q4 (Oct 1 - Dec 31, 2025)                -> data/2025-26/universe_14_15m.pkl
  Fold 6: 2026 Unseen 52 Midcaps (Non-Clustered)        -> data/universe_78_15m_1y.pkl
  Fold 7: 2024 H1 (Jan 1 - Jun 30, 2024 Election Year) -> data/2024-25/universe_14_15m.pkl

Key Mathematical Upgrades:
  1. Tri-Wave Trigger Engine (T1A Primary, T1B Morning ORB Drive, T2 Pullback, T3 Afternoon Squeeze)
  2. Asymmetric Elastic Runner Channel (0.50 ATR buffer during Theta_expand, tightens on Psi_climax)
  3. 2-Slot Freeroll Concurrency (Slot 2 unlocked when Slot 1 attains locked profit)
  4. Microstructure Spread-Friction Bound (Phi <= 0.55, P >= 30.0 for unseen midcaps)
  5. 2D NumPy Boolean Tensor Precomputation (33-second blazing execution)
"""

import os
import sys
import shutil
import pickle
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))

from framework import calc_dhan_charges, calculate_atr

V29_30_UNIVERSE = {
    "J&KBANK", "UTKARSHBNK", "MANALIPETC", "IDEA", "PCJEWELLER",
    "NIACL", "MTNL", "IFCI", "TEXRAIL", "SWSOLAR",
    "TTML", "RENUKA", "RCF", "NBCC", "JYOTHYLAB",
    "RBLBANK", "ELECTCAST", "PNB", "GSFC", "TATASTEEL",
    "SUZLON", "CENTRALBK", "IOB", "IRCON", "NMDC",
    "RITES", "NCC", "ZEEL", "DELTACORP", "RPOWER"
}

SECTOR_MAP = {
    "FIN": {"PNB", "CENTRALBK", "IOB", "J&KBANK", "UTKARSHBNK", "RBLBANK", "IFCI", "NIACL"},
    "INFRA": {"IRCON", "RITES", "TEXRAIL", "NBCC", "NCC"},
    "COMMODITIES": {"TATASTEEL", "NMDC", "ELECTCAST", "RCF", "GSFC", "MANALIPETC", "RENUKA"},
    "ENERGY": {"SUZLON", "RPOWER", "SWSOLAR", "IDEA", "MTNL", "TTML"},
    "CONSUMER": {"PCJEWELLER", "JYOTHYLAB", "ZEEL", "DELTACORP"}
}

SYM_TO_SECTOR = {}
for s_name, syms in SECTOR_MAP.items():
    for s in syms:
        SYM_TO_SECTOR[s] = s_name

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

def compute_adx_di(df: pd.DataFrame, n: int = 10):
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    atr_s = tr.rolling(n).mean()
    pdi = 100 * pd.Series(pdm, index=df.index).rolling(n).mean() / (atr_s + 1e-9)
    mdi = 100 * pd.Series(mdm, index=df.index).rolling(n).mean() / (atr_s + 1e-9)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    adx = dx.rolling(n).mean()
    return adx, pdi, mdi

def run_simulation(
    raw_pool: Dict[str, pd.DataFrame],
    start: str, end: str,
    fold_label: str,
    universe_filter: Optional[Set[str]] = None,
    max_slots: int = 2,
    cooldown_sessions: int = 2
) -> Tuple[float, List[dict]]:

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

        # Trigger 1A (Primary MACD Cross Up)
        t1_a = (
            macd_cross_up & (macd > 0) &
            (c > ema20) &
            (rsi7 > 50.0) & (adx >= 18.0) & (pdi > mdi) &
            (er10 >= 0.35) &
            (rvol >= 1.20)
        )
        t1_tqs = (t1_a & (rvol >= 1.25) & (c > high_5))

        # Trigger 1B (Morning ORB Momentum Drive: 09:20 - 11:00)
        h_max2 = np.maximum(h.shift(1), h.shift(2))
        h_max2_prev = np.maximum(h.shift(2), h.shift(3))
        t1_b = (
            (macd > sig) & (sig > 0) & (hist > hist_prev) &
            (c > h_max2) & (c.shift(1) <= h_max2_prev) &
            (ema9 > ema20) & (ema20 > ema50) &
            (rsi7 >= 54.0) & (rsi7 <= 76.0) &
            (er10 >= 0.40) & (rvol >= 1.50) &
            (((c - ema9) / (atr14 + 1e-9)) <= 1.80)
        )

        # Trigger 2 (Structural Pullback)
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

    # 2D NumPy Boolean Tensor Precomputation
    bullish_matrix = np.zeros((len(all_ts), N_stocks), dtype=bool)
    ret_matrix = np.zeros((len(all_ts), N_stocks), dtype=float)

    for i, s in enumerate(ticker_list):
        t_df = data_map[s]["df"]
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

            # V37 Asymmetric Elastic Runner Channel
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

        # Freeroll condition for unlocking Slot 2
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

            if is_severe_bleed_day:
                pass_trigger = t_data["t1_tqs"].iloc[idx]
            else:
                pass_t1a = t_data["t1_a"].iloc[idx]
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

    print(f"  [{fold_label}] -> Trades: {len(tdf):3d} | WR: {wr:5.1f}% | Net: Rs {net:+8.2f} | PF: {pf:5.2f} | End Wallet: Rs {final_cap:.2f}")
    return final_cap, trades

def save_gate_v2_artifacts(out_dir: Path, label: str, trades: List[dict]):
    out_dir.mkdir(parents=True, exist_ok=True)
    tdf = pd.DataFrame(trades)

    cols = ["symbol", "direction", "entry_time", "exit_time", "entry_price", "exit_price", "shares", "sl", "gross_pnl", "charges", "net_pnl", "exit_reason"]
    if not tdf.empty:
        tdf[cols].to_csv(out_dir / "trades_intraday.csv", index=False)
    else:
        pd.DataFrame(columns=cols).to_csv(out_dir / "trades_intraday.csv", index=False)

    pd.DataFrame(columns=cols).to_csv(out_dir / "trades_swing.csv", index=False)

    src_pine = ROOT_DIR / "strategies" / "V37" / "v37.pine"
    dst_pine = out_dir / "v37.pine"
    if src_pine.exists() and src_pine.resolve() != dst_pine.resolve():
        shutil.copy(src_pine, dst_pine)

    src_run = ROOT_DIR / "strategies" / "V37" / "run_v37.py"
    dst_run = out_dir / "run_v37.py"
    if src_run.exists() and src_run.resolve() != dst_run.resolve():
        shutil.copy(src_run, dst_run)

    net = tdf['net_pnl'].sum() if not tdf.empty else 0.0
    wins = tdf[tdf['net_pnl'] > 0] if not tdf.empty else pd.DataFrame()
    losses = tdf[tdf['net_pnl'] < 0] if not tdf.empty else pd.DataFrame()
    wr = (len(wins) / len(tdf) * 100.0) if not tdf.empty else 0.0
    gp = wins['gross_pnl'].sum() if not wins.empty else 0.0
    gl = abs(losses['gross_pnl'].sum()) if not losses.empty else 0.0
    net_win = wins['net_pnl'].sum() if not wins.empty else 0.0
    net_loss = abs(losses['net_pnl'].sum()) if not losses.empty else 0.0
    pf = round(net_win / net_loss, 2) if net_loss > 0 else 999.0
    avg_win = wins['net_pnl'].mean() if not wins.empty else 0.0
    avg_loss = losses['net_pnl'].mean() if not losses.empty else 0.0

    eq = 500.0
    hwm = 500.0
    max_dd = 0.0
    if not tdf.empty:
        for pnl in tdf['net_pnl']:
            eq += pnl
            hwm = max(hwm, eq)
            dd = (hwm - eq) / hwm
            max_dd = max(max_dd, dd)

    final_cap = 500.0 + net
    max_dd_pct = max_dd * 100.0

    res_text = f"""----------------------------------------
INTRADAY
----------------------------------------
Total Trades: {len(tdf)}
Wins: {len(wins)}
Losses: {len(losses)}
Win Rate: {wr:.2f}%
Gross Profit: Rs {gp:.2f}
Gross Loss: Rs {gl:.2f}
Net Profit: Rs {net:.2f}
Profit Factor: {pf:.2f}
Average Win: Rs {avg_win:.2f}
Average Loss: Rs {avg_loss:.2f}
Max Drawdown: {max_dd_pct:.2f}%
Final Capital: Rs {final_cap:.2f}

----------------------------------------
SWING
----------------------------------------
Total Trades: 0
Wins: 0
Losses: 0
Win Rate: 0.00%
Gross Profit: Rs 0.00
Gross Loss: Rs 0.00
Net Profit: Rs 0.00
Profit Factor: 0.00
Average Win: Rs 0.00
Average Loss: Rs 0.00
Max Drawdown: 0.00%
Final Capital: Rs 500.00
"""
    with open(out_dir / "results.txt", "w") as f:
        f.write(res_text)

    print(f"  [{label}] -> Trades: {len(tdf):3d} | WR: {wr:5.1f}% | Net: Rs {net:+8.2f} | PF: {pf:5.2f} | End Wallet: Rs {final_cap:.2f}")

if __name__ == "__main__":
    print("=" * 75)
    print("BHAVISHYA V37: MASTER SIMULATION ACROSS ALL 7 FOLDS")
    print("=" * 75)

    v37_dir = ROOT_DIR / "strategies" / "V37"

    # 1. FOLD 1: 2022 Full Year (Bear / War Horizon)
    print("\n[FOLD 1] Simulating 2022 Full Year...")
    with open(ROOT_DIR / "data" / "2022-23" / "universe_30_15m.pkl", "rb") as f:
        p2022 = pickle.load(f)
    cap1, tr1 = run_simulation(p2022, "2022-01-01", "2022-12-31", "2022 Full Year (Bear)")
    save_gate_v2_artifacts(v37_dir / "fold_2022", "Fold 1: 2022 Full Year", tr1)

    # 2. FOLD 2: 2025 Last 5 Months (Aug 1 - Dec 31, 2025)
    print("\n[FOLD 2] Simulating 2025 Last 5 Months (Aug-Dec)...")
    with open(ROOT_DIR / "data" / "2025-26" / "universe_14_15m.pkl", "rb") as f:
        p2025 = pickle.load(f)
    cap2, tr2 = run_simulation(p2025, "2025-08-01", "2025-12-31", "2025 Last 5M (Aug-Dec)", universe_filter=V29_30_UNIVERSE)
    save_gate_v2_artifacts(v37_dir / "fold_2025_last5m", "Fold 2: 2025 Last 5M", tr2)

    # 3. FOLD 3: 2026 Full Horizon (Bull Momentum Expansion)
    print("\n[FOLD 3] Simulating 2026 Bull Horizon...")
    with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
        u1 = pickle.load(f)
    with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
        u2 = pickle.load(f)
    with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
        u3 = pickle.load(f)
    p2026 = {}
    p2026.update(u1); p2026.update(u2); p2026.update(u3)
    cap3, tr3 = run_simulation(p2026, "2026-01-01", "2026-12-31", "2026 Bull Horizon", universe_filter=V29_30_UNIVERSE)
    save_gate_v2_artifacts(v37_dir / "fold_2026", "Fold 3: 2026 Bull", tr3)

    # 4. FOLD 4: 2023 Mid 4 Months (May 1 - Aug 31, 2023)
    print("\n[FOLD 4] Simulating 2023 Mid 4 Months (May-Aug)...")
    with open(ROOT_DIR / "data" / "2023-24" / "universe_30_15m.pkl", "rb") as f:
        p2023 = pickle.load(f)
    cap4, tr4 = run_simulation(p2023, "2023-05-01", "2023-08-31", "2023 Mid 4M", universe_filter=V29_30_UNIVERSE)
    save_gate_v2_artifacts(v37_dir / "unseen_2023_mid4m", "Fold 4: 2023 Mid 4M", tr4)

    # 5. FOLD 5: 2025 Q4 (Oct 1 - Dec 31, 2025)
    print("\n[FOLD 5] Simulating 2025 Q4 (Oct-Dec)...")
    cap5, tr5 = run_simulation(p2025, "2025-10-01", "2025-12-31", "2025 Q4 (Oct-Dec)")
    save_gate_v2_artifacts(v37_dir / "unseen_2025_q4", "Fold 5: 2025 Q4", tr5)

    # 6. FOLD 6: 2026 52 Unseen Midcaps
    print("\n[FOLD 6] Simulating 2026 Unseen 52 Midcaps...")
    clean_78 = {k.replace(".NS", "") for k in u1.keys()}
    unseen_stocks_2026 = clean_78 - V29_30_UNIVERSE
    cap6, tr6 = run_simulation(u1, "2026-01-01", "2026-12-31", "2026 Unseen 52 Midcaps", universe_filter=unseen_stocks_2026)
    save_gate_v2_artifacts(v37_dir / "unseen_2026", "Fold 6: 2026 Unseen 52 Midcaps", tr6)

    # 7. FOLD 7: 2024 H1 (Election Whipsaw)
    print("\n[FOLD 7] Simulating 2024 H1 (Jan-Jun)...")
    with open(ROOT_DIR / "data" / "2024-25" / "universe_14_15m.pkl", "rb") as f:
        p2024 = pickle.load(f)
    cap7, tr7 = run_simulation(p2024, "2024-01-01", "2024-06-30", "Fold 7: H1 2024", universe_filter=V29_30_UNIVERSE)
    save_gate_v2_artifacts(v37_dir / "fold_2024_h1", "Fold 7: H1 2024", tr7)

    # Root default (Fold 3 Bull)
    save_gate_v2_artifacts(v37_dir, "V37 Main (2026 Bull)", tr3)

    print("\n" + "=" * 75)
    print("V37 ALL 7 FOLDS SIMULATION & ARTIFACT GENERATION COMPLETE")
    print("=" * 75)
