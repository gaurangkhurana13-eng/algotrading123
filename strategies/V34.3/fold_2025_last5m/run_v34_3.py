#!/usr/bin/env python3
"""
BHAVISHYA V34.2: Sharpened Multi-Wave Compounding Engine
=============================================================================
3 Strict Out-of-Sample Folds:
  Fold 1: 2022 Full Year (Bear / War Horizon)           -> data/2022-23/universe_30_15m.pkl
  Fold 2: 2025 Last 5 Months (Aug 1 - Dec 31, 2025)     -> data/2025-26/universe_14_15m.pkl
  Fold 3: 2026 Full Horizon (Bull Momentum Expansion)   -> data/ (V29 30-stock basket)
"""

import os, sys, shutil, pickle, csv
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
    universe_filter: Optional[Set[str]] = None
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
            penny_trap = (c >= 30.0) & (atr14 >= 0.30)
            univ_cond = univ_cond & penny_trap

        # Trigger 1 (Primary Wave - V33 proven core)
        t1_base = (
            macd_cross_up & (macd > 0) &
            (c > ema20) &
            (rsi7 > 50.0) & (adx >= 18.0) & (pdi > mdi) &
            (er10 >= 0.35) &
            (rvol >= 1.20)
        )
        t1_tqs = (t1_base & (rvol >= 1.25) & (c > high_5))

        # Sharpened Trigger 2 (Institutional Pullback: Close > EMA9, RVOL >= 1.20)
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

    # Fast NumPy Bullish Matrix: (c > vwap_lag) & (excess_1h > 0)
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

    t2_start_t = pd.Timestamp("09:45").time()
    t2_end_t = pd.Timestamp("13:15").time()

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
            cur_rvol = t_data["rvol"].loc[ts]

            p["bars_held"] += 1
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0

            shooting_star = (cur_rvol >= 2.60) and (((cur_h - cur_c) / (cur_h - cur_l + 1e-6)) >= 0.45)
            extreme_rsi = (cur_rsi >= 86.0)
            break_ema9 = (cur_c < p.get("prev_l", cur_l)) and (cur_c < cur_ema9)
            circuit_prox = (mfe_atr >= 8.00)
            psi_climax = shooting_star or extreme_rsi or break_ema9 or circuit_prox

            # Hybrid Stage Ratchet
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

        # 2. Dynamic Equity & Risk Budget
        ek = initial_equity + sum(t["net_pnl"] for t in trades)
        hwm = max(hwm, ek)
        dd_pct = (hwm - ek) / max(hwm, initial_equity)

        if len(positions) >= 1 or ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
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

        if consec_losses >= 2 or (ek > 2500.0 and dd_pct > 0.16):
            alpha, beta = 0.015, 0.010
        elif consec_losses == 1:
            alpha, beta = 0.035, 0.020
        else:
            alpha, beta = alpha_streak, beta_streak

        if ek <= 2500.0:
            raw_r = 14.0 + (alpha * max(0.0, ek - 500.0))
        else:
            raw_r = 14.0 + (alpha * 2000.0) + (beta * (ek - 2500.0))

        gate_v2_cap = min(0.090 * ek, 258.0)
        if dd_pct > 0.15:
            gate_v2_cap *= 0.50
        target_risk = min(raw_r, gate_v2_cap)

        b_day_now = bench_day.get(ts, 0.0)
        b300_now = c300_breadth.get(ts, 0.5)
        is_severe_bleed_day = (b_day_now < -0.006)
        is_bear_env = (b300_now < 0.35) and (b_day_now < 0.0)

        # Fast Vectorized Cluster Confirmation
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
            else:
                k_cluster[clean] = float(sum_bullish - t_bullish[tick_i]) / max(1, N_stocks - 1)

        candidates = []
        for ticker in data_map.keys():
            if ticker in positions or ticker in daily_traded[cur_date]:
                continue
            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            idx = t_data["df"].index.get_loc(ts)
            if idx < 100:
                continue

            if not (t_data["univ_cond"].iloc[idx] and t_data["g1_cond"].iloc[idx] and t_data["g2_cond"].iloc[idx]):
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

    final_cap = initial_equity + sum(t["net_pnl"] for t in trades)
    wr = (sum(1 for t in trades if t["net_pnl"] > 0) / len(trades) * 100.0) if trades else 0.0
    net = sum(t["net_pnl"] for t in trades)
    wins = [t["net_pnl"] for t in trades if t["net_pnl"] > 0]
    losses = [t["net_pnl"] for t in trades if t["net_pnl"] < 0]
    gp = sum(wins)
    gl = abs(sum(losses))
    pf = round(gp / gl, 2) if gl > 0 else 999.0

    print(f"[{fold_label}] -> Trades: {len(trades):3d} | WR: {wr:5.1f}% | Net: Rs {net:+8.2f} | PF: {pf:5.2f} | End Wallet: Rs {final_cap:.2f}")
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

    src_pine = ROOT_DIR / "strategies" / "V34.3" / "v34_3.pine"
    dst_pine = out_dir / "v34_3.pine"
    if src_pine.resolve() != dst_pine.resolve():
        shutil.copy(src_pine, dst_pine)

    src_run = ROOT_DIR / "strategies" / "V34.3" / "run_v34_3.py"
    dst_run = out_dir / "run_v34_3.py"
    if src_run.resolve() != dst_run.resolve():
        shutil.copy(src_run, dst_run)

    initial_equity = 500.0
    wins = tdf[tdf["net_pnl"] > 0] if not tdf.empty else pd.DataFrame()
    losses = tdf[tdf["net_pnl"] < 0] if not tdf.empty else pd.DataFrame()
    wr = len(wins) / len(tdf) * 100.0 if len(tdf) > 0 else 0.0
    net = tdf["net_pnl"].sum() if not tdf.empty else 0.0
    cum = initial_equity + (tdf["net_pnl"].cumsum() if not tdf.empty else pd.Series([0.0]))
    hwm = cum.cummax()
    dd = (hwm - cum) / hwm * 100.0
    max_dd = dd.max() if len(dd) > 0 else 0.0

    net_win = wins["net_pnl"].sum() if not wins.empty else 0.0
    net_loss = abs(losses["net_pnl"].sum()) if not losses.empty else 0.0
    pf = round(net_win / net_loss, 2) if net_loss > 0 else 999.0
    avg_win = round(wins["net_pnl"].mean(), 2) if not wins.empty else 0.0
    avg_loss = round(losses["net_pnl"].mean(), 2) if not losses.empty else 0.0
    final_cap = cum.iloc[-1] if not cum.empty else initial_equity

    res_text = f"""----------------------------------------
INTRADAY
----------------------------------------
Total Trades: {len(tdf)}
Wins: {len(wins)}
Losses: {len(losses)}
Win Rate: {wr:.2f}%
Gross Profit: Rs {wins['gross_pnl'].sum() if not wins.empty else 0.0:.2f}
Gross Loss: Rs {abs(losses['gross_pnl'].sum()) if not losses.empty else 0.0:.2f}
Net Profit: Rs {net:.2f}
Profit Factor: {pf}
Average Win: Rs {avg_win:.2f}
Average Loss: Rs {avg_loss:.2f}
Max Drawdown: {max_dd:.2f}%
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
    print("BHAVISHYA V34.3: THE MASTER EXECUTION SIMULATION")
    print("=" * 75)

    v34_3_dir = ROOT_DIR / "strategies" / "V34.3"

    # 1. FOLD 1: 2022 Full Year (Bear / War Horizon)
    print("\n[FOLD 1] Simulating 2022 Full Year...")
    with open(ROOT_DIR / "data" / "2022-23" / "universe_30_15m.pkl", "rb") as f:
        p2022 = pickle.load(f)
    cap1, tr1 = run_simulation(p2022, "2022-01-01", "2022-12-31", "2022 Full Year (Bear)")
    save_gate_v2_artifacts(v34_3_dir / "fold_2022", "Fold 1: 2022 Full Year", tr1)

    # 2. FOLD 2: 2025 Last 5 Months (Aug 1 - Dec 31, 2025)
    print("\n[FOLD 2] Simulating 2025 Last 5 Months (Aug-Dec)...")
    with open(ROOT_DIR / "data" / "2025-26" / "universe_14_15m.pkl", "rb") as f:
        p2025 = pickle.load(f)
    cap2, tr2 = run_simulation(p2025, "2025-08-01", "2025-12-31", "2025 Last 5M (Aug-Dec)", universe_filter=V29_30_UNIVERSE)
    save_gate_v2_artifacts(v34_3_dir / "fold_2025_last5m", "Fold 2: 2025 Last 5M", tr2)

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
    save_gate_v2_artifacts(v34_3_dir / "fold_2026", "Fold 3: 2026 Bull", tr3)

    # 4. FOLD 4: 2023 Mid 4 Months (May 1 - Aug 31, 2023)
    print("\n[FOLD 4] Simulating 2023 Mid 4 Months (May-Aug)...")
    with open(ROOT_DIR / "data" / "2023-24" / "universe_30_15m.pkl", "rb") as f:
        p2023 = pickle.load(f)
    cap4, tr4 = run_simulation(p2023, "2023-05-01", "2023-08-31", "2023 Mid 4M", universe_filter=V29_30_UNIVERSE)
    save_gate_v2_artifacts(v34_3_dir / "unseen_2023_mid4m", "Fold 4: 2023 Mid 4M", tr4)

    # 5. FOLD 5: 2025 Q4 (Oct 1 - Dec 31, 2025)
    print("\n[FOLD 5] Simulating 2025 Q4 (Oct-Dec)...")
    cap5, tr5 = run_simulation(p2025, "2025-10-01", "2025-12-31", "2025 Q4 (Oct-Dec)")
    save_gate_v2_artifacts(v34_3_dir / "unseen_2025_q4", "Fold 5: 2025 Q4", tr5)

    # 6. FOLD 6: 2026 52 Unseen Midcaps
    print("\n[FOLD 6] Simulating 2026 Unseen 52 Midcaps...")
    clean_78 = {k.replace(".NS", "") for k in u1.keys()}
    unseen_stocks_2026 = clean_78 - V29_30_UNIVERSE
    cap6, tr6 = run_simulation(u1, "2026-01-01", "2026-12-31", "2026 Unseen 52 Midcaps", universe_filter=unseen_stocks_2026)
    save_gate_v2_artifacts(v34_3_dir / "unseen_2026", "Fold 6: 2026 Unseen 52 Midcaps", tr6)

    # 7. FOLD 7: 2024 H1 (First 6 Months)
    print("\n[FOLD 7] Simulating 2024 H1 (First 6 Months)...")
    with open(ROOT_DIR / "data" / "2024-25" / "universe_14_15m.pkl", "rb") as f:
        p2024 = pickle.load(f)
    cap7, tr7 = run_simulation(p2024, "2024-01-01", "2024-06-30", "Fold 7: H1 2024", universe_filter=V29_30_UNIVERSE)
    save_gate_v2_artifacts(v34_3_dir / "fold_2024_h1", "Fold 7: H1 2024", tr7)

    # Root default (Fold 3 Bull)
    save_gate_v2_artifacts(v34_3_dir, "V34.3 Main (2026 Bull)", tr3)

    print("\n" + "=" * 75)
    print("V34.3 ALL 7 FOLDS SIMULATION & ARTIFACT GENERATION COMPLETE")
    print("=" * 75)
