#!/usr/bin/env python3
"""
BHAVISHYA V33: Cross-Sectional Regime-Gated Multi-Fold Compounding Engine
=============================================================================
3-Fold Out-of-Sample Certification:
  Fold 1: 2022 Full Year (Bear / War Horizon)               -> universe_30_15m.pkl
  Fold 2: 2025 H2 (July - December 2025, Sideways/Chop)     -> universe_14_15m.pkl
  Fold 3: 2026 Full Horizon (Bull Momentum Expansion)       -> V29 30-stock basket
"""

import os, sys, shutil, pickle, csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

# V29 proven 30 high-beta momentum stocks
V29_30_UNIVERSE = {
    "J&KBANK", "UTKARSHBNK", "MANALIPETC", "IDEA", "PCJEWELLER",
    "NIACL", "MTNL", "IFCI", "TEXRAIL", "SWSOLAR",
    "TTML", "RENUKA", "RCF", "NBCC", "JYOTHYLAB",
    "RBLBANK", "ELECTCAST", "PNB", "GSFC", "TATASTEEL",
    "SUZLON", "CENTRALBK", "IOB", "IRCON", "NMDC",
    "RITES", "NCC", "ZEEL", "DELTACORP", "RPOWER"
}

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
        c, h, l, v = df_full["close"], df_full["high"], df_full["low"], df_full["volume"]
        atr14 = calculate_atr(df_full, 14)
        atr_pct = (atr14 / c) * 100.0
        ema9 = c.ewm(span=9).mean()
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
        mom_cond = (c > c.shift(lookback))

        g4_v26 = (
            macd_cross_up & (macd > 0) &
            (c > ema50) & (c > ema110) &
            (c > vwap_lag) &
            (rsi7 > 50.0) & (adx >= 18.0) & (pdi > mdi) &
            (er10 >= 0.35) &
            (rvol >= 1.20) &
            (atr_pct >= 0.50) &
            mom_cond &
            (c <= 450.0) & (c >= 10.0)
        )

        g4_v28 = (g4_v26 & (rvol >= 1.25) & (c > high_5))

        df_full["_date"] = df_full.index.date
        day_open = df_full.groupby("_date")["open"].transform("first")
        df_full["day_ret"] = (df_full["close"] / day_open) - 1.0
        df_full["1h_ret"] = (df_full["close"] / df_full["close"].shift(4)) - 1.0

        data_map[clean] = {
            "df": df_full, "c": c, "h": h, "l": l, "v": v, "atr14": atr14,
            "ema9": ema9, "rsi7": rsi7, "vwap_lag": vwap_lag,
            "g4_v26": g4_v26, "g4_v28": g4_v28,
            "day_ret": df_full["day_ret"], "1h_ret": df_full["1h_ret"]
        }

    bench_day = pd.concat([d["day_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    bench_1h  = pd.concat([d["1h_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)

    for clean, d in data_map.items():
        b_d = bench_day.reindex(d["day_ret"].index)
        b_1 = bench_1h.reindex(d["1h_ret"].index)
        d["g1_cond"] = (d["day_ret"] > b_d)
        d["g2_cond"] = (d["1h_ret"] > b_1)
        d["excess_1h"] = d["1h_ret"] - b_1

    all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
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

    for ts in all_ts:
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

            p["bars_held"] += 1
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0

            if mfe_atr >= 4.00:
                stop_ladder = p["high_seen"] - (0.10 * p["entry_atr"])
            elif mfe_atr >= 3.00:
                stop_ladder = p["high_seen"] - (0.14 * p["entry_atr"])
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
                finalize_trade(p["active_sl"] * (1.0 - slippage), "STOP")
                continue
            if p["bars_held"] >= 4 and mfe_atr < 0.40 and cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                finalize_trade(cur_c * (1.0 - slippage), "SCRATCH")
                continue
            if t_time >= pd.Timestamp("14:45").time():
                finalize_trade(cur_c * (1.0 - slippage), "EOD")
                continue

        # 2. Dynamic Equity & Risk Budget
        ek = initial_equity + sum(t["net_pnl"] for t in trades)
        hwm = max(hwm, ek)
        dd_pct = (hwm - ek) / max(hwm, initial_equity)

        if len(positions) >= 1 or ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
            continue

        avail_margin = (0.95 * ek) - (sum(p["entry_p"] * p["qty"] for p in positions.values()) / 5.0)
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

        # 3. Dynamic Market Regime Detection (Bleed vs Expansion)
        b_day_now = bench_day.get(ts, 0.0)
        is_severe_bleed_day = (b_day_now < -0.006)

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

            gate_to_use = t_data["g4_v28"].iloc[idx] if is_severe_bleed_day else t_data["g4_v26"].iloc[idx]
            if not (gate_to_use and t_data["g1_cond"].iloc[idx] and t_data["g2_cond"].iloc[idx]):
                continue

            candidates.append((
                ticker,
                t_data["c"].iloc[idx],
                t_data["h"].iloc[idx],
                t_data["l"].iloc[idx],
                t_data["v"].iloc[idx],
                max(t_data["atr14"].iloc[idx], 0.05),
                t_data["excess_1h"].iloc[idx]
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
        qty = min(qty_r, qty_m)
        if cur_v > 0:
            qty = min(qty, int(0.045 * cur_v))

        if qty >= 1:
            positions[ticker] = {
                "entry_p": entry_p,
                "qty": qty,
                "entry_atr": entry_atr,
                "active_sl": active_sl,
                "initial_sl": active_sl,
                "high_seen": entry_p,
                "bars_held": 0,
                "entry_t": ts
            }
            daily_traded[cur_date].add(ticker)

    return ek, trades

def save_gate_v2_artifacts(out_dir: Path, label: str, trades: List[dict]):
    out_dir.mkdir(parents=True, exist_ok=True)
    tdf = pd.DataFrame(trades)

    cols = ["symbol", "direction", "entry_time", "exit_time", "entry_price", "exit_price", "shares", "sl", "gross_pnl", "charges", "net_pnl", "exit_reason"]
    if not tdf.empty:
        tdf[cols].to_csv(out_dir / "trades_intraday.csv", index=False)
    else:
        pd.DataFrame(columns=cols).to_csv(out_dir / "trades_intraday.csv", index=False)

    pd.DataFrame(columns=cols).to_csv(out_dir / "trades_swing.csv", index=False)

    # Safely copy pine and run files if not destination
    src_pine = ROOT_DIR / "strategies" / "V33" / "v33.pine"
    dst_pine = out_dir / "v33.pine"
    if src_pine.resolve() != dst_pine.resolve():
        shutil.copy(src_pine, dst_pine)

    src_run = ROOT_DIR / "strategies" / "V33" / "run_v33.py"
    dst_run = out_dir / "run_v33.py"
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
    print("BHAVISHYA V33 — 3-FOLD FORENSIC CERTIFICATION RUN")
    print("=" * 75)

    v33_dir = ROOT_DIR / "strategies" / "V33"

    # 1. FOLD 1: 2022 Full Year (Bear / War)
    print("\n[FOLD 1] Simulating 2022 Full Year (Bear / War)...")
    with open(ROOT_DIR / "data" / "2022-23" / "universe_30_15m.pkl", "rb") as f:
        p2022 = pickle.load(f)
    cap1, tr1 = run_simulation(p2022, "2022-01-01", "2022-12-31", "2022 Bear (Full)")
    save_gate_v2_artifacts(v33_dir / "fold_2022", "Fold 1: 2022 Bear", tr1)

    # 2. FOLD 2: 2025 H2 Chop
    print("\n[FOLD 2] Simulating 2025 H2 Monsoon Chop...")
    with open(ROOT_DIR / "data" / "2025-26" / "universe_14_15m.pkl", "rb") as f:
        p2025 = pickle.load(f)
    cap2, tr2 = run_simulation(p2025, "2025-07-01", "2025-12-31", "2025 H2 Chop", universe_filter=V29_30_UNIVERSE)
    save_gate_v2_artifacts(v33_dir / "fold_2025h2", "Fold 2: 2025 H2 Chop", tr2)

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
    save_gate_v2_artifacts(v33_dir / "fold_2026", "Fold 3: 2026 Bull", tr3)

    # Main folder default (Fold 3 Bull for general audit)
    save_gate_v2_artifacts(v33_dir, "V33 Main (2026 Bull)", tr3)

    print("\n" + "=" * 75)
    print("V33 MULTI-FOLD SIMULATION COMPLETE")
    print("=" * 75)
