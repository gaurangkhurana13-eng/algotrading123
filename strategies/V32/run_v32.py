#!/usr/bin/env python3
"""
BHAVISHYA V32: Cross-Sectional Regime-Gated Multi-Wave Compounding Engine
Architecture by ARGON | Coded for Gate V2 Compliance
==========================================================================
3-Fold Out-of-Sample Test:
  Fold 1: 2022 Full Year (Bear / Russia-Ukraine War)         -> 30-stock universe
  Fold 2: 2025 H2 (July - December 2025, Sideways/Chop)     -> V29 30-stock universe
  Fold 3: 2026 Full Year till date (Bull Momentum Expansion) -> V29 30-stock universe
"""

import os, sys, pickle, csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

# V29's exact 30 high-beta stocks — the same universe V29 was trained on
# We restrict Folds 2 & 3 to ONLY these stocks for a fair cross-sectional MRC signal
V29_30_UNIVERSE = {
    "J&KBANK", "UTKARSHBNK", "MANALIPETC", "IDEA", "PCJEWELLER",
    "NIACL", "MTNL", "IFCI", "TEXRAIL", "SWSOLAR",
    "TTML", "RENUKA", "RCF", "NBCC", "JYOTHYLAB",
    "RBLBANK", "ELECTCAST", "PNB", "GSFC", "TATASTEEL",
    "SUZLON", "CENTRALBK", "IOB", "IRCON", "NMDC",
    "RITES", "NCC", "ZEEL", "DELTACORP", "RPOWER"
}

# ============================================================================
# CONFIG (All Argon V32 Parameters - Fixed Across All 3 Folds)
# ============================================================================
@dataclass
class V32Config:
    initial_capital: float = 500.0
    slippage: float = 0.0005          # 0.05% per leg

    # MRC Breadth Thresholds
    mrc_vwap_lockout: float = 0.40
    mrc_vwap_expansion: float = 0.60
    mrc_ema_lockout: float = 0.30
    mrc_ema_expansion: float = 0.50
    mrc_bench_panic: float = -0.0020
    mrc_bench_thrust: float = 0.0020

    # Gate 4 Filters
    rsi_min: float = 52.0
    rsi_max: float = 78.0
    adx_min: float = 18.0
    er_min: float = 0.35
    rvol_min: float = 1.25
    atr_pct_min: float = 0.0050
    max_ema9_stretch_atr: float = 1.35
    min_price: float = 10.0
    max_price: float = 450.0
    lookback: int = 300

    # Risk Scaling
    risk_selective: float = 0.0110
    risk_expansion: float = 0.0210
    risk_dd_throttle: float = 0.0050
    dd_throttle_at: float = 0.035
    max_lev_selective: float = 1.35
    max_lev_expansion: float = 2.50

    # Exit Ladder
    init_sl_atr: float = 0.80
    stagnation_bars: int = 4
    stagnation_mfe_thresh: float = 0.35
    stagnation_sl_atr: float = 0.35
    lock_mfe_atr: float = 0.85
    lock_offset_atr: float = 0.40
    phase1b_mfe_atr: float = 1.40
    phase2_mfe_atr: float = 2.80
    climax_mfe_atr: float = 4.00

    # Freeroll Slot
    enable_freeroll: bool = True

CFG = V32Config()

# ============================================================================
# INDICATORS (Zero Lookahead, Causal Only)
# ============================================================================
def wilder_rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0/n, adjust=False).mean()

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index().copy()
    c, h, l, v = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float), df["volume"].astype(float)

    df["ema9"]  = c.ewm(span=9,   adjust=False).mean()
    df["ema20"] = c.ewm(span=20,  adjust=False).mean()
    df["ema50"] = c.ewm(span=50,  adjust=False).mean()
    df["ema110"]= c.ewm(span=110, adjust=False).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    hist  = macd - sig
    df["macd"] = macd; df["sig"] = sig; df["hist"] = hist

    delta = c.diff()
    avg_g = wilder_rma(delta.clip(lower=0), 7)
    avg_l = wilder_rma((-delta).clip(lower=0), 7)
    rs = avg_g / avg_l.replace(0, np.nan)
    df["rsi7"] = (100 - 100/(1+rs)).fillna(50)

    prev_c = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    df["atr14"] = wilder_rma(tr, 14)
    df["atr_pct"] = df["atr14"] / c.replace(0, np.nan)

    up = h - h.shift(1); dn = l.shift(1) - l
    pdm = np.where((up>dn)&(up>0), up, 0.0)
    mdm = np.where((dn>up)&(dn>0), dn, 0.0)
    atr10 = wilder_rma(tr, 10).replace(0, np.nan)
    df["pdi"] = 100*wilder_rma(pd.Series(pdm, index=df.index), 10) / atr10
    df["mdi"] = 100*wilder_rma(pd.Series(mdm, index=df.index), 10) / atr10
    dx = 100*(df["pdi"]-df["mdi"]).abs() / (df["pdi"]+df["mdi"]).replace(0, np.nan)
    df["adx10"] = wilder_rma(dx.fillna(0), 10)

    net10 = (c - c.shift(10)).abs()
    sum10 = c.diff().abs().rolling(10).sum().replace(0, np.nan)
    df["er10"] = (net10/sum10).fillna(0)

    vol_ma20 = v.rolling(20).mean().shift(1).replace(0, np.nan)
    df["rvol20"] = (v/vol_ma20).fillna(0)

    # Intraday VWAP (zero-lookahead: shift by 1 bar)
    dates = df.index.date
    tp = (h+l+c)/3
    pv = tp*v
    cum_pv = pv.groupby(dates).cumsum()
    cum_v  = v.groupby(dates).cumsum().replace(0, np.nan)
    df["vwap"]     = cum_pv/cum_v
    df["vwap_lag"] = df["vwap"].groupby(dates).shift(1).fillna(df["vwap"])
    df["vwap_rising"] = df["vwap"] >= df["vwap_lag"]

    df["ret1h"]    = (c - c.shift(4)) / c.shift(4).replace(0, np.nan)
    df["close300"] = c.shift(CFG.lookback)
    df["h6_lag1"]  = h.shift(1).rolling(6).max()
    df["h6_lag2"]  = h.shift(2).rolling(6).max()

    # Triggers
    trig_a = (macd>sig) & (macd.shift(1)<=sig.shift(1)) & (macd>0)
    trig_b = (
        (macd>sig) & (sig>0) &
        (hist.shift(1)<hist.shift(2)) & (hist>hist.shift(1)) &
        (l.shift(1) <= df["ema9"].shift(1) + 0.25*df["atr14"]) &
        (c > h.shift(1))
    )
    trig_c = (
        (c > df["h6_lag1"]) & (c.shift(1) <= df["h6_lag2"]) &
        (df["rvol20"]>=1.75) & (df["er10"]>=0.45) &
        (macd>sig) & (macd>0)
    )
    df["tA"] = trig_a; df["tB"] = trig_b; df["tC"] = trig_c

    df["_date"] = df.index.date
    return df

# ============================================================================
# MRC PANEL BUILDER
# ============================================================================
def build_mrc_panel(processed: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(sorted(set().union(*(df.index for df in processed.values()))))
    above_vwap, ema_trend, ret1h = {}, {}, {}

    for sym, df in processed.items():
        above_vwap[sym] = (df["close"] > df["vwap_lag"]).astype(float).reindex(idx)
        ema_trend[sym]  = ((df["close"]>df["ema50"]) & (df["ema50"]>df["ema110"])).astype(float).reindex(idx)
        ret1h[sym]      = df["ret1h"].reindex(idx)

    mrc = pd.DataFrame(index=idx)
    mrc["b_vwap"]  = pd.DataFrame(above_vwap).mean(axis=1)
    mrc["b_ema"]   = pd.DataFrame(ema_trend).mean(axis=1)
    mrc["bench1h"] = pd.DataFrame(ret1h).mean(axis=1)

    state = np.ones(len(mrc), dtype=int)
    lockout   = (mrc["b_vwap"]<CFG.mrc_vwap_lockout) | (mrc["b_ema"]<CFG.mrc_ema_lockout) | (mrc["bench1h"]<CFG.mrc_bench_panic)
    expansion = (mrc["b_vwap"]>=CFG.mrc_vwap_expansion) & (mrc["b_ema"]>=CFG.mrc_ema_expansion) & (mrc["bench1h"]>=CFG.mrc_bench_thrust)
    state = np.where(expansion, 2, state)
    state = np.where(lockout, 0, state)
    mrc["mrc"] = state
    return mrc

# ============================================================================
# MAIN SIMULATION ENGINE
# ============================================================================
def run_simulation(
    raw_pool: Dict[str, pd.DataFrame],
    start: str, end: str,
    fold_label: str,
    universe_filter: set = None   # if set, only these clean-name tickers are used
) -> Tuple[float, List[dict]]:

    # Build deduplicated ticker map: prefer bare key over .NS key
    ticker_map = {}  # clean_name -> raw_key_in_pool
    for k in raw_pool.keys():
        clean = k.replace(".NS", "")
        if clean not in ticker_map:
            ticker_map[clean] = k
        elif not k.endswith(".NS"):
            # prefer non-.NS key if both exist
            ticker_map[clean] = k

    # Apply universe filter if provided
    if universe_filter is not None:
        ticker_map = {c: r for c, r in ticker_map.items() if c in universe_filter}

    processed = {}
    for clean_name, raw_key in ticker_map.items():
        df = raw_pool[raw_key]
        df_slice = df.loc[start:end]
        if len(df_slice) < 200:
            continue
        processed[clean_name] = prepare_features(df_slice)

    print(f"  [{fold_label}] {len(processed)} stocks loaded | {start} to {end}")
    mrc_panel = build_mrc_panel(processed)

    all_ts = sorted(mrc_panel.index)
    capital = CFG.initial_capital
    hwm     = capital
    positions = {}   # {ticker: {...}}
    trades = []
    last_exit_ts = set()

    for ts in all_ts:
        if ts.weekday() >= 5:
            continue

        mrc_state = int(mrc_panel.loc[ts, "mrc"])
        bench1h   = float(mrc_panel.loc[ts, "bench1h"])

        # -------- MANAGE OPEN POSITIONS --------
        for ticker in list(positions.keys()):
            p = positions[ticker]
            df_s = processed[ticker]
            if ts not in df_s.index:
                continue

            row   = df_s.loc[ts]
            cur_c = float(row["close"]); cur_h = float(row["high"]); cur_l = float(row["low"])
            cur_ema9  = float(row["ema9"])
            cur_rsi   = float(row["rsi7"])
            cur_rvol  = float(row["rvol20"])
            cur_vwap  = float(row["vwap"])

            p["bars"] += 1
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["raw_entry"]) / max(p["atr"], 1e-6)

            new_sl = p["sl"]

            # Stagnation guard (cut dead chop early)
            if p["bars"] >= CFG.stagnation_bars and mfe_atr < CFG.stagnation_mfe_thresh and cur_c < cur_vwap:
                new_sl = max(new_sl, p["raw_entry"] - CFG.stagnation_sl_atr * p["atr"])

            # Phase 1A: Green lock
            if mfe_atr >= CFG.lock_mfe_atr:
                new_sl = max(new_sl, p["raw_entry"] + CFG.lock_offset_atr * p["atr"])
                p["freeroll"] = True

            # Phase 1B: EMA9 structural channel (WIDE breathing room)
            if mfe_atr >= CFG.phase1b_mfe_atr:
                if cur_c >= cur_ema9:
                    struct_sl = max(cur_ema9 - 0.30*p["atr"], p["high_seen"] - 0.75*p["atr"])
                else:
                    struct_sl = p["high_seen"] - 0.28*p["atr"]
                new_sl = max(new_sl, struct_sl)

            # Phase 2: Parabolic
            if mfe_atr >= CFG.phase2_mfe_atr:
                bar_rng = max(cur_h - cur_l, 1e-6)
                upper_wick = (cur_h - cur_c) / bar_rng
                is_climax = (cur_rvol >= 2.80 and upper_wick > 0.45) or (cur_rsi >= 84.0)
                offset = 0.14 if is_climax else 0.35
                new_sl = max(new_sl, p["high_seen"] - offset*p["atr"])

            # Macro climax
            if mfe_atr >= CFG.climax_mfe_atr:
                new_sl = max(new_sl, p["high_seen"] - 0.10*p["atr"])

            p["sl"] = round(new_sl, 2)

            # Exit check
            exit_fill, exit_reason = None, None
            if cur_l <= p["sl"]:
                exit_fill  = min(cur_c, p["sl"]) * (1 - CFG.slippage)
                exit_reason = "TRAIL_SL"
            elif ts.hour == 15 and ts.minute >= 10:
                exit_fill  = cur_c * (1 - CFG.slippage)
                exit_reason = "EOD_1510"

            if exit_fill is not None:
                qty      = p["qty"]
                gross    = round((exit_fill - p["entry"]) * qty, 4)
                charges  = round(calc_dhan_charges(p["entry"], exit_fill, qty, is_intraday=True), 4)
                net      = round(gross - charges, 4)
                capital += net
                hwm      = max(hwm, capital)
                last_exit_ts.add(ts)
                trades.append({
                    "symbol":       ticker,
                    "direction":    "LONG",
                    "entry_time":   p["entry_t"].strftime("%Y-%m-%d %H:%M:%S"),
                    "exit_time":    ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "entry_price":  round(p["entry"], 4),
                    "exit_price":   round(exit_fill, 4),
                    "qty":          round(qty, 4),
                    "gross_pnl":    gross,
                    "charges":      charges,
                    "net_pnl":      net,
                    "mfe_atr":      round(mfe_atr, 3),
                    "exit_reason":  exit_reason,
                    "mrc_at_entry": p["mrc"],
                    "trigger":      p["trigger"],
                })
                del positions[ticker]

        # -------- ENTRY CHECK --------
        if ts.hour == 9 and ts.minute < 30:
            continue
        if (ts.hour == 14 and ts.minute > 45) or ts.hour >= 15:
            continue
        if ts in last_exit_ts:
            continue
        if mrc_state == 0:
            continue   # LOCKOUT — pure cash immunity

        # Slot eligibility
        n_open = len(positions)
        can_enter = False
        is_freeroll_slot = False
        if n_open == 0:
            can_enter = True
        elif (n_open == 1 and CFG.enable_freeroll and mrc_state == 2
              and list(positions.values())[0].get("freeroll", False)):
            can_enter = True
            is_freeroll_slot = True

        if not can_enter:
            continue

        # Candidate scan
        active_syms = set(positions.keys())
        candidates = []
        for t, df_s in processed.items():
            if t in active_syms or ts not in df_s.index:
                continue
            row = df_s.loc[ts]
            c_val = float(row["close"])
            if np.isnan(c_val) or c_val < CFG.min_price or c_val > CFG.max_price:
                continue
            if pd.isna(row["close300"]) or c_val <= float(row["close300"]):
                continue

            # Trigger check
            trig = ""
            if bool(row["tA"]):
                trig = "A"
            elif mrc_state == 2 and bool(row["tB"]):
                trig = "B"
            elif mrc_state == 2 and bool(row["tC"]):
                trig = "C"
            if not trig:
                continue

            # Gate 4 alignment
            ok = (
                c_val > float(row["ema20"]) and float(row["ema20"]) > float(row["ema50"])
                and c_val > float(row["ema110"])
                and c_val > float(row["vwap_lag"])
                and bool(row["vwap_rising"])
                and CFG.rsi_min <= float(row["rsi7"]) <= CFG.rsi_max
                and float(row["adx10"]) >= CFG.adx_min
                and float(row["pdi"]) > float(row["mdi"])
                and float(row["er10"]) >= CFG.er_min
                and float(row["rvol20"]) >= CFG.rvol_min
                and float(row["atr_pct"]) >= CFG.atr_pct_min
            )
            if not ok:
                continue

            # Anti-chasing guard
            stretch = (c_val - float(row["ema9"])) / max(float(row["atr14"]), 1e-6)
            if stretch > CFG.max_ema9_stretch_atr:
                continue

            excess1h = float(row["ret1h"]) - bench1h
            if excess1h <= 0:
                continue

            alpha = excess1h * np.sqrt(float(row["er10"])) * min(2.5, np.sqrt(float(row["rvol20"])))
            candidates.append((alpha, t, row, trig))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, best_t, best_row, best_trig = candidates[0]

        raw_c   = float(best_row["close"])
        atr_val = float(best_row["atr14"])
        fill    = round(raw_c * (1 + CFG.slippage), 4)
        sl_dist_pct = (CFG.init_sl_atr * atr_val) / raw_c

        cur_dd = (hwm - capital) / max(hwm, 1e-6)
        if cur_dd >= CFG.dd_throttle_at:
            risk_frac = CFG.risk_dd_throttle; max_lev = 1.0
        elif mrc_state == 2:
            risk_frac = CFG.risk_expansion;   max_lev = CFG.max_lev_expansion
        else:
            risk_frac = CFG.risk_selective;   max_lev = CFG.max_lev_selective

        if is_freeroll_slot:
            max_lev = min(max_lev, 1.0); risk_frac *= 0.65

        eff_lev  = float(np.clip(risk_frac / max(sl_dist_pct, 0.002), 0.50, max_lev))
        notional = capital * eff_lev
        qty      = notional / fill
        init_sl  = round(raw_c - CFG.init_sl_atr * atr_val, 4)

        positions[best_t] = {
            "entry":     fill,
            "raw_entry": raw_c,
            "entry_t":   ts,
            "qty":       qty,
            "atr":       atr_val,
            "sl":        init_sl,
            "high_seen": float(best_row["high"]),
            "bars":      0,
            "freeroll":  False,
            "mrc":       mrc_state,
            "trigger":   best_trig,
        }

    return capital, trades

# ============================================================================
# GATE V2 OUTPUT: results.txt + trades_intraday.csv
# ============================================================================
def save_gate_v2_outputs(fold_label: str, capital: float, trades: List[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    net_pnl   = capital - CFG.initial_capital
    n_trades  = len(trades)
    wins      = [t for t in trades if t["net_pnl"] > 0]
    losses    = [t for t in trades if t["net_pnl"] < 0]
    wr        = 100*len(wins)/n_trades if n_trades else 0
    gross_w   = sum(t["net_pnl"] for t in wins)
    gross_l   = abs(sum(t["net_pnl"] for t in losses))
    pf        = round(gross_w/gross_l, 4) if gross_l > 0 else float("inf")
    avg_w     = round(gross_w/len(wins), 4) if wins else 0
    avg_l     = round(gross_l/len(losses), 4) if losses else 0
    total_chg = round(sum(t["charges"] for t in trades), 4)

    # running equity for max drawdown
    eq = CFG.initial_capital; hwm = eq; max_dd = 0.0
    for t in trades:
        eq += t["net_pnl"]
        hwm = max(hwm, eq)
        max_dd = max(max_dd, (hwm-eq)/max(hwm, 1e-6))

    results_txt = f"""Bhavishya V32 — {fold_label}
{'='*60}

Intraday Results
----------------
Initial Capital : Rs {CFG.initial_capital:.2f}
Final Capital   : Rs {capital:.4f}
Net PnL         : Rs {net_pnl:.4f}
Total Trades    : {n_trades}
Win Rate        : {wr:.2f}%
Profit Factor   : {pf}
Avg Win         : Rs {avg_w}
Avg Loss        : Rs {avg_l}
Max Drawdown    : {max_dd*100:.2f}%
Total Charges   : Rs {total_chg}
Multiplier      : {capital/CFG.initial_capital:.4f}x

Swing Results
-------------
Initial Capital : Rs {CFG.initial_capital:.2f}
Final Capital   : Rs {CFG.initial_capital:.2f}
Net PnL         : Rs 0.00
Total Trades    : 0
"""
    (out_dir / "results.txt").write_text(results_txt)

    fieldnames = [
        "symbol", "direction", "entry_time", "exit_time",
        "entry_price", "exit_price", "qty",
        "gross_pnl", "charges", "net_pnl",
        "mfe_atr", "exit_reason", "mrc_at_entry", "trigger"
    ]
    with open(out_dir / "trades_intraday.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(trades)

    # Empty swing CSV (Gate V2 expects it)
    with open(out_dir / "trades_swing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()

    print(f"  [{fold_label}] Trades={n_trades} | WR={wr:.1f}% | PF={pf} | Net=Rs {net_pnl:.2f} | {capital/CFG.initial_capital:.3f}x | MaxDD={max_dd*100:.2f}%")

# ============================================================================
# MAIN — 3-FOLD EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("BHAVISHYA V32 — 3-FOLD OOS TEST (GATE V2 COMPLIANT)")
    print("=" * 70)

    V32_DIR = ROOT_DIR / "strategies" / "V32"

    # -------------------------------------------------------------------
    # FOLD 1: 2022 Full Year (Bear/War) — 30-stock universe
    # -------------------------------------------------------------------
    print("\n[FOLD 1] Loading 2022 Bear/War data (30-stock universe)...")
    with open(ROOT_DIR / "data" / "2022-23" / "universe_30_15m.pkl", "rb") as f:
        pool_2022 = pickle.load(f)

    cap1, trades1 = run_simulation(pool_2022, "2022-01-01", "2022-12-31", "2022 Full Year")
    save_gate_v2_outputs("2022 Full Year Bear/War", cap1, trades1, V32_DIR / "fold_2022")

    # -------------------------------------------------------------------
    # FOLD 2: 2025 H2 (July-December 2025) — 2025-26 pool filtered to V29 30-stock universe
    # -------------------------------------------------------------------
    print("\n[FOLD 2] Loading 2025 H2 Chop data (2025-26 pool, V29 30-stock filter)...")
    with open(ROOT_DIR / "data" / "2025-26" / "universe_14_15m.pkl", "rb") as f:
        pool_2025 = pickle.load(f)

    cap2, trades2 = run_simulation(pool_2025, "2025-07-01", "2025-12-31", "2025 H2", universe_filter=V29_30_UNIVERSE)
    save_gate_v2_outputs("2025 H2 Chop (July-Dec)", cap2, trades2, V32_DIR / "fold_2025h2")

    # -------------------------------------------------------------------
    # FOLD 3: 2026 Full Year till date — V29 30-stock universe only (from combined pool)
    # -------------------------------------------------------------------
    print("\n[FOLD 3] Loading 2026 Bull data (V29 30-stock universe, filtered from 78+10+10)...")
    with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
        u1 = pickle.load(f)
    with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
        u2 = pickle.load(f)
    with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
        u3 = pickle.load(f)

    pool_v29 = {}
    pool_v29.update(u1); pool_v29.update(u2); pool_v29.update(u3)

    cap3, trades3 = run_simulation(pool_v29, "2026-01-01", "2026-12-31", "2026 Full Year", universe_filter=V29_30_UNIVERSE)
    save_gate_v2_outputs("2026 Full Year Bull", cap3, trades3, V32_DIR / "fold_2026")

    # -------------------------------------------------------------------
    # MASTER SUMMARY
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MASTER 3-FOLD SUMMARY (Base: Rs 500 per fold)")
    print("=" * 70)
    for label, cap, trs in [
        ("2022 Bear/War", cap1, trades1),
        ("2025 H2 Chop",  cap2, trades2),
        ("2026 Bull Run", cap3, trades3),
    ]:
        n = len(trs); net = cap - 500
        wr = 100*len([t for t in trs if t["net_pnl"]>0])/max(n,1)
        print(f"  {label:<20} Trades={n:>3} | WR={wr:>5.1f}% | Net=Rs {net:>+9.2f} | {cap/500:.3f}x")
    print("=" * 70)
    print("\nOutputs saved in strategies/V32/fold_2022/, fold_2025h2/, fold_2026/")
    print("Run Gate V2 on each fold subfolder next.")
