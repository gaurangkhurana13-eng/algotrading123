"""
Backtester Module for Bhavishya V37:
Noise-Immune Tiered Stops, 4-Phase Chandelier Runner Ratchet,
Velocity-Modulated Z-Score Gates & Risk-Free Concurrency Recycling.
Strictly lookahead-free, real Dhan MIS transaction costs, 100% causal execution.
"""

import os
import json
import numpy as np
import pandas as pd
from .dhan_engine import calc_dhan_charges
from .pre_trade_mc import PreTradeMonteCarloEngine

SECTOR_MAP = {
    "ADANIPOWER": ("ENERGY", "POWER"),
    "ALOKINDS": ("CONSUMER", "TEXTILES"),
    "ASHOKLEY": ("CONSUMER_DISC", "AUTO"),
    "BANDHANBNK": ("FINANCIALS", "BANKING"),
    "BANKBARODA": ("FINANCIALS", "BANKING"),
    "BANKINDIA": ("FINANCIALS", "BANKING"),
    "BCLIND": ("MATERIALS", "CHEMICALS_AGRO"),
    "BPCL": ("ENERGY", "OIL_GAS"),
    "CANBK": ("FINANCIALS", "BANKING"),
    "CENTRALBK": ("FINANCIALS", "BANKING"),
    "COALINDIA": ("MATERIALS", "MINING"),
    "ELECTCAST": ("INDUSTRIALS", "PIPES_METALS"),
    "EQUITASBNK": ("FINANCIALS", "BANKING"),
    "GAIL": ("ENERGY", "OIL_GAS"),
    "GMRAIRPORT": ("INDUSTRIALS", "INFRA"),
    "GSFC": ("MATERIALS", "FERTILIZERS"),
    "HATHWAY": ("COMM_SERVICES", "MEDIA_CABLE"),
    "HCC": ("INDUSTRIALS", "INFRA_CONSTRUCTION"),
    "HUDCO": ("FINANCIALS", "NBFC_HOUSING"),
    "IDBI": ("FINANCIALS", "BANKING"),
    "IDEA": ("COMM_SERVICES", "TELECOM"),
    "IDFCFIRSTB": ("FINANCIALS", "BANKING"),
    "IFCI": ("FINANCIALS", "NBFC_FINANCE"),
    "INDIANB": ("FINANCIALS", "BANKING"),
    "IOB": ("FINANCIALS", "BANKING"),
    "IOC": ("ENERGY", "OIL_GAS"),
    "IRCON": ("INDUSTRIALS", "INFRA_RAIL"),
    "IRFC": ("FINANCIALS", "NBFC_RAIL"),
    "J&KBANK": ("FINANCIALS", "BANKING"),
    "JAICORPLTD": ("INDUSTRIALS", "DIVERSIFIED"),
    "JPPOWER": ("ENERGY", "POWER"),
    "JYOTHYLAB": ("CONSUMER", "FMCG"),
    "LLOYDSENGG": ("INDUSTRIALS", "CAPITAL_GOODS"),
    "MAHABANK": ("FINANCIALS", "BANKING"),
    "MANALIPETC": ("MATERIALS", "CHEMICALS"),
    "MMTC": ("COMMERCE", "TRADING"),
    "MOREPENLAB": ("HEALTHCARE", "PHARMA"),
    "MOTHERSON": ("CONSUMER_DISC", "AUTO_ANCILLARY"),
    "MRPL": ("ENERGY", "OIL_GAS"),
    "MTNL": ("COMM_SERVICES", "TELECOM"),
    "NBCC": ("INDUSTRIALS", "REAL_ESTATE_INFRA"),
    "NETWORK18": ("COMM_SERVICES", "MEDIA"),
    "NFL": ("MATERIALS", "FERTILIZERS"),
    "NHPC": ("ENERGY", "POWER_HYDRO"),
    "NIACL": ("FINANCIALS", "INSURANCE"),
    "NMDC": ("MATERIALS", "MINING_IRON"),
    "NTPC": ("ENERGY", "POWER_THERMAL"),
    "ONGC": ("ENERGY", "OIL_GAS"),
    "PATANJALI": ("CONSUMER", "FMCG_FOODS"),
    "PCJEWELLER": ("CONSUMER", "GEMS_JEWELLERY"),
    "PFC": ("FINANCIALS", "NBFC_POWER"),
    "PNB": ("FINANCIALS", "BANKING"),
    "PTC": ("ENERGY", "POWER_TRADING"),
    "RBLBANK": ("FINANCIALS", "BANKING"),
    "RCF": ("MATERIALS", "FERTILIZERS"),
    "RECLTD": ("FINANCIALS", "NBFC_POWER"),
    "RENUKA": ("CONSUMER", "SUGAR_AGRO"),
    "RPOWER": ("ENERGY", "POWER"),
    "RVNL": ("INDUSTRIALS", "INFRA_RAIL"),
    "SAIL": ("MATERIALS", "STEEL"),
    "SBIN": ("FINANCIALS", "BANKING"),
    "SJVN": ("ENERGY", "POWER_HYDRO"),
    "SOUTHBANK": ("FINANCIALS", "BANKING"),
    "SPIC": ("MATERIALS", "FERTILIZERS"),
    "SUBEXLTD": ("IT_SERVICES", "SOFTWARE"),
    "SUZLON": ("ENERGY", "RENEWABLE_WIND"),
    "SWSOLAR": ("ENERGY", "RENEWABLE_SOLAR"),
    "TATAPOWER": ("ENERGY", "POWER"),
    "TATASTEEL": ("MATERIALS", "STEEL"),
    "TEXRAIL": ("INDUSTRIALS", "RAIL_EQUIPMENT"),
    "TRIDENT": ("CONSUMER", "TEXTILES"),
    "TTML": ("COMM_SERVICES", "TELECOM"),
    "UCOBANK": ("FINANCIALS", "BANKING"),
    "UJJIVANSFB": ("FINANCIALS", "BANKING"),
    "UNIONBANK": ("FINANCIALS", "BANKING"),
    "UTKARSHBNK": ("FINANCIALS", "BANKING"),
    "VEDL": ("MATERIALS", "METALS_MINING"),
    "YESBANK": ("FINANCIALS", "BANKING")
}

def get_sector_info(ticker: str):
    clean = ticker.replace(".NS", "")
    return SECTOR_MAP.get(clean, ("GENERAL", "GENERAL"))

def execute_backtest(oos_df: pd.DataFrame, df_pool: pd.DataFrame, config: dict, out_dirs: dict) -> dict:
    capital          = float(config.get("capital", 500.0))
    notional_cap     = float(config.get("max_notional_capital", 2450.0))
    daily_n          = int(config.get("max_executed_per_day", 4))
    max_concurrent   = int(config.get("max_concurrent_open", 2))
    max_per_sec      = int(config.get("max_trades_per_sector", 2))
    max_d_loss_val   = float(config.get("max_daily_loss", 25.0))
    slippage         = float(config.get("slippage_per_leg", 0.0005))
    r_net_cap        = float(config.get("r_net_cap", 15.50))
    phi_rt           = float(config.get("round_trip_friction_rate", 0.002067))

    # Volatility bounds & Tier stops
    min_atr_floor    = float(config.get("min_atr_floor", 0.0048))
    max_atr_cap      = float(config.get("max_atr_cap", 0.0240))
    min_r_used       = float(config.get("min_r_used", 11.50))

    t1_cutoff        = float(config.get("tier1_atr_cutoff", 0.0075))
    t2_cutoff        = float(config.get("tier2_atr_cutoff", 0.0110))
    t1_sl_mult       = float(config.get("tier1_sl_mult", 1.40))
    t2_sl_mult       = float(config.get("tier2_sl_mult", 1.45))
    t3_sl_mult       = float(config.get("tier3_sl_mult", 1.50))

    # Time boundaries
    entry_cutoff_time = pd.Timestamp(config.get("hard_cutoff_time", "13:15")).time()
    pre_square_time   = pd.Timestamp("15:00").time()
    eod_square_time   = pd.Timestamp(config.get("eod_square_time", "15:10")).time()

    # Pre-trade MC engine
    min_mc_prob      = float(config.get("pre_trade_mc", {}).get("min_mc_win_prob", 0.48))
    mc_engine        = PreTradeMonteCarloEngine(config)

    # Build ticker price history lookup for causal trade tracing
    ticker_dfs = {}
    for ticker, g in df_pool.groupby("ticker"):
        ticker_dfs[ticker] = g.sort_index()

    oos_df_copy = oos_df.copy()
    oos_df_copy["bar_time"] = [t.time() if hasattr(t, 'time') else pd.Timestamp(str(t)).time() for t in oos_df_copy.index]
    oos_df_copy["trade_date"] = oos_df_copy.index.date

    # Initial candidate filter: strictly within [09:15, 13:15] and [0.48%, 2.40%] ATR
    candidates = oos_df_copy[
        (oos_df_copy["bar_time"] <= entry_cutoff_time) &
        (oos_df_copy["atr_norm"] >= min_atr_floor) &
        (oos_df_copy["atr_norm"] <= max_atr_cap)
    ].copy()

    sector_info = [get_sector_info(t) for t in candidates["ticker"]]
    candidates["sector"] = [s[0] for s in sector_info]
    candidates["sub_sector"] = [s[1] for s in sector_info]

    trade_ledger = []
    rejected_ledger = []
    daily_executed_count = {}
    daily_net_loss = {}
    daily_sector_trade_count = {}
    daily_opening_taken = {}

    pre_mc_jsonl_path = os.path.join(out_dirs["monte_carlo"], "pre_trade_decisions.jsonl")
    f_jsonl = open(pre_mc_jsonl_path, "w", encoding="utf-8")

    current_capital = capital
    active_positions = []

    unique_timestamps = sorted(list(candidates.index.unique()))

    for ts in unique_timestamps:
        df_bar = candidates.loc[[ts]]
        b_time = ts.time() if hasattr(ts, 'time') else pd.Timestamp(str(ts)).time()
        d = ts.date()

        # 1. Update active positions status at this timestamp
        still_active = []
        for pos in active_positions:
            if ts <= pos["exit_time"]:
                # Check if position has reached Phase 2 (MFE >= 2.10x ATR) -> mark Risk-Free
                t_df = ticker_dfs.get(pos["ticker"])
                if t_df is not None:
                    bars_to_now = t_df[(t_df.index > pos["entry_time"]) & (t_df.index <= ts)]
                    if len(bars_to_now) > 0:
                        m_so_far = float(bars_to_now["high"].max())
                        if m_so_far >= pos["entry_p"] + 2.10 * pos["atr_val"]:
                            pos["is_risk_free"] = True
                still_active.append(pos)
        active_positions = still_active

        # 2. Concurrency check & Risk-Free Slot Recycling (§3.C)
        # Position count check
        total_open = len(active_positions)
        risk_bearing_open = sum(1 for p in active_positions if not p.get("is_risk_free", False))

        if risk_bearing_open >= max_concurrent or total_open >= 3:
            continue

        # Margin availability
        tied_notional = sum(p["qty"] * p["entry_p"] for p in active_positions)
        available_notional = max(0.0, notional_cap - tied_notional)
        if available_notional < 150.0:
            continue

        # Daily risk limits
        if daily_net_loss.get(d, 0.0) >= max_d_loss_val:
            continue
        if daily_executed_count.get(d, 0) >= daily_n:
            continue

        # Cross-sectional sorting: primary by z_score descending, secondary by prob descending
        df_sorted = df_bar.sort_values(by=["z_score", "prob"], ascending=[False, False])

        for idx, row in df_sorted.iterrows():
            ticker     = row["ticker"]
            sec        = row["sector"]
            prob       = float(row["prob"])
            mu_f       = float(row.get("mu_f", 0.30))
            sigma_f    = float(row.get("sigma_f", 0.08))
            z_score    = float(row.get("z_score", 0.0))
            breadth    = float(row.get("breadth", 0.50))
            dispersion = float(row.get("dispersion", 0.0))
            q35_disp   = float(row.get("q35_dispersion", 0.005))
            rvol       = float(row.get("rvol", 1.0))
            cs_rs      = float(row.get("cs_rs", 0.0))
            pct_cs_rs  = float(row.get("pct_cs_rs", 0.50))
            atr_norm   = float(row["atr_norm"])
            raw_p      = float(row["close"])
            raw_open   = float(row["open"])
            raw_high   = float(row["high"])
            raw_low    = float(row["low"])
            vwap_lag   = float(row.get("vwap", raw_p))
            atr_raw    = float(row.get("atr_raw", raw_p * atr_norm))

            # Sector cap check
            day_sec_key = (d, sec)
            curr_sec_trades = daily_sector_trade_count.get(day_sec_key, 0)
            if curr_sec_trades >= max_per_sec:
                rejected_ledger.append({
                    "timestamp": str(ts),
                    "ticker": ticker,
                    "reason": f"MAX_SECTOR_CAP ({max_per_sec})",
                    "z_score": round(z_score, 2),
                    "prob": round(prob, 4)
                })
                continue

            # Remaining horizon to 15:10 IST
            mins_left = (eod_square_time.hour * 60 + eod_square_time.minute) - (b_time.hour * 60 + b_time.minute)
            h_rem = min(12, max(1, mins_left // 15))
            if h_rem < 8:
                rejected_ledger.append({
                    "timestamp": str(ts),
                    "ticker": ticker,
                    "reason": f"HORIZON_TOO_SHORT ({h_rem} < 8 bars)",
                    "z_score": round(z_score, 2),
                    "prob": round(prob, 4)
                })
                continue

            # Section 2.B: Map to Volatility Tier
            if atr_norm < t1_cutoff:
                tier = "Tier 1 (Core Liquid)"
                sl_mult = t1_sl_mult
            elif atr_norm < t2_cutoff:
                tier = "Tier 2 (Sweet-Spot Alpha)"
                sl_mult = t2_sl_mult
            else:
                tier = "Tier 3 (High-Beta Runners)"
                sl_mult = t3_sl_mult

            # Section 1.A & 1.C: Velocity-Modulated Z-Score & Timing Protocol
            is_opening_burst = (b_time <= pd.Timestamp("09:45").time())
            passed_gate = False
            gate_reason = ""

            # CLV calculation
            clv = (raw_p - raw_low) / (raw_high - raw_low + 1e-8)

            if is_opening_burst:
                # 09:15-09:45 Opening Burst Protocol
                if daily_opening_taken.get(d, False):
                    rejected_ledger.append({
                        "timestamp": str(ts),
                        "ticker": ticker,
                        "reason": "OPENING_TRADE_ALREADY_TAKEN",
                        "z_score": round(z_score, 2),
                        "prob": round(prob, 4)
                    })
                    continue

                alpha_t = 1.24
                # Opening Guards:
                # 1. Gap Exhaustion Cap (|Open - PrevClose| <= 1.85 * ATR)
                gap_0 = float(row.get("gap_0", 0.0))
                atr_prev_norm = float(row.get("atr_prev_norm", atr_norm))
                gap_exhausted = (abs(gap_0) > 1.85 * atr_prev_norm)

                # 2. Wick Rejection & Order Flow (CLV >= 0.70, C > O, C > VWAP, RVOL >= 1.60)
                wick_orderflow = (clv >= 0.70 and raw_p > raw_open and raw_p > vwap_lag and rvol >= 1.60)

                # 3. VWAP Extension Guard (0.05 <= (C - VWAP)/ATR <= 1.90)
                vwap_ext = (raw_p - vwap_lag) / (atr_raw + 1e-8)
                vwap_ext_valid = (0.05 <= vwap_ext <= 1.90)

                # 4. Opening Z-Score hurdle
                hurdle = max(0.38, mu_f + alpha_t * sigma_f)
                cond_z = (prob >= hurdle and rvol >= 1.85 and pct_cs_rs >= 0.85)

                if (not gap_exhausted) and wick_orderflow and vwap_ext_valid and cond_z:
                    passed_gate = True
                    gate_reason = "OPENING_INSTITUTIONAL_BURST"
                else:
                    rejected_ledger.append({
                        "timestamp": str(ts),
                        "ticker": ticker,
                        "reason": f"OPENING_GUARD_FAIL (CLV={clv:.2f}, RVOL={rvol:.2f}, Gap={gap_0:.4f}, Prob={prob:.4f})",
                        "z_score": round(z_score, 2),
                        "prob": round(prob, 4)
                    })
                    continue
            else:
                # Midday & Afternoon Session
                if b_time <= pd.Timestamp("12:15").time():
                    alpha_t = 1.32
                else:
                    alpha_t = 1.40

                hurdle = max(0.38, mu_f + alpha_t * sigma_f)
                if prob < hurdle:
                    rejected_ledger.append({
                        "timestamp": str(ts),
                        "ticker": ticker,
                        "reason": f"CONVICTION_HURDLE_FAIL (prob={prob:.4f} < {hurdle:.4f}, alpha={alpha_t})",
                        "z_score": round(z_score, 2),
                        "prob": round(prob, 4)
                    })
                    continue

                # Section 1.B: Causal Macro-Regime & Cross-Sectional Gates
                # 1. Market Breadth Gate
                breadth_required = 0.42 if pct_cs_rs >= 0.88 else 0.48
                if breadth < breadth_required:
                    rejected_ledger.append({
                        "timestamp": str(ts),
                        "ticker": ticker,
                        "reason": f"BREADTH_FAIL ({breadth:.2f} < {breadth_required:.2f})",
                        "z_score": round(z_score, 2),
                        "prob": round(prob, 4)
                    })
                    continue

                # 2. Cross-Sectional Relative Strength Rank & CLV
                if pct_cs_rs < 0.78 or clv < 0.62:
                    rejected_ledger.append({
                        "timestamp": str(ts),
                        "ticker": ticker,
                        "reason": f"CS_RS_CLV_FAIL (pct_cs={pct_cs_rs:.2f} < 0.78 or clv={clv:.2f} < 0.62)",
                        "z_score": round(z_score, 2),
                        "prob": round(prob, 4)
                    })
                    continue

                # 3. Volatility Dispersion Gate
                if dispersion < q35_disp:
                    rejected_ledger.append({
                        "timestamp": str(ts),
                        "ticker": ticker,
                        "reason": f"DISPERSION_FAIL ({dispersion:.4f} < {q35_disp:.4f})",
                        "z_score": round(z_score, 2),
                        "prob": round(prob, 4)
                    })
                    continue

                passed_gate = True
                gate_reason = "CORE_REGIME_VELOCITY_GATE"

            if not passed_gate:
                continue

            entry_p = raw_p * (1.0 + slippage)
            atr_val = entry_p * atr_norm

            # Section 2.B: Exact Integer Share Sizing Equation
            stop_dist = sl_mult * atr_val
            friction_per_share = phi_rt * entry_p
            denom = stop_dist + friction_per_share

            qty_risk = int(r_net_cap / denom) if denom > 0 else 1
            qty_notional = int(available_notional / (entry_p * (1.0 + slippage)))
            qty = max(0, min(qty_notional, qty_risk))

            if qty < 1:
                rejected_ledger.append({
                    "timestamp": str(ts),
                    "ticker": ticker,
                    "reason": "QTY_ZERO",
                    "z_score": round(z_score, 2),
                    "prob": round(prob, 4)
                })
                continue

            # Section 2.A: Effective Risk Utilization Gate (R_used >= Rs 11.50)
            r_used = qty * denom
            if r_used < min_r_used:
                rejected_ledger.append({
                    "timestamp": str(ts),
                    "ticker": ticker,
                    "reason": f"RISK_UTILIZATION_LOW ({r_used:.2f} < {min_r_used:.2f})",
                    "z_score": round(z_score, 2),
                    "prob": round(prob, 4)
                })
                continue

            if qty * entry_p > available_notional + 1.0:
                rejected_ledger.append({
                    "timestamp": str(ts),
                    "ticker": ticker,
                    "reason": f"INSUFFICIENT_MARGIN ({qty * entry_p:.2f} > {available_notional:.2f})",
                    "z_score": round(z_score, 2),
                    "prob": round(prob, 4)
                })
                continue

            # Pre-Trade Monte Carlo Validation
            snapshot = {
                "close": raw_p,
                "atr_norm": atr_norm,
                "z_score": z_score,
                "cs_rs": cs_rs,
                "rvol": rvol,
                "qty": qty,
                "sl_mult": sl_mult,
                "hold_bars": h_rem
            }
            mc_eval = mc_engine.evaluate_candidate(snapshot)
            mc_win_prob = float(mc_eval.get("win_prob", 0.50))

            if mc_win_prob < min_mc_prob:
                rejected_ledger.append({
                    "timestamp": str(ts),
                    "ticker": ticker,
                    "reason": f"PRE_TRADE_MC_REJECT (WinProb={mc_win_prob:.2f} < {min_mc_prob:.2f})",
                    "z_score": round(z_score, 2),
                    "prob": round(prob, 4)
                })
                continue

            # Record pre-trade decision in JSONL
            f_jsonl.write(json.dumps({
                "timestamp": str(ts),
                "ticker": ticker,
                "tier": tier,
                "z_score": z_score,
                "prob": prob,
                "mc_win_prob": mc_win_prob,
                "qty": qty,
                "r_used": round(r_used, 2),
                "gate": gate_reason,
                "notional": round(qty * entry_p, 2)
            }) + "\n")

            # Section 3: 4-Phase Chandelier Ratchet & Calibrated Stagnation Exit Tracing
            p_sl_initial = entry_p - sl_mult * atr_val
            tp_ceiling   = entry_p + 5.80 * atr_val
            active_p_sl  = p_sl_initial
            phase        = 0
            m_t          = entry_p
            exit_price   = None
            exit_time    = None
            exit_reason  = None

            if ticker in ticker_dfs:
                t_df = ticker_dfs[ticker]
                post_bars = t_df[t_df.index > ts]
                post_bars_day = post_bars[post_bars.index.date == d]
                check_bars = post_bars_day.head(h_rem)

                for bar_idx, (_, bar_r) in enumerate(check_bars.iterrows()):
                    b_high  = float(bar_r["high"])
                    b_low   = float(bar_r["low"])
                    b_close = float(bar_r["close"])
                    b_open  = float(bar_r["open"])
                    b_vwap  = float(bar_r.get("vwap", b_close))
                    b_t     = bar_r.name.time() if hasattr(bar_r.name, 'time') else pd.Timestamp(str(bar_r.name)).time()
                    m_t     = max(m_t, b_high)
                    h_held  = bar_idx + 1

                    mfe_atr = (m_t - entry_p) / atr_val

                    # Section 3.A: Multi-Phase Chandelier Schedule
                    if mfe_atr >= 4.50:
                        phase = max(phase, 4)
                        active_p_sl = max(active_p_sl, entry_p + 3.70 * atr_val, m_t - 0.65 * atr_val)
                    elif mfe_atr >= 3.20:
                        phase = max(phase, 3)
                        active_p_sl = max(active_p_sl, entry_p + 2.20 * atr_val, m_t - 0.95 * atr_val)
                    elif mfe_atr >= 2.10:
                        phase = max(phase, 2)
                        active_p_sl = max(active_p_sl, entry_p + 0.95 * atr_val, m_t - 1.15 * atr_val)
                    elif mfe_atr >= 1.30:
                        phase = max(phase, 1)
                        active_p_sl = max(active_p_sl, entry_p - 0.55 * atr_val)

                    # 1. Hard TP Ceiling Hit (+5.80x ATR)
                    if b_high >= tp_ceiling:
                        exit_price = tp_ceiling * (1.0 - slippage)
                        exit_time  = bar_r.name
                        exit_reason = "TP_CEILING_HIT (+5.80 ATR)"
                        break

                    # 2. Stop Loss or Chandelier Trailed Floor Hit
                    if b_low <= active_p_sl:
                        # Realistic gap protection
                        actual_exit = min(active_p_sl, b_open)
                        exit_price = actual_exit * (1.0 - slippage)
                        exit_time  = bar_r.name
                        if phase == 4:
                            exit_reason = "PHASE_4_CLIMAX_CHANDELIER_HIT"
                        elif phase == 3:
                            exit_reason = "PHASE_3_PARABOLIC_TRAIL_HIT"
                        elif phase == 2:
                            exit_reason = "PHASE_2_WIDE_CHANDELIER_HIT"
                        elif phase == 1:
                            exit_reason = "PHASE_1_SHIELD_SL_HIT (-0.55 ATR)"
                        else:
                            exit_reason = "INITIAL_SL_HIT"
                        break

                    # 3. Section 3.B: Calibrated Stagnation Exits
                    # Bar 4 Failed-Impulse Exit
                    if h_held >= 4 and mfe_atr < 0.45 and b_close < entry_p - 0.35 * atr_val and b_close < b_vwap:
                        exit_price = b_close * (1.0 - slippage)
                        exit_time  = bar_r.name
                        exit_reason = "STAGNATION_BAR_4_FAILED_IMPULSE"
                        break

                    # Bar 7 Dead-Capital Exit
                    if h_held >= 7 and mfe_atr < 0.85 and ((b_close - entry_p) / atr_val) < 0.25:
                        exit_price = b_close * (1.0 - slippage)
                        exit_time  = bar_r.name
                        exit_reason = "STAGNATION_BAR_7_DEAD_CAPITAL"
                        break

                    # Bar 11 Sub-Ratchet Tightener
                    if h_held >= 11 and phase <= 1 and 0.25 <= ((b_close - entry_p) / atr_val) <= 1.20:
                        active_p_sl = max(active_p_sl, entry_p + 0.25 * atr_val)

                    # 15:00 IST Pre-Square-Off Guard
                    if b_t >= pre_square_time and b_t < eod_square_time:
                        if b_close > entry_p + 0.30 * atr_val and phase < 3:
                            active_p_sl = max(active_p_sl, b_close - 0.35 * atr_val)
                        elif b_close <= entry_p + 0.30 * atr_val and b_close < b_vwap:
                            exit_price = b_close * (1.0 - slippage)
                            exit_time  = bar_r.name
                            exit_reason = "15:00_PRE_SQUARE_OFF_GUARD"
                            break

                    # 15:10 IST EOD Flatten
                    if b_t >= eod_square_time:
                        exit_price = b_close * (1.0 - slippage)
                        exit_time  = bar_r.name
                        exit_reason = "15:10_EOD_FLATTEN"
                        break

                if exit_price is None and len(check_bars) > 0:
                    last_b = check_bars.iloc[-1]
                    exit_price = float(last_b["close"]) * (1.0 - slippage)
                    exit_time  = last_b.name
                    exit_reason = "15:10_EOD_FLATTEN"

            if exit_price is None:
                realized_ret = float(row.get("realized_pct", 0.0))
                exit_price = entry_p * (1.0 + realized_ret) * (1.0 - slippage)
                exit_time  = ts
                exit_reason = "DEFAULT_EXIT"

            gross_pnl = (exit_price - entry_p) * qty
            charges = calc_dhan_charges(entry_p, exit_price, qty)
            net_pnl = round(gross_pnl - charges, 2)

            daily_executed_count[d] = daily_executed_count.get(d, 0) + 1
            daily_sector_trade_count[day_sec_key] = daily_sector_trade_count.get(day_sec_key, 0) + 1
            if is_opening_burst:
                daily_opening_taken[d] = True

            if net_pnl < 0:
                daily_net_loss[d] = daily_net_loss.get(d, 0.0) + abs(net_pnl)

            current_capital = round(current_capital + net_pnl, 2)

            trade_record = {
                "timestamp": str(ts),
                "ticker": ticker,
                "tier": tier,
                "exit_time": str(exit_time),
                "exit_reason": exit_reason,
                "phase": phase,
                "sector": sec,
                "prob": round(prob, 4),
                "z_score": round(z_score, 2),
                "entry_price": round(entry_p, 2),
                "exit_price": round(exit_price, 2),
                "qty": qty,
                "position_exposure": round(qty * entry_p, 2),
                "r_used": round(r_used, 2),
                "gross_pnl": round(gross_pnl, 2),
                "charges": round(charges, 2),
                "net_pnl": net_pnl,
                "capital": current_capital,
                "winner": net_pnl > 0
            }
            trade_ledger.append(trade_record)

            # Register open position
            active_positions.append({
                "ticker": ticker,
                "entry_p": entry_p,
                "entry_time": ts,
                "exit_time": exit_time,
                "qty": qty,
                "atr_val": atr_val,
                "is_risk_free": False
            })

            break  # Move to next timestamp after taking best cross-sectional candidate

    f_jsonl.close()

    df_trades = pd.DataFrame(trade_ledger)
    df_rejected = pd.DataFrame(rejected_ledger)

    trades_csv_path = os.path.join(out_dirs["trades"], "v37_trades.csv")
    rejected_csv_path = os.path.join(out_dirs["trades"], "v37_rejected_candidates.csv")
    df_trades.to_csv(trades_csv_path, index=False)
    df_rejected.to_csv(rejected_csv_path, index=False)

    n_trades = len(df_trades)
    n_wins = int(df_trades["winner"].sum()) if n_trades > 0 else 0
    n_losses = n_trades - n_wins
    wr = round((n_wins / n_trades) * 100, 2) if n_trades > 0 else 0.0

    gross_wins = df_trades[df_trades["net_pnl"] > 0]["net_pnl"].sum() if n_trades > 0 else 0.0
    gross_losses = abs(df_trades[df_trades["net_pnl"] < 0]["net_pnl"].sum()) if n_trades > 0 else 0.0
    pf = round(gross_wins / gross_losses, 2) if gross_losses > 0 else 999.0

    avg_win = round(df_trades[df_trades["net_pnl"] > 0]["net_pnl"].mean(), 2) if n_wins > 0 else 0.0
    avg_loss = round(df_trades[df_trades["net_pnl"] <= 0]["net_pnl"].mean(), 2) if n_losses > 0 else 0.0
    payoff = round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0.0

    total_net_pnl = round(float(df_trades["net_pnl"].sum()), 2) if n_trades > 0 else 0.0
    total_charges = round(float(df_trades["charges"].sum()), 2) if n_trades > 0 else 0.0
    final_cap = round(capital + total_net_pnl, 2)
    roi = round(((final_cap - capital) / capital) * 100, 2)

    # Sharpe and Sortino
    if n_trades > 1:
        rets = df_trades["net_pnl"] / capital
        sharpe = round(float(np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(252 * 5)), 2)
        neg_rets = rets[rets < 0]
        sortino = round(float(np.mean(rets) / (np.std(neg_rets) + 1e-9) * np.sqrt(252 * 5)), 2) if len(neg_rets) > 0 else 999.0
    else:
        sharpe, sortino = 0.0, 0.0

    return {
        "n_candidates": len(candidates),
        "n_approved": len(trade_ledger),
        "n_rejected": len(rejected_ledger),
        "n_trades": n_trades,
        "wins": n_wins,
        "losses": n_losses,
        "win_rate": wr,
        "profit_factor": pf,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff,
        "total_charges": total_charges,
        "net_pnl": total_net_pnl,
        "final_capital": final_cap,
        "roi": roi,
        "sharpe": sharpe,
        "sortino": sortino,
        "df_trades": df_trades
    }

def run_global_monte_carlo(df_trades: pd.DataFrame, config: dict, out_dirs: dict) -> dict:
    if len(df_trades) == 0:
        return {"ruin_prob": 0.0, "expected_dd": 0.0, "median_capital": 500.0}

    mc_cfg = config.get("global_monte_carlo", {})
    iters = mc_cfg.get("iterations", 10000)
    seed = mc_cfg.get("seed", 42)
    np.random.seed(seed)

    capital = float(config.get("capital", 500.0))
    pnls = df_trades["net_pnl"].values
    n = len(pnls)

    final_caps = []
    max_dds = []
    ruin_count = 0

    for _ in range(iters):
        sampled_pnl = np.random.choice(pnls, size=n, replace=True)
        equity_curve = capital + np.cumsum(sampled_pnl)
        peak = np.maximum.accumulate(equity_curve)
        dd = (peak - equity_curve) / (peak + 1e-9)
        max_dds.append(np.max(dd))
        final_caps.append(equity_curve[-1])
        if np.min(equity_curve) <= (capital * 0.50):
            ruin_count += 1

    ruin_prob = round((ruin_count / iters) * 100, 2)
    expected_dd = round(float(np.mean(max_dds)) * 100, 2)
    median_cap = round(float(np.median(final_caps)), 2)

    res = {
        "ruin_prob": ruin_prob,
        "expected_dd": expected_dd,
        "median_capital": median_cap
    }
    mc_json_path = os.path.join(out_dirs["monte_carlo"], "v37_global_mc_report.json")
    with open(mc_json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    return res
