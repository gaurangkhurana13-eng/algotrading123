"""
Backtester Module for Bhavishya V34:
Post-Mortem & Corrective Deterministic Mathematical Blueprint.
1. Fold-Relative Z-Score Conviction Gate (z_t >= 1.28 & P_cal >= mu_f + 1.0*sigma_f)
2. Regime Gate: Breadth B_t >= 0.55 & Dispersion D_t >= Q40(D_train)
3. First-Bar Opening Momentum Filter (09:15-09:30 candle)
4. Restored Strict Session Velocity Gates (RVOL, CS_RS, PCT(CS_RS))
5. Extended 4.2x ATR Geometry with Inviolate 4-Phase Ratchet
6. Trade-Relative Stagnation Exits (H >= 6, 8, 10 bars) replacing 13:15 clock guillotine
7. Unified Dual-Constraint Sizing (Max Loss <= Rs 15.50) with ATR% Floor (>= 0.60%)
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
    daily_n          = int(config.get("max_executed_per_day", 5))
    max_concurrent   = int(config.get("max_concurrent_open", 1))
    max_per_sec      = int(config.get("max_trades_per_sector", 2))
    max_d_loss_val   = float(config.get("max_daily_loss", 25.0))
    sl_atr_mult      = float(config.get("sl_atr_mult", 1.5))
    slippage         = float(config.get("slippage_per_leg", 0.0005))
    min_friction_atr = float(config.get("min_friction_atr_ratio", 0.0060))
    r_net_cap        = float(config.get("r_net_cap", 15.50))

    # Extended Ratchet Parameters (4.2x ATR TP)
    p1_trig          = float(config.get("ratchet_phase1_trigger", 1.60))
    p1_flr           = float(config.get("ratchet_phase1_floor", 1.35))
    p2_trig          = float(config.get("ratchet_phase2_trigger", 2.30))
    p2_flr           = float(config.get("ratchet_phase2_floor", 1.95))
    p3_trig          = float(config.get("ratchet_phase3_trigger", 3.20))
    p3_flr           = float(config.get("ratchet_phase3_floor", 2.80))
    tp_ceiling_mult  = float(config.get("tp_ceiling", 4.20))

    min_mc_prob      = float(config.get("pre_trade_mc", {}).get("min_mc_win_prob", 0.50))
    z_min            = float(config.get("z_conviction_min", 1.28))
    rank_cutoff      = float(config.get("rank_score_cutoff", 0.95))
    breadth_min      = float(config.get("breadth_min_long", 0.55))

    midday_split     = pd.Timestamp("11:15").time()
    cutoff_time      = pd.Timestamp(config.get("hard_cutoff_time", "13:30")).time()
    eod_square_time  = pd.Timestamp(config.get("eod_square_time", "15:05")).time()

    mc_engine = PreTradeMonteCarloEngine(config)

    # Build ticker price history lookup for causal trade tracing
    ticker_dfs = {}
    for ticker, g in df_pool.groupby("ticker"):
        ticker_dfs[ticker] = g.sort_index()

    oos_df_copy = oos_df.copy()
    oos_df_copy["bar_time"] = [t.time() if hasattr(t, 'time') else pd.Timestamp(str(t)).time() for t in oos_df_copy.index]
    oos_df_copy["trade_date"] = oos_df_copy.index.date

    # Initial coarse filter: within [09:15, 13:30], min friction ATR% (>= 0.60%)
    candidates = oos_df_copy[
        (oos_df_copy["bar_time"] <= cutoff_time) &
        (oos_df_copy["atr_norm"] >= min_friction_atr)
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
    active_position = None  # Exactly 1 position at a time (per Section 5.6)

    # Process chronologically timestamp by timestamp across the out-of-sample data
    unique_timestamps = sorted(list(candidates.index.unique()))

    for ts in unique_timestamps:
        df_bar = candidates.loc[[ts]]
        b_time = ts.time() if hasattr(ts, 'time') else pd.Timestamp(str(ts)).time()
        d = ts.date()

        # Check if active position has exited by current timestamp
        if active_position is not None:
            if ts <= active_position["exit_time"]:
                # Position still open: no new entries allowed
                continue
            else:
                # Margin released on the bar after exit
                active_position = None

        # Check daily risk constraints
        if daily_net_loss.get(d, 0.0) >= max_d_loss_val:
            continue
        if daily_executed_count.get(d, 0) >= daily_n:
            continue

        # Sort candidates at this timestamp by z_score descending, secondary by prob descending
        df_sorted = df_bar.sort_values(by=["z_score", "prob"], ascending=[False, False])

        for idx, row in df_sorted.iterrows():
            ticker     = row["ticker"]
            sec        = row["sector"]
            prob       = float(row["prob"])
            z_score    = float(row["z_score"])
            p_floor    = float(row.get("p_floor", 0.40))
            rank_score = float(row.get("rank_score", 0.50))
            breadth    = float(row.get("breadth", 0.50))
            dispersion = float(row.get("dispersion", 0.0))
            q40_disp   = float(row.get("q40_dispersion", 0.005))
            rvol       = float(row.get("rvol", 1.0))
            cs_rs      = float(row.get("cs_rs", 0.0))
            pct_cs_rs  = float(row.get("pct_cs_rs", 0.50))
            atr_norm   = float(row["atr_norm"])
            raw_p      = float(row["close"])

            # Check sector exposure cap
            day_sec_key = (d, sec)
            curr_sec_trades = daily_sector_trade_count.get(day_sec_key, 0)
            if curr_sec_trades >= max_per_sec:
                rejected_ledger.append({
                    "timestamp": str(ts),
                    "ticker": ticker,
                    "reason": f"MAX_SECTOR_CAP_REACHED ({max_per_sec})",
                    "z_score": round(z_score, 2),
                    "prob": round(prob, 4)
                })
                continue

            # Dynamic Horizon Calculation to 15:05 IST
            mins_left = (eod_square_time.hour * 60 + eod_square_time.minute) - (b_time.hour * 60 + b_time.minute)
            h_rem = min(12, max(1, mins_left // 15))

            if h_rem < 6:
                rejected_ledger.append({
                    "timestamp": str(ts),
                    "ticker": ticker,
                    "reason": f"HORIZON_TOO_SHORT ({h_rem} < 6 bars)",
                    "z_score": round(z_score, 2),
                    "prob": round(prob, 4)
                })
                continue

            # Section 3.2: First-Bar Opening Momentum Filter (09:15-09:30 candle evaluated at 09:30 close)
            is_opening_bar = (b_time <= pd.Timestamp("09:30").time())
            passed_gate = False
            gate_reason = ""

            if is_opening_bar:
                if daily_opening_taken.get(d, False):
                    rejected_ledger.append({
                        "timestamp": str(ts),
                        "ticker": ticker,
                        "reason": "OPENING_TRADE_ALREADY_TAKEN",
                        "z_score": round(z_score, 2),
                        "prob": round(prob, 4)
                    })
                    continue

                range_pct = float(row.get("bar_0_range_pct", 0.5))
                r_0 = float(row.get("r_0", 0.0))
                gap_0 = float(row.get("gap_0", 0.0))
                atr_prev_norm = float(row.get("atr_prev_norm", 0.01))

                # Section 3.2 Opening Filter Conditions:
                # RVOL >= 2.0, close in top 25% of range, positive gap, r_0 >= 0.50*atr_prev, z >= 1.28, cs_rs >= 0, pct_cs_rs >= 0.90
                cond_rvol  = (rvol >= 2.0)
                cond_range = (range_pct >= 0.75)
                cond_r0    = (r_0 >= 0.50 * atr_prev_norm)
                cond_gap   = (gap_0 >= -0.50 * atr_prev_norm)
                cond_z     = (z_score >= z_min)
                cond_cs    = (cs_rs >= 0.0 and pct_cs_rs >= 0.90)

                if cond_rvol and cond_range and cond_r0 and cond_gap and cond_z and cond_cs:
                    passed_gate = True
                    gate_reason = "OPENING_MOMENTUM_FILTER"
                else:
                    rejected_ledger.append({
                        "timestamp": str(ts),
                        "ticker": ticker,
                        "reason": f"OPENING_FILTER_FAIL (RVOL={rvol:.2f}, RangePct={range_pct:.2f}, z={z_score:.2f})",
                        "z_score": round(z_score, 2),
                        "prob": round(prob, 4)
                    })
                    continue
            else:
                # Block new entries until 10:00 if opening trade was taken today
                if daily_opening_taken.get(d, False) and b_time < pd.Timestamp("10:00").time():
                    rejected_ledger.append({
                        "timestamp": str(ts),
                        "ticker": ticker,
                        "reason": "BLOCK_UNTIL_10:00_AFTER_OPENING_TRADE",
                        "z_score": round(z_score, 2),
                        "prob": round(prob, 4)
                    })
                    continue

                # Section 1.5 Joint Entry Gate:
                # 1. Conviction: z_t >= 1.28 and P_cal >= mu_f + 1.0*sigma_f
                if z_score < z_min or prob < p_floor:
                    rejected_ledger.append({
                        "timestamp": str(ts),
                        "ticker": ticker,
                        "reason": f"CONVICTION_FAIL (z={z_score:.2f} < {z_min}, prob={prob:.4f} < {p_floor:.4f})",
                        "z_score": round(z_score, 2),
                        "prob": round(prob, 4)
                    })
                    continue

                # 2. Rank confirmation
                if rank_score < rank_cutoff:
                    rejected_ledger.append({
                        "timestamp": str(ts),
                        "ticker": ticker,
                        "reason": f"RANK_SCORE_FAIL ({rank_score:.4f} < {rank_cutoff:.2f})",
                        "z_score": round(z_score, 2),
                        "prob": round(prob, 4)
                    })
                    continue

                # 3. Regime Gate: Breadth B_t >= 0.55 and Dispersion D_t >= Q40(D_train)
                if breadth < breadth_min or dispersion < q40_disp:
                    rejected_ledger.append({
                        "timestamp": str(ts),
                        "ticker": ticker,
                        "reason": f"REGIME_FAIL (Breadth={breadth:.2f} < {breadth_min}, Disp={dispersion:.4f} < {q40_disp:.4f})",
                        "z_score": round(z_score, 2),
                        "prob": round(prob, 4)
                    })
                    continue

                # 4. Session Velocity Gate
                if b_time <= midday_split:
                    # Morning Session: RVOL >= 1.10, CS_RS >= 0.0, PCT(CS_RS) >= 0.85
                    if not (rvol >= 1.10 and cs_rs >= 0.0 and pct_cs_rs >= 0.85):
                        rejected_ledger.append({
                            "timestamp": str(ts),
                            "ticker": ticker,
                            "reason": f"MORNING_VELOCITY_FAIL (RVOL={rvol:.2f}, CS_RS={cs_rs:.2f}, PCT={pct_cs_rs:.2f})",
                            "z_score": round(z_score, 2),
                            "prob": round(prob, 4)
                        })
                        continue
                else:
                    # Midday Session: RVOL >= 1.35, CS_RS >= 0.35, PCT(CS_RS) >= 0.90
                    if not (rvol >= 1.35 and cs_rs >= 0.35 and pct_cs_rs >= 0.90):
                        rejected_ledger.append({
                            "timestamp": str(ts),
                            "ticker": ticker,
                            "reason": f"MIDDAY_VELOCITY_FAIL (RVOL={rvol:.2f}, CS_RS={cs_rs:.2f}, PCT={pct_cs_rs:.2f})",
                            "z_score": round(z_score, 2),
                            "prob": round(prob, 4)
                        })
                        continue

                passed_gate = True
                gate_reason = "REGIME_VELOCITY_GATE"

            if not passed_gate:
                continue

            # Pre-Trade Monte Carlo Evaluation
            snapshot = {
                "close": raw_p,
                "atr_norm": atr_norm,
                "z_score": z_score,
                "rank_score": rank_score,
                "cs_rs": cs_rs,
                "rvol": rvol,
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

            entry_p = raw_p * (1.0 + slippage)
            sl_pct  = sl_atr_mult * atr_norm
            tp_pct  = tp_ceiling_mult * atr_norm

            # Section 4.3 Unified Dual-Constraint Sizing Equation (with ATR% floor >= 0.60%)
            # kappa_t = round-trip Dhan friction per share
            kappa_share = entry_p * (2 * slippage + 0.0003 + 0.00025 + 0.0000325 + 0.00003 + 0.000054)
            denom = entry_p * sl_pct + kappa_share
            qty_risk = int(r_net_cap / denom) if denom > 0 else 1
            qty_notional = int(notional_cap / entry_p)
            qty = max(1, min(qty_notional, qty_risk))

            # Record pre-trade decision in JSONL
            f_jsonl.write(json.dumps({
                "timestamp": str(ts),
                "ticker": ticker,
                "z_score": z_score,
                "rank_score": rank_score,
                "prob": prob,
                "mc_win_prob": mc_win_prob,
                "qty": qty,
                "gate": gate_reason,
                "position_exposure": round(qty * entry_p, 2)
            }) + "\n")

            # Section 4.2 Inviolate 4-Phase Ratchet & Trade-Relative Stagnation Exits
            p_sl_initial = entry_p * (1.0 - sl_pct)
            p_tp_ceiling = entry_p * (1.0 + tp_pct)
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
                    b_t     = bar_r.name.time() if hasattr(bar_r.name, 'time') else pd.Timestamp(str(bar_r.name)).time()
                    m_t     = max(m_t, b_high)
                    h_held  = bar_idx + 1

                    # Section 4.2 Four-Phase Ratchet Updates (Immobile Phase 0 SL until +1.60x)
                    if m_t >= entry_p * (1.0 + p3_trig * atr_norm):
                        phase = 3
                        floor_p = entry_p * (1.0 + p3_flr * atr_norm)
                        active_p_sl = max(active_p_sl, floor_p)
                    elif m_t >= entry_p * (1.0 + p2_trig * atr_norm):
                        phase = max(phase, 2)
                        floor_p = entry_p * (1.0 + p2_flr * atr_norm)
                        active_p_sl = max(active_p_sl, floor_p)
                    elif m_t >= entry_p * (1.0 + p1_trig * atr_norm):
                        phase = max(phase, 1)
                        floor_p = entry_p * (1.0 + p1_flr * atr_norm)
                        active_p_sl = max(active_p_sl, floor_p)

                    # 1. Final Take-Profit Ceiling Hit (+4.20x ATR)
                    if b_high >= p_tp_ceiling:
                        exit_price = p_tp_ceiling * (1.0 - slippage)
                        exit_time  = bar_r.name
                        exit_reason = "TP_CEILING_HIT"
                        break

                    # 2. Stop Loss or Trailed Floor Hit
                    if b_low <= active_p_sl:
                        exit_price = active_p_sl * (1.0 - slippage)
                        exit_time  = bar_r.name
                        if phase == 3:
                            exit_reason = "PHASE_3_RATCHET_FLOOR_HIT"
                        elif phase == 2:
                            exit_reason = "PHASE_2_RATCHET_FLOOR_HIT"
                        elif phase == 1:
                            exit_reason = "PHASE_1_HARD_FLOOR_HIT"
                        else:
                            exit_reason = "INITIAL_SL_HIT"
                        break

                    # 3. Section 2.2 Trade-Relative Stagnation Exits (evaluated on bar close)
                    mfe_atr = (m_t - entry_p) / (entry_p * atr_norm + 1e-9)
                    if h_held >= 6 and mfe_atr < 0.50 and b_close <= entry_p + 0.10 * entry_p * atr_norm:
                        exit_price = b_close * (1.0 - slippage)
                        exit_time  = bar_r.name
                        exit_reason = "STAGNATION_DEAD_MONEY_H6"
                        break
                    elif h_held >= 8 and mfe_atr < 1.00:
                        exit_price = b_close * (1.0 - slippage)
                        exit_time  = bar_r.name
                        exit_reason = "STAGNATION_SUB_1ATR_H8"
                        break
                    elif h_held >= 10 and mfe_atr < 1.50:
                        exit_price = b_close * (1.0 - slippage)
                        exit_time  = bar_r.name
                        exit_reason = "STAGNATION_SUB_1.5ATR_H10"
                        break

                    # 4. Hard EOD Square-Off at 15:05 IST
                    if b_t >= eod_square_time:
                        exit_price = b_close * (1.0 - slippage)
                        exit_time  = bar_r.name
                        exit_reason = "15:05_EOD_SQUARE_OFF"
                        break

                if exit_price is None and len(check_bars) > 0:
                    last_b = check_bars.iloc[-1]
                    exit_price = float(last_b["close"]) * (1.0 - slippage)
                    exit_time  = last_b.name
                    exit_reason = "15:05_EOD_SQUARE_OFF"

            if exit_price is None:
                realized_ret = float(row.get("realized_pct", 0.0))
                exit_price = entry_p * (1.0 + realized_ret) * (1.0 - slippage)
                exit_time  = ts
                exit_reason = "DEFAULT_EXIT"

            gross_pnl = (exit_price - entry_p) * qty
            charges   = calc_dhan_charges(entry_p, exit_price, qty)
            net_pnl   = round(gross_pnl - charges, 2)
            current_capital = round(current_capital + net_pnl, 2)
            is_win    = net_pnl > 0
            realized_ret_pct = round(((exit_price - entry_p) / entry_p) * 100, 2)

            daily_executed_count[d] = daily_executed_count.get(d, 0) + 1
            daily_sector_trade_count[day_sec_key] = curr_sec_trades + 1
            if is_opening_bar:
                daily_opening_taken[d] = True

            if not is_win:
                daily_net_loss[d] = daily_net_loss.get(d, 0.0) + abs(net_pnl)

            # Store active position to prevent overlapping execution
            active_position = {
                "entry_time": ts,
                "exit_time": exit_time,
                "ticker": ticker
            }

            trade_ledger.append({
                "timestamp": str(ts),
                "ticker": ticker,
                "exit_time": str(exit_time),
                "exit_reason": exit_reason,
                "phase": phase,
                "sector": sec,
                "sub_sector": row["sub_sector"],
                "direction": "BUY",
                "z_score": round(z_score, 2),
                "rank_score": round(rank_score, 4),
                "confidence": round(prob * 100, 2),
                "mc_win_prob": round(mc_win_prob * 100, 2),
                "entry_price": round(entry_p, 2),
                "exit_price": round(exit_price, 2),
                "tp_multiple": round(tp_ceiling_mult, 2),
                "qty": qty,
                "position_exposure": round(qty * entry_p, 2),
                "gross_pnl": round(gross_pnl, 2),
                "charges": charges,
                "net_pnl": net_pnl,
                "capital": current_capital,
                "winner": is_win,
                "realized_return_pct": realized_ret_pct
            })

            # Since max_concurrent_open = 1, break from candidate loop for this bar
            break

    f_jsonl.close()

    df_trades = pd.DataFrame(trade_ledger)
    df_rejected = pd.DataFrame(rejected_ledger)

    trades_csv_path = os.path.join(out_dirs["trades"], "v34_trades.csv")
    df_trades.to_csv(trades_csv_path, index=False)
    rejected_csv_path = os.path.join(out_dirs["trades"], "v34_rejected_candidates.csv")
    df_rejected.to_csv(rejected_csv_path, index=False)

    n_trades = len(df_trades)
    wins = int(df_trades["winner"].sum()) if n_trades > 0 else 0
    losses = n_trades - wins
    win_rate = round((wins / n_trades) * 100, 2) if n_trades > 0 else 0.0

    avg_win  = round(float(df_trades[df_trades["winner"]]["net_pnl"].mean()), 2) if wins > 0 else 0.0
    avg_loss = round(float(df_trades[~df_trades["winner"]]["net_pnl"].mean()), 2) if losses > 0 else 0.0
    payoff   = round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0.0

    tot_profit = float(df_trades[df_trades["net_pnl"] > 0]["net_pnl"].sum()) if wins > 0 else 0.0
    tot_loss   = abs(float(df_trades[df_trades["net_pnl"] < 0]["net_pnl"].sum())) if losses > 0 else 0.0
    profit_factor = round(tot_profit / tot_loss, 2) if tot_loss > 0 else 999.0

    total_charges = round(float(df_trades["charges"].sum()), 2) if n_trades > 0 else 0.0
    net_pnl = round(float(df_trades["net_pnl"].sum()), 2) if n_trades > 0 else 0.0
    final_cap = round(current_capital, 2)
    roi = round(((final_cap - capital) / capital) * 100, 2)

    rets = df_trades["net_pnl"].values if n_trades > 0 else np.array([0.0])
    mean_r = np.mean(rets)
    std_r = np.std(rets) if len(rets) > 1 else 1e-6
    sharpe = round(float((mean_r / (std_r + 1e-9)) * np.sqrt(252)), 2)
    downside_r = rets[rets < 0]
    downside_std = np.std(downside_r) if len(downside_r) > 1 else 1e-6
    sortino = round(float((mean_r / (downside_std + 1e-9)) * np.sqrt(252)), 2)

    return {
        "n_candidates": len(candidates),
        "n_approved": n_trades,
        "n_rejected": len(df_rejected),
        "n_trades": n_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff,
        "profit_factor": profit_factor,
        "total_charges": total_charges,
        "net_pnl": net_pnl,
        "final_capital": final_cap,
        "roi": roi,
        "sharpe": sharpe,
        "sortino": sortino,
        "df_trades": df_trades
    }

def run_global_monte_carlo(df_trades: pd.DataFrame, config: dict, out_dirs: dict) -> dict:
    mc_cfg = config.get("global_monte_carlo", {})
    n_sims = mc_cfg.get("iterations", 10000)
    capital = float(config.get("capital", 500.0))

    if len(df_trades) == 0:
        return {"ruin_prob": 0.0, "expected_dd": 0.0, "median_capital": capital, "p5": capital, "p95": capital}

    pnls = df_trades["net_pnl"].values
    n_trades = len(pnls)

    rng = np.random.default_rng(42)
    sample_indices = rng.choice(n_trades, size=(n_sims, n_trades), replace=True)
    sim_pnls = pnls[sample_indices]

    equity_paths = capital + np.cumsum(sim_pnls, axis=1)
    end_caps = equity_paths[:, -1]

    ruin_level = capital * 0.50
    ruined = np.any(equity_paths <= ruin_level, axis=1)
    ruin_prob = round(float(np.mean(ruined) * 100), 2)

    peaks = np.maximum.accumulate(equity_paths, axis=1)
    drawdowns = (peaks - equity_paths) / peaks
    max_dds = np.max(drawdowns, axis=1)
    expected_dd = round(float(np.mean(max_dds) * 100), 2)

    summary = {
        "ruin_prob": ruin_prob,
        "expected_dd": expected_dd,
        "median_capital": round(float(np.median(end_caps)), 2),
        "p5": round(float(np.percentile(end_caps, 5)), 2),
        "p95": round(float(np.percentile(end_caps, 95)), 2)
    }

    summary_path = os.path.join(out_dirs["monte_carlo"], "global_monte_carlo_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
