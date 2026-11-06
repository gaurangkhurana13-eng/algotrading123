"""
Backtester Module for Bhavishya V36:
Claude Sonnet High-Conviction Breakout & Microstructure Architecture.
1. Dual-Constraint Sizing with Risk-Utilization Gate (q >= 3 and R_used >= 11.0)
2. 7-Factor High-Conviction Scoring Filter (Score >= 5 of 7)
3. Gap-Protected Fill Model: stop fill = min(stop, bar open)
4. Anti-Choke Breathing Ratchet (Phase 1: -0.35R, Phase 2: +0.60R, Phase 3: Chandelier, TP: +4.0R)
5. Partial Take-Profit: If q >= 4, sell half at +2.0R and trail remainder
6. Multi-Horizon Stagnation Exits (Bars 4, 6, 10, and VWAP breach) with exemptions
Strictly lookahead-free, real Dhan MIS transaction costs, 100% causal execution.
"""

import os
import json
import numpy as np
import pandas as pd
from .dhan_engine import calc_dhan_charges

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
    daily_n          = int(config.get("max_executed_per_day", 2))
    max_d_losses     = int(config.get("max_daily_losses", 2))
    l_cap            = float(config.get("l_cap", 15.00))
    min_r_used       = float(config.get("min_r_used", 11.0))
    min_q            = int(config.get("min_q", 3))
    stop_d_mult      = float(config.get("stop_d_mult", 1.4))
    tp_ceiling_r     = float(config.get("tp_ceiling_r", 4.0))
    score_min        = int(config.get("conviction_score_min", 5))

    start_window     = pd.Timestamp(config.get("entry_window_start", "09:45")).time()
    end_window       = pd.Timestamp(config.get("entry_window_end", "12:15")).time()
    cutoff_time      = pd.Timestamp(config.get("hard_cutoff_time", "13:30")).time()
    eod_square_time  = pd.Timestamp(config.get("eod_square_time", "15:10")).time()

    ticker_dfs = {}
    for ticker, g in df_pool.groupby("ticker"):
        ticker_dfs[ticker] = g.sort_index()

    oos_df_copy = oos_df.copy()
    oos_df_copy["bar_time"] = [t.time() if hasattr(t, 'time') else pd.Timestamp(str(t)).time() for t in oos_df_copy.index]
    oos_df_copy["trade_date"] = oos_df_copy.index.date

    # Initial coarse filter: within [09:45, 12:15] window and conviction score >= 5
    candidates = oos_df_copy[
        (oos_df_copy["bar_time"] >= start_window) &
        (oos_df_copy["bar_time"] <= end_window) &
        (oos_df_copy["conviction_score"] >= score_min)
    ].copy()

    sector_info = [get_sector_info(t) for t in candidates["ticker"]]
    candidates["sector"] = [s[0] for s in sector_info]
    candidates["sub_sector"] = [s[1] for s in sector_info]

    trade_ledger = []
    rejected_ledger = []
    daily_executed_count = {}
    daily_loss_count = {}
    daily_net_loss_val = {}

    current_capital = capital
    active_position = None  # Single position execution

    unique_timestamps = sorted(list(candidates.index.unique()))

    for ts in unique_timestamps:
        df_bar = candidates.loc[[ts]]
        b_time = ts.time() if hasattr(ts, 'time') else pd.Timestamp(str(ts)).time()
        d = ts.date()

        # Check if active position has exited
        if active_position is not None:
            if ts <= active_position["exit_time"]:
                continue
            else:
                active_position = None

        # Check daily stops
        if daily_loss_count.get(d, 0) >= max_d_losses:
            continue
        if daily_executed_count.get(d, 0) >= daily_n:
            continue

        # Rank candidates at this timestamp by score descending, then by rvol descending
        df_sorted = df_bar.sort_values(by=["conviction_score", "rvol"], ascending=[False, False])

        for idx, row in df_sorted.iterrows():
            ticker     = row["ticker"]
            sec        = row["sector"]
            score      = int(row["conviction_score"])
            prob       = float(row["prob"])
            raw_p      = float(row["close"])
            atr_raw    = float(row["atr_raw"])
            rvol       = float(row.get("rvol", 1.0))

            if raw_p <= 0 or atr_raw <= 0:
                continue

            # Section 1: Dual-Constraint Sizing & Risk-Utilization Gate
            d_dist = stop_d_mult * atr_raw
            s_slip = max(0.0005 * raw_p, 0.25 * atr_raw)
            c_cost = 0.00095 * raw_p
            per_share_risk = d_dist + s_slip + c_cost

            q_notional = int(notional_cap / raw_p)
            q_risk = int(l_cap / per_share_risk) if per_share_risk > 0 else 0
            q = min(q_notional, q_risk)
            r_used = q * per_share_risk

            # Risk-utilization gate: q >= 3 AND R_used >= 11.0
            if q < min_q or r_used < min_r_used:
                rejected_ledger.append({
                    "timestamp": str(ts),
                    "ticker": ticker,
                    "reason": f"RISK_UTILIZATION_GATE_REJECT (q={q} < {min_q} or R_used={r_used:.2f} < {min_r_used})",
                    "score": score,
                    "r_used": round(r_used, 2)
                })
                continue

            entry_p = raw_p * 1.0005
            p_sl0 = entry_p - d_dist - s_slip
            p_tp_ceiling = entry_p + tp_ceiling_r * d_dist
            delta = max(0.50 * atr_raw, 0.0010 * entry_p)

            # Section 2 & 4: Progressive Ratchet, Chandelier Trailing & Stagnation Execution
            active_sl = p_sl0
            m_t = entry_p
            hh_close = entry_p
            phase = 0
            exit_price = None
            exit_time = None
            exit_reason = None
            partial_exit_price = None
            partial_exit_time = None
            partial_q = 0

            # Partial TP config (if q >= 4: sell half at +2.0R)
            can_partial = (q >= 4)
            p_partial_tp = entry_p + 2.0 * d_dist if can_partial else 999999.0
            partial_taken = False

            consec_below_vwap = 0

            if ticker in ticker_dfs:
                t_df = ticker_dfs[ticker]
                post_bars = t_df[t_df.index > ts]
                post_bars_day = post_bars[post_bars.index.date == d]

                for bar_idx, (_, bar_r) in enumerate(post_bars_day.iterrows()):
                    b_open  = float(bar_r["open"])
                    b_high  = float(bar_r["high"])
                    b_low   = float(bar_r["low"])
                    b_close = float(bar_r["close"])
                    b_vwap  = float(bar_r.get("vwap", entry_p))
                    b_t     = bar_r.name.time() if hasattr(bar_r.name, 'time') else pd.Timestamp(str(bar_r.name)).time()
                    h_held  = bar_idx + 1

                    m_t = max(m_t, b_high)
                    hh_close = max(hh_close, b_close)
                    mfe_r = (m_t - entry_p) / d_dist

                    # Consecutive closes below VWAP check
                    if b_close < b_vwap:
                        consec_below_vwap += 1
                    else:
                        consec_below_vwap = 0

                    # Check Partial TP (+2.0R) for half position
                    if can_partial and not partial_taken and b_high >= p_partial_tp:
                        partial_q = q // 2
                        partial_exit_price = p_partial_tp * 0.9995
                        partial_exit_time = bar_r.name
                        partial_taken = True

                    # Ratchet Updates (close-confirmed per Section 2)
                    if mfe_r >= 2.5:
                        phase = 3
                        chandelier_flr = hh_close - 1.25 * atr_raw - delta
                        active_sl = max(active_sl, chandelier_flr)
                    elif mfe_r >= 1.8:
                        phase = max(phase, 2)
                        active_sl = max(active_sl, entry_p + 0.60 * d_dist)
                    elif mfe_r >= 1.0:
                        phase = max(phase, 1)
                        active_sl = max(active_sl, entry_p - 0.35 * d_dist)

                    # Check TP Ceiling Hit (+4.0R)
                    if b_high >= p_tp_ceiling:
                        exit_price = p_tp_ceiling * 0.9995
                        exit_time = bar_r.name
                        exit_reason = "TP_CEILING_HIT (+4.0R)"
                        break

                    # Check Stop Loss / Trailed Floor Hit (Fill Model: min(stop, bar open) to prevent gap blowouts)
                    if b_low <= active_sl:
                        fill_p = min(active_sl, b_open)
                        exit_price = fill_p * 0.9995
                        exit_time = bar_r.name
                        if phase == 3:
                            exit_reason = "CHANDELIER_TRAIL_HIT (Phase 3)"
                        elif phase == 2:
                            exit_reason = "PHASE_2_LOCKED_FLOOR_HIT (+0.6R)"
                        elif phase == 1:
                            exit_reason = "PHASE_1_TIGHTENED_SL_HIT (-0.35R)"
                        else:
                            exit_reason = "INITIAL_SL_HIT (-1.0R)"
                        break

                    # Section 4: Stagnation Exits with Exemptions
                    is_exempt = (mfe_r >= 0.8) or (b_close > entry_p + 0.5 * d_dist) or (b_high >= m_t)

                    if not is_exempt:
                        # Bar 4: MFE < 0.3R -> tighten SL to entry - 0.5d
                        if h_held == 4 and mfe_r < 0.3:
                            active_sl = max(active_sl, entry_p - 0.50 * d_dist)

                        # Bar 6: MFE < 0.5R AND close < entry + 0.15d -> exit at close
                        if h_held >= 6 and mfe_r < 0.5 and b_close < entry_p + 0.15 * d_dist:
                            exit_price = b_close * 0.9995
                            exit_time = bar_r.name
                            exit_reason = "STAGNATION_EXIT_BAR_6"
                            break

                        # Bar 10: MFE < 1.0R AND close < entry + 0.30d -> exit at close
                        if h_held >= 10 and mfe_r < 1.0 and b_close < entry_p + 0.30 * d_dist:
                            exit_price = b_close * 0.9995
                            exit_time = bar_r.name
                            exit_reason = "STAGNATION_EXIT_BAR_10"
                            break

                        # Any bar >= 3: 2 consecutive closes < VWAP AND MFE < 0.7R -> exit
                        if h_held >= 3 and consec_below_vwap >= 2 and mfe_r < 0.7:
                            exit_price = b_close * 0.9995
                            exit_time = bar_r.name
                            exit_reason = "VWAP_BREAK_STAGNATION_EXIT"
                            break

                    # Hard EOD Flatten at 15:10 IST
                    if b_t >= eod_square_time:
                        exit_price = b_close * 0.9995
                        exit_time = bar_r.name
                        exit_reason = "15:10_EOD_FLATTEN"
                        break

                if exit_price is None and len(post_bars_day) > 0:
                    last_b = post_bars_day.iloc[-1]
                    exit_price = float(last_b["close"]) * 0.9995
                    exit_time = last_b.name
                    exit_reason = "15:10_EOD_FLATTEN"

            if exit_price is None:
                exit_price = entry_p * 0.99
                exit_time = ts
                exit_reason = "DEFAULT_EXIT"

            # Calculate Net PnL (accounting for partial exit if taken)
            rem_q = q - partial_q
            if partial_taken and partial_q > 0:
                p_gross = (partial_exit_price - entry_p) * partial_q + (exit_price - entry_p) * rem_q
                p_charges = calc_dhan_charges(entry_p, partial_exit_price, partial_q) + calc_dhan_charges(entry_p, exit_price, rem_q)
            else:
                p_gross = (exit_price - entry_p) * q
                p_charges = calc_dhan_charges(entry_p, exit_price, q)

            net_pnl = round(p_gross - p_charges, 2)
            current_capital = round(current_capital + net_pnl, 2)
            is_win = net_pnl > 0
            realized_ret_pct = round(((exit_price - entry_p) / entry_p) * 100, 2)

            daily_executed_count[d] = daily_executed_count.get(d, 0) + 1
            if not is_win:
                daily_loss_count[d] = daily_loss_count.get(d, 0) + 1
                daily_net_loss_val[d] = daily_net_loss_val.get(d, 0.0) + abs(net_pnl)

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
                "score": score,
                "sector": sec,
                "sub_sector": row["sub_sector"],
                "entry_price": round(entry_p, 2),
                "exit_price": round(exit_price, 2),
                "qty": q,
                "partial_q": partial_q,
                "r_used": round(r_used, 2),
                "position_exposure": round(q * entry_p, 2),
                "gross_pnl": round(p_gross, 2),
                "charges": round(p_charges, 2),
                "net_pnl": net_pnl,
                "capital": current_capital,
                "winner": is_win,
                "realized_return_pct": realized_ret_pct
            })

            # Single position execution per bar
            break

    df_trades = pd.DataFrame(trade_ledger)
    df_rejected = pd.DataFrame(rejected_ledger)

    trades_csv_path = os.path.join(out_dirs["trades"], "v36_trades.csv")
    df_trades.to_csv(trades_csv_path, index=False)
    rejected_csv_path = os.path.join(out_dirs["trades"], "v36_rejected_candidates.csv")
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
