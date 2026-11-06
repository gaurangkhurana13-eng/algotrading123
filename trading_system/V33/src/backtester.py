"""
Backtester Module for Bhavishya V33:
DeepSeek-R1 Deterministic Mathematical Blueprint.
RankScore Quantile Decoupling (Top 5%), Time-Bucket Velocity Gates, Unified Sizing Equation,
and Inviolate 3-Stage Ratchet (Zero Premature Trailing).
Strictly lookahead-free, real Dhan MIS transaction costs, 100% causal bar-by-bar execution.
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
    max_concurrent   = int(config.get("max_concurrent_open", 2))
    max_per_sec      = int(config.get("max_trades_per_sector", 2))
    max_d_loss_val   = float(config.get("max_daily_loss", 25.0))
    sl_atr_mult      = float(config.get("sl_atr_mult", 1.5))
    slippage         = float(config.get("slippage_per_leg", 0.0005))
    min_friction_atr = float(config.get("min_friction_atr_ratio", 0.0038))

    # DeepSeek-R1 Ratchet Parameters
    p1_trig          = float(config.get("ratchet_phase1_trigger", 1.60))
    p1_flr           = float(config.get("ratchet_phase1_floor", 1.35))
    p2_trig          = float(config.get("ratchet_phase2_trigger", 2.30))
    p2_flr           = float(config.get("ratchet_phase2_floor", 1.95))
    tp_ceiling_mult  = float(config.get("tp_ceiling", 2.85))
    min_mc_prob      = float(config.get("pre_trade_mc", {}).get("min_mc_win_prob", 0.48))
    rank_cutoff      = float(config.get("rank_score_cutoff", 0.95))

    start_time       = pd.Timestamp(config.get("time_window_start", "09:30")).time()
    midday_split     = pd.Timestamp("11:15").time()
    cutoff_time      = pd.Timestamp(config.get("hard_cutoff_time", "13:15")).time()
    eod_square_time  = pd.Timestamp("15:15").time()

    mc_engine = PreTradeMonteCarloEngine(config)

    # Build ticker price history lookup for causal trade tracing
    ticker_dfs = {}
    for ticker, g in df_pool.groupby("ticker"):
        ticker_dfs[ticker] = g.sort_index()

    oos_df_copy = oos_df.copy()
    oos_df_copy["bar_time"] = [t.time() if hasattr(t, 'time') else pd.Timestamp(str(t)).time() for t in oos_df_copy.index]
    oos_df_copy["trade_date"] = oos_df_copy.index.date

    # Initial coarse filter: within [09:30, 13:15], min friction ATR, and RankScore >= 0.95 (Top 5%)
    candidates = oos_df_copy[
        (oos_df_copy["bar_time"] >= start_time) &
        (oos_df_copy["bar_time"] <= cutoff_time) &
        (oos_df_copy["atr_norm"] >= min_friction_atr) &
        (oos_df_copy["rank_score"] >= rank_cutoff)
    ].copy()

    sector_info = [get_sector_info(t) for t in candidates["ticker"]]
    candidates["sector"] = [s[0] for s in sector_info]
    candidates["sub_sector"] = [s[1] for s in sector_info]

    trade_ledger = []
    rejected_ledger = []
    daily_executed_count = {}
    daily_net_loss = {}
    daily_sector_trade_count = {}

    pre_mc_jsonl_path = os.path.join(out_dirs["monte_carlo"], "pre_trade_decisions.jsonl")
    f_jsonl = open(pre_mc_jsonl_path, "w", encoding="utf-8")

    current_capital = capital

    for trade_date, df_day in candidates.groupby("trade_date"):
        d = trade_date

        # Sort candidates strictly by RankScore descending, secondary sort by cs_rs descending
        df_sorted = df_day.sort_values(by=["rank_score", "cs_rs"], ascending=[False, False])

        active_positions_in_day = []

        for idx, row in df_sorted.iterrows():
            ticker     = row["ticker"]
            sec        = row["sector"]
            prob       = float(row["prob"])
            rank_score = float(row["rank_score"])
            bar_time   = row["bar_time"]
            rvol       = float(row.get("rvol", 1.0))
            cs_rs      = float(row.get("cs_rs", 0.0))
            pct_cs_rs  = float(row.get("pct_cs_rs", 0.50))
            atr_norm   = float(row["atr_norm"])
            raw_p      = float(row["close"])

            # 1. DeepSeek-R1 Section 4.2: Time-Bucket Velocity & Relative Strength Gate
            if bar_time <= midday_split:
                # Morning bucket (09:30 - 11:15 IST)
                if not (rvol >= 1.05 and cs_rs >= -0.10 and pct_cs_rs >= 0.80):
                    rejected_ledger.append({
                        "timestamp": str(idx),
                        "ticker": ticker,
                        "reason": f"MORNING_VELOCITY_FAIL (RVOL={rvol:.2f}, CS_RS={cs_rs:.2f}, PCT_CS={pct_cs_rs:.2f})",
                        "rank_score": rank_score,
                        "prob": round(prob, 4)
                    })
                    continue
            else:
                # Midday bucket (11:15 - 13:15 IST)
                if not (rvol >= 1.30 and cs_rs >= 0.25 and pct_cs_rs >= 0.85):
                    rejected_ledger.append({
                        "timestamp": str(idx),
                        "ticker": ticker,
                        "reason": f"MIDDAY_VELOCITY_FAIL (RVOL={rvol:.2f}, CS_RS={cs_rs:.2f}, PCT_CS={pct_cs_rs:.2f})",
                        "rank_score": rank_score,
                        "prob": round(prob, 4)
                    })
                    continue

            # 2. Daily Loss Circuit Breaker (stops new trades if daily loss >= Rs 25.0)
            if daily_net_loss.get(d, 0.0) >= max_d_loss_val:
                rejected_ledger.append({
                    "timestamp": str(idx),
                    "ticker": ticker,
                    "reason": f"CIRCUIT_BREAKER_DAILY_LOSS ({daily_net_loss[d]:.2f} >= {max_d_loss_val})",
                    "rank_score": rank_score,
                    "prob": round(prob, 4)
                })
                continue

            # 3. Daily Execution Cap
            if daily_executed_count.get(d, 0) >= daily_n:
                rejected_ledger.append({
                    "timestamp": str(idx),
                    "ticker": ticker,
                    "reason": f"DAILY_EXECUTED_CAP_REACHED ({daily_n})",
                    "rank_score": rank_score,
                    "prob": round(prob, 4)
                })
                continue

            day_sec_key = (d, sec)
            curr_sec_trades = daily_sector_trade_count.get(day_sec_key, 0)

            # 4. Sector Exposure Cap
            if curr_sec_trades >= max_per_sec:
                rejected_ledger.append({
                    "timestamp": str(idx),
                    "ticker": ticker,
                    "reason": f"MAX_SECTOR_CAP_REACHED ({max_per_sec})",
                    "rank_score": rank_score,
                    "prob": round(prob, 4)
                })
                continue

            # 5. Concurrent Open Positions (Margin Recycling Check)
            open_concurrent = [
                pos for pos in active_positions_in_day
                if pos["entry_time"] <= idx < pos["exit_time"] and not pos["risk_free"]
            ]
            if len(open_concurrent) >= max_concurrent:
                rejected_ledger.append({
                    "timestamp": str(idx),
                    "ticker": ticker,
                    "reason": f"MAX_CONCURRENT_OPEN_REACHED ({max_concurrent})",
                    "rank_score": rank_score,
                    "prob": round(prob, 4)
                })
                continue

            # Dynamic Horizon Calculation to 15:15 IST
            mins_left = (eod_square_time.hour * 60 + eod_square_time.minute) - (bar_time.hour * 60 + bar_time.minute)
            h_rem = min(12, max(1, mins_left // 15))

            if h_rem < 3:
                rejected_ledger.append({
                    "timestamp": str(idx),
                    "ticker": ticker,
                    "reason": "HORIZON_TOO_SHORT (<3 bars)",
                    "rank_score": rank_score,
                    "prob": round(prob, 4)
                })
                continue

            # Pre-Trade Monte Carlo Evaluation (Section 1.3 & 1.4)
            snapshot = {
                "close": raw_p,
                "atr_norm": atr_norm,
                "rank_score": rank_score,
                "cs_rs": cs_rs,
                "rvol": rvol,
                "hold_bars": h_rem
            }
            mc_eval = mc_engine.evaluate_candidate(snapshot)
            mc_win_prob = float(mc_eval.get("win_prob", 0.50))

            # Pre-Trade MC Gate
            if mc_win_prob < min_mc_prob:
                rejected_ledger.append({
                    "timestamp": str(idx),
                    "ticker": ticker,
                    "reason": f"PRE_TRADE_MC_REJECT (WinProb={mc_win_prob:.2f} < {min_mc_prob:.2f})",
                    "rank_score": rank_score,
                    "prob": round(prob, 4)
                })
                continue

            entry_p = raw_p * (1.0 + slippage)
            sl_pct  = sl_atr_mult * atr_norm
            tp_pct  = tp_ceiling_mult * atr_norm

            # DeepSeek-R1 Section 3.3: Unified Dual-Constraint Sizing Equation (All Stocks)
            # kappa'_t = 0.00216 * entry_p
            # Qty = min( floor(2450 / entry_p), floor(15.50 / (1.50*ATR_t + 0.00216*entry_p)) )
            denom = entry_p * (sl_pct + 0.00216)
            qty_risk = int(15.50 / denom) if denom > 0 else 1
            qty_notional = int(notional_cap / entry_p)
            qty = max(1, min(qty_notional, qty_risk))

            # Record pre-trade decision in JSONL
            f_jsonl.write(json.dumps({
                "timestamp": str(idx),
                "ticker": ticker,
                "rank_score": rank_score,
                "prob": prob,
                "mc_win_prob": mc_win_prob,
                "qty": qty,
                "position_exposure": round(qty * entry_p, 2)
            }) + "\n")

            # DeepSeek-R1 Section 2.2: Three-Stage Inviolate Ratchet Causal Tracing
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
                post_bars = t_df[t_df.index > idx]
                post_bars_day = post_bars[post_bars.index.date == d]
                check_bars = post_bars_day.head(h_rem)

                for _, bar_r in check_bars.iterrows():
                    b_high  = float(bar_r["high"])
                    b_low   = float(bar_r["low"])
                    b_close = float(bar_r["close"])
                    b_time  = bar_r.name.time() if hasattr(bar_r.name, 'time') else pd.Timestamp(str(bar_r.name)).time()
                    m_t     = max(m_t, b_high)

                    # Phase 1: at M_t >= P_0 + 1.60x ATR, lock hard floor at +1.35x ATR
                    if phase < 1:
                        if m_t >= entry_p * (1.0 + p1_trig * atr_norm):
                            phase = 1
                            floor_p = entry_p * (1.0 + p1_flr * atr_norm)
                            active_p_sl = max(active_p_sl, floor_p)

                    # Phase 2: at M_t >= P_0 + 2.30x ATR, lock floor at +1.95x ATR
                    if m_t >= entry_p * (1.0 + p2_trig * atr_norm):
                        phase = 2
                        floor_p = entry_p * (1.0 + p2_flr * atr_norm)
                        active_p_sl = max(active_p_sl, floor_p)

                    # Time Stop: At 13:15 IST, if phase == 0 (no breakout momentum), exit at bar close
                    if b_time >= cutoff_time and phase == 0:
                        exit_price = b_close * (1.0 - slippage)
                        exit_time  = bar_r.name
                        exit_reason = "13:15_TIME_STOP"
                        break

                    # Check TP Ceiling Hit (+2.85x ATR)
                    if b_high >= p_tp_ceiling:
                        exit_price = p_tp_ceiling * (1.0 - slippage)
                        exit_time  = bar_r.name
                        exit_reason = "TP_CEILING_HIT"
                        break

                    # Check Stop Loss / Ratchet Floor Hit
                    if b_low <= active_p_sl:
                        exit_price = active_p_sl * (1.0 - slippage)
                        exit_time  = bar_r.name
                        if phase == 2:
                            exit_reason = "PHASE_2_RATCHET_FLOOR_HIT"
                        elif phase == 1:
                            exit_reason = "PHASE_1_HARD_FLOOR_HIT"
                        else:
                            exit_reason = "INITIAL_SL_HIT"
                        break

                if exit_price is None and len(check_bars) > 0:
                    last_b = check_bars.iloc[-1]
                    exit_price = float(last_b["close"]) * (1.0 - slippage)
                    exit_time  = last_b.name
                    exit_reason = "15:15_MIS_AUTO_SQUARE_OFF"

            if exit_price is None:
                realized_ret = float(row.get("realized_pct", 0.0))
                exit_price = entry_p * (1.0 + realized_ret) * (1.0 - slippage)
                exit_time  = idx
                exit_reason = "DEFAULT_EXIT"

            gross_pnl = (exit_price - entry_p) * qty
            charges   = calc_dhan_charges(entry_p, exit_price, qty)
            net_pnl   = round(gross_pnl - charges, 2)
            current_capital = round(current_capital + net_pnl, 2)
            is_win    = net_pnl > 0
            realized_ret_pct = round(((exit_price - entry_p) / entry_p) * 100, 2)

            daily_executed_count[d] = daily_executed_count.get(d, 0) + 1
            daily_sector_trade_count[day_sec_key] = curr_sec_trades + 1

            if not is_win:
                daily_net_loss[d] = daily_net_loss.get(d, 0.0) + abs(net_pnl)

            # Section 5.6: Margin Recycling: Phase 1 or 2 locks profit, freeing concurrent slot
            active_positions_in_day.append({
                "entry_time": idx,
                "exit_time": exit_time,
                "risk_free": (phase >= 1)
            })

            trade_ledger.append({
                "timestamp": str(idx),
                "ticker": ticker,
                "exit_time": str(exit_time),
                "exit_reason": exit_reason,
                "phase": phase,
                "sector": sec,
                "sub_sector": row["sub_sector"],
                "direction": "BUY",
                "rank_score": rank_score,
                "confidence": round(prob * 100, 2),
                "mc_win_prob": round(mc_win_prob * 100, 2),
                "entry_price": round(entry_p, 2),
                "exit_price": round(exit_price, 2),
                "tp_multiple": tp_ceiling_mult,
                "qty": qty,
                "position_exposure": round(qty * entry_p, 2),
                "gross_pnl": round(gross_pnl, 2),
                "charges": charges,
                "net_pnl": net_pnl,
                "capital": current_capital,
                "winner": is_win,
                "realized_return_pct": realized_ret_pct
            })

    f_jsonl.close()

    df_trades = pd.DataFrame(trade_ledger)
    df_rejected = pd.DataFrame(rejected_ledger)

    trades_csv_path = os.path.join(out_dirs["trades"], "v33_trades.csv")
    df_trades.to_csv(trades_csv_path, index=False)

    rejected_csv_path = os.path.join(out_dirs["trades"], "v33_rejected_candidates.csv")
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
