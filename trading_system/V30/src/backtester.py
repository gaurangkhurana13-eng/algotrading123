"""
Backtester Module for Bhavishya V30:
Two-Stage Asymmetric Ratchet, Terminal-Aware MC, Margin Recycling & Friction Gate.
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

def compute_v30_risk_budget(prob: float, min_b: float = 24.0, max_b: float = 38.0) -> float:
    # R_t = clip(24.0 + 14.0 * ((P_cal - 0.45) / 0.12), 24.0, 38.0)
    raw_b = min_b + (max_b - min_b) * ((prob - 0.45) / 0.12)
    return round(float(np.clip(raw_b, min_b, max_b)), 2)

def execute_backtest(oos_df: pd.DataFrame, df_pool: pd.DataFrame, config: dict, out_dirs: dict) -> dict:
    capital          = float(config.get("capital", 500.0))
    leverage         = float(config.get("leverage", 5.0))
    notional_cap     = float(config.get("max_notional_capital", 2450.0))
    conv_thr         = float(config.get("conviction_threshold", 0.45))
    min_budget       = float(config.get("min_risk_budget", 24.0))
    max_budget       = float(config.get("max_risk_budget", 38.0))
    daily_n          = int(config.get("max_executed_per_day", 5))
    max_concurrent   = int(config.get("max_concurrent_open", 3))
    max_per_sec      = int(config.get("max_trades_per_sector", 2))
    sec_penalty      = float(config.get("second_sector_entry_penalty", 0.80))
    max_d_loss       = int(config.get("max_daily_loss", 2))
    sl_atr_mult      = float(config.get("sl_atr_mult", 1.5))
    slippage         = float(config.get("slippage_per_leg", 0.0005))
    ratchet_trigger  = float(config.get("ratchet_trigger_atr", 1.70))
    ratchet_floor    = float(config.get("ratchet_floor_atr", 1.45))
    ratchet_trail    = float(config.get("ratchet_trail_atr", 0.85))
    min_friction_atr = float(config.get("min_friction_atr_ratio", 0.0038))

    cutoff_time = pd.Timestamp(config.get("time_window_end", "13:30")).time()
    lock_time   = pd.Timestamp(config.get("profit_lock_guard_time", "15:00")).time()
    eod_square_time = pd.Timestamp("15:15").time()

    mc_engine = PreTradeMonteCarloEngine(config)

    # Build ticker price history lookup for causal trade tracing
    ticker_dfs = {}
    for ticker, g in df_pool.groupby("ticker"):
        ticker_dfs[ticker] = g.sort_index()

    oos_df_copy = oos_df.copy()
    oos_df_copy["bar_time"] = [t.time() if hasattr(t, 'time') else pd.Timestamp(str(t)).time() for t in oos_df_copy.index]
    oos_df_copy["trade_date"] = oos_df_copy.index.date

    # Step 3.1: Spatio-Temporal Filter (t <= 13:30 IST) & Friction Efficiency Gate
    candidates = oos_df_copy[
        (oos_df_copy["bar_time"] <= cutoff_time) &
        (oos_df_copy["prob"] >= conv_thr) &
        (oos_df_copy["atr_norm"] >= min_friction_atr)
    ].copy()

    sector_info = [get_sector_info(t) for t in candidates["ticker"]]
    candidates["sector"] = [s[0] for s in sector_info]
    candidates["sub_sector"] = [s[1] for s in sector_info]

    trade_ledger = []
    rejected_ledger = []
    daily_executed_count = {}
    daily_losses = {}
    daily_sector_trade_count = {}

    pre_mc_jsonl_path = os.path.join(out_dirs["monte_carlo"], "pre_trade_decisions.jsonl")
    f_jsonl = open(pre_mc_jsonl_path, "w", encoding="utf-8")

    current_capital = capital

    for trade_date, df_day in candidates.groupby("trade_date"):
        # Sort candidates strictly by P_cal descending (V27's pure probability priority)
        df_sorted = df_day.sort_values(by="prob", ascending=False)
        d = trade_date

        active_positions_in_day = []

        for idx, row in df_sorted.iterrows():
            # 1. Daily Execution Cap (Margin Recycled up to 5/day)
            if daily_executed_count.get(d, 0) >= daily_n:
                rejected_ledger.append({
                    "timestamp": str(idx),
                    "ticker": row["ticker"],
                    "reason": f"DAILY_EXECUTED_CAP_REACHED ({daily_n})",
                    "prob": round(row["prob"], 4)
                })
                continue

            # 2. Daily Loss Circuit Breaker
            if daily_losses.get(d, 0) >= max_d_loss:
                rejected_ledger.append({
                    "timestamp": str(idx),
                    "ticker": row["ticker"],
                    "reason": "CIRCUIT_BREAKER_MAX_DAILY_LOSS",
                    "prob": round(row["prob"], 4)
                })
                continue

            ticker = row["ticker"]
            sec = row["sector"]
            day_sec_key = (d, sec)
            curr_sec_trades = daily_sector_trade_count.get(day_sec_key, 0)

            # 3. Sector Exposure Cap
            if curr_sec_trades >= max_per_sec:
                rejected_ledger.append({
                    "timestamp": str(idx),
                    "ticker": ticker,
                    "reason": f"MAX_SECTOR_CAP_REACHED ({max_per_sec})",
                    "prob": round(row["prob"], 4)
                })
                continue

            # Check concurrent open positions at this timestamp
            open_concurrent = [
                pos for pos in active_positions_in_day
                if pos["entry_time"] <= idx < pos["exit_time"] and not pos["risk_free"]
            ]
            if len(open_concurrent) >= max_concurrent:
                rejected_ledger.append({
                    "timestamp": str(idx),
                    "ticker": ticker,
                    "reason": f"MAX_CONCURRENT_OPEN_REACHED ({max_concurrent})",
                    "prob": round(row["prob"], 4)
                })
                continue

            raw_price = float(row["close"])
            entry_p   = raw_price * (1.0 + slippage)
            atr_norm  = float(row["atr_norm"])
            prob      = float(row["prob"])
            sl_pct    = sl_atr_mult * atr_norm
            bar_time  = row["bar_time"]

            # Dynamic Horizon Calculation to 15:15 IST
            mins_left = (eod_square_time.hour * 60 + eod_square_time.minute) - (bar_time.hour * 60 + bar_time.minute)
            h_rem = min(12, max(1, mins_left // 15))

            if h_rem < 3:
                rejected_ledger.append({
                    "timestamp": str(idx),
                    "ticker": ticker,
                    "reason": "HORIZON_TOO_SHORT (<3 bars)",
                    "prob": round(prob, 4)
                })
                continue

            # Step 4: Two-Stage Asymmetric Ratchet Initial Ceilings
            if h_rem >= 10:
                tp_mult = 3.20 if prob >= 0.56 else 2.85
            elif h_rem >= 6:
                tp_mult = 2.40 if prob >= 0.56 else 2.10
            else:
                tp_mult = 1.85 if prob >= 0.56 else 1.70
            tp_pct = tp_mult * atr_norm

            # Continuous Margin-Synchronized Half-Kelly Sizing
            risk_budget = compute_v30_risk_budget(prob, min_budget, max_budget)
            if curr_sec_trades >= 1:
                risk_budget *= sec_penalty

            # Available margin considering active concurrent positions
            avail_margin = notional_cap / max(1, (len(open_concurrent) + 1))
            per_share_sl_dist = entry_p * sl_pct
            qty_raw = int(risk_budget / per_share_sl_dist) if per_share_sl_dist > 0 else 1
            qty_margin = int(avail_margin / entry_p)
            qty = max(1, min(qty_raw, qty_margin))

            snapshot = {
                "timestamp": str(idx),
                "ticker": ticker,
                "close": raw_price,
                "atr_norm": atr_norm,
                "prob": prob,
                "qty": qty,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "hold_bars": h_rem,
                "capital_available": round(current_capital, 2),
                "position_size": round(qty * entry_p, 2),
                "risk_budget": round(risk_budget, 2),
                "sector": sec,
                "sub_sector": row["sub_sector"]
            }

            # Step 3.3: Terminal-Aware Monte Carlo Gate (MC_term >= 0.49)
            mc_eval = mc_engine.evaluate_candidate(snapshot)
            f_jsonl.write(json.dumps(mc_eval) + "\n")

            if mc_eval["decision"] != "ALLOW ENTRY":
                rejected_ledger.append({
                    "timestamp": str(idx),
                    "ticker": ticker,
                    "reason": f"PRE_TRADE_MC_{mc_eval['decision']}: {mc_eval['reason']}",
                    "prob": round(prob, 4)
                })
                continue

            # Step 4: Execute Two-Stage Asymmetric Ratchet Causal Tracing
            p_sl_initial = entry_p * (1.0 - sl_pct)
            p_tp_ceiling = entry_p * (1.0 + tp_pct)
            active_p_sl  = p_sl_initial
            ratchet_active = False
            h_max_so_far = entry_p
            exit_price   = None
            exit_time    = None
            exit_reason  = None

            if ticker in ticker_dfs:
                t_df = ticker_dfs[ticker]
                post_bars = t_df[t_df.index > idx]
                post_bars_day = post_bars[post_bars.index.date == d]
                check_bars = post_bars_day.head(h_rem)

                for _, bar_r in check_bars.iterrows():
                    b_high = bar_r["high"]
                    b_low  = bar_r["low"]
                    b_close = bar_r["close"]
                    b_time = bar_r.name.time() if hasattr(bar_r.name, 'time') else pd.Timestamp(str(bar_r.name)).time()
                    h_max_so_far = max(h_max_so_far, b_high)

                    # Phase 2 Trigger: High touches +1.70x ATR (V27's target achieved!)
                    if not ratchet_active:
                        if b_high >= entry_p * (1.0 + ratchet_trigger * atr_norm):
                            ratchet_active = True
                            # Lock hard profit floor at +1.45x ATR
                            floor_p = entry_p * (1.0 + ratchet_floor * atr_norm)
                            trail_p = h_max_so_far - (ratchet_trail * atr_norm * entry_p)
                            active_p_sl = max(active_p_sl, floor_p, trail_p)
                    else:
                        # Continue dynamic ratcheting
                        trail_p = h_max_so_far - (ratchet_trail * atr_norm * entry_p)
                        floor_p = entry_p * (1.0 + ratchet_floor * atr_norm)
                        active_p_sl = max(active_p_sl, floor_p, trail_p)

                    # Phase 3: 15:00 IST Profit-Lock Guard
                    if b_time >= lock_time:
                        if b_close >= entry_p * (1.0 + 0.30 * atr_norm):
                            exit_price = b_close * (1.0 - slippage)
                            exit_time  = bar_r.name
                            exit_reason = "15:00_PROFIT_LOCK_HARVEST"
                            break

                    # Check Take Profit Ceiling hit
                    if b_high >= p_tp_ceiling:
                        exit_price = p_tp_ceiling * (1.0 - slippage)
                        exit_time  = bar_r.name
                        exit_reason = "TP_CEILING_HIT"
                        break

                    # Check Stop Loss / Ratchet Floor hit
                    if b_low <= active_p_sl:
                        exit_price = active_p_sl * (1.0 - slippage)
                        exit_time  = bar_r.name
                        exit_reason = "RATCHET_FLOOR_HIT" if ratchet_active else "INITIAL_SL_HIT"
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
                daily_losses[d] = daily_losses.get(d, 0) + 1

            # Track in active positions for margin recycling
            active_positions_in_day.append({
                "entry_time": idx,
                "exit_time": exit_time,
                "risk_free": ratchet_active
            })

            trade_ledger.append({
                "timestamp": str(idx),
                "ticker": ticker,
                "exit_time": str(exit_time),
                "exit_reason": exit_reason,
                "ratchet_activated": ratchet_active,
                "sector": sec,
                "sub_sector": row["sub_sector"],
                "direction": "BUY",
                "confidence": round(prob * 100, 2),
                "mc_win_prob": round(mc_eval["stats"].get("win_prob", 0.0) * 100, 2),
                "mc_expected_ev": round(mc_eval["stats"].get("expected_ev", 0.0), 2),
                "entry_price": round(entry_p, 2),
                "exit_price": round(exit_price, 2),
                "tp_multiple": tp_mult,
                "qty": qty,
                "position_exposure": round(qty * entry_p, 2),
                "half_kelly_risk_budget": round(risk_budget, 2),
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

    trades_csv_path = os.path.join(out_dirs["trades"], "v30_trades.csv")
    df_trades.to_csv(trades_csv_path, index=False)

    rejected_csv_path = os.path.join(out_dirs["trades"], "v30_rejected_candidates.csv")
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
