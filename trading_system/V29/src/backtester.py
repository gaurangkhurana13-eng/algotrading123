"""
Backtester Module for Bhavishya V29:
Spatio-Temporal Horizon Protection, Dynamic First-Passage Scaling & Margin-Synchronized Half-Kelly.
Enforces strict 13:15 IST entry cutoff, horizon-truncated Monte Carlo, 15:00 IST profit-lock guard,
invariant 1.50x ATR structural stop loss, and real Dhan MIS transaction costs.
"""

import os
import json
import numpy as np
import pandas as pd
from .dhan_engine import calc_dhan_charges
from .pre_trade_mc import PreTradeMonteCarloEngine

SECTOR_MAP = {
    "MOREPENLAB": ("HEALTHCARE", "PHARMA"),
    "LLOYDSENGG": ("INDUSTRIALS", "CAPITAL_GOODS"),
    "MRPL": ("ENERGY", "OIL_GAS"),
    "ALOKINDS": ("CONSUMER", "TEXTILES"),
    "PCJEWELLER": ("CONSUMER", "GEMS_JEWELLERY")
}

def get_sector_info(ticker: str):
    clean = ticker.replace(".NS", "")
    return SECTOR_MAP.get(clean, ("GENERAL", "GENERAL"))

def compute_continuous_half_kelly(prob: float, min_b: float = 22.0, max_b: float = 35.0) -> float:
    # R_t = clip(22.0 + 13.0 * ((P_cal - 0.48) / 0.10), 22.0, 35.0)
    raw_b = min_b + (max_b - min_b) * ((prob - 0.48) / 0.10)
    return round(float(np.clip(raw_b, min_b, max_b)), 2)

def execute_backtest(oos_df: pd.DataFrame, df_pool: pd.DataFrame, config: dict, out_dirs: dict) -> dict:
    capital       = float(config.get("capital", 500.0))
    leverage      = float(config.get("leverage", 5.0))
    notional_cap  = float(config.get("max_notional_capital", 2450.0))
    conv_thr      = float(config.get("conviction_threshold", 0.48))
    min_budget    = float(config.get("min_risk_budget", 22.0))
    max_budget    = float(config.get("max_risk_budget", 35.0))
    daily_n       = int(config.get("daily_top_n", 3))
    max_per_sec   = int(config.get("max_trades_per_sector", 2))
    sec_penalty   = float(config.get("second_sector_entry_penalty", 0.80))
    max_d_loss    = int(config.get("max_daily_loss", 2))
    sl_atr_mult   = float(config.get("sl_atr_mult", 1.5))
    slippage      = float(config.get("slippage_per_leg", 0.0005))
    apex_tp_mult  = float(config.get("apex_tp_mult", 2.65))
    std_tp_mult   = float(config.get("standard_tp_mult", 2.35))
    mid_tp_mult   = float(config.get("midday_tp_mult", 1.95))
    cutoff_time_s = config.get("time_window_end", "13:15")
    lock_time_s   = config.get("profit_lock_guard_time", "15:00")
    lock_min_atr  = float(config.get("profit_lock_min_atr", 0.35))
    lock_buf_rs   = float(config.get("profit_lock_buffer_rs", 3.0))

    cutoff_time = pd.Timestamp(cutoff_time_s).time()
    lock_time   = pd.Timestamp(lock_time_s).time()
    eod_square_time = pd.Timestamp("15:15").time()

    mc_engine = PreTradeMonteCarloEngine(config)

    # Build ticker price history lookup for causal trade tracing
    ticker_dfs = {}
    for ticker, g in df_pool.groupby("ticker"):
        ticker_dfs[ticker] = g.sort_index()

    # Spatio-Temporal Filter: t <= 13:15 IST and prob >= 0.48
    oos_df_copy = oos_df.copy()
    oos_df_copy["bar_time"] = [t.time() if hasattr(t, 'time') else pd.Timestamp(str(t)).time() for t in oos_df_copy.index]
    oos_df_copy["trade_date"] = oos_df_copy.index.date

    # Strict slot-protected morning/midday filter (t <= 13:15 IST)
    candidates = oos_df_copy[
        (oos_df_copy["bar_time"] <= cutoff_time) &
        (oos_df_copy["prob"] >= conv_thr)
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
        # Sort candidates of the day by prob descending
        df_sorted = df_day.sort_values(by="prob", ascending=False)
        d = trade_date

        for idx, row in df_sorted.iterrows():
            # 1. Post-MC Daily Execution Cap
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

            raw_price = float(row["close"])
            entry_p   = raw_price * (1.0 + slippage)
            atr_norm  = float(row["atr_norm"])
            prob      = float(row["prob"])
            sl_pct    = sl_atr_mult * atr_norm
            bar_time  = row["bar_time"]

            # Dynamic Horizon Calculation to 15:15 IST
            mins_left = (eod_square_time.hour * 60 + eod_square_time.minute) - (bar_time.hour * 60 + bar_time.minute)
            h_mc = min(12, max(1, mins_left // 15))

            if h_mc < 4:
                rejected_ledger.append({
                    "timestamp": str(idx),
                    "ticker": ticker,
                    "reason": "HORIZON_TOO_SHORT (<4 bars)",
                    "prob": round(prob, 4)
                })
                continue

            # Continuous Margin-Synchronized Sizing
            risk_budget = compute_continuous_half_kelly(prob, min_budget, max_budget)
            if curr_sec_trades >= 1:
                risk_budget *= sec_penalty

            per_share_sl_dist = entry_p * sl_pct
            qty_raw = int(risk_budget / per_share_sl_dist) if per_share_sl_dist > 0 else 1
            qty_margin = int(notional_cap / entry_p)
            qty = max(1, min(qty_raw, qty_margin))

            # Time-Decay Target Scaling
            di_diff = float(row.get("di_diff", 1.0))
            adx_slope = float(row.get("adx_slope_3", 0.5))

            if bar_time <= pd.Timestamp("11:30").time() and prob >= 0.54 and di_diff > 0 and adx_slope > 0:
                eff_tp_mult = apex_tp_mult   # 2.65x ATR
                regime_tag = "APEX_EXPANDED"
            elif bar_time <= pd.Timestamp("12:15").time():
                eff_tp_mult = std_tp_mult    # 2.35x ATR
                regime_tag = "STANDARD_HIGH_VOL"
            else:
                eff_tp_mult = mid_tp_mult    # 1.95x ATR
                regime_tag = "MIDDAY_HORIZON_ADJUSTED"

            tp_pct = eff_tp_mult * atr_norm

            snapshot = {
                "timestamp": str(idx),
                "ticker": ticker,
                "close": raw_price,
                "atr_norm": atr_norm,
                "vwap_dev": float(row["vwap_dev"]),
                "above_vwap": int(row["above_vwap"]),
                "adx": float(row["adx"]),
                "rvol": float(row["rvol"]),
                "prob": prob,
                "qty": qty,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "hold_bars": h_mc,
                "capital_available": round(current_capital, 2),
                "position_size": round(qty * entry_p, 2),
                "risk_budget": round(risk_budget, 2),
                "sector": sec,
                "sub_sector": row["sub_sector"]
            }

            mc_eval = mc_engine.evaluate_candidate(snapshot)

            # Step-down fallback to 1.75x ATR if midday MC requires
            if mc_eval["decision"] != "ALLOW ENTRY" and eff_tp_mult == mid_tp_mult:
                eff_tp_mult = 1.75
                tp_pct = eff_tp_mult * atr_norm
                snapshot["tp_pct"] = tp_pct
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

            # Causal Trade Execution with 15:00 IST Profit-Lock Guard
            p_tp = entry_p * (1.0 + tp_pct)
            active_p_sl = entry_p * (1.0 - sl_pct)
            trade_hit = False
            exit_price = None
            profit_locked = False

            if ticker in ticker_dfs:
                t_df = ticker_dfs[ticker]
                post_bars = t_df[t_df.index > idx]
                post_bars_day = post_bars[post_bars.index.date == d]
                check_bars = post_bars_day.head(h_mc)

                for _, bar_r in check_bars.iterrows():
                    b_time = bar_r.name.time() if hasattr(bar_r.name, 'time') else pd.Timestamp(str(bar_r.name)).time()

                    # 15:00 IST Profit-Lock Guard
                    if b_time >= lock_time and not profit_locked:
                        curr_gain = (bar_r["close"] - entry_p) / entry_p
                        if curr_gain >= lock_min_atr * atr_norm:
                            lock_price = entry_p + (lock_buf_rs / qty)
                            active_p_sl = max(active_p_sl, lock_price)
                            profit_locked = True

                    if bar_r["high"] >= p_tp:
                        exit_price = p_tp * (1.0 - slippage)
                        trade_hit = True
                        break
                    elif bar_r["low"] <= active_p_sl:
                        exit_price = active_p_sl * (1.0 - slippage)
                        trade_hit = True
                        break

                if not trade_hit and len(check_bars) > 0:
                    exit_price = float(check_bars.iloc[-1]["close"]) * (1.0 - slippage)

            if exit_price is None:
                realized_ret = float(row.get("realized_pct", 0.0))
                exit_price = entry_p * (1.0 + realized_ret) * (1.0 - slippage)

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

            trade_ledger.append({
                "timestamp": str(idx),
                "ticker": ticker,
                "regime": regime_tag,
                "sector": sec,
                "sub_sector": row["sub_sector"],
                "direction": "BUY",
                "confidence": round(prob * 100, 2),
                "mc_win_prob": round(mc_eval["stats"].get("win_prob", 0.0) * 100, 2),
                "mc_expected_ev": round(mc_eval["stats"].get("expected_net_ev", 0.0), 2),
                "entry_price": round(entry_p, 2),
                "exit_price": round(exit_price, 2),
                "tp_multiple": eff_tp_mult,
                "horizon_bars": h_mc,
                "profit_locked": profit_locked,
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

    trades_csv_path = os.path.join(out_dirs["trades"], "v29_trades.csv")
    df_trades.to_csv(trades_csv_path, index=False)

    rejected_csv_path = os.path.join(out_dirs["trades"], "v29_rejected_candidates.csv")
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
