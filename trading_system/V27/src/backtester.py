"""
Backtester Module for Bhavishya V27:
Two-Tier Execution (Tier 1 Direct Fill + Tier 2 MC Rescue Pullback) & Uncapped Post-MC Daily Slot Architecture.
Ensures zero lookahead, realistic Dhan friction, Half-Kelly sizing, and risk controls.
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

def compute_half_kelly_budget(prob: float, base_budget: float, b: float, min_b: float, max_b: float) -> float:
    p = prob
    q = 1.0 - p
    edge = p * b - q
    f_star = edge / b if b > 0 else 0.0
    f_half = max(0.0, f_star / 2.0)
    budget = base_budget * (1.0 + f_half)
    return round(float(np.clip(budget, min_b, max_b)), 2)

def execute_backtest(oos_df: pd.DataFrame, df_pool: pd.DataFrame, config: dict, out_dirs: dict) -> dict:
    capital      = float(config.get("capital", 500.0))
    leverage     = float(config.get("leverage", 5.0))
    base_budget  = float(config.get("base_risk_budget", 20.0))
    min_budget   = float(config.get("min_risk_budget", 18.0))
    max_budget   = float(config.get("max_risk_budget", 35.0))
    payoff_b     = float(config.get("payoff_b", 1.333))
    conv_thr     = float(config.get("conviction_threshold", 0.45))
    tier2_conv   = float(config.get("tier2_conviction_threshold", 0.48))
    pullback_atr = float(config.get("pullback_offset_atr", 0.35))
    pullback_bars= int(config.get("pullback_max_bars", 2))
    daily_n      = int(config.get("daily_top_n", 3))
    max_per_sec  = int(config.get("max_trades_per_sector", 2))
    sec_penalty  = float(config.get("second_sector_entry_penalty", 0.80))
    max_d_loss   = int(config.get("max_daily_loss", 2))
    sl_atr_mult  = float(config.get("sl_atr_mult", 1.5))
    slippage     = float(config.get("slippage_per_leg", 0.0005))
    hold_bars    = int(config.get("hold_bars", 12))

    # Pre-Trade Monte Carlo Engine
    mc_engine = PreTradeMonteCarloEngine(config)
    tier2_min_mc_win = float(config.get("pre_trade_mc", {}).get("tier2_min_win_prob", 0.45))

    # Build ticker price history lookup for causal pullback execution
    ticker_dfs = {}
    for ticker, g in df_pool.groupby("ticker"):
        ticker_dfs[ticker] = g.sort_index()

    # Candidates filtered by primary conviction gate
    candidates = oos_df[oos_df["prob"] >= conv_thr].copy()
    candidates["trade_date"] = candidates.index.date
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

    # Uncapped candidate evaluation sorted by prob descending per day
    for trade_date, df_day in candidates.groupby("trade_date"):
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
            tp_pct    = float(row["target_pct"])
            sl_pct    = float(row["sl_pct"])
            prob      = float(row["prob"])

            # Compute Half-Kelly risk budget
            risk_budget = compute_half_kelly_budget(prob, base_budget, payoff_b, min_budget, max_budget)
            if curr_sec_trades >= 1:
                risk_budget *= sec_penalty

            max_leverage_capital = current_capital * leverage
            per_share_sl_dist = entry_p * sl_pct
            qty = max(1, min(int(risk_budget / per_share_sl_dist), int(max_leverage_capital / entry_p)))

            snapshot_t1 = {
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
                "hold_bars": hold_bars,
                "capital_available": round(current_capital, 2),
                "position_size": round(qty * entry_p, 2),
                "risk_budget": round(risk_budget, 2),
                "sector": sec,
                "sub_sector": row["sub_sector"]
            }

            # Evaluate Tier 1 (Bar Close Entry)
            mc_eval_t1 = mc_engine.evaluate_candidate(snapshot_t1)
            mc_win_prob_t1 = mc_eval_t1["stats"].get("win_prob", 0.0)
            f_jsonl.write(json.dumps(mc_eval_t1) + "\n")

            executed_trade = None

            if mc_eval_t1["decision"] == "ALLOW ENTRY":
                # Tier 1 Direct Execution
                realized_ret = float(row["realized_pct"])
                exit_p    = entry_p * (1.0 + realized_ret) * (1.0 - slippage)
                gross_pnl = (exit_p - entry_p) * qty
                charges   = calc_dhan_charges(entry_p, exit_p, qty)
                net_pnl   = round(gross_pnl - charges, 2)
                executed_trade = {
                    "timestamp": str(idx),
                    "ticker": ticker,
                    "tier": "TIER_1_DIRECT",
                    "sector": sec,
                    "sub_sector": row["sub_sector"],
                    "direction": "BUY",
                    "confidence": round(prob * 100, 2),
                    "mc_win_prob": round(mc_win_prob_t1 * 100, 2),
                    "mc_expected_ev": round(mc_eval_t1["stats"].get("expected_net_ev", 0.0), 2),
                    "entry_price": round(entry_p, 2),
                    "exit_price": round(exit_p, 2),
                    "qty": qty,
                    "position_exposure": round(qty * entry_p, 2),
                    "half_kelly_risk_budget": round(risk_budget, 2),
                    "gross_pnl": round(gross_pnl, 2),
                    "charges": charges,
                    "net_pnl": net_pnl,
                    "capital": round(current_capital + net_pnl, 2),
                    "winner": net_pnl > 0,
                    "realized_return_pct": round(realized_ret * 100, 2)
                }
            elif prob >= tier2_conv and mc_win_prob_t1 >= tier2_min_mc_win:
                # Tier 2 Pullback Rescue Evaluation
                p_limit = raw_price * (1.0 - pullback_atr * atr_norm)
                p_sl = raw_price * (1.0 - sl_pct)
                p_tp = raw_price * (1.0 + tp_pct)
                sl_pct_limit = (p_limit - p_sl) / p_limit
                tp_pct_limit = (p_tp - p_limit) / p_limit

                per_share_sl_lim = p_limit - p_sl
                qty_lim = max(1, min(int(risk_budget / per_share_sl_lim), int(max_leverage_capital / p_limit)))

                snapshot_t2 = snapshot_t1.copy()
                snapshot_t2["close"] = p_limit
                snapshot_t2["tp_pct"] = tp_pct_limit
                snapshot_t2["sl_pct"] = sl_pct_limit
                snapshot_t2["qty"] = qty_lim
                mc_eval_t2 = mc_engine.evaluate_candidate(snapshot_t2)
                mc_win_prob_t2 = mc_eval_t2["stats"].get("win_prob", 0.0)

                # Tier 2 positive expectancy condition
                is_tier2_approved = (
                    mc_eval_t2["stats"].get("expected_ev", -1.0) > 0.0 and
                    mc_eval_t2["stats"].get("cvar_loss", -100.0) >= -45.0 and
                    mc_win_prob_t2 >= tier2_min_mc_win
                )

                if is_tier2_approved and ticker in ticker_dfs:
                    t_df = ticker_dfs[ticker]
                    post_bars = t_df[t_df.index > idx]
                    post_bars_day = post_bars[post_bars.index.date == d]

                    filled = False
                    fill_idx = None
                    if len(post_bars_day) >= 1:
                        b1 = post_bars_day.iloc[0]
                        if b1["low"] <= p_limit:
                            if b1["low"] > p_sl:
                                filled = True
                                fill_idx = post_bars_day.index[0]
                    if not filled and len(post_bars_day) >= pullback_bars:
                        b2 = post_bars_day.iloc[1]
                        if b2["low"] <= p_limit:
                            if b2["low"] > p_sl:
                                filled = True
                                fill_idx = post_bars_day.index[1]

                    if filled:
                        bars_remaining = post_bars_day[post_bars_day.index >= fill_idx]
                        trade_hit = False
                        exit_price_lim = None
                        for _, bar_r in bars_remaining.iterrows():
                            if bar_r["high"] >= p_tp:
                                exit_price_lim = p_tp * (1.0 - slippage)
                                trade_hit = True
                                break
                            elif bar_r["low"] <= p_sl:
                                exit_price_lim = p_sl * (1.0 - slippage)
                                trade_hit = True
                                break
                        if not trade_hit:
                            exit_price_lim = float(bars_remaining.iloc[-1]["close"]) * (1.0 - slippage)

                        entry_fill = p_limit * (1.0 + slippage)
                        gross_pnl = (exit_price_lim - entry_fill) * qty_lim
                        charges = calc_dhan_charges(entry_fill, exit_price_lim, qty_lim)
                        net_pnl = round(gross_pnl - charges, 2)
                        realized_ret_lim = (exit_price_lim - entry_fill) / entry_fill
                        executed_trade = {
                            "timestamp": str(idx),
                            "ticker": ticker,
                            "tier": "TIER_2_PULLBACK",
                            "sector": sec,
                            "sub_sector": row["sub_sector"],
                            "direction": "BUY",
                            "confidence": round(prob * 100, 2),
                            "mc_win_prob": round(mc_win_prob_t2 * 100, 2),
                            "mc_expected_ev": round(mc_eval_t2["stats"].get("expected_net_ev", 0.0), 2),
                            "entry_price": round(entry_fill, 2),
                            "exit_price": round(exit_price_lim, 2),
                            "qty": qty_lim,
                            "position_exposure": round(qty_lim * entry_fill, 2),
                            "half_kelly_risk_budget": round(risk_budget, 2),
                            "gross_pnl": round(gross_pnl, 2),
                            "charges": charges,
                            "net_pnl": net_pnl,
                            "capital": round(current_capital + net_pnl, 2),
                            "winner": net_pnl > 0,
                            "realized_return_pct": round(realized_ret_lim * 100, 2)
                        }

            if executed_trade is not None:
                trade_ledger.append(executed_trade)
                current_capital = round(current_capital + executed_trade["net_pnl"], 2)
                daily_executed_count[d] = daily_executed_count.get(d, 0) + 1
                daily_sector_trade_count[day_sec_key] = curr_sec_trades + 1
                if not executed_trade["winner"]:
                    daily_losses[d] = daily_losses.get(d, 0) + 1
            else:
                rejected_ledger.append({
                    "timestamp": str(idx),
                    "ticker": ticker,
                    "reason": f"PRE_TRADE_MC_{mc_eval_t1['decision']}: {mc_eval_t1['reason']}",
                    "prob": round(prob, 4)
                })

    f_jsonl.close()

    df_trades = pd.DataFrame(trade_ledger)
    df_rejected = pd.DataFrame(rejected_ledger)

    trades_csv_path = os.path.join(out_dirs["trades"], "v27_trades.csv")
    df_trades.to_csv(trades_csv_path, index=False)

    rejected_csv_path = os.path.join(out_dirs["trades"], "v27_rejected_candidates.csv")
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
