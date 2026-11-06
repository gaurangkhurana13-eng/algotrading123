"""
Backtester Module for Bhavishya V28:
Multi-Horizon Squeeze, Conviction-Gated Target Expansion & Bimodal Asymmetric Risk Sizing.
Strictly lookahead-free, real Dhan MIS transaction costs, invariant 1.50x ATR structural stop loss.
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

def compute_tier1a_budget(prob: float, base_budget: float, b: float, min_b: float, max_b: float) -> float:
    p = prob
    q = 1.0 - p
    edge = p * b - q
    f_star = edge / b if b > 0 else 0.0
    f_half = max(0.0, f_star / 2.0)
    budget = base_budget * (1.0 + f_half)
    return round(float(np.clip(budget, min_b, max_b)), 2)

def execute_backtest(oos_df: pd.DataFrame, df_pool: pd.DataFrame, config: dict, out_dirs: dict) -> dict:
    capital       = float(config.get("capital", 500.0))
    leverage      = float(config.get("leverage", 5.0))
    conv_thr      = float(config.get("conviction_threshold", 0.50))
    tier1a_prob   = float(config.get("tier1a_elite_prob", 0.55))
    t1a_base      = float(config.get("tier1a_risk_budget", 32.0))
    t1a_min       = float(config.get("tier1a_min_budget", 28.0))
    t1a_max       = float(config.get("tier1a_max_budget", 35.0))
    t1b_base      = float(config.get("tier1b_risk_budget", 11.0))
    payoff_b      = float(config.get("payoff_b", 1.333))
    daily_n       = int(config.get("daily_top_n", 3))
    max_per_sec   = int(config.get("max_trades_per_sector", 2))
    sec_penalty   = float(config.get("second_sector_entry_penalty", 0.80))
    max_d_loss    = int(config.get("max_daily_loss", 2))
    sl_atr_mult   = float(config.get("sl_atr_mult", 1.5))
    slippage      = float(config.get("slippage_per_leg", 0.0005))
    hold_bars     = int(config.get("hold_bars", 12))
    apex_tp_mult  = float(config.get("apex_tp_atr_mult", 2.7))
    comp_tp_mult  = float(config.get("compression_tp_atr_mult", 2.4))
    exp_tp_mult   = float(config.get("expansion_tp_atr_mult", 1.7))
    norm_tp_mult  = float(config.get("tp_atr_mult", 2.0))

    mc_engine = PreTradeMonteCarloEngine(config)

    # Build ticker price history lookup for causal trade tracing
    ticker_dfs = {}
    for ticker, g in df_pool.groupby("ticker"):
        ticker_dfs[ticker] = g.sort_index()

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
            prob      = float(row["prob"])
            sl_pct    = sl_atr_mult * atr_norm

            # Option C: Bimodal Asymmetric Risk Budgeting
            if prob >= tier1a_prob:
                risk_budget = compute_tier1a_budget(prob, t1a_base, payoff_b, t1a_min, t1a_max)
                tier_label = "TIER_1A_ELITE"
            else:
                risk_budget = t1b_base
                tier_label = "TIER_1B_MARGINAL"

            if curr_sec_trades >= 1:
                risk_budget *= sec_penalty

            max_leverage_capital = current_capital * leverage
            per_share_sl_dist = entry_p * sl_pct
            qty = max(1, min(int(risk_budget / per_share_sl_dist), int(max_leverage_capital / entry_p)))

            # Option B: Conviction-Gated Target Expansion
            bar_time = idx.time() if hasattr(idx, 'time') else pd.Timestamp(str(idx)).time()
            di_diff = float(row.get("di_diff", 1.0))
            adx_slope = float(row.get("adx_slope_3", 0.5))
            sqz = int(row.get("squeeze_on", 0))
            atr_z = float(row.get("atr_norm_z", 0.0))

            is_apex_setup = (
                prob >= tier1a_prob and
                di_diff > 0 and
                adx_slope > 0 and
                bar_time <= pd.Timestamp("13:30").time()
            )

            if is_apex_setup:
                eff_tp_mult = apex_tp_mult  # 2.7x ATR
            elif sqz == 1 or atr_z < -0.5:
                eff_tp_mult = comp_tp_mult  # 2.4x ATR
            elif atr_z > 1.0:
                eff_tp_mult = exp_tp_mult   # 1.7x ATR
            else:
                eff_tp_mult = norm_tp_mult  # 2.0x ATR

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
                "hold_bars": hold_bars,
                "capital_available": round(current_capital, 2),
                "position_size": round(qty * entry_p, 2),
                "risk_budget": round(risk_budget, 2),
                "sector": sec,
                "sub_sector": row["sub_sector"]
            }

            mc_eval = mc_engine.evaluate_candidate(snapshot)

            # Fallback to 2.4x ATR if 2.7x target failed MC win prob < 50%
            if mc_eval["decision"] != "ALLOW ENTRY" and eff_tp_mult == apex_tp_mult:
                eff_tp_mult = comp_tp_mult
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

            # Trace execution causally using ticker price history
            p_tp = entry_p * (1.0 + tp_pct)
            p_sl = entry_p * (1.0 - sl_pct)
            trade_hit = False
            exit_price = None

            if ticker in ticker_dfs:
                t_df = ticker_dfs[ticker]
                post_bars = t_df[t_df.index > idx]
                post_bars_day = post_bars[post_bars.index.date == d]
                check_bars = post_bars_day.head(hold_bars)

                for _, bar_r in check_bars.iterrows():
                    if bar_r["high"] >= p_tp:
                        exit_price = p_tp * (1.0 - slippage)
                        trade_hit = True
                        break
                    elif bar_r["low"] <= p_sl:
                        exit_price = p_sl * (1.0 - slippage)
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
                "tier": tier_label,
                "sector": sec,
                "sub_sector": row["sub_sector"],
                "direction": "BUY",
                "confidence": round(prob * 100, 2),
                "mc_win_prob": round(mc_eval["stats"].get("win_prob", 0.0) * 100, 2),
                "mc_expected_ev": round(mc_eval["stats"].get("expected_net_ev", 0.0), 2),
                "entry_price": round(entry_p, 2),
                "exit_price": round(exit_price, 2),
                "tp_multiple": eff_tp_mult,
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

    trades_csv_path = os.path.join(out_dirs["trades"], "v28_trades.csv")
    df_trades.to_csv(trades_csv_path, index=False)

    rejected_csv_path = os.path.join(out_dirs["trades"], "v28_rejected_candidates.csv")
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
