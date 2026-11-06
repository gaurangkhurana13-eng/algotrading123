"""
Backtester Engine with Dual-Target Asymmetric Scale-Out (Bhavishya V15)
- Leg 1 (50% Qty): Exits at 1.6 * ATR Target
- On Leg 1 fill: Stop Loss on Leg 2 is moved to Breakeven
- Leg 2 (50% Qty): Aims for 2.6 * ATR Target
- Initial Stop: 1.5 * ATR for both legs
"""

import os
import json
import numpy as np
import pandas as pd
from .dhan_engine import calc_dhan_charges
from .pre_trade_mc import PreTradeMonteCarloEngine

def get_tiered_risk_budget(capital: float, tiers: list) -> float:
    for tier in tiers:
        if capital < tier["max_capital"]:
            return float(tier["risk_budget"])
    return float(tiers[-1]["risk_budget"])

def execute_backtest(oos_df: pd.DataFrame, config: dict, out_dirs: dict) -> dict:
    conv_thr   = config.get("conviction_threshold", 0.75)
    daily_n    = config.get("daily_top_n", 3)
    max_d_loss = config.get("max_daily_loss", 2)
    leverage   = config.get("leverage", 5.0)
    tiers      = config.get("tiers", [{"max_capital": 999999.0, "risk_budget": 20.0}])
    slippage   = config.get("slippage_per_leg", 0.0005)
    capital    = config.get("capital", 500.0)
    sl_atr_mult= config.get("sl_atr_mult", 1.5)
    tp1_mult   = config.get("tp1_atr_mult", 1.6)
    tp2_mult   = config.get("tp2_atr_mult", 2.6)

    mc_engine = PreTradeMonteCarloEngine(config)

    candidates = oos_df[oos_df["prob"] >= conv_thr].copy()
    candidates["trade_date"] = candidates.index.date

    daily_cands = candidates.groupby("trade_date", group_keys=False).apply(
        lambda g: g.sort_values("prob", ascending=False).head(daily_n)
    )

    trade_ledger = []
    rejected_ledger = []
    daily_losses = {}

    pre_mc_jsonl_path = os.path.join(out_dirs["monte_carlo"], "pre_trade_decisions.jsonl")
    f_jsonl = open(pre_mc_jsonl_path, "w", encoding="utf-8")

    current_capital = capital

    for idx, row in daily_cands.iterrows():
        d = idx.date() if hasattr(idx, 'date') else pd.Timestamp(str(idx)).date()
        if daily_losses.get(d, 0) >= max_d_loss:
            rejected_ledger.append({
                "timestamp": str(idx),
                "ticker": row["ticker"],
                "reason": "CIRCUIT_BREAKER_MAX_DAILY_LOSS",
                "prob": round(row["prob"], 4)
            })
            continue

        raw_price = float(row["close"])
        entry_p   = raw_price * (1.0 + slippage)
        atr_norm  = float(row["atr_norm"])

        risk_budget = get_tiered_risk_budget(current_capital, tiers)
        max_leverage_capital = current_capital * leverage

        per_share_sl_dist = entry_p * (sl_atr_mult * atr_norm)
        if per_share_sl_dist <= 0:
            qty = 1
        else:
            raw_qty = int(risk_budget / per_share_sl_dist)
            cap_limit_qty = int(max_leverage_capital / entry_p)
            qty = max(2, min(raw_qty, cap_limit_qty))

        snapshot = {
            "timestamp": str(idx),
            "ticker": row["ticker"],
            "close": raw_price,
            "atr_norm": atr_norm,
            "vwap_dev": float(row["vwap_dev"]),
            "above_vwap": int(row["above_vwap"]),
            "adx": float(row["adx"]),
            "rvol": float(row["rvol"]),
            "prob": float(row["prob"]),
            "qty": qty,
            "tp_pct": tp1_mult * atr_norm,
            "sl_pct": sl_atr_mult * atr_norm,
            "hold_bars": int(config.get("hold_bars", 12)),
            "capital_available": round(current_capital, 2),
            "position_size": round(qty * entry_p, 2),
            "risk_budget": risk_budget
        }

        mc_eval = mc_engine.evaluate_candidate(snapshot)
        f_jsonl.write(json.dumps(mc_eval) + "\n")

        if mc_eval["decision"] != "ALLOW ENTRY":
            rejected_ledger.append({
                "timestamp": str(idx),
                "ticker": row["ticker"],
                "reason": f"PRE_TRADE_MC_{mc_eval['decision']}: {mc_eval['reason']}",
                "prob": round(row["prob"], 4)
            })
            continue

        # Realized exit simulation with dual scale-out
        realized_ret = float(row["realized_pct"])
        tp1_ret = tp1_mult * atr_norm
        tp2_ret = tp2_mult * atr_norm
        sl_ret  = -sl_atr_mult * atr_norm

        qty_1 = max(1, qty // 2)
        qty_2 = qty - qty_1

        # Check path outcome based on realized return
        if realized_ret >= tp2_ret:
            # Reached TP2: Leg 1 exits at TP1, Leg 2 exits at TP2
            exit_p1 = entry_p * (1.0 + tp1_ret) * (1.0 - slippage)
            exit_p2 = entry_p * (1.0 + tp2_ret) * (1.0 - slippage)
            gross_pnl = (exit_p1 - entry_p) * qty_1 + (exit_p2 - entry_p) * qty_2
            charges   = calc_dhan_charges(entry_p, exit_p1, qty_1) + calc_dhan_charges(entry_p, exit_p2, qty_2)
            exit_p    = (exit_p1 * qty_1 + exit_p2 * qty_2) / qty
            exit_type = "TP1_AND_TP2"
        elif realized_ret >= tp1_ret:
            # Reached TP1, Leg 2 trailed to breakeven
            exit_p1 = entry_p * (1.0 + tp1_ret) * (1.0 - slippage)
            exit_p2 = entry_p * (1.0 - slippage)
            gross_pnl = (exit_p1 - entry_p) * qty_1 + (exit_p2 - entry_p) * qty_2
            charges   = calc_dhan_charges(entry_p, exit_p1, qty_1) + calc_dhan_charges(entry_p, exit_p2, qty_2)
            exit_p    = (exit_p1 * qty_1 + exit_p2 * qty_2) / qty
            exit_type = "TP1_AND_BREAKEVEN"
        elif realized_ret <= sl_ret:
            # Hit Stop Loss on entire position
            exit_p = entry_p * (1.0 + sl_ret) * (1.0 - slippage)
            gross_pnl = (exit_p - entry_p) * qty
            charges = calc_dhan_charges(entry_p, exit_p, qty)
            exit_type = "STOP_LOSS"
        else:
            # Market close
            exit_p = entry_p * (1.0 + realized_ret) * (1.0 - slippage)
            gross_pnl = (exit_p - entry_p) * qty
            charges = calc_dhan_charges(entry_p, exit_p, qty)
            exit_type = "MARKET_TIME_EXIT"

        net_pnl   = round(gross_pnl - charges, 2)
        current_capital = round(current_capital + net_pnl, 2)
        is_win    = net_pnl > 0

        if not is_win:
            daily_losses[d] = daily_losses.get(d, 0) + 1

        trade_ledger.append({
            "timestamp": str(idx),
            "ticker": row["ticker"],
            "direction": "BUY",
            "confidence": round(row["prob"] * 100, 2),
            "mc_win_prob": round(mc_eval["stats"].get("win_prob", 0.0) * 100, 2),
            "mc_expected_ev": round(mc_eval["stats"].get("expected_net_ev", 0.0), 2),
            "entry_price": round(entry_p, 2),
            "exit_price": round(exit_p, 2),
            "qty": qty,
            "exit_type": exit_type,
            "position_exposure": round(qty * entry_p, 2),
            "risk_budget": risk_budget,
            "gross_pnl": round(gross_pnl, 2),
            "charges": charges,
            "net_pnl": net_pnl,
            "capital": current_capital,
            "winner": is_win,
            "realized_return_pct": round(realized_ret * 100, 2)
        })

    f_jsonl.close()

    df_trades   = pd.DataFrame(trade_ledger)
    df_rejected = pd.DataFrame(rejected_ledger)

    trades_path = os.path.join(out_dirs["trades"], f"{config['version'].lower()}_trades.csv")
    rejections_path = os.path.join(out_dirs["trades"], f"{config['version'].lower()}_rejected_candidates.csv")
    df_trades.to_csv(trades_path, index=False)
    df_rejected.to_csv(rejections_path, index=False)

    n_trades = len(df_trades)
    n_rejected = len(df_rejected)
    n_candidates = len(daily_cands)

    if n_trades == 0:
        return {
            "n_candidates": n_candidates, "n_approved": 0, "n_rejected": n_rejected,
            "n_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "payoff_ratio": 0.0,
            "profit_factor": 0.0, "charges": 0.0, "total_pnl": 0.0,
            "final_capital": current_capital, "roi": 0.0, "sharpe": 0.0, "sortino": 0.0,
            "df_trades": df_trades, "df_rejected": df_rejected
        }

    wins = df_trades["winner"].sum()
    losses = n_trades - wins
    win_rate = (wins / n_trades) * 100.0
    total_net = df_trades["net_pnl"].sum()
    total_fees = df_trades["charges"].sum()
    roi = ((current_capital - capital) / capital) * 100.0

    gross_win_sum = df_trades[df_trades["net_pnl"] > 0]["net_pnl"].sum()
    gross_loss_sum = abs(df_trades[df_trades["net_pnl"] < 0]["net_pnl"].sum())
    pf = round(gross_win_sum / (gross_loss_sum + 1e-6), 2)
    avg_win = df_trades[df_trades["winner"]]["net_pnl"].mean() if wins > 0 else 0.0
    avg_loss = df_trades[~df_trades["winner"]]["net_pnl"].mean() if losses > 0 else 0.0
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    pnls = df_trades["net_pnl"].values
    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls)
    downside_std = np.std(pnls[pnls < 0]) if np.any(pnls < 0) else 1e-6
    sharpe = round(mean_pnl / (std_pnl + 1e-6) * np.sqrt(252 * 2), 2)
    sortino = round(mean_pnl / (downside_std + 1e-6) * np.sqrt(252 * 2), 2)

    return {
        "n_candidates": n_candidates, "n_approved": n_trades, "n_rejected": n_rejected,
        "n_trades": n_trades, "wins": int(wins), "losses": int(losses), "win_rate": round(win_rate, 2),
        "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2), "payoff_ratio": round(payoff, 2),
        "profit_factor": pf, "total_pnl": round(total_net, 2), "charges": round(total_fees, 2),
        "final_capital": round(current_capital, 2), "roi": round(roi, 2),
        "sharpe": sharpe, "sortino": sortino,
        "df_trades": df_trades, "df_rejected": df_rejected
    }

def run_global_monte_carlo(df_trades: pd.DataFrame, config: dict, out_dirs: dict) -> dict:
    if len(df_trades) < 5:
        return {"ruin_prob": 0.0, "avg_max_dd": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0}

    n_sims = config.get("global_monte_carlo", {}).get("iterations", 10000)
    seed = config.get("global_monte_carlo", {}).get("seed", 42)
    rng = np.random.default_rng(seed)

    pnls = df_trades["net_pnl"].values
    n_trades = len(pnls)
    capital = config.get("capital", 500.0)

    final_caps = []
    max_dds = []
    ruin_count = 0

    for _ in range(n_sims):
        sim_pnls = rng.choice(pnls, size=n_trades, replace=True)
        curve = capital + np.cumsum(sim_pnls)
        peak = np.maximum.accumulate(curve)
        dd = (peak - curve) / (peak + 1e-6)
        max_dds.append(np.max(dd) * 100.0)
        final_caps.append(curve[-1])
        if np.min(curve) <= capital * 0.50:
            ruin_count += 1

    ruin_prob = round((ruin_count / n_sims) * 100.0, 2)
    p5  = round(float(np.percentile(final_caps, 5)), 2)
    p50 = round(float(np.percentile(final_caps, 50)), 2)
    p95 = round(float(np.percentile(final_caps, 95)), 2)
    avg_dd = round(float(np.mean(max_dds)), 2)

    mc_res = {
        "n_sims": n_sims,
        "ruin_prob_50pct": ruin_prob,
        "average_max_drawdown": avg_dd,
        "p5_worst_capital": p5,
        "p50_median_capital": p50,
        "p95_best_capital": p95
    }

    out_path = os.path.join(out_dirs["monte_carlo"], "global_monte_carlo.json")
    with open(out_path, "w") as f:
        json.dump(mc_res, f, indent=2)

    return mc_res
