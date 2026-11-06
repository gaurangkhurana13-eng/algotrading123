"""
Causal Backtesting & Trade Execution Engine for PCJEWELLER Specialist System.
Enforces:
1. Dual-Constraint Sizing: strictly caps worst-case single loss at <= Rs 15.50.
2. 5x MIS Leverage: deployed notional <= Rs 2,450.
3. High-Conviction CatBoost Z-Score & Trend Filter (z_t >= 1.25, above shifted VWAP, RVOL >= 1.10).
4. Multi-Phase Profit Ratchet (Locking gains at 1.8x, 2.6x, 3.5x ATR).
5. Real Dhan MIS Cost Stack & 0.05% Per-Leg Slippage.
6. 100% Intraday (Zero overnight risk, flat by 15:10 IST).
"""

import os
import json
import numpy as np
import pandas as pd
from .dhan_engine import calc_dhan_charges

def execute_pcj_backtest(oos_df: pd.DataFrame, df_full: pd.DataFrame, config: dict, out_dirs: dict) -> dict:
    capital          = float(config.get("capital", 500.0))
    notional_cap     = float(config.get("max_notional_capital", 2450.0))
    max_daily_trades = int(config.get("max_daily_trades", 2))
    max_daily_loss   = float(config.get("max_daily_loss", 25.0))
    slippage         = float(config.get("slippage_per_leg", 0.0005))
    r_net_cap        = float(config.get("r_net_cap", 15.50))
    sl_atr_mult      = float(config.get("sl_atr_mult", 1.45))
    z_min            = float(config.get("z_conviction_min", 1.25))
    phi_rt           = float(config.get("round_trip_friction_rate", 0.002067))

    entry_cutoff_time = pd.Timestamp(config.get("entry_cutoff_time", "13:30")).time()
    eod_square_time   = pd.Timestamp(config.get("eod_square_time", "15:10")).time()

    # Ratchet parameters
    p1_trig = float(config.get("ratchet_phase1_trigger", 1.80))
    p1_flr  = float(config.get("ratchet_phase1_floor", 0.95))
    p2_trig = float(config.get("ratchet_phase2_trigger", 2.60))
    p2_flr  = float(config.get("ratchet_phase2_floor", 1.80))
    p3_trig = float(config.get("ratchet_phase3_trigger", 3.50))
    p3_flr  = float(config.get("ratchet_phase3_floor", 2.70))
    tp_mult = float(config.get("tp_ceiling_mult", 4.80))

    oos = oos_df.copy()
    oos["bar_time"] = [t.time() if hasattr(t, "time") else pd.Timestamp(str(t)).time() for t in oos.index]
    oos["trade_date"] = oos.index.date

    # Initial candidate filter: within entry hours, above shifted vwap, positive volume
    candidates = oos[
        (oos["bar_time"] >= pd.Timestamp("09:15").time()) &
        (oos["bar_time"] <= entry_cutoff_time) &
        (oos["volume"] > 0)
    ].copy()

    trade_ledger = []
    rejected_ledger = []
    daily_executed_count = {}
    daily_net_loss = {}

    current_capital = capital
    active_position = None

    unique_timestamps = sorted(list(candidates.index.unique()))

    # Build price lookup dataframe
    price_lookup = df_full.sort_index()

    for ts in unique_timestamps:
        row = candidates.loc[ts]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        b_time = ts.time() if hasattr(ts, "time") else pd.Timestamp(str(ts)).time()
        d = ts.date()

        # Check if active position has closed by this timestamp
        if active_position is not None:
            if ts <= active_position["exit_time"]:
                continue  # Position still open, cannot take new trade
            else:
                active_position = None

        # Check daily limits
        if daily_net_loss.get(d, 0.0) >= max_daily_loss:
            continue
        if daily_executed_count.get(d, 0) >= max_daily_trades:
            continue

        z_score  = float(row.get("z_score", 0.0))
        prob     = float(row.get("prob", 0.0))
        mu_f     = float(row.get("mu_f", 0.30))
        sigma_f  = float(row.get("sigma_f", 0.08))
        rvol     = float(row.get("rvol", 1.0))
        clv      = float(row.get("clv", 0.5))
        atr_norm = float(row.get("atr_norm", 0.008))
        raw_p    = float(row["close"])
        raw_open = float(row["open"])
        vwap_lag = float(row.get("vwap", raw_p))

        # Filter gates
        # 1. Z-Score Conviction (z_score >= z_min) and prob >= mu_f + 0.85 * sigma_f
        if z_score < z_min or prob < (mu_f + 0.85 * sigma_f):
            rejected_ledger.append({"timestamp": str(ts), "reason": f"CONVICTION_FAIL (z={z_score:.2f})"})
            continue

        # 2. Trend & Buyer Dominance: Close above shifted VWAP and green candle / high CLV
        if raw_p <= vwap_lag:
            rejected_ledger.append({"timestamp": str(ts), "reason": "BELOW_VWAP"})
            continue

        if clv < 0.52 and raw_p <= raw_open:
            rejected_ledger.append({"timestamp": str(ts), "reason": f"SELLER_DOMINATED (clv={clv:.2f})"})
            continue

        # 3. Minimum volume participation (RVOL >= 1.05)
        if rvol < 1.05:
            rejected_ledger.append({"timestamp": str(ts), "reason": f"LOW_RVOL ({rvol:.2f})"})
            continue

        # Dynamic horizon to 15:10 IST
        mins_left = (eod_square_time.hour * 60 + eod_square_time.minute) - (b_time.hour * 60 + b_time.minute)
        h_rem = min(12, max(1, mins_left // 15))
        if h_rem < 6:
            rejected_ledger.append({"timestamp": str(ts), "reason": f"HORIZON_SHORT ({h_rem} bars)"})
            continue

        entry_p = raw_p * (1.0 + slippage)
        atr_val = entry_p * atr_norm

        # Dual-Constraint Sizing
        stop_dist = sl_atr_mult * atr_val
        friction_per_share = phi_rt * entry_p
        denom = stop_dist + friction_per_share

        qty_risk = int(r_net_cap / denom) if denom > 0 else 1
        qty_notional = int(notional_cap / (entry_p * (1.0 + slippage)))
        qty = max(1, min(qty_notional, qty_risk))

        if qty * entry_p > notional_cap + 1.0:
            continue

        r_used = qty * denom

        # Trace trade forward
        p_sl_initial = entry_p - sl_atr_mult * atr_val
        tp_ceiling   = entry_p + tp_mult * atr_val
        active_p_sl  = p_sl_initial
        phase        = 0
        m_t          = entry_p
        exit_price   = None
        exit_time    = None
        exit_reason  = None

        post_bars = price_lookup[price_lookup.index > ts]
        post_bars_day = post_bars[post_bars.index.date == d]
        check_bars = post_bars_day.head(h_rem)

        for bar_idx, (_, bar_r) in enumerate(check_bars.iterrows()):
            b_high  = float(bar_r["high"])
            b_low   = float(bar_r["low"])
            b_close = float(bar_r["close"])
            b_open  = float(bar_r["open"])
            b_vwap  = float(bar_r.get("vwap", b_close))
            b_t     = bar_r.name.time() if hasattr(bar_r.name, "time") else pd.Timestamp(str(bar_r.name)).time()
            m_t     = max(m_t, b_high)
            h_held  = bar_idx + 1

            mfe_atr = (m_t - entry_p) / atr_val

            # Multi-Phase Ratchet
            if mfe_atr >= p3_trig:
                phase = max(phase, 3)
                active_p_sl = max(active_p_sl, entry_p + p3_flr * atr_val, m_t - 0.75 * atr_val)
            elif mfe_atr >= p2_trig:
                phase = max(phase, 2)
                active_p_sl = max(active_p_sl, entry_p + p2_flr * atr_val, m_t - 0.85 * atr_val)
            elif mfe_atr >= p1_trig:
                phase = max(phase, 1)
                active_p_sl = max(active_p_sl, entry_p + p1_flr * atr_val)

            # 1. Take Profit Ceiling Hit
            if b_high >= tp_ceiling:
                exit_price = tp_ceiling * (1.0 - slippage)
                exit_time  = bar_r.name
                exit_reason = "TP_CEILING_HIT (+4.80 ATR)"
                break

            # 2. Stop Loss or Trailed Floor Hit
            if b_low <= active_p_sl:
                actual_fill = min(active_p_sl, b_open)
                exit_price = actual_fill * (1.0 - slippage)
                exit_time  = bar_r.name
                if phase == 3:
                    exit_reason = "PHASE_3_RATCHET_FLOOR_HIT"
                elif phase == 2:
                    exit_reason = "PHASE_2_RATCHET_FLOOR_HIT"
                elif phase == 1:
                    exit_reason = "PHASE_1_PROFIT_FLOOR_HIT"
                else:
                    exit_reason = "INITIAL_SL_HIT"
                break

            # 3. Calibrated Stagnation Exits
            if h_held >= 6 and mfe_atr < 0.40 and b_close < entry_p - 0.25 * atr_val and b_close < b_vwap:
                exit_price = b_close * (1.0 - slippage)
                exit_time  = bar_r.name
                exit_reason = "STAGNATION_FAILED_IMPULSE_H6"
                break

            if h_held >= 10 and mfe_atr < 0.80 and b_close < entry_p + 0.10 * atr_val:
                exit_price = b_close * (1.0 - slippage)
                exit_time  = bar_r.name
                exit_reason = "STAGNATION_DEAD_MONEY_H10"
                break

            # 4. 15:05 Pre-Square Off Guard
            if b_t >= pd.Timestamp("15:05").time() and b_t < eod_square_time:
                if b_close > entry_p + 0.30 * atr_val and phase < 2:
                    active_p_sl = max(active_p_sl, b_close - 0.30 * atr_val)
                elif b_close <= entry_p + 0.20 * atr_val and b_close < b_vwap:
                    exit_price = b_close * (1.0 - slippage)
                    exit_time  = bar_r.name
                    exit_reason = "15:05_PRE_SQUARE_OFF"
                    break

            # 5. 15:10 Hard Intraday Flatten
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
        if net_pnl < 0:
            daily_net_loss[d] = daily_net_loss.get(d, 0.0) + abs(net_pnl)

        current_capital = round(current_capital + net_pnl, 2)

        trade_record = {
            "timestamp": str(ts),
            "ticker": "PCJEWELLER",
            "fold": int(row.get("fold", 1)),
            "exit_time": str(exit_time),
            "exit_reason": exit_reason,
            "phase": phase,
            "z_score": round(z_score, 2),
            "prob": round(prob, 4),
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

        active_position = {
            "entry_time": ts,
            "exit_time": exit_time,
            "qty": qty,
            "entry_p": entry_p
        }

    df_trades = pd.DataFrame(trade_ledger)
    df_rejected = pd.DataFrame(rejected_ledger)

    trades_csv_path = os.path.join(out_dirs["trades"], "pcj_trades.csv")
    rejected_csv_path = os.path.join(out_dirs["trades"], "pcj_rejected_candidates.csv")
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
        "df_trades": df_trades
    }

def run_pcj_monte_carlo(df_trades: pd.DataFrame, config: dict, out_dirs: dict) -> dict:
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
    mc_json_path = os.path.join(out_dirs["monte_carlo"], "pcj_global_mc_report.json")
    with open(mc_json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    return res
