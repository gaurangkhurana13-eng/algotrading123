"""
Backtester Module for Bhavishya V31:
Two-Part Velocity + Stagnation Recycling Engine, Friction-Inclusive Inverse-ATR Cap,
Three-Gear Asymmetric Ratchet & Margin Recycling.
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
    max_concurrent   = int(config.get("max_concurrent_open", 3))
    max_per_sec      = int(config.get("max_trades_per_sector", 2))
    sec_penalty      = float(config.get("second_sector_entry_penalty", 0.80))
    max_d_loss       = int(config.get("max_daily_loss", 2))
    sl_atr_mult      = float(config.get("sl_atr_mult", 1.5))
    slippage         = float(config.get("slippage_per_leg", 0.0005))
    min_friction_atr = float(config.get("min_friction_atr_ratio", 0.0038))

    # Risk Caps
    r_cap_core       = float(config.get("r_net_cap_core", 15.50))
    r_cap_high_atr   = float(config.get("r_net_cap_high_atr", 12.00))
    high_atr_thr     = float(config.get("high_atr_threshold", 0.0080))

    # Ratchet Specifications
    gear1_trig       = float(config.get("ratchet_gear1_trigger", 1.65))
    gear1_flr        = float(config.get("ratchet_gear1_floor", 1.35))
    gear2_trig       = float(config.get("ratchet_gear2_trigger", 2.30))
    gear2_flr        = float(config.get("ratchet_gear2_floor", 1.95))
    ratchet_trail    = float(config.get("ratchet_trail_mult", 0.70))

    tp_ceil_std      = float(config.get("tp_ceiling_standard", 2.65))
    tp_ceil_run      = float(config.get("tp_ceiling_runner", 3.60))

    cutoff_time      = pd.Timestamp(config.get("hard_cutoff_time", "13:15")).time()
    midday_split     = pd.Timestamp("11:30").time()
    conv_step1_time  = pd.Timestamp("11:15").time()
    conv_step2_time  = pd.Timestamp("12:15").time()
    lock_time        = pd.Timestamp(config.get("profit_lock_guard_time", "15:00")).time()
    eod_square_time  = pd.Timestamp("15:15").time()

    mc_engine = PreTradeMonteCarloEngine(config)

    # Build ticker price history lookup for causal trade tracing
    ticker_dfs = {}
    for ticker, g in df_pool.groupby("ticker"):
        ticker_dfs[ticker] = g.sort_index()

    oos_df_copy = oos_df.copy()
    oos_df_copy["bar_time"] = [t.time() if hasattr(t, 'time') else pd.Timestamp(str(t)).time() for t in oos_df_copy.index]
    oos_df_copy["trade_date"] = oos_df_copy.index.date

    # Initial coarse filter: within trade hours up to 13:15 IST and minimum friction ATR
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
            ticker   = row["ticker"]
            sec      = row["sector"]
            prob     = float(row["prob"])
            bar_time = row["bar_time"]
            rvol     = float(row.get("rvol", 1.0))
            cs_rs    = float(row.get("cs_rs", 0.0))
            adx_slp  = float(row.get("adx_slope_3", 0.0))
            atr_norm = float(row["atr_norm"])

            # 1. Hard Cutoff: t <= 13:15 IST
            if bar_time > cutoff_time:
                rejected_ledger.append({
                    "timestamp": str(idx),
                    "ticker": ticker,
                    "reason": "HARD_CUTOFF_AFTER_13:15",
                    "prob": round(prob, 4)
                })
                continue

            # 2. Time-Stepped Conviction Hurdle P_min(t)
            if bar_time <= conv_step1_time:
                p_min = 0.46
            elif bar_time <= conv_step2_time:
                p_min = 0.49
            else:
                p_min = 0.52

            if prob < p_min:
                rejected_ledger.append({
                    "timestamp": str(idx),
                    "ticker": ticker,
                    "reason": f"LOW_CONVICTION_TIME_STEPPED ({prob:.4f} < {p_min:.2f})",
                    "prob": round(prob, 4)
                })
                continue

            # 3. Session Velocity Gate
            if bar_time <= midday_split:
                # Morning Session: RVOL >= 1.10 and CS_RS >= 0.0
                if rvol < 1.10 or cs_rs < 0.0:
                    rejected_ledger.append({
                        "timestamp": str(idx),
                        "ticker": ticker,
                        "reason": f"VELOCITY_MORNING_FAIL (RVOL={rvol:.2f}, CS_RS={cs_rs:.2f})",
                        "prob": round(prob, 4)
                    })
                    continue
            else:
                # Midday Session: RVOL >= 1.35 and CS_RS >= +0.35 and adx_slope_3 > 0
                if rvol < 1.35 or cs_rs < 0.35 or adx_slp <= 0.0:
                    rejected_ledger.append({
                        "timestamp": str(idx),
                        "ticker": ticker,
                        "reason": f"VELOCITY_MIDDAY_FAIL (RVOL={rvol:.2f}, CS_RS={cs_rs:.2f}, ADX_SLOPE={adx_slp:.2f})",
                        "prob": round(prob, 4)
                    })
                    continue

            # 4. Daily Execution Cap (with recycling recovery)
            if daily_executed_count.get(d, 0) >= daily_n:
                rejected_ledger.append({
                    "timestamp": str(idx),
                    "ticker": ticker,
                    "reason": f"DAILY_EXECUTED_CAP_REACHED ({daily_n})",
                    "prob": round(prob, 4)
                })
                continue

            # 5. Daily Loss Circuit Breaker
            if daily_losses.get(d, 0) >= max_d_loss:
                rejected_ledger.append({
                    "timestamp": str(idx),
                    "ticker": ticker,
                    "reason": "CIRCUIT_BREAKER_MAX_DAILY_LOSS",
                    "prob": round(prob, 4)
                })
                continue

            day_sec_key = (d, sec)
            curr_sec_trades = daily_sector_trade_count.get(day_sec_key, 0)

            # 6. Sector Exposure Cap
            if curr_sec_trades >= max_per_sec:
                rejected_ledger.append({
                    "timestamp": str(idx),
                    "ticker": ticker,
                    "reason": f"MAX_SECTOR_CAP_REACHED ({max_per_sec})",
                    "prob": round(prob, 4)
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
                    "prob": round(prob, 4)
                })
                continue

            raw_price = float(row["close"])
            entry_p   = raw_price * (1.0 + slippage)
            sl_pct    = sl_atr_mult * atr_norm

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

            # Regime-Adaptive Target Ceiling
            is_runner = (rvol >= 1.60 and cs_rs >= 0.80)
            tp_mult = tp_ceil_run if is_runner else tp_ceil_std
            tp_pct = tp_mult * atr_norm

            # Friction-Inclusive Inverse-ATR Risk Cap
            r_net_cap = r_cap_high_atr if atr_norm > high_atr_thr else r_cap_core
            if curr_sec_trades >= 1:
                r_net_cap *= sec_penalty

            # Sizing: Qty_t = min( floor(R_cap / (1.5*ATR_t + 0.0014*C_t)), floor(M_avail / C_t) )
            per_share_risk = entry_p * (sl_pct + 0.0014)
            qty_risk = int(r_net_cap / per_share_risk) if per_share_risk > 0 else 1

            avail_margin = notional_cap / max(1, (len(open_concurrent) + 1))
            qty_margin = int(avail_margin / entry_p)
            qty = max(1, min(qty_risk, qty_margin))

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
                "risk_budget": round(r_net_cap, 2),
                "sector": sec,
                "sub_sector": row["sub_sector"]
            }

            # Pre-Trade Monte Carlo Gate
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

            # Execute Three-Gear Asymmetric Ratchet Causal Tracing
            p_sl_initial = entry_p * (1.0 - sl_pct)
            p_tp_ceiling = entry_p * (1.0 + tp_pct)
            active_p_sl  = p_sl_initial
            ratchet_gear = 0
            h_max_so_far = entry_p
            exit_price   = None
            exit_time    = None
            exit_reason  = None
            recycled_slot = False

            if ticker in ticker_dfs:
                t_df = ticker_dfs[ticker]
                post_bars = t_df[t_df.index > idx]
                post_bars_day = post_bars[post_bars.index.date == d]
                check_bars = post_bars_day.head(h_rem)

                bar_count = 0
                for _, bar_r in check_bars.iterrows():
                    bar_count += 1
                    b_high = float(bar_r["high"])
                    b_low  = float(bar_r["low"])
                    b_close = float(bar_r["close"])
                    b_time = bar_r.name.time() if hasattr(bar_r.name, 'time') else pd.Timestamp(str(bar_r.name)).time()
                    b_vwap = float(bar_r.get("vwap", b_close))
                    h_max_so_far = max(h_max_so_far, b_high)

                    # Gear 1 Trigger (+1.65x ATR)
                    if ratchet_gear < 1:
                        if b_high >= entry_p * (1.0 + gear1_trig * atr_norm):
                            ratchet_gear = 1
                            floor_p = entry_p * (1.0 + gear1_flr * atr_norm)
                            active_p_sl = max(active_p_sl, floor_p)

                    # Gear 2 Trigger (+2.30x ATR)
                    if b_high >= entry_p * (1.0 + gear2_trig * atr_norm):
                        ratchet_gear = 2
                        floor_p = entry_p * (1.0 + gear2_flr * atr_norm)
                        trail_p = h_max_so_far - (ratchet_trail * atr_norm * entry_p)
                        active_p_sl = max(active_p_sl, floor_p, trail_p)
                    elif ratchet_gear == 2:
                        floor_p = entry_p * (1.0 + gear2_flr * atr_norm)
                        trail_p = h_max_so_far - (ratchet_trail * atr_norm * entry_p)
                        active_p_sl = max(active_p_sl, floor_p, trail_p)

                    # 6-Bar Dead-Money Time-Stop + Slot Recycling
                    if bar_count == 6 and ratchet_gear == 0:
                        unrealized_ret = (b_close - entry_p) / entry_p
                        in_dead_zone = (-0.25 * atr_norm <= unrealized_ret <= 0.35 * atr_norm)
                        below_vwap = (b_close < b_vwap)
                        if in_dead_zone and below_vwap:
                            exit_price = b_close * (1.0 - slippage)
                            exit_time  = bar_r.name
                            exit_reason = "6_BAR_DEAD_MONEY_TIME_STOP"
                            recycled_slot = True
                            break

                    # 15:00 IST Profit-Lock Guard
                    if b_time >= lock_time:
                        if b_close >= entry_p * (1.0 + 0.25 * atr_norm):
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
                        if ratchet_gear == 2:
                            exit_reason = "GEAR_2_TRAIL_FLOOR_HIT"
                        elif ratchet_gear == 1:
                            exit_reason = "GEAR_1_RATCHET_FLOOR_HIT"
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

            # Slot management & recycling:
            # If recycled due to dead money, we do not permanently consume a daily execution slot
            if not recycled_slot:
                daily_executed_count[d] = daily_executed_count.get(d, 0) + 1
            else:
                # Still record count, but keep a slot open for fresh breakout
                daily_executed_count[d] = max(0, daily_executed_count.get(d, 0))

            daily_sector_trade_count[day_sec_key] = curr_sec_trades + 1

            if not is_win:
                daily_losses[d] = daily_losses.get(d, 0) + 1

            # Track in active positions for margin recycling
            active_positions_in_day.append({
                "entry_time": idx,
                "exit_time": exit_time,
                "risk_free": (ratchet_gear >= 1)
            })

            trade_ledger.append({
                "timestamp": str(idx),
                "ticker": ticker,
                "exit_time": str(exit_time),
                "exit_reason": exit_reason,
                "ratchet_gear": ratchet_gear,
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
                "risk_net_cap": round(r_net_cap, 2),
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

    trades_csv_path = os.path.join(out_dirs["trades"], "v31_trades.csv")
    df_trades.to_csv(trades_csv_path, index=False)

    rejected_csv_path = os.path.join(out_dirs["trades"], "v31_rejected_candidates.csv")
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
