"""
Prototype test script for Bhavishya V27: Two-Tier Execution Engine
Tests Tier 1 Direct Fill + Tier 2 Pullback Rescue on Fold 5 out-of-sample data.
"""

import sys
from pathlib import Path
sys.path.insert(0, r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
import json
import pandas as pd
import numpy as np

with open(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\trading_system\V24\config\config.json") as f:
    config = json.load(f)

from trading_system.V24.src.strategy import build_dataset, run_walk_forward_pipeline
from trading_system.V24.src.pre_trade_mc import PreTradeMonteCarloEngine
from trading_system.V24.src.dhan_engine import calc_dhan_charges

cached_data = r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\data\universe_78_15m.pkl"
df_pool = build_dataset(cached_data, config)
oos_df = run_walk_forward_pipeline(df_pool, config)

print("Starting V27 Two-Tier Prototype Simulation...")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Evaluate on full walk-forward out-of-sample dataset (identical to V24)
eval_oos = oos_df.copy()
eval_oos["trade_date"] = eval_oos.index.date

mc_engine = PreTradeMonteCarloEngine(config)

SECTOR_MAP = {
    "MOREPENLAB": ("HEALTHCARE", "PHARMA"),
    "LLOYDSENGG": ("INDUSTRIALS", "CAPITAL_GOODS"),
    "MRPL": ("ENERGY", "OIL_GAS"),
    "ALOKINDS": ("CONSUMER", "TEXTILES"),
    "PCJEWELLER": ("CONSUMER", "GEMS_JEWELLERY")
}

def get_sector(ticker):
    return SECTOR_MAP.get(ticker, ("GENERAL", "GENERAL"))

eval_oos["sector"] = [get_sector(t)[0] for t in eval_oos["ticker"]]
eval_oos["sub_sector"] = [get_sector(t)[1] for t in eval_oos["ticker"]]

# Candidates with prob >= 0.45 (same as V24, but with Tier 2 rescue for 0.48 <= prob < 0.52)
cands = eval_oos[eval_oos["prob"] >= 0.45].copy()
print(f"Total candidates with prob >= 0.45 across full OOS: {len(cands)}")

# Build lookup by (ticker, timestamp) for df_pool bars to check pullbacks
ticker_dfs = {}
for ticker, g in df_pool.groupby("ticker"):
    ticker_dfs[ticker] = g.sort_index()

trade_ledger = []
rejected_ledger = []
daily_executed_count = {}
daily_losses = {}
daily_sector_trade_count = {}

capital = 500.0
current_capital = capital
leverage = 5.0
max_daily_loss = 2
daily_n = 3
max_per_sec = 2
sec_penalty = 0.80
slippage = 0.0005
base_budget = 20.0
min_budget = 18.0
max_budget = 35.0
payoff_b = 1.333

def compute_half_kelly(prob):
    p = prob
    q = 1.0 - p
    b = payoff_b
    edge = p * b - q
    f_star = edge / b if b > 0 else 0.0
    f_half = max(0.0, f_star / 2.0)
    budget = base_budget * (1.0 + f_half)
    return round(float(np.clip(budget, min_budget, max_budget)), 2)

# Uncapped daily queue sorted by prob descending
for trade_date, df_day in cands.groupby("trade_date"):
    df_sorted = df_day.sort_values(by="prob", ascending=False)
    d = trade_date

    for idx, row in df_sorted.iterrows():
        # Check daily executed cap
        if daily_executed_count.get(d, 0) >= daily_n:
            rejected_ledger.append({"timestamp": str(idx), "ticker": row["ticker"], "reason": "DAILY_EXECUTED_CAP_REACHED", "prob": row["prob"]})
            continue

        if daily_losses.get(d, 0) >= max_daily_loss:
            rejected_ledger.append({"timestamp": str(idx), "ticker": row["ticker"], "reason": "CIRCUIT_BREAKER_MAX_DAILY_LOSS", "prob": row["prob"]})
            continue

        ticker = row["ticker"]
        sec = row["sector"]
        day_sec_key = (d, sec)
        curr_sec_trades = daily_sector_trade_count.get(day_sec_key, 0)
        if curr_sec_trades >= max_per_sec:
            rejected_ledger.append({"timestamp": str(idx), "ticker": ticker, "reason": "MAX_TRADES_PER_SECTOR_REACHED", "prob": row["prob"]})
            continue

        raw_price = float(row["close"])
        entry_p   = raw_price * (1.0 + slippage)
        atr_norm  = float(row["atr_norm"])
        tp_pct    = float(row["target_pct"])
        sl_pct    = float(row["sl_pct"])
        prob      = float(row["prob"])

        risk_budget = compute_half_kelly(prob)
        if curr_sec_trades >= 1:
            risk_budget *= sec_penalty

        max_lev_cap = current_capital * leverage
        per_share_sl = entry_p * sl_pct
        qty = max(1, min(int(risk_budget / per_share_sl), int(max_lev_cap / entry_p)))

        # Tier 1 MC Evaluation
        snapshot_t1 = {
            "timestamp": str(idx), "ticker": ticker, "close": raw_price, "atr_norm": atr_norm,
            "vwap_dev": float(row["vwap_dev"]), "above_vwap": int(row["above_vwap"]),
            "adx": float(row["adx"]), "rvol": float(row["rvol"]), "prob": prob, "qty": qty,
            "tp_pct": tp_pct, "sl_pct": sl_pct, "hold_bars": 12,
            "capital_available": round(current_capital, 2), "position_size": round(qty * entry_p, 2),
            "risk_budget": round(risk_budget, 2), "sector": sec, "sub_sector": row["sub_sector"]
        }
        mc_eval_t1 = mc_engine.evaluate_candidate(snapshot_t1)
        mc_win_prob = mc_eval_t1["stats"].get("win_prob", 0.0)

        executed_trade = None

        if mc_eval_t1["decision"] == "ALLOW ENTRY":
            # Tier 1 Direct Execution
            realized_ret = float(row["realized_pct"])
            exit_p = entry_p * (1.0 + realized_ret) * (1.0 - slippage)
            gross_pnl = (exit_p - entry_p) * qty
            charges = calc_dhan_charges(entry_p, exit_p, qty)
            net_pnl = round(gross_pnl - charges, 2)
            executed_trade = {
                "timestamp": str(idx), "ticker": ticker, "tier": "TIER_1_DIRECT",
                "prob": prob, "mc_win_prob": mc_win_prob, "entry_price": entry_p,
                "exit_price": exit_p, "qty": qty, "gross_pnl": gross_pnl, "charges": charges,
                "net_pnl": net_pnl, "winner": net_pnl > 0
            }
        elif prob >= 0.48 and mc_win_prob >= 0.45:
            # Tier 2 Pullback Rescue Evaluation
            p_limit = raw_price * (1.0 - 0.35 * atr_norm)
            p_sl = raw_price * (1.0 - sl_pct)
            p_tp = raw_price * (1.0 + tp_pct)
            sl_pct_limit = (p_limit - p_sl) / p_limit
            tp_pct_limit = (p_tp - p_limit) / p_limit

            per_share_sl_lim = p_limit - p_sl
            qty_lim = max(1, min(int(risk_budget / per_share_sl_lim), int(max_lev_cap / p_limit)))

            snapshot_t2 = snapshot_t1.copy()
            snapshot_t2["close"] = p_limit
            snapshot_t2["tp_pct"] = tp_pct_limit
            snapshot_t2["sl_pct"] = sl_pct_limit
            snapshot_t2["qty"] = qty_lim
            mc_eval_t2 = mc_engine.evaluate_candidate(snapshot_t2)
            mc_win_prob_t2 = mc_eval_t2["stats"].get("win_prob", 0.0)

            # Tier 2 approval condition for asymmetric payoff:
            # Requires positive EV, controlled CVaR (>= -45), and MC win prob >= 0.45
            is_tier2_approved = (
                mc_eval_t2["stats"].get("expected_ev", -1.0) > 0.0 and
                mc_eval_t2["stats"].get("cvar_loss", -100.0) >= -45.0 and
                mc_win_prob_t2 >= 0.45
            )

            if is_tier2_approved:
                # Check bar t+1 and t+2 in df_pool
                if ticker in ticker_dfs:
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
                    if not filled and len(post_bars_day) >= 2:
                        b2 = post_bars_day.iloc[1]
                        if b2["low"] <= p_limit:
                            if b2["low"] > p_sl:
                                filled = True
                                fill_idx = post_bars_day.index[1]

                    if filled:
                        # Trace outcome from fill_idx onwards
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
                        executed_trade = {
                            "timestamp": str(idx), "ticker": ticker, "tier": "TIER_2_PULLBACK",
                            "prob": prob, "mc_win_prob": mc_win_prob_t2, "entry_price": entry_fill,
                            "exit_price": exit_price_lim, "qty": qty_lim, "gross_pnl": gross_pnl, "charges": charges,
                            "net_pnl": net_pnl, "winner": net_pnl > 0
                        }

        if executed_trade is not None:
            trade_ledger.append(executed_trade)
            current_capital = round(current_capital + executed_trade["net_pnl"], 2)
            daily_executed_count[d] = daily_executed_count.get(d, 0) + 1
            daily_sector_trade_count[day_sec_key] = curr_sec_trades + 1
            if not executed_trade["winner"]:
                daily_losses[d] = daily_losses.get(d, 0) + 1
        else:
            rejected_ledger.append({"timestamp": str(idx), "ticker": ticker, "reason": f"MC_REJECT_{mc_eval_t1['decision']}", "prob": prob})

trades_df = pd.DataFrame(trade_ledger)
print("\n" + "=" * 60)
print("🎯 V27 TWO-TIER EXECUTION RESULTS")
print("=" * 60)
print(f"Total Completed Trades: {len(trades_df)}")
if len(trades_df) > 0:
    wins = trades_df["winner"].sum()
    losses = len(trades_df) - wins
    win_rate = (wins / len(trades_df)) * 100
    avg_win = trades_df[trades_df["winner"]]["net_pnl"].mean() if wins > 0 else 0.0
    avg_loss = trades_df[~trades_df["winner"]]["net_pnl"].mean() if losses > 0 else 0.0
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
    tot_profit = trades_df[trades_df["net_pnl"] > 0]["net_pnl"].sum()
    tot_loss = abs(trades_df[trades_df["net_pnl"] < 0]["net_pnl"].sum())
    pf = tot_profit / tot_loss if tot_loss > 0 else 999.0
    net_pnl = trades_df["net_pnl"].sum()

    print(f"Wins / Losses:          {wins} Wins / {losses} Losses")
    print(f"Win Rate:               {win_rate:.2f}%")
    print(f"Average Win:            ₹{avg_win:.2f}")
    print(f"Average Loss:           ₹{avg_loss:.2f}")
    print(f"Payoff Ratio:           {payoff:.2f} : 1")
    print(f"Profit Factor:          {pf:.2f}")
    print(f"Net PnL:                ₹{net_pnl:.2f} (Final Capital: ₹{current_capital:.2f})")
    print("\nTrades Breakdown by Tier:")
    print(trades_df["tier"].value_counts())
    print("\nIndividual Trades:")
    for _, tr in trades_df.iterrows():
        print(f"[{tr['tier']}] {tr['timestamp']} {tr['ticker']} P={tr['prob']:.3f} Entry={tr['entry_price']:.2f} Exit={tr['exit_price']:.2f} PnL={tr['net_pnl']:.2f} Win={tr['winner']}")
