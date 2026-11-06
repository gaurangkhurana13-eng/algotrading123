"""
Strategy V11: Institutional Multi-Asset Sniper & Climax Moonshot Engine
Basket: PCJEWELLER, RPOWER, IDEA, MTNL, HCC, UJJIVANSFB
Dual-Book Architecture:
  1. Intraday MIS Book (5x leverage, max notional Rs 2,450, 15:10 square-off, 0.05% slippage, Dhan charges)
  2. Swing Delivery Book (1x equity, multi-day holding, STT delivery charges)
Includes 10,000 Monte Carlo Bootstrap Iterations.
"""

import sys
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

STRATEGY_DIR = Path(__file__).parent
ROOT_DIR = STRATEGY_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR / "strategies"))
sys.path.insert(0, str(ROOT_DIR / "strategies" / "factory"))

from framework import calc_dhan_charges, calculate_atr, compute_metrics
from data_manager import load_clean_ticker_data, BASKET
from monte_carlo import run_monte_carlo_simulation

def compute_vwap_shifted(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    df_temp["_date"] = df_temp.index.date
    tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
    df_temp["_tpv"] = tp * df_temp["volume"]
    cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
    cum_vol = df_temp.groupby("_date")["volume"].cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).shift(1)

def prepare_signals(df: pd.DataFrame):
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    
    atr14 = calculate_atr(df, 14)
    ema50 = c.ewm(span=50).mean()
    ema110 = c.ewm(span=110).mean()
    vwap_lag = compute_vwap_shifted(df)
    
    # Fast RSI 7
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(7).mean()
    loss = (-delta.clip(upper=0)).rolling(7).mean()
    rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))
    
    # MACD 12/26/9
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()
    
    # ADX 10
    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    atr10 = tr.rolling(10).mean()
    up_move = h - h.shift(1)
    down_move = l.shift(1) - l
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(10).mean() / (atr10 + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(10).mean() / (atr10 + 1e-9))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    adx = dx.rolling(10).mean()
    
    macd_cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    
    # V11 Confluence:
    # 1. Structural Regime Gate: Close > 50 EMA and Close > 110 EMA
    # 2. Institutional Anchor: Close > Shifted VWAP
    # 3. Momentum Inflection: MACD Zero-Cross & MACD > 0
    # 4. Buyer Dominance: RSI 7 > 50.0, ADX 10 > 18.0, +DI > -DI
    entry_cond = (
        macd_cross_up &
        (macd > 0) &
        (c > ema50) &
        (c > ema110) &
        (c > vwap_lag) &
        (rsi7 > 50.0) &
        (adx > 18.0) &
        (plus_di > minus_di)
    )
    
    return {
        "df": df, "c": c, "h": h, "l": l, "atr14": atr14, "rsi7": rsi7,
        "vwap_lag": vwap_lag, "entry_cond": entry_cond
    }

def run_portfolio_backtest(data_map, basket, params, mode="intraday"):
    all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
    capital = 500.0
    max_portfolio_notional = 2450.0 if mode == "intraday" else 500.0
    slippage = 0.0005
    sl_mult = params.get("sl_mult", 1.00)
    p1_trig = params.get("p1_trig", 0.85)
    p1_lock = params.get("p1_lock", 0.40)
    p3_trig = params.get("p3_trig", 1.35)
    p3_trail = params.get("p3_trail", 0.45)
    trail_dist = params.get("trail_dist", 0.65)
    
    positions = {}
    trades = []
    
    for ts in all_ts:
        t_time = ts.time()
        live_tickers = list(positions.keys())
        for ticker in live_tickers:
            p = positions[ticker]
            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            row = t_data["df"].loc[ts]
            cur_c = row["close"]
            cur_h = row["high"]
            cur_l = row["low"]
            cur_rsi = t_data["rsi7"].loc[ts]
            cur_vwap = t_data["vwap_lag"].loc[ts]
            p["bars_held"] += 1
            
            # Intraday 15:10 square-off check
            if mode == "intraday" and t_time >= pd.Timestamp("15:10").time():
                raw_exit = cur_c * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                clean_sym = ticker.replace(".NS", "").replace("NSE:", "")
                trades.append({
                    "symbol": clean_sym, "direction": "long",
                    "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                    "entry_price": p["entry_p"], "exit_price": exit_p,
                    "shares": p["qty"], "qty": p["qty"], "gross_pnl": gross, "charges": chg,
                    "net_pnl": net, "exit_reason": "15:10_INTRADAY_SQUARE_OFF"
                })
                del positions[ticker]
                continue
                
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
            
            # Structural Stall Scratch: 3+ bars held, MFE < 0.40x ATR, Close < Entry and (RSI < 46 or Close < VWAP)
            if p["bars_held"] >= 3 and p["phase"] == 0 and mfe_atr < 0.40:
                if cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                    raw_exit = cur_c * (1.0 - slippage)
                    exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                    gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                    chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=(mode=="intraday")), 2)
                    net = round(gross - chg, 2)
                    clean_sym = ticker.replace(".NS", "").replace("NSE:", "")
                    trades.append({
                        "symbol": clean_sym, "direction": "long",
                        "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                        "entry_price": p["entry_p"], "exit_price": exit_p,
                        "shares": p["qty"], "qty": p["qty"], "gross_pnl": gross, "charges": chg,
                        "net_pnl": net, "exit_reason": "STRUCTURAL_STALL_SCRATCH"
                    })
                    del positions[ticker]
                    continue
                    
            # Tri-Phase Harvester & Climax Tightener
            if mfe_atr >= p3_trig:
                p["phase"] = 3
                p["active_sl"] = max(p["active_sl"], p["high_seen"] - (p3_trail * p["entry_atr"]))
            elif mfe_atr >= 1.70:
                p["phase"] = 2
                p["active_sl"] = max(p["active_sl"], p["high_seen"] - (trail_dist * p["entry_atr"]))
            elif mfe_atr >= p1_trig and p["phase"] < 1:
                p["phase"] = 1
                p["active_sl"] = max(p["active_sl"], p["entry_p"] + (p1_lock * p["entry_atr"]))
                
            if cur_l <= p["active_sl"]:
                raw_exit = p["active_sl"] * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=(mode=="intraday")), 2)
                net = round(gross - chg, 2)
                reason = "PHASE_3_CLIMAX_TRAIL_HIT" if p["phase"] == 3 else ("PHASE_2_MOONSHOT_TRAIL_HIT" if p["phase"] == 2 else ("PHASE_1_RATCHET_FLOOR_HIT" if p["phase"] == 1 else "INITIAL_SL_HIT"))
                clean_sym = ticker.replace(".NS", "").replace("NSE:", "")
                trades.append({
                    "symbol": clean_sym, "direction": "long",
                    "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                    "entry_price": p["entry_p"], "exit_price": exit_p,
                    "shares": p["qty"], "qty": p["qty"], "gross_pnl": gross, "charges": chg,
                    "net_pnl": net, "exit_reason": reason
                })
                del positions[ticker]
                
        # Entries
        current_live_notional = sum(p["entry_p"] * p["qty"] for p in positions.values())
        avail_notional = max_portfolio_notional - current_live_notional
        if avail_notional < 15.0:
            continue
        time_ok = (t_time >= pd.Timestamp("09:20").time()) and (t_time <= pd.Timestamp("13:30").time() if mode=="intraday" else True)
        if not time_ok:
            continue
            
        for ticker in basket:
            if ticker in positions:
                continue
            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            idx = t_data["df"].index.get_loc(ts)
            if idx < 100:
                continue
                
            if not t_data["entry_cond"].iloc[idx]:
                continue
                
            cur_c = t_data["df"]["close"].iloc[idx]
            cur_h = t_data["df"]["high"].iloc[idx]
            cur_l = t_data["df"]["low"].iloc[idx]
            entry_atr = max(t_data["atr14"].iloc[idx], 0.05)
            
            raw_entry = cur_c * (1.0 + slippage)
            entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
            stop_dist = sl_mult * entry_atr
            active_sl = entry_p - stop_dist
            high_seen = entry_p
            
            friction_share = (0.002067 if mode == "intraday" else 0.0045) * entry_p
            denom = stop_dist + friction_share
            eff_rcap = 14.00 if mode == "intraday" else 11.00
            qty_risk = int(eff_rcap / denom) if denom > 0 else 1
            qty_margin = int(avail_notional / entry_p)
            qty = min(qty_risk, qty_margin)
            
            if qty >= 1:
                notional_used = entry_p * qty
                if current_live_notional + notional_used <= max_portfolio_notional:
                    positions[ticker] = {
                        "direction": "long", "entry_p": entry_p, "entry_t": ts,
                        "qty": qty, "entry_atr": entry_atr, "active_sl": active_sl,
                        "high_seen": high_seen, "phase": 0, "bars_held": 0
                    }
                    current_live_notional += notional_used
                    avail_notional = max_portfolio_notional - current_live_notional
                    if avail_notional < 15.0:
                        break
                        
    return pd.DataFrame(trades)

def main():
    params = {
        "sl_mult": 1.00,
        "p1_trig": 0.85,
        "p1_lock": 0.40,
        "p3_trig": 1.35,
        "p3_trail": 0.45,
        "trail_dist": 0.65
    }
    
    print("Loading clean OHLCV data for all 6 basket stocks...")
    data_map = {}
    for t in BASKET:
        df = load_clean_ticker_data(t)
        data_map[t] = prepare_signals(df)
        
    print("Running Intraday MIS simulation...")
    df_intra = run_portfolio_backtest(data_map, BASKET, params, mode="intraday")
    print(f"Intraday trades generated: {len(df_intra)}")
    
    print("Running Swing Delivery simulation...")
    df_swing = run_portfolio_backtest(data_map, BASKET, params, mode="swing")
    print(f"Swing trades generated: {len(df_swing)}")
    
    # Save CSVs
    intra_csv = STRATEGY_DIR / "trades_intraday.csv"
    swing_csv = STRATEGY_DIR / "trades_swing.csv"
    df_intra.to_csv(intra_csv, index=False)
    df_swing.to_csv(swing_csv, index=False)
    print("CSVs saved.")
    
    # Compute metrics
    m_intra = compute_metrics(df_intra, capital=500.0)
    m_swing = compute_metrics(df_swing, capital=500.0)
    
    # Run Monte Carlo 10,000 Iterations
    print("Executing Institutional Monte Carlo Simulation (10,000 iterations)...")
    mc_intra, mc_intra_report = run_monte_carlo_simulation(df_intra, initial_capital=500.0, num_simulations=10000)
    mc_swing, mc_swing_report = run_monte_carlo_simulation(df_swing, initial_capital=500.0, num_simulations=10000)
    
    # Write Monte Carlo reports
    (STRATEGY_DIR / "monte_carlo_report.txt").write_text(
        "=== INTRADAY MONTE CARLO (10,000 RUNS) ===\n" + mc_intra_report + "\n\n" +
        "=== SWING MONTE CARLO (10,000 RUNS) ===\n" + mc_swing_report,
        encoding="utf-8"
    )
    with open(STRATEGY_DIR / "monte_carlo_report.json", "w", encoding="utf-8") as f:
        json.dump({"intraday": mc_intra, "swing": mc_swing}, f, indent=2)
    print("Monte Carlo reports saved.")
    
    payoff_intra = round(abs(m_intra['avg_win'] / m_intra['avg_loss']), 2) if m_intra['avg_loss'] != 0 else 0.0
    payoff_swing = round(abs(m_swing['avg_win'] / m_swing['avg_loss']), 2) if m_swing['avg_loss'] != 0 else 0.0
    
    # Format results.txt with exact reconciliation matching
    results_txt = f"""================================================================================
  BHAVISHYA STRATEGY V11: INSTITUTIONAL MULTI-ASSET SNIPER & CLIMAX MOONSHOT
================================================================================

1. LOGIC SUMMARY
   - Portfolio Basket: High-Beta Retail Leaders (PCJEWELLER, RPOWER, IDEA, MTNL, HCC, UJJIVANSFB)
   - Structural Regime Gate: Close > 50 EMA and Close > 110 EMA
   - Institutional Anchor: Close > Shifted VWAP.shift(1)
   - Momentum Inflection: MACD Zero-Cross (12, 26, 9) while MACD > 0
   - Buyer Dominance: RSI 7 > 50.0, ADX 10 > 18.0, Directional Movement +DI > -DI
   - Dynamic Tri-Phase Harvester & Climax Tightener:
     * Phase 1: At +0.85x ATR MFE -> Lock profit at +0.40x ATR floor
     * Phase 3 Climax: At +1.35x ATR MFE -> Tighten trailing stop to HighSeen - 0.45x ATR
     * Phase 2: At +1.70x ATR MFE -> Moonshot Trailing (HighSeen - 0.65x ATR)
     * Structural Stall Scratch: 3+ bars held, MFE < 0.40x ATR, Close < Entry & (RSI < 46 or Close < VWAP)
     * Initial Stop: 1.00x ATR
   - Realistic Execution: Penny-exact Dhan charges + 0.05% slippage, Rs 500 base, 5x leverage max Rs 2,450
   - Multi-Stock Portfolio Concurrency: Real-time margin pool tracking ensuring total open notional <= Rs 2,450
   - Institutional Monte Carlo: 10,000 bootstrap shuffles for ruin risk, VaR 95%, and execution slippage stress test

--------------------------------------------------------------------------------
INTRADAY
--------------------------------------------------------------------------------
Trades: {m_intra['total_trades']}
Wins: {m_intra['wins']}
Losses: {m_intra['losses']}
Win Rate: {m_intra['win_rate']}%
Net Profit: Rs {m_intra['net_pnl']}
Profit Factor: {m_intra['profit_factor']}
Avg Win: Rs {m_intra['avg_win']}
Avg Loss: {m_intra['avg_loss']}
Payoff Ratio: {payoff_intra}x
Avg Dhan Charges: Rs {m_intra['avg_charge']}
Max Drawdown: Rs {m_intra['max_drawdown']}
Final Capital: Rs {m_intra['final_capital']} (ROI: {m_intra['roi_pct']}%)

--------------------------------------------------------------------------------
SWING
--------------------------------------------------------------------------------
Trades: {m_swing['total_trades']}
Wins: {m_swing['wins']}
Losses: {m_swing['losses']}
Win Rate: {m_swing['win_rate']}%
Net Profit: Rs {m_swing['net_pnl']}
Profit Factor: {m_swing['profit_factor']}
Avg Win: Rs {m_swing['avg_win']}
Avg Loss: {m_swing['avg_loss']}
Payoff Ratio: {payoff_swing}x
Avg Dhan Charges: Rs {m_swing['avg_charge']}
Max Drawdown: Rs {m_swing['max_drawdown']}
Final Capital: Rs {m_swing['final_capital']} (ROI: {m_swing['roi_pct']}%)

================================================================================
"""
    (STRATEGY_DIR / "results.txt").write_text(results_txt, encoding="utf-8")
    print("results.txt written.")
    
    # Config JSON
    config = {
        "version": "V11",
        "name": "Institutional Multi-Asset Sniper & Climax Moonshot Engine",
        "basket": BASKET,
        "capital": 500.0,
        "max_notional": 2450.0,
        "params": params,
        "monte_carlo_runs": 10000,
        "intraday_metrics": m_intra,
        "swing_metrics": m_swing
    }
    with open(STRATEGY_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print("config.json written.")

if __name__ == "__main__":
    main()
