"""
Test script for Strategy V10: 4-Layer Institutional Engine
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, 'C:/Users/admin/.gemini/antigravity/scratch/new_trade/strategies')
sys.path.insert(0, 'C:/Users/admin/.gemini/antigravity/scratch/new_trade/strategies/factory')
from data_manager import load_clean_ticker_data, BASKET
from framework import calculate_atr, compute_metrics, calc_dhan_charges

def get_shifted_vwap(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    d["_d"] = d.index.date
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    d["_tpv"] = tp * d["volume"]
    return (d.groupby("_d")["_tpv"].cumsum() / (d.groupby("_d")["volume"].cumsum() + 1e-9)).shift(1)

def compute_v10_signals(df_15m: pd.DataFrame):
    c = df_15m['close']
    h = df_15m['high']
    l = df_15m['low']
    o = df_15m['open']
    v = df_15m['volume']
    
    # 1. 15m Indicators
    atr14 = calculate_atr(df_15m, 14)
    atr10 = calculate_atr(df_15m, 10)
    sma_atr25 = atr10.rolling(25).mean()
    sma_atr30 = atr10.rolling(30).mean()
    nu_t = atr10 / (sma_atr30 + 1e-9)
    
    ema20 = c.ewm(span=20).mean()
    ema50 = c.ewm(span=50).mean()
    vwap_lag = get_shifted_vwap(df_15m)
    
    # Fast RSI 7
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(7).mean()
    loss = (-delta.clip(upper=0)).rolling(7).mean()
    rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))
    
    # MACD 12/26/9
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()
    macd_hist = macd - sig
    
    # ADX 10
    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    atr10_dmi = tr.rolling(10).mean()
    up_move = h - h.shift(1)
    down_move = l.shift(1) - l
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * (pd.Series(plus_dm, index=df_15m.index).rolling(10).mean() / (atr10_dmi + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm, index=df_15m.index).rolling(10).mean() / (atr10_dmi + 1e-9))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    adx10 = dx.rolling(10).mean()
    
    # 2. Layer 1: Multi-Timeframe 1H Regime Gate (Zero-Lookahead backward merge)
    df_1h = df_15m.resample('1h', closed='left', label='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    ema50_1h = df_1h['close'].ewm(span=50).mean()
    ema20_1h = df_1h['close'].ewm(span=20).mean()
    slope_20_1h = ema20_1h > ema20_1h.shift(1)
    atr14_1h = calculate_atr(df_1h, 14)
    sma_atr20_1h = atr14_1h.rolling(20).mean()
    exhaustion_1h = (atr14_1h / (sma_atr20_1h + 1e-9)) <= 1.35
    bullish_1h_series = (df_1h['close'] > ema50_1h) & slope_20_1h & exhaustion_1h
    
    df_1h_features = pd.DataFrame({
        'ts': df_1h.index,
        'bullish_1h': bullish_1h_series.shift(1).fillna(False)
    })
    
    df_temp = pd.DataFrame({'ts': df_15m.index})
    merged_1h = pd.merge_asof(df_temp, df_1h_features, on='ts', direction='backward')
    bullish_1h = pd.Series(merged_1h['bullish_1h'].to_numpy(), index=df_15m.index).fillna(False)
    
    # Universal Pre-Entry Filters
    vwap_ext = (c - vwap_lag) / (atr14 + 1e-9)
    universal_ok = (c > vwap_lag) & (c > ema50) & (vwap_ext <= 1.15) & bullish_1h
    
    # Trigger Mode A: Volatility-Compressed Momentum Inflection
    coiling_ok = atr10 <= 1.05 * sma_atr25
    macd_hist_rising = (macd_hist > 0) & (macd_hist > macd_hist.shift(1))
    mode_a = coiling_ok & macd_hist_rising & (macd > sig) & (macd.shift(1) <= sig.shift(1)) & (rsi7 > 52.0) & (adx10 > 18.0) & (plus_di > minus_di)
    
    # Trigger Mode B: Institutional VWAP / EMA-20 Continuation Pullback
    strong_trend = (adx10 > 22.0) & (ema20 > ema50)
    low_touch = (l <= ema20 + 0.25 * atr14) | (l <= vwap_lag + 0.25 * atr14)
    low_touch_2b = low_touch.shift(1) | low_touch.shift(2)
    resumption = (c > o) & (c > h.shift(1)) & (rsi7 > 54.0)
    mode_b = strong_trend & low_touch_2b & resumption & (c > ema50)
    
    entry_signal = universal_ok & (mode_a | mode_b)
    
    return {
        "df": df_15m, "c": c, "h": h, "l": l, "o": o, "atr14": atr14, "atr10": atr10,
        "nu_t": nu_t, "rsi7": rsi7, "entry_signal": entry_signal
    }

def run_v10_backtest(data_map, basket, mode="intraday"):
    all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
    capital = 500.0
    max_portfolio_notional = 2450.0 if mode == "intraday" else 500.0
    single_stock_max_notional = 815.0 if mode == "intraday" else 500.0
    slippage = 0.0005
    sl_mult = 0.85
    
    positions = {}
    trades = []
    
    for ts in all_ts:
        t_time = ts.time()
        live_tickers = list(positions.keys())
        
        # 1. Process EXITS
        for ticker in live_tickers:
            p = positions[ticker]
            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            row = t_data["df"].loc[ts]
            cur_c = row["close"]
            cur_h = row["high"]
            cur_l = row["low"]
            cur_atr = max(t_data["atr14"].loc[ts], 0.05)
            rsi_cur = t_data["rsi7"].loc[ts]
            
            p["bars_held"] += 1
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
            
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
                
            # Phase 0: 4-Bar Stall-Kill Scratch (Reduces average loss)
            if p["bars_held"] >= 4 and mfe_atr < 0.25 and rsi_cur < 48.0:
                p["active_sl"] = max(p["active_sl"], cur_l - 0.10 * p["entry_atr"])
                
            # Phase 1: Scalp Lock & Cost-Free Shield at +0.55x ATR
            if mfe_atr >= 0.55 and p["phase"] < 1:
                p["phase"] = 1
                p["active_sl"] = max(p["active_sl"], p["entry_p"] + (0.08 * p["entry_atr"]))
                
            # Phase 2: Meat of Trend Lock at +1.15x ATR
            if mfe_atr >= 1.15 and p["phase"] < 2:
                p["phase"] = 2
                p["active_sl"] = max(p["active_sl"], p["entry_p"] + (0.65 * p["entry_atr"]))
                
            # Phase 3: Moonshot Structural Runner at +1.60x ATR
            if mfe_atr >= 1.60:
                p["phase"] = 3
                chandelier = p["high_seen"] - (0.60 * cur_atr)
                p["active_sl"] = max(p["active_sl"], chandelier)
                
            # Check Stop hit
            if cur_l <= p["active_sl"]:
                raw_exit = p["active_sl"] * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=(mode=="intraday")), 2)
                net = round(gross - chg, 2)
                
                if p["phase"] == 3:
                    reason = "PHASE_3_MOONSHOT_RUNNER_HIT"
                elif p["phase"] == 2:
                    reason = "PHASE_2_MEAT_OF_TREND_LOCKED"
                elif p["phase"] == 1:
                    reason = "PHASE_1_SCALP_SHIELD_HIT"
                elif p["bars_held"] >= 4 and mfe_atr < 0.25:
                    reason = "PHASE_0_STALL_KILL_SCRATCH"
                else:
                    reason = "INITIAL_SL_HIT"
                    
                clean_sym = ticker.replace(".NS", "").replace("NSE:", "")
                trades.append({
                    "symbol": clean_sym, "direction": "long",
                    "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                    "entry_price": p["entry_p"], "exit_price": exit_p,
                    "shares": p["qty"], "qty": p["qty"], "gross_pnl": gross, "charges": chg,
                    "net_pnl": net, "exit_reason": reason
                })
                del positions[ticker]
                
        # 2. Process ENTRIES
        current_live_notional = sum(p["entry_p"] * p["qty"] for p in positions.values())
        avail_portfolio_notional = max_portfolio_notional - current_live_notional
        if avail_portfolio_notional < 15.0:
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
            if idx < 100 or not t_data["entry_signal"].iloc[idx]:
                continue
                
            cur_c = t_data["df"]["close"].iloc[idx]
            cur_h = t_data["df"]["high"].iloc[idx]
            cur_l = t_data["df"]["low"].iloc[idx]
            entry_atr = max(t_data["atr14"].iloc[idx], 0.05)
            nu = t_data["nu_t"].iloc[idx]
            
            raw_entry = cur_c * (1.0 + slippage)
            entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
            stop_dist = sl_mult * entry_atr
            active_sl = entry_p - stop_dist
            
            # Layer 3: 3-Tier Volatility-Normalized Risk Budget
            if nu <= 0.85:
                r_budget = 13.80
            elif nu <= 1.10:
                r_budget = 11.50
            else:
                r_budget = 8.50
                
            friction_share = (0.002067 if mode == "intraday" else 0.0045) * entry_p
            denom = stop_dist + friction_share
            qty_risk = int(r_budget / denom) if denom > 0 else 1
            
            alloc_notional = min(avail_portfolio_notional, single_stock_max_notional)
            qty_margin = int(alloc_notional / entry_p)
            qty = min(qty_risk, qty_margin)
            
            if qty >= 1:
                notional_used = entry_p * qty
                if current_live_notional + notional_used <= max_portfolio_notional:
                    positions[ticker] = {
                        "entry_p": entry_p, "entry_t": ts, "qty": qty,
                        "entry_atr": entry_atr, "active_sl": active_sl,
                        "high_seen": entry_p, "bars_held": 0, "phase": 0
                    }
                    current_live_notional += notional_used
                    avail_portfolio_notional = max_portfolio_notional - current_live_notional
                    if avail_portfolio_notional < 15.0:
                        break
                        
    return pd.DataFrame(trades)

def main():
    print("Loading data and computing V10 4-Layer Indicators across basket...")
    data_map = {}
    for t in BASKET:
        df = load_clean_ticker_data(t)
        data_map[t] = compute_v10_signals(df)
        
    df_intra = run_v10_backtest(data_map, BASKET, mode="intraday")
    df_swing = run_v10_backtest(data_map, BASKET, mode="swing")
    
    m_intra = compute_metrics(df_intra, 500.0)
    m_swing = compute_metrics(df_swing, 500.0)
    
    print("\n=== STRATEGY V10 INTRADAY RESULTS ===")
    print("Trades:", m_intra["total_trades"], "Wins:", m_intra["wins"], "Losses:", m_intra["losses"])
    print("Win Rate:", m_intra["win_rate"], "% | PF:", m_intra["profit_factor"], "| Net PnL: Rs", m_intra["net_pnl"])
    print("Avg Win: Rs", m_intra["avg_win"], "| Avg Loss: Rs", m_intra["avg_loss"])
    print("Max Drawdown: Rs", m_intra["max_drawdown"], "| Worst Loss: Rs", df_intra["net_pnl"].min())
    
    print("\n=== STRATEGY V10 SWING RESULTS ===")
    print("Trades:", m_swing["total_trades"], "Wins:", m_swing["wins"], "Losses:", m_swing["losses"])
    print("Win Rate:", m_swing["win_rate"], "% | PF:", m_swing["profit_factor"], "| Net PnL: Rs", m_swing["net_pnl"])
    print("Avg Win: Rs", m_swing["avg_win"], "| Avg Loss: Rs", m_swing["avg_loss"])
    print("Max Drawdown: Rs", m_swing["max_drawdown"], "| Worst Loss: Rs", df_swing["net_pnl"].min())

if __name__ == "__main__":
    main()
