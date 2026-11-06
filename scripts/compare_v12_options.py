#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_v12_options.py
Rigorous side-by-side benchmark of:
  - Option 1: Argon's 78-Stock Cross-Sectional Architecture (4 Layers)
  - Option 2: Ultra-Elite Champion Basket (Top Alpha Movers)
Evaluated across full 10-month data (30-Dec-2025 to 29-Oct-2026, 5000 bars/stock).
"""

import sys
import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
sys.path.insert(0, str(ROOT_DIR / "strategies" / "factory"))

from framework import calc_dhan_charges, calculate_atr

# Load full 78-stock universe
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)

# 6 Sectors mapping
SECTOR_MAP = {
    # Power & Energy / Infrastructure
    **{s: "POWER_INFRA" for s in ["ADANIPOWER.NS", "BPCL.NS", "COALINDIA.NS", "GAIL.NS", "GMRAIRPORT.NS", "HCC.NS", "HUDCO.NS", "IOC.NS", "IRCON.NS", "IRFC.NS", "JPPOWER.NS", "NHPC.NS", "NTPC.NS", "ONGC.NS", "PTC.NS", "RPOWER.NS", "RVNL.NS", "SJVN.NS", "SWSOLAR.NS", "TATAPOWER.NS"]},
    # Banking & Financial Services
    **{s: "BANK_FIN" for s in ["BANDHANBNK.NS", "BANKBARODA.NS", "BANKINDIA.NS", "CANBK.NS", "CENTRALBK.NS", "EQUITASBNK.NS", "IDBI.NS", "IDFCFIRSTB.NS", "IFCI.NS", "INDIANB.NS", "IOB.NS", "KBANK.NS", "J&KBANK.NS", "MAHABANK.NS", "PFC.NS", "PNB.NS", "RBLBANK.NS", "RECLTD.NS", "SBIN.NS", "SOUTHBANK.NS", "UCOBANK.NS", "UJJIVANSFB.NS", "UNIONBANK.NS", "UTKARSHBNK.NS", "YESBANK.NS"]},
    # Metals, Mining & Heavy Engineering
    **{s: "METALS_ENGG" for s in ["ELECTCAST.NS", "LLOYDSENGG.NS", "NMDC.NS", "SAIL.NS", "SUZLON.NS", "TATASTEEL.NS", "TEXRAIL.NS", "VEDL.NS"]},
    # Telecom, Media & Technology
    **{s: "TELECOM_TECH" for s in ["HATHWAY.NS", "IDEA.NS", "MTNL.NS", "NETWORK18.NS", "SUBEXLTD.NS", "TTML.NS"]},
    # Retail, Gems & FMCG
    **{s: "RETAIL_FMCG" for s in ["ASHOKLEY.NS", "BCLIND.NS", "JYOTHYLAB.NS", "MANALIPETC.NS", "MMTC.NS", "MOREPENLAB.NS", "MOTHERSON.NS", "PATANJALI.NS", "PCJEWELLER.NS", "RENUKA.NS", "TRIDENT.NS"]},
    # Chemicals, Fertilizers & Real Estate / Infra
    **{s: "CHEM_FERT_REALTY" for s in ["ALOKINDS.NS", "GSFC.NS", "JAICORPLTD.NS", "MRPL.NS", "NBCC.NS", "NFL.NS", "NIACL.NS", "RCF.NS", "SPIC.NS"]}
}

def compute_vwap_shifted(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    df_temp["_date"] = df_temp.index.date
    tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
    df_temp["_tpv"] = tp * df_temp["volume"]
    cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
    cum_vol = df_temp.groupby("_date")["volume"].cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).shift(1)

def prepare_all_data(symbols):
    data_map = {}
    for ticker in symbols:
        if ticker not in univ:
            continue
        df = univ[ticker].copy()
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df["volume"]
        
        atr14 = calculate_atr(df, 14)
        natr14 = (atr14 / c) * 100.0
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
        macd_cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
        
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
        
        # RVOL 20
        vol_sma20 = v.rolling(20).mean()
        rvol20 = v / (vol_sma20 + 1e-9)
        
        # Surge Score
        surge_score = rvol20 * adx * natr14
        
        # V11 Baseline Entry Condition
        v11_entry = (
            macd_cross_up &
            (macd > 0) &
            (c > ema50) &
            (c > ema110) &
            (c > vwap_lag) &
            (rsi7 > 50.0) &
            (adx > 18.0) &
            (plus_di > minus_di)
        )
        
        data_map[ticker] = {
            "df": df, "c": c, "h": h, "l": l, "v": v, "atr14": atr14,
            "natr14": natr14, "rsi7": rsi7, "vwap_lag": vwap_lag,
            "v11_entry": v11_entry, "surge_score": surge_score,
            "adx10": adx, "rvol20": rvol20,
            "sector": SECTOR_MAP.get(ticker, "UNKNOWN")
        }
    return data_map

def run_simulation(data_map, symbols, config):
    all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
    max_portfolio_notional = 2450.0
    slot_cap = config.get("slot_cap", 816.66) # max 3 concurrent
    max_concurrent = config.get("max_concurrent", 3)
    max_per_sector = config.get("max_per_sector", 1)
    use_natr_filter = config.get("use_natr_filter", False)
    min_natr = config.get("min_natr", 0.65)
    max_price = config.get("max_price", 450.0)
    use_surge_rank = config.get("use_surge_rank", False)
    slippage = 0.0005
    
    # Exit params (V11 untouched)
    sl_mult = 1.00
    p1_trig = 0.85
    p1_lock = 0.40
    p3_trig = 1.35
    p3_trail = 0.45
    trail_dist = 0.65
    
    positions = {}
    trades = []
    daily_traded_stocks = {} # date -> set of tickers already traded today
    
    for ts in all_ts:
        cur_date = ts.date()
        if cur_date not in daily_traded_stocks:
            daily_traded_stocks[cur_date] = set()
            
        t_time = ts.time()
        
        # 1. Update live positions & check exits
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
            if t_time >= pd.Timestamp("15:10").time():
                raw_exit = cur_c * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                trades.append({
                    "symbol": ticker.replace(".NS", ""), "direction": "long",
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
                    chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                    net = round(gross - chg, 2)
                    trades.append({
                        "symbol": ticker.replace(".NS", ""), "direction": "long",
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
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                reason = "PHASE_3_CLIMAX_TRAIL_HIT" if p["phase"] == 3 else ("PHASE_2_MOONSHOT_TRAIL_HIT" if p["phase"] == 2 else ("PHASE_1_RATCHET_FLOOR_HIT" if p["phase"] == 1 else "INITIAL_SL_HIT"))
                trades.append({
                    "symbol": ticker.replace(".NS", ""), "direction": "long",
                    "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                    "entry_price": p["entry_p"], "exit_price": exit_p,
                    "shares": p["qty"], "qty": p["qty"], "gross_pnl": gross, "charges": chg,
                    "net_pnl": net, "exit_reason": reason
                })
                del positions[ticker]
                
        # 2. Candidate Evaluation & Entry Allocation
        if len(positions) >= max_concurrent:
            continue
            
        current_live_notional = sum(p["entry_p"] * p["qty"] for p in positions.values())
        avail_notional = max_portfolio_notional - current_live_notional
        if avail_notional < 15.0:
            continue
            
        time_ok = (t_time >= pd.Timestamp("09:20").time()) and (t_time <= pd.Timestamp("13:30").time())
        if not time_ok:
            continue
            
        # Collect candidates
        candidates = []
        for ticker in symbols:
            if ticker in positions:
                continue
            if ticker in daily_traded_stocks[cur_date]:
                continue
                
            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            idx = t_data["df"].index.get_loc(ts)
            if idx < 100:
                continue
                
            if not t_data["v11_entry"].iloc[idx]:
                continue
                
            cur_c = t_data["c"].iloc[idx]
            cur_natr = t_data["natr14"].iloc[idx]
            
            if use_natr_filter and (cur_natr < min_natr):
                continue
            if cur_c > max_price:
                continue
                
            score = t_data["surge_score"].iloc[idx] if use_surge_rank else 1.0
            candidates.append({
                "ticker": ticker,
                "score": score,
                "idx": idx,
                "sector": t_data["sector"]
            })
            
        if not candidates:
            continue
            
        # Sort candidates
        if use_surge_rank:
            candidates.sort(key=lambda x: x["score"], reverse=True)
            
        # Allocate slots obeying sector limits
        active_sectors = [data_map[t]["sector"] for t in positions.keys()]
        
        for cand in candidates:
            if len(positions) >= max_concurrent:
                break
            cand_sec = cand["sector"]
            if active_sectors.count(cand_sec) >= max_per_sector:
                continue
                
            ticker = cand["ticker"]
            t_data = data_map[ticker]
            idx = cand["idx"]
            cur_c = t_data["c"].iloc[idx]
            cur_h = t_data["h"].iloc[idx]
            cur_l = t_data["l"].iloc[idx]
            entry_atr = max(t_data["atr14"].iloc[idx], 0.05)
            
            raw_entry = cur_c * (1.0 + slippage)
            entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
            stop_dist = sl_mult * entry_atr
            active_sl = entry_p - stop_dist
            high_seen = entry_p
            
            friction_share = 0.002067 * entry_p
            denom = stop_dist + friction_share
            eff_rcap = 14.00
            qty_risk = int(eff_rcap / denom) if denom > 0 else 1
            slot_available = min(slot_cap, avail_notional)
            qty_margin = int(slot_available / entry_p)
            qty = min(qty_risk, qty_margin)
            
            if qty >= 1:
                notional_used = entry_p * qty
                if current_live_notional + notional_used <= max_portfolio_notional:
                    positions[ticker] = {
                        "direction": "long", "entry_p": entry_p, "entry_t": ts,
                        "qty": qty, "entry_atr": entry_atr, "active_sl": active_sl,
                        "high_seen": high_seen, "phase": 0, "bars_held": 0
                    }
                    daily_traded_stocks[cur_date].add(ticker)
                    active_sectors.append(cand_sec)
                    current_live_notional += notional_used
                    avail_notional = max_portfolio_notional - current_live_notional
                    if avail_notional < 15.0:
                        break
                        
    df_trades = pd.DataFrame(trades)
    return df_trades

def analyze_trades(df_trades, name):
    if df_trades.empty:
        print(f"\n[{name}] NO TRADES GENERATED!")
        return {}
        
    wins = df_trades[df_trades["net_pnl"] > 0]
    losses = df_trades[df_trades["net_pnl"] < 0]
    total_trades = len(df_trades)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades) * 100.0 if total_trades > 0 else 0.0
    gross_profit = wins["gross_pnl"].sum()
    gross_loss = abs(losses["gross_pnl"].sum())
    total_charges = df_trades["charges"].sum()
    net_pnl = df_trades["net_pnl"].sum()
    pf = (gross_profit / gross_loss) if gross_loss > 0 else 999.0
    worst_loss = df_trades["net_pnl"].min()
    best_win = df_trades["net_pnl"].max()
    avg_win = wins["net_pnl"].mean() if win_count > 0 else 0.0
    avg_loss = losses["net_pnl"].mean() if loss_count > 0 else 0.0
    
    # Cumulative PnL & Max Drawdown
    cum_pnl = df_trades["net_pnl"].cumsum()
    peak = cum_pnl.cummax()
    dd = peak - cum_pnl
    max_dd = dd.max()
    
    print(f"\n=================================================================")
    print(f"BENCHMARK SCORECARD: {name}")
    print(f"=================================================================")
    print(f"Total Trades         : {total_trades}")
    print(f"Wins / Losses        : {win_count} Wins / {loss_count} Losses")
    print(f"Win Rate             : {win_rate:.2f}%")
    print(f"Gross Profit         : +Rs {gross_profit:.2f}")
    print(f"Gross Loss           : -Rs {gross_loss:.2f}")
    print(f"Total Dhan Friction  : Rs {total_charges:.2f}")
    print(f"Net Realized PnL     : +Rs {net_pnl:.2f} ({net_pnl/500.0*100:+.2f}% ROI on Rs 500)")
    print(f"Profit Factor        : {pf:.2f}")
    print(f"Average Win / Loss   : +Rs {avg_win:.2f} / Rs {avg_loss:.2f}")
    print(f"Worst Single Loss    : Rs {worst_loss:.2f} (Cap <= Rs 15.50: {'PASS' if worst_loss >= -15.50 else 'FAIL'})")
    print(f"Best Single Win      : +Rs {best_win:.2f}")
    print(f"Max Intra-Book DD    : Rs {max_dd:.2f}")
    
    # Monthly breakdown
    df_trades["month"] = pd.to_datetime(df_trades["entry_time"]).dt.to_period("M")
    monthly = df_trades.groupby("month")["net_pnl"].agg(["count", "sum"]).reset_index()
    print("\n--- Monthly PnL Breakdown ---")
    for _, row in monthly.iterrows():
        print(f"  {row['month']}: {row['count']:2d} trades | Net PnL: Rs {row['sum']:+7.2f}")
        
    # Top 5 contributing tickers
    by_sym = df_trades.groupby("symbol")["net_pnl"].agg(["count", "sum"]).reset_index()
    by_sym = by_sym.sort_values(by="sum", ascending=False)
    print("\n--- Top Ticker Contributions ---")
    for _, row in by_sym.head(8).iterrows():
        print(f"  {row['symbol']:12s}: {row['count']:2d} trades | Net PnL: Rs {row['sum']:+7.2f}")
    print("--- Worst Ticker Contributions ---")
    for _, row in by_sym.tail(5).iterrows():
        print(f"  {row['symbol']:12s}: {row['count']:2d} trades | Net PnL: Rs {row['sum']:+7.2f}")
    print(f"=================================================================\n")
    
    return {
        "trades": total_trades, "wins": win_count, "losses": loss_count,
        "win_rate": win_rate, "net_pnl": net_pnl, "pf": pf,
        "worst_loss": worst_loss, "max_dd": max_dd
    }

def main():
    all_78_symbols = list(univ.keys())
    ultra_elite_symbols = [
        "PCJEWELLER.NS", "RPOWER.NS", "MTNL.NS", "IDEA.NS", 
        "UTKARSHBNK.NS", "TEXRAIL.NS"
    ]
    
    print("Loading data and calculating technical indicators for all 78 symbols...")
    data_map = prepare_all_data(all_78_symbols)
    print(f"Successfully calculated indicators for {len(data_map)} tickers.")
    
    # OPTION 1: Argon 78-Stock 4-Layer Architecture
    opt1_config = {
        "slot_cap": 816.66,
        "max_concurrent": 3,
        "max_per_sector": 1,
        "use_natr_filter": True,
        "min_natr": 0.65,
        "max_price": 450.0,
        "use_surge_rank": True
    }
    print("\nExecuting OPTION 1: Argon 78-Stock 4-Layer Architecture...")
    trades_opt1 = run_simulation(data_map, all_78_symbols, opt1_config)
    res_opt1 = analyze_trades(trades_opt1, "OPTION 1: Argon 78-Stock 4-Layer Cross-Sectional Sniper")
    
    # OPTION 2: Ultra-Elite Champion Basket
    opt2_config = {
        "slot_cap": 1225.0,
        "max_concurrent": 3,
        "max_per_sector": 2,
        "use_natr_filter": False,
        "max_price": 450.0,
        "use_surge_rank": True
    }
    print("\nExecuting OPTION 2: Ultra-Elite Champion Basket (Top Alpha Movers)...")
    trades_opt2 = run_simulation(data_map, ultra_elite_symbols, opt2_config)
    res_opt2 = analyze_trades(trades_opt2, "OPTION 2: Ultra-Elite Champion Basket")

if __name__ == "__main__":
    main()
