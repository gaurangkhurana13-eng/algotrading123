import pickle
import pandas as pd
import numpy as np
import sys
import math
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)

tickers = [
    "PCJEWELLER.NS", "UTKARSHBNK.NS", "RPOWER.NS", "MTNL.NS", 
    "IDEA.NS", "IFCI.NS", "SWSOLAR.NS", "TEXRAIL.NS"
]

def compute_vwap_shifted(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    df_temp["_date"] = df_temp.index.date
    tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
    df_temp["_tpv"] = tp * df_temp["volume"]
    cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
    cum_vol = df_temp.groupby("_date")["volume"].cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).shift(1)

data_map = {}
for t in tickers:
    df = univ[t].copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    atr14 = calculate_atr(df, 14)
    ema50 = c.ewm(span=50).mean()
    ema110 = c.ewm(span=110).mean()
    vwap_lag = compute_vwap_shifted(df)
    
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(7).mean()
    loss = (-delta.clip(upper=0)).rolling(7).mean()
    rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))
    
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()
    macd_cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    
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
    
    entry_cond = (
        macd_cross_up & (macd > 0) &
        (c > ema50) & (c > ema110) &
        (c > vwap_lag) &
        (rsi7 > 50.0) & (adx > 18.0) & (plus_di > minus_di) &
        (c <= 450.0)
    )
    data_map[t] = {
        "df": df, "c": c, "h": h, "l": l, "atr14": atr14,
        "rsi7": rsi7, "vwap_lag": vwap_lag, "entry_cond": entry_cond
    }

all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
slippage = 0.0005
p1_trig = 0.85
p1_lock = 0.40
p3_trig = 1.35
p3_trail = 0.45
trail_dist = 0.65

for quarantine_enabled in [False, True]:
    positions = {}
    parent_trades = []
    ticker_history = {t: [] for t in tickers}
    daily_traded = {}
    
    wallet_equity = 500.0
    wallet_hwm = 500.0
    
    for ts in all_ts:
        cur_date = ts.date()
        if cur_date not in daily_traded:
            daily_traded[cur_date] = set()
        t_time = ts.time()
        
        for ticker in list(positions.keys()):
            p = positions[ticker]
            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            row = t_data["df"].loc[ts]
            cur_c, cur_h, cur_l = row["close"], row["high"], row["low"]
            cur_rsi = t_data["rsi7"].loc[ts]
            cur_vwap = t_data["vwap_lag"].loc[ts]
            p["bars_held"] += 1
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
            
            # Harvester
            if mfe_atr >= p3_trig:
                p["phase"] = 3
                p["active_sl"] = max(p["active_sl"], p["high_seen"] - (p3_trail * p["entry_atr"]))
            elif mfe_atr >= 1.70:
                p["phase"] = 2
                p["active_sl"] = max(p["active_sl"], p["high_seen"] - (trail_dist * p["entry_atr"]))
            elif mfe_atr >= p1_trig and p["phase"] < 1:
                p["phase"] = 1
                p["active_sl"] = max(p["active_sl"], p["entry_p"] + (p1_lock * p["entry_atr"]))
                
            # SL
            if cur_l <= p["active_sl"]:
                raw_exit = p["active_sl"] * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                tot_qty = p["total_qty"]
                trade_gross = round((exit_p - p["entry_p"]) * tot_qty, 2)
                trade_chg = round(calc_dhan_charges(p["entry_p"], exit_p, tot_qty, is_intraday=True), 2)
                trade_net = round(trade_gross - trade_chg, 2)
                wallet_equity += trade_net
                wallet_hwm = max(wallet_hwm, wallet_equity)
                parent_trades.append({"symbol": ticker, "pnl": trade_net, "wallet": wallet_equity})
                ticker_history[ticker].append(trade_net)
                del positions[ticker]
                continue
                
            # Scratch
            if p["bars_held"] >= 3 and p["phase"] == 0 and mfe_atr < 0.40:
                if cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                    raw_exit = cur_c * (1.0 - slippage)
                    exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                    tot_qty = p["total_qty"]
                    trade_gross = round((exit_p - p["entry_p"]) * tot_qty, 2)
                    trade_chg = round(calc_dhan_charges(p["entry_p"], exit_p, tot_qty, is_intraday=True), 2)
                    trade_net = round(trade_gross - trade_chg, 2)
                    wallet_equity += trade_net
                    wallet_hwm = max(wallet_hwm, wallet_equity)
                    parent_trades.append({"symbol": ticker, "pnl": trade_net, "wallet": wallet_equity})
                    ticker_history[ticker].append(trade_net)
                    del positions[ticker]
                    continue
                    
            # 15:00 Square-off
            if t_time >= pd.Timestamp("15:00").time():
                raw_exit = cur_c * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                tot_qty = p["total_qty"]
                trade_gross = round((exit_p - p["entry_p"]) * tot_qty, 2)
                trade_chg = round(calc_dhan_charges(p["entry_p"], exit_p, tot_qty, is_intraday=True), 2)
                trade_net = round(trade_gross - trade_chg, 2)
                wallet_equity += trade_net
                wallet_hwm = max(wallet_hwm, wallet_equity)
                parent_trades.append({"symbol": ticker, "pnl": trade_net, "wallet": wallet_equity})
                ticker_history[ticker].append(trade_net)
                del positions[ticker]
                continue
                
        if len(positions) >= 2:
            continue
            
        current_notional = sum(p["entry_p"] * p["total_qty"] for p in positions.values())
        avail = (5.0 * wallet_equity) - current_notional
        if avail < 15.0 or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("13:30").time():
            continue
            
        for ticker in tickers:
            if ticker in positions or ticker in daily_traded[cur_date]:
                continue
                
            # Quarantine check: if last 4 trades have PF < 0.8
            if quarantine_enabled and len(ticker_history[ticker]) >= 4:
                recent = ticker_history[ticker][-4:]
                w_rec = sum(x for x in recent if x > 0)
                l_rec = abs(sum(x for x in recent if x < 0))
                rec_pf = w_rec / (l_rec + 1e-9)
                if rec_pf < 0.80 and sum(recent) < 0:
                    # In quarantine: skip trade or take minimal probe
                    continue
                    
            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            idx = t_data["df"].index.get_loc(ts)
            if idx < 100 or not t_data["entry_cond"].iloc[idx]:
                continue
                
            cur_c = t_data["c"].iloc[idx]
            cur_h = t_data["h"].iloc[idx]
            cur_l = t_data["l"].iloc[idx]
            entry_atr = max(t_data["atr14"].iloc[idx], 0.05)
            raw_entry = cur_c * (1.0 + slippage)
            entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
            stop_dist = entry_atr
            active_sl = entry_p - stop_dist
            
            friction = 0.002067 * entry_p
            denom = stop_dist + friction
            base_risk = 12.80
            base_qty = int(base_risk / denom) if denom > 0 else 1
            
            cushion = max(0.0, wallet_equity - 500.0)
            if wallet_equity >= 0.88 * wallet_hwm:
                total_risk = base_risk + (0.24 * cushion)
            else:
                total_risk = max(base_risk, 0.03 * wallet_equity)
                
            num_tks = max(1, math.ceil(total_risk / base_risk))
            tot_qty = base_qty * num_tks
            
            if (entry_p * tot_qty) > avail:
                tot_qty = int(avail / entry_p)
                num_tks = max(1, math.ceil(tot_qty / base_qty))
                
            if tot_qty >= 1:
                positions[ticker] = {
                    "entry_p": entry_p, "entry_t": ts, "total_qty": tot_qty,
                    "num_tickets": num_tks, "qty_per_ticket": max(1, tot_qty // num_tks),
                    "entry_atr": entry_atr, "active_sl": active_sl,
                    "high_seen": entry_p, "phase": 0, "bars_held": 0
                }
                daily_traded[cur_date].add(ticker)
                current_notional += (entry_p * tot_qty)
                avail = (5.0 * wallet_equity) - current_notional
                if avail < 15.0 or len(positions) >= 2:
                    break
                    
    df_pt = pd.DataFrame(parent_trades)
    w = df_pt[df_pt["pnl"] > 0]
    l = df_pt[df_pt["pnl"] < 0]
    gp = w["pnl"].sum()
    gl = abs(l["pnl"].sum())
    pf = gp / gl if gl > 0 else 999.0
    print(f"Quarantine: {quarantine_enabled:<5} | Trades: {len(df_pt):<3} | WR: {len(w)/len(df_pt)*100:.1f}% | PF: {pf:.2f} | Final Wallet: Rs {wallet_equity:.2f} | Peak HWM: Rs {wallet_hwm:.2f}")
