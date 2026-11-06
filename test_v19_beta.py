import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

# Load Data Pools
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    d_beta = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
    d_unseen = pickle.load(f)

all_pools = {}
all_pools.update(univ)
all_pools.update(d_beta)
all_pools.update(d_unseen)

tickers_combined = [
    "PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS", "RITES.NS",
    "IDEA.NS", "MTNL.NS", "ZEEL.NS", "UTKARSHBNK.NS", "IFCI.NS", "NCC.NS",
    "TEXRAIL.NS", "JPPOWER.NS", "PTC.NS", "NIACL.NS"
]

def compute_vwap_shifted(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    df_temp["_date"] = df_temp.index.date
    tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
    df_temp["_tpv"] = tp * df_temp["volume"]
    cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
    cum_vol = df_temp.groupby("_date")["volume"].cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).shift(1)

def compute_er10(series_close: pd.Series) -> pd.Series:
    change = (series_close - series_close.shift(10)).abs()
    volatility = series_close.diff().abs().rolling(10).sum()
    return change / (volatility + 1e-9)

data_map = {}
for t in tickers_combined:
    if t not in all_pools:
        continue
    df = all_pools[t].copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    atr14 = calculate_atr(df, 14)
    ema50 = c.ewm(span=50).mean()
    ema110 = c.ewm(span=110).mean()
    vwap_lag = compute_vwap_shifted(df)
    v_sma20 = v.rolling(20).mean()
    rvol = v / (v_sma20 + 1e-9)
    er10 = compute_er10(c)
    
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
        (c <= 450.0) & (c >= 10.0)
    )
    
    data_map[t] = {
        "df": df, "c": c, "h": h, "l": l, "v": v, "atr14": atr14,
        "rsi7": rsi7, "vwap_lag": vwap_lag, "rvol": rvol, "er10": er10,
        "entry_cond": entry_cond
    }

all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
slippage = 0.0005
hard_sl_mult = 0.90
trail_dist = 0.28
climax_dist = 0.20
ratchet_dist = 0.40

for beta in [0.035, 0.055, 0.065, 0.075]:
    positions = {}
    trades = []
    daily_traded = {}
    last_exit_times = set()

    for ts in all_ts:
        if ts.weekday() >= 5:
            continue
        cur_date = ts.date()
        if cur_date not in daily_traded:
            daily_traded[cur_date] = set()
        t_time = ts.time()
        t_time_int = t_time.hour * 100 + t_time.minute
        
        # Manage active positions
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
            
            # --- 3-PHASE ASYMMETRIC LIFECYCLE ---
            if mfe_atr >= 2.50 or (mfe_atr >= 1.25 and t_time_int >= 1415):
                p["active_sl"] = max(p["active_sl"], round(p["high_seen"] - (climax_dist * p["entry_atr"]), 2))
                p["phase"] = 3
            elif mfe_atr >= 1.15:
                p["active_sl"] = max(p["active_sl"], round(p["high_seen"] - (trail_dist * p["entry_atr"]), 2))
                p["phase"] = 2
            elif mfe_atr >= 0.85:
                p["active_sl"] = max(p["active_sl"], round(p["entry_p"] + (ratchet_dist * p["entry_atr"]), 2))
                p["phase"] = 1

            # Stop hit check
            if cur_l <= p["active_sl"]:
                raw_exit = p["active_sl"] * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                reason = "PHASE_3_TRAILING_STOP_HIT" if p["phase"] == 3 else ("PHASE_2_TRAILING_STOP_HIT" if p["phase"] == 2 else ("PHASE_1_RATCHET_STOP_HIT" if p["phase"] == 1 else "STOP_LOSS_HIT"))
                trades.append({"net_pnl": net, "gross_pnl": gross, "chg": chg})
                last_exit_times.add(ts)
                del positions[ticker]
                continue
                
            # Stagnation scratch
            if p["bars_held"] >= 3 and mfe_atr < 0.40 and cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                raw_exit = cur_c * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                trades.append({"net_pnl": net, "gross_pnl": gross, "chg": chg})
                last_exit_times.add(ts)
                del positions[ticker]
                continue
                
            # 14:45 Square-off
            if t_time >= pd.Timestamp("14:45").time():
                raw_exit = cur_c * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                trades.append({"net_pnl": net, "gross_pnl": gross, "chg": chg})
                last_exit_times.add(ts)
                del positions[ticker]
                continue

        # Sizing
        ek = 500.0 + sum(t["net_pnl"] for t in trades)
        hwm = 500.0
        cum = 500.0
        for t in trades:
            cum += t["net_pnl"]
            if cum > hwm:
                hwm = cum
                
        consec_losses = 0
        for t in reversed(trades):
            if t["net_pnl"] < 0:
                consec_losses += 1
            else:
                break
                
        dd_pct = (hwm - ek) / max(hwm, 500.0)
        
        if len(positions) >= 1 or ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
            continue
            
        current_notional = sum(p["entry_p"] * p["qty"] for p in positions.values())
        avail_margin = (0.95 * ek) - (current_notional / 5.0)
        if avail_margin < 10.0:
            continue
        avail_notional = avail_margin * 5.0
        
        # Gate V2 official risk cap
        gate_v2_cap = min(0.09 * ek, 260.0)
        if dd_pct > 0.15:
            gate_v2_cap *= 0.5
            
        if consec_losses >= 2 or (ek > 2500.0 and dd_pct > 0.16):
            cur_alpha, cur_beta = 0.015, 0.010
        elif consec_losses == 1:
            cur_alpha, cur_beta = 0.035, 0.020
        else:
            cur_alpha, cur_beta = 0.082, beta
            
        if ek <= 2500.0:
            raw_r = 14.0 + (cur_alpha * max(0.0, ek - 500.0))
        else:
            raw_r = 14.0 + (cur_alpha * 2000.0) + (cur_beta * (ek - 2500.0))
            
        candidates = []
        for ticker in tickers_combined:
            if ticker in positions or ticker in daily_traded[cur_date]:
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
            cur_v = t_data["v"].iloc[idx]
            entry_atr = max(t_data["atr14"].iloc[idx], 0.05)
            conviction_score = t_data["rvol"].iloc[idx] * t_data["er10"].iloc[idx]
            candidates.append({
                "ticker": ticker, "c": cur_c, "h": cur_h, "l": cur_l, "v": cur_v,
                "atr": entry_atr, "score": conviction_score, "er": t_data["er10"].iloc[idx], "rvol": t_data["rvol"].iloc[idx]
            })
            
        if not candidates:
            continue
            
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]
        ticker = best["ticker"]
        cur_c = best["c"]
        cur_h = best["h"]
        cur_l = best["l"]
        cur_v = best["v"]
        entry_atr = best["atr"]
        cur_er = best["er"]
        cur_rvol = best["rvol"]
        
        er_bonus = 0.25 * max(0.0, cur_er - 0.35)
        rvol_bonus = 0.05 * min(max(0.0, cur_rvol - 1.5), 3.0)
        weight = min(1.20, max(0.85, 0.85 + er_bonus + rvol_bonus))
        
        target_risk = min(raw_r * weight, gate_v2_cap)
        raw_entry = cur_c * (1.0 + slippage)
        entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
        active_sl = round(entry_p - (hard_sl_mult * entry_atr), 2)
        sl_dist = max(entry_p - active_sl, 0.05)
        friction = 0.002067 * entry_p
        denom = sl_dist + friction
        
        qty_r = int(target_risk / denom)
        qty_m = int((avail_notional * 0.90) / entry_p)
        qty = min(qty_r, qty_m)
        if cur_v > 0:
            qty = min(qty, int(0.045 * cur_v))
            
        if qty >= 1:
            positions[ticker] = {
                "entry_p": entry_p, "qty": qty, "entry_atr": entry_atr,
                "active_sl": active_sl, "initial_sl": active_sl,
                "high_seen": entry_p, "bars_held": 0, "entry_t": ts, "phase": 0
            }
            daily_traded[cur_date].add(ticker)

    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["net_pnl"] > 0]
    wr = len(wins) / len(tdf) * 100.0 if len(tdf) > 0 else 0
    net_pnl = tdf["net_pnl"].sum()
    fin_wallet = 500.0 + net_pnl
    worst = tdf["net_pnl"].min()
    print(f"Beta Slope: {beta:<5} | Trades: {len(tdf)} | WR: {wr:.2f}% | Net: Rs {net_pnl:>8.2f} | Final: Rs {fin_wallet:>8.2f} ({fin_wallet/500:.1f}x) | Worst: Rs {worst:.2f}")
