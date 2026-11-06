import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

DATA_FILE = ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl"
with open(DATA_FILE, "rb") as f:
    univ = pickle.load(f)

tickers = list(univ.keys())

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
    
    # Exact V11 Entry
    entry_cond = (
        macd_cross_up & (macd > 0) &
        (c > ema50) & (c > ema110) &
        (c > vwap_lag) &
        (rsi7 > 50.0) & (adx > 18.0) & (plus_di > minus_di)
    )
    data_map[t] = {
        "df": df, "c": c, "h": h, "l": l, "atr14": atr14,
        "rsi7": rsi7, "vwap_lag": vwap_lag, "entry_cond": entry_cond
    }

# Run Individual Unconstrained Ticker Performance
ticker_summaries = []
all_trades = []

slippage = 0.0005
p1_trig = 0.85
p1_lock = 0.40
p3_trig = 1.35
p3_trail = 0.45
trail_dist = 0.65

for t in tickers:
    t_data = data_map[t]
    df = t_data["df"]
    c = t_data["c"]
    h = t_data["h"]
    l = t_data["l"]
    atr14 = t_data["atr14"]
    rsi7 = t_data["rsi7"]
    vwap_lag = t_data["vwap_lag"]
    entry_cond = t_data["entry_cond"]
    
    in_pos = False
    p = {}
    trades = []
    
    for i in range(100, len(df)):
        ts = df.index[i]
        t_time = ts.time()
        cur_c = c.iloc[i]
        cur_h = h.iloc[i]
        cur_l = l.iloc[i]
        cur_rsi = rsi7.iloc[i]
        cur_vwap = vwap_lag.iloc[i]
        
        if in_pos:
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
                
            # SL check
            if cur_l <= p["active_sl"]:
                raw_exit = p["active_sl"] * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                trades.append({"symbol": t.replace(".NS", ""), "gross": gross, "chg": chg, "net": net, "exit_reason": "SL"})
                in_pos = False
                continue
                
            # Scratch
            if p["bars_held"] >= 3 and p["phase"] == 0 and mfe_atr < 0.40:
                if cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                    raw_exit = cur_c * (1.0 - slippage)
                    exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                    gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                    chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                    net = round(gross - chg, 2)
                    trades.append({"symbol": t.replace(".NS", ""), "gross": gross, "chg": chg, "net": net, "exit_reason": "SCRATCH"})
                    in_pos = False
                    continue
                    
            # 15:00 Square-off
            if t_time >= pd.Timestamp("15:00").time():
                raw_exit = cur_c * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                trades.append({"symbol": t.replace(".NS", ""), "gross": gross, "chg": chg, "net": net, "exit_reason": "SQUAREOFF"})
                in_pos = False
                continue
        else:
            if entry_cond.iloc[i] and pd.Timestamp("09:20").time() <= t_time <= pd.Timestamp("13:30").time():
                entry_atr = max(atr14.iloc[i], 0.05)
                raw_entry = cur_c * (1.0 + slippage)
                entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
                stop_dist = entry_atr
                active_sl = entry_p - stop_dist
                friction = 0.002067 * entry_p
                # Risk budget Rs 12.80 (cap <= Rs 15.50 PASS)
                qty_risk = int(12.80 / (stop_dist + friction))
                qty_cap = int(2450.0 / entry_p)
                qty = min(qty_risk, qty_cap)
                if qty >= 1:
                    p = {
                        "entry_p": entry_p, "qty": qty, "entry_atr": entry_atr,
                        "active_sl": active_sl, "high_seen": entry_p, "phase": 0, "bars_held": 0
                    }
                    in_pos = True
                    
    df_t = pd.DataFrame(trades)
    if len(df_t) > 0:
        w = df_t[df_t["net"] > 0]
        l = df_t[df_t["net"] < 0]
        gp = w["gross"].sum()
        gl = abs(l["gross"].sum())
        pf = gp / gl if gl > 0 else 999.0
        ticker_summaries.append({
            "ticker": t.replace(".NS", ""),
            "trades": len(df_t),
            "wins": len(w),
            "losses": len(l),
            "win_rate": round(len(w) / len(df_t) * 100, 1),
            "net_pnl": round(df_t["net"].sum(), 2),
            "pf": round(pf, 2),
            "avg_win": round(w["net"].mean(), 2) if len(w) > 0 else 0.0,
            "avg_loss": round(l["net"].mean(), 2) if len(l) > 0 else 0.0,
            "worst_loss": round(df_t["net"].min(), 2)
        })
        all_trades.extend(trades)

df_sum = pd.DataFrame(ticker_summaries).sort_values("net_pnl", ascending=False)
print("=========================================================================================")
print("100% UNTOUCHED OUT-OF-SAMPLE TEST: 10 HIGH-BETA LOW-PRICE STOCKS")
print("=========================================================================================")
print(df_sum.to_string(index=False))

df_all = pd.DataFrame(all_trades)
w_all = df_all[df_all["net"] > 0]
l_all = df_all[df_all["net"] < 0]
gp_all = w_all["gross"].sum()
gl_all = abs(l_all["gross"].sum())
pf_all = gp_all / gl_all if gl_all > 0 else 999.0

print("\n=========================================================================================")
print(f"OUT-OF-SAMPLE AGGREGATE RESULTS (10 STOCKS COMBINED):")
print(f"Total Trades        : {len(df_all)}")
print(f"Wins / Losses       : {len(w_all)} Wins / {len(l_all)} Losses")
print(f"Win Rate            : {len(w_all)/len(df_all)*100:.2f}%")
print(f"Gross Profit        : +Rs {gp_all:.2f}")
print(f"Gross Loss          : -Rs {gl_all:.2f}")
print(f"Net Realized PnL    : +Rs {df_all['net'].sum():.2f}")
print(f"Profit Factor       : {pf_all:.2f}")
print(f"Avg Win / Avg Loss  : +Rs {w_all['net'].mean():.2f} / Rs {l_all['net'].mean():.2f} (Payoff: {w_all['net'].mean()/abs(l_all['net'].mean()):.2f}x)")
print(f"Worst Single Loss   : Rs {df_all['net'].min():.2f} (Cap <= Rs 15.50: {'PASS' if df_all['net'].min() >= -15.50 else 'FAIL'})")
print("=========================================================================================")
