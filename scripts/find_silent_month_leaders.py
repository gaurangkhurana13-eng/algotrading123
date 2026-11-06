import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)

# Filter universe for stocks with reasonable price and liquidity
silent_months = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]

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

results = []

for ticker, df in univ.items():
    if len(df) < 500:
        continue
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    
    avg_price = c.mean()
    if avg_price < 15.0 or avg_price > 350.0:
        continue
    avg_daily_vol = v.rolling(25).sum().mean()
    if avg_daily_vol < 100000:
        continue
        
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
        (rsi7 > 50.0) & (adx > 18.0) & (plus_di > minus_di)
    )
    
    # Simulate single stock trades in silent months
    pos = None
    trades = []
    for ts, row in df.iterrows():
        m_str = str(ts.to_period("M"))
        if m_str not in silent_months:
            continue
            
        cur_c, cur_h, cur_l = row["close"], row["high"], row["low"]
        cur_atr = atr14.loc[ts]
        cur_rsi = rsi7.loc[ts]
        cur_vwap = vwap_lag.loc[ts]
        t_time = ts.time()
        t_time_int = t_time.hour * 100 + t_time.minute
        
        if pos is not None:
            pos["bars_held"] += 1
            pos["high_seen"] = max(pos["high_seen"], cur_h)
            mfe_atr = (pos["high_seen"] - pos["entry_p"]) / pos["entry_atr"] if pos["entry_atr"] > 0 else 0.0
            
            # V19 3-Phase Harvester
            if mfe_atr >= 2.50 or (mfe_atr >= 1.25 and t_time_int >= 1415):
                pos["active_sl"] = max(pos["active_sl"], round(pos["high_seen"] - (0.20 * pos["entry_atr"]), 2))
                pos["phase"] = 3
            elif mfe_atr >= 1.15:
                pos["active_sl"] = max(pos["active_sl"], round(pos["high_seen"] - (0.28 * pos["entry_atr"]), 2))
                pos["phase"] = 2
            elif mfe_atr >= 0.85:
                pos["active_sl"] = max(pos["active_sl"], round(pos["entry_p"] + (0.40 * pos["entry_atr"]), 2))
                pos["phase"] = 1
                
            if cur_l <= pos["active_sl"]:
                exit_p = round(pos["active_sl"] * 0.9995, 2)
                gross = (exit_p - pos["entry_p"]) * pos["qty"]
                chg = calc_dhan_charges(pos["entry_p"], exit_p, pos["qty"], is_intraday=True)
                trades.append(gross - chg)
                pos = None
                continue
                
            if pos["bars_held"] >= 3 and mfe_atr < 0.40 and cur_c < pos["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                exit_p = round(cur_c * 0.9995, 2)
                gross = (exit_p - pos["entry_p"]) * pos["qty"]
                chg = calc_dhan_charges(pos["entry_p"], exit_p, pos["qty"], is_intraday=True)
                trades.append(gross - chg)
                pos = None
                continue
                
            if t_time >= pd.Timestamp("14:45").time():
                exit_p = round(cur_c * 0.9995, 2)
                gross = (exit_p - pos["entry_p"]) * pos["qty"]
                chg = calc_dhan_charges(pos["entry_p"], exit_p, pos["qty"], is_intraday=True)
                trades.append(gross - chg)
                pos = None
                continue
                
        elif entry_cond.loc[ts] and pd.Timestamp("09:20").time() <= t_time <= pd.Timestamp("14:15").time():
            entry_p = round(cur_c * 1.0005, 2)
            active_sl = round(entry_p - (0.90 * cur_atr), 2)
            denom = cur_atr + 0.002067 * entry_p
            # Standard test size Rs 20 unit risk
            qty = max(1, int(20.0 / denom))
            pos = {
                "entry_p": entry_p, "qty": qty, "entry_atr": cur_atr,
                "active_sl": active_sl, "high_seen": entry_p, "bars_held": 0, "phase": 0
            }
            
    if len(trades) >= 5:
        df_t = pd.Series(trades)
        wins = (df_t > 0).sum()
        wr = (wins / len(df_t)) * 100
        net = df_t.sum()
        gw = df_t[df_t > 0].sum()
        gl = abs(df_t[df_t < 0].sum())
        pf = (gw / gl) if gl > 0 else 0
        results.append({
            "ticker": ticker.replace(".NS", ""),
            "avg_price": round(avg_price, 1),
            "trades": len(trades),
            "win_rate": round(wr, 1),
            "net_pnl": round(net, 2),
            "pf": round(pf, 2)
        })

df_res = pd.DataFrame(results).sort_values("net_pnl", ascending=False)
print("=== TOP PERFORMERS DURING SILENT MONTHS (JAN - MAY 2026) ===")
print(df_res.head(15).to_string(index=False))
