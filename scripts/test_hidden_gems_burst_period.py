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

gems = ["TEXRAIL.NS", "JPPOWER.NS", "PTC.NS", "NIACL.NS", "MANALIPETC.NS"]
burst_months = ["2026-07", "2026-08", "2026-09", "2026-10"]

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

for ticker in gems:
    if ticker not in univ:
        continue
    df = univ[ticker]
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
        (rsi7 > 50.0) & (adx > 18.0) & (plus_di > minus_di)
    )
    
    # Check Price Action in Burst Period
    df["m_str"] = df.index.to_period("M").astype(str)
    df_burst_price = df[df["m_str"].isin(burst_months)]
    start_p = df_burst_price["close"].iloc[0] if len(df_burst_price) else 0
    end_p = df_burst_price["close"].iloc[-1] if len(df_burst_price) else 0
    high_p = df_burst_price["high"].max() if len(df_burst_price) else 0
    low_p = df_burst_price["low"].min() if len(df_burst_price) else 0
    
    # Simulate single stock trades in burst months
    pos = None
    trades = []
    for ts, row in df.iterrows():
        m_str = str(ts.to_period("M"))
        if m_str not in burst_months:
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
            # Flat Rs 20 unit risk for pure comparison
            qty = max(1, int(20.0 / denom))
            pos = {
                "entry_p": entry_p, "qty": qty, "entry_atr": cur_atr,
                "active_sl": active_sl, "high_seen": entry_p, "bars_held": 0, "phase": 0
            }
            
    df_t = pd.Series(trades) if len(trades) else pd.Series([0])
    wins = (df_t > 0).sum() if len(trades) else 0
    wr = (wins / len(trades)) * 100 if len(trades) else 0
    net = df_t.sum() if len(trades) else 0
    gw = df_t[df_t > 0].sum() if len(trades) else 0
    gl = abs(df_t[df_t < 0].sum()) if len(trades) else 1e-9
    pf = (gw / gl) if gl > 0 else 0
    
    results.append({
        "Ticker": ticker.replace(".NS", ""),
        "Trades (Jul-Oct)": len(trades),
        "Win Rate (%)": round(wr, 1),
        "Net PnL (Rs)": round(net, 2),
        "Profit Factor": round(pf, 2),
        "Start Price": round(start_p, 2),
        "End Price": round(end_p, 2),
        "Price Change (%)": round(((end_p - start_p) / start_p) * 100, 1) if start_p else 0
    })

df_gems_burst = pd.DataFrame(results)
print("=== PERFORMANCE OF THE 5 GEMS DURING BURST MONTHS (JUL - OCT 2026) ===")
print(df_gems_burst.to_string(index=False))
