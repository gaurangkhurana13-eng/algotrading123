import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    universe = pickle.load(f)

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

def compute_adx_di(df: pd.DataFrame, n: int = 10):
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    atr_s = tr.rolling(n).mean()
    pdi = 100 * pd.Series(pdm, index=df.index).rolling(n).mean() / (atr_s + 1e-9)
    mdi = 100 * pd.Series(mdm, index=df.index).rolling(n).mean() / (atr_s + 1e-9)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    adx = dx.rolling(n).mean()
    return adx, pdi, mdi

results = []

for sym, df in universe.items():
    if len(df) < 500:
        continue
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    recent_p = c.iloc[-1]
    if recent_p < 10.0 or recent_p > 450.0:
        continue
        
    atr14 = calculate_atr(df, 14)
    atr_pct = (atr14 / c) * 100.0
    if atr_pct.dropna().mean() < 0.40:
        continue
        
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
    
    adx, pdi, mdi = compute_adx_di(df, 10)
    lookback = min(300, len(c) - 1)
    mom_cond = (c > c.shift(lookback))
    
    df["_date"] = df.index.date
    day_open = df.groupby("_date")["open"].transform("first")
    day_ret = (c / day_open) - 1.0
    h1_ret = (c / c.shift(4)) - 1.0
    
    cond = (
        macd_cross_up & (macd > 0) &
        (c > ema50) & (c > ema110) &
        (c > vwap_lag) &
        (rsi7 > 50.0) & (adx >= 18.0) & (pdi > mdi) &
        (er10 >= 0.35) &
        (rvol >= 1.2) &
        (atr_pct >= 0.50) &
        mom_cond &
        (c <= 450.0) & (c >= 10.0)
    )
    
    # Backtest individual symbol
    trades = []
    pos = None
    slippage = 0.0005
    hard_sl_mult = 0.90
    fixed_risk = 25.0
    
    for idx in range(100, len(df)):
        ts = df.index[idx]
        cur_c = c.iloc[idx]
        cur_h = h.iloc[idx]
        cur_l = l.iloc[idx]
        cur_v = v.iloc[idx]
        t_time = ts.time()
        
        if pos is not None:
            pos["bars_held"] += 1
            pos["high_seen"] = max(pos["high_seen"], cur_h)
            mfe_atr = (pos["high_seen"] - pos["entry_p"]) / pos["entry_atr"] if pos["entry_atr"] > 0 else 0
            
            # V26 Ladder
            if mfe_atr >= 3.00:
                stop_ladder = pos["high_seen"] - (0.14 * pos["entry_atr"])
            elif mfe_atr >= 2.00:
                stop_ladder = pos["high_seen"] - (0.20 * pos["entry_atr"])
            elif mfe_atr >= 1.15:
                stop_ladder = pos["high_seen"] - (0.28 * pos["entry_atr"])
            elif mfe_atr >= 0.85:
                stop_ladder = pos["entry_p"] + (0.40 * pos["entry_atr"])
            else:
                stop_ladder = pos["entry_p"] - (0.90 * pos["entry_atr"])
                
            pos["active_sl"] = max(pos["active_sl"], round(stop_ladder, 2))
            
            exit_fill, reason = None, None
            if cur_l <= pos["active_sl"]:
                exit_fill = pos["active_sl"] * (1.0 - slippage)
                reason = "STOP"
            elif pos["bars_held"] >= 4 and mfe_atr < 0.40 and cur_c < pos["entry_p"]:
                exit_fill = cur_c * (1.0 - slippage)
                reason = "STAG"
            elif t_time >= pd.Timestamp("14:45").time():
                exit_fill = cur_c * (1.0 - slippage)
                reason = "EOD"
                
            if exit_fill is not None:
                qty = pos["qty"]
                gross = qty * (exit_fill - pos["entry_p"])
                chg = calc_dhan_charges(pos["entry_p"], exit_fill, qty, is_intraday=True)
                net = gross - chg
                trades.append(net)
                pos = None
                
        if pos is None and cond.iloc[idx] and pd.Timestamp("09:20").time() <= t_time <= pd.Timestamp("14:15").time():
            entry_p = round(cur_c * (1.0 + slippage), 2)
            entry_atr = max(atr14.iloc[idx], 0.05)
            active_sl = round(entry_p - (hard_sl_mult * entry_atr), 2)
            sl_dist = max(entry_p - active_sl, 0.05)
            qty = max(1, int(fixed_risk / sl_dist))
            if cur_v > 0:
                qty = min(qty, int(0.045 * cur_v))
            pos = {
                "entry_p": entry_p, "qty": qty, "entry_atr": entry_atr,
                "active_sl": active_sl, "high_seen": entry_p, "bars_held": 0
            }
            
    if len(trades) > 0:
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]
        wr = len(wins) / len(trades) * 100.0
        tot_net = sum(trades)
        net_w = sum(wins)
        net_l = abs(sum(losses))
        pf = round(net_w / net_l, 2) if net_l > 0 else 999.0
        results.append({
            "symbol": sym, "trades": len(trades), "wr": round(wr, 1),
            "net": round(tot_net, 2), "pf": pf, "avg_win": round(np.mean(wins), 2) if wins else 0,
            "avg_loss": round(np.mean(losses), 2) if losses else 0
        })

rdf = pd.DataFrame(results).sort_values("net", ascending=False)
print("TOP 25 PROFITABLE STOCKS ACROSS 78 UNIVERSE:")
print(rdf.head(25).to_string(index=False))
print("\nOVERALL STATS:")
pos_stocks = rdf[rdf["net"] > 0]
print(f"Profitable Stocks: {len(pos_stocks)} out of {len(rdf)}")
print(f"Total Net PnL of Top 20 stocks: Rs {rdf.head(20)['net'].sum():.2f}")
