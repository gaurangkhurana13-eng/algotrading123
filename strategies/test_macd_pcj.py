"""
Test Intraday vs Swing for MACD + RSI + ADX + Ratchet on 1-Year PC Jeweller Data
"""

import pandas as pd
import numpy as np
from framework import calc_dhan_charges, calculate_atr, compute_metrics

df = pd.read_pickle("C:/Users/admin/.gemini/antigravity/scratch/new_trade/data/pcjeweller_15m_1y.pkl")
c = df["close"]
h = df["high"]
l = df["low"]
o = df["open"]
v = df["volume"]
atr14 = calculate_atr(df, 14)

ema50 = c.ewm(span=50).mean()

delta = c.diff()
gain = delta.clip(lower=0).rolling(7).mean()
loss = (-delta.clip(upper=0)).rolling(7).mean()
rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))

macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
sig = macd.ewm(span=9).mean()

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
    (macd > sig) & 
    (macd.shift(1) <= sig.shift(1)) & 
    (macd > 0) & 
    (c > ema50) & 
    (rsi7 > 52) & 
    (adx > 18) & 
    (plus_di > minus_di)
)

def run_pcj_sim(mode="intraday"):
    capital = 500.0
    notional_cap = 2450.0 if mode == "intraday" else 500.0
    slippage = 0.0005
    sl_mult = 1.15
    p1_trig = 0.90
    p1_lock = 0.35
    p2_trig = 2.00
    p2_lock = 1.20
    tp_mult = 3.50 if mode == "swing" else 2.50
    
    trades = []
    pos = 0
    entry_p = 0.0
    entry_t = None
    qty = 0
    active_sl = 0.0
    active_tp = 0.0
    high_seen = 0.0
    phase = 0
    entry_atr = 0.0
    
    for i in range(100, len(df)):
        ts = df.index[i]
        t = ts.time()
        cur_c = c.iloc[i]
        cur_h = h.iloc[i]
        cur_l = l.iloc[i]
        
        if mode == "intraday" and pos == 1 and t >= pd.Timestamp("15:10").time():
            exit_p = cur_c * (1.0 - slippage)
            gross = (exit_p - entry_p) * qty
            chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=True)
            net = round(gross - chg, 2)
            trades.append({
                "entry_time": str(entry_t), "exit_time": str(ts),
                "entry_price": round(entry_p, 2), "exit_price": round(exit_p, 2),
                "qty": qty, "gross_pnl": round(gross, 2), "charges": round(chg, 2),
                "net_pnl": net, "exit_reason": "15:10_INTRADAY_SQUARE_OFF"
            })
            pos = 0
            continue
            
        if pos == 1:
            high_seen = max(high_seen, cur_h)
            mfe_atr = (high_seen - entry_p) / entry_atr if entry_atr > 0 else 0
            
            if mfe_atr >= p2_trig and phase < 2:
                phase = 2
                active_sl = max(active_sl, entry_p + p2_lock * entry_atr)
            elif mfe_atr >= p1_trig and phase < 1:
                phase = 1
                active_sl = max(active_sl, entry_p + p1_lock * entry_atr)
                
            if cur_h >= active_tp:
                exit_p = active_tp * (1.0 - slippage)
                gross = (exit_p - entry_p) * qty
                chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=(mode=="intraday"))
                net = round(gross - chg, 2)
                trades.append({
                    "entry_time": str(entry_t), "exit_time": str(ts),
                    "entry_price": round(entry_p, 2), "exit_price": round(exit_p, 2),
                    "qty": qty, "gross_pnl": round(gross, 2), "charges": round(chg, 2),
                    "net_pnl": net, "exit_reason": "TAKE_PROFIT_HIT"
                })
                pos = 0
            elif cur_l <= active_sl:
                fill_p = min(active_sl, o.iloc[i])
                exit_p = fill_p * (1.0 - slippage)
                gross = (exit_p - entry_p) * qty
                chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=(mode=="intraday"))
                net = round(gross - chg, 2)
                trades.append({
                    "entry_time": str(entry_t), "exit_time": str(ts),
                    "entry_price": round(entry_p, 2), "exit_price": round(exit_p, 2),
                    "qty": qty, "gross_pnl": round(gross, 2), "charges": round(chg, 2),
                    "net_pnl": net, "exit_reason": "SL_OR_RATCHET_HIT"
                })
                pos = 0
                
        if pos == 0:
            time_ok = (t >= pd.Timestamp("09:20").time()) and (t <= pd.Timestamp("13:30").time() if mode=="intraday" else True)
            if entry_cond.iloc[i] and time_ok:
                pos = 1
                entry_t = ts
                entry_p = cur_c * (1.0 + slippage)
                entry_atr = max(atr14.iloc[i], 0.05)
                active_sl = entry_p - sl_mult * entry_atr
                active_tp = entry_p + tp_mult * entry_atr
                high_seen = entry_p
                phase = 0
                
                # Sizing
                stop_dist = sl_mult * entry_atr
                friction_share = 0.002067 * entry_p
                denom = stop_dist + friction_share
                qty_risk = int(15.50 / denom) if denom > 0 else 1
                qty_notional = int(notional_cap / entry_p)
                qty = max(1, min(qty_risk, qty_notional))
                
    return pd.DataFrame(trades)

tdf_intra = run_pcj_sim(mode="intraday")
tdf_swing = run_pcj_sim(mode="swing")

m_in = compute_metrics(tdf_intra, 500.0)
m_sw = compute_metrics(tdf_swing, 500.0)

print("=== INTRADAY MIS ===")
print(f"Trades: {m_in['total_trades']}, WR: {m_in['win_rate']}%, Net PnL: Rs {m_in['net_pnl']}, PF: {m_in['profit_factor']}, AvgWin: Rs {m_in['avg_win']}, AvgLoss: Rs {m_in['avg_loss']}")
print("\n=== SWING / DELIVERY ===")
print(f"Trades: {m_sw['total_trades']}, WR: {m_sw['win_rate']}%, Net PnL: Rs {m_sw['net_pnl']}, PF: {m_sw['profit_factor']}, AvgWin: Rs {m_sw['avg_win']}, AvgLoss: Rs {m_sw['avg_loss']}")
