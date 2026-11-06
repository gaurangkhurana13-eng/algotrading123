import pickle, sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from framework import calc_dhan_charges

# Load cached trade contexts
from scripts.test_dynamic_r_architect import trade_contexts, slippage

print("=" * 90)
print("DEEP INVESTIGATION: HOW MANY TRADES REACH TARGETS BEFORE HITTING INITIAL SL?")
print("No early choke, letting the setup breathe with full 1R initial stop")
print("=" * 90)

results = []
for target_r in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]:
    wins = 0
    losses = 0
    eod = 0
    pnl_list = []
    
    for tc in trade_contexts:
        entry_p = tc['entry_p']
        initial_sl = tc['initial_sl']
        risk = tc['risk']
        shares = tc['shares']
        bars = tc['bars']
        target_p = round(entry_p + target_r * risk, 2)
        
        trade_exit = None
        trade_reason = None
        
        for ts, row in bars.iterrows():
            cur_h = row['high']
            cur_l = row['low']
            cur_c = row['close']
            
            # Check target hit first
            if cur_h >= target_p:
                trade_exit = target_p
                trade_reason = "TARGET"
                break
                
            # Check initial SL hit
            if cur_l <= initial_sl:
                trade_exit = round(max(cur_l, min(cur_h, initial_sl * (1 - slippage))), 2)
                trade_reason = "SL"
                break
                
            # EOD
            if ts.time() >= pd.Timestamp("14:45").time():
                trade_exit = round(cur_c * (1 - slippage), 2)
                trade_reason = "EOD"
                break
                
        if trade_exit is None:
            last_c = bars.iloc[-1]['close'] if len(bars) > 0 else entry_p
            trade_exit = round(last_c * (1 - slippage), 2)
            trade_reason = "EOD"
            
        gross = round((trade_exit - entry_p) * shares, 2)
        chg = round(calc_dhan_charges(entry_p, trade_exit, shares, is_intraday=True), 2)
        net = round(gross - chg, 2)
        pnl_list.append(net)
        
        if trade_reason == "TARGET":
            wins += 1
        elif trade_reason == "SL":
            losses += 1
        else:
            if net > 0: wins += 1
            else: losses += 1
            eod += 1
            
    pnl_series = pd.Series(pnl_list)
    win_trades = pnl_series[pnl_series > 0]
    loss_trades = pnl_series[pnl_series < 0]
    wr = len(win_trades) / len(pnl_series) * 100.0
    tot_net = pnl_series.sum()
    avg_win = win_trades.mean() if len(win_trades) > 0 else 0
    avg_loss = loss_trades.mean() if len(loss_trades) > 0 else 0
    pf = win_trades.sum() / abs(loss_trades.sum()) if abs(loss_trades.sum()) > 0 else 999.0
    
    results.append({
        "Target R": f"{target_r:.2f}R",
        "Target Price Dist": f"+{target_r * 0.90:.2f} ATR",
        "Win Rate (%)": round(wr, 1),
        "Avg Win (Rs)": round(avg_win, 2),
        "Avg Loss (Rs)": round(avg_loss, 2),
        "Win/Loss Ratio": round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0,
        "Profit Factor": round(pf, 2),
        "Total Net (Rs)": round(tot_net, 2),
        "Target Hits": wins - (eod if eod > 0 else 0)
    })

df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))
