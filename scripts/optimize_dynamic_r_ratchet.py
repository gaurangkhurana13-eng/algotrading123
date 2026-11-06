import pickle, sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from framework import calc_dhan_charges
from scripts.test_dynamic_r_architect import trade_contexts, slippage

print("=" * 105)
print("OPTIMIZING DYNAMIC R-MULTIPLE RATCHET ARCHITECTURE FOR BHAVISHYA V42")
print("Goal: Maximize Average Win (target > Rs 50 - Rs 80) and Total Net PnL while minimizing Drawdown")
print("=" * 105)

def run_ratchet_simulation(
    be_trigger_r: float,       # R at which BE is activated (e.g. 1.0R)
    be_lock_offset_atr: float, # ATR above entry for BE (e.g. 0.12 ATR to cover fees)
    lock1_trigger_r: float,    # R at which Stage 1 profit is locked (e.g. 1.5R)
    lock1_lock_r: float,       # R locked at Stage 1 (e.g. 0.8R)
    lock2_trigger_r: float,    # R at which Stage 2 profit is locked (e.g. 2.0R)
    lock2_lock_r: float,       # R locked at Stage 2 (e.g. 1.5R)
    target_r: float,           # Asymmetric Target (e.g. 2.0R, 2.5R, 3.0R)
    trailing_buffer_r: float = 0.50 # ATR/R buffer above lock2
):
    net_pnls = []
    reasons = []
    realized_rs = []
    
    for tc in trade_contexts:
        entry_p = tc['entry_p']
        initial_sl = tc['initial_sl']
        risk = tc['risk']
        entry_atr = tc['entry_atr']
        shares = tc['shares']
        bars = tc['bars']
        
        target_p = round(entry_p + target_r * risk, 2)
        be_p = round(entry_p + be_lock_offset_atr * entry_atr, 2)
        
        active_sl = initial_sl
        high_seen = entry_p
        
        exit_fill = None
        exit_reason = None
        
        for ts, row in bars.iterrows():
            cur_h = row['high']
            cur_l = row['low']
            cur_c = row['close']
            high_seen = max(high_seen, cur_h)
            mfe_r = (high_seen - entry_p) / risk
            
            # --- Dynamic R Ladder Update ---
            if mfe_r >= lock2_trigger_r:
                # Stage 2 Lock
                trail = high_seen - (trailing_buffer_r * risk)
                active_sl = max(active_sl, round(max(entry_p + lock2_lock_r * risk, trail), 2))
            elif mfe_r >= lock1_trigger_r:
                # Stage 1 Lock
                active_sl = max(active_sl, round(entry_p + lock1_lock_r * risk, 2))
            elif mfe_r >= be_trigger_r:
                # Breakeven Lock
                active_sl = max(active_sl, be_p)
                
            # Check Target Hit
            if cur_h >= target_p:
                exit_fill = target_p
                exit_reason = f"TARGET_{target_r}R"
                break
                
            # Check Stop Hit
            if cur_l <= active_sl:
                raw_fill = min(cur_h, active_sl) * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                exit_reason = "STOP"
                break
                
            # EOD
            if ts.time() >= pd.Timestamp("14:45").time():
                raw_fill = cur_c * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                exit_reason = "EOD"
                break
                
        if exit_fill is None:
            last_c = bars.iloc[-1]['close'] if len(bars) > 0 else entry_p
            exit_fill = round(last_c * (1.0 - slippage), 2)
            exit_reason = "EOD"

        gross = round((exit_fill - entry_p) * shares, 2)
        chg = round(calc_dhan_charges(entry_p, exit_fill, shares, is_intraday=True), 2)
        net = round(gross - chg, 2)
        realized_r = round((exit_fill - entry_p) / risk, 2)
        
        net_pnls.append(net)
        reasons.append(exit_reason)
        realized_rs.append(realized_r)
        
    s_pnl = pd.Series(net_pnls)
    wins = s_pnl[s_pnl > 0]
    losses = s_pnl[s_pnl < 0]
    
    n = len(s_pnl)
    wr = len(wins) / n * 100.0
    tot_net = s_pnl.sum()
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    med_win = wins.median() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    pf = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else 999.0
    
    eq = [500.0]
    for p in net_pnls: eq.append(eq[-1] + p)
    eq_s = pd.Series(eq)
    peak = eq_s.cummax()
    mdd = ((peak - eq_s) / peak * 100.0).max()
    
    return {
        "wr": wr,
        "tot_net": tot_net,
        "avg_win": avg_win,
        "med_win": med_win,
        "avg_loss": avg_loss,
        "wl_ratio": abs(avg_win / avg_loss) if avg_loss != 0 else 0.0,
        "pf": pf,
        "mdd": mdd,
        "target_hits": sum(1 for r in reasons if r.startswith("TARGET"))
    }

# Run Grid Search across potential candidate combinations
grid = []
for be_trig in [0.75, 0.90, 1.00, 1.20]:
    for be_off in [0.08, 0.12, 0.15]:
        for l1_trig, l1_lock in [(1.30, 0.60), (1.40, 0.70), (1.50, 0.80), (1.50, 1.00)]:
            for l2_trig, l2_lock in [(1.80, 1.30), (2.00, 1.50), (2.20, 1.60)]:
                for tgt in [2.0, 2.25, 2.5, 2.75, 3.0]:
                    if tgt <= l2_trig: continue
                    res = run_ratchet_simulation(be_trig, be_off, l1_trig, l1_lock, l2_trig, l2_lock, tgt)
                    grid.append({
                        "BE_Trig": be_trig, "BE_Off": be_off,
                        "L1_Trig": l1_trig, "L1_Lock": l1_lock,
                        "L2_Trig": l2_trig, "L2_Lock": l2_lock,
                        "Target_R": tgt,
                        **res
                    })

df_grid = pd.DataFrame(grid)
print(f"Total Grid Permutations Tested: {len(df_grid)}")

# Top architectures by Total Net PnL
print("\nTop 10 Architectures by Total Net PnL:")
top_net = df_grid.sort_values(by="tot_net", ascending=False).head(10)
print(top_net[["BE_Trig", "L1_Trig", "L1_Lock", "L2_Trig", "L2_Lock", "Target_R", "wr", "avg_win", "avg_loss", "pf", "tot_net", "mdd", "target_hits"]].to_string(index=False))

# Top architectures by Average Win (with Net > 0)
print("\nTop 10 Architectures by Average Win (with Net PnL > Rs 1,000):")
top_win = df_grid[df_grid["tot_net"] > 1000].sort_values(by="avg_win", ascending=False).head(10)
print(top_win[["BE_Trig", "L1_Trig", "L1_Lock", "L2_Trig", "L2_Lock", "Target_R", "wr", "avg_win", "avg_loss", "pf", "tot_net", "mdd", "target_hits"]].to_string(index=False))

# Top architectures by Profit Factor (with Net > 1,500)
print("\nTop 10 Architectures by Profit Factor (with Net PnL > Rs 1,500):")
top_pf = df_grid[df_grid["tot_net"] > 1500].sort_values(by="pf", ascending=False).head(10)
print(top_pf[["BE_Trig", "L1_Trig", "L1_Lock", "L2_Trig", "L2_Lock", "Target_R", "wr", "avg_win", "avg_loss", "pf", "tot_net", "mdd", "target_hits"]].to_string(index=False))

df_grid.to_csv(ROOT / "reports/dynamic_r_grid_results.csv", index=False)
print("\nSaved full grid results to reports/dynamic_r_grid_results.csv")
