"""
Triple Barrier Labeling Module for Pure BUY-Only Trades.
Barriers:
  Target = Entry + (tp_mult * ATR)
  Stop Loss = Entry - (sl_mult * ATR)
  Time Barrier = 12 bars (3 hours max holding)
"""
import numpy as np
import pandas as pd

def apply_triple_barrier_labels(df: pd.DataFrame, tp_mult=2.0, sl_mult=1.5, min_atr=0.005, hold_bars=12) -> pd.DataFrame:
    df   = df.copy()
    n    = len(df)
    c    = df["close"].values
    h    = df["high"].values
    l    = df["low"].values
    atr  = df["atr_norm"].values

    y_buy          = np.zeros(n, dtype=int)
    target_pcts    = np.zeros(n, dtype=float)
    sl_pcts        = np.zeros(n, dtype=float)
    realized_pcts  = np.zeros(n, dtype=float)

    for i in range(n - hold_bars):
        entry = c[i]
        curr_atr = max(atr[i], min_atr)
        
        tp_dist = curr_atr * tp_mult      # +2.0 * ATR
        sl_dist = curr_atr * sl_mult      # -1.5 * ATR
        target_pcts[i] = tp_dist
        sl_pcts[i]     = sl_dist

        tp_price = entry * (1.0 + tp_dist)
        sl_price = entry * (1.0 - sl_dist)

        hit = False
        for j in range(1, hold_bars + 1):
            hi = h[i + j]
            lo = l[i + j]

            if hi >= tp_price:
                y_buy[i] = 1
                realized_pcts[i] = tp_dist
                hit = True
                break
            elif lo <= sl_price:
                y_buy[i] = 0
                realized_pcts[i] = -sl_dist
                hit = True
                break

        if not hit:
            final_p = c[i + hold_bars]
            ret = (final_p - entry) / entry
            realized_pcts[i] = ret
            y_buy[i] = 1 if ret >= tp_dist * 0.5 else 0

    df["y_buy"]         = y_buy
    df["target_pct"]    = target_pcts
    df["sl_pct"]        = sl_pcts
    df["realized_pct"]  = realized_pcts
    return df.iloc[:n - hold_bars]
