"""
Triple Barrier Labeling Module for Bhavishya V35
Adapts target multiples based on local volatility regime:
  - Volatility Compression (Squeeze / Low ATR z-score < -0.5): TP expanded to 2.4 * ATR
  - Volatility Expansion (High ATR z-score > 1.0): TP tightened to 1.7 * ATR
  - Normal Regime: TP = 2.0 * ATR
  - Stop Loss: Invariant 1.5 * ATR (strict capital protection)
Strictly trims hold_bars at the tail so no label leaks into future bars.
"""
import numpy as np
import pandas as pd

def apply_triple_barrier_labels(df: pd.DataFrame, tp_mult=2.0, sl_mult=1.5, min_atr=0.0045, hold_bars=12) -> pd.DataFrame:
    df   = df.copy()
    n    = len(df)
    c    = df["close"].values
    h    = df["high"].values
    l    = df["low"].values
    atr  = df["atr_norm"].values
    atr_z = df["atr_norm_z"].values if "atr_norm_z" in df.columns else np.zeros(n)
    sqz  = df["squeeze_on"].values if "squeeze_on" in df.columns else np.zeros(n)

    y_buy          = np.zeros(n, dtype=int)
    target_pcts    = np.zeros(n, dtype=float)
    sl_pcts        = np.zeros(n, dtype=float)
    realized_pcts  = np.zeros(n, dtype=float)

    for i in range(n - hold_bars):
        entry = c[i]
        curr_atr = max(atr[i], min_atr)
        
        # Volatility Regime Switching Target Multiplier
        if sqz[i] == 1 or atr_z[i] < -0.5:
            eff_tp_mult = 2.4
        elif atr_z[i] > 1.0:
            eff_tp_mult = 1.7
        else:
            eff_tp_mult = 2.0

        tp_dist = curr_atr * eff_tp_mult
        sl_dist = curr_atr * sl_mult
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
            final_ret = (c[i + hold_bars] - entry) / entry
            realized_pcts[i] = final_ret
            y_buy[i] = 1 if final_ret >= (tp_dist * 0.5) else 0

    df["y_buy"]         = y_buy
    df["target_pct"]    = target_pcts
    df["sl_pct"]        = sl_pcts
    df["realized_pct"]  = realized_pcts

    return df.iloc[:n - hold_bars].copy()
