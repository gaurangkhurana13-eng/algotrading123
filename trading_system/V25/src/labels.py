"""
Labeling and Dynamic Exit Engine for Bhavishya V25
Implements Continuous Volatility-Regime TP + Quadratic MFE Ratchet & Stagnation Time-Decay Stop:
  - Upper Barrier (TP): Scaled continuously by ATR ratio: clip(2.4 - 0.8 * (SMA50(ATR)/ATR - 1), 1.6, 3.2)
  - Lower Barrier (SL):
      * Initial Stop: 1.40 * ATR
      * Stagnation Decay (lambda = 0.12): If 3 bars pass and MFE < 0.5 ATR, stop tightens by 0.12 ATR/bar.
      * Quadratic MFE Ratchet (gamma = 0.28): Ratchets stop upwards quadratically with MFE:
          At MFE=0.5 ATR -> -1.33 ATR
          At MFE=1.0 ATR -> -1.12 ATR
          At MFE=2.0 ATR -> -0.28 ATR (nearly breakeven)
          At MFE=2.5 ATR -> +0.35 ATR (profit locked)
"""
import numpy as np
import pandas as pd

def apply_triple_barrier_labels(df: pd.DataFrame, min_atr=0.005, hold_bars=12) -> pd.DataFrame:
    df   = df.copy()
    n    = len(df)
    c    = df["close"].values
    h    = df["high"].values
    l    = df["low"].values
    atr  = df["atr_norm"].values
    
    # Rolling 50-period mean of ATR for continuous regime scaling
    atr_series = pd.Series(atr)
    atr_sma50  = atr_series.rolling(50, min_periods=10).mean().bfill().values

    y_buy          = np.zeros(n, dtype=int)
    target_pcts    = np.zeros(n, dtype=float)
    sl_pcts        = np.zeros(n, dtype=float)
    realized_pcts  = np.zeros(n, dtype=float)

    for i in range(n - hold_bars):
        entry = c[i]
        curr_atr = max(atr[i], min_atr)
        sma_atr  = max(atr_sma50[i], min_atr)

        # Continuous Volatility-Regime Upper Barrier (TP)
        vol_ratio = (sma_atr / curr_atr) - 1.0
        eff_tp_mult = float(np.clip(2.4 - 0.8 * vol_ratio, 1.6, 3.2))

        tp_dist = curr_atr * eff_tp_mult
        initial_sl_dist = curr_atr * 1.40
        target_pcts[i] = tp_dist
        sl_pcts[i]     = initial_sl_dist

        tp_price = entry * (1.0 + tp_dist)
        running_sl_price = entry * (1.0 - initial_sl_dist)

        hit = False
        highest_high = entry

        for j in range(1, hold_bars + 1):
            curr_h = h[i + j]
            curr_l = l[i + j]
            if curr_h > highest_high:
                highest_high = curr_h

            # Maximum Favorable Excursion in ATR units
            mfe_atr = max(0.0, (highest_high - entry) / (curr_atr * entry + 1e-9))

            # Stagnation Time-Decay (if k > 3 bars and MFE < 0.5 ATR)
            stagnation_decay = 0.12 * max(0, j - 3) if mfe_atr < 0.5 else 0.0

            # Quadratic MFE Ratchet
            mfe_ratchet = 0.28 * (mfe_atr ** 2)

            # Net dynamic stop distance in ATR below entry (or above if ratcheted into profit)
            net_sl_atr = 1.40 - stagnation_decay - mfe_ratchet
            dyn_sl_price = entry - (net_sl_atr * curr_atr * entry)

            # Ratchet guarantee: Stop never loosens downwards
            if dyn_sl_price > running_sl_price:
                running_sl_price = dyn_sl_price

            # Barrier Check
            if curr_h >= tp_price:
                y_buy[i] = 1
                realized_pcts[i] = tp_dist
                hit = True
                break
            elif curr_l <= running_sl_price:
                y_buy[i] = 0
                realized_ret = (running_sl_price - entry) / entry
                realized_pcts[i] = realized_ret
                hit = True
                break

        if not hit:
            # Time exit at bar 12 close
            final_p = c[i + hold_bars]
            ret = (final_p - entry) / entry
            realized_pcts[i] = ret
            y_buy[i] = 1 if ret >= tp_dist * 0.4 else 0

    df["y_buy"]         = y_buy
    df["target_pct"]    = target_pcts
    df["sl_pct"]        = sl_pcts
    df["realized_pct"]  = realized_pcts
    return df.iloc[:n - hold_bars]
