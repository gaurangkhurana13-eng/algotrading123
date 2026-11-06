"""
Benchmark Soft Gate for Bhavishya V25: Asymmetric Nifty 50 Liquidation Filter
Computes Nifty 50 Z-score relative to intraday mean and 3-bar EMA21 slope.
Blocks long entries ONLY when Z_Nifty < -1.25 AND Delta EMA21 < 0 (macro waterfall sell-off).
Strictly shifted by 1 bar to eliminate future lookahead.
"""
import pandas as pd
import numpy as np

def compute_nifty_regime(raw_data: pd.DataFrame) -> pd.Series:
    if "^NSEI" not in raw_data.columns.levels[0]:
        return pd.Series(dtype=bool)

    nifty = raw_data["^NSEI"].dropna().copy()
    c = nifty["Close"]
    
    # 21-period EMA and 3-bar slope
    ema21 = c.ewm(span=21).mean()
    delta_ema21 = ema21.diff(3)

    # Intraday rolling mean and standard deviation
    rolling_mean = c.rolling(20, min_periods=5).mean()
    rolling_std  = c.rolling(20, min_periods=5).std() + 1e-9
    z_nifty = (c - rolling_mean) / rolling_std

    # Macro Waterfall condition: Z < -1.25 AND Delta EMA21 < 0
    macro_waterfall = (z_nifty < -1.25) & (delta_ema21 < 0)
    
    # Allow entry unless macro waterfall is happening
    allowed = ~macro_waterfall
    # Shift by 1 bar to strictly guarantee out-of-sample decision validity
    allowed_lagged = allowed.shift(1).fillna(True).astype(bool)
    return allowed_lagged
