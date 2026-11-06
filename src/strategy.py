"""
Strategy Engine
---------------
Implements:
1. DailyMidpointStrategy: Exact replication of Previous Day 50% Midpoint Breakout (Pine Script v5 logic)
2. AIConfluenceStrategy: Triple-Filter Trend, Momentum & Volatility Engine (Pine Script v5 logic)
"""

from typing import Tuple, Optional
import pandas as pd
import numpy as np


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Wilder's ATR matching TradingView Pine Script ta.atr(14)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's RMA (standard in TradingView)
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    return atr


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates RSI matching TradingView Pine Script ta.rsi(14)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's Smoothing
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


class DailyMidpointStrategy:
    """
    Replicates TradingView '50% Daily Midpoint Strategy'.
    Maps previous day High and Low onto intraday candles without lookahead bias.
    """
    def __init__(self, sl_multiplier: float = 1.5, tp_multiplier: float = 3.0, atr_period: int = 14):
        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier
        self.atr_period = atr_period

    def prepare_data(self, intraday_df: pd.DataFrame, daily_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Merges daily High/Low onto intraday bars.
        If daily_df is not provided, extracts daily bars from intraday data.
        """
        df = intraday_df.copy()
        df["date"] = df.index.date

        if daily_df is not None:
            d_df = daily_df.copy()
            d_df["date"] = d_df.index.date
            daily_stats = d_df.groupby("date").agg({
                "high": "max",
                "low": "min"
            }).rename(columns={"high": "prev_high", "low": "prev_low"})
        else:
            # Aggregate from intraday
            daily_stats = df.groupby("date").agg({
                "high": "max",
                "low": "min"
            }).rename(columns={"high": "prev_high", "low": "prev_low"})

        # Shift by 1 day so today gets yesterday's High and Low (ZERO lookahead bias!)
        daily_stats_shifted = daily_stats.shift(1)
        daily_stats_shifted["prev_midpoint"] = (daily_stats_shifted["prev_high"] + daily_stats_shifted["prev_low"]) / 2.0

        # Merge onto intraday DataFrame
        merged = df.merge(daily_stats_shifted, left_on="date", right_index=True, how="left")
        merged.index = df.index
        merged = merged.drop(columns=["date"])

        # Forward fill any edge cases
        merged["prev_high"] = merged["prev_high"].ffill()
        merged["prev_low"] = merged["prev_low"].ffill()
        merged["prev_midpoint"] = merged["prev_midpoint"].ffill()

        # ATR calculation
        merged["atr"] = calculate_atr(merged, period=self.atr_period)

        # Drop initial rows where prev_midpoint is NaN
        merged = merged.dropna(subset=["prev_midpoint", "atr"]).copy()

        return merged

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates buy and sell signals on candle close crossovers.
        """
        df = df.copy()
        close = df["close"]
        prev_close = close.shift(1)
        midpoint = df["prev_midpoint"]
        prev_midpoint = midpoint.shift(1)

        # ta.crossover(close, midpoint)
        df["buy_signal"] = (prev_close <= prev_midpoint) & (close > midpoint)
        # ta.crossunder(close, midpoint)
        df["sell_signal"] = (prev_close >= prev_midpoint) & (close < midpoint)

        # Dynamic Stop Loss and Take Profit
        df["long_sl"] = df["close"] - (df["atr"] * self.sl_multiplier)
        df["long_tp"] = df["close"] + (df["atr"] * self.tp_multiplier)
        df["short_sl"] = df["close"] + (df["atr"] * self.sl_multiplier)
        df["short_tp"] = df["close"] - (df["atr"] * self.tp_multiplier)

        return df


class AIConfluenceStrategy:
    """
    Replicates TradingView 'AI Confluence Signal Engine':
    - EMA 20, 50, 200
    - RSI 14
    - ATR 14
    - Confluence filtering + State Debounce
    """
    def __init__(self, ema_fast: int = 20, ema_slow: int = 50, ema_base: int = 200, 
                 rsi_len: int = 14, atr_len: int = 14, sl_mult: float = 1.5, tp_mult: float = 3.0):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_base = ema_base
        self.rsi_len = rsi_len
        self.atr_len = atr_len
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # EMAs
        df["ema_fast"] = df["close"].ewm(span=self.ema_fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.ema_slow, adjust=False).mean()
        df["ema_base"] = df["close"].ewm(span=self.ema_base, adjust=False).mean()

        # RSI & ATR
        df["rsi"] = calculate_rsi(df["close"], period=self.rsi_len)
        df["atr"] = calculate_atr(df, period=self.atr_len)

        # 1. Macro Filter
        macro_bull = (df["close"] > df["ema_base"]) & (df["ema_fast"] > df["ema_slow"])
        macro_bear = (df["close"] < df["ema_base"]) & (df["ema_fast"] < df["ema_slow"])

        # 2. Momentum Filter
        momentum_bull = (df["rsi"] > 50) & (df["rsi"] < 70)
        momentum_bear = (df["rsi"] < 50) & (df["rsi"] > 30)

        # 3. Volatility Breakout Filter
        vol_bull = df["close"] > (df["ema_fast"] + (0.5 * df["atr"]))
        vol_bear = df["close"] < (df["ema_fast"] - (0.5 * df["atr"]))

        raw_buy = macro_bull & momentum_bull & vol_bull
        raw_sell = macro_bear & momentum_bear & vol_bear

        # Debounce State (State Machine matching Pine Script)
        buy_signals = np.zeros(len(df), dtype=bool)
        sell_signals = np.zeros(len(df), dtype=bool)
        current_state = 0  # 1 for Long, -1 for Short

        for i in range(len(df)):
            if raw_buy.iloc[i] and current_state != 1:
                buy_signals[i] = True
                current_state = 1
            elif raw_sell.iloc[i] and current_state != -1:
                sell_signals[i] = True
                current_state = -1

        df["buy_signal"] = buy_signals
        df["sell_signal"] = sell_signals

        df["long_sl"] = df["close"] - (df["atr"] * self.sl_mult)
        df["long_tp"] = df["close"] + (df["atr"] * self.tp_mult)
        df["short_sl"] = df["close"] + (df["atr"] * self.sl_mult)
        df["short_tp"] = df["close"] - (df["atr"] * self.tp_mult)

        return df
