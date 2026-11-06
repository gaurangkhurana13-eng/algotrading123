"""
Data Fetcher & Validation Engine
---------------------------------
Supports dual-source fetching:
1. Yahoo Finance (yfinance) - deep daily and multi-year hourly history
2. TradingView Datafeed (tvDatafeed) - deep intraday history (up to 5000 candles per interval)

Includes rigorous data validation:
- Zero duplication guarantee
- OHLC consistency validation (High >= Low, High >= Open/Close, Low <= Open/Close)
- Automatic deduplication and chronological ordering
- Local caching in CSV and Parquet formats
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import numpy as np
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataFetcher")

# Attempt tvDatafeed import
try:
    from tvDatafeed import TvDatafeed, Interval
    TV_AVAILABLE = True
except ImportError:
    TV_AVAILABLE = False
    logger.warning("tvDatafeed not available, falling back to yfinance only.")


TIMEFRAME_MAP_YF = {
    "1d":  {"interval": "1d",  "period": "5y"},
    "1h":  {"interval": "1h",  "period": "2y"},
    "30m": {"interval": "30m", "period": "60d"},
    "15m": {"interval": "15m", "period": "60d"},
    "5m":  {"interval": "5m",  "period": "60d"},
    "1m":  {"interval": "1m",  "period": "7d"},
}


class DataFetcher:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.tv_client = None
        if TV_AVAILABLE:
            try:
                self.tv_client = TvDatafeed()
            except Exception as e:
                logger.warning(f"Could not initialize TvDatafeed: {e}")

    @staticmethod
    def format_symbols(raw_symbol: str) -> Tuple[str, str, str]:
        """
        Parses symbol into (clean_symbol, yf_symbol, tv_exchange, tv_symbol).
        Example:
            'PCJEWELLER' -> ('PCJEWELLER', 'PCJEWELLER.NS', 'NSE', 'PCJEWELLER')
            'RELIANCE'   -> ('RELIANCE', 'RELIANCE.NS', 'NSE', 'RELIANCE')
            'AAPL'       -> ('AAPL', 'AAPL', 'NASDAQ', 'AAPL')
        """
        sym = raw_symbol.upper().strip()
        if sym.endswith(".NS") or sym.endswith(".BO"):
            base = sym.split(".")[0]
            ex = "NSE" if sym.endswith(".NS") else "BSE"
            return base, sym, ex
        elif ":" in sym:
            ex, base = sym.split(":")
            return base, f"{base}.NS" if ex.upper() == "NSE" else base, ex.upper()
        else:
            # Default Indian equity convention if alphanumeric without exchange
            return sym, f"{sym}.NS", "NSE"

    def fetch_yfinance(self, yf_symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch historical candles via Yahoo Finance."""
        config = TIMEFRAME_MAP_YF.get(timeframe.lower())
        if not config:
            logger.error(f"Unsupported timeframe: {timeframe}")
            return None

        logger.info(f"[yfinance] Fetching {yf_symbol} | interval={config['interval']} period={config['period']}")
        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=config["period"], interval=config["interval"], auto_adjust=True)
            if df is None or df.empty:
                logger.warning(f"[yfinance] No data returned for {yf_symbol}")
                return None

            df = df.reset_index()
            # Standardize column names
            date_col = "Datetime" if "Datetime" in df.columns else "Date"
            df = df.rename(columns={
                date_col: "datetime",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })
            df = df[["datetime", "open", "high", "low", "close", "volume"]]
            return df
        except Exception as e:
            logger.error(f"[yfinance] Error fetching {yf_symbol}: {e}")
            return None

    def fetch_tradingview(self, exchange: str, symbol: str, timeframe: str, n_bars: int = 5000) -> Optional[pd.DataFrame]:
        """Fetch historical candles via TradingView Datafeed (tvDatafeed)."""
        if not self.tv_client:
            logger.warning("[tvDatafeed] Client not initialized.")
            return None

        tf_map_tv = {
            "1d": Interval.in_daily,
            "1h": Interval.in_1_hour,
            "30m": Interval.in_30_minute,
            "15m": Interval.in_15_minute,
            "5m": Interval.in_5_minute,
            "1m": Interval.in_1_minute,
        }
        interval = tf_map_tv.get(timeframe.lower())
        if not interval:
            logger.error(f"[tvDatafeed] Unsupported timeframe: {timeframe}")
            return None

        logger.info(f"[tvDatafeed] Fetching {exchange}:{symbol} | interval={timeframe} n_bars={n_bars}")
        for attempt in range(3):
            try:
                df = self.tv_client.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)
                if df is not None and not df.empty:
                    df = df.reset_index()
                    df = df.rename(columns={
                        "datetime": "datetime",
                        "open": "open",
                        "high": "high",
                        "low": "low",
                        "close": "close",
                        "volume": "volume"
                    })
                    df = df[["datetime", "open", "high", "low", "close", "volume"]]
                    return df
            except Exception as e:
                logger.warning(f"[tvDatafeed] Attempt {attempt+1} failed: {e}")
                time.sleep(1)

        logger.error(f"[tvDatafeed] Failed to fetch data after 3 attempts.")
        return None

    @staticmethod
    def validate_and_clean(df: pd.DataFrame, source_label: str = "") -> pd.DataFrame:
        """
        Performs rigorous data validation:
        1. Datetime conversion and sorting
        2. Deduplication on timestamp (keeps last)
        3. Drop rows with nulls in OHLC
        4. Validates OHLC integrity (High >= Low, High >= Open/Close, Low <= Open/Close)
        5. Fills or cleans volume
        """
        initial_len = len(df)
        df = df.copy()

        # 1. Datetime parsing
        df["datetime"] = pd.to_datetime(df["datetime"])
        # Remove timezone offset for seamless alignment across sources
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

        # 2. Sort chronologically
        df = df.sort_values("datetime").reset_index(drop=True)

        # 3. Deduplicate
        dup_count = df.duplicated(subset=["datetime"]).sum()
        if dup_count > 0:
            df = df.drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)
            logger.info(f"[{source_label}] Removed {dup_count} duplicate timestamps.")

        # 4. Check for null values
        df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

        # 5. OHLC Consistency Validation
        invalid_mask = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"]) |
            (df["close"] <= 0) |
            (df["open"] <= 0)
        )
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            logger.warning(f"[{source_label}] Removed {invalid_count} corrupted/anomalous candles.")
            df = df[~invalid_mask].reset_index(drop=True)

        # 6. Volume sanity
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0).clip(lower=0)

        # Set datetime index
        df = df.set_index("datetime")

        logger.info(
            f"[{source_label} Cleaned] Candles: {len(df)} (Original: {initial_len}) | "
            f"Range: {df.index.min().strftime('%Y-%m-%d %H:%M')} -> {df.index.max().strftime('%Y-%m-%d %H:%M')}"
        )
        return df

    def get_data(self, raw_symbol: str, timeframe: str, force_download: bool = False, prefer_tv: bool = False) -> pd.DataFrame:
        """
        Gets clean dataset.
        Checks local cache first, otherwise fetches from TradingView / Yahoo Finance.
        """
        base_sym, yf_sym, tv_ex = self.format_symbols(raw_symbol)
        csv_path = os.path.join(self.data_dir, f"{base_sym}_{timeframe.lower()}.csv")

        if not force_download and os.path.exists(csv_path):
            logger.info(f"Loading cached data from {csv_path}")
            df = pd.read_csv(csv_path, parse_dates=["datetime"], index_col="datetime")
            return df

        df = None
        # Try preferred source or fallback
        if prefer_tv and self.tv_client:
            df = self.fetch_tradingview(tv_ex, base_sym, timeframe, n_bars=5000)
            source = "TradingView"

        if df is None or df.empty:
            df = self.fetch_yfinance(yf_sym, timeframe)
            source = "yfinance"

        # If yfinance failed or had few rows and tv is available, try tv
        if (df is None or len(df) < 100) and self.tv_client:
            df_tv = self.fetch_tradingview(tv_ex, base_sym, timeframe, n_bars=5000)
            if df_tv is not None and not df_tv.empty:
                df = df_tv
                source = "TradingView (Fallback)"

        if df is None or df.empty:
            raise ValueError(f"Could not fetch data for {raw_symbol} on timeframe {timeframe} from any source.")

        # Validate and clean
        df_clean = self.validate_and_clean(df, source_label=f"{base_sym} {timeframe} via {source}")

        # Save to cache
        df_clean.to_csv(csv_path)
        logger.info(f"Saved verified dataset to {csv_path}")
        return df_clean

    def download_all_timeframes(self, raw_symbol: str, timeframes: Tuple[str, ...] = ("1d", "1h", "30m", "15m", "5m")) -> Dict[str, pd.DataFrame]:
        """Downloads and caches multiple timeframes for a symbol."""
        datasets = {}
        for tf in timeframes:
            try:
                # For intraday (15m, 5m), TradingView datafeed has more candles (up to 5000) than Yahoo (60 days)
                prefer_tv = tf in ["5m", "15m", "30m"]
                df = self.get_data(raw_symbol, tf, force_download=True, prefer_tv=prefer_tv)
                datasets[tf] = df
            except Exception as e:
                logger.error(f"Failed to fetch {raw_symbol} for timeframe {tf}: {e}")
        return datasets
