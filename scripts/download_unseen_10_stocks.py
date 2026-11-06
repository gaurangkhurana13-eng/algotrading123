import sys
import os
import time
from pathlib import Path
import pandas as pd
import pickle
from tvDatafeed import TvDatafeed, Interval

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
OUT_FILE = ROOT_DIR / "data" / "unseen_10_15m_1y.pkl"

# 10 100% GENUINELY UNSEEN, NEVER-BEFORE-TOUCHED STOCKS (Outside the 78 universe)
UNSEEN_TICKERS = [
    "BHEL", "BEL", "NATIONALUM", "TATACHEM", "EXIDEIND",
    "BIOCON", "ZEEL", "HINDALCO", "SUNTV", "ESCORTS"
]

def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep='last')]
    df = df.sort_index()
    df.columns = [c.lower() for c in df.columns]
    required = ['open', 'high', 'low', 'close', 'volume']
    for req in required:
        if req not in df.columns:
            return None
    df = df[required]
    df = df.dropna(subset=['close'])
    df['volume'] = df['volume'].fillna(0)
    return df

tv = TvDatafeed()
data_dict = {}

print("Starting download of 10 UNSEEN stocks...")
for sym in UNSEEN_TICKERS:
    try:
        raw_df = tv.get_hist(sym, "NSE", interval=Interval.in_15_minute, n_bars=5000)
        cleaned = clean_ohlcv(raw_df)
        if cleaned is not None and len(cleaned) >= 4000:
            full_sym = f"{sym}.NS"
            data_dict[full_sym] = cleaned
            print(f"SUCCESS: {sym} -> {len(cleaned)} bars ({cleaned.index[0].date()} to {cleaned.index[-1].date()})")
        else:
            print(f"FAILED or insufficient bars: {sym}")
        time.sleep(1.0)
    except Exception as e:
        print(f"ERROR downloading {sym}: {e}")

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "wb") as f:
    pickle.dump(data_dict, f)

print(f"\nCompleted! Saved {len(data_dict)} UNSEEN stocks to {OUT_FILE}")
