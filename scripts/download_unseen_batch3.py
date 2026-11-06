import sys
import os
import time
from pathlib import Path
import pandas as pd
import pickle
from tvDatafeed import TvDatafeed, Interval

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
OUT_FILE = ROOT_DIR / "data" / "unseen_batch3_10_15m_1y.pkl"

CANDIDATES = [
    "RTNPOWER", "INFIBEAM", "DISHTV", "DEN", "ORIENTPPR",
    "BODALCHEM", "SEPC", "VIKASLIFE", "GTLINFRA", "JPASSOCIAT",
    "HBLPOWER", "FCSSOFT", "TV18BRDCST", "HEG", "GRAPHITE"
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

print("Downloading 10 Fresh 100% Unseen High-Beta Stocks from NSE...")
for sym in CANDIDATES:
    if len(data_dict) >= 10:
        break
    try:
        raw_df = tv.get_hist(sym, "NSE", interval=Interval.in_15_minute, n_bars=5000)
        cleaned = clean_ohlcv(raw_df)
        if cleaned is not None and len(cleaned) >= 4500:
            full_sym = f"{sym}.NS"
            data_dict[full_sym] = cleaned
            print(f"SUCCESS ({len(data_dict)}/10): {sym:<12} -> {len(cleaned)} bars ({cleaned.index[0].date()} to {cleaned.index[-1].date()}) | Last: Rs {cleaned['close'].iloc[-1]:.2f}")
        else:
            bars_count = len(cleaned) if cleaned is not None else 0
            print(f"SKIPPED {sym}: insufficient bars ({bars_count})")
        time.sleep(1.0)
    except Exception as e:
        print(f"ERROR downloading {sym}: {e}")

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "wb") as f:
    pickle.dump(data_dict, f)

print(f"\nSaved {len(data_dict)} 100% UNSEEN HIGH-BETA STOCKS to {OUT_FILE}")
