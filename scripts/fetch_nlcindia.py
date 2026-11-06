import pickle
from pathlib import Path
from tvDatafeed import TvDatafeed, Interval

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
tv = TvDatafeed()
try:
    df = tv.get_hist("NLCINDIA", "NSE", interval=Interval.in_15_minute, n_bars=5000)
    if df is not None and len(df) >= 4500:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']].dropna(subset=['close'])
        df['volume'] = df['volume'].fillna(0)
        out_path = ROOT_DIR / "data" / "nlcindia_15m_1y.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(df, f)
        print(f"SUCCESS: NLCINDIA downloaded {len(df)} bars to {out_path}")
    else:
        print(f"FAILED or insufficient bars: {len(df) if df is not None else 0}")
except Exception as e:
    print(f"ERROR: {e}")
