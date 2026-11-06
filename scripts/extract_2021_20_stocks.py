import os
import pickle
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
OUT_DIR = ROOT_DIR / "data" / "2021-22"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KAGGLE_DIR = Path(r"C:\Users\admin\.cache\kagglehub\datasets\yeashusemwal\nse-stocks-15min-data-from-2019-03-01\versions\5")

# 20 stocks available for 2021
STOCKS_20 = [
    'CENTRALBK', 'DELTACORP', 'GSFC', 'IDEA', 'IOB', 'JYOTHYLAB', 'NBCC', 'NCC',
    'NIACL', 'NMDC', 'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'SUZLON',
    'SWSOLAR', 'TATASTEEL', 'TTML', 'ZEEL'
]

universe_30 = {}
audit_list = []

print("=== STARTING 2021 DATA EXTRACTION FOR 20 CORE HIGH-BETA STOCKS ===")

for sym in STOCKS_20:
    csv_file = KAGGLE_DIR / f"{sym}.csv"
    if not csv_file.exists():
        print(f"[ERROR] File not found: {csv_file}")
        continue
        
    df_raw = pd.read_csv(csv_file)
    
    # Rename columns to lowercase standard
    df_raw.columns = [c.lower() for c in df_raw.columns]
    
    # Date parsing
    date_col = 'datetime' if 'datetime' in df_raw.columns else 'date'
    df_raw['datetime'] = pd.to_datetime(df_raw[date_col])
    
    # Filter for full year 2021 (2021-01-01 to 2021-12-31)
    df_2021 = df_raw[(df_raw['datetime'] >= '2021-01-01') & (df_raw['datetime'] <= '2021-12-31 23:59:59')].copy()
    
    # Set index
    df_2021 = df_2021.sort_values('datetime').set_index('datetime')
    
    # Keep only standard OHLCV
    req_cols = ['open', 'high', 'low', 'close', 'volume']
    df_2021 = df_2021[req_cols].copy()
    
    # Filter market hours 09:15 to 15:15
    df_2021 = df_2021.between_time('09:15', '15:15')
    
    # Drop any NaNs
    df_2021 = df_2021.dropna()
    
    # Save individual CSV
    sym_csv_path = OUT_DIR / f"{sym}.csv"
    df_2021.to_csv(sym_csv_path)
    
    # Save individual PKL for Gate 1 Check 2 (<symbol>_15m_1y.pkl)
    sym_pkl_path = OUT_DIR / f"{sym.lower()}_15m_1y.pkl"
    with open(sym_pkl_path, 'wb') as f:
        pickle.dump(df_2021, f)
        
    # Store both clean and .NS mapping in master dictionary
    universe_30[sym] = df_2021
    universe_30[f"{sym}.NS"] = df_2021
    
    days = len(set(df_2021.index.date))
    bars = len(df_2021)
    min_dt = df_2021.index.min()
    max_dt = df_2021.index.max()
    
    print(f"  -> {sym:12s}: {bars:5d} bars | {days:3d} days | {min_dt.strftime('%Y-%m-%d')} to {max_dt.strftime('%Y-%m-%d')}")
    audit_list.append({
        'symbol': sym,
        'bars': bars,
        'days': days,
        'start_date': str(min_dt),
        'end_date': str(max_dt)
    })

# Save master universe_30_15m.pkl
master_file = OUT_DIR / "universe_30_15m.pkl"
with open(master_file, "wb") as f:
    pickle.dump(universe_30, f)

print(f"\n[SUCCESS] Master 2021 dataset saved: {master_file}")
print(f"Total Unique Stocks: {len(universe_30)//2} | Total Mappings: {len(universe_30)}")

audit_df = pd.DataFrame(audit_list)
print("\n=== AUDIT SUMMARY ===")
print(audit_df.describe())
