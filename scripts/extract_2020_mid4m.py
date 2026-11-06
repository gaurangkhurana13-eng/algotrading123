import pickle
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
OUT_DIR = ROOT_DIR / "data" / "2020_MID4M"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KAGGLE_DIR = Path(r"C:\Users\admin\.cache\kagglehub\datasets\yeashusemwal\nse-stocks-15min-data-from-2019-03-01\versions\5")

core_30 = [
    'CANBK', 'DELTACORP', 'DEVYANI', 'EQUITASBNK', 'FEDERALBNK', 'GMRINFRA', 'GSFC',
    'IDFCFIRSTB', 'IOC', 'IRFC', 'JYOTHYLAB', 'MANAPPURAM', 'MOTHERSON', 'NBCC',
    'NCC', 'NATIONALUM', 'NMDC', 'POONAWALLA', 'PNB', 'RBLBANK', 'RCF', 'SAIL',
    'SUZLON', 'TATACHEM', 'TATASTEEL', 'TATAPOWER', 'TRIDENT', 'UNIONBANK', 'YESBANK', 'ZEEL'
]

universe_30 = {}
audit_list = []

print("=== EXTRACTING 2020 MID 4 MONTHS (01 MAY 2020 TO 31 AUG 2020) ===")

for sym in core_30:
    csv_file = KAGGLE_DIR / f"{sym}.csv"
    if not csv_file.exists():
        print(f"[-] Missing CSV: {sym}")
        continue
        
    df_raw = pd.read_csv(csv_file)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    
    date_col = 'datetime' if 'datetime' in df_raw.columns else 'date'
    df_raw['datetime'] = pd.to_datetime(df_raw[date_col])
    
    # 2020 Mid 4 Months: May 1 to Aug 31
    sub = df_raw[(df_raw['datetime'] >= '2020-05-01') & (df_raw['datetime'] <= '2020-08-31 23:59:59')].copy()
    
    if len(sub) == 0:
        print(f"[-] {sym}: No data in May-Aug 2020")
        continue
        
    sub = sub.sort_values('datetime').set_index('datetime')
    req_cols = ['open', 'high', 'low', 'close', 'volume']
    sub = sub[req_cols].copy()
    sub = sub.between_time('09:15', '15:15')
    sub = sub.dropna()
    
    # Save individual CSV
    sym_csv = OUT_DIR / f"{sym}.csv"
    sub.to_csv(sym_csv)
    
    # Save individual pkl for Gate 1 Check 2
    sym_pkl = OUT_DIR / f"{sym.lower()}_15m_1y.pkl"
    with open(sym_pkl, 'wb') as f:
        pickle.dump(sub, f)
        
    universe_30[sym] = sub
    universe_30[f"{sym}.NS"] = sub
    
    days = len(set(sub.index.date))
    bars = len(sub)
    min_dt = sub.index.min()
    max_dt = sub.index.max()
    
    print(f"  -> {sym:12s}: {bars:4d} bars | {days:2d} days | {min_dt.strftime('%Y-%m-%d')} to {max_dt.strftime('%Y-%m-%d')}")
    audit_list.append({
        'symbol': sym,
        'bars': bars,
        'days': days,
        'min_date': str(min_dt),
        'max_date': str(max_dt)
    })

master_file = OUT_DIR / "universe_30_15m.pkl"
with open(master_file, "wb") as f:
    pickle.dump(universe_30, f)

print(f"\n[SUCCESS] Master 2020 Mid 4M dataset saved: {master_file}")
print(f"Total Valid Stocks: {len(universe_30)//2} | Total Dict Keys: {len(universe_30)}")

audit_df = pd.DataFrame(audit_list)
print("\n=== AUDIT SUMMARY ===")
print(audit_df.describe())
