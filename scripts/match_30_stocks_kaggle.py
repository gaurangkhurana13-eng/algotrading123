import os
import glob
import pandas as pd

KAGGLE_DIR = r"C:\Users\admin\.cache\kagglehub\datasets\yeashusemwal\nse-stocks-15min-data-from-2019-03-01\versions\5"

v29_universe = [
    'CENTRALBK', 'DELTACORP', 'ELECTCAST', 'GSFC', 'IDEA', 'IFCI', 'IOB', 'IRCON',
    'J&KBANK', 'JYOTHYLAB', 'MANALIPETC', 'MTNL', 'NBCC', 'NCC', 'NIACL', 'NMDC',
    'PCJEWELLER', 'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'RPOWER', 'SUZLON',
    'SWSOLAR', 'TATASTEEL', 'TEXRAIL', 'TTML', 'UTKARSHBNK', 'ZEEL'
]

print("=== INSPECTING KAGGLE DOWNLOADED 15M CSV FILES ===")
files = os.listdir(KAGGLE_DIR)
file_map = {f.upper().replace(".CSV", ""): f for f in files if f.endswith(".csv")}

found = []
missing = []

for sym in v29_universe:
    # Handle possible symbol variants: J&KBANK -> JKBANK or J&KBANK
    clean_sym = sym.upper().replace("&", "")
    target = None
    if sym in file_map:
        target = file_map[sym]
    elif clean_sym in file_map:
        target = file_map[clean_sym]
    elif f"{sym}.NS" in file_map:
        target = file_map[f"{sym}.NS"]
        
    if target:
        found.append((sym, target))
    else:
        missing.append(sym)

print(f"\nTotal Target Stocks: {len(v29_universe)}")
print(f"Matched Stocks: {len(found)} / {len(v29_universe)}")
print(f"Missing Stocks: {len(missing)}: {missing}")

# Check sample matched stock for 2021 date range and format
if found:
    sym, fname = found[0]
    fpath = os.path.join(KAGGLE_DIR, fname)
    df = pd.read_csv(fpath)
    print(f"\n--- Sample Stock: {sym} ({fname}) ---")
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
    print(df.head(3))
    print(df.tail(3))
    
    # Check date column
    date_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()][0]
    df[date_col] = pd.to_datetime(df[date_col])
    df_2021 = df[(df[date_col] >= '2021-01-01') & (df[date_col] <= '2021-12-31 23:59:59')]
    print(f"Total 2021 bars for {sym}: {len(df_2021)} (Min: {df_2021[date_col].min()}, Max: {df_2021[date_col].max()})")
