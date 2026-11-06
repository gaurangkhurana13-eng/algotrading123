import os
import duckdb
import pandas as pd
import pickle
import datetime

# Define folders
BASE_DATA_DIR = "data"
FOLDERS = {
    "2022-23": ("2022-01-01", "2023-01-01"),
    "2023-24": ("2023-01-01", "2024-01-01"),
    "2024-25": ("2024-01-01", "2025-01-01"),
    "2025-26": ("2025-01-01", "2026-02-01")
}

for folder in FOLDERS.keys():
    os.makedirs(os.path.join(BASE_DATA_DIR, folder), exist_ok=True)

# Shard mapping discovered from locate_14_stocks.py
SHARD_MAP = {
    1: ['DELTACORP'],
    2: ['IFCI', 'IDEA'],
    3: ['JPPOWER'],
    4: ['PCJEWELLER', 'MTNL', 'NCC', 'NIACL'],
    5: ['RPOWER', 'PTC', 'RITES'],
    6: ['TEXRAIL', 'UTKARSHBNK'],
    7: ['ZEEL']
}

base_url = "https://huggingface.co/datasets/xxparthparekhxx/indian-stock-market-minute-data/resolve/main/minute/"

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")

# Containers for each folder
universe_by_folder = {f: {} for f in FOLDERS.keys()}
audit_stats = []

print("=== STARTING MULTI-YEAR 14-STOCK DATA EXTRACTION (2022 - 2026) ===")

for shard_id, symbols in SHARD_MAP.items():
    shard_url = f"{base_url}train-0000{shard_id}.parquet"
    sym_list_str = ",".join([f"'{s}'" for s in symbols])
    print(f"\n[Shard {shard_id}] Fetching {len(symbols)} symbols: {symbols}...")
    
    query = f"""
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM '{shard_url}'
        WHERE symbol IN ({sym_list_str})
        ORDER BY symbol, timestamp
    """
    
    df_raw = con.execute(query).df()
    print(f"  -> Downloaded {len(df_raw)} minute rows from Shard {shard_id}.")
    
    df_raw['datetime'] = pd.to_datetime(df_raw['timestamp']).dt.tz_localize(None)
    
    for sym in symbols:
        sym_df = df_raw[df_raw['symbol'] == sym].sort_values('datetime').set_index('datetime')
        if len(sym_df) == 0:
            print(f"  [WARNING] No data found for {sym}")
            continue
            
        # Resample to 15m
        sym_15m = sym_df[['open', 'high', 'low', 'close', 'volume']].resample(
            '15min', origin='start_day', offset='15min'
        ).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Filter for market hours (09:15 to 15:15)
        sym_15m = sym_15m.between_time('09:15', '15:15')
        
        total_15m_bars = len(sym_15m)
        earliest_dt = sym_15m.index.min()
        latest_dt = sym_15m.index.max()
        print(f"  -> {sym}: {total_15m_bars} 15m bars ({earliest_dt.strftime('%Y-%m-%d')} to {latest_dt.strftime('%Y-%m-%d')})")
        
        # Split into the 4 folders
        for folder_name, (start_dt, end_dt) in FOLDERS.items():
            sub_df = sym_15m.loc[start_dt:end_dt].copy()
            # If end_dt is 2023-01-01, exclude exact 2023-01-01 if needed
            sub_df = sub_df[sub_df.index < end_dt]
            
            bars_count = len(sub_df)
            trading_days = len(set(sub_df.index.date))
            
            csv_path = os.path.join(BASE_DATA_DIR, folder_name, f"{sym}.csv")
            sub_df.to_csv(csv_path)
            
            # Store both with and without .NS for seamless strategy compatibility
            universe_by_folder[folder_name][sym] = sub_df
            universe_by_folder[folder_name][f"{sym}.NS"] = sub_df
            
            audit_stats.append({
                "Folder": folder_name,
                "Symbol": sym,
                "Trading_Days": trading_days,
                "Bars_15m": bars_count,
                "Min_Date": str(sub_df.index.min().date()) if bars_count > 0 else "N/A",
                "Max_Date": str(sub_df.index.max().date()) if bars_count > 0 else "N/A"
            })

# Save universe_14_15m.pkl in each folder
print("\nSaving universe_14_15m.pkl in each folder...")
for folder_name, data_dict in universe_by_folder.items():
    pkl_path = os.path.join(BASE_DATA_DIR, folder_name, "universe_14_15m.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"  -> Saved {pkl_path} with {len(data_dict)//2} unique stocks.")

# Save audit report
audit_df = pd.DataFrame(audit_stats)
audit_csv_path = os.path.join(BASE_DATA_DIR, "universe_4_years_audit_report.csv")
audit_df.to_csv(audit_csv_path, index=False)
print(f"\nAudit report saved to: {audit_csv_path}")

print("\n=== SUMMARY PER FOLDER ===")
summary = audit_df.groupby('Folder').agg({
    'Bars_15m': ['count', 'min', 'max', 'sum'],
    'Trading_Days': ['min', 'max']
})
print(summary)
