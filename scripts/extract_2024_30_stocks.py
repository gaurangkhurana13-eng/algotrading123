import os
import duckdb
import pandas as pd
import pickle
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
OUT_DIR = ROOT_DIR / "data" / "2024-25"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHARD_MAPPING = {
    1: ["CENTRALBK", "DELTACORP", "ELECTCAST"],
    2: ["GSFC", "IOB", "IRCON", "IFCI", "IDEA"],
    3: ["J&KBANK", "MANALIPETC", "JYOTHYLAB"],
    4: ["PCJEWELLER", "NMDC", "NIACL", "NBCC", "MTNL", "NCC"],
    5: ["RITES", "RENUKA", "RCF", "RPOWER", "PNB", "RBLBANK"],
    6: ["TTML", "SUZLON", "TATASTEEL", "SWSOLAR", "TEXRAIL", "UTKARSHBNK"],
    7: ["ZEEL"]
}

base_url = "https://huggingface.co/datasets/xxparthparekhxx/indian-stock-market-minute-data/resolve/main/minute/"

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")

universe_30_2024 = {}
audit_list = []

print("=== STARTING 2024 FULL YEAR EXTRACTION FOR ALL 30 CORE STOCKS ===")

for shard_id, symbols in SHARD_MAPPING.items():
    shard_url = f"{base_url}train-0000{shard_id}.parquet"
    sym_str = ",".join([f"'{s}'" for s in symbols])
    print(f"\n[Shard {shard_id}] Fetching 2024 data for: {symbols}...")
    
    query = f"""
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM '{shard_url}'
        WHERE symbol IN ({sym_str})
          AND timestamp >= '2024-01-01'
          AND timestamp < '2025-01-01'
        ORDER BY symbol, timestamp
    """
    try:
        df_raw = con.execute(query).df()
        print(f"  -> Downloaded {len(df_raw)} minute rows from Shard {shard_id}.")
        
        if len(df_raw) == 0:
            continue
            
        df_raw['datetime'] = pd.to_datetime(df_raw['timestamp']).dt.tz_localize(None)
        
        for sym in symbols:
            sym_df = df_raw[df_raw['symbol'] == sym].sort_values('datetime').set_index('datetime')
            if len(sym_df) == 0:
                print(f"  [NOTE] No 2024 data for {sym}")
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
            
            sym_15m = sym_15m.between_time('09:15', '15:15')
            
            # Save individual CSV
            csv_path = OUT_DIR / f"{sym}.csv"
            sym_15m.to_csv(csv_path)
            
            # Save individual PKL for Gate 1 Check 2 (<symbol>_15m_1y.pkl)
            pkl_path = OUT_DIR / f"{sym.lower()}_15m_1y.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(sym_15m, f)
                
            universe_30_2024[sym] = sym_15m
            universe_30_2024[f"{sym}.NS"] = sym_15m
            
            days = len(set(sym_15m.index.date))
            bars = len(sym_15m)
            print(f"  -> Processed {sym:12s}: {bars:5d} bars | {days:3d} days | ({sym_15m.index.min().strftime('%Y-%m-%d')} to {sym_15m.index.max().strftime('%Y-%m-%d')})")
            audit_list.append({'symbol': sym, 'bars': bars, 'days': days})
    except Exception as e:
        print(f"  [ERROR] on Shard {shard_id}: {e}")

out_file = OUT_DIR / "universe_30_15m.pkl"
with open(out_file, "wb") as f:
    pickle.dump(universe_30_2024, f)

print(f"\n[DONE] Successfully saved {len(universe_30_2024)//2} unique stocks to {out_file}!")
