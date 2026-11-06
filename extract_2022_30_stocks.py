import os
import duckdb
import pandas as pd
import pickle
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
OUT_DIR = ROOT_DIR / "data" / "2022-23"
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

universe_30 = {}

print("=== STARTING 2022 EXTRACTION FOR ALL 30 HIGH-BETA LEADERS ===")

for shard_id, symbols in SHARD_MAPPING.items():
    shard_url = f"{base_url}train-0000{shard_id}.parquet"
    sym_str = ",".join([f"'{s}'" for s in symbols])
    print(f"\n[Shard {shard_id}] Downloading 2022 data for: {symbols}...")
    
    query = f"""
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM '{shard_url}'
        WHERE symbol IN ({sym_str})
          AND timestamp >= '2022-01-01'
          AND timestamp < '2023-01-01'
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
                print(f"  [NOTE] No 2022 data for {sym} (likely listed later)")
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
            
            # Map clean symbol & .NS symbol
            universe_30[sym] = sym_15m
            universe_30[f"{sym}.NS"] = sym_15m
            print(f"  -> Processed {sym}: {len(sym_15m)} 15m bars ({sym_15m.index.min()} to {sym_15m.index.max()})")
    except Exception as e:
        print(f"  [ERROR] on Shard {shard_id}: {e}")

out_file = OUT_DIR / "universe_30_15m.pkl"
with open(out_file, "wb") as f:
    pickle.dump(universe_30, f)

print(f"\n[DONE] Successfully saved {len(universe_30)//2} unique stocks to {out_file}!")
