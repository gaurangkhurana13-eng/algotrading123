import duckdb
import pandas as pd

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")

symbols_target = [
    "JKBANK", "J&KBANK", "UTKARSHBNK", "MANALIPETC", "IDEA", "PCJEWELLER",
    "NIACL", "MTNL", "IFCI", "TEXRAIL", "SWSOLAR",
    "TTML", "RENUKA", "RCF", "NBCC", "JYOTHYLAB",
    "RBLBANK", "ELECTCAST", "PNB", "GSFC", "TATASTEEL",
    "SUZLON", "CENTRALBK", "IOB", "IRCON", "NMDC",
    "RITES", "NCC", "ZEEL", "DELTACORP", "RPOWER"
]

base_url = "https://huggingface.co/datasets/xxparthparekhxx/indian-stock-market-minute-data/resolve/main/minute/"
found_map = {}

print("Scanning shards 0 to 9 on HuggingFace...")
for i in range(10):
    shard_url = f"{base_url}train-0000{i}.parquet"
    sym_str = ",".join([f"'{s}'" for s in symbols_target])
    q = f"""
        SELECT symbol, MIN(timestamp) as min_ts, MAX(timestamp) as max_ts, COUNT(*) as rows
        FROM '{shard_url}'
        WHERE symbol IN ({sym_str})
        GROUP BY symbol
    """
    try:
        df = con.execute(q).df()
        for idx, row in df.iterrows():
            sym = row["symbol"]
            if sym not in found_map:
                found_map[sym] = (i, str(row["min_ts"])[:10], str(row["max_ts"])[:10], int(row["rows"]))
                print(f"Found {sym:<12} in Shard {i}: {row['min_ts']} to {row['max_ts']} ({row['rows']} rows)")
    except Exception as e:
        print(f"Shard {i}: {e}")

print(f"\nTotal Found: {len(found_map)} out of {len(symbols_target)}")
missing = [s for s in symbols_target if s not in found_map and s not in ["JKBANK", "J&KBANK"]]
print("Missing:", missing)
