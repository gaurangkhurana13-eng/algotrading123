import duckdb

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
url = "https://huggingface.co/datasets/xxparthparekhxx/indian-stock-market-minute-data/resolve/main/minute/train-00005.parquet"
try:
    res = con.execute(f"SELECT min(timestamp), max(timestamp) FROM '{url}'").fetchall()
    print("Shard 5 min/max timestamp:", res)
except Exception as e:
    print("Error:", e)
