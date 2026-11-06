from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq

fs = HfFileSystem()
info = fs.info("datasets/xxparthparekhxx/indian-stock-market-minute-data/minute/train-00000.parquet")
print("Size of train-00000.parquet:", round(info['size'] / (1024*1024), 2), "MB")

total_size = sum(fs.info(f"datasets/xxparthparekhxx/indian-stock-market-minute-data/minute/train-{i:05d}.parquet")['size'] for i in range(8))
print("Total size of all 8 files:", round(total_size / (1024*1024*1024), 2), "GB")
