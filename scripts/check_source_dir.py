import sys
from pathlib import Path

SOURCE_DIR = Path(r"C:\Users\admin\.cache\kagglehub\datasets\debashis74017\stock-market-data-nifty-100-stocks-15-min-data\versions\2")

print("Exists:", SOURCE_DIR.exists())
if SOURCE_DIR.exists():
    files = list(SOURCE_DIR.glob("*"))
    print("Total files:", len(files))
    for f in files[:10]:
        print(" ", f.name)
