import os

KAGGLE_DIR = r"C:\Users\admin\.cache\kagglehub\datasets\yeashusemwal\nse-stocks-15min-data-from-2019-03-01\versions\5"
files = [f.upper().replace(".CSV", "") for f in os.listdir(KAGGLE_DIR) if f.endswith(".csv")]

targets = {
    'ELECTCAST': ['ELECT', 'CAST'],
    'IFCI': ['IFCI'],
    'IRCON': ['IRCON'],
    'J&KBANK': ['JK', 'JAMMU', 'BANK'],
    'MANALIPETC': ['MANALI', 'PETC'],
    'MTNL': ['MTNL', 'MAHA'],
    'PCJEWELLER': ['PC', 'JEWEL'],
    'RPOWER': ['RPOWER', 'REL', 'POWER'],
    'TEXRAIL': ['TEX', 'RAIL'],
}

print(f"Total files in Kaggle dataset: {len(files)}")
for sym, keywords in targets.items():
    matches = [f for f in files if any(k in f for k in keywords)]
    print(f"\nTarget {sym} (keywords: {keywords}):")
    print("  Matches:", matches[:10])
