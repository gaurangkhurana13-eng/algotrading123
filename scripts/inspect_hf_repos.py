import json
from huggingface_hub import HfApi

api = HfApi()

datasets_to_check = [
    "xxparthparekhxx/indian-stock-market-minute-data",
    "destinybound/NSE-stock-market-historical-data",
    "Tiru8055/NSE_top100_OHLC_1min_tf",
    "neerajnigam6/nifty_stock_data"
]

print("=== CHECKING HUGGINGFACE DATASET FILES ===")
for ds in datasets_to_check:
    print(f"\n--- Checking {ds} ---")
    try:
        files = api.list_repo_files(repo_id=ds, repo_type="dataset")
        print(f"Total files: {len(files)}")
        # Print first 20 files
        for f in files[:25]:
            print(f"  {f}")
        if len(files) > 25:
            print(f"  ... and {len(files)-25} more files.")
            # Check if any of our 30 stocks are in files
            symbols = ['CENTRALBK', 'DELTACORP', 'ELECTCAST', 'GSFC', 'IDEA', 'IFCI', 'IOB', 'IRCON', 
                       'JKBANK', 'JYOTHYLAB', 'MANALIPETC', 'MTNL', 'NBCC', 'NCC', 'NIACL', 'NMDC', 
                       'PCJEWELLER', 'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'RPOWER', 'SUZLON', 
                       'SWSOLAR', 'TATASTEEL', 'TEXRAIL', 'TTML', 'ZEEL']
            matched = [f for f in files if any(s.lower() in f.lower() for s in symbols)]
            print(f"  Matched symbols in repo: {len(matched)} files: {matched[:10]}")
    except Exception as e:
        print(f"Error checking {ds}: {e}")
