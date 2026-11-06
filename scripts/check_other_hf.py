from huggingface_hub import HfApi

api = HfApi()

repos = [
    "mveen3/Six_Year_Indian_Stock_Market_Dataset-News_and_Ticker",
    "Saintforest/indian-stock-market-minute-data",
    "GalacticWanderer/indian-stock-market-minute-data",
    "Jhanvi-29112000/indian-stock-market-minute-data",
]

for r in repos:
    print(f"\n--- Checking {r} ---")
    try:
        files = api.list_repo_files(repo_id=r, repo_type="dataset")
        print(f"Files count: {len(files)}")
        for f in files[:10]:
            print(f"  {f}")
    except Exception as e:
        print(f"Error {r}: {e}")
