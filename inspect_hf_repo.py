from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_files(repo_id="xxparthparekhxx/indian-stock-market-minute-data", repo_type="dataset")
print(f"Total files in repo: {len(files)}")
for f in files:
    print(f)
