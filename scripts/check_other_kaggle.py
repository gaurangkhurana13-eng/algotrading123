import requests
import json

datasets = [
    "ramamet4/nse-stocks-database",
    "zafaraliii/nse-bse-nifty-sensex-stocks-securities-minute-data",
    "hk7797/stock-market-india"
]

headers = {'User-Agent': 'Mozilla/5.0'}

for ds in datasets:
    print(f"\n--- Checking Kaggle dataset: {ds} ---")
    url = f"https://www.kaggle.com/api/v1/datasets/view/{ds}"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        print("Status:", r.status_code)
        if r.status_code == 200:
            data = r.json()
            print("Title:", data.get("title"))
            print("Description snippet:", data.get("description", "")[:300].replace("\n", " "))
            print("Total bytes:", data.get("totalBytes"))
    except Exception as e:
        print("Error:", e)
