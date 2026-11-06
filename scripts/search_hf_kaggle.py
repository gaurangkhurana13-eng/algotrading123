import requests
import json

print("--- Searching Hugging Face Datasets ---")
searches = ["nifty", "nse", "indian stock", "nse minute"]
for q in searches:
    try:
        url = f"https://huggingface.co/api/datasets?search={requests.utils.quote(q)}&limit=10"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            items = r.json()
            print(f"Query '{q}': found {len(items)}")
            for item in items[:5]:
                print(f"  - {item['id']}")
    except Exception as e:
        print(f"Error for '{q}': {e}")

print("\n--- Inspecting KaggleHub capabilities ---")
try:
    import kagglehub
    print("kagglehub version:", kagglehub.__version__)
    print("kagglehub attributes:", dir(kagglehub))
except Exception as e:
    print("kagglehub error:", e)
