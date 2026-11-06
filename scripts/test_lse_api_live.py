import requests
import json

base = "https://download.londonstrategicedge.com"
endpoints = [
    "/v3/health",
    "/v3/markets",
    "/v3/overview",
    "/v3/catalogue",
    "/v3/factors"
]

print("=== PROBING LIVE LSE API ENDPOINTS ===")
for ep in endpoints:
    url = f"{base}{ep}"
    try:
        r = requests.get(url, timeout=10)
        print(f"\n{ep} -> Status: {r.status_code}")
        print("  Headers:", {k: v for k, v in r.headers.items() if k in ['content-type', 'access-control-allow-origin']})
        if r.status_code == 200:
            try:
                data = r.json()
                print("  JSON Type:", type(data))
                if isinstance(data, list):
                    print(f"  Count: {len(data)}")
                    print("  Sample:", data[:3])
                elif isinstance(data, dict):
                    print("  Keys:", list(data.keys())[:10])
                    print("  Sample:", str(data)[:200])
            except:
                print("  Text:", r.text[:200])
        else:
            print("  Body:", r.text[:200])
    except Exception as e:
        print(f"  Error: {e}")
