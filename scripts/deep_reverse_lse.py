import requests
import re
import json

base_url = "https://www.londonstrategicedge.com"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 1. Inspect api-documentation
print("--- Fetching /api-documentation ---")
try:
    r_api = requests.get(f"{base_url}/api-documentation", headers=headers, timeout=10)
    print("API Doc status:", r_api.status_code)
    print("API Doc length:", len(r_api.text))
    with open("lse_api_doc.html", "w", encoding="utf-8") as f:
        f.write(r_api.text)
except Exception as e:
    print("Error fetching api docs:", e)

# 2. Fetch and reverse engineer JS bundle
print("\n--- Inspecting JS bundle /assets/index-B0teSxym.js ---")
js_url = f"{base_url}/assets/index-B0teSxym.js"
try:
    r_js = requests.get(js_url, headers=headers, timeout=20)
    print("JS bundle status:", r_js.status_code)
    print("JS bundle size:", len(r_js.text))
    with open("lse_bundle.js", "w", encoding="utf-8") as f:
        f.write(r_js.text)
        
    js_text = r_js.text
    
    # Search for API endpoints, URLs, domains
    urls = set(re.findall(r'https?://[a-zA-Z0-9\.\-_/]+', js_text))
    print(f"\nUnique external/internal URLs found in bundle ({len(urls)}):")
    for u in sorted(urls):
        if any(k in u for k in ['api', 'data', 'london', 'lse', 'aws', 'cdn', 'storage', 'backend']):
            print("  ->", u)
            
    # Search for endpoint paths like "/api/v1/..." or "/v1/..."
    endpoints = set(re.findall(r'["\'](/api/[a-zA-Z0-9\-_/]+)["\']', js_text))
    endpoints.update(re.findall(r'["\'](/v[0-9]/[a-zA-Z0-9\-_/]+)["\']', js_text))
    endpoints.update(re.findall(r'["\'](/data/[a-zA-Z0-9\-_/]+)["\']', js_text))
    print(f"\nAPI Endpoints found in bundle ({len(endpoints)}):")
    for ep in sorted(endpoints):
        print("  ->", ep)

except Exception as e:
    print("Error analyzing JS bundle:", e)
