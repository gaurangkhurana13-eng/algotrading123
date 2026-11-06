import re
import json

with open("lse_bundle.js", "r", encoding="utf-8") as f:
    js = f.read()

print("=== DEEP CODEBASE SCAN OF LSE BUNDLE ===")

# 1. Search for data-api calls
matches = re.findall(r'["\'](https://data-api\.londonstrategicedge\.com[^"\']*)["\']', js)
print(f"data-api URLs found: {len(matches)}")
for m in set(matches):
    print("  ->", m)

# 2. Search for all API routes with path templates
api_calls = re.findall(r'`https://data-api\.londonstrategicedge\.com([^`]+)`', js)
print(f"\nTemplate URLs on data-api: {len(api_calls)}")
for a in set(api_calls):
    print("  ->", a)

# 3. Check covered markets / asset classes / exchanges
exchanges = re.findall(r'\b(NSE|BSE|NASDAQ|NYSE|LSE|FOREX|CRYPTO|CME|BINANCE|CBOE)\b', js)
print(f"\nExchange mentions: {set(exchanges)}")

# 4. Check for headers/auth
headers_match = re.findall(r'["\']([xX]-[a-zA-Z0-9\-_]+)["\']', js)
print(f"\nCustom Headers found: {set(headers_match)}")

# 5. Extract text snippets around 'data-api' or 'api.londonstrategicedge.com'
print("\n--- Snippets around data-api ---")
for m in re.finditer(r'data-api\.londonstrategicedge\.com', js):
    start = max(0, m.start() - 200)
    end = min(len(js), m.end() + 200)
    print("Snippet:\n", js[start:end])
    print("-" * 50)
