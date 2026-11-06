import requests
import re
import json

base_url = "https://www.londonstrategicedge.com/assets"
headers = {"User-Agent": "Mozilla/5.0"}

chunks_to_fetch = [
    "Databank3-C8r139Eb.js",
    "ApiDocumentation-BnzHNOsH.js",
    "LoadInPython-7yza-U1U.js",
    "TickData-CG-5OcR5.js",
    "WebSocketDocumentation-9ZFgR8V2.js"
]

print("=== FETCHING AND DISSECTING CRITICAL LSE CHUNKS ===")
for c in chunks_to_fetch:
    url = f"{base_url}/{c}"
    r = requests.get(url, headers=headers, timeout=15)
    print(f"\n--- {c} (Status {r.status_code}, Size {len(r.text)}) ---")
    if r.status_code == 200:
        with open(f"lse_{c}", "w", encoding="utf-8") as f:
            f.write(r.text)
            
        # Extract endpoints, sample code, python snippets, data schemas
        py_snippets = re.findall(r'import\s+[a-zA-Z0-9_\.]+|pip\s+install\s+[a-zA-Z0-9_\-]+', r.text)
        print("  Python/Package references:", set(py_snippets))
        
        # Extract URLs
        urls = set(re.findall(r'https?://[a-zA-Z0-9\.\-_/]+', r.text))
        print("  URLs found:", urls)
        
        # Look for API paths
        paths = set(re.findall(r'["\'](/v[0-9]/[a-zA-Z0-9\-_/]+|/api/[a-zA-Z0-9\-_/]+)["\']', r.text))
        print("  API Paths:", paths)
