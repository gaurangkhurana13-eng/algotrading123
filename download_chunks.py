import urllib.request
import re
import os

base_url = 'https://dhan.co'
js_files = [
    '/_next/static/chunks/pages/calculators/brokerage-calculator-84e252e92a7f7dbf.js',
    '/_next/static/chunks/4288-54d201d7d8f0f112.js',
    '/_next/static/chunks/4880-41211650d4629467.js',
    '/_next/static/chunks/3122-c596326344926e72.js'
]

for js_path in js_files:
    full_url = base_url + js_path
    filename = os.path.basename(js_path)
    print(f"Fetching {full_url}...")
    req = urllib.request.Request(full_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    try:
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode('utf-8', errors='ignore')
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Saved {filename} (len: {len(content)})")
    except Exception as e:
        print(f"Failed {filename}: {e}")
