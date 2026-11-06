import re

with open("lse_bundle.js", "r", encoding="utf-8") as f:
    js = f.read()

chunks = set(re.findall(r'["\']\./([a-zA-Z0-9\-_]+\.js)["\']', js))
chunks.update(re.findall(r'["\']([a-zA-Z0-9\-_]+\.js)["\']', js))
print(f"Chunks found ({len(chunks)}):")
for c in sorted(chunks):
    if any(k in c for k in ['data', 'market', 'stock', 'api', 'vendor', 'page', 'trade']):
        print("  ->", c)
