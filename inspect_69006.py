with open("_app.js", "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

idx = content.find("69006:function")
chunk = content[idx:idx+40000]

# Find definition of F, cR, etc.
import re
print("Searching for F, cR, api_calling...")
# Look for var F = ... or const F = ... or F = ...
matches = re.findall(r'\bF\s*=\s*["\'][^"\']+["\']', chunk)
print("F matches:", matches)

matches_cR = re.findall(r'\bcR:function\(\)\{return\s*([^}]+)\}', chunk)
print("cR export:", matches_cR)

# Search all string literals inside chunk
urls = re.findall(r'["\'](https?://[^"\']+|/[^"\']+)["\']', chunk)
print("Endpoints inside 69006:", set(urls))
