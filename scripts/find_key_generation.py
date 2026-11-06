import re

with open("lse_bundle.js", "r", encoding="utf-8") as f:
    js = f.read()

# Search for key generation or api-key endpoints
matches = re.findall(r'["\']([^"\']*(?:api-key|generate|create-key|auth/key)[^"\']*)["\']', js)
print("Key endpoints found:", set(matches))

# Search for /data/api-keys component or routing
idx = js.find("/data/api-keys")
if idx != -1:
    print("\n=== /data/api-keys snippet ===")
    print(js[max(0, idx - 200): min(len(js), idx + 800)])
