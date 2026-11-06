import requests
import re

url = "https://www.londonstrategicedge.com/assets/vendor-auth-hRK_PeSe.js"
headers = {"User-Agent": "Mozilla/5.0"}

r = requests.get(url, headers=headers, timeout=15)
print("vendor-auth size:", len(r.text))

# Search for supabase, firebase, auth0, cognito or api urls
matches = set(re.findall(r'https?://[a-zA-Z0-9\.\-_/]+', r.text))
print("Auth URLs found:")
for m in matches:
    print("  ->", m)

# Search for auth providers
providers = set(re.findall(r'\b(supabase|firebase|auth0|cognito|clerk|google|github)\b', r.text, re.I))
print("Auth providers:", providers)
