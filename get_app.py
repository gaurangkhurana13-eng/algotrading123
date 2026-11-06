import urllib.request
import re

url = 'https://dhan.co/_next/static/chunks/pages/_app-7a6e91a3944e58d4.js'
print("Fetching _app...")
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as resp:
        content = resp.read().decode('utf-8', errors='ignore')
        with open("_app.js", "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved _app.js ({len(content)})")
        if "69006" in content:
            print("Found 69006 in _app.js!")
            # extract where 69006 is
            idx = content.find("69006:")
            if idx == -1:
                idx = content.find("69006 :")
            if idx != -1:
                print("Context:", content[idx:idx+800])
except Exception as e:
    print("Error:", e)
