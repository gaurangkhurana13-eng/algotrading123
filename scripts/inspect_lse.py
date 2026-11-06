import requests
from bs4 import BeautifulSoup
import json
import re

url = "https://www.londonstrategicedge.com/data"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

try:
    r = requests.get(url, headers=headers, timeout=15)
    print("Status:", r.status_code)
    print("URL:", r.url)
    print("Content length:", len(r.text))
    
    soup = BeautifulSoup(r.text, "html.parser")
    title = soup.find("title")
    print("Title:", title.text if title else "No title")
    
    # Meta tags
    metas = {m.get("name") or m.get("property"): m.get("content") for m in soup.find_all("meta") if m.get("content")}
    print("\nMeta tags:")
    for k, v in list(metas.items())[:10]:
        print(f"  {k}: {v}")
        
    # Check for Next.js or React hydration data
    next_data = soup.find("script", id="__NEXT_DATA__")
    if next_data:
        print("\nFound __NEXT_DATA__!")
        data_json = json.loads(next_data.string)
        print("BuildId:", data_json.get("buildId"))
        print("Page:", data_json.get("page"))
        print("Keys in props:", data_json.get("props", {}).keys())
        with open("lse_next_data.json", "w", encoding="utf-8") as f:
            json.dump(data_json, f, indent=2)
            print("Saved lse_next_data.json")
            
    # Check script tags
    scripts = [s.get("src") for s in soup.find_all("script") if s.get("src")]
    print(f"\nScript sources ({len(scripts)}):")
    for s in scripts[:15]:
        print("  ", s)
        
    # Check links
    links = [a.get("href") for a in soup.find_all("a") if a.get("href")]
    print(f"\nLinks found ({len(links)}):")
    for l in links[:15]:
        print("  ", l)
        
    # Save full HTML for deep analysis
    with open("lse_data.html", "w", encoding="utf-8") as f:
        f.write(r.text)
        print("\nSaved lse_data.html")

except Exception as e:
    print("Error:", e)
