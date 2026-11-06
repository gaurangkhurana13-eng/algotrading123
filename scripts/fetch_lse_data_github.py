import requests

urls = [
    "https://raw.githubusercontent.com/londonstrategicedge/lse-data/main/README.md",
    "https://raw.githubusercontent.com/londonstrategicedge/lse-data/master/README.md",
    "https://api.github.com/repos/londonstrategicedge/lse-data",
    "https://pypi.org/pypi/lse-data/json"
]

headers = {"User-Agent": "Mozilla/5.0"}

print("=== CHECKING LSE-DATA ON GITHUB AND PYPI ===")
for u in urls:
    try:
        r = requests.get(u, headers=headers, timeout=10)
        print(f"\n{u} -> Status: {r.status_code}")
        if r.status_code == 200:
            if "json" in u:
                data = r.json()
                if "info" in data:
                    print("  Package summary:", data["info"].get("summary"))
                    print("  Version:", data["info"].get("version"))
                    print("  Project URLs:", data["info"].get("project_urls"))
                elif "description" in data:
                    print("  Repo description:", data.get("description"))
                    print("  Stars:", data.get("stargazers_count"))
            else:
                print("  Snippet:\n", r.text[:1200])
    except Exception as e:
        print("  Error:", e)
