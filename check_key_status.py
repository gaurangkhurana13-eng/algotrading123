import urllib.request
import json
import ssl

API_KEY = "os.getenv("OPENROUTER_API_KEY", "DUMMY_KEY")"
URL = "https://openrouter.ai/api/v1/auth/key"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "HTTP-Referer": "https://antigravity.google.com"
}

req = urllib.request.Request(URL, headers=headers)
context = ssl.create_default_context()

try:
    with urllib.request.urlopen(req, context=context, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))
        print("KEY DETAILS:")
        print(json.dumps(data, indent=2))
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}")
except Exception as ex:
    print(f"Error: {ex}")
