import urllib.request
import json
import ssl
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

API_KEY = "os.getenv("OPENROUTER_API_KEY", "DUMMY_KEY")"
URL = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://antigravity.google.com",
    "X-Title": "Bhavishya Autonomous Quant Engine"
}

payload = {
    "model": "openrouter/free",
    "messages": [
        {"role": "user", "content": "Say hello!"}
    ],
    "temperature": 0.2,
    "max_tokens": 50
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(URL, data=data, headers=headers)
context = ssl.create_default_context()

try:
    with urllib.request.urlopen(req, context=context, timeout=30) as response:
        res = json.loads(response.read().decode("utf-8"))
        print("[FULL RESPONSE]:")
        print(json.dumps(res, indent=2))
except Exception as ex:
    print(f"[ERROR]: {ex}")
