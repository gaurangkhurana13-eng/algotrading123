import urllib.request
import json
import ssl

API_KEY = "os.getenv("OPENROUTER_API_KEY", "DUMMY_KEY")"
URL = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://antigravity.google.com",
    "X-Title": "Bhavishya Autonomous Quant Engine"
}

prompt = "def calculate_sma(close):"

payload = {
    "model": "cohere/north-mini-code:free",
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.1,
    "max_tokens": 500
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(URL, data=data, headers=headers)
context = ssl.create_default_context()

try:
    with urllib.request.urlopen(req, context=context, timeout=30) as response:
        res = json.loads(response.read().decode("utf-8"))
        print(json.dumps(res, indent=2))
except Exception as ex:
    print(f"Error: {ex}")
