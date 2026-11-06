import urllib.request
import json
import ssl
import sys

API_KEY = "os.getenv("OPENROUTER_API_KEY", "DUMMY_KEY")"
URL = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://antigravity.google.com",
    "X-Title": "Bhavishya Autonomous Quant Engine"
}

prompt = "Write a 5-line python function that calculates SMA of a list."

payload = {
    "model": "openrouter/free",
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.2,
    "max_tokens": 500
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(URL, data=data, headers=headers)
context = ssl.create_default_context()

try:
    with urllib.request.urlopen(req, context=context, timeout=60) as response:
        res = json.loads(response.read().decode("utf-8"))
        with open("res_dump.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print("Dumped raw response to res_dump.json successfully.")
except Exception as ex:
    print(f"Error: {ex}")
