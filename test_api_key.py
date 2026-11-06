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

payload = {
    "model": "deepseek/deepseek-r1:free",
    "messages": [
        {"role": "user", "content": "Respond with only one word: READY"}
    ],
    "temperature": 0.1
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(URL, data=data, headers=headers)

context = ssl.create_default_context()

try:
    print("Testing OpenRouter API with model deepseek/deepseek-r1:free...")
    with urllib.request.urlopen(req, context=context, timeout=30) as response:
        res = json.loads(response.read().decode("utf-8"))
        print("STATUS: SUCCESS ✅")
        print("Response:", res.get("choices", [{}])[0].get("message", {}).get("content"))
except urllib.error.HTTPError as e:
    err_body = e.read().decode("utf-8")
    print(f"HTTP Error {e.code}: {err_body}")
except Exception as ex:
    print(f"Error: {ex}")
