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

candidates = [
    "deepseek/deepseek-chat",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-chat:free",
    "deepseek/deepseek-r1:free"
]

context = ssl.create_default_context()

for model in candidates:
    print(f"Testing model: {model}...")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Ping. Respond with READY."}],
        "max_tokens": 10
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(URL, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, context=context, timeout=20) as response:
            res = json.loads(response.read().decode("utf-8"))
            print(f"[SUCCESS ✅] Model {model} is ACTIVE!")
            break
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")
        print(f"Failed {model} ({e.code}): {err[:150]}")
    except Exception as ex:
        print(f"Error {model}: {ex}")
