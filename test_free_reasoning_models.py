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
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "thinkingmachines/inkling:free",
    "recursive/flint-v1.0:free"
]

context = ssl.create_default_context()

for model in candidates:
    print(f"Testing reasoning model: {model}...")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Respond with ONLY: REASONING_READY"}],
        "max_tokens": 15
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(URL, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, context=context, timeout=25) as response:
            res = json.loads(response.read().decode("utf-8"))
            print(f"[ACTIVE ✅] Model {model} is LIVE and responding!")
            break
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")
        print(f"Failed {model} ({e.code}): {err[:120]}")
    except Exception as ex:
        print(f"Error {model}: {ex}")
