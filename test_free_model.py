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

models_to_test = [
    "openrouter/free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "cohere/north-mini-code:free"
]

context = ssl.create_default_context()

for model in models_to_test:
    print(f"\nTesting model: {model}...")
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Return ONLY the string 'QUANT_READY' and nothing else."}
        ],
        "temperature": 0.1,
        "max_tokens": 50
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(URL, data=data, headers=headers)
    
    try:
        with urllib.request.urlopen(req, context=context, timeout=25) as response:
            res = json.loads(response.read().decode("utf-8"))
            content = res.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            print(f"SUCCESS ✅ -> Model responded: {content}")
            break # Got our working model!
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode('utf-8')[:120]}")
    except Exception as ex:
        print(f"Error: {ex}")
