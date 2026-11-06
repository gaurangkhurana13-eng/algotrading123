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

# Test 1: Disable reasoning via extra body
payload = {
    "model": "openrouter/free",
    "messages": [
        {"role": "user", "content": "Write a 3-line python function to calculate SMA."}
    ],
    "temperature": 0.1,
    "max_tokens": 500,
    "reasoning": {"max_tokens": 0}
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(URL, data=data, headers=headers)
context = ssl.create_default_context()

try:
    print("Testing payload with reasoning max_tokens=0...")
    with urllib.request.urlopen(req, context=context, timeout=30) as response:
        res = json.loads(response.read().decode("utf-8"))
        msg = res.get("choices", [{}])[0].get("message", {})
        print("Content:", msg.get("content"))
        print("Reasoning:", msg.get("reasoning")[:100] if msg.get("reasoning") else "None")
except Exception as ex:
    print(f"Error test 1: {ex}")

# Test 2: Target google/gemma-4-26b-a4b-it:free
payload2 = {
    "model": "google/gemma-4-26b-a4b-it:free",
    "messages": [
        {"role": "user", "content": "Write a 3-line python function to calculate SMA."}
    ],
    "temperature": 0.1,
    "max_tokens": 500
}

data2 = json.dumps(payload2).encode("utf-8")
req2 = urllib.request.Request(URL, data=data2, headers=headers)

try:
    print("\nTesting google/gemma-4-26b-a4b-it:free...")
    with urllib.request.urlopen(req2, context=context, timeout=30) as response:
        res2 = json.loads(response.read().decode("utf-8"))
        msg2 = res2.get("choices", [{}])[0].get("message", {})
        print("Gemma Content:", msg2.get("content"))
except Exception as ex:
    print(f"Error test 2: {ex}")
