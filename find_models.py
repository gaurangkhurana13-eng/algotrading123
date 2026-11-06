import urllib.request
import json
import ssl

URL = "https://openrouter.ai/api/v1/models"
req = urllib.request.Request(URL, headers={"HTTP-Referer": "https://antigravity.google.com"})
context = ssl.create_default_context()

try:
    with urllib.request.urlopen(req, context=context, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))
        models = data.get("data", [])
        print(f"Total models fetched: {len(models)}")
        
        free_models = []
        deepseek_models = []
        for m in models:
            mid = m.get("id", "")
            pricing = m.get("pricing", {})
            prompt_cost = float(pricing.get("prompt", "0") or "0")
            completion_cost = float(pricing.get("completion", "0") or "0")
            
            if prompt_cost == 0 and completion_cost == 0:
                free_models.append(mid)
            if "deepseek" in mid.lower():
                deepseek_models.append((mid, prompt_cost, completion_cost))
                
        print("\n=== TOP FREE MODELS AVAILABLE NOW ===")
        for fm in free_models[:20]:
            print(f"- {fm}")
            
        print("\n=== DEEPSEEK MODELS AVAILABLE ===")
        for dm, p, c in deepseek_models[:15]:
            print(f"- {dm} (Prompt: ${p*1e6:.2f}/M, Completion: ${c*1e6:.2f}/M)")

except Exception as ex:
    print(f"Error: {ex}")
