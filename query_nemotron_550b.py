import urllib.request
import json
import ssl
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
API_KEY = "os.getenv("OPENROUTER_API_KEY", "DUMMY_KEY")"
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

prompt_file = ROOT_DIR / "bhavishya_v27_deepseek_prompt.md"
with open(prompt_file, "r", encoding="utf-8") as f:
    master_prompt = f.read()

system_prompt = (
    "You are an Elite Senior Quantitative Researcher and Microstructure Scientist for Indian Equities. "
    "Do NOT write a 400-line Python backtest script—Antigravity will code the production implementation. "
    "We need your PURE QUANTITATIVE REASONING, MATHEMATICAL FORMULAS, AND ARCHITECTURAL BLUEPRINT for Bhavishya V27: "
    "\n1. INSTITUTIONAL SHORTING TRIGGERS: Exact indicators, thresholds, and false-breakdown filters for intraday MIS 5x shorting on 15m Indian equities. "
    "\n2. VOLATILITY DECOUPLING (1.5%+ MOVE CAPTURE): Exact mathematical conditions (ATR%, RVOL, Kaufman ER, candle range) to guarantee trade expansions exceed statutory friction. "
    "\n3. DYNAMIC REGIME GATE: Concrete mathematical formulation (Session VWAP, EMA slopes, ADX) to switch between Long, Short, and Cash (preserving capital in brutal sideways chop). "
    "\n4. MICROSTRUCTURE TRAILING RATCHET: Exact mathematical logic for trailing stops and risk management without same-bar lookahead bias. "
    "\nBe mathematically rigorous, concrete with formulas, and provide exact parameters."
)

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://antigravity.google.com",
    "X-Title": "Bhavishya Autonomous Quant Engine"
}

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": master_prompt}
    ],
    "temperature": 0.25,
    "max_tokens": 6000
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(URL, data=data, headers=headers)
context = ssl.create_default_context()

print(f"Connecting to 550B Monster Reasoning Model ({MODEL})...")
print("Sending Master Prompt with full historical forensics...")

try:
    with urllib.request.urlopen(req, context=context, timeout=240) as response:
        res = json.loads(response.read().decode("utf-8"))
        choice = res.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or ""
        
        full_output = ""
        if reasoning:
            full_output += f"### DEEP REASONING & THOUGHT PROCESS:\n\n{reasoning}\n\n---\n\n"
        full_output += f"### ARCHITECTURAL SUGGESTIONS & MATHEMATICAL BLUEPRINT:\n\n{content}\n"
        
        out_dir = ROOT_DIR / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "nemotron_550b_v27_suggestions.md"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(full_output)
            
        print(f"[SUCCESS] 550B reasoning completed successfully!")
        print(f"Output saved to {out_file}")
        print(f"Tokens used: {res.get('usage', {}).get('total_tokens')}")
        print(f"Content length: {len(content)}, Reasoning length: {len(reasoning)}")
        
except urllib.error.HTTPError as e:
    err = e.read().decode("utf-8")
    print(f"HTTP Error {e.code}: {err}")
except Exception as ex:
    print(f"Error: {ex}")
