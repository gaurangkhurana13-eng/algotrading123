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
MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

prompt = """You are a Senior Quantitative Microstructure Researcher for Indian Equities (NSE 15m intraday, Dhan MIS 5x leverage, ₹500 starting wallet).
We need your rigorous mathematical formulas and exact parameters for Bhavishya V27.

HISTORICAL CONTEXT FROM BHAVISHYA V26:
- 2026 Modern Horizon: 83 trades, 59% Win Rate, Net +₹3,327 (7.6x profit, Max DD 13.69%).
- 2025 Chop Horizon: 56 trades, 51.8% Win Rate, Gross +₹107.45, Net +₹13.89. Capital was 100% preserved, but Dhan statutory round-trip friction (~₹2.47/trade) ate 87% of gross profit.
- V26 was 100% Long-Only, so it suffered in bear chop.

DELIVER YOUR CONCRETE MATHEMATICAL SPECIFICATIONS FOR BHAVISHYA V27:

1. SHORT BREAKDOWN IGNITION (MIS 5X):
   - What are the exact mathematical causal indicator conditions (EMA 9/20/50/110, VWAP shift(1), DMI/ADX, Kaufman ER10) to trigger high-conviction Short entries?
   - How to filter out false bear traps on 15m bars?

2. VOLATILITY DECOUPLING (TARGETING >= 1.5% MOVES):
   - What exact formula ensures we only enter trades when expected expansion is >= 1.5% so Dhan statutory friction (<0.20%) is less than 10% of gross gain?
   - Specify ATR% threshold, RVOL formula, and candle body/range ratio.

3. DYNAMIC REGIME GOVERNOR (BULL / BEAR / CHOP):
   - What exact mathematical gate distinguishes Bull (Longs enabled), Bear (Shorts enabled), and Sideways Chop (Stay in 100% Cash)?

4. ASYMMETRIC EXIT LADDER & MICROSTRUCTURE TRAILING RATCHET:
   - Provide the exact causal formula for initial stop loss and trailing ratchet step logic (e.g. at +1.0R, +1.5R, +2.0R) without same-bar lookahead bias.

Output exact formulas, variable names, and numerical parameters immediately without boilerplate.
"""

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://antigravity.google.com",
    "X-Title": "Bhavishya Autonomous Quant Engine"
}

payload = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.2,
    "max_tokens": 3000
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(URL, data=data, headers=headers)
context = ssl.create_default_context()

print(f"Connecting to reasoning model ({MODEL})...")
print("Sending clean targeted quant specification prompt...")

try:
    with urllib.request.urlopen(req, context=context, timeout=180) as response:
        res = json.loads(response.read().decode("utf-8"))
        choice = res.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or ""
        
        full_output = ""
        if content:
            full_output += f"### ARCHITECTURAL SPECIFICATIONS:\n\n{content}\n\n---\n\n"
        if reasoning:
            full_output += f"### INTERNAL REASONING DERIVATION:\n\n{reasoning}\n"
            
        out_dir = ROOT_DIR / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "clean_v27_quant_specifications.md"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(full_output)
            
        print(f"[SUCCESS] Clean Quant Specs received!")
        print(f"Saved to {out_file}")
        print(f"Content length: {len(content)}, Reasoning length: {len(reasoning)}")
        
except Exception as ex:
    print(f"Error: {ex}")
