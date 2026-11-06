import urllib.request
import json
import ssl
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

API_KEY = "os.getenv("OPENROUTER_API_KEY", "DUMMY_KEY")"
URL = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://antigravity.google.com",
    "X-Title": "Bhavishya Autonomous Quant Engine"
}

prompt = """You are an institutional quant engineer for Indian Equities (NSE 15m intraday, Dhan MIS 5x).
Generate Bhavishya V27 quantitative strategy adhering strictly to Gate V2 rules.

CORE REQUIREMENTS:
1. Bi-directional: Both LONG and SHORT trades allowed (Dhan MIS 5x leverage).
2. Universe: Top 30 High-Beta Momentum Midcaps (J&KBANK, UTKARSHBNK, MANALIPETC, IDEA, PCJEWELLER, NIACL, MTNL, IFCI, TEXRAIL, SWSOLAR, etc.).
3. Regime Gate: Session VWAP + EMA50/EMA110 trend alignment + ER10 >= 0.35 + ADX10 >= 20.
4. Risk: 1.5% max risk per trade, Dhan MIS 5x statutory friction model.
5. Files to produce: trades_intraday.csv, trades_swing.csv, results.txt.

STRICT INSTRUCTION:
Do NOT output verbose reasoning or conversational fluff.
Provide the complete executable python script in ```python block for run_v27.py.
Provide the complete v27.pine in ```pinescript block.
"""

payload = {
    "model": "openrouter/free",
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.2,
    "max_tokens": 3500
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(URL, data=data, headers=headers)
context = ssl.create_default_context()

try:
    print("Testing compact direct code generation...")
    with urllib.request.urlopen(req, context=context, timeout=120) as response:
        res = json.loads(response.read().decode("utf-8"))
        choice = res.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "")
        reasoning = msg.get("reasoning", "")
        print(f"Content length: {len(content)}")
        print(f"Reasoning length: {len(reasoning) if reasoning else 0}")
        print(f"Finish reason: {choice.get('finish_reason')}")
        print("First 300 chars of content:\n", content[:300])
except Exception as ex:
    print(f"Error: {ex}")
