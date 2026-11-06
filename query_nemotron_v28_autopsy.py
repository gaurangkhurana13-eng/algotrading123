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

prompt = """You are a Senior Quantitative Microstructure Researcher for Indian Equities.
We implemented your Bhavishya V27 recommendations, and we now bring you the BRUTAL FORENSIC POST-MORTEM from real backtest data (NSE 15m, Dhan broker, MIS 5x leverage, starting wallet Rs 500).

======================================================================
1. THE V27 CATASTROPHE AUTOPSY (WHAT YOUR SHORTING LOGIC DID):
======================================================================
Overall V27 Performance:
- Trades: 339 trades (massive over-trading!)
- Net PnL: -Rs 289.37 (ACCOUNT CRASHED from Rs 500 to Rs 210.63, Max DD 70.19%!)
- Dhan Friction: Rs 413.54 eaten!

FORENSIC BREAKDOWN (LONG VS SHORT):
- DIRECTION: LONG
  Total Trades: 212 | Win Rate: 48.11% | Gross PnL: +Rs 233.30 | Net PnL: -Rs 24.15
- DIRECTION: SHORT (YOUR RECOMMENDED LOGIC)
  Total Trades: 127 | Win Rate: 45.67% | Gross PnL: -Rs 109.13 | Dhan Friction: Rs 156.09 | NET LOSS: -Rs 265.22!

EMPIRICAL LESSON:
In Indian Midcaps (PC Jeweller, Suzlon, IDEA, Manali Petro, IFCI), shorting cash equity intraday on 15m is TOXIC. Stocks have a structural upward drift and suffer violent bear-market short-covering spikes ("bear traps"). Every short trade triggered false breakdowns and bled capital.

======================================================================
2. CONTRAST WITH BHAVISHYA V26 (THE CERTIFIED KING):
======================================================================
Bhavishya V26 is 100% LONG-ONLY and passed Gate V2 with EXIT CODE: 0.
- 2026 Modern Horizon: 83 trades, 59.04% Win Rate, Net +Rs 3,327.88 (7.66x profit, PF 2.84, Max DD 13.69%, Final Wallet Rs 3,827.88).
- Unseen 2025 Monsoon Chop H2 (Jun-Dec 2025): 26 trades, 53.85% Win Rate, Net +Rs 32.24 (GREEN!), Max DD 10.10%.
- Full Continuous Year 2025: 56 trades, 51.79% Win Rate, Net +Rs 13.89 (GREEN, 100% capital preserved).

V26's CORE ARCHITECTURE:
- 100% Long-Only.
- Relative Excess Momentum (G1 & G2): Compares each stock's 1-hour return against the median benchmark of the top 30 midcaps. Only buys when the stock is outperforming the market.
- Single-Position Focus: Exactly 1 trade at a time. Zero margin blowout.
- Extended Climax Ladder: At MFE >= 0.85 ATR, stop moves to Entry + 0.40 ATR (locking profit). Tightens at 1.15 ATR, 2.0 ATR, 3.0 ATR.
- Stagnation Scratch: Exits at bar 4 if MFE < 0.40 ATR and RSI < 46.
- 14:45 EOD Square-off.

======================================================================
3. ARCHITECTURAL MANDATE FOR BHAVISHYA V28:
======================================================================
We DO NOT WANT SHORTING. Shorting is permanently rejected for Indian cash midcaps.

DeepThink on how to enhance Bhavishya V26 into BHAVISHYA V28:
1. HOW TO INCREASE 2025 CHOP GAINS WITHOUT TAKING MORE TRADES:
   In 2025, V26 made +Rs 107.45 Gross, but Dhan friction took Rs 93.56 (leaving +Rs 13.89 net).
   How can we mathematically filter out the 10 lowest-quality trades in 2025 while letting the winning runners expand further, so Net Cash in chop jumps from +Rs 13 to +Rs 150+?
2. DYNAMIC VOLATILITY SCALING:
   How should position sizing or ATR targets scale when market breadth is low vs high?
3. RUNNER EXPANSION:
   In 2026, V26 made +Rs 3,327. How can the Climax Ladder be mathematically calibrated to capture 4.0 ATR+ macro runners without getting prematurely shaken out on minor wicks?

Deliver exact mathematical formulas, threshold values, and concrete indicator rules for BHAVISHYA V28. Be concise, direct, and rigorous.
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
    "max_tokens": 3500
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(URL, data=data, headers=headers)
context = ssl.create_default_context()

print(f"Connecting to reasoning model ({MODEL})...")
print("Sending V27 Autopsy and V28 Blueprint Mandate...")

try:
    with urllib.request.urlopen(req, context=context, timeout=180) as response:
        res = json.loads(response.read().decode("utf-8"))
        choice = res.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or ""
        
        full_output = ""
        if content:
            full_output += f"### BHAVISHYA V28 ARCHITECTURAL SPECIFICATIONS:\n\n{content}\n\n---\n\n"
        if reasoning:
            full_output += f"### REASONING & DERIVATION:\n\n{reasoning}\n"
            
        out_dir = ROOT_DIR / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "bhavishya_v28_autopsy_and_blueprint.md"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(full_output)
            
        print(f"[SUCCESS] V28 Quant Blueprint received from reasoning engine!")
        print(f"Saved to {out_file}")
        print(f"Content length: {len(content)}, Reasoning length: {len(reasoning)}")
        
except Exception as ex:
    print(f"Error: {ex}")
