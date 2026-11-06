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

prompt = """You are our Senior Quantitative Microstructure Researcher for Indian Equities.
You have been our lead quant brain across V26, V27, and V28. DO NOT forget your history and DO NOT hallucinate a completely new strategy.
Here is the exact ground reality of what we just tested and discovered together:

========================================================================================
1. THE CONTINUOUS EMPIRICAL AUDIT TRAIL (V26 -> V27 -> V28):
========================================================================================

A. BHAVISHYA V26 (THE COMPOUNDING BEAST):
   - 2026 Horizon: 83 trades, 59.04% Win Rate, Net +Rs 3,327.88 (7.66x profit, Final Wallet Rs 3,827.88). Gate V2 Exit Code 0 certified!
   - 2025 Full Year: 56 trades, 51.79% Win Rate, Net +Rs 13.89 (Capital preserved, but Dhan friction ate Rs 93.56 of the Rs 107 gross gain).
   - Core Engine: 100% Long-only, G1 & G2 Relative Excess Momentum against 30-stock median, Single-position focus, Climax Ratchet ladder, Streak scaling (alpha 0.082, beta 0.035).

B. BHAVISHYA V27 (THE SHORTING DISASTER):
   - You suggested bi-directional shorting.
   - Result: 339 trades, Net -Rs 289.37 loss (Wallet crashed from Rs 500 to Rs 210.63, Max DD 70.19%).
   - Autopsy: Short trades alone lost -Rs 265.22! Indian cash midcaps suffer from structural upward drift and violent bear-trap short-covering spikes. SHORTING IS PERMANENTLY DEAD.

C. BHAVISHYA V28 (THE CHOP SHIELD - TESTED MINUTES AGO):
   - We implemented your Trade Quality Score (TQS), Breadth-Adjusted Scaling, and Extended Climax Ladder.
   - Gate V2 Result: EXIT CODE: 0 (ZERO FAILURES) Certified!
   - 2025 Chop Horizon (H2): 5 trades, 100% Win Rate (5W/0L), Net +Rs 60.66, ZERO DRAWDOWN (0.00%)!
   - 2025 Full Year: 16 trades, 62.50% Win Rate, Net +Rs 40.04 (almost 3x higher than V26!), Max DD only 4.57% (down from 14.96%!), Dhan friction down to Rs 19.04.
   - BUT THE FATAL FLAW IN 2026:
     In 2026, V28's Net Profit collapsed from +Rs 3,327 down to just +Rs 213.95 (less than 10% of V26)!
     Why? Because:
     1) V28 was so strict (rvol >= 1.25, c > high_5) that trades collapsed from 83 to 34 trades (49 massive momentum runners were discarded!).
     2) Breadth Scaling (f_size) cut position sizing to 0.3x-0.5x, completely choking the geometric compounding snowball!

========================================================================================
2. THE ARCHITECTURAL MANDATE FOR BHAVISHYA V29 (THE HOLY GRAIL HYBRID):
========================================================================================
We DO NOT WANT to sacrifice 2026's +Rs 3,327 compounding beast to get 2025's +Rs 40 chop protection.
WE DEMAND BOTH:
- In Trending / Momentum Expansion Regimes: Run pure V26 full throttle (full 83 trades, full streak compounding, capturing Rs 3,327+).
- In Chop / Bleeding Regimes: Automatically switch into V28's ultra-protective Chop Shield (cutting trades to 16, preserving capital with 4.5% DD).

DeepThink on the following exact quant challenges for Bhavishya V29:
1. DYNAMIC REGIME SWITCHING FORMULA:
   What exact mathematical formula (e.g., 5-day market breadth trend, Nifty/Universe 20-DMA slope, or 10-bar Kaufman Efficiency of the universe median) cleanly switches between:
   - REGIME 1 (MOMENTUM EXPANSION): Activate V26 baseline rules (83 trades, full compounding).
   - REGIME 2 (MONSOON CHOP): Activate V28 TQS filter and defensive risk sizing.
2. ZERO OVERFITTING & ZERO LOOKAHEAD:
   The regime state must be computed strictly at the start of the session or bar (causal, lag-1).
3. RUNNER RATIO CALIBRATION:
   How to ensure the trailing stop allows 4.0 ATR runners in Regime 1 without choking exits in Regime 2.

Provide exact mathematical formulas, threshold numbers, and python-ready logic. DO NOT deviate from V26/V28 lineage.
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
    "max_tokens": 5000
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(URL, data=data, headers=headers)
context = ssl.create_default_context()

print(f"Connecting to reasoning model ({MODEL})...")
print("Sending complete V26/V27/V28 continuity and V29 Hybrid mandate (5000 max tokens)...")

try:
    with urllib.request.urlopen(req, context=context, timeout=240) as response:
        res = json.loads(response.read().decode("utf-8"))
        choice = res.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or ""
        
        full_output = ""
        if content:
            full_output += f"### BHAVISHYA V29 HYBRID ARCHITECTURAL BLUEPRINT:\n\n{content}\n\n---\n\n"
        if reasoning:
            full_output += f"### REASONING & DERIVATION:\n\n{reasoning}\n"
            
        out_dir = ROOT_DIR / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "bhavishya_v29_hybrid_blueprint.md"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(full_output)
            
        print(f"[SUCCESS] V29 Hybrid Blueprint received from reasoning engine!")
        print(f"Saved to {out_file}")
        print(f"Content length: {len(content)}, Reasoning length: {len(reasoning)}")
        
except Exception as ex:
    print(f"Error: {ex}")
