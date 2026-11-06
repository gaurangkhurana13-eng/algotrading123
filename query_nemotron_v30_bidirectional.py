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
We have worked through Bhavishya V26, V27, V28, and V29 together.
Here is the raw, unvarnished empirical truth of what our real-tick backtests and Gate V2 forensic audits have proven:

========================================================================================
1. THE CONTINUOUS EMPIRICAL RECORD:
========================================================================================

A. BHAVISHYA V29 (THE BULL MOMENTUM KING):
   - 2026 Horizon: 83 trades, 59.04% Win Rate, Net +Rs 3,388.80 on Rs 500 capital (7.78x growth, Final Wallet Rs 3,888.80, Profit Factor 2.84, Max DD 13.80%).
   - Gate V2 Forensic Audit: EXIT CODE 0 (ZERO FAILURES) certified!
   - Core Long Engine: G1 & G2 relative excess momentum against universe median, 4.0 ATR Macro Climax Ladder, Streak scaling (alpha 0.082, beta 0.035).

B. BHAVISHYA V27 (THE SHORTING DISASTER):
   - We tested bi-directional shorting previously.
   - Result: 339 trades, Net -Rs 289.37 loss (Account collapsed from Rs 500 to Rs 210, Max DD 70.19%).
   - The Autopsy: Longs made +Rs 233.30 gross; Shorts alone LOST -Rs 265.22 net (-Rs 109 gross + Rs 156 Dhan friction)!
   - WHY IT FAILED: Indian cash equities have an inherent structural upward drift. Midcaps suffer violent short-covering bounces (bear traps). When shorting on simple indicators (MACD cross down or price below VWAP), the system shorted at the bottom of pullbacks and got slaughtered by friction (Rs 2.47 per trade).

C. Q4 2024 UNSEEN CRASH STRESS TEST (Oct 1 - Dec 31, 2024):
   - In Q4 2024, the Indian market suffered a historic crash (Nifty plunged 2,500 points from 26,277, FIIs dumped >Rs 1,00,000 Crore, universe mean daily return = -0.215% per day).
   - In that crash:
     * V26 (pure long) lost -Rs 42.80 (DD 13.87%).
     * V28 (chop shield with breadth scaling) lost only -Rs 22.57 (DD only 5.28%!).
     * V29 (hybrid) lost -Rs 30.01 (DD 13.87%).

========================================================================================
2. THE USER'S UNBREAKABLE MANDATE FOR BHAVISHYA V30:
========================================================================================
The user demands:
"Hame aisa engine chahiye jo V29 ki tarah bull market me 500 ka 3388 kare, aur jab market short/crash ho rahi ho, tab bhi paisa banaye (profit on the short side too)! How can we make money both when the market is roaring AND when it is crashing?"

DeepThink through the following foundational quant challenges with extreme rigor:

1. THE MICROSTRUCTURE OF CASH EQUITIES SHORTING:
   In Indian cash equity intraday trading (MIS margin 5x, no overnight shorting, must square off before 15:10 IST, Dhan friction ~Rs 2.50 per trade):
   Why did 99% of shorting algorithms fail, and what is the ONLY mathematically sound regime where shorting cash midcaps has positive mathematical expectancy?
   - Is it during Market Breadth collapse (e.g. Breadth 20-DMA < 25%)?
   - Is it Institutional Capitulation days (Nifty / Universe Day Return < -1.0% with heavy volume)?

2. BREAKDOWN QUALITY SCORE (BQS) - THE REVERSE TQS:
   What exact mathematical conditions are required to filter out bear traps and false bounces when shorting?
   - RVOL on red volume (e.g. RVOL >= 1.50 on down bars)?
   - Breakdown of prior multi-bar lows (e.g. Close < low_5)?
   - Relative Weakness: Stock falling significantly faster than the falling benchmark (Excess 1h < -0.015)?
   - Kaufman Efficiency Ratio ER10 >= 0.40 (ensuring clean straight downward trend without wicked intraday whipsaws)?
   - Time of day: Why must shorts NEVER be initiated after 13:00 IST (to avoid 14:00 short-covering rallies)?

3. ASYMMETRIC POSITION SIZING & RISK ARCHITECTURE:
   Because bull markets trend smoothly for months while crashes are violent and short-lived:
   - Should Longs retain the full compounding engine (alpha 0.082, beta 0.035, 4.0 ATR ladder)?
   - How should Shorts be sized? (e.g. Fixed risk budget, tight 1.5 ATR ratchet, fast profit taking at 2.0 ATR before the snap-back)?

4. MATHEMATICAL FORMULATION FOR BHAVISHYA V30:
   Provide the complete, causal (lag-1), zero-lookahead specification for:
   - Macro Regime Classifier (Bull Expansion vs Normal Chop vs Panic Crash Breakdown).
   - Long Entry Rule (V29 perfection).
   - Short Entry Rule (Ultra-selective institutional capitulation filter).
   - Short Exit & Stop Mechanism (Rapid ratchet to prevent giving back gains to short covering).
   - Friction Control (Preventing churn: maximum 1 short position per day, strictly single-position).

Think step by step with deep mathematical proofs and explicit threshold constants.
"""

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://antigravity.google.com",
    "X-Title": "Bhavishya Autonomous Quant Engine V30"
}

payload = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.2,
    "max_tokens": 7000
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(URL, data=data, headers=headers)
context = ssl.create_default_context()

print(f"Connecting to reasoning model ({MODEL}) with 7000 max tokens...")
print("Sending complete V26/V27/V28/V29 history and V30 Bi-Directional mandate...")

try:
    with urllib.request.urlopen(req, context=context, timeout=300) as response:
        res = json.loads(response.read().decode("utf-8"))
        choice = res.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or ""
        
        full_output = ""
        if content:
            full_output += f"### BHAVISHYA V30 BI-DIRECTIONAL ARCHITECTURAL BLUEPRINT:\n\n{content}\n\n---\n\n"
        if reasoning:
            full_output += f"### REASONING & DERIVATION:\n\n{reasoning}\n"
            
        out_dir = ROOT_DIR / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "bhavishya_v30_bidirectional_blueprint.md"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(full_output)
            
        print(f"[SUCCESS] V30 Bi-Directional Blueprint received from reasoning engine!")
        print(f"Saved to {out_file}")
        print(f"Content length: {len(content)}, Reasoning length: {len(reasoning)}")
        
except Exception as ex:
    print(f"Error: {ex}")
