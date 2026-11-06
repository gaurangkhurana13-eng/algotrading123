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

prompt = """You are our Chief Quantitative Strategist and Microstructure Researcher for Indian Equities.
We have worked together continuously across Bhavishya V26, V27, V28, V29, and V30.
Here is the complete, line-by-line function, parameter, and empirical scorecard of all 5 versions tested on raw NSE 15m tick data and audited via Gate V2 forensic rules:

========================================================================================
1. THE 5-VERSION FUNCTIONAL & EMPIRICAL DOSSIER (V26 TO V30):
========================================================================================

A. BHAVISHYA V26 (Baseline Compounding Beast):
   - Entry Parameters:
     * MACD cross up & MACD > 0 (12, 26, 9)
     * Close > EMA50 and Close > EMA110
     * Close > Shifted VWAP (lag-1, zero lookahead)
     * RSI(7) > 50.0
     * ADX(10) >= 18.0 & PDI > MDI
     * ER10 >= 0.35 (Kaufman Efficiency Ratio)
     * RVOL >= 1.20 (Volume / 20-SMA Volume)
     * ATR% >= 0.50% of Close
     * 300-bar Momentum: Close > Close.shift(lookback)
     * Relative Excess Momentum: G1 (day_ret > bench_day_ret) & G2 (1h_ret > bench_1h_ret)
     * Single-position focus (1 active trade)
   - Sizing & Compounding:
     * Streak scaling: alpha = 0.082, beta = 0.035
     * Gate V2 cap: min(0.09 * equity, Rs 258)
     * Dhan friction modeled: Rs 2.47 per roundtrip + 0.05% adverse slippage on entry/exit
   - Exit Ladder (4.0 ATR Macro Climax):
     * MFE >= 4.00 ATR -> Stop = High - 0.10 * ATR
     * MFE >= 3.00 ATR -> Stop = High - 0.14 * ATR
     * MFE >= 2.00 ATR -> Stop = High - 0.20 * ATR
     * MFE >= 1.15 ATR -> Stop = High - 0.28 * ATR
     * MFE >= 0.85 ATR -> Stop = Entry + 0.40 * ATR
     * Initial Hard Stop = Entry - 0.90 * ATR
     * Scratch: bars_held >= 4, MFE < 0.40, Close < Entry & (RSI < 46 or Close < VWAP)
     * EOD Exit: 14:45 IST bar close (100% flat before 15:10 cutoff)
   - Empirical Audited Results:
     * 2026 Modern Horizon: 83 trades, 59.04% WR, Net +Rs 3,327.88 (7.66x growth, Final Rs 3,827.88), PF 2.76, DD 14.80%. Gate V2 EXIT CODE 0!
     * Full Year 2025: 56 trades, 51.79% WR, Net +Rs 13.89, DD 14.96%.
     * Q4 2024 Unseen Crash: 13 trades, 30.77% WR, Net -Rs 42.80, DD 13.87%.

B. BHAVISHYA V27 (Bi-Directional Failure):
   - Added naive symmetric shorting (MACD cross down, Close < VWAP, etc.).
   - Empirical Result: 339 trades, Net -Rs 289.37 (Account crashed from Rs 500 to Rs 210, DD 70.19%).
   - Autopsy: Longs made +Rs 233.30 gross; Shorts lost -Rs 265.22 net (-Rs 109 gross + Rs 156 Dhan friction)! Naive shorting on cash equities is suicidal.

C. BHAVISHYA V28 (Ultra-Strict Chop Shield):
   - Parameters Added:
     * Trade Quality Score (TQS): RVOL >= 1.25, Close > 5-bar High (c > high_5), Close > EMA9 & EMA60, RSI between 48.0 and 78.0.
     * Breadth-Adjusted Scaling: f_size = 0.35 if Universe Breadth (stocks > 20-DMA) < 50%, else 1.0.
   - Empirical Results:
     * 2025 H2 Monsoon Chop: 5 trades, 100% WR (5W/0L), Net +Rs 60.66, DD 0.00%!
     * Full Year 2025: 16 trades, 62.50% WR, Net +Rs 40.04 (3x V26!), DD only 4.57% (down from 14.96%!).
     * Q4 2024 Unseen Crash: 7 trades, Net -Rs 22.57, DD only 5.28%!
     * FATAL FLAW IN 2026: Profit collapsed from +Rs 3,327 down to just +Rs 213.95 because:
       1) Trade count collapsed from 83 to 34 (49 massive multi-session runners were discarded by high_5).
       2) Breadth scaling cut position sizes to 0.35x, choking the geometric compounding snowball.

D. BHAVISHYA V29 (Holy Grail Dynamic Hybrid):
   - Mathematical Synthesis:
     * Dynamic Market Regime Switch: Causal benchmark daily return measured at bar open:
       If bench_day_ret < -0.006 (Severe Bleed Day) -> Activate V28 TQS Filter.
       Else (Normal / Momentum Expansion Day) -> Run pure V26 full compounding engine.
   - Empirical Results:
     * 2026 Modern Horizon: 83 trades, 59.04% WR, Net +Rs 3,388.80 (7.78x growth, Final Rs 3,888.80), PF 2.84, Max DD 13.80%. Gate V2 EXIT CODE 0!
     * Full Year 2025: 55 trades, 52.73% WR, Net +Rs 25.55 (Capital preserved, green).
     * Q4 2024 Unseen Crash: 12 trades, Net -Rs 30.01 (Loss capped at 6%, DD 13.87%).

E. BHAVISHYA V30 (Bi-Directional Breakdown-Retest Experiment):
   - Added institutional Breakdown Quality Score (BQS) for shorts during panic crashes (b_day < -0.012, breadth < 0.25, RVOL >= 1.50, Close < low_5, time <= 12:45, 14:10 exit).
   - Empirical Results:
     * In Q4 2024 Crash: 24 trades, Net -Rs 104.24 (Shorts lost -Rs 90.49!).
     * In 2026 Horizon: 108 trades, Net +Rs 2,109.37 (Shorts lost -Rs 58.28, killing Rs 1,279 of profit vs V29!).
     * Pure-Short: In Q4 2024 crash, 3 trades made +Rs 11.72 (66.7% WR, PF 8.42, DD 0.31%), but in 2025-26 it had negative friction drag.
   - DEFINITIVE QUANT CONCLUSION: Shorting cash midcaps full-time is toxic. SHORTING IS PERMANENTLY DEAD.

========================================================================================
2. THE USER'S UNBREAKABLE MANDATE FOR BHAVISHYA V31:
========================================================================================
The user demands:
"Bhavishya V29 ने ₹500 को ₹3,388 बनाया (7.78x). अब हमें ऐसा Next-Gen Compounding Engine चाहिए जो ₹3,000 नहीं, बल्कि ₹8,000+ (16x Multi-Bagger) लेकर जाए!
Higher Average Win, Lower Average Loss, Higher Profit Factor (PF > 3.5), with controlled drawdown. Trades badhein toh chalega!"

The user has specified 5 MANDATORY ARCHITECTURAL GOALS:

GOAL 1: BALANCED ENTRY FILTER (Neither too strict nor too loose):
- V28 was too strict (34 trades, missed 49 runners). V26 was balanced (83 trades).
- How can we capture 110 - 140 high-conviction momentum expansion trades without letting chop noise in?
- What exact combination of RSI, MACD momentum slope, dynamic VWAP proximity, and Volume Expansion captures early runner liftoff?

GOAL 2: RUNNER HOLDING ENGINE (No Early Exits on Trending Days):
- If a stock is strongly trending upwards on the same day, NEVER exit early!
- How to detect an "Intraday Trend Day" (e.g., Close > EMA9 on 15m, ADX expanding > 25, stock outperforming universe median by > 1.5%) and let the runner milk the maximum multi-ATR meat until 14:45 EOD?

GOAL 3: NOISE-IMMUNE TRAILING STOP (The "Chota Dip" Protection):
- Why do standard trailing stops get whipsawed on small intraday pullbacks?
- How to design a structural trailing ratchet that gives breathing room to shallow dips (e.g., trailing below the structural 2-bar swing low or 1.25 ATR rather than a tight tick stop) while aggressively ratcheting as new highs are conquered?
- Provide the exact mathematical formula for the "Noise-Immune Dynamic Ratchet".

GOAL 4: TRADE FREQUENCY & CONCURRENT SIZING CAPACITY:
- On Dhan, friction is low (~Rs 2.47 per roundtrip).
- Can the system support 2 concurrent positions (Dual-Slot Architecture) or higher trade velocity (e.g. 100-140 trades across the year) so the geometric compounding snowballs much faster towards Rs 8,000+?
- What is the exact margin and risk budget allocation per slot under Gate V2 constitution (total margin <= 95% of equity at 5x MIS, total planned risk <= min(9% * equity, Rs 260))?

GOAL 5: RIGOROUS MATHEMATICAL FORMULATION & CONSTITUTIONAL INTEGRITY:
- Provide the complete, causal (lag-1), zero-lookahead specification for Bhavishya V31:
  * Dynamic Market Regime Switch 2.0
  * Stock Selection & Ranking (Top Relative Excess Momentum)
  * Entry Conditions (exact boolean formulas with constants)
  * Intraday Trend-Holding Logic & Structural Ratchet
  * Dual-Slot Position Sizing & Margin Budget Formula
  * EOD Square-Off discipline

Provide deep mathematical proofs, explicit threshold constants, and full Python implementation guidelines.
"""

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://antigravity.google.com",
    "X-Title": "Bhavishya V31 Master Compounding Blueprint"
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
print("Sending complete V26-V30 dossier and V31 16x Compounding mandate...")

try:
    with urllib.request.urlopen(req, context=context, timeout=300) as response:
        res = json.loads(response.read().decode("utf-8"))
        choice = res.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or ""
        
        full_output = ""
        if content:
            full_output += f"### BHAVISHYA V31 MASTER COMPOUNDING ARCHITECTURAL BLUEPRINT:\n\n{content}\n\n---\n\n"
        if reasoning:
            full_output += f"### REASONING & DERIVATION:\n\n{reasoning}\n"
            
        out_dir = ROOT_DIR / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "bhavishya_v31_master_blueprint.md"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(full_output)
            
        print(f"[SUCCESS] V31 Master Blueprint received from reasoning engine!")
        print(f"Saved to {out_file}")
        print(f"Content length: {len(content)}, Reasoning length: {len(reasoning)}")
        
except Exception as ex:
    print(f"Error: {ex}")
