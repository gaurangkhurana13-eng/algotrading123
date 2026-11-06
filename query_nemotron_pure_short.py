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
The user poses a radical, pure-quant challenge:
"What if an algorithmic trader wants to trade EXCLUSIVELY SHORT (100% Short-Only, ZERO long trades) on Indian cash equities intraday (MIS 5x leverage, 15:10 IST mandatory square-off, Dhan friction ~Rs 2.50 per trade)? How can an intraday trader consistently generate positive mathematical expectancy purely from shorting?"

We have already observed the empirical reality:
- Naive shorts (MACD cross down, Close < VWAP, or RSI < 40) got slaughtered because of violent morning dip-buying wicks, bear squeezes, and Dhan brokerage friction.
- In Q4 2024 crash, naive shorts had a 7.7% win rate because 12 out of 13 were stopped out by intraday pullback wicks before the day ended!

We need you to DeepThink across the entire microstructure of short-only equity trading in India:

1. THE ANATOMY OF INDICATORS IN A PROFITABLE SHORT:
   Why do typical indicators behave completely differently on the short side compared to the long side?
   - RSI: Why is buying RSI > 50 great for momentum, but shorting RSI < 30 or < 40 an oversold bounce trap? What is the true RSI signature for a high-conviction short (e.g. Bearish Rejection at RSI 45-55 on a pullback to EMA)?
   - MACD: How does MACD histogram divergence or zero-line rejection confirm genuine institutional liquidation versus a retail fade?
   - EMAs (EMA9, EMA20, EMA50, EMA110): What is the "Death Cross Cascade" or "Bearish Dynamic Resistance" setup? (e.g. Price rallies into falling EMA20/EMA50 from below and prints a high-volume rejection candle).
   - VWAP & Anchored VWAP: Why is shorting below VWAP on an extended move dangerous, but shorting a failed test of VWAP from below (VWAP Rejection Wick) lethal?

2. THE INTRADAY TIMING & SQUEEZE CYCLE:
   Indian market sessions have distinct microstructure phases:
   - 09:15 - 09:45: Opening gap volatility, trap zone.
   - 09:45 - 11:30: Primary institutional trend / distribution phase.
   - 11:30 - 13:00: European open & midday liquidity vacuum (sideways chop).
   - 13:00 - 14:15: Secondary institutional dumping wave (if market is genuinely weak).
   - 14:15 - 15:10: Compulsory intraday short covering squeeze!
   When EXACTLY can a short entry be taken, and at what exact time MUST all shorts be covered to avoid the 14:30 short-covering bounce?

3. BREAKDOWN-RETEST VS DIRECT BREAKDOWN:
   Why does breaking a 5-bar low directly cause 80% false breakouts in midcaps, whereas "Breakdown -> Weak Low-Volume Pullback -> Rejection -> Lower High" has 3x higher statistical win rate?
   Provide the exact formula for a Causal (Lag-1) Breakdown-Retest Gate.

4. THE SHORT RATCHET & ASYMMETRIC EXIT MECHANISM:
   Because stocks drop faster than they rise, but bounce violently:
   - What should the initial stop-loss be? (e.g., Prior Bar High + 0.15 ATR, or fixed 0.75 ATR?)
   - How should the dynamic trailing ratchet operate? (e.g. Once in profit by 1.0 ATR, immediately lock breakeven; trail by 0.25 ATR behind prior bar highs?)
   - When to take hard profit (e.g. 2.0 - 2.5 ATR or fixed R:R 2.5:1)?

5. THE MATHEMATICAL SPECIFICATION OF "BHAVISHYA PURE-SHORT":
   Provide the full, causal, zero-lookahead quantitative algorithm:
   - Macro Regime Filter (e.g. only active on days when Universe or Index is in confirmed downtrend).
   - Stock Selection & Relative Weakness.
   - Entry Condition (exact boolean formulas with explicit constants).
   - Dynamic Ratchet & Exit Rules.
   - Risk Sizing formula for Rs 500 base wallet ensuring ruin probability is 0%.

Provide deep mathematical rigor, explicit formulas, and cold quantitative truth.
"""

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://antigravity.google.com",
    "X-Title": "Bhavishya Pure-Short Quant Engine"
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
print("Sending Pure-Short Intraday Quant Architecture prompt...")

try:
    with urllib.request.urlopen(req, context=context, timeout=300) as response:
        res = json.loads(response.read().decode("utf-8"))
        choice = res.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or ""
        
        full_output = ""
        if content:
            full_output += f"### BHAVISHYA PURE-SHORT INTRADAY ARCHITECTURAL BLUEPRINT:\n\n{content}\n\n---\n\n"
        if reasoning:
            full_output += f"### REASONING & DERIVATION:\n\n{reasoning}\n"
            
        out_dir = ROOT_DIR / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "bhavishya_pure_short_blueprint.md"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(full_output)
            
        print(f"[SUCCESS] Pure-Short Blueprint received from reasoning engine!")
        print(f"Saved to {out_file}")
        print(f"Content length: {len(content)}, Reasoning length: {len(reasoning)}")
        
except Exception as ex:
    print(f"Error: {ex}")
