import re

with open("brokerage-calculator-84e252e92a7f7dbf.js", "r", encoding="utf-8") as f:
    code = f.read()

# Let's search for calculations, numbers like 0.03, 0.025, 0.05, 0.18, 0.00325, etc.
# Also search for words like brokerage, stt, stamp, sebi, turnover, gst, dpCharges, etc.

matches = re.findall(r'(function\s+\w+\s*\([^)]*\)\s*\{[^}]+\})', code)
print(f"Functions found: {len(matches)}")

# Search for patterns where variables like brokerage, stt are computed
patterns = [
    r'(\w+\.brokerage\s*=[^;]+;)',
    r'(\w+\.stt\s*=[^;]+;)',
    r'(\w+\.stamp\s*=[^;]+;)',
    r'(\w+\.sebi\s*=[^;]+;)',
    r'(\w+\.gst\s*=[^;]+;)',
    r'(\w+\.turnover\s*=[^;]+;)',
    r'(\bbrokerage\s*:[^,}]+)',
    r'(\bstt\s*:[^,}]+)',
    r'(\bturnover\s*:[^,}]+)',
]

for p in patterns:
    m = re.findall(p, code, re.IGNORECASE)
    print(f"Pattern {p}: {len(m)} matches")
    for item in m[:5]:
        print("  ", item)

# Let's search for the main calculator functions or switch statements
# Find all occurrences of "brokerage" in context
brokerage_contexts = [m.start() for m in re.finditer(r'brokerage', code, re.IGNORECASE)]
print(f"\nTotal 'brokerage' occurrences: {len(brokerage_contexts)}")

for idx in brokerage_contexts[:10]:
    start = max(0, idx - 150)
    end = min(len(code), idx + 250)
    print("--- CONTEXT ---")
    print(code[start:end])
