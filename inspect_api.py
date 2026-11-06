import re
import os

with open("brokerage-calculator-84e252e92a7f7dbf.js", "r", encoding="utf-8") as f:
    code = f.read()

# Find all occurrences of p.BT or URLs or endpoints
endpoints = re.findall(r'["\'](https?://[^"\']+|/[^"\']+)["\']', code)
print("Endpoints found:", len(endpoints))
for ep in endpoints:
    if any(k in ep for k in ['api', 'brokerage', 'calc', 'dhan']):
        print("  ", ep)

# Also let's inspect the function where p.cR is called
idx = code.find("calculateBrokerageValues")
if idx != -1:
    print("\n--- Surrounding calculateBrokerageValues ---")
    start = max(0, idx - 1000)
    end = min(len(code), idx + 1000)
    print(code[start:end])
