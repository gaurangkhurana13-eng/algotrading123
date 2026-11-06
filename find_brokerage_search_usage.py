with open("brokerage-calculator-84e252e92a7f7dbf.js", "r", encoding="utf-8") as f:
    code = f.read()

import re
matches = [m.start() for m in re.finditer(r'<BrokerageSearch', code)]
if not matches:
    matches = [m.start() for m in re.finditer(r'BrokerageSearch', code)]
print(f"Matches count: {len(matches)}")
for idx in matches:
    print(code[max(0, idx-50):min(len(code), idx+300)])
    print("=" * 50)
