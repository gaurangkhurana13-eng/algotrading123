with open("brokerage-calculator-84e252e92a7f7dbf.js", "r", encoding="utf-8") as f:
    code = f.read()

import re
matches = [m.start() for m in re.finditer(r'\bs\s*=', code)]
print(f"Matches for s = : {len(matches)}")
for idx in matches[:10]:
    print(code[max(0, idx-50):min(len(code), idx+100)])
    print("-" * 40)
