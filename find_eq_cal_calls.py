with open("brokerage-calculator-84e252e92a7f7dbf.js", "r", encoding="utf-8") as f:
    code = f.read()

import re
matches = [m.start() for m in re.finditer(r'eq_cal\(', code)]
print(f"eq_cal calls count: {len(matches)}")
for idx in matches:
    print(code[max(0, idx-100):min(len(code), idx+200)])
    print("-" * 50)
