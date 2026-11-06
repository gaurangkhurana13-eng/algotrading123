with open("brokerage-calculator-84e252e92a7f7dbf.js", "r", encoding="utf-8") as f:
    code = f.read()

import re
# Find where stock search onSelect or onClick is handled
idx = code.find("BrokerageSearch")
print(code[idx:idx+2500])
