with open("brokerage-calculator-84e252e92a7f7dbf.js", "r", encoding="utf-8") as f:
    code = f.read()

import re
# Search for default security id, or stock name, or default values
matches = re.findall(r'security_id[^,}]+', code)
print("security_id matches:", matches)

# Search for sid:
matches_sid = re.findall(r'sid:[^,}]+', code)
print("sid matches:", matches_sid[:10])

# Search for initial state or default stock
matches_def = re.findall(r'default[^,}]+', code)
print("default matches:", matches_def[:10])

# Search for initial stocks like RELIANCE, TATA, INFY, etc.
stocks = re.findall(r'["\'](RELIANCE|TCS|INFY|HDFCBANK|SBIN|SUZLON|[A-Z]{3,10})["\']', code)
print("Stock symbols found:", set(stocks))
