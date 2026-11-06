with open("_app.js", "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

idx = content.find("69006:function")
chunk = content[idx:idx+40000]

import re
matches = re.findall(r'\bnZ:function\(\)\{return\s*([^}]+)\}', chunk)
print("nZ export:", matches)
if matches:
    var_name = matches[0]
    var_matches = re.findall(rf'\b{var_name}\s*=\s*["\'][^"\']+["\']', chunk)
    print("Variable value:", var_matches)
