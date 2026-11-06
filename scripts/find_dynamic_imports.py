import re

with open("lse_bundle.js", "r", encoding="utf-8") as f:
    js = f.read()

dynamic_imports = re.findall(r'import\(\s*["\']\./([a-zA-Z0-9\-_]+\.js)["\']\s*\)', js)
print(f"Dynamic imports found: {len(dynamic_imports)}")
for di in dynamic_imports:
    print("  ->", di)
