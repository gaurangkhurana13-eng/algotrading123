import re

with open("lse_ApiDocumentation-BnzHNOsH.js", "r", encoding="utf-8") as f:
    text = f.read()

# Find all market types, parameters, documentation text
markets = set(re.findall(r'["\'](us_stocks|uk_stocks|eu_stocks|forex|crypto|futures|indices|commodities|bonds|macro|in_stocks|india_stocks|nse)[a-zA-Z0-9_\-]*["\']', text))
print("Market identifiers found:", markets)

# Find query parameters documented
params = set(re.findall(r'["\']([a-zA-Z_]+)=["\']|params:\s*\{([^}]+)\}', text))
print("\nParam patterns found:")
for p in list(params)[:10]:
    print(" ", p)

# Search for /v3/markets documentation
idx = text.find("/v3/markets")
if idx != -1:
    print("\n=== /v3/markets CONTEXT ===")
    print(text[max(0, idx - 100): min(len(text), idx + 1000)])

# Search for catalogue context
idx2 = text.find("/v3/catalogue")
if idx2 != -1:
    print("\n=== /v3/catalogue CONTEXT ===")
    print(text[max(0, idx2 - 100): min(len(text), idx2 + 1000)])
