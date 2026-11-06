with open("lse_LoadInPython-7yza-U1U.js", "r", encoding="utf-8") as f:
    text1 = f.read()
print("=== LOAD IN PYTHON CODE SNIPPETS ===")
print(text1[:3000])

with open("lse_ApiDocumentation-BnzHNOsH.js", "r", encoding="utf-8") as f:
    text2 = f.read()
print("\n=== API DOCUMENTATION EXCERPTS ===")
# search for code snippets or curl examples
matches = [m.start() for m in re.finditer(r'curl|python|https://download\.londonstrategicedge\.com', text2)]
for idx in matches[:5]:
    print("--- Snippet ---")
    print(text2[max(0, idx - 100): min(len(text2), idx + 500)])
