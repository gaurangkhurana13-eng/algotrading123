from pathlib import Path

root = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")

targets = ["V16", "V17", "V18", "V19", "V20", "V21", "V22"]

print("=== SEARCHING FOR V16 TO V22 IN STRATEGIES & REAL_VERSIONS ===")
for t in targets:
    found = []
    for p in root.rglob("*.py"):
        if t.lower() in p.name.lower() and ("run" in p.name.lower() or "test" in p.name.lower()):
            found.append(str(p.relative_to(root)))
    print(f"{t}: {found}")
