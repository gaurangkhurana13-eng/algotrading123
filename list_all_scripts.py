from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
scripts = list(ROOT_DIR.glob("*.py"))
for s in scripts:
    print(s.name)
