from pathlib import Path

f26 = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V26\run_v26.py").read_text(encoding="utf-8")
f29 = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V29\run_v29.py").read_text(encoding="utf-8")

lines26 = f26.splitlines()
lines29 = f29.splitlines()

print(f"V26 total lines: {len(lines26)}")
print(f"V29 total lines: {len(lines29)}")

# Let's inspect differences in lines 250-320
print("\n--- V26 lines 270-300 ---")
for i, l in enumerate(lines26[270:305], 271):
    print(f"{i}: {l}")

print("\n--- V29 lines 270-305 ---")
for i, l in enumerate(lines29[270:305], 271):
    print(f"{i}: {l}")
