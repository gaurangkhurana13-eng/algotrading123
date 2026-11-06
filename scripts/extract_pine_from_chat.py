import json
from pathlib import Path

t_path = Path(r"C:\Users\admin\.gemini\antigravity\brain\e3df3247-acde-4f9a-bae6-cc79289782db\.system_generated\logs\transcript_full.jsonl")

print("Searching for Pine Script in transcript...")
with open(t_path, "r", encoding="utf-8", errors="ignore") as f:
    for idx, line in enumerate(f):
        try:
            data = json.loads(line)
            content = data.get("content", "")
            if "//@version=6" in content or "//@version=5" in content:
                print(f"\nFound Pine Script in step {idx}:")
                # print first 30 lines of pine script
                for l in content.splitlines():
                    if "//@version" in l or "strategy(" in l or "ta.ema" in l or "input" in l or "strategy.entry" in l:
                        print("  ", l)
        except Exception as e:
            pass
