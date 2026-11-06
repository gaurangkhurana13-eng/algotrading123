import json
from pathlib import Path

t_path = Path(r"C:\Users\admin\.gemini\antigravity\brain\e3df3247-acde-4f9a-bae6-cc79289782db\.system_generated\logs\transcript_full.jsonl")

with open(t_path, "r", encoding="utf-8", errors="ignore") as f:
    for idx, line in enumerate(f):
        if idx == 56:
            data = json.loads(line)
            content = data.get("content", "")
            # write out content to a file
            with open(Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\scripts\official_v3_pine.pine"), "w", encoding="utf-8") as out:
                out.write(content)
            print("Successfully extracted step 56 to official_v3_pine.pine")
